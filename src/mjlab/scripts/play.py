"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.estimation.mdp import LowLevelTrackerActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiTargetMotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  tracker_wandb_run_path: str | None = None
  """W&B run path for a pretrained low-level tracker (used by the estimation
  task). Resolves both the tracker checkpoint and the motion artifacts
  linked to the tracker's training run."""
  tracker_wandb_checkpoint_name: str | None = None
  """Optional checkpoint filename within the tracker's W&B run."""
  tracker_checkpoint_file: str | None = None
  """Local path to a tracker ``.pt`` checkpoint. Overrides W&B resolution."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  is_multi_target_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MultiTargetMotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_multi_target_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  # If the environment uses a LowLevelTrackerAction, resolve the tracker
  # checkpoint and (optionally) motion files from the tracker's W&B run.
  tracker_action_cfg: LowLevelTrackerActionCfg | None = None
  for act_cfg in env_cfg.actions.values():
    if isinstance(act_cfg, LowLevelTrackerActionCfg):
      tracker_action_cfg = act_cfg
      break

  if tracker_action_cfg is not None:
    # Copy the tracker's RL actor config so the model architecture matches.
    tracker_rl_cfg = load_rl_cfg("Mjlab-MultiTarget-Tracking-Flat-Unitree-G1")
    assert isinstance(tracker_rl_cfg, RslRlOnPolicyRunnerCfg)
    tracker_action_cfg.tracker_model_cfg = tracker_rl_cfg.actor

    if cfg.tracker_checkpoint_file is not None:
      ckpt = Path(cfg.tracker_checkpoint_file)
      if not ckpt.exists():
        raise FileNotFoundError(f"Tracker checkpoint not found: {ckpt}")
      tracker_action_cfg.checkpoint_path = str(ckpt)
      print(f"[INFO]: Tracker checkpoint (local): {ckpt.name}")
    elif cfg.tracker_wandb_run_path is not None:
      tracker_log_root = (
        Path("logs") / "rsl_rl" / tracker_rl_cfg.experiment_name
      ).resolve()
      tracker_ckpt, was_cached = get_wandb_checkpoint_path(
        tracker_log_root,
        Path(cfg.tracker_wandb_run_path),
        cfg.tracker_wandb_checkpoint_name,
      )
      tracker_action_cfg.checkpoint_path = str(tracker_ckpt)
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Tracker checkpoint: {tracker_ckpt.name} "
        f"(run: {tracker_ckpt.parent.name}, {cached_str})"
      )

      # Pre-populate motion files from the tracker's training run
      # so the downstream multi-target block can skip its own resolution.
      if is_multi_target_task and cfg.motion_file is None and cfg.registry_name is None:
        import wandb

        api = wandb.Api()
        tracker_run = api.run(str(cfg.tracker_wandb_run_path))
        arts = [a for a in tracker_run.used_artifacts() if a.type == "motions"]
        if not arts:
          raise RuntimeError("No motion artifacts found in the tracker's W&B run.")
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
        resolved_files: list[str] = []
        for art in arts:
          p = str(Path(art.download()) / "motion.npz")
          resolved_files.append(p)
          print(f"[INFO]: Tracker motion artifact: {art.name} -> {p}")
        motion_cmd.motion_files = resolved_files

  if is_multi_target_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)

    if motion_cmd.motion_files:
      # Already resolved (e.g. by the tracker block above).
      pass
    elif cfg.motion_file is not None:
      # Comma-separated local paths.
      files = [f.strip() for f in cfg.motion_file.split(",")]
      for f in files:
        if not Path(f).exists():
          raise FileNotFoundError(f"Motion file not found: {f}")
      motion_cmd.motion_files = files
      print(f"[INFO]: Using local motion files: {files}")
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Multi-target tracking tasks require either:\n"
          "  --motion-file f1.npz,f2.npz (comma-separated local files)\n"
          "  --registry-name name1,name2 (comma-separated WandB registry names)"
        )
      import wandb

      api = wandb.Api()
      registry_names = [r.strip() for r in cfg.registry_name.split(",")]
      motion_files: list[str] = []
      for rn in registry_names:
        if ":" not in rn:
          rn = rn + ":latest"
        artifact = api.artifact(rn)
        motion_files.append(str(Path(artifact.download()) / "motion.npz"))
        print(f"[INFO]: Downloaded motion: {rn} -> {motion_files[-1]}")
      motion_cmd.motion_files = motion_files
    else:
      # Trained mode: resolve motion artifacts from the W&B training run.
      import wandb

      api = wandb.Api()
      if cfg.registry_name:
        # Explicit comma-separated registry names override artifact resolution.
        registry_names = [r.strip() for r in cfg.registry_name.split(",")]
        motion_files = []
        for rn in registry_names:
          if ":" not in rn:
            rn = rn + ":latest"
          artifact = api.artifact(rn)
          motion_files.append(str(Path(artifact.download()) / "motion.npz"))
          print(f"[INFO]: Downloaded motion: {rn} -> {motion_files[-1]}")
        motion_cmd.motion_files = motion_files
      elif cfg.wandb_run_path is not None:
        wandb_run = api.run(str(cfg.wandb_run_path))
        arts = [a for a in wandb_run.used_artifacts() if a.type == "motions"]
        if not arts:
          raise RuntimeError("No motion artifacts found in the run.")
        motion_files = []
        for art in arts:
          motion_files.append(str(Path(art.download()) / "motion.npz"))
          print(f"[INFO]: Downloaded motion artifact: {art.name} -> {motion_files[-1]}")
        motion_cmd.motion_files = motion_files
      else:
        raise ValueError(
          "Multi-target tracking tasks require motion files. Provide one of:\n"
          "  --motion-file f1.npz,f2.npz (comma-separated local files)\n"
          "  --registry-name name1,name2 (comma-separated WandB registry names)\n"
          "  --wandb-run-path entity/project/run_id (resolve from training run)"
        )

  elif is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
