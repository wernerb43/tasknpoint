"""Evaluate a trained tracking policy with goal-conditioned position and phase metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from tasknpoint_project.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg
from tasknpoint_project.goal_cond_tracking.mdp.commands import MultiTargetMotionCommand
from tasknpoint_project.motion_sets.motion_set import filter_motion_cmd_cfg
from tasknpoint_project.goal_cond_tracking.mdp.metrics import compute_goal_position_error
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for policy evaluation."""

  wandb_run_path: str
  """W&B run path in format 'entity/project/run_id'."""
  target_x_min: float
  """Minimum x of goal point range in anchor (pelvis) frame."""
  target_x_max: float
  """Maximum x of goal point range in anchor (pelvis) frame."""
  target_y_min: float
  """Minimum y of goal point range in anchor (pelvis) frame."""
  target_y_max: float
  """Maximum y of goal point range in anchor (pelvis) frame."""
  target_z_min: float
  """Minimum z of goal point range in anchor (pelvis) frame."""
  target_z_max: float
  """Maximum z of goal point range in anchor (pelvis) frame."""
  motion_config: Path | None = None
  """Path to a motion set TOML. Drives --registry-name and robot XML when provided."""
  registry_name: str | None = None
  """Comma-separated W&B registry artifact names for motion files (e.g. 'entity/proj/right_kick:v0,entity/proj/left_kick:v0').
  If omitted (and no --motion-config), motion artifacts are pulled from the run's used artifacts."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  num_envs: int = 4096
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  reset_after_motion_assignment: bool = False
  """Re-trigger env reset after assigning per-env motions so robot initial state matches
  the chosen motion's start frame. When False, motions and targets are overridden in-place."""
  output_file: str | None = None
  """Optional path to save metrics as JSON."""


def _plot_phase_histogram(
  time_offset: torch.Tensor,
  window_half_s: float,
  save_path: str,
) -> None:
  """Save a matplotlib histogram of closest-approach time offsets from the goal phase."""
  import matplotlib.pyplot as plt

  offsets = time_offset.cpu().numpy()

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.hist(offsets, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
  ax.axvline(0, color="crimson", linewidth=1.5, label="goal phase")
  ax.axvspan(-window_half_s, window_half_s, alpha=0.2, color="crimson", label="goal window")
  ax.set_xlabel("Time offset from goal phase (s)")
  ax.set_ylabel("Environments")
  ax.set_title("Closest-approach phase distribution")
  ax.legend()
  fig.tight_layout()
  plt.savefig(save_path, dpi=150)
  print(f"[INFO] Phase histogram saved to {save_path}")
  plt.close(fig)


def _assign_motions_and_targets(
  cfg: EvaluateConfig,
  command: MultiTargetMotionCommand,
  env: RslRlVecEnvWrapper,
  device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample per-env goal points, assign closest motions, and inject world-frame targets.

  Returns:
    goal_pts: (num_envs, 3) sampled goal points in anchor frame.
    chosen: (num_envs,) motion index assigned to each env.
  """
  num_envs = cfg.num_envs

  goal_pts = torch.empty(num_envs, 3, device=device)
  goal_pts[:, 0].uniform_(cfg.target_x_min, cfg.target_x_max)
  goal_pts[:, 1].uniform_(cfg.target_y_min, cfg.target_y_max)
  goal_pts[:, 2].uniform_(cfg.target_z_min, cfg.target_z_max)

  # nominal_targets: (num_motions, 3) — position sub-target means, anchor frame
  nominal_targets = command._target_pos_means_t[:, 0, :]
  dists = torch.cdist(goal_pts, nominal_targets)  # (num_envs, num_motions)
  chosen = dists.argmin(dim=1)                    # (num_envs,)

  command.which_motion[:] = chosen
  command.time_steps[:] = 0

  if cfg.reset_after_motion_assignment:
    env.unwrapped.reset()

  # Convert goal points from anchor frame to world frame and inject.
  anchor_pos = command.robot.data.body_link_pos_w[:, command.robot_anchor_body_index]
  anchor_quat = command.robot.data.body_link_quat_w[:, command.robot_anchor_body_index]
  goal_world = quat_apply(anchor_quat, goal_pts) + anchor_pos
  command.target_position_w[:, 0, :] = goal_world

  # Log motion assignment distribution.
  num_motions = len(command.motion_loaders)
  counts = [(chosen == i).sum().item() for i in range(num_motions)]
  print("[INFO] Per-motion env counts:", counts)

  return goal_pts, chosen


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute goal-conditioned metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configs.
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd_cfg, MultiTargetMotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  # Load motion set from TOML when --motion-config is provided.
  motion_set = None
  if cfg.motion_config is not None:
    import mujoco
    from tasknpoint_project.motion_sets.motion_set import MotionSet
    motion_set = MotionSet.from_toml(cfg.motion_config)

    if motion_set.robot_xml is not None:
      _xml = motion_set.robot_xml
      robot_entity = env_cfg.scene.entities.get("robot")
      if robot_entity is not None:
        robot_entity.spec_fn = lambda: mujoco.MjSpec.from_file(_xml)
      print(f"[INFO] Robot XML from motion config: {_xml}")

  # Load motion files from W&B.
  api = wandb.Api()
  registry_name = motion_set.eval_registry() if (motion_set and cfg.registry_name is None) else cfg.registry_name
  if registry_name:
    registry_names = [r.strip() for r in registry_name.split(",")]
    motion_files: list[str] = []
    motion_names: list[str] = []
    for rn in registry_names:
      if ":" not in rn:
        rn = rn + ":latest"
      art = api.artifact(rn)
      src = api.artifact(art.source_qualified_name)
      motion_files.append(str(Path(src.download()) / "motion.npz"))
      motion_names.append(rn.split("/")[-1].split(":")[0])
      print(f"[INFO] Downloaded motion: {rn} -> {motion_files[-1]}")
    motion_cmd_cfg.motion_files = motion_files
    filter_motion_cmd_cfg(motion_cmd_cfg, motion_set.enabled_names if motion_set else motion_names)
  else:
    run = api.run(cfg.wandb_run_path)
    arts = [a for a in run.used_artifacts() if a.type == "motions"]
    if not arts:
      raise RuntimeError("No motion artifacts found in the run. Pass --registry-name to specify them explicitly.")
    motion_files = []
    for art in arts:
      motion_files.append(str(Path(art.download()) / "motion.npz"))
      print(f"[INFO] Downloaded motion artifact: {art.name} -> {motion_files[-1]}")
    motion_cmd_cfg.motion_files = motion_files

  # Evaluation config.
  motion_cmd_cfg.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  resume_path, _ = get_wandb_checkpoint_path(
    log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
  )
  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MultiTargetMotionCommand, env.unwrapped.command_manager.get_term("motion"))

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

  # Assign per-env motions and goal targets.
  goal_pts, chosen = _assign_motions_and_targets(cfg, command, env, device)

  # Per-env accumulators.
  min_goal_pos_error = torch.full((cfg.num_envs,), float("inf"), device=device)
  min_error_time_step = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)

  done_envs = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

  print(f"[INFO] Running {cfg.num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)

    active = ~done_envs
    if active.any():
      cur_err = compute_goal_position_error(command, ())
      improved = active & (cur_err < min_goal_pos_error)
      min_goal_pos_error = torch.where(improved, cur_err, min_goal_pos_error)
      min_error_time_step = torch.where(improved, command.time_steps, min_error_time_step)

    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly_done = dones.bool() & ~done_envs

    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{cfg.num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  # Goal position metric.
  mean_min_goal_pos_error = min_goal_pos_error.mean().item()

  # Goal phase metric: did the closest approach occur within the motion's phase window?
  motion_lengths = command._time_step_totals[command.which_motion].float()  # (num_envs,)
  phase = min_error_time_step.float() / motion_lengths                      # (num_envs,)

  phase_starts = torch.tensor(
    [command.motion_configs[m].sub_targets[0].target_phase_start for m in chosen.tolist()],
    device=device,
  )
  phase_ends = torch.tensor(
    [command.motion_configs[m].sub_targets[0].target_phase_end for m in chosen.tolist()],
    device=device,
  )

  step_dt = env.unwrapped.step_dt
  motion_lengths_s = motion_lengths * step_dt          # (num_envs,) in seconds

  phase_window_increase_s = 0.16                                      # seconds
  phase_window_increase = phase_window_increase_s / motion_lengths_s  # (num_envs,) in phase
  phase_starts -= phase_window_increase
  phase_ends += phase_window_increase
  in_phase = (phase >= phase_starts) & (phase <= phase_ends)
  goal_phase_success_rate = in_phase.float().mean().item()

  nominal_phase = (phase_starts + phase_ends) / 2  # (num_envs,)
  phase_offset = phase - nominal_phase              # signed offset from goal phase center

  time_offset_s = phase_offset * motion_lengths_s      # (num_envs,) in seconds
  window_half_s = ((phase_ends - phase_starts) / 2 * motion_lengths_s).mean().item()

  plot_path = str(Path(cfg.output_file).with_suffix(".png")) if cfg.output_file else "phase_histogram.png"
  _plot_phase_histogram(time_offset_s, window_half_s, plot_path)

  metrics = {
    "success_rate": success.float().mean().item(),
    "min_goal_pos_error": mean_min_goal_pos_error,
    "goal_phase_success_rate": goal_phase_success_rate,
  }

  print("\n" + "=" * 50)
  print("Evaluation Results")
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)

  if cfg.output_file:
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
      json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


def main():
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
