import os
from typing import cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand, MultiTargetMotionCommand


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class _OnnxMultiTargetMotionModel(nn.Module):
  """ONNX-exportable model for multi-target motion tracking.

  Stores stacked motion data for all motions. Takes ``which_motion`` and
  ``time_step`` to index into the correct motion's reference data.
  """

  def __init__(self, actor, cmd: MultiTargetMotionCommand):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    # Stacked motion tensors: (num_motions, max_timesteps, ...)
    self.register_buffer("joint_pos", cmd._stacked_joint_pos.to("cpu"))
    self.register_buffer("joint_vel", cmd._stacked_joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", cmd._stacked_body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", cmd._stacked_body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", cmd._stacked_body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", cmd._stacked_body_ang_vel_w.to("cpu"))
    self.register_buffer("time_step_totals", cmd._time_step_totals.to("cpu").long())
    self.num_motions: int = len(cmd.motion_loaders)

  def forward(self, x, which_motion, time_step):
    which_motion_clamped = torch.clamp(
      which_motion.long().squeeze(-1), max=self.num_motions - 1
    )
    per_motion_max = self.time_step_totals[which_motion_clamped]  # type: ignore[index]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=per_motion_max - 1
    )
    return (
      self.policy(x),
      self.joint_pos[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.joint_vel[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_pos_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_quat_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[which_motion_clamped, time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def _export_single_motion_onnx(
    self, path: str, filename: str, verbose: bool = False
  ) -> None:
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def _export_multi_target_onnx(
    self, path: str, filename: str, verbose: bool = False
  ) -> None:
    cmd = cast(
      MultiTargetMotionCommand,
      self.env.unwrapped.command_manager.get_term("motion"),
    )
    model = _OnnxMultiTargetMotionModel(self.alg.get_policy(), cmd)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    which_motion = torch.zeros(1, 1)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, which_motion, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "which_motion", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def _is_multi_target(self) -> bool:
    cmd = self.env.unwrapped.command_manager.get_term("motion")
    return isinstance(cmd, MultiTargetMotionCommand)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    if self._is_multi_target():
      self._export_multi_target_onnx(path, filename, verbose)
    else:
      self._export_single_motion_onnx(path, filename, verbose)

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      if self._is_multi_target():
        cmd = cast(
          MultiTargetMotionCommand,
          self.env.unwrapped.command_manager.get_term("motion"),
        )
        metadata.update(
          {
            "command_type": "multi_target",
            "anchor_body_name": cmd.cfg.anchor_body_name,
            "body_names": list(cmd.cfg.body_names),
            "num_motions": len(cmd.motion_loaders),
            "source_link_names": list(cmd.cfg.source_link_names),
            "source_link_types": [mc.source_type for mc in cmd.motion_configs],
            "target_phase_starts": [mc.target_phase_start for mc in cmd.motion_configs],
            "target_phase_ends": [mc.target_phase_end for mc in cmd.motion_configs],
            "time_step_totals": cmd._time_step_totals.cpu().tolist(),
          }
        )
      else:
        motion_term = cast(
          MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
        )
        metadata.update(
          {
            "anchor_body_name": motion_term.cfg.anchor_body_name,
            "body_names": list(motion_term.cfg.body_names),
          }
        )
      attach_metadata_to_onnx(os.path.join(policy_path, filename), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        if self.registry_name is not None:
          for rn in self.registry_name.split(","):
            rn = rn.strip()
            if not rn:
              continue
            # Build registry artifact reference for use_artifact.
            # Input may be "org/wandb-registry-collection/name[:alias]".
            # use_artifact needs "wandb-registry-collection/name:alias".
            parts = rn.split("/")
            if len(parts) >= 3:
              # Full registry path — take last two components.
              name_part = parts[-2] + "/" + parts[-1]
            else:
              name_part = parts[-1]
            if ":" not in name_part.split("/")[-1]:
              name_part = name_part + ":latest"
            try:
              wandb.run.use_artifact(name_part)  # type: ignore
            except Exception as e:
              print(f"[WARN] Could not link artifact '{name_part}' to run: {e}")
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
