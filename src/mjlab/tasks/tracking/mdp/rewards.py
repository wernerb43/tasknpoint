from __future__ import annotations

from dbm import error
from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from .commands import MotionCommand, MultiTargetMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def all_motions_target_position_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
  per_motion_weights: list[float],
) -> torch.Tensor:
  """Phase-gated exponential reward for target position tracking.

  Each env gets the reward for its currently active motion, scaled by the
  corresponding entry in *per_motion_weights*.  Set a weight to 0.0 to
  disable the position reward for that motion.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  which_motion = command.which_motion

  source_positions = command.get_source_pos_w()

  error = torch.sum(torch.square(command.target_position_w - source_positions), dim=-1)

  reward = torch.exp(-error / std**2)

  time_step_totals = torch.tensor(
    [loader.time_step_total for loader in command.motion_loaders],
    device=command.device,
    dtype=torch.float32,
  )
  phase = command.time_steps / time_step_totals[which_motion]
  active = (phase >= command.target_phase_start) & (phase <= command.target_phase_end)

  weights = torch.tensor(per_motion_weights, device=command.device, dtype=torch.float32)
  per_env_weight = weights[which_motion]

  return reward * active.float() * per_env_weight


def all_motions_target_orientation_axis_alignment_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
  axis: str,
  per_motion_weights: list[float],
) -> torch.Tensor:
  """Phase-gated exponential reward for single-axis orientation alignment.

  Each env gets the reward for its currently active motion, scaled by the
  corresponding entry in *per_motion_weights*.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  which_motion = command.which_motion

  source_quats = command.get_source_quat_w()

  axis_map = {
    "x": torch.tensor([1.0, 0.0, 0.0], device=command.device),
    "y": torch.tensor([0.0, 1.0, 0.0], device=command.device),
    "z": torch.tensor([0.0, 0.0, 1.0], device=command.device),
  }
  if axis not in axis_map:
    raise ValueError(f"Invalid axis '{axis}'. Expected 'x', 'y', or 'z'.")

  axis_vec = axis_map[axis].expand(command.num_envs, 3)
  target_axis_w = quat_apply(command.target_orientation_w, axis_vec)
  source_axis_w = quat_apply(source_quats, axis_vec)

  dot = torch.sum(target_axis_w * source_axis_w, dim=-1).clamp(-1.0, 1.0)
  error = 1.0 - dot
  reward = torch.exp(-(error**2) / std**2)

  time_step_totals = torch.tensor(
    [loader.time_step_total for loader in command.motion_loaders],
    device=command.device,
    dtype=torch.float32,
  )
  phase = command.time_steps / time_step_totals[which_motion]
  active = (phase >= command.target_phase_start) & (phase <= command.target_phase_end)

  weights = torch.tensor(per_motion_weights, device=command.device, dtype=torch.float32)
  per_env_weight = weights[which_motion]

  return reward * active.float() * per_env_weight


def action_rate_l2_clamped(
  env: ManagerBasedRlEnv, max_value: float = 50.0
) -> torch.Tensor:
  """Penalize the rate of change of actions, clamped to a max value."""
  raw = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )
  return torch.clamp(raw, max=max_value)
