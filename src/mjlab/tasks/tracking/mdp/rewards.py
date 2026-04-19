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


def _phase_and_subtarget_weights(
  command: MultiTargetMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return ``(phase, active, pos_weights, ori_weights)`` for the current step.

  *phase*       – (num_envs,) normalised motion phase in [0, 1]
  *active*      – (num_envs, max_subtargets) bool, True when sub-target's
                  phase window contains the current phase
  *pos_weights* – (num_envs, max_subtargets) per-subtarget position reward weights
  *ori_weights* – (num_envs, max_subtargets) per-subtarget orientation reward weights
  """
  which_motion = command.which_motion
  time_step_totals = command._time_step_totals.float()
  phase = command.time_steps.float() / time_step_totals[which_motion]  # (E,)
  phase_starts = command._target_phase_starts_t[which_motion]  # (E, S)
  phase_ends = command._target_phase_ends_t[which_motion]  # (E, S)
  active = (phase[:, None] >= phase_starts) & (phase[:, None] <= phase_ends)
  pos_weights = command._target_pos_reward_weights_t[which_motion]  # (E, S)
  ori_weights = command._target_ori_reward_weights_t[which_motion]  # (E, S)
  return phase, active, pos_weights, ori_weights


def all_motions_target_position_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
) -> torch.Tensor:
  """Phase-gated exponential reward for target position tracking.

  Each active sub-target contributes independently; rewards are summed over
  sub-targets weighted by per-subtarget ``pos_reward_weight``.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  source_pos = command.get_source_pos_w()  # (E, S, 3)
  error = torch.sum(
    torch.square(command.target_position_w - source_pos), dim=-1
  )  # (E, S)
  reward = torch.exp(-error / std**2)  # (E, S)
  _, active, pos_weights, _ = _phase_and_subtarget_weights(command)
  return (reward * active.float() * pos_weights).sum(dim=-1)


def all_motions_target_orientation_axis_alignment_error_exp(
  env: ManagerBasedRlEnv,
  target_command_name: str,
  std: float,
) -> torch.Tensor:
  """Phase-gated exponential reward for per-subtarget axis orientation alignment.

  Each active sub-target contributes independently using its configured
  ``ori_axis``; rewards are summed over sub-targets weighted by ``ori_reward_weight``.
  """
  command = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(target_command_name),
  )
  # axis_vecs: (E, S, 3) — per-subtarget axis vectors for this env's motion
  axis_vecs = command._target_ori_axes_t[command.which_motion]  # (E, S, 3)
  axis_vecs_flat = axis_vecs.reshape(-1, 3)  # (E*S, 3)
  # target_orientation_w / source_quat: (E, S, 4) → reshape to (E*S, 4)
  target_axis_w = quat_apply(
    command.target_orientation_w.reshape(-1, 4), axis_vecs_flat
  ).reshape(command.num_envs, command.max_subtargets, 3)
  source_axis_w = quat_apply(
    command.get_source_quat_w().reshape(-1, 4), axis_vecs_flat
  ).reshape(command.num_envs, command.max_subtargets, 3)

  dot = torch.sum(target_axis_w * source_axis_w, dim=-1).clamp(-1.0, 1.0)  # (E, S)
  error = 1.0 - dot
  reward = torch.exp(-(error**2) / std**2)  # (E, S)
  _, active, _, ori_weights = _phase_and_subtarget_weights(command)
  return (reward * active.float() * ori_weights).sum(dim=-1)


def action_rate_l2_clamped(
  env: ManagerBasedRlEnv, max_value: float = 50.0
) -> torch.Tensor:
  """Penalize the rate of change of actions, clamped to a max value."""
  raw = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )
  return torch.clamp(raw, max=max_value)
