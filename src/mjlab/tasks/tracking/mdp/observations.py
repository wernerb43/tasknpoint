from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)

from .commands import MotionCommand, MultiTargetMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def current_target_pos_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return all sub-target positions and orientations in the robot anchor frame.

  Returns a ``(num_envs, max_subtargets * 7)`` tensor: 3 pos + 4 quat per sub-target.
  """
  command = cast(MultiTargetMotionCommand, env.command_manager.get_term(command_name))
  anchor_quat_inv = quat_inv(command.robot_anchor_quat_w)  # (E, 4)
  S = command.max_subtargets
  quat_inv_exp = anchor_quat_inv[:, None, :].expand(-1, S, -1).reshape(-1, 4)
  pos_b = quat_apply(
    quat_inv_exp,
    (command.target_position_w - command.robot_anchor_pos_w[:, None, :]).reshape(-1, 3),
  ).reshape(command.num_envs, S * 3)
  ori_b = quat_mul(quat_inv_exp, command.target_orientation_w.reshape(-1, 4)).reshape(
    command.num_envs, S * 4
  )
  return torch.cat([pos_b, ori_b], dim=-1)
