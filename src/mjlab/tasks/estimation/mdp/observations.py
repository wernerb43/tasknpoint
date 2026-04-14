"""Observation terms specific to the estimation task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.estimation.mdp.ball_command import BallCommand
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ball_pos_b(
  env: ManagerBasedRlEnv,
  command_name: str = "ball",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball position expressed in the robot root-link frame.

  Returns shape ``(num_envs, 3)``.
  """
  ball_cmd = cast(BallCommand, env.command_manager.get_term(command_name))
  ball_pos_w = ball_cmd.command  # (num_envs, 3)

  asset: Entity = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w
  # print(f"ball_pos_w: {ball_pos_w}")
  # print(quat_apply_inverse(root_quat_w, ball_pos_w - root_pos_w))

  return quat_apply_inverse(root_quat_w, ball_pos_w - root_pos_w)


def ball_pos_root_centered_w(
  env: ManagerBasedRlEnv,
  command_name: str = "ball",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ball position minus root position in world frame.

  Returns shape ``(num_envs, 3)``.
  """
  ball_cmd = cast(BallCommand, env.command_manager.get_term(command_name))
  ball_pos_w = ball_cmd.command

  asset: Entity = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w

  return ball_pos_w - root_pos_w


def ball_vel_w(
  env: ManagerBasedRlEnv,
  command_name: str = "ball",
) -> torch.Tensor:
  """Ball velocity in world frame (constant -Y).

  Returns shape ``(num_envs, 3)``.
  """
  ball_cmd = cast(BallCommand, env.command_manager.get_term(command_name))
  vel = torch.zeros(env.num_envs, 3, device=env.device)
  vel[:, 1] = -ball_cmd.cfg.velocity
  return vel
