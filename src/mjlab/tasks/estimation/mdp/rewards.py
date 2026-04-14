"""Reward terms for the estimation task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.estimation.mdp.ball_command import BallCommand
from mjlab.tasks.tracking.mdp.commands import MultiTargetMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def hit_ball_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  ball_command_name: str = "ball",
) -> torch.Tensor:
  """exp(-||source_pos - ball_pos|| / std).

  Rewards the motion's source link being close to the ball.
  """
  motion_cmd = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(command_name),
  )
  ball_cmd = cast(BallCommand, env.command_manager.get_term(ball_command_name))

  source_pos_w = motion_cmd.get_source_pos_w()  # (num_envs, 3)
  ball_pos_w = ball_cmd.command  # (num_envs, 3)

  error = torch.norm(source_pos_w - ball_pos_w, dim=-1)
  return torch.exp(-error / std)


def penalize_action(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Negative reward while a motion is actively playing (not paused).

  Returns -1 for envs that are playing and 0 for paused envs, so
  the agent is penalised for triggering at the wrong time.
  """
  motion_cmd = cast(
    MultiTargetMotionCommand,
    env.command_manager.get_term(command_name),
  )
  return (~motion_cmd.is_paused).float() * -1.0
