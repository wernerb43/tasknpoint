"""Ball command for the estimation task.

Manages a virtual ball that travels at constant velocity along the
-Y world axis toward the robot's motion target position. When the
ball passes a configurable limit it resets to its starting position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MultiTargetMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class BallCommandCfg(CommandTermCfg):
  """Configuration for :class:`BallCommand`."""

  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  motion_command_name: str = "motion"
  """Name of the motion command term whose ``target_position_w`` is used
  as the ball's destination."""

  start_y_offset: float = 5.0
  """How far in front (+Y world) of the target the ball starts."""

  y_limit: float = -2.0
  """Y-world value (relative to the target) at which the ball resets."""

  velocity: float = 3.0
  """Constant speed of the ball in the -Y direction (m/s)."""

  ball_radius: float = 0.04
  """Radius used for debug visualisation."""

  ball_color: tuple[float, float, float, float] = (1.0, 0.3, 0.1, 1.0)
  """RGBA colour for the debug sphere."""

  debug_vis: bool = True

  def build(self, env: ManagerBasedRlEnv) -> BallCommand:
    return BallCommand(self, env)


class BallCommand(CommandTerm):
  """Ball travelling along -Y toward the motion target, resetting on limit."""

  cfg: BallCommandCfg

  def __init__(self, cfg: BallCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._ball_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def _motion_cmd(self) -> MultiTargetMotionCommand:
    cmd = self._env.command_manager.get_term(self.cfg.motion_command_name)
    assert isinstance(cmd, MultiTargetMotionCommand)
    return cmd

  # -- CommandTerm interface --------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """The current ball position in world frame ``(num_envs, 3)``."""
    return self._ball_pos_w

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._reset_ball(env_ids)

  def _update_command(self) -> None:
    # Move the ball in -Y at constant speed.
    self._ball_pos_w[:, 1] -= self.cfg.velocity * self._env.step_dt

    # Reset any ball that passed the limit.
    target_y = self._motion_cmd.target_position_w[:, 1]
    past_limit = self._ball_pos_w[:, 1] < (target_y + self.cfg.y_limit)
    if torch.any(past_limit):
      self._reset_ball(torch.where(past_limit)[0])

  def _update_metrics(self) -> None:
    pass

  # -- Debug visualisation ----------------------------------------------------

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for idx in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self._ball_pos_w[idx].cpu(),
        radius=self.cfg.ball_radius,
        color=self.cfg.ball_color,
        label=f"ball_{idx}",
      )

  # -- Internals --------------------------------------------------------------

  def _reset_ball(self, env_ids: torch.Tensor) -> None:
    """Place the ball at its starting position for *env_ids*."""
    target_pos = self._motion_cmd.target_position_w[env_ids]  # (n, 3)
    self._ball_pos_w[env_ids] = target_pos.clone()
    self._ball_pos_w[env_ids, 1] += self.cfg.start_y_offset
