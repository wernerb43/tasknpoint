"""Estimation-specific motion command.

Extends the multi-target motion command with trigger-based motion start.
The motion holds a paused pose until the high-level policy fires the
trigger action, then plays the motion clip through once and returns to
the paused state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import (
  MultiTargetMotionCommand,
  MultiTargetMotionCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class EstimationMotionCommandCfg(MultiTargetMotionCommandCfg):
  """Multi-target motion command gated by a trigger action."""

  trigger_action_name: str = "trigger"
  """Name of the :class:`TriggerAction` term in the action manager."""

  def build(self, env: ManagerBasedRlEnv) -> EstimationMotionCommand:
    return EstimationMotionCommand(self, env)


class EstimationMotionCommand(MultiTargetMotionCommand):
  """Plays a motion only when the trigger action fires.

  On reset the command enters the paused state immediately (time_steps
  is set past the motion length). The parent's pause logic holds the
  robot's current pose as the reference. When the trigger fires for a
  paused env, a new motion is sampled and played from the start.
  """

  def __init__(self, cfg: EstimationMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Start envs in paused state so the motion waits for a trigger."""
    super()._resample_command(env_ids)
    # super()._resample_command writes root/joint state and samples targets.
    # Refresh derived transforms before reading body_link poses for pause refs.
    self._env.sim.forward()
    # Snapshot robot state as paused reference now because we mark envs
    # as paused immediately and therefore bypass the parent's newly-paused
    # snapshot path in _update_command.
    paused_body_pos = self.robot.data.body_link_pos_w[env_ids][
      :, self.body_indexes
    ]
    paused_body_quat = self.robot.data.body_link_quat_w[env_ids][
      :, self.body_indexes
    ]
    self._paused_body_pos_w[env_ids] = paused_body_pos
    self._paused_body_quat_w[env_ids] = paused_body_quat
    self._paused_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids]
    self._paused_joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]

    # Force envs into paused state by pushing time_steps past the
    # motion length.
    timestep_limits = self._time_step_totals[self.which_motion[env_ids]]
    self.time_steps[env_ids] = timestep_limits
    self.is_paused[env_ids] = True
    self.between_motion_pause_time[env_ids] = 0.0

  def _update_command(self) -> None:
    # Check which paused envs should be triggered.
    from mjlab.tasks.estimation.mdp.trigger_action import TriggerAction

    cfg = cast(EstimationMotionCommandCfg, self.cfg)
    term = self._env.action_manager.get_term(cfg.trigger_action_name)
    assert isinstance(term, TriggerAction)
    triggered = term.triggered & self.is_paused

    if torch.any(triggered):
      trigger_ids = torch.where(triggered)[0]
      self._sample_next_motion(trigger_ids)
      self.is_paused[trigger_ids] = False
      self.between_motion_pause_time[trigger_ids] = 0.0

    # Run the parent update (advances time_steps, handles motion
    # completion → pause transition, updates relative poses).
    # Set pause_length to infinity so the parent never auto-resamples;
    # only the trigger above can start a new motion.
    orig = self.between_motion_pause_length
    self.between_motion_pause_length = float("inf")
    super()._update_command()
    self.between_motion_pause_length = orig


# @dataclass(kw_only=True)
# class EstimationMotionCommandCfg(MultiTargetMotionCommandCfg):
#   """Multi-target motion command with a randomized pause between motions."""

#   pause_range: tuple[float, float] = (3.0, 6.0)
#   """Range (min, max) in seconds for the random pause between motions."""

#   def build(self, env: ManagerBasedRlEnv) -> EstimationMotionCommand:
#     return EstimationMotionCommand(self, env)


# class EstimationMotionCommand(MultiTargetMotionCommand):
#   """Samples a per-env random pause duration each time a motion completes."""

#   def __init__(self, cfg: EstimationMotionCommandCfg, env: ManagerBasedRlEnv):
#     super().__init__(cfg, env)
#     self._pause_lo, self._pause_hi = cfg.pause_range
#     # Per-env pause durations, sampled on init and re-sampled each time
#     # an env finishes its pause and starts a new motion.
#     self._pause_durations = torch.empty(self.num_envs, device=self.device).uniform_(
#       self._pause_lo, self._pause_hi
#     )

#   def _update_command(self) -> None:
#     # Snapshot which envs are currently paused before the parent updates.
#     was_paused = self.is_paused.clone()

#     # Temporarily install per-env pause durations so the parent's
#     # comparison (pause_time >= pause_length) uses our random values.
#     orig = self.between_motion_pause_length
#     self.between_motion_pause_length = self._pause_durations
#     super()._update_command()
#     self.between_motion_pause_length = orig

#     # Detect envs that just exited pause (were paused, now not) and
#     # resample their next pause duration.
#     just_continued = was_paused & ~self.is_paused
#     if torch.any(just_continued):
#       ids = torch.where(just_continued)[0]
#       self._pause_durations[ids] = torch.empty(len(ids), device=self.device).uniform_(
#         self._pause_lo, self._pause_hi
#       )
