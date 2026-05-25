from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


# motion loader takes in the motion file, body indexes, and constructs an object which has positions velocities and angles for the motion. We index into this during motion tracking.
class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]


@dataclass  # this means that each of the field: type will be populated in an implicit __init__ method.
class MotionGoalCfg:  # this indicates the conditioned goal(s) in each motion
  """Configuration for a single target within a motion."""

  goal_type: Literal["position", "orientation", "velocity"] = "position"
  goal_weight: float = 1.0  # this is the weight of the position, orientation, or velocity reward for this target depending on the obs type
  source_link: str | None = None
  source_type: Literal["body", "site"] = "body"
  target_link: str | None = None
  target_type: Literal["body", "site"] = "body"
  target_pos_mean: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_pos_std: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_vel_mean: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_vel_std: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_orientation_mean: dict[str, float] = field(
    default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
  )
  target_orientation_std: dict[str, float] = field(
    default_factory=lambda: {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
  )
  orientation_axis: Literal["x", "y", "z"] = "y"

  target_phase_start: float = 0.0  # phase start and end
  target_phase_end: float = 1.0


@dataclass
class MotionCfg:
  """Per-motion target configuration — a list of sub-targets, each active during its own phase window."""

  name: str = ""
  sub_targets: list[MotionGoalCfg] = field(default_factory=list)
  sampling_weight: float = 1.0  # how often we sample this motion
  probe_points: list[tuple[str, float]] = field(default_factory=list)  # (site_or_body_name, phase)


class MultiTargetMotionCommand(CommandTerm):
  """Motion command supporting multiple motions with per-motion target points.

  Each motion has a source link that should reach a target position during a
  specific phase window of the motion.  Targets can be static (sampled from
  a Gaussian) or dynamic (tracking another body link).
  """

  cfg: MultiTargetMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiTargetMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Build per-motion loaders; configs come directly from cfg.
    self.motion_loaders: list[MotionLoader] = []
    for motion_file in self.cfg.motion_files:
      self.motion_loaders.append(
        MotionLoader(motion_file, self.body_indexes, device=self.device)
      )
    self.motion_configs: list[MotionCfg] = list(self.cfg.motion_target_cfgs)

    # Source/target indices per motion per sub-target (body or site depending on type).
    self.source_body_indices: list[list[int]] = []
    self.source_is_site: list[list[bool]] = []
    self.target_body_indices: list[list[int | None]] = []
    self.target_is_site: list[list[bool]] = []
    for mc in self.motion_configs:
      m_src, m_src_site, m_tgt, m_tgt_site = [], [], [], []
      for st in mc.sub_targets:
        if st.source_type == "site":
          m_src.append(self.robot.find_sites(st.source_link, preserve_order=True)[0][0])
          m_src_site.append(True)
        else:
          m_src.append(self.robot.body_names.index(st.source_link))
          m_src_site.append(False)
        if st.target_link is not None:
          if st.target_type == "site":
            m_tgt.append(
              self.robot.find_sites(st.target_link, preserve_order=True)[0][0]
            )
            m_tgt_site.append(True)
          else:
            m_tgt.append(self.robot.body_names.index(st.target_link))
            m_tgt_site.append(False)
        else:
          m_tgt.append(None)
          m_tgt_site.append(False)
      self.source_body_indices.append(m_src)
      self.source_is_site.append(m_src_site)
      self.target_body_indices.append(m_tgt)
      self.target_is_site.append(m_tgt_site)

    # Pre-stack motion data for vectorized access.
    num_motions = len(self.motion_loaders)
    max_t = max(m.time_step_total for m in self.motion_loaders)

    def _pad_stack(tensors: list[torch.Tensor], max_len: int) -> torch.Tensor:
      padded = []
      for t in tensors:
        if t.shape[0] < max_len:
          pad = t[-1:].expand((max_len - t.shape[0],) + t.shape[1:])
          t = torch.cat([t, pad], dim=0)
        padded.append(t)
      return torch.stack(padded)

    self._stacked_joint_pos = _pad_stack(
      [m.joint_pos for m in self.motion_loaders], max_t
    )
    self._stacked_joint_vel = _pad_stack(
      [m.joint_vel for m in self.motion_loaders], max_t
    )
    self._stacked_body_pos_w = _pad_stack(
      [m.body_pos_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_quat_w = _pad_stack(
      [m.body_quat_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_lin_vel_w = _pad_stack(
      [m.body_lin_vel_w for m in self.motion_loaders], max_t
    )
    self._stacked_body_ang_vel_w = _pad_stack(
      [m.body_ang_vel_w for m in self.motion_loaders], max_t
    )
    self._time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motion_loaders], device=self.device
    )

    # Pre-compute per-motion, per-subtarget indices and parameters as tensors.
    # Shape convention: (num_motions, max_subtargets, ...), padded with zeros/False.
    max_subtargets = max(len(mc.sub_targets) for mc in self.motion_configs)
    self.max_subtargets = max_subtargets

    # Per-subtarget observation type — taken from the first motion config and
    # padded with "position" for any slots that don't exist in that motion.
    first_sub_targets = self.motion_configs[0].sub_targets
    self._subtarget_goal_types: list[str] = [
      first_sub_targets[s].goal_type if s < len(first_sub_targets) else "position"
      for s in range(max_subtargets)
    ]

    def _pad_to(vals: list, length: int, pad):
      return vals + [pad] * (length - len(vals))

    self._source_body_indices_t = torch.tensor(
      [_pad_to(row, max_subtargets, 0) for row in self.source_body_indices],
      device=self.device,
      dtype=torch.long,
    )  # (num_motions, max_subtargets)
    self._target_body_indices_t = torch.tensor(
      [
        _pad_to([idx if idx is not None else 0 for idx in row], max_subtargets, 0)
        for row in self.target_body_indices
      ],
      device=self.device,
      dtype=torch.long,
    )  # (num_motions, max_subtargets)
    self._has_target_link = torch.tensor(
      [
        _pad_to([idx is not None for idx in row], max_subtargets, False)
        for row in self.target_body_indices
      ],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)
    self._source_is_site_t = torch.tensor(
      [_pad_to(row, max_subtargets, False) for row in self.source_is_site],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)
    self._target_is_site_t = torch.tensor(
      [_pad_to(row, max_subtargets, False) for row in self.target_is_site],
      device=self.device,
      dtype=torch.bool,
    )  # (num_motions, max_subtargets)

    # Pre-compute target sampling parameters as tensors.
    # (num_motions, max_subtargets, 3)
    self._target_pos_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_pos_mean.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )
    self._target_pos_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_pos_std.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )
    self._target_ori_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [
                st.target_orientation_mean.get(k, 0.0) for k in ["roll", "pitch", "yaw"]
              ],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_ori_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_orientation_std.get(k, 0.0) for k in ["roll", "pitch", "yaw"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_phase_starts_t = torch.tensor(
      [
        _pad_to([st.target_phase_start for st in mc.sub_targets], max_subtargets, 0.0)
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    self._target_phase_ends_t = torch.tensor(
      [
        _pad_to([st.target_phase_end for st in mc.sub_targets], max_subtargets, 0.0)
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)

    def _goal_weight(st: MotionGoalCfg, kind: str) -> float:
      return st.goal_weight if st.goal_type == kind else 0.0

    self._target_pos_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "position") for st in mc.sub_targets], max_subtargets, 0.0
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    self._target_ori_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "orientation") for st in mc.sub_targets],
          max_subtargets,
          0.0,
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)
    _axis_vec_map = {
      "x": [1.0, 0.0, 0.0],
      "y": [0.0, 1.0, 0.0],
      "z": [0.0, 0.0, 1.0],
    }
    self._target_ori_axes_t = torch.tensor(
      [
        _pad_to(
          [_axis_vec_map[st.orientation_axis] for st in mc.sub_targets],
          max_subtargets,
          [0.0, 0.0, 0.0],
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_means_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_vel_mean.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_stds_t = torch.stack(
      [
        torch.stack(
          [
            torch.tensor(
              [st.target_vel_std.get(k, 0.0) for k in ["x", "y", "z"]],
              device=self.device,
            )
            for st in _pad_to(mc.sub_targets, max_subtargets, mc.sub_targets[0])
          ]
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, max_subtargets, 3)
    self._target_vel_reward_weights_t = torch.tensor(
      [
        _pad_to(
          [_goal_weight(st, "velocity") for st in mc.sub_targets], max_subtargets, 0.0
        )
        for mc in self.motion_configs
      ],
      device=self.device,
    )  # (num_motions, max_subtargets)

    # Motion sampling weights — normalised to sum to 1.
    num_motions = len(self.motion_loaders)
    raw_weights = self.cfg.motion_sampling_weights
    w = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
    self._motion_weights_t = w / w.sum()

    # Per-env state.
    self.which_motion = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.target_pos_std_scale: float = 1.0

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Target tracking (world frame) — (num_envs, max_subtargets, 3/4).
    # Per-episode sampled offset for moving targets, stored in anchor-aligned world frame
    # so _update_command can apply it every step without re-randomizing.
    self._moving_target_offset_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.target_position_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self.target_orientation_w = torch.zeros(
      self.num_envs, max_subtargets, 4, device=self.device
    )
    self.target_orientation_w[:, :, 0] = 1.0
    self.target_velocity_w = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )

    # Source-state cache — refreshed once per step in _update_command so that
    # get_source_pos_w / get_source_quat_w / get_source_lin_vel_w are free reads.
    self._source_pos_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )
    self._source_quat_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 4, device=self.device
    )
    self._source_quat_w_cache[:, :, 0] = 1.0
    self._source_lin_vel_w_cache = torch.zeros(
      self.num_envs, max_subtargets, 3, device=self.device
    )

    # Adaptive sampling per motion.
    self.bin_counts = [
      int(self.motion_loaders[i].time_step_total // (1 / self._env.step_dt)) + 1
      for i in range(num_motions)
    ]
    self.bin_failed_counts = [
      torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device)
      for i in range(num_motions)
    ]
    self._current_bin_failed = [
      torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device)
      for i in range(num_motions)
    ]
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    # Between-motion pause.
    self.between_motion_pause_range = self.cfg.between_motion_pause_range
    self.between_motion_pause_time = torch.zeros(self.num_envs, device=self.device)
    self._sampled_pause_lengths = torch.zeros(self.num_envs, device=self.device)
    self.is_paused = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    num_bodies = len(self.cfg.body_names)
    num_joints = self.robot.data.joint_pos.shape[-1]
    self._paused_body_pos_w = torch.zeros(
      self.num_envs, num_bodies, 3, device=self.device
    )
    self._paused_body_quat_w = torch.zeros(
      self.num_envs, num_bodies, 4, device=self.device
    )
    self._paused_body_quat_w[..., 0] = 1.0
    self._paused_joint_pos = torch.zeros(self.num_envs, num_joints, device=self.device)
    self._paused_joint_vel = torch.zeros(self.num_envs, num_joints, device=self.device)

    # Metrics.
    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_target_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model for visualization (created lazily).
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  # ------------------------------------------------------------------
  # Time-step helpers
  # ------------------------------------------------------------------

  def _clamped_time_steps(self) -> torch.Tensor:
    totals = self._time_step_totals[self.which_motion]
    return torch.where(
      self.time_steps >= totals,
      torch.zeros_like(self.time_steps),
      self.time_steps,
    )

  # ------------------------------------------------------------------
  # Properties: command observation
  # ------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Joint pos/vel + per-subtarget observation (position, orientation, or velocity).

    Each subtarget emits only the quantity selected by its ``goal_type``:
      "position"    → 3 values:
                        static target  → anchor-frame position (changes as robot moves)
                        dynamic target → world-frame position  (constant over the episode)
      "orientation" → 4 values (anchor-frame quaternion)
      "velocity"    → 3 values (world-frame linear velocity)

    For dynamic (moving) targets the world-frame position is used directly so
    that the observation encodes the fixed per-episode task goal and does not
    change as the robot moves toward the target.
    """
    robot_quat_inv = quat_inv(self.robot_anchor_quat_w)  # (E, 4)
    quat_inv_exp = robot_quat_inv[:, None, :].expand(-1, self.max_subtargets, -1)
    flat_quat_inv = quat_inv_exp.reshape(-1, 4)

    # Pelvis-frame position — used for static targets.
    target_pos_b = quat_apply(
      flat_quat_inv,
      (self.target_position_w - self.robot_anchor_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(self.num_envs, self.max_subtargets, 3)
    target_ori_b = quat_mul(
      flat_quat_inv,
      self.target_orientation_w.reshape(-1, 4),
    ).reshape(self.num_envs, self.max_subtargets, 4)
    target_vel_b = self.target_velocity_w.reshape(
      self.num_envs, self.max_subtargets, 3
    )  # velocity is already in world frame, no need to transform

    # Per-env, per-subtarget dynamic flag: True when a target_link is set.  (E, S)
    is_dynamic = self._has_target_link[self.which_motion]

    parts: list[torch.Tensor] = [
      self.joint_pos,
      self.joint_vel,
    ]  # always use joint pos and vel

    for s, goal_type in enumerate(
      self._subtarget_goal_types
    ):  # for each of the subtarget, add their goal type
      if goal_type == "position":
        # Static → pelvis-relative (tells the policy where the goal is w.r.t. the robot).
        # Dynamic → world-frame (constant per-episode goal; does not shift as robot moves).
        dynamic_mask = is_dynamic[:, s].unsqueeze(-1)  # (E, 1)
        pos_obs = torch.where(
          dynamic_mask,
          self.target_position_w[:, s],  # world frame — constant over the motion
          target_pos_b[:, s],            # pelvis frame — for static targets
        )
        parts.append(pos_obs)
      elif goal_type == "orientation":
        parts.append(target_ori_b[:, s])
      else:  # velocity
        parts.append(target_vel_b[:, s])

    return torch.cat(parts, dim=1)

  # ------------------------------------------------------------------
  # Properties: motion data
  # ------------------------------------------------------------------
  @property
  def joint_pos(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_joint_pos[self.which_motion, t]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_joint_pos[self.is_paused]
    return val

  @property
  def joint_vel(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_joint_vel[self.which_motion, t]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_joint_vel[self.is_paused]
    return val

  @property
  def body_pos_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    body_pos = (
      self._stacked_body_pos_w[self.which_motion, t]
      + self._env.scene.env_origins[:, None, :]
    )
    if torch.any(self.is_paused):
      body_pos[self.is_paused] = self._paused_body_pos_w[self.is_paused]
    return body_pos

  @property
  def body_quat_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_body_quat_w[self.which_motion, t]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_body_quat_w[self.is_paused]
    return val

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    vel = self._stacked_body_lin_vel_w[self.which_motion, t]
    vel[self.is_paused] = 0.0
    return vel

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    vel = self._stacked_body_ang_vel_w[self.which_motion, t]
    vel[self.is_paused] = 0.0
    return vel

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    anchor_pos = (
      self._stacked_body_pos_w[self.which_motion, t, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )
    if torch.any(self.is_paused):
      anchor_pos[self.is_paused] = self._paused_body_pos_w[
        self.is_paused, self.motion_anchor_body_index
      ]
    return anchor_pos

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    val = self._stacked_body_quat_w[self.which_motion, t, self.motion_anchor_body_index]
    if torch.any(self.is_paused):
      val[self.is_paused] = self._paused_body_quat_w[
        self.is_paused, self.motion_anchor_body_index
      ]
    return val

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    vel = self._stacked_body_lin_vel_w[
      self.which_motion, t, self.motion_anchor_body_index
    ]
    vel[self.is_paused] = 0.0
    return vel

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    t = self._clamped_time_steps()
    vel = self._stacked_body_ang_vel_w[
      self.which_motion, t, self.motion_anchor_body_index
    ]
    vel[self.is_paused] = 0.0
    return vel

  # ------------------------------------------------------------------
  # Properties: robot state
  # ------------------------------------------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # ------------------------------------------------------------------
  # Link-state helpers (body or site)
  # ------------------------------------------------------------------

  def _fetch_link_state(
    self,
    env_ids: torch.Tensor,
    indices_t: torch.Tensor,
    is_site_t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat)`` in world frame for each env in *env_ids*.

    *indices_t* and *is_site_t* are per-env body or site indices and flags
    (already indexed for this batch).
    """
    pos = torch.zeros(len(env_ids), 3, device=self.device)
    quat = torch.zeros(len(env_ids), 4, device=self.device)
    quat[:, 0] = 1.0
    body_mask = ~is_site_t
    if torch.any(body_mask):
      bi = env_ids[body_mask]
      pos[body_mask] = self.robot.data.body_link_pos_w[bi, indices_t[body_mask]]
      quat[body_mask] = self.robot.data.body_link_quat_w[bi, indices_t[body_mask]]
    if torch.any(is_site_t):
      si = env_ids[is_site_t]
      pos[is_site_t] = self.robot.data.site_pos_w[si, indices_t[is_site_t]]
      quat[is_site_t] = self.robot.data.site_quat_w[si, indices_t[is_site_t]]
    return pos, quat

  def _fetch_link_state_batched(
    self,
    env_ids: torch.Tensor,
    indices_t: torch.Tensor,
    is_site_t: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pos, quat)`` of shape ``(E, S, 3/4)`` for *env_ids* × sub-targets.

    *indices_t* and *is_site_t* are ``(E, S)`` body/site indices and flags.
    Fetches all sub-targets in one vectorized gather, blending body and site
    data via ``torch.where``.
    """
    E, S = indices_t.shape
    env_ids_exp = env_ids[:, None].expand(E, S)  # (E, S)
    # Clamp to valid range for each array so out-of-domain indices don't OOB.
    # Values at clamped-invalid positions are masked out by torch.where below.
    body_idx = indices_t.clamp(max=self.robot.data.body_link_pos_w.shape[1] - 1)
    site_idx = indices_t.clamp(max=self.robot.data.site_pos_w.shape[1] - 1)
    body_pos = self.robot.data.body_link_pos_w[env_ids_exp, body_idx]  # (E, S, 3)
    body_quat = self.robot.data.body_link_quat_w[env_ids_exp, body_idx]  # (E, S, 4)
    site_pos = self.robot.data.site_pos_w[env_ids_exp, site_idx]  # (E, S, 3)
    site_quat = self.robot.data.site_quat_w[env_ids_exp, site_idx]  # (E, S, 4)
    is_site = is_site_t.unsqueeze(-1)  # (E, S, 1)
    pos = torch.where(is_site, site_pos, body_pos)
    quat = torch.where(is_site.expand(-1, -1, 4), site_quat, body_quat)
    return pos, quat

  def _fetch_target_pos_quat(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame ``(pos, quat)`` of shape ``(E, S, 3/4)`` for each target link."""
    return self._fetch_link_state_batched(
      env_ids,
      self._target_body_indices_t[motion_ids],
      self._target_is_site_t[motion_ids],
    )

  # ------------------------------------------------------------------
  # Target sampling
  # ------------------------------------------------------------------

  def _sample_targets(self, env_ids: torch.Tensor) -> None:
    """Sample target positions/orientations for *env_ids* across all sub-targets."""
    motion_ids = self.which_motion[env_ids]

    anchor_pos_w = self.robot.data.body_link_pos_w[
      env_ids, self.robot_anchor_body_index
    ]
    anchor_quat_w = self.robot.data.body_link_quat_w[
      env_ids, self.robot_anchor_body_index
    ]

    for s in range(self.max_subtargets):
      has_moving = self._has_target_link[motion_ids, s]  # (E,)

      # Moving targets — fetch only when needed, using the non-batched helper.
      if torch.any(has_moving):
        mv_eids = env_ids[has_moving]
        mv_mids = motion_ids[has_moving]
        mv_idx = torch.where(has_moving)[0]
        tgt_pos, tgt_quat = self._fetch_link_state(
          mv_eids,
          self._target_body_indices_t[mv_mids, s],
          self._target_is_site_t[mv_mids, s],
        )
        pos_offset = self._target_pos_means_t[mv_mids, s]  # (E_mv, 3), target body local frame
        pos_std = self._target_pos_stds_t[mv_mids, s] * self.target_pos_std_scale
        rand_offset = pos_offset + torch.randn(len(mv_eids), 3, device=self.device) * pos_std

        # Persist per-episode random offset (target-body local frame) so _update_command
        # reapplies it every step without re-randomizing.
        self._moving_target_offset_w[mv_eids, s] = rand_offset

        self.target_position_w[mv_eids, s] = tgt_pos + quat_apply(tgt_quat, rand_offset)
        self.target_orientation_w[mv_eids, s] = tgt_quat
        vel_mean_mv = self._target_vel_means_t[mv_mids, s]
        vel_std_mv = self._target_vel_stds_t[mv_mids, s]
        rand_vel_mv = (
          vel_mean_mv + torch.randn(len(mv_eids), 3, device=self.device) * vel_std_mv
        )
        self.target_velocity_w[mv_eids, s] = quat_apply(
          anchor_quat_w[mv_idx], rand_vel_mv
        )

      # Static targets — sampled from Gaussian in anchor frame.
      has_static = ~has_moving
      if torch.any(has_static):
        st_idx = torch.where(has_static)[0]
        st_eids = env_ids[has_static]
        st_mids = motion_ids[has_static]
        ns = int(has_static.sum().item())

        pos_mean = self._target_pos_means_t[st_mids, s]
        pos_std = self._target_pos_stds_t[st_mids, s] * self.target_pos_std_scale
        rand_pos = pos_mean + torch.randn(ns, 3, device=self.device) * pos_std

        ori_mean = self._target_ori_means_t[st_mids, s]
        ori_std = self._target_ori_stds_t[st_mids, s]
        rand_euler = ori_mean + torch.randn(ns, 3, device=self.device) * ori_std
        rand_quat = quat_from_euler_xyz(
          rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2]
        )

        st_anchor_pos = anchor_pos_w[st_idx]
        st_anchor_quat = anchor_quat_w[st_idx]
        self.target_position_w[st_eids, s] = (
          quat_apply(st_anchor_quat, rand_pos) + st_anchor_pos
        )
        self.target_orientation_w[st_eids, s] = quat_mul(st_anchor_quat, rand_quat)
        vel_mean_st = self._target_vel_means_t[st_mids, s]
        vel_std_st = self._target_vel_stds_t[st_mids, s]
        rand_vel_st = vel_mean_st + torch.randn(ns, 3, device=self.device) * vel_std_st
        self.target_velocity_w[st_eids, s] = quat_apply(st_anchor_quat, rand_vel_st)

  # ------------------------------------------------------------------
  # Source position / orientation helpers (body or site)
  # ------------------------------------------------------------------

  def get_source_pos_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 3)`` world-frame source positions."""
    return self._source_pos_w_cache

  def get_source_quat_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 4)`` world-frame source quaternions."""
    return self._source_quat_w_cache

  def get_source_lin_vel_w(self) -> torch.Tensor:
    """Return ``(num_envs, max_subtargets, 3)`` world-frame source linear velocities."""
    return self._source_lin_vel_w_cache

  # ------------------------------------------------------------------
  # Metrics
  # ------------------------------------------------------------------

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    self.metrics["error_target_pos"] = torch.norm(
      self.get_source_pos_w() - self.target_position_w, dim=-1
    ).mean(dim=-1)

  # ------------------------------------------------------------------
  # Adaptive sampling
  # ------------------------------------------------------------------

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    episode_failed = self._env.termination_manager.terminated[env_ids]

    if torch.any(episode_failed):
      for i in range(len(self.motion_loaders)):
        motion_mask = self.which_motion[env_ids] == i
        failed_mask = episode_failed & motion_mask
        if torch.any(failed_mask):
          current_bin_index = torch.clamp(
            (self.time_steps[env_ids[failed_mask]] * self.bin_counts[i])
            // max(self.motion_loaders[i].time_step_total, 1),
            0,
            self.bin_counts[i] - 1,
          )
          self._current_bin_failed[i] = torch.bincount(
            current_bin_index, minlength=self.bin_counts[i]
          )

    sampling_probs_per_motion: list[torch.Tensor] = []
    for i in range(len(self.motion_loaders)):
      probs = self.bin_failed_counts[i] + self.cfg.adaptive_uniform_ratio / float(
        self.bin_counts[i]
      )
      probs = torch.nn.functional.pad(
        probs.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
      probs = probs / probs.sum()
      sampling_probs_per_motion.append(probs)

    sampled_bins = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    for i in range(len(self.motion_loaders)):
      motion_mask = self.which_motion[env_ids] == i
      if torch.any(motion_mask):
        num_samples = int(motion_mask.sum().item())
        sampled = torch.multinomial(
          sampling_probs_per_motion[i], num_samples, replacement=True
        )
        sampled_bins[motion_mask] = sampled

    motion_indices = self.which_motion[env_ids]
    bin_counts_for_envs = torch.tensor(
      [self.bin_counts[m] for m in motion_indices.tolist()],
      device=self.device,
    )
    timestep_totals = torch.tensor(
      [self.motion_loaders[m].time_step_total for m in motion_indices.tolist()],
      device=self.device,
    )

    random_offsets = torch.rand(len(env_ids), device=self.device)
    new_timesteps = (
      (sampled_bins.float() + random_offsets)
      / bin_counts_for_envs.float()
      * (timestep_totals.float() - 1)
    ).long()
    self.time_steps[env_ids] = new_timesteps

    for i in range(len(self.motion_loaders)):
      motion_mask = self.which_motion[env_ids] == i
      if torch.any(motion_mask):
        probs = sampling_probs_per_motion[i]
        h = -(probs * (probs + 1e-12).log()).sum()
        h_norm = h / math.log(self.bin_counts[i]) if self.bin_counts[i] > 1 else 1.0
        pmax, imax = probs.max(dim=0)
        env_indices = torch.where(motion_mask)[0]
        actual_env_ids = env_ids[env_indices]
        self.metrics["sampling_entropy"][actual_env_ids] = h_norm
        self.metrics["sampling_top1_prob"][actual_env_ids] = pmax
        self.metrics["sampling_top1_bin"][actual_env_ids] = (
          imax.float() / self.bin_counts[i]
        )

  # ------------------------------------------------------------------
  # Resampling
  # ------------------------------------------------------------------

  def _sample_motion_ids(self, n: int) -> torch.Tensor:
    """Sample *n* motion indices according to ``motion_sampling_weights``.

    Weights are honoured proportionally: each motion gets exactly
    ``round(weight * n)`` slots, with any remainder filled by multinomial
    sampling.  The result is shuffled before being returned.
    """
    num_motions = len(self.motion_loaders)
    counts = (self._motion_weights_t * n).floor().long()
    remainder = n - counts.sum().item()
    if remainder > 0:
      extra = torch.multinomial(
        self._motion_weights_t, int(remainder), replacement=True
      )
      for idx in extra.tolist():
        counts[idx] += 1
    ids = torch.repeat_interleave(torch.arange(num_motions, device=self.device), counts)
    return ids[torch.randperm(n, device=self.device)]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    self.which_motion[env_ids] = self._sample_motion_ids(len(env_ids))

    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      motion_indices = self.which_motion[env_ids]
      timestep_totals = torch.tensor(
        [self.motion_loaders[m].time_step_total for m in motion_indices.tolist()],
        device=self.device,
      )
      self.time_steps[env_ids] = (
        torch.rand(len(env_ids), device=self.device) * (timestep_totals.float() - 1)
      ).long()
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids],
      soft_joint_pos_limits[:, :, 0],
      soft_joint_pos_limits[:, :, 1],
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

    # _sample_targets reads body/site world transforms; ensure they reflect
    # the newly written reset state before sampling.
    self._env.sim.forward()
    self._sample_targets(env_ids)
    self._update_source_cache()

  def _sample_next_motion(self, env_ids: torch.Tensor) -> None:
    """Pick a new random motion from the start without resetting robot."""
    if len(env_ids) == 0:
      return
    self.which_motion[env_ids] = self._sample_motion_ids(len(env_ids))
    self.time_steps[env_ids] = 0
    self._sample_targets(env_ids)

  # ------------------------------------------------------------------
  # Step update
  # ------------------------------------------------------------------

  def _update_command(self) -> None:
    self.time_steps += 1

    # Update moving targets each step — fully vectorized over (E, S).
    has_moving = self._has_target_link[self.which_motion]  # (E, S)
    if torch.any(has_moving):
      E = self.num_envs
      S = self.max_subtargets
      motion_ids = self.which_motion  # (E,)
      all_env_ids = torch.arange(E, device=self.device)
      tgt_pos_all, tgt_quat_all = self._fetch_link_state_batched(
        all_env_ids,
        self._target_body_indices_t[motion_ids],  # (E, S)
        self._target_is_site_t[motion_ids],  # (E, S)
      )  # (E, S, 3/4)
      # Use the per-episode random offset sampled at reset (target-body local frame),
      # rotated by the live target-body orientation each step.
      rand_offset_all = self._moving_target_offset_w[all_env_ids]  # (E, S, 3), body local
      world_offset_all = quat_apply(
        tgt_quat_all.reshape(-1, 4),
        rand_offset_all.reshape(-1, 3),
      ).reshape(E, S, 3)
      self.target_position_w[has_moving] = (tgt_pos_all + world_offset_all)[has_moving]
      self.target_orientation_w[has_moving] = tgt_quat_all[has_moving]

    # Handle motion completion with between-motion pause.
    timestep_limits = self._time_step_totals[self.which_motion]
    env_ids_at_end = torch.where(self.time_steps >= timestep_limits)[0]

    newly_paused = env_ids_at_end[~self.is_paused[env_ids_at_end]]
    if len(newly_paused) > 0:
      lo, hi = self.between_motion_pause_range
      self._sampled_pause_lengths[newly_paused] = (
        torch.rand(len(newly_paused), device=self.device) * (hi - lo) + lo
      )
      motion_ids = self.which_motion[newly_paused]
      self._paused_body_pos_w[newly_paused] = (
        self._stacked_body_pos_w[motion_ids, 0]
        + self._env.scene.env_origins[newly_paused, None, :]
      )
      self._paused_body_quat_w[newly_paused] = self._stacked_body_quat_w[motion_ids, 0]
      self._paused_joint_pos[newly_paused] = self._stacked_joint_pos[motion_ids, 0]
      self._paused_joint_vel[newly_paused] = self._stacked_joint_vel[motion_ids, 0]

    self.is_paused[:] = False
    self.is_paused[env_ids_at_end] = True
    self.between_motion_pause_time[env_ids_at_end] += self._env.step_dt
    env_ids_to_continue = env_ids_at_end[
      self.between_motion_pause_time[env_ids_at_end]
      >= self._sampled_pause_lengths[env_ids_at_end]
    ]
    self.between_motion_pause_time[env_ids_to_continue] = 0.0

    if len(env_ids_to_continue) > 0:
      self._sample_next_motion(env_ids_to_continue)

    # Update relative body poses.
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      for i in range(len(self.motion_loaders)):
        self.bin_failed_counts[i] = (
          self.cfg.adaptive_alpha * self._current_bin_failed[i]
          + (1 - self.cfg.adaptive_alpha) * self.bin_failed_counts[i]
        )
        self._current_bin_failed[i].zero_()

    self._update_source_cache()

  # ------------------------------------------------------------------
  # Source-state cache
  # ------------------------------------------------------------------

  def _update_source_cache(self) -> None:
    """Fetch source link pos/quat/vel once and store in cache tensors.

    Called at the end of every _update_command and after _sample_targets so
    that get_source_pos_w / get_source_quat_w / get_source_lin_vel_w are
    simple tensor reads with no redundant data fetches.
    """
    all_env_ids = torch.arange(self.num_envs, device=self.device)
    source_indices = self._source_body_indices_t[self.which_motion]  # (E, S)
    source_is_site = self._source_is_site_t[self.which_motion]  # (E, S)

    pos, quat = self._fetch_link_state_batched(
      all_env_ids, source_indices, source_is_site
    )
    self._source_pos_w_cache = pos
    self._source_quat_w_cache = quat

    E, S = source_indices.shape
    env_ids_exp = all_env_ids[:, None].expand(E, S)
    body_idx = source_indices.clamp(
      max=self.robot.data.body_link_lin_vel_w.shape[1] - 1
    )
    site_idx = source_indices.clamp(max=self.robot.data.site_lin_vel_w.shape[1] - 1)
    body_vel = self.robot.data.body_link_lin_vel_w[env_ids_exp, body_idx]
    site_vel = self.robot.data.site_lin_vel_w[env_ids_exp, site_idx]
    self._source_lin_vel_w_cache = torch.where(
      source_is_site.unsqueeze(-1), site_vel, body_vel
    )

  # ------------------------------------------------------------------
  # Visualization
  # ------------------------------------------------------------------

  _VIZ_LINK_COLORS = ((1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0))

  def _add_target_vis(self, visualizer: DebugVisualizer, batch: int) -> None:
    """Draw target spheres and source/target link frames for one env (all sub-targets)."""
    source_pos_all = self.get_source_pos_w()[batch].cpu().numpy()  # (S, 3)
    source_quat_all = self.get_source_quat_w()[batch]  # (S, 4)
    num_active = len(self.motion_configs[self.which_motion[batch].item()].sub_targets)
    for s in range(num_active):
      target_pos = self.target_position_w[batch, s].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.05,
        color=(0.0, 1.0, 0.0, 0.7),
        label=f"target_{batch}_{s}",
      )
      source_rotm = matrix_from_quat(source_quat_all[s].unsqueeze(0))[0].cpu().numpy()
      visualizer.add_frame(
        position=source_pos_all[s],
        rotation_matrix=source_rotm,
        scale=0.12,
        label=f"source_{batch}_{s}",
        axis_colors=self._VIZ_LINK_COLORS,
      )
      target_rotm = (
        matrix_from_quat(self.target_orientation_w[batch, s].unsqueeze(0))[0]
        .cpu()
        .numpy()
      )
      visualizer.add_frame(
        position=target_pos,
        rotation_matrix=target_rotm,
        scale=0.12,
        label=f"target_frame_{batch}_{s}",
        axis_colors=self._VIZ_LINK_COLORS,
      )
      target_vel = self.target_velocity_w[batch, s].cpu().numpy()
      vel_norm = np.linalg.norm(target_vel)
      if vel_norm > 1e-4:
        visualizer.add_arrow(
          start=target_pos,
          end=target_pos + target_vel / vel_norm * 0.3,
          color=(1.0, 0.5, 0.0, 0.9),
          label=f"target_vel_{batch}_{s}",
        )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_relative_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_relative_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
        self._add_target_vis(visualizer, batch)

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_rotm = matrix_from_quat(self.body_quat_w[batch]).cpu().numpy()
        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_rotm = (
          matrix_from_quat(self.robot_body_quat_w[batch]).cpu().numpy()
        )

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )
        self._add_target_vis(visualizer, batch)


@dataclass(kw_only=True)
class MultiTargetMotionCommandCfg(CommandTermCfg):
  """Configuration for the multi-target motion command."""

  entity_name: str
  motion_files: list[str] = field(default_factory=list)
  anchor_body_name: str = ""
  body_names: tuple[str, ...] = ()

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)

  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  motion_sampling_weights: list[float] = field(default_factory=list)
  """Per-motion sampling proportions. Defaults to uniform when empty."""

  motion_target_cfgs: list[MotionCfg] = field(default_factory=list)
  """Per-motion target configurations, each containing one or more sub-targets."""

  between_motion_pause_range: tuple[float, float] = (0.0, 1.0)
  """(min, max) seconds for the between-motion pause, sampled uniformly per env."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiTargetMotionCommand:
    return MultiTargetMotionCommand(self, env)
