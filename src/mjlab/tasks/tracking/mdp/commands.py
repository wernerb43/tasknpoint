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


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
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

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

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
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

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

  def _update_metrics(self):
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

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
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
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
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

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

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
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
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
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

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

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


@dataclass
class MotionTargetCfg:
  """Per-motion target configuration."""

  source_link: str
  source_type: Literal["body", "site"] = "body"
  target_link: str | None = None
  target_type: Literal["body", "site"] = "body"
  target_pos_mean: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_pos_std: dict[str, float] = field(
    default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
  )
  target_euler_angle_range: dict[str, tuple[float, float]] = field(
    default_factory=lambda: {
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
  )
  target_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  target_euler_angle_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  target_phase_start: float = 0.0
  target_phase_end: float = 1.0


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

    def _get(lst: list, idx: int, default):  # noqa: ANN001, ANN202
      return lst[idx] if idx < len(lst) else default

    # Build per-motion configs and loaders.
    self.motion_loaders: list[MotionLoader] = []
    self.motion_configs: list[MotionTargetCfg] = []
    for i, motion_file in enumerate(self.cfg.motion_files):
      self.motion_loaders.append(
        MotionLoader(motion_file, self.body_indexes, device=self.device)
      )

      source_types = self.cfg.source_link_types
      target_types = self.cfg.target_link_types
      self.motion_configs.append(
        MotionTargetCfg(
          source_link=self.cfg.source_link_names[i],
          source_type=source_types[i] if source_types else "body",
          target_link=_get(self.cfg.target_link_names, i, None),
          target_type=target_types[i] if target_types else "body",
          target_pos_mean=_get(
            self.cfg.target_pos_means,
            i,
            {"x": 0.0, "y": 0.0, "z": 0.0},
          ),
          target_pos_std=_get(
            self.cfg.target_pos_stds,
            i,
            {"x": 0.0, "y": 0.0, "z": 0.0},
          ),
          target_euler_angle_range=_get(
            self.cfg.target_euler_angle_ranges,
            i,
            {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
          ),
          target_pos_offset=_get(self.cfg.target_pos_offsets, i, (0.0, 0.0, 0.0)),
          target_euler_angle_offset=_get(
            self.cfg.target_euler_angle_offsets, i, (0.0, 0.0, 0.0)
          ),
          target_phase_start=_get(self.cfg.target_phase_starts, i, 0.0),
          target_phase_end=_get(self.cfg.target_phase_ends, i, 1.0),
        )
      )

    # Source/target indices per motion (body or site depending on type).
    self.source_body_indices: list[int] = []
    self.source_is_site: list[bool] = []
    self.target_body_indices: list[int | None] = []
    self.target_is_site: list[bool] = []
    for mc in self.motion_configs:
      if mc.source_type == "site":
        idx = self.robot.find_sites(mc.source_link, preserve_order=True)[0][0]
        self.source_body_indices.append(idx)
        self.source_is_site.append(True)
      else:
        self.source_body_indices.append(self.robot.body_names.index(mc.source_link))
        self.source_is_site.append(False)
      if mc.target_link is not None:
        if mc.target_type == "site":
          idx = self.robot.find_sites(mc.target_link, preserve_order=True)[0][0]
          self.target_body_indices.append(idx)
          self.target_is_site.append(True)
        else:
          self.target_body_indices.append(self.robot.body_names.index(mc.target_link))
          self.target_is_site.append(False)
      else:
        self.target_body_indices.append(None)
        self.target_is_site.append(False)

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

    # Pre-compute per-motion indices as tensors.
    self._source_body_indices_t = torch.tensor(
      self.source_body_indices, device=self.device, dtype=torch.long
    )
    self._target_body_indices_t = torch.tensor(
      [idx if idx is not None else 0 for idx in self.target_body_indices],
      device=self.device,
      dtype=torch.long,
    )
    self._has_target_link = torch.tensor(
      [mc.target_link is not None for mc in self.motion_configs],
      device=self.device,
      dtype=torch.bool,
    )
    self._source_is_site_t = torch.tensor(
      self.source_is_site, device=self.device, dtype=torch.bool
    )
    self._target_is_site_t = torch.tensor(
      self.target_is_site, device=self.device, dtype=torch.bool
    )

    # Pre-compute target sampling parameters as tensors.
    self._target_pos_means_t = torch.stack(
      [
        torch.tensor(
          [mc.target_pos_mean.get(k, 0.0) for k in ["x", "y", "z"]],
          device=self.device,
        )
        for mc in self.motion_configs
      ]
    )
    self._target_pos_stds_t = torch.stack(
      [
        torch.tensor(
          [mc.target_pos_std.get(k, 0.0) for k in ["x", "y", "z"]],
          device=self.device,
        )
        for mc in self.motion_configs
      ]
    )
    self._target_pos_offsets_t = torch.stack(
      [
        torch.tensor(mc.target_pos_offset, device=self.device, dtype=torch.float32)
        for mc in self.motion_configs
      ]
    )
    self._target_euler_ranges_t = torch.stack(
      [
        torch.tensor(
          [
            mc.target_euler_angle_range.get(k, (0.0, 0.0))
            for k in ["roll", "pitch", "yaw"]
          ],
          device=self.device,
        )
        for mc in self.motion_configs
      ]
    )  # (num_motions, 3, 2)
    self._target_offset_quats = torch.stack(
      [
        quat_from_euler_xyz(
          torch.tensor([mc.target_euler_angle_offset[0]], device=self.device),
          torch.tensor([mc.target_euler_angle_offset[1]], device=self.device),
          torch.tensor([mc.target_euler_angle_offset[2]], device=self.device),
        ).squeeze(0)
        for mc in self.motion_configs
      ]
    )
    self._target_phase_starts_t = torch.tensor(
      [mc.target_phase_start for mc in self.motion_configs],
      device=self.device,
    )
    self._target_phase_ends_t = torch.tensor(
      [mc.target_phase_end for mc in self.motion_configs],
      device=self.device,
    )

    # Motion sampling weights — normalised to sum to 1.
    num_motions = len(self.motion_loaders)
    raw_weights = self.cfg.motion_sampling_weights
    if raw_weights:
      assert len(raw_weights) == num_motions, (
        f"motion_sampling_weights length ({len(raw_weights)}) must match "
        f"number of motions ({num_motions})"
      )
      w = torch.tensor(raw_weights, dtype=torch.float32, device=self.device)
    else:
      w = torch.ones(num_motions, dtype=torch.float32, device=self.device)
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

    # Target tracking (world frame).
    self.target_position_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_orientation_w = torch.zeros(self.num_envs, 4, device=self.device)
    self.target_orientation_w[:, 0] = 1.0

    # Target tracking (body frame, for observation).

    # Target phase windows per env.
    self.target_phase_start = torch.zeros(self.num_envs, device=self.device)
    self.target_phase_end = torch.zeros(self.num_envs, device=self.device)

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
    self.between_motion_pause_length = self.cfg.between_motion_pause_length
    self.between_motion_pause_time = torch.zeros(self.num_envs, device=self.device)
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
    """Joint pos/vel + target position/orientation in anchor frame."""
    robot_quat_inv = quat_inv(self.robot_anchor_quat_w)
    target_pos_b = quat_apply(
      robot_quat_inv, self.target_position_w - self.robot_anchor_pos_w
    )
    target_ori_b = quat_mul(robot_quat_inv, self.target_orientation_w)
    return torch.cat(
      [
        self.joint_pos,
        self.joint_vel,
        target_pos_b,
        target_ori_b,
      ],
      dim=1,
    )

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

  def _fetch_target_pos_quat(
    self, env_ids: torch.Tensor, motion_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Read world-frame position and quaternion of the target link for each env."""
    return self._fetch_link_state(
      env_ids,
      self._target_body_indices_t[motion_ids],
      self._target_is_site_t[motion_ids],
    )

  # ------------------------------------------------------------------
  # Target sampling
  # ------------------------------------------------------------------

  def _sample_targets(self, env_ids: torch.Tensor) -> None:
    """Sample target positions/orientations for *env_ids*."""
    motion_ids = self.which_motion[env_ids]

    # Moving targets (tracking another body link or site).
    has_moving = self._has_target_link[motion_ids]
    if torch.any(has_moving):
      moving_env_ids = env_ids[has_moving]
      moving_motion_ids = motion_ids[has_moving]
      target_pos, target_quat = self._fetch_target_pos_quat(
        moving_env_ids, moving_motion_ids
      )
      offset_pos_w = quat_apply(
        target_quat, self._target_pos_offsets_t[moving_motion_ids]
      )
      self.target_position_w[moving_env_ids] = target_pos + offset_pos_w
      self.target_orientation_w[moving_env_ids] = quat_mul(
        target_quat, self._target_offset_quats[moving_motion_ids]
      )

    # Static targets (sampled from Gaussian).
    has_static = ~has_moving
    if torch.any(has_static):
      static_env_ids = env_ids[has_static]
      static_motion_ids = motion_ids[has_static]
      ns = len(static_env_ids)

      pos_mean = self._target_pos_means_t[static_motion_ids]
      pos_std = self._target_pos_stds_t[static_motion_ids] * self.target_pos_std_scale
      rand_pos = pos_mean + torch.randn(ns, 3, device=self.device) * pos_std

      euler_ranges = self._target_euler_ranges_t[static_motion_ids]
      rand_euler = euler_ranges[:, :, 0] + torch.rand(ns, 3, device=self.device) * (
        euler_ranges[:, :, 1] - euler_ranges[:, :, 0]
      )
      rand_quat = quat_from_euler_xyz(
        rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2]
      )

      anchor_pos_w = self.robot.data.body_link_pos_w[
        static_env_ids, self.robot_anchor_body_index
      ]
      anchor_quat_w = self.robot.data.body_link_quat_w[
        static_env_ids, self.robot_anchor_body_index
      ]
      # rand_pos is expressed in anchor-body frame; rotate offset the same way.
      offset_pos_w = quat_apply(
        anchor_quat_w, self._target_pos_offsets_t[static_motion_ids]
      )
      self.target_position_w[static_env_ids] = (
        quat_apply(anchor_quat_w, rand_pos) + anchor_pos_w + offset_pos_w
      )

      offset_quats = self._target_offset_quats[static_motion_ids]
      self.target_orientation_w[static_env_ids] = quat_mul(
        quat_mul(anchor_quat_w, rand_quat), offset_quats
      )

    self.target_phase_start[env_ids] = self._target_phase_starts_t[motion_ids]
    self.target_phase_end[env_ids] = self._target_phase_ends_t[motion_ids]

  # ------------------------------------------------------------------
  # Source position / orientation helpers (body or site)
  # ------------------------------------------------------------------

  def get_source_pos_w(self) -> torch.Tensor:
    """Return (num_envs, 3) world-frame positions of each env's active source."""
    pos, _ = self._fetch_link_state(
      torch.arange(self.num_envs, device=self.device),
      self._source_body_indices_t[self.which_motion],
      self._source_is_site_t[self.which_motion],
    )
    return pos

  def get_source_quat_w(self) -> torch.Tensor:
    """Return (num_envs, 4) world-frame quaternions of each env's active source."""
    _, quat = self._fetch_link_state(
      torch.arange(self.num_envs, device=self.device),
      self._source_body_indices_t[self.which_motion],
      self._source_is_site_t[self.which_motion],
    )
    return quat

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
    )

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

    # Update moving targets each step.
    envs_with_moving_target = self._has_target_link[self.which_motion]
    if torch.any(envs_with_moving_target):
      env_indices = torch.where(envs_with_moving_target)[0]
      motion_ids = self.which_motion[env_indices]
      target_pos, target_quat = self._fetch_target_pos_quat(env_indices, motion_ids)
      offset_pos_w = quat_apply(target_quat, self._target_pos_offsets_t[motion_ids])
      self.target_position_w[env_indices] = target_pos + offset_pos_w
      self.target_orientation_w[env_indices] = quat_mul(
        target_quat, self._target_offset_quats[motion_ids]
      )

    # Handle motion completion with between-motion pause.
    timestep_limits = self._time_step_totals[self.which_motion]
    env_ids_at_end = torch.where(self.time_steps >= timestep_limits)[0]

    newly_paused = env_ids_at_end[~self.is_paused[env_ids_at_end]]
    if len(newly_paused) > 0:
      self._paused_body_pos_w[newly_paused] = self.robot.data.body_link_pos_w[
        newly_paused
      ][:, self.body_indexes]
      self._paused_body_quat_w[newly_paused] = self.robot.data.body_link_quat_w[
        newly_paused
      ][:, self.body_indexes]
      self._paused_joint_pos[newly_paused] = self.robot.data.joint_pos[newly_paused]
      self._paused_joint_vel[newly_paused] = self.robot.data.joint_vel[newly_paused]

    self.is_paused[:] = False
    self.is_paused[env_ids_at_end] = True
    self.between_motion_pause_time[env_ids_at_end] += self._env.step_dt
    env_ids_to_continue = env_ids_at_end[
      self.between_motion_pause_time[env_ids_at_end] >= self.between_motion_pause_length
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

  # ------------------------------------------------------------------
  # Visualization
  # ------------------------------------------------------------------

  _VIZ_LINK_COLORS = ((1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0))

  def _add_target_vis(self, visualizer: DebugVisualizer, batch: int) -> None:
    """Draw target sphere and source/target link frames for one env."""
    target_pos = self.target_position_w[batch].cpu().numpy()
    visualizer.add_sphere(
      center=target_pos,
      radius=0.05,
      color=(0.0, 1.0, 0.0, 0.7),
      label=f"target_{batch}",
    )
    source_pos = self.get_source_pos_w()[batch].cpu().numpy()
    source_rotm = (
      matrix_from_quat(self.get_source_quat_w()[batch].unsqueeze(0))[0].cpu().numpy()
    )
    visualizer.add_frame(
      position=source_pos,
      rotation_matrix=source_rotm,
      scale=0.12,
      label=f"source_{batch}",
      axis_colors=self._VIZ_LINK_COLORS,
    )
    target_rotm = (
      matrix_from_quat(self.target_orientation_w[batch].unsqueeze(0))[0].cpu().numpy()
    )
    visualizer.add_frame(
      position=target_pos,
      rotation_matrix=target_rotm,
      scale=0.12,
      label=f"target_frame_{batch}",
      axis_colors=self._VIZ_LINK_COLORS,
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
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
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
  """Per-motion sampling proportions (e.g. ``[0.33, 0.33, 0.34]``).
  Must sum to approximately 1.0 and have one entry per motion.
  Defaults to uniform sampling when empty."""

  source_link_names: list[str] = field(default_factory=list)
  """Per-motion source link names (the link that should reach the target)."""

  source_link_types: list[Literal["body", "site"]] | None = None
  """Per-motion source link types. Defaults to ``"body"`` for all motions."""

  target_link_names: list[str | None] = field(default_factory=list)
  """Per-motion target link names. ``None`` means static target (Gaussian
  sampled)."""

  target_link_types: list[Literal["body", "site"]] | None = None
  """Per-motion target link types. Defaults to ``"body"`` for all motions."""

  target_pos_means: list[dict[str, float]] = field(default_factory=list)
  """Per-motion nominal target positions ``{"x": ..., "y": ..., "z": ...}``."""

  target_pos_stds: list[dict[str, float]] = field(default_factory=list)
  """Per-motion target position standard deviations (max, scaled by
  curriculum)."""

  target_euler_angle_ranges: list[dict[str, tuple[float, float]]] = field(
    default_factory=list
  )
  """Per-motion target orientation euler-angle ranges."""

  target_pos_offsets: list[tuple[float, float, float]] = field(default_factory=list)
  """Per-motion position offsets applied after sampling."""

  target_euler_angle_offsets: list[tuple[float, float, float]] = field(
    default_factory=list
  )
  """Per-motion orientation offsets (euler angles) applied after sampling."""

  target_phase_starts: list[float] = field(default_factory=list)
  """Per-motion phase window start (0.0-1.0)."""

  target_phase_ends: list[float] = field(default_factory=list)
  """Per-motion phase window end (0.0-1.0)."""

  between_motion_pause_length: float = 0.3
  """Seconds the reference is held frozen at the final motion pose before a
  new motion is sampled. Increase (e.g. 1e9) to make the command hold the
  pose indefinitely, which is useful when evaluating a low-level tracker in
  an estimation/playback setting."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MultiTargetMotionCommand:
    return MultiTargetMotionCommand(self, env)
