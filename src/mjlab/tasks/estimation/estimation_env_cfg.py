"""Estimation environment configuration.

The goal of this task is, given a pretrained low-level tracking policy, learn
a map from observations of a ball flying along a ballistic trajectory to a
target position + motion choice + motion timing that lets the robot
intercept the ball (hit, catch, etc).

The current MVP only stands up the *interface* to the low-level tracker so
that the play script can be used to verify the tracker runs inside this
environment and holds a pose (via the tracker's existing between-motion
pause logic). Rewards, ball commands, and the high-level policy's own
observations are intentionally left as stubs and will be filled in
incrementally.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.estimation import mdp
from mjlab.tasks.estimation.mdp import (
  BallCommandCfg,
  EstimationMotionCommandCfg,
  LowLevelTrackerActionCfg,
  TriggerActionCfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import MOTIONS, VELOCITY_RANGE
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_estimation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create a base estimation environment configuration.

  Wires the multi-target motion command and a :class:`LowLevelTrackerAction`
  that consumes the tracker-observation group at inference time. Rewards
  and the high-level ball/target command are intentionally empty for now.
  """

  ##
  # Observations
  ##

  # The "tracker" group MUST match the actor observations of the multi-target
  # tracking task (order and term semantics), so the pretrained tracker sees
  # the same input distribution it was trained on.
  tracker_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    # Read the tracker's own previous output, not the high-level action.
    "actions": ObservationTermCfg(
      func=mdp.last_action, params={"action_name": "tracker"}
    ),
  }

  actor_terms = {
    "ball_pos_b": ObservationTermCfg(
      func=mdp.ball_pos_b,
      params={"command_name": "ball"},
      history_length=30,
    ),
  }

  critic_terms = {
    "ball_pos_b": ObservationTermCfg(
      func=mdp.ball_pos_b,
      params={"command_name": "ball"},
      history_length=30,
    ),
    "ball_pos_root_centered_w": ObservationTermCfg(
      func=mdp.ball_pos_root_centered_w,
      params={"command_name": "ball"},
    ),
    "ball_vel_w": ObservationTermCfg(
      func=mdp.ball_vel_w,
      params={"command_name": "ball"},
    ),
  }

  observations = {
    "tracker": ObservationGroupCfg(
      terms=tracker_terms,
      concatenate_terms=True,
      # Use the same corruption as training so the tracker's input statistics
      # match (the tracker was trained against corrupted observations).
      enable_corruption=True,
    ),
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "trigger": TriggerActionCfg(
      entity_name="robot",
    ),
    "tracker": LowLevelTrackerActionCfg(
      entity_name="robot",
      tracker_obs_group="tracker",
    ),
  }

  ##
  # Commands — reuse the multi-target motion command. Per-motion details are
  # pulled from the tracking task so the structure the tracker saw at
  # training is preserved verbatim.
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": EstimationMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      motion_files=[],  # Set per-play from the tracker's W&B run.
      anchor_body_name="",  # Set per-robot.
      body_names=(),  # Set per-robot.
      motion_sampling_weights=[m.sampling_weight for m in MOTIONS],
      source_link_names=[m.source_link for m in MOTIONS],
      source_link_types=[m.source_type for m in MOTIONS],
      target_link_names=[m.target_link for m in MOTIONS],
      target_link_types=[m.target_type for m in MOTIONS],
      target_pos_means=[m.target_pos_mean for m in MOTIONS],
      target_pos_stds=[m.target_pos_std for m in MOTIONS],
      target_euler_angle_ranges=[m.target_euler_range for m in MOTIONS],
      target_pos_offsets=[m.target_pos_offset for m in MOTIONS],
      target_euler_angle_offsets=[m.target_euler_offset for m in MOTIONS],
      target_phase_starts=[m.phase_start for m in MOTIONS],
      target_phase_ends=[m.phase_end for m in MOTIONS],
      trigger_action_name="trigger",
    ),
    "ball": BallCommandCfg(
      motion_command_name="motion",
      start_y_offset=5.0,
      y_limit=-2.0,
      velocity=1.0,
    ),
  }

  ##
  # Events — startup-only DR so reset picks plausible friction/com. No
  # push_robot since we're not training.
  ##

  events: dict[str, EventTermCfg] = {
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
  }

  ##
  # Rewards — empty until the high-level ball/target task is defined.
  ##

  rewards: dict[str, RewardTermCfg] = {
    "hit_ball": RewardTermCfg(  # Reward for moving the target link to the ball position
      func=mdp.hit_ball_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.1,
      },
    ),
    "penalize_action": RewardTermCfg(  # Penalize performing the motion when we miss the ball, to encourage learning to time the motion correctly.
      func=mdp.penalize_action,
      weight=0.1,
      params={
        "command_name": "motion",
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
