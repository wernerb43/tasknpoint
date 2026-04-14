"""Unitree G1 flat estimation environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.estimation.estimation_env_cfg import make_estimation_env_cfg
from mjlab.tasks.estimation.mdp import (
  EstimationMotionCommandCfg,
  LowLevelTrackerActionCfg,
)

# Bodies tracked by the multi-target tracker on the G1. Must match the
# tracker's training config exactly.
_G1_TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def unitree_g1_flat_estimation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain estimation configuration.

  The inner joint-position action is configured to match the G1 tracking
  task (``scale = G1_ACTION_SCALE``, ``use_default_offset = True``), so the
  pretrained tracker sees the same action transform it was trained with.
  """
  cfg = make_estimation_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # Configure the inner joint-position action to match the tracking task.
  tracker_action = cfg.actions["tracker"]
  assert isinstance(tracker_action, LowLevelTrackerActionCfg)
  assert isinstance(tracker_action.inner_action, JointPositionActionCfg)
  tracker_action.inner_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, EstimationMotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = _G1_TRACKED_BODIES

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  if play:
    # Effectively infinite episode length so the viewer stays up.
    cfg.episode_length_s = int(1e9)

    # Disable terminations so the tracker can hold a pose indefinitely
    # without the env resetting under it.
    cfg.terminations = {}

    # Disable observation corruption in every group during play.
    for group in cfg.observations.values():
      group.enable_corruption = False

    # Deterministic motion start, no RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
