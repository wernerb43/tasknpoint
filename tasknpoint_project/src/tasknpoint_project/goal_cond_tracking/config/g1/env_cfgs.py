"""Unitree G1 flat tracking environment configurations."""

import mujoco
from pathlib import Path
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from tasknpoint_project.goal_cond_tracking.mdp import MultiTargetMotionCommandCfg
from tasknpoint_project.goal_cond_tracking.tracking_env_cfg import (
  make_multi_target_tracking_env_cfg,
)

_LOCAL_XML = Path(__file__).parents[6] / "robots" / "g1_tennis_27dof.xml"


def _get_local_g1_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_LOCAL_XML))


def _get_g1_tennis_robot_cfg():
  cfg = get_g1_robot_cfg()
  cfg.spec_fn = _get_local_g1_spec
  return cfg


def unitree_g1_multi_target_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 multi-target tracking configuration."""
  cfg = make_multi_target_tracking_env_cfg()

  cfg.scene.entities = {"robot": _get_g1_tennis_robot_cfg()}

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

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiTargetMotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"  # TODO note that this can be changed to torso_link if we want torso imu instead of pelvis
  motion_cmd.body_names = (
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
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
