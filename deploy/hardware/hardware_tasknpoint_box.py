##
#
# Deployment code for Unitree G1 robot.
#
# Common G1 "hg" topics: https://support.unitree.com/home/en/G1_developer/dds_services_interface
#
##

# standard imports
import argparse
import numpy as np
import time
import threading

# other imports
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray, String
from geometry_msgs.msg import PoseStamped

# directory imports
import os
import sys
from pathlib import Path

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(Path(__file__).resolve().parents[1])
REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.append(DEPLOY_DIR)

# custom imports
from utils.math_utils import (
  quat_to_rotation_matrix,
)

# Unitree SDK imports
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
  MotionSwitcherClient,
)


########################################################################
# GLOBAL VARIABLES (DO NOT CHANGE)
########################################################################

G1_NUM_MOTOR = 29


class G1JointIndex:
  LeftHipPitch = 0
  LeftHipRoll = 1
  LeftHipYaw = 2
  LeftKnee = 3
  LeftAnklePitch = 4
  LeftAnkleB = 4
  LeftAnkleRoll = 5
  LeftAnkleA = 5
  RightHipPitch = 6
  RightHipRoll = 7
  RightHipYaw = 8
  RightKnee = 9
  RightAnklePitch = 10
  RightAnkleB = 10
  RightAnkleRoll = 11
  RightAnkleA = 11
  WaistYaw = 12
  WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
  WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15
  LeftShoulderRoll = 16
  LeftShoulderYaw = 17
  LeftElbow = 18
  LeftWristRoll = 19
  LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
  LeftWristYaw = 21  # NOTE: INVALID for g1 23dof
  RightShoulderPitch = 22
  RightShoulderRoll = 23
  RightShoulderYaw = 24
  RightElbow = 25
  RightWristRoll = 26
  RightWristPitch = 27  # NOTE: INVALID for g1 23dof
  RightWristYaw = 28  # NOTE: INVALID for g1 23dof


class Mode:
  PR = 0  # Series Control for Pitch/Roll Joints
  AB = 1  # Parallel Control for A/B Joints


# low-level control frequency
LOW_LEVEL_CONTROL_DT = 0.002  # [sec]

# ROS2 sensor publishing frequency
ROS_SENSOR_PUBLISH_DT = 0.01  # [sec]

# safety: max allowable pelvis roll/pitch before forcing damp (when you fall)
SAFETY_MAX_TILT = np.radians(60.0)  # [rad]

# Only switch motion when the new nominal is this much closer than the current one.
# Prevents rapid oscillation when the ball sits near a boundary between two targets.
_MOTION_SWITCH_HYSTERESIS = 0.25  # metres


########################################################################
# CONTROL
########################################################################


class ControlNode(Node):
  def __init__(self, config_path: str):
    super().__init__("hardware_node")

    # import config
    self.config = self.load_config(config_path)

    # load parameters
    self.load_params()

    # IMU states
    self.pelvis_imu_rpy = None  # roll, pitch, yaw
    self.pelvis_imu_quaternion = None  # orientation
    self.pelvis_imu_gyroscope = None  # angular velocity
    self.pelvis_imu_accelerometer = None  # linear acceleration

    # Joint states
    self.q = np.zeros(G1_NUM_MOTOR)  # joint positions
    self.dq = np.zeros(G1_NUM_MOTOR)  # joint velocities
    self.ddq = np.zeros(G1_NUM_MOTOR)  # joint accelerations
    self.tau_est = np.zeros(G1_NUM_MOTOR)  # estimated joint torques

    # command arrays
    self.q_cmd = np.array(self.default_joint_pos, dtype=np.float64)
    self.dq_cmd = np.zeros(G1_NUM_MOTOR)
    self.Kp_cmd = np.array(self.Kp, dtype=np.float64)
    self.Kd_cmd = np.array(self.Kd, dtype=np.float64)
    self.tau_ff_cmd = np.zeros(G1_NUM_MOTOR)

    # locks for thread safety
    self.fsm_lock = threading.Lock()  # protects FSM state
    self.sensor_lock = threading.Lock()  # protects sensor state arrays
    self.cmd_lock = threading.Lock()  # protects command arrays

    # finite state machine state
    self.fsm_state = "init"
    self.prev_fsm_state = "init"
    self.fsm_start_time = 0.0
    self.fsm_start_q = np.zeros(G1_NUM_MOTOR)
    self.fsm_time = 0.0

    # safety flags
    self.safety_triggered = False

    # perception: pelvis pose from external system (world frame)
    self.pelvis_pose_position   = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    self.pelvis_pose_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # box pose from perception (world frame)
    self.box_pose_position   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    self.box_pose_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # motion frame from control node
    self.motion_frame = 0
    self.motion_idx = 0

    # Otherhand grip-width toggle (Y button, rising-edge).
    # True  → nominal offset from YAML (e.g. [0, 0.25, 0]).
    # False → wide offset [0, 0.40, 0].
    self._otherhand_wide  = True
    self._prev_y_pressed  = False

    # other stuff from unitree's example
    self.time_ = 0.0
    self.mode_machine_ = 0
    self.low_cmd = unitree_hg_msg_dds__LowCmd_()
    self.low_state = None
    self.update_mode_machine_ = False
    self.crc = CRC()

  #################################################################
  # INITIALIZATION
  #################################################################

  # load the config file
  def load_config(self, config_path: str):
    # open the config file and load it
    config_path_full = DEPLOY_DIR + "/configs/" + config_path
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)

    print("Config file loaded successfully from: [{}].".format(config_path_full))

    return config

  # load params from config
  def load_params(self):
    # time to interpolate to initial
    self.home_pos_duration = self.config["home_pos_duration"]  # float

    # default joint positions
    self.default_joint_pos = self.config["default_joint_pos"]  # list

    # motion and contact params
    contact_phase_cfg = self.config["contact_phase"]
    self.contact_phases = (
      contact_phase_cfg if isinstance(contact_phase_cfg, list) else [contact_phase_cfg]
    )
    self.contact_duration = float(self.config["contact_duration"])
    self.motion_num_frames = [
      int(np.load(mp if os.path.isabs(mp) else os.path.join(REPO_ROOT, mp))["joint_pos"].shape[0])
      for mp in self.config["motion_paths"]
    ]

    # Parse YAML goals to extract per-motion grasp offsets (mirrors simulation_box.py).
    goals_cfg = self.config.get("goals", [])
    _right_by_idx = {
      g["motion_index"]: np.array(g["vector"], dtype=np.float32)
      for g in goals_cfg
      if g["type"] == "position"
      and "left"       not in g["name"]
      and "otherhand"  not in g["name"]
    }
    _left_by_idx = {
      g["motion_index"]: np.array(g["vector"], dtype=np.float32)
      for g in goals_cfg
      if g["type"] == "position" and "left" in g["name"]
    }
    _otherhand_by_idx = {
      g["motion_index"]: np.array(g["vector"], dtype=np.float32)
      for g in goals_cfg
      if g["type"] == "position" and "otherhand" in g["name"]
    }
    n_motions = (max(max(_right_by_idx.keys()), max(_left_by_idx.keys())) + 1
                 if _right_by_idx else 1)
    _nominal_positions = np.array(
      [_right_by_idx[i] for i in range(n_motions)], dtype=np.float32
    )  # right-palm pelvis-frame position from YAML
    _left_positions = np.array(
      [_left_by_idx[i]  for i in range(n_motions)], dtype=np.float32
    )  # left-palm pelvis-frame position from YAML
    _default_otherhand = np.array([0.0, 0.20, 0.0], dtype=np.float32)
    self._otherhand_offsets = np.array(
      [_otherhand_by_idx.get(i, _default_otherhand) for i in range(n_motions)],
      dtype=np.float32,
    )  # (n_motions, 3) — constant offset in right-palm local frame

    # Box centre in pelvis frame = midpoint of right and left pre-grab targets.
    self._nominal_box_centers = (
      (_nominal_positions + _left_positions) / 2.0
    ).astype(np.float32)  # (n_motions, 3) — used for motion auto-selection

    # Per-motion grasp offsets from box centre (defined in box frame in the YAML).
    self._right_grasp_offsets = (
      _nominal_positions - self._nominal_box_centers
    ).astype(np.float32)  # (n_motions, 3)
    self._left_grasp_offsets = (
      _left_positions - self._nominal_box_centers
    ).astype(np.float32)  # (n_motions, 3)

    # Human-readable motion names derived from motion_paths filenames
    self._motion_names = [
      Path(mp).stem for mp in self.config["motion_paths"]
    ]

    # PD gains
    self.Kp = self.config["Kp"]  # list
    self.Kd = self.config["Kd"]  # list

    # type checks
    assert type(self.home_pos_duration) in [float], "home_pos_duration must be a float."
    assert type(self.default_joint_pos) == list, "default_joint_pos must be a list."
    assert type(self.Kp) == list, "Kp must be a list."
    assert type(self.Kd) == list, "Kd must be a list."

    # length checks
    assert len(self.Kp) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} Kp values, got {len(self.Kp)}."
    )
    assert len(self.Kd) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} Kd values, got {len(self.Kd)}."
    )

    # value checks
    assert self.home_pos_duration >= 3.0, (
      "home_pos_duration must take at least 3 seconds."
    )
    assert len(self.default_joint_pos) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} default joint positions, "
      f"got {len(self.default_joint_pos)}"
    )
    for i in range(G1_NUM_MOTOR):
      assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
      assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

    print("Config parameters loaded successfully.")

  # initialize the motion switcher client, publishers, and subscribers
  def Init(self):
    # initialize motion switcher client
    self.msc = MotionSwitcherClient()
    self.msc.SetTimeout(5.0)
    self.msc.Init()

    # wait until we have low-level control of the robot before proceeding
    status, result = self.msc.CheckMode()
    while result["name"]:
      self.msc.ReleaseMode()
      status, result = self.msc.CheckMode()
      time.sleep(1)

    # create publisher #
    self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.lowcmd_publisher_.Init()

    # create subscribers
    self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    print("Unitree SDK publishers and subscribers initialized successfully.")

    # ROS2 publishers
    self.pelvis_imu_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
    )
    self.joint_state_pub = self.create_publisher(
      Float32MultiArray, "deploy_robot/joint_state", 10
    )
    self.hardware_time_pub = self.create_publisher(
      Float64, "deploy_robot/hardware_time", 10
    )
    self.goals_pub = self.create_publisher(Float32MultiArray, "deploy_robot/goals", 10)
    self.which_motion_pub = self.create_publisher(
      Float64, "deploy_robot/which_motion", 10
    )

    # Hardware specific publishers
    self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 10)

    # ROS2 subscribers
    self.command_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/command", self.command_callback, 10
    )
    self.motion_frame_sub = self.create_subscription(
      Float64, "deploy_robot/motion_frame", self.motion_frame_callback, 10
    )

    # Hardware specific subscribers
    self.fsm_sub = self.create_subscription(
      String, "deploy_robot/fsm", self.fsm_callback, 10
    )

    # Perception subscribers
    self.pelvis_pose_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )
    self.box_pose_sub = self.create_subscription(
      PoseStamped, "/box/pose", self.box_pose_callback, 10
    )
    # Joystick subscriber for grip-width toggle (Y button)
    self.joystick_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/joystick", self._joystick_callback, 10
    )
    # sensor publish timer
    self.pub_timer = self.create_timer(ROS_SENSOR_PUBLISH_DT, self.publish_sensor_data)

    print("ROS2 publishers and subscribers initialized successfully.")

  # create a thread to run the low-level control loop
  def Start(self):
    # create a thread for low-level control loop, but do not start it yet
    self.lowCmdWriteThreadPtr = RecurrentThread(
      interval=LOW_LEVEL_CONTROL_DT, target=self.LowCmdWrite, name="control"
    )

    # wait until we receive the first low state message
    while self.update_mode_machine_ == False:
      time.sleep(1)

    # start the low-level control thread
    if self.update_mode_machine_ == True:
      self.lowCmdWriteThreadPtr.Start()
      print("Low-level robot control thread started successfully.")

  #################################################################
  # ROS PUBLISHING AND CALLBACKS
  #################################################################

  # callback to receive FSM state from joystick
  def fsm_callback(self, msg: String):
    with self.fsm_lock:
      self.fsm_state = msg.data

  # callback to receive command messages from ROS2
  def command_callback(self, msg: Float32MultiArray):
    # expected layout: [q(29), dq(29), Kp(29), Kd(29), tau_ff(29)] = 145 floats
    data = np.array(msg.data, dtype=np.float64)

    # safety check on command length
    if len(data) != 5 * G1_NUM_MOTOR:
      self.get_logger().warn(
        f"Expected {5 * G1_NUM_MOTOR} values in command, got {len(data)}"
      )
      return

    # update command arrays under lock
    nu = G1_NUM_MOTOR
    with self.cmd_lock:
      self.q_cmd[:] = data[0 * nu : 1 * nu]
      self.dq_cmd[:] = data[1 * nu : 2 * nu]
      self.Kp_cmd[:] = data[2 * nu : 3 * nu]
      self.Kd_cmd[:] = data[3 * nu : 4 * nu]
      self.tau_ff_cmd[:] = data[4 * nu : 5 * nu]

  # callback to receive motion frame from control node
  def motion_frame_callback(self, msg: Float64):
    self.motion_frame = int(msg.data)

  # publish sensor data to ROS2 topics
  def publish_sensor_data(self):
    # read sensor data under lock
    with self.sensor_lock:
      # pelvis IMU state
      pelvis_imu_rpy = (
        np.array(self.pelvis_imu_rpy, dtype=np.float64)
        if self.pelvis_imu_rpy is not None
        else np.zeros(3)
      )
      pelvis_imu_quat = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float64)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
      )
      pelvis_imu_gyro = (
        np.array(self.pelvis_imu_gyroscope, dtype=np.float64)
        if self.pelvis_imu_gyroscope is not None
        else np.zeros(3)
      )
      pelvis_imu_accel = (
        np.array(self.pelvis_imu_accelerometer, dtype=np.float64)
        if self.pelvis_imu_accelerometer is not None
        else np.zeros(3)
      )

      # joint state
      q = self.q.copy()
      dq = self.dq.copy()
      ddq = self.ddq.copy()
      tau_est = self.tau_est.copy()

    # imu_state: [rpy(3), quaternion(4), gyroscope(3), accelerometer(3)] = 13 floats
    pelvis_imu_msg = Float32MultiArray()
    pelvis_imu_msg.data = np.concatenate(
      [pelvis_imu_rpy, pelvis_imu_quat, pelvis_imu_gyro, pelvis_imu_accel]
    ).tolist()

    # joint_state: [q(29), dq(29), ddq(29), tau_est(29)] = 116 floats
    joint_msg = Float32MultiArray()
    joint_msg.data = np.concatenate([q, dq, ddq, tau_est]).tolist()

    # hardware_time: single float
    time_msg = Float64()
    time_msg.data = self.time_

    # fsm_time: time since entering current state
    fsm_time_msg = Float64()
    fsm_time_msg.data = self.fsm_time
    # Compute box-pickup goals (mirrors simulation_box.py _publish_goals).
    # Goals 1 & 2: box-frame grasp offsets rotated to pelvis frame via live box pose.
    # Goal 3: constant otherhand offset in right-palm local frame (no FK needed).
    goals_msg = Float32MultiArray()
    with self.sensor_lock:
      pelvis_pos_w = np.array(self.pelvis_pose_position, dtype=np.float32)
      pelvis_quat  = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float32)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
      )
      box_pos_w  = np.array(self.box_pose_position,   dtype=np.float32)
      box_quat   = np.array(self.box_pose_quaternion, dtype=np.float32)

    R             = quat_to_rotation_matrix(pelvis_quat)
    box_in_pelvis = R.T @ (box_pos_w - pelvis_pos_w)
    R_box         = quat_to_rotation_matrix(box_quat)

    right_goal     = (box_in_pelvis + R.T @ R_box @ self._right_grasp_offsets[self.motion_idx]).astype(np.float32)
    left_goal      = (box_in_pelvis + R.T @ R_box @ self._left_grasp_offsets[self.motion_idx]).astype(np.float32)
    otherhand_goal = (self._otherhand_offsets[self.motion_idx]           # [0, 0.25, 0] nominal
                      if self._otherhand_wide
                      else np.array([0.0, 0.5, 0.0], dtype=np.float32))  # [0, 0.40, 0] wide

    goals_msg.data = np.concatenate([right_goal, left_goal, otherhand_goal]).tolist()

    # publish to ROS2 topics
    which_motion_msg = Float64()
    which_motion_msg.data = float(self.motion_idx)

    self.pelvis_imu_state_pub.publish(pelvis_imu_msg)
    self.joint_state_pub.publish(joint_msg)
    self.hardware_time_pub.publish(time_msg)
    self.fsm_time_pub.publish(fsm_time_msg)
    self.goals_pub.publish(goals_msg)
    self.which_motion_pub.publish(which_motion_msg)

  #################################################################
  # SDK HARDWARE
  #################################################################

  # callback to receive low state messages
  def LowStateHandler(self, msg: LowState_):
    self.low_state = msg

    if self.update_mode_machine_ == False:
      self.mode_machine_ = self.low_state.mode_machine
      self.update_mode_machine_ = True

    # update sensor states under lock
    with self.sensor_lock:
      # update IMU states
      self.pelvis_imu_rpy = self.low_state.imu_state.rpy
      self.pelvis_imu_quaternion = self.low_state.imu_state.quaternion
      self.pelvis_imu_gyroscope = self.low_state.imu_state.gyroscope
      self.pelvis_imu_accelerometer = self.low_state.imu_state.accelerometer

      # update joint states
      for i in range(G1_NUM_MOTOR):
        self.q[i] = self.low_state.motor_state[i].q
        self.dq[i] = self.low_state.motor_state[i].dq
        self.ddq[i] = self.low_state.motor_state[i].ddq
        self.tau_est[i] = self.low_state.motor_state[i].tau_est

  # callback to receive pelvis pose messages from perception
  def pelvis_pose_callback(self, msg: PoseStamped):
    """Update pelvis world-frame pose from motion-capture system."""
    p = msg.pose.position
    q = msg.pose.orientation  # ROS: (x, y, z, w)
    with self.sensor_lock:
      self.pelvis_pose_position   = np.array([p.x, p.y, p.z], dtype=np.float64)
      self.pelvis_pose_quaternion = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)

  def box_pose_callback(self, msg: PoseStamped):
    """Receive live box pose (world frame) from motion capture; auto-select nearest motion."""
    p = msg.pose.position
    q = msg.pose.orientation  # ROS: (x, y, z, w)
    with self.sensor_lock:
      self.box_pose_position   = np.array([p.x, p.y, p.z], dtype=np.float32)
      self.box_pose_quaternion = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

    # Don't switch motion while a pickup is in progress.
    if self.motion_frame > 0 or len(self._nominal_box_centers) <= 1:
      return

    # Auto-select the nearest motion based on box position in pelvis frame
    # (mirrors simulation_box.py _step_simulation motion selection with hysteresis).
    with self.sensor_lock:
      pelvis_pos_w = np.array(self.pelvis_pose_position, dtype=np.float32)
      pelvis_quat  = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float32)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
      )

    R             = quat_to_rotation_matrix(pelvis_quat)
    box_pos_w     = np.array(self.box_pose_position, dtype=np.float32)
    box_in_pelvis = (R.T @ (box_pos_w - pelvis_pos_w)).astype(np.float32)

    dists     = np.linalg.norm(self._nominal_box_centers - box_in_pelvis, axis=1)
    best_idx  = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    curr_dist = float(dists[self.motion_idx])

    if (
      best_idx != self.motion_idx
      and best_dist + _MOTION_SWITCH_HYSTERESIS < curr_dist
    ):
      nominal = self._nominal_box_centers[best_idx]
      name    = (self._motion_names[best_idx]
                 if best_idx < len(self._motion_names) else str(best_idx))
      print(
        f"Motion switch -> {name} (idx {best_idx})  "
        f"nominal=({nominal[0]:.3f}, {nominal[1]:.3f}, {nominal[2]:.3f})  "
        f"box_b=({box_in_pelvis[0]:.3f}, {box_in_pelvis[1]:.3f}, {box_in_pelvis[2]:.3f})"
      )
      self.motion_idx = best_idx

  def _joystick_callback(self, msg: Float32MultiArray):
    """Toggle grip-width via Y button (index 6), rising-edge only."""
    y_pressed = len(msg.data) > 6 and float(msg.data[6]) > 0.5
    if y_pressed and not self._prev_y_pressed:
      self._otherhand_wide = not self._otherhand_wide
      state_str = "nominal [0,0.25,0]" if self._otherhand_wide else "wide    [0,0.40,0]"
      self.get_logger().info(f"Otherhand grip-width toggled → {state_str}")
    self._prev_y_pressed = y_pressed

  # main control loop to send low-level commands
  def LowCmdWrite(self):
    # update hardware time
    self.time_ += LOW_LEVEL_CONTROL_DT

    # read FSM state under lock
    with self.fsm_lock:
      fsm_state = self.fsm_state

    # detect state transition
    if fsm_state != self.prev_fsm_state:
      print(f"FSM: {self.prev_fsm_state} -> {fsm_state}")
      self.fsm_start_time = self.time_
      with self.sensor_lock:
        self.fsm_start_q = self.q.copy()
      self.prev_fsm_state = fsm_state

    # update fsm time
    self.fsm_time = self.time_ - self.fsm_start_time

    # safety: force damp if pelvis tilts beyond specified threshold
    if not self.safety_triggered:
      with self.sensor_lock:
        rpy = self.pelvis_imu_rpy
      if rpy is not None:
        roll, pitch = abs(rpy[0]), abs(rpy[1])
        if roll > SAFETY_MAX_TILT or pitch > SAFETY_MAX_TILT:
          print()
          print("*" * 70)
          print(
            f"SAFETY: roll={np.degrees(roll):.2f} pitch={np.degrees(pitch):.2f} -> FORCING DAMP. PLEASE RESTART!"
          )
          print("*" * 70)
          self.safety_triggered = True
    if self.safety_triggered:
      fsm_state = "damp"

    # [init]: zero out all commands
    if fsm_state == "init":
      for i in range(G1_NUM_MOTOR):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        self.low_cmd.motor_cmd[i].mode = 1
        self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.motor_cmd[i].q = 0.0
        self.low_cmd.motor_cmd[i].dq = 0.0
        self.low_cmd.motor_cmd[i].kp = 0.0
        self.low_cmd.motor_cmd[i].kd = 0.0

    # [damp]: Kd damping, no position tracking
    elif fsm_state == "damp":
      for i in range(G1_NUM_MOTOR):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        self.low_cmd.motor_cmd[i].mode = 1
        self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.motor_cmd[i].q = 0.0
        self.low_cmd.motor_cmd[i].dq = 0.0
        self.low_cmd.motor_cmd[i].kp = 0.0
        self.low_cmd.motor_cmd[i].kd = 3.0

    # [home]: interpolate to default joint positions and gains
    elif fsm_state == "home":
      ratio = np.clip(self.fsm_time / self.home_pos_duration, 0.0, 1.0)
      for i in range(G1_NUM_MOTOR):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        self.low_cmd.motor_cmd[i].mode = 1
        self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.fsm_start_q[
          i
        ] + ratio * self.default_joint_pos[i]
        self.low_cmd.motor_cmd[i].dq = 0.0
        self.low_cmd.motor_cmd[i].kp = ratio * self.Kp[i]
        self.low_cmd.motor_cmd[i].kd = ratio * self.Kd[i]

    # [control]: read from ROS2 command subscriber
    elif fsm_state == "control":
      with self.cmd_lock:
        q_cmd = self.q_cmd.copy()
        dq_cmd = self.dq_cmd.copy()
        Kp_cmd = self.Kp_cmd.copy()
        Kd_cmd = self.Kd_cmd.copy()
        tau_ff_cmd = self.tau_ff_cmd.copy()
      for i in range(G1_NUM_MOTOR):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        self.low_cmd.motor_cmd[i].mode = 1
        self.low_cmd.motor_cmd[i].tau = tau_ff_cmd[i]
        self.low_cmd.motor_cmd[i].q = q_cmd[i]
        self.low_cmd.motor_cmd[i].dq = dq_cmd[i]
        self.low_cmd.motor_cmd[i].kp = Kp_cmd[i]
        self.low_cmd.motor_cmd[i].kd = Kd_cmd[i]

    # check sum commands for safety and then publish
    self.low_cmd.crc = self.crc.Crc(self.low_cmd)
    self.lowcmd_publisher_.Write(self.low_cmd)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  # init ROS2
  rclpy.init()

  # parse arguments
  parser = argparse.ArgumentParser(
    description="Hardware deployment node using Unitree SDK2 for Python."
  )
  # network interface name argument
  parser.add_argument(
    "--network",
    type=str,
    required=True,
    help='Network interface name for robot communication. Example: "enp8s0".',
  )
  # config path argument
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help='Path to the config yaml file for hardware. Example: "g1_29dof_hardware.yaml".',
  )
  args = parser.parse_args()

  print()
  while input("Press [Enter] to continue: ") != "":
    pass
  print()

  # initialize the channel factory with the specified network interface
  ChannelFactoryInitialize(0, args.network)

  # instantiate the custom control class
  ctrl_node = ControlNode(args.config)
  ctrl_node.Init()

  # spin ROS2 node in background thread
  ros_running = True

  def spin_ros():
    while ros_running and rclpy.ok():
      try:
        rclpy.spin_once(ctrl_node, timeout_sec=0.1)
      except Exception:
        break

  ros_thread = threading.Thread(target=spin_ros, daemon=True)
  ros_thread.start()

  # start the control loop
  ctrl_node.Start()

  # run normally
  try:
    while True:
      time.sleep(1)
  # ctrl + C
  except KeyboardInterrupt:
    print("\nExiting...")
  # graceful shutdown on any exception
  finally:
    ros_running = False
    ros_thread.join(timeout=1.0)
    try:
      ctrl_node.destroy_node()
    except Exception:
      pass
    try:
      if rclpy.ok():
        rclpy.shutdown()
    except Exception:
      pass

  print("Hardware shutdown complete.")


if __name__ == "__main__":
  main()
