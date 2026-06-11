##
#
# Deployment code for Unitree G1 robot — box pickup, autonomous grip control.
#
# Identical to hardware_tasknpoint_box.py except the otherhand grip-width observation
# is driven automatically by motion phase rather than a joystick button.
#
# Grip schedule (per-motion, from g1_tasknpoint_pickup.yaml):
#   motion_frame == 0                                  → OPEN   ([0, 0.5, 0])
#   contact_frame  <= motion_frame < end_contact_frame → CLOSED (YAML otherhand vector, e.g. [0, 0.20, 0])
#   motion_frame >= end_contact_frame                  → OPEN   ([0, 0.5, 0])
#
#   contact_frame     = int(contact_phase[motion_idx]     * motion_num_frames[motion_idx])
#   end_contact_frame = int(end_contact_phase[motion_idx] * motion_num_frames[motion_idx])
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

# Unitree SDK imports (C++ wrapper: submodules/unitree_sdk2_wrapper)
import unitree_interface
from utils.unitree_utils import RecurrentThread


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
_MOTION_SWITCH_HYSTERESIS = 0.05  # metres

# Half the box width: each hand is placed this far to the left/right of the box centre
# in the pelvis Y-axis.  Right hand → -BOX_WIDTH, left hand → +BOX_WIDTH.
BOX_WIDTH = 0.25  # metres

# Grip-width values (y-component of otherhand offset in right-palm local frame).
_GRIP_OPEN_OFFSET   = np.array([0.0, 0.55, 0.0], dtype=np.float32)   # hands apart — pre/post contact
# Closed value comes from the YAML otherhand vector (e.g. [0, 0.20, 0]) per motion.


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
    self.fsm_lock = threading.Lock()    # protects FSM state
    self.sensor_lock = threading.Lock() # protects sensor state arrays
    self.cmd_lock = threading.Lock()    # protects command arrays

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

    # motion frame from control node (updated via ROS topic)
    self.motion_frame = 0
    self.motion_idx = 0

    # Autonomous grip state — tracked to log transitions only.
    # True  = grip CLOSED (contact phase active).
    # False = grip OPEN   (pre-contact or post-release).
    self._grip_closed = False

    # robot interface (created in Init) and network interface name
    self.robot = None
    self.network = None

    # hardware time and latest state flag
    self.time_ = 0.0
    self.state_ready = False

  #################################################################
  # INITIALIZATION
  #################################################################

  def load_config(self, config_path: str):
    config_path_full = DEPLOY_DIR + "/configs/" + config_path
    with open(config_path_full, "r") as f:
      config = yaml.safe_load(f)
    print("Config file loaded successfully from: [{}].".format(config_path_full))
    return config

  def load_params(self):
    # time to interpolate to initial position
    self.home_pos_duration = self.config["home_pos_duration"]  # float

    # default joint positions
    self.default_joint_pos = self.config["default_joint_pos"]  # list

    # contact phase fractions — one per motion
    contact_phase_cfg = self.config["contact_phase"]
    self.contact_phases = (
      contact_phase_cfg if isinstance(contact_phase_cfg, list) else [contact_phase_cfg]
    )
    # end-contact phase fractions — one per motion (grip opens at this point)
    end_contact_phase_cfg = self.config["end_contact_phase"]
    self.end_contact_phases = (
      end_contact_phase_cfg if isinstance(end_contact_phase_cfg, list) else [end_contact_phase_cfg]
    )

    # total frame counts per motion (used to convert phase fractions to frame indices)
    self.motion_num_frames = [
      int(np.load(mp if os.path.isabs(mp) else os.path.join(REPO_ROOT, mp))["joint_pos"].shape[0])
      for mp in self.config["motion_paths"]
    ]

    # sanity check: phase lists must cover all motions
    n_motions = len(self.motion_num_frames)
    assert len(self.contact_phases)     == n_motions, (
      f"contact_phase must have {n_motions} entries, got {len(self.contact_phases)}"
    )
    assert len(self.end_contact_phases) == n_motions, (
      f"end_contact_phase must have {n_motions} entries, got {len(self.end_contact_phases)}"
    )

    # pre-compute absolute contact/end-contact frame indices per motion
    self.contact_frames = [
      int(self.contact_phases[i] * self.motion_num_frames[i])
      for i in range(n_motions)
    ]
    self.end_contact_frames = [
      int(self.end_contact_phases[i] * self.motion_num_frames[i])
      for i in range(n_motions)
    ]

    # Parse YAML goals to extract per-motion grasp offsets (mirrors simulation_box.py).
    goals_cfg = self.config.get("goals", [])
    _right_by_idx = {
      g["motion_index"]: np.array(g["vector"], dtype=np.float32)
      for g in goals_cfg
      if g["type"] == "position"
      and "left"      not in g["name"]
      and "otherhand" not in g["name"]
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
    _default_otherhand = np.array([0.0, 0.20, 0.0], dtype=np.float32)
    self._otherhand_closed_offsets = np.array(
      [_otherhand_by_idx.get(i, _default_otherhand) for i in range(n_motions)],
      dtype=np.float32,
    )  # (n_motions, 3) — closed-grip offset in right-palm local frame, from YAML

    _nominal_positions = np.array(
      [_right_by_idx[i] for i in range(n_motions)], dtype=np.float32
    )
    _left_positions = np.array(
      [_left_by_idx[i] for i in range(n_motions)], dtype=np.float32
    )

    # Box centre and per-hand grasp offsets for goal computation.
    self._nominal_box_centers = (
      (_nominal_positions + _left_positions) / 2.0
    ).astype(np.float32)
    self._right_grasp_offsets = (
      _nominal_positions - self._nominal_box_centers
    ).astype(np.float32)
    self._left_grasp_offsets = (
      _left_positions - self._nominal_box_centers
    ).astype(np.float32)

    # Human-readable motion names
    self._motion_names = [Path(mp).stem for mp in self.config["motion_paths"]]

    # PD gains
    self.Kp = self.config["Kp"]  # list
    self.Kd = self.config["Kd"]  # list

    # type / length / value checks
    assert type(self.home_pos_duration) in [float], "home_pos_duration must be a float."
    assert type(self.default_joint_pos) == list,    "default_joint_pos must be a list."
    assert type(self.Kp) == list, "Kp must be a list."
    assert type(self.Kd) == list, "Kd must be a list."
    assert len(self.Kp) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kp values, got {len(self.Kp)}."
    assert len(self.Kd) == G1_NUM_MOTOR, f"Expected {G1_NUM_MOTOR} Kd values, got {len(self.Kd)}."
    assert self.home_pos_duration >= 3.0, "home_pos_duration must take at least 3 seconds."
    assert len(self.default_joint_pos) == G1_NUM_MOTOR, (
      f"Expected {G1_NUM_MOTOR} default joint positions, got {len(self.default_joint_pos)}"
    )
    for i in range(G1_NUM_MOTOR):
      assert self.Kp[i] >= 0.0, f"Kp for joint {i} must be non-negative."
      assert self.Kd[i] >= 0.0, f"Kd for joint {i} must be non-negative."

    print("Config parameters loaded successfully.")
    for i in range(n_motions):
      print(
        f"  Motion {i} ({self._motion_names[i]}): "
        f"frames={self.motion_num_frames[i]}  "
        f"contact=[{self.contact_frames[i]}, {self.end_contact_frames[i]})  "
        f"closed_offset={self._otherhand_closed_offsets[i].tolist()}"
      )

  # initialize the robot interface, publishers, and subscribers
  def Init(self):
    # create the G1 interface (HG messages); this also initializes DDS on the
    # given network interface and starts the wrapper's internal 500Hz command
    # writer + state subscriber threads.
    self.robot = unitree_interface.UnitreeInterface.create_g1(self.network)

    # release any active high-level motion service (loco/sport) so our LowCmd
    # writes are not fought by it. Replaces the MotionSwitcherClient loop.
    self.robot.release_motion_control()

    # Series control for pitch/roll joints (matches previous Mode.PR usage).
    self.robot.set_control_mode(unitree_interface.ControlMode.PR)

    print("Unitree robot interface initialized successfully.")

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
    self.which_motion_pub = self.create_publisher(Float64, "deploy_robot/which_motion", 10)
    self.fsm_time_pub = self.create_publisher(Float64, "deploy_robot/fsm_time", 10)

    # ROS2 subscribers
    self.command_sub = self.create_subscription(
      Float32MultiArray, "deploy_robot/command", self.command_callback, 10
    )
    self.motion_frame_sub = self.create_subscription(
      Float64, "deploy_robot/motion_frame", self.motion_frame_callback, 10
    )
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
    # NOTE: no joystick subscriber — grip width is driven autonomously by motion phase.

    # sensor publish timer
    self.pub_timer = self.create_timer(ROS_SENSOR_PUBLISH_DT, self.publish_sensor_data)

    print("ROS2 publishers and subscribers initialized successfully.")

  def Start(self):
    self.lowCmdWriteThreadPtr = RecurrentThread(
      interval=LOW_LEVEL_CONTROL_DT, target=self.LowCmdWrite, name="control"
    )
    while not self.read_robot_state():
      print("Waiting for first low state from robot...")
      time.sleep(1)
    self.state_ready = True
    self.lowCmdWriteThreadPtr.Start()
    print("Low-level robot control thread started successfully.")

  #################################################################
  # GRIP SCHEDULE
  #################################################################

  def _compute_grip_closed(self) -> bool:
    """Return True when the current motion frame is inside the contact window.

    Grip is CLOSED when:
      contact_frames[motion_idx] <= motion_frame < end_contact_frames[motion_idx]

    Grip is OPEN (returns False) when motion_frame == 0 (idle), before contact,
    or after end_contact.
    """
    if self.motion_frame == 0:
      return False  # no motion in progress
    idx = self.motion_idx
    return self.contact_frames[idx] <= self.motion_frame < self.end_contact_frames[idx]

  #################################################################
  # ROS PUBLISHING AND CALLBACKS
  #################################################################

  def fsm_callback(self, msg: String):
    with self.fsm_lock:
      self.fsm_state = msg.data

  def command_callback(self, msg: Float32MultiArray):
    data = np.array(msg.data, dtype=np.float64)
    if len(data) != 5 * G1_NUM_MOTOR:
      self.get_logger().warn(
        f"Expected {5 * G1_NUM_MOTOR} values in command, got {len(data)}"
      )
      return
    nu = G1_NUM_MOTOR
    with self.cmd_lock:
      self.q_cmd[:]      = data[0 * nu : 1 * nu]
      self.dq_cmd[:]     = data[1 * nu : 2 * nu]
      self.Kp_cmd[:]     = data[2 * nu : 3 * nu]
      self.Kd_cmd[:]     = data[3 * nu : 4 * nu]
      self.tau_ff_cmd[:] = data[4 * nu : 5 * nu]

  def motion_frame_callback(self, msg: Float64):
    self.motion_frame = int(msg.data)

  def publish_sensor_data(self):
    # read sensor data under lock
    with self.sensor_lock:
      pelvis_imu_rpy = (
        np.array(self.pelvis_imu_rpy, dtype=np.float64)
        if self.pelvis_imu_rpy is not None else np.zeros(3)
      )
      pelvis_imu_quat = (
        np.array(self.pelvis_imu_quaternion, dtype=np.float64)
        if self.pelvis_imu_quaternion is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
      )
      pelvis_imu_gyro = (
        np.array(self.pelvis_imu_gyroscope, dtype=np.float64)
        if self.pelvis_imu_gyroscope is not None else np.zeros(3)
      )
      pelvis_imu_accel = (
        np.array(self.pelvis_imu_accelerometer, dtype=np.float64)
        if self.pelvis_imu_accelerometer is not None else np.zeros(3)
      )
      q       = self.q.copy()
      dq      = self.dq.copy()
      ddq     = self.ddq.copy()
      tau_est = self.tau_est.copy()

    # imu_state: [rpy(3), quaternion(4), gyroscope(3), accelerometer(3)]
    pelvis_imu_msg = Float32MultiArray()
    pelvis_imu_msg.data = np.concatenate(
      [pelvis_imu_rpy, pelvis_imu_quat, pelvis_imu_gyro, pelvis_imu_accel]
    ).tolist()

    # joint_state: [q(29), dq(29), ddq(29), tau_est(29)]
    joint_msg = Float32MultiArray()
    joint_msg.data = np.concatenate([q, dq, ddq, tau_est]).tolist()

    time_msg = Float64()
    time_msg.data = self.time_

    fsm_time_msg = Float64()
    fsm_time_msg.data = self.fsm_time

    # ------------------------------------------------------------------
    # Compute box-pickup goals (mirrors simulation_box.py _publish_goals).
    # Goals 1 & 2: box-frame grasp offsets rotated to pelvis frame.
    # Goal 3:      otherhand offset — CLOSED during contact window, OPEN otherwise.
    # ------------------------------------------------------------------
    goals_msg = Float32MultiArray()
    with self.sensor_lock:
      pelvis_pos_w = np.array(self.pelvis_pose_position,   dtype=np.float32)
      pelvis_quat  = np.array(self.pelvis_pose_quaternion, dtype=np.float32)  # mocap rigid-body orientation
      box_pos_w    = np.array(self.box_pose_position,      dtype=np.float32)

    # Box centre expressed in the pelvis frame.
    R             = quat_to_rotation_matrix(pelvis_quat)
    box_in_pelvis = R.T @ (box_pos_w - pelvis_pos_w)

    # Grasp targets: BOX_WIDTH offset either side of the box centre in pelvis frame.
    right_goal = (box_in_pelvis + np.array([0.0, -BOX_WIDTH, 0.0], dtype=np.float32)).astype(np.float32)
    left_goal  = (box_in_pelvis + np.array([0.0, +BOX_WIDTH, 0.0], dtype=np.float32)).astype(np.float32)

    # Autonomous grip schedule.
    grip_closed = self._compute_grip_closed()
    if grip_closed != self._grip_closed:
      # Log transitions so they are easy to spot in the terminal.
      idx = self.motion_idx
      if grip_closed:
        print(
          f"[grip] CLOSE  motion={idx}  frame={self.motion_frame}"
          f"  (contact_frame={self.contact_frames[idx]})"
          f"  offset={self._otherhand_closed_offsets[idx].tolist()}"
        )
      else:
        print(
          f"[grip] OPEN   motion={idx}  frame={self.motion_frame}"
          f"  (end_contact_frame={self.end_contact_frames[idx]})"
          f"  offset={_GRIP_OPEN_OFFSET.tolist()}"
        )
      self._grip_closed = grip_closed

    otherhand_goal = (
      self._otherhand_closed_offsets[self.motion_idx]  # YAML value, e.g. [0, 0.20, 0]
      if grip_closed
      else _GRIP_OPEN_OFFSET                           # [0, 0.5, 0]
    )

    goals_msg.data = np.concatenate([right_goal, left_goal, otherhand_goal]).tolist()

    # publish
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

  # read the latest low state from the robot interface and update sensor arrays.
  # returns True once a valid state has been received from the robot.
  def read_robot_state(self):
    st = self.robot.read_low_state()
    q = np.asarray(st.motor.q, dtype=np.float64)
    quat = np.asarray(st.imu.quat, dtype=np.float64)

    # before any DDS state has arrived the wrapper buffers are zero-filled; a
    # valid IMU quaternion has unit norm, so use that to gate readiness.
    if q.shape[0] < G1_NUM_MOTOR or np.linalg.norm(quat) < 0.5:
      return False

    with self.sensor_lock:
      # wrapper field names: rpy / quat / omega / accel
      self.pelvis_imu_rpy           = np.asarray(st.imu.rpy, dtype=np.float64)
      self.pelvis_imu_quaternion    = quat
      self.pelvis_imu_gyroscope     = np.asarray(st.imu.omega, dtype=np.float64)
      self.pelvis_imu_accelerometer = np.asarray(st.imu.accel, dtype=np.float64)

      self.q[:]       = q[:G1_NUM_MOTOR]
      self.dq[:]      = np.asarray(st.motor.dq, dtype=np.float64)[:G1_NUM_MOTOR]
      self.tau_est[:] = np.asarray(st.motor.tau_est, dtype=np.float64)[:G1_NUM_MOTOR]
      # wrapper exposes no joint acceleration; downstream uses only q and dq.
      self.ddq[:]     = 0.0
    return True

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

    # Auto-select the nearest motion based on box position in pelvis frame.
    with self.sensor_lock:
      pelvis_pos_w = np.array(self.pelvis_pose_position,   dtype=np.float32)
      pelvis_quat  = np.array(self.pelvis_pose_quaternion, dtype=np.float32)  # mocap rigid-body orientation

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

  def LowCmdWrite(self):
    # refresh the latest robot state (IMU + joints) from the wrapper buffers
    self.read_robot_state()

    self.time_ += LOW_LEVEL_CONTROL_DT

    with self.fsm_lock:
      fsm_state = self.fsm_state

    # detect state transition
    if fsm_state != self.prev_fsm_state:
      print(f"FSM: {self.prev_fsm_state} -> {fsm_state}")
      self.fsm_start_time = self.time_
      with self.sensor_lock:
        self.fsm_start_q = self.q.copy()
      self.prev_fsm_state = fsm_state

    self.fsm_time = self.time_ - self.fsm_start_time

    # safety: force damp if pelvis tilts beyond threshold
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

    # Build the per-joint command arrays for the active FSM state. The wrapper's
    # internal 500Hz writer fills mode=1, mode_pr, mode_machine, and CRC; we only
    # provide q/dq/kp/kd/tau_ff.
    q_t = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    dq_t = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    kp = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    kd = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
    tau = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

    # [init] (arrays already zero)
    if fsm_state == "init":
      pass

    # [damp]
    elif fsm_state == "damp":
      kd[:] = 3.0

    # [home]
    elif fsm_state == "home":
      ratio = np.clip(self.fsm_time / self.home_pos_duration, 0.0, 1.0)
      q_t[:] = (1.0 - ratio) * self.fsm_start_q + ratio * np.asarray(
        self.default_joint_pos, dtype=np.float64
      )
      kp[:] = ratio * np.asarray(self.Kp, dtype=np.float64)
      kd[:] = ratio * np.asarray(self.Kd, dtype=np.float64)

    # [control]
    elif fsm_state == "control":
      with self.cmd_lock:
        q_t[:] = self.q_cmd
        dq_t[:] = self.dq_cmd
        kp[:] = self.Kp_cmd
        kd[:] = self.Kd_cmd
        tau[:] = self.tau_ff_cmd

    # send the command to the robot (whole-list assignment: pybind11 list
    # properties are copy-on-read, so element-wise writes would not persist).
    cmd = self.robot.create_zero_command()
    cmd.q_target = q_t.tolist()
    cmd.dq_target = dq_t.tolist()
    cmd.kp = kp.tolist()
    cmd.kd = kd.tolist()
    cmd.tau_ff = tau.tolist()
    self.robot.write_low_command(cmd)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  rclpy.init()

  parser = argparse.ArgumentParser(
    description="Hardware deployment node — box pickup with autonomous grip control."
  )
  parser.add_argument(
    "--network", type=str, required=True,
    help='Network interface name. Example: "enp8s0".',
  )
  parser.add_argument(
    "--config", type=str, required=True,
    help='Config YAML filename. Example: "g1_tasknpoint_pickup.yaml".',
  )
  args = parser.parse_args()

  print()
  while input("Press [Enter] to continue: ") != "":
    pass
  print()

  # DDS is initialized inside the wrapper using the network interface below.
  ctrl_node = ControlNode(args.config)
  ctrl_node.network = args.network
  ctrl_node.Init()

  ros_running = True

  def spin_ros():
    while ros_running and rclpy.ok():
      try:
        rclpy.spin_once(ctrl_node, timeout_sec=0.1)
      except Exception:
        break

  ros_thread = threading.Thread(target=spin_ros, daemon=True)
  ros_thread.start()

  ctrl_node.Start()

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    print("\nExiting...")
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
