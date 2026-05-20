##
#
# Ball pose estimator node.
#
# Subscribes to raw ball detections and publishes estimated pose to /ball/pose.
#
##

# standard imports
import argparse
import numpy as np
import threading
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64
from geometry_msgs.msg import PoseStamped

# directory imports
import os
import sys

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)


############################################################################
# ESTIMATOR NODE
############################################################################
GRAVITY = 9.81
COEFF_OF_RESTITUTION = 0.8


class BallEstimatorNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_node")

    # state
    self.ball_pos = np.zeros(3, dtype=np.float64)
    self.ball_vel = np.zeros(3, dtype=np.float64)
    self.pelvis_pos = np.zeros(3, dtype=np.float64)
    self.pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self.target_pos = np.zeros(3, dtype=np.float64)
    self.target_time = -1.0
    self.ball_trajectory_positions = np.zeros((0, 3), dtype=np.float64)
    self.ball_trajectory_times = np.zeros(0, dtype=np.float64)
    self.ball_trajectory_time_length = 5.0  # this is the amount of time in the future that the trajectory prediction should cover
    self.cutoff_distance = 0.5

    self.dt = 0.02

    # Kalman filter state: [x, y, z, vx, vy, vz]
    self.kf_state = np.zeros(6, dtype=np.float64)
    self.kf_P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    self.kf_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
    self.kf_R = np.eye(3, dtype=np.float64) * 0.01
    self.kf_initialized = False
    self.kf_last_time: float | None = None

    config_path = os.path.join(
      os.path.dirname(__file__), "..", "configs", "g1_29dof_tasknpoint_multimotion.yaml"
    )
    with open(config_path) as f:
      config = yaml.safe_load(f)
    position_goals = [g for g in config["goals"] if g["type"] == "position"]
    self.nominal_target_pos_pelvis = [
      np.array(g["vector"], dtype=np.float64) for g in position_goals
    ]
    self.nominal_target_pos = [np.zeros(3, dtype=np.float64) for _ in position_goals]
    self.nominal_motion_indices = [int(g["motion_index"]) for g in position_goals]
    self.target_motion_idx = self.nominal_motion_indices[0]

    self.lock = threading.Lock()

    # subscribers
    self.ball_pos_sub = self.create_subscription(
      PoseStamped, "/ball/pose", self.ball_pose_callback, 10
    )
    self.pelvis_pos_sub = self.create_subscription(
      PoseStamped, "/g1_pelvis/pose", self.pelvis_pose_callback, 10
    )

    # publishers
    self.ball_pose_pub = self.create_publisher(PoseStamped, "/ball/target_pose", 10)
    self.ball_target_time = self.create_publisher(Float64, "/ball/target_time", 10)
    self.ball_trajectory_pub = self.create_publisher(
      Float32MultiArray, "/ball/trajectory", 10
    )
    # self.which_motion = self.create_publisher(Float64, "/ball/which_motion", 10)

    self.create_timer(self.dt, self.timer_callback)

    print("Ball estimator node initialized.")

  # callback: ball position [x, y, z] in world frame
  def ball_pose_callback(self, msg: PoseStamped):
    z = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    self.filter_pose(z)

  # callback: pelvis pose in world frame
  def pelvis_pose_callback(self, msg: PoseStamped):
    self.pelvis_pos = np.array(
      [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
    )
    self.pelvis_quat = np.array(
      [
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
      ],
      dtype=np.float64,
    )
    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    for i, body_pos in enumerate(self.nominal_target_pos_pelvis):
      t = 2.0 * np.cross(q_vec, body_pos)
      self.nominal_target_pos[i] = (
        body_pos + qw * t + np.cross(q_vec, t) + self.pelvis_pos
      )

  # kalman filter update: estimate velocity and position of the ball from the measurements
  def filter_pose(self, z: np.ndarray):
    now = self.get_clock().now().nanoseconds * 1e-9

    if not self.kf_initialized:
      self.kf_state[:3] = z
      self.kf_initialized = True
      self.kf_last_time = now
      self.ball_pos = self.kf_state[:3].copy()
      self.ball_vel = self.kf_state[3:].copy()
      return

    dt = now - self.kf_last_time
    self.kf_last_time = now
    if dt <= 0:
      return

    # State transition: position integrates velocity, velocity is constant (gravity handled as input)
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Deterministic gravity input
    u = np.array([0.0, 0.0, -0.5 * GRAVITY * dt**2, 0.0, 0.0, -GRAVITY * dt])

    # Predict
    x_pred = F @ self.kf_state + u
    P_pred = F @ self.kf_P @ F.T + self.kf_Q

    # Update: observation is position only
    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    S = H @ P_pred @ H.T + self.kf_R
    K = P_pred @ H.T @ np.linalg.inv(S)

    self.kf_state = x_pred + K @ (z - H @ x_pred)
    self.kf_P = (np.eye(6) - K @ H) @ P_pred

    self.ball_pos = self.kf_state[:3].copy()
    self.ball_vel = self.kf_state[3:].copy()

  # given the estimated position and velocity, find the point closest to either nominal target along the ball's trajectory
  def estimate_target_point(self):
    if len(self.ball_trajectory_positions) == 0:
      dists = [np.sum((self.ball_pos - wp) ** 2) for wp in self.nominal_target_pos]
      best_target_idx = int(np.argmin(dists))
      self.target_pos = self.nominal_target_pos_pelvis[best_target_idx].copy()
      self.target_motion_idx = self.nominal_motion_indices[best_target_idx]
      self.target_time = -1.0
      return

    best_dist_sq = float("inf")
    best_traj_idx = 0
    best_target_idx = 0
    for i, world_target in enumerate(self.nominal_target_pos):
      dists_sq = np.sum((self.ball_trajectory_positions - world_target) ** 2, axis=1)
      idx = int(np.argmin(dists_sq))
      if dists_sq[idx] < best_dist_sq:
        best_dist_sq = float(dists_sq[idx])
        best_traj_idx = idx
        best_target_idx = i

    self.target_motion_idx = self.nominal_motion_indices[best_target_idx]

    if (
      best_dist_sq > self.cutoff_distance
    ):  # closest point still far from both targets, fall back
      self.target_pos = self.nominal_target_pos_pelvis[best_target_idx].copy()
      self.target_time = -1.0
    else:
      v = self.ball_trajectory_positions[best_traj_idx] - self.pelvis_pos
      q_vec = self.pelvis_quat[:3]
      qw = self.pelvis_quat[3]
      t = 2.0 * np.cross(q_vec, v)

      self.target_pos = v - qw * t + np.cross(q_vec, t)
      self.target_time = float(self.ball_trajectory_times[best_traj_idx])

      # TODO OFFSETS HERE ARE HACKED, FIND OUT WHY AND FIX PROPERLY
      self.target_pos[2] += (
        0.13  # this is a hack to make the target point slightly above the ball, which seems to help with hitting
      )

  def estimate_ball_trajectory(self):
    """
    Returns (times, positions) for the ball's predicted trajectory.
    times: shape (N,), positions: shape (N, 3), both in world frame.
    Accounts for gravity and ground bounces via COEFF_OF_RESTITUTION.
    """
    x, y, z = self.ball_pos
    vx, vy, vz = self.ball_vel
    t_elapsed = 0.0

    max_pts = int(np.ceil(self.ball_trajectory_time_length / self.dt)) + 1
    buf_times = np.empty(max_pts, dtype=np.float64)
    buf_positions = np.empty((max_pts, 3), dtype=np.float64)
    n = 0

    while t_elapsed < self.ball_trajectory_time_length:
      remaining = self.ball_trajectory_time_length - t_elapsed

      # Time to ground contact: z + vz*t - 0.5*g*t^2 = 0
      # t = (vz + sqrt(vz^2 + 2*g*z)) / g  (first positive root)
      discriminant = vz**2 + 2.0 * GRAVITY * max(z, 0.0)
      t_bounce = (vz + np.sqrt(discriminant)) / GRAVITY
      if t_bounce < 1e-6:
        break

      arc_duration = min(t_bounce, remaining)
      ts = np.arange(0.0, arc_duration, self.dt)
      if len(ts) == 0 or ts[-1] < arc_duration - 1e-9:
        ts = np.append(ts, arc_duration)

      k = len(ts)
      if n + k > max_pts:
        ts = ts[: max_pts - n]
        k = len(ts)
      buf_times[n : n + k] = t_elapsed + ts
      buf_positions[n : n + k, 0] = x + vx * ts
      buf_positions[n : n + k, 1] = y + vy * ts
      buf_positions[n : n + k, 2] = z + vz * ts - 0.5 * GRAVITY * ts**2
      n += k
      if n >= max_pts:
        break

      t_elapsed += t_bounce

      # Bounce: vz at impact is negative, flip and attenuate
      vz_impact = vz - GRAVITY * t_bounce
      vz = -COEFF_OF_RESTITUTION * vz_impact
      x = x + vx * t_bounce
      y = y + vy * t_bounce
      z = 0.0

    self.ball_trajectory_times = buf_times[:n]
    self.ball_trajectory_positions = buf_positions[:n]

  def timer_callback(self):
    if not self.kf_initialized:
      return
    self.estimate_ball_trajectory()
    self.estimate_target_point()
    self.publish_pose()
    self.publish_target_time()
    self.publish_trajectory()
    # self.publish_which_motion()

  def publish_pose(self):
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = self.target_pos[0]  # in pelvis frame (for all of these)
    msg.pose.position.y = self.target_pos[1]
    msg.pose.position.z = self.target_pos[2]
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    self.ball_pose_pub.publish(msg)

  def publish_target_time(self):
    # publish the time until the ball reaches the target point
    msg = Float64()
    msg.data = self.target_time
    self.ball_target_time.publish(msg)

  def publish_trajectory(self):
    msg = Float32MultiArray()
    msg.data = self.ball_trajectory_positions.flatten().tolist()
    self.ball_trajectory_pub.publish(msg)

  # def publish_which_motion(self):
  #   msg = Float64()
  #   msg.data = float(self.target_motion_idx)
  #   self.which_motion.publish(msg)


############################################################################
# MAIN FUNCTION
############################################################################


def main(args=None):
  rclpy.init()

  parser = argparse.ArgumentParser(description="Ball pose estimator node.")
  args = parser.parse_args()

  node = BallEstimatorNode()

  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()

  print("Ball estimator shutdown complete.")


if __name__ == "__main__":
  main()
