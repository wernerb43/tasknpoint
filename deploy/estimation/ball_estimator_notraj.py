##
#
# Ball pose estimator node.
#
# Subscribes to raw ball detections and publishes estimated pose to /ball/target_pose.
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

DEPLOY_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(DEPLOY_DIR)

# TODO add extended kalman filter for pelvis estimation

############################################################################
# ESTIMATOR NODE
############################################################################


class BallEstimatorNode(Node):
  def __init__(self):
    super().__init__("ball_estimator_node")

    # state
    self.ball_pos = np.zeros(3, dtype=np.float64)
    self.ball_received = False
    self.pelvis_pos = np.zeros(3, dtype=np.float64)
    self.pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    self.target_pos = np.zeros(3, dtype=np.float64)
    self.target_time = -1.0

    config_path = os.path.join(
      os.path.dirname(__file__), "..", "configs", "g1_tasknpoint.yaml"
    )
    with open(config_path) as f:
      config = yaml.safe_load(f)
    goals = {g["name"]: g for g in config["goals"]}
    self.nominal_target_pos_pelvis = np.array(
      goals["forehand_position"]["vector"], dtype=np.float64
    )
    self.nominal_target_pos = np.zeros(3, dtype=np.float64)

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
    # self.ball_target_time = self.create_publisher(Float64, "/ball/target_time", 10)

    self.create_timer(0.02, self.timer_callback)

    print("Ball estimator node initialized.")

  def ball_pose_callback(self, msg: PoseStamped):
    p = msg.pose.position
    self.ball_pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    self.ball_received = True

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
    t = 2.0 * np.cross(q_vec, self.nominal_target_pos_pelvis)
    self.nominal_target_pos = (
      self.nominal_target_pos_pelvis + qw * t + np.cross(q_vec, t) + self.pelvis_pos
    )

  # given the estimated position and velocity, find the point closest to the robot target point in world frame along the ball's trajectory, this will be the target pos
  def estimate_target_point(self):
    v = self.ball_pos - self.pelvis_pos
    q_vec = self.pelvis_quat[:3]
    qw = self.pelvis_quat[3]
    t = 2.0 * np.cross(q_vec, v)
    self.target_pos = v - qw * t + np.cross(q_vec, t)

  def timer_callback(self):
    if not self.ball_received:
      return
    self.estimate_target_point()
    self.publish_pose()

  def publish_pose(self):
    msg = PoseStamped()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = "world"
    msg.pose.position.x = self.target_pos[0]  # in world frame (for all of these)
    msg.pose.position.y = self.target_pos[1]
    msg.pose.position.z = self.target_pos[2]
    msg.pose.orientation.w = 1.0
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    self.ball_pose_pub.publish(msg)


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
