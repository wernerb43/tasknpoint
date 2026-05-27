##
#
# Box estimator visualization node.
#
# Live 3D plot showing:
#   - G1 pelvis position and orientation (world frame, from /g1_pelvis/pose + IMU)
#   - Box position and orientation (world frame, from /box/pose mocap)
#   - Box wireframe outline (20×20×40 cm geometry)
#   - Right-palm target in world frame  (goals[0:3], transformed from pelvis frame)
#   - Left-palm  target in world frame  (goals[3:6], transformed from pelvis frame)
#   - Otherhand offset label            (goals[6:9], constant in right-palm local frame)
#   - Active motion index
#
# Topics consumed:
#   /box/pose                    PoseStamped   — box world pose from motion capture
#   /g1_pelvis/pose              PoseStamped   — pelvis world position from motion capture
#   deploy_robot/pelvis_imu_state Float32MultiArray — [rpy(3), quat(4)=[w,x,y,z], gyro(3), acc(3)]
#   deploy_robot/goals            Float32MultiArray — 9D goal vector from hardware_tasknpoint.py
#   deploy_robot/which_motion     Float64           — active motion index
#
##

import threading
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import os
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

ROOT_DIR = os.getenv("DEPLOY_ROOT_DIR")
sys.path.append(ROOT_DIR)

DEPLOY_DIR = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(DEPLOY_DIR)

# Box geometry (half-extents in metres, matching simulation_box.py)
_BOX_HALF = np.array([0.10, 0.10, 0.20])

# 8 corners in box-local frame
_BOX_CORNERS_LOCAL = (
    np.array([
        [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
        [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
    ], dtype=np.float64) * _BOX_HALF
)

# 12 edges as corner-index pairs
_BOX_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]

HISTORY_LEN = 80


def _quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → 3×3 rotation matrix (matches quat_to_rotation_matrix in math_utils)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)     ],
        [    2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)     ],
        [    2*(x*z - w*y),      2*(y*z + w*x),  1 - 2*(x*x + y*y) ],
    ])


class BoxEstimatorPlottingNode(Node):
    def __init__(self):
        super().__init__("box_estimator_plotting_node")

        self.lock = threading.Lock()

        # Box state (world frame)
        self.box_pos:  np.ndarray | None = None
        self.box_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.box_history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

        # Pelvis state
        self.pelvis_pos:       np.ndarray | None = None  # world frame from mocap
        self.pelvis_pose_quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z] from mocap rigid-body
        self.pelvis_imu_quat:  np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z] from IMU (kept for reference)

        # Goals from hardware_tasknpoint (pelvis frame for 0:6, right-palm frame for 6:9)
        self.goals: np.ndarray | None = None  # shape (9,)

        # Active motion index
        self.motion_idx: int = 0

        self.create_subscription(PoseStamped,       "/box/pose",                    self._box_callback,       10)
        self.create_subscription(PoseStamped,       "/g1_pelvis/pose",              self._pelvis_callback,    10)
        self.create_subscription(Float32MultiArray, "deploy_robot/pelvis_imu_state",self._imu_callback,       10)
        self.create_subscription(Float32MultiArray, "deploy_robot/goals",           self._goals_callback,     10)
        self.create_subscription(Float64,           "deploy_robot/which_motion",    self._motion_callback,    10)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _box_callback(self, msg: PoseStamped):
        q = msg.pose.orientation  # ROS: (x, y, z, w)
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        with self.lock:
            self.box_pos  = pos
            self.box_quat = np.array([q.w, q.x, q.y, q.z])  # → [w, x, y, z]
            self.box_history.append(pos.copy())

    def _pelvis_callback(self, msg: PoseStamped):
        q = msg.pose.orientation  # ROS: (x, y, z, w)
        with self.lock:
            self.pelvis_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])
            self.pelvis_pose_quat = np.array([q.w, q.x, q.y, q.z])  # → [w, x, y, z]

    def _imu_callback(self, msg: Float32MultiArray):
        # layout: [rpy(3), quat(4)=[w,x,y,z], gyro(3), acc(3)]
        data = np.array(msg.data, dtype=np.float64)
        if len(data) >= 7:
            with self.lock:
                self.pelvis_imu_quat = data[3:7].copy()  # [w, x, y, z]

    def _goals_callback(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float64)
        if len(data) == 9:
            with self.lock:
                self.goals = data.copy()

    def _motion_callback(self, msg: Float64):
        with self.lock:
            self.motion_idx = int(msg.data)

    # ------------------------------------------------------------------
    # State snapshot for the animation loop
    # ------------------------------------------------------------------

    def get_state(self):
        with self.lock:
            box_pos   = self.box_pos.copy()  if self.box_pos  is not None else None
            box_quat  = self.box_quat.copy()
            pelvis    = self.pelvis_pos.copy() if self.pelvis_pos is not None else None
            pose_quat = self.pelvis_pose_quat.copy()  # mocap rigid-body orientation
            goals     = self.goals.copy() if self.goals is not None else None
            motion    = self.motion_idx
            history   = [p.copy() for p in self.box_history]

        # Transform goal targets from pelvis frame → world frame using pelvis mocap orientation.
        # goals[0:3] = right-palm target in pelvis frame
        # goals[3:6] = left-palm  target in pelvis frame
        # goals[6:9] = otherhand offset in right-palm local frame (constant, not transformable)
        right_target_w = None
        left_target_w  = None
        otherhand_offset = None
        if goals is not None and pelvis is not None:
            R_pelvis = _quat_to_rotmat(pose_quat)
            right_target_w   = R_pelvis @ goals[0:3] + pelvis
            left_target_w    = R_pelvis @ goals[3:6] + pelvis
            otherhand_offset = goals[6:9]

        return box_pos, box_quat, pelvis, pose_quat, right_target_w, left_target_w, otherhand_offset, motion, history


def main():
    rclpy.init()
    node = BoxEstimatorPlottingNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    def _draw_box_wireframe(ax, center: np.ndarray, R: np.ndarray) -> None:
        """Draw 12-edge wireframe of the box at *center* with rotation *R*."""
        corners = (R @ _BOX_CORNERS_LOCAL.T).T + center  # (8, 3)
        for i, j in _BOX_EDGES:
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                color="saddlebrown", linewidth=1.2, alpha=0.8,
            )

    def _draw_frame_axes(ax, origin: np.ndarray, R: np.ndarray, length: float = 0.25) -> None:
        """Draw RGB X/Y/Z axes at *origin* using rotation *R*."""
        for col, axis in zip(("red", "limegreen", "dodgerblue"), R.T):
            ax.quiver(
                origin[0], origin[1], origin[2],
                axis[0] * length, axis[1] * length, axis[2] * length,
                color=col, linewidth=1.5, arrow_length_ratio=0.2,
            )

    def update(_frame):
        (box_pos, box_quat, pelvis, pose_quat,
         right_w, left_w, otherhand_offset, motion, history) = node.get_state()

        ax.cla()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        otherhand_str = (
            f"[{otherhand_offset[0]:.2f}, {otherhand_offset[1]:.2f}, {otherhand_offset[2]:.2f}]"
            if otherhand_offset is not None else "—"
        )
        ax.set_title(
            f"Box Pickup — motion {motion}    "
            f"otherhand offset (right-palm frame): {otherhand_str}",
            fontsize=10,
        )

        # --- Box history trail ---
        if len(history) > 1:
            hist = np.array(history)
            ax.plot(
                hist[:, 0], hist[:, 1], hist[:, 2],
                ".", color="peru", markersize=3, alpha=0.4, label="Box history",
            )

        # --- Box: wireframe + axes + centre marker ---
        if box_pos is not None:
            R_box = _quat_to_rotmat(box_quat)
            _draw_box_wireframe(ax, box_pos, R_box)
            _draw_frame_axes(ax, box_pos, R_box, length=0.20)
            ax.scatter(
                *box_pos, c="saddlebrown", s=80, marker="s",
                label="Box", zorder=5,
            )

        # --- Pelvis: marker + orientation axes ---
        if pelvis is not None:
            R_pelvis = _quat_to_rotmat(pose_quat)
            ax.scatter(
                *pelvis, c="orange", s=120, marker="s",
                label="Pelvis", zorder=5,
            )
            _draw_frame_axes(ax, pelvis, R_pelvis, length=0.30)

        # --- Right-palm target (world frame) ---
        if right_w is not None:
            ax.scatter(
                *right_w, c="red", s=180, marker="*",
                label="Right-palm target", zorder=6,
            )
            # dashed line from pelvis to target for context
            if pelvis is not None:
                ax.plot(
                    [pelvis[0], right_w[0]], [pelvis[1], right_w[1]], [pelvis[2], right_w[2]],
                    "--", color="red", linewidth=0.8, alpha=0.5,
                )

        # --- Left-palm target (world frame) ---
        if left_w is not None:
            ax.scatter(
                *left_w, c="limegreen", s=180, marker="*",
                label="Left-palm target", zorder=6,
            )
            if pelvis is not None:
                ax.plot(
                    [pelvis[0], left_w[0]], [pelvis[1], left_w[1]], [pelvis[2], left_w[2]],
                    "--", color="limegreen", linewidth=0.8, alpha=0.5,
                )

        # --- Axis frame colour legend patches ---
        extra_handles = [
            mpatches.Patch(color="red",        label="Frame X axis"),
            mpatches.Patch(color="limegreen",  label="Frame Y axis"),
            mpatches.Patch(color="dodgerblue", label="Frame Z axis"),
        ]

        # --- Axis limits centred on pelvis (or box if no pelvis yet) ---
        centre = pelvis if pelvis is not None else box_pos
        if centre is not None:
            r = 3.0
            ax.set_xlim(centre[0] - r, centre[0] + r)
            ax.set_ylim(centre[1] - r, centre[1] + r)
            ax.set_zlim(0.0, max(centre[2] + r, 2.5))

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles + extra_handles,
            labels  + [h.get_label() for h in extra_handles],
            loc="upper left", fontsize=8,
        )
        return []

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841
    plt.show()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
