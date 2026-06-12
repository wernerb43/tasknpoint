##
#
# Simulation node for the box-pickup task.
#
# The box is a real MuJoCo free-body that the robot can physically interact
# with.  Arrow keys move the box; B button on the joystick triggers the motion.
#
# Goal vector published each tick (9 floats):
#   [0:3]  right-palm target  = box_center_in_pelvis + R_pelvis.T @ R_box @ right_grasp_offset  (per-motion, from YAML)
#   [3:6]  left-palm target   = box_center_in_pelvis + R_pelvis.T @ R_box @ left_grasp_offset   (per-motion, from YAML)
#   [6:9]  otherhand target   = otherhand_offset in right-palm local frame   (constant per motion, from YAML)
#            Y button toggles between nominal [0,0.25,0] and wide [0,0.40,0].
#
# right/left offsets are pre-computed from the explicit right/left hand positions stored in
# the deploy config YAML (cast_pickup_N_position and cast_pickup_N_left_position).
# Offsets are defined in the box frame so both hand goals track the box live as it is
# moved AND as it changes orientation (R_box rotates the offset into world/pelvis frame).
# The otherhand goal (3rd) matches the training generated_commands output for dynamic
# targets: commands.py passes _moving_target_offset_w directly (the per-episode sampled
# offset in the target body's local frame) with NO frame transformation.  No live FK needed.
#
##

import argparse
import os
import tempfile
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32MultiArray

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.math_utils import (
    quat_to_rotation_matrix,
    quat_to_rpy,
)

# ---------------------------------------------------------------------------
# Box geometry constants
# ---------------------------------------------------------------------------
_BOX_HALF_X = 0.10    # depth  half-extent (m)
_BOX_HALF_Y = 0.10    # width  half-extent (m)
_BOX_HALF_Z = 0.20    # height half-extent (m)
_BOX_MASS   = 0.1     # kg

# Per-motion grasp offsets (right and left from box centre) are computed at
# startup from the YAML goals — see _init_params.  _BOX_HALF_Y only controls
# the MuJoCo box geom; it is no longer used as the goal offset.

# ---------------------------------------------------------------------------
# Arrow-key / joystick movement parameters
# ---------------------------------------------------------------------------
_BOX_SPEED   = 0.5    # m/s
_BOX_X_LIMIT = 5.0    # world-frame X limits (m)
_BOX_Y_LIMIT = 5.0    # world-frame Y limits (m)
_BOX_Z_MAX   = 1.2    # world-frame Z ceiling (m)

# GLFW key codes used by the MuJoCo viewer
_KEY_UP       = 265   # move box +X (deeper)
_KEY_DOWN     = 264   # move box -X (closer)
_KEY_LEFT     = 263   # move box +Y (robot's left)
_KEY_RIGHT    = 262   # move box -Y (robot's right)
_KEY_PAGE_UP  = 266   # move box +Z (up)
_KEY_PAGE_DN  = 267   # move box -Z (down)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw to quaternion [w, x, y, z]."""
    r, p, y = rpy
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


############################################################################
# SIMULATION NODE
############################################################################


class SimulationNode(Node):
    """
    Asynchronous simulation node for the box-pickup task.

    The cardboard box is a physics-enabled free body so the robot can
    actually pick it up.  Arrow keys reposition it while it is not being
    held; the B button on the joystick triggers the policy motion.

    Goal vector (9 floats) published each control tick:
      [0:3]  right-palm target in pelvis frame
             = box_center_in_pelvis  +  R_pelvis.T @ R_box @ right_grasp_offset[motion_idx]
      [3:6]  left-palm target in pelvis frame
             = box_center_in_pelvis  +  R_pelvis.T @ R_box @ left_grasp_offset[motion_idx]
      [6:9]  otherhand target in right-palm local frame  (constant per motion)
             = otherhand_offset[motion_idx]  (e.g. [0, 0.25, 0])

    The first two offsets are derived from the explicit right/left pre-grab positions
    stored in the YAML (cast_pickup_N_position / cast_pickup_N_left_position) and are
    defined in the box frame.  Multiplying by R_box rotates them into world frame so
    that the hand targets follow the box as it both translates and rotates.
    The third (otherhand) goal matches the training generated_commands output for
    dynamic targets: the per-episode offset in the target body's local frame is passed
    directly with no frame transformation (commands.py _moving_target_offset_w).
    """

    def __init__(self, config_path: str, apply_noise: bool = False):
        super().__init__("simulation_node")

        self.config      = self._load_config(config_path)
        self.apply_noise = apply_noise

        self._init_params()
        self._init_simulation()   # loads model with box injected, launches viewer

        # ------------------------------------------------------------------
        # Place the box at its initial world position.
        # Box centre = midpoint of the right-palm and left-palm targets for
        # motion 0, expressed in the default pelvis frame.
        # ------------------------------------------------------------------
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        _R0 = quat_to_rotation_matrix(self.mj_data.body("pelvis").xquat.astype(np.float32))
        _p0 = self.mj_data.body("pelvis").xpos.astype(np.float32)

        _box_center_pelvis = self._nominal_box_centers[0]
        _box_center_world  = _R0 @ _box_center_pelvis + _p0
        _box_center_world[2] = max(float(_box_center_world[2]), _BOX_HALF_Z)

        self.mj_data.qpos[self._box_qpos_adr : self._box_qpos_adr + 3] = _box_center_world
        self.mj_data.qpos[self._box_qpos_adr + 3]                      = 1.0  # quat w
        self.mj_data.qpos[self._box_qpos_adr + 4 : self._box_qpos_adr + 7] = 0.0

        self._init_goals()

        # ------------------------------------------------------------------
        # ROS publishers
        # ------------------------------------------------------------------
        self.pelvis_imu_state_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/pelvis_imu_state", 10
        )
        self.joint_state_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/joint_state", 10
        )
        self.simulation_time_pub = self.create_publisher(
            Float64, "deploy_robot/simulation_time", 10
        )
        self.goal_pub = self.create_publisher(
            Float32MultiArray, "deploy_robot/goals", 10
        )
        self.which_motion_pub = self.create_publisher(
            Float64, "deploy_robot/which_motion", 10
        )

        # ------------------------------------------------------------------
        # ROS subscribers
        # ------------------------------------------------------------------
        self.create_subscription(
            Float32MultiArray, "deploy_robot/command", self._command_callback, 10
        )
        self.create_subscription(
            Float64, "deploy_robot/motion_frame", self._motion_frame_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "deploy_robot/joystick", self._joystick_callback, 10
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.command_received    = False
        self.qpos_des            = np.zeros(self.nu)
        self.qvel_des            = np.zeros(self.nu)
        self.tau_ff              = np.zeros(self.nu)
        self.Kp                  = np.zeros(self.nu)
        self.Kd                  = np.zeros(self.nu)
        self.motion_frame        = 0
        self._motion_in_progress = False

        # Otherhand-width toggle (Y button).
        # When True  (nominal): otherhand goal = YAML offset ([0, 0.25, 0]) in right-palm frame.
        # When False (wide):    otherhand goal = [0, 0.40, 0] in right-palm frame.
        self._otherhand_wide   = True   # starts in nominal mode
        self._prev_y_pressed   = False  # for rising-edge detection

        self.create_timer(0.0,         self._step_simulation)
        self.create_timer(self.sim_dt, self._publish_pelvis_imu)
        self.create_timer(self.sim_dt, self._publish_joint_state)
        self.create_timer(self.sim_dt, self._publish_goals)

        print("Simulation node initialized.")
        print("    Press [Tab] to toggle the left UI.")
        print("    Press [Shift + Tab] to toggle the right UI.")
        print("    Arrow keys : move box  (Up/Down = X,  Left/Right = Y,  PgUp/PgDn = Z)")
        print("    Joystick B : trigger pickup motion")
        print("    Joystick Y : toggle otherhand grip-width  (wide ↔ zero)")
        print(f"    Box initial position (world): {_box_center_world.tolist()}")

    #################################################################
    # INITIALIZATION
    #################################################################

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT.parent / path

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.is_absolute():
            candidate = REPO_ROOT / "configs" / config_path
            if candidate.exists():
                path = candidate
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{path}].")
        return config

    def _init_params(self):
        self.default_base   = np.array(self.config["default_base_pos"])
        self.default_joints = np.array(self.config["default_joint_pos"])

        self.motion_idx = 0

        goals_cfg = self.config.get("goals", [])

        # Right-palm targets: goals whose name does NOT contain "left" or "otherhand".
        # NOTE: "otherhand" goals must be excluded explicitly because they also lack
        # "left" in their name but represent a grip-width offset, not a palm position.
        _right_by_idx = {
            g["motion_index"]: np.array(g["vector"], dtype=np.float32)
            for g in goals_cfg
            if g["type"] == "position"
            and "left" not in g["name"]
            and "otherhand" not in g["name"]
        }
        # Left-palm targets: goals whose name contains "left".
        _left_by_idx = {
            g["motion_index"]: np.array(g["vector"], dtype=np.float32)
            for g in goals_cfg
            if g["type"] == "position" and "left" in g["name"]
        }
        # Otherhand offsets: goals whose name contains "otherhand".
        # These are the grip-width offsets added to the live right-palm position
        # to produce the 3rd goal (target position for left_palm, in pelvis frame).
        _otherhand_by_idx = {
            g["motion_index"]: np.array(g["vector"], dtype=np.float32)
            for g in goals_cfg
            if g["type"] == "position" and "otherhand" in g["name"]
        }

        n_motions = max(max(_right_by_idx.keys()), max(_left_by_idx.keys())) + 1 \
                    if _right_by_idx else 0

        # Per-motion right-palm targets (pelvis frame from motion data).
        self._nominal_positions = np.array(
            [_right_by_idx[i] for i in range(n_motions)], dtype=np.float32
        )  # shape (n_motions, 3)

        # Per-motion left-palm targets (pelvis frame from motion data).
        self._left_positions = np.array(
            [_left_by_idx[i] for i in range(n_motions)], dtype=np.float32
        )  # shape (n_motions, 3)

        # Per-motion otherhand offsets (grip-width in pelvis frame, e.g. [0, 0.25, 0]).
        # Fall back to [0, 0.25, 0] if not present in YAML.
        _default_otherhand = np.array([0.0, 0.25, 0.0], dtype=np.float32)
        self._otherhand_offsets = np.array(
            [_otherhand_by_idx.get(i, _default_otherhand) for i in range(n_motions)],
            dtype=np.float32,
        )  # shape (n_motions, 3)

        # Box centre in pelvis frame = midpoint of right and left pre-grab targets.
        self._nominal_box_centers = (
            self._nominal_positions + self._left_positions
        ) / 2.0  # shape (n_motions, 3)

        # Per-motion grasp offsets from box centre in pelvis frame.
        # right_grasp_offsets[i] ≈ [0, -0.2, 0]  (robot's right)
        # left_grasp_offsets[i]  ≈ [0, +0.2, 0]  (robot's left)
        self._right_grasp_offsets = (
            self._nominal_positions - self._nominal_box_centers
        ).astype(np.float32)  # shape (n_motions, 3)
        self._left_grasp_offsets = (
            self._left_positions - self._nominal_box_centers
        ).astype(np.float32)  # shape (n_motions, 3)

    def _init_simulation(self):
        xml_path = REPO_ROOT.parent / "robots" / self.config["xml_path"]

        # ------------------------------------------------------------------
        # Inject a physics-enabled box free-body into the robot XML.
        # Writing to a temp file in the same directory preserves relative
        # mesh / include paths used by the robot XML.
        # ------------------------------------------------------------------
        with open(xml_path, "r") as f:
            xml_str = f.read()

        box_xml = (
            f'  <body name="box" pos="0 0 {_BOX_HALF_Z:.4f}">\n'
            f'    <freejoint name="box_joint"/>\n'
            f'    <geom name="box_geom" type="box"'
            f' size="{_BOX_HALF_X} {_BOX_HALF_Y} {_BOX_HALF_Z}"\n'
            f'          mass="{_BOX_MASS}" rgba="0.72 0.45 0.20 1.0"\n'
            f'          friction="0.8 0.005 0.0001" condim="4"\n'
            f'          solimp="0.99 0.999 0.001" solref="0.01 1"/>\n'
            f'  </body>'
        )
        assert "</worldbody>" in xml_str, "Could not find </worldbody> in robot XML."
        # Use count=1 — the XML contains two </worldbody> sections (robot + floor).
        # We only want to inject the box into the first (robot) worldbody.
        xml_str = xml_str.replace("</worldbody>", box_xml + "\n  </worldbody>", 1)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=str(xml_path.parent))
        try:
            with os.fdopen(tmp_fd, "w") as tmp_f:
                tmp_f.write(xml_str)
            self.mj_model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            os.unlink(tmp_path)

        self.mj_data = mujoco.MjData(self.mj_model)

        self.nq     = self.mj_model.nq
        self.nv     = self.mj_model.nv
        self.nu     = self.mj_model.nu   # unchanged — box has no actuators
        self.sim_dt = self.mj_model.opt.timestep

        assert len(self.default_joints) == self.nu, (
            f"default_joint_pos must have {self.nu} entries, got {len(self.default_joints)}."
        )

        # Robot base + joint defaults (indices unchanged — box joint comes after)
        self.mj_data.qpos[:7]              = self.default_base
        self.mj_data.qpos[7 : 7 + self.nu] = self.default_joints

        # Box freejoint addresses in qpos / qvel
        _box_jid              = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint"
        )
        self._box_qpos_adr    = int(self.mj_model.jnt_qposadr[_box_jid])
        self._box_dof_adr     = int(self.mj_model.jnt_dofadr[_box_jid])

        # per-joint sensor name lists (matching actuator order)
        self.joint_pos_sensor_names = []
        self.joint_vel_sensor_names = []
        for i in range(self.nu):
            joint_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            self.joint_pos_sensor_names.append(f"{joint_name}_pos_sensor")
            self.joint_vel_sensor_names.append(f"{joint_name}_vel_sensor")

        print(f"Loaded Mujoco model from [{xml_path}] (box injected).")
        print(f"    sim_dt={self.sim_dt}s  nq={self.nq}  nv={self.nv}  nu={self.nu}")
        print(f"    box_qpos_adr={self._box_qpos_adr}  box_dof_adr={self._box_dof_adr}")

        # ------------------------------------------------------------------
        # Arrow-key velocity state (set from viewer thread via key_callback).
        #
        # MuJoCo's Python viewer KeyCallbackType is Callable[[int], None] —
        # only the key code is passed; there is no press/release/repeat flag.
        # We use a timestamp heartbeat instead: each key event refreshes
        # _box_key_last_t.  The simulation loop zeros the velocity once the
        # timestamp goes stale (key released → GLFW repeat stops firing).
        # ------------------------------------------------------------------
        _KEY_TIMEOUT = 0.20   # seconds — safely longer than GLFW repeat interval (~33ms)

        self._box_key_vel    = np.zeros(3, dtype=np.float64)
        self._box_key_last_t = 0.0   # real-clock time of last key event

        def _key_callback(key: int) -> None:
            """Set box velocity from GLFW arrow-key events."""
            if   key == _KEY_UP:       self._box_key_vel[:] = [ _BOX_SPEED, 0, 0]
            elif key == _KEY_DOWN:     self._box_key_vel[:] = [-_BOX_SPEED, 0, 0]
            elif key == _KEY_LEFT:     self._box_key_vel[:] = [0,  _BOX_SPEED, 0]
            elif key == _KEY_RIGHT:    self._box_key_vel[:] = [0, -_BOX_SPEED, 0]
            elif key == _KEY_PAGE_UP:  self._box_key_vel[:] = [0, 0,  _BOX_SPEED]
            elif key == _KEY_PAGE_DN:  self._box_key_vel[:] = [0, 0, -_BOX_SPEED]
            else:
                return   # unrecognised key — don't update timestamp
            self._box_key_last_t = time.perf_counter()

        self.viewer = mujoco.viewer.launch_passive(
            self.mj_model, self.mj_data,
            show_left_ui=False, show_right_ui=False,
            key_callback=_key_callback,
        )

        self._viewer_font_scale = getattr(
            mujoco.mjtFontScale,
            "mjFONTSCALE_250",
            getattr(
                mujoco.mjtFontScale,
                "mjFONTSCALE_200",
                mujoco.mjtFontScale.mjFONTSCALE_150,
            ),
        )

        self.viewer.cam.azimuth   = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance  = 2.5
        self.viewer.cam.lookat[:] = list(self.default_base[:3])

        self.viewer_render_hz    = 50.0
        self._last_viewer_sync   = 0.0
        self._real_start_time    = time.perf_counter()
        self._next_step_deadline = self._real_start_time + self.sim_dt

    def _init_goals(self):
        """
        Log per-motion grasp offsets.

        The actual offsets are pre-computed in _init_params from the YAML
        cast_pickup_N_position (right palm) and cast_pickup_N_left_position
        (left palm) goals.  Both are expressed relative to the box centre
        so they track the box live as it is moved.
        """
        goals_cfg = self.config.get("goals", [])
        names     = [g["name"] for g in goals_cfg if g["motion_index"] == self.motion_idx]
        r_off     = self._right_grasp_offsets[self.motion_idx]
        l_off     = self._left_grasp_offsets[self.motion_idx]

        print(f"Goals initialized for pickup motion {self.motion_idx}: {names}")
        print(f"    Right grasp offset from box centre: {r_off.tolist()}")
        print(f"    Left  grasp offset from box centre: {l_off.tolist()}")

    #################################################################
    # CALLBACKS
    #################################################################

    def _command_callback(self, msg):
        data = np.array(msg.data)
        self.command_received = True
        self.qpos_des = data[0 * self.nu : 1 * self.nu]
        self.qvel_des = data[1 * self.nu : 2 * self.nu]
        self.Kp       = data[2 * self.nu : 3 * self.nu]
        self.Kd       = data[3 * self.nu : 4 * self.nu]
        self.tau_ff   = data[4 * self.nu : 5 * self.nu]

    def _motion_frame_callback(self, msg):
        new_frame = int(msg.data)
        if self._motion_in_progress and self.motion_frame > 0 and new_frame == 0:
            self._motion_in_progress = False
        self.motion_frame = new_frame

    def _joystick_callback(self, msg):
        """
        Joystick button handler.

        Layout (matches joystick_ros.py):
          data[4] — B : trigger pickup motion
          data[6] — Y : toggle otherhand grip-width (wide ↔ zero), rising-edge only
        """
        # B button — start pickup motion.
        if len(msg.data) > 4 and float(msg.data[4]) > 0.5:
            self._motion_in_progress = True

        # Y button — toggle otherhand width on rising edge only (avoids rapid toggling
        # while the button is held down across multiple callback invocations).
        y_pressed = len(msg.data) > 6 and float(msg.data[6]) > 0.5
        if y_pressed and not self._prev_y_pressed:
            self._otherhand_wide = not self._otherhand_wide
            state_str   = "nominal [0,0.25,0]" if self._otherhand_wide else "wide    [0,0.40,0]"
            self.get_logger().info(f"Otherhand grip-width toggled → {state_str}")
        self._prev_y_pressed = y_pressed

    #################################################################
    # PUBLISHING
    #################################################################

    def _publish_pelvis_imu(self):
        pelvis_quat = self.mj_data.sensor("pelvis_imu_quat_sensor").data.copy()
        pelvis_gyro = self.mj_data.sensor("pelvis_imu_gyro_sensor").data.copy()
        pelvis_acc  = self.mj_data.sensor("pelvis_imu_acc_sensor").data.copy()
        pelvis_rpy  = quat_to_rpy(pelvis_quat)

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [pelvis_rpy, pelvis_quat, pelvis_gyro, pelvis_acc]
        ).tolist()
        self.pelvis_imu_state_pub.publish(msg)

    def _publish_joint_state(self):
        qpos_joints    = np.array(
            [self.mj_data.sensor(n).data[0] for n in self.joint_pos_sensor_names]
        )
        qvel_joints    = np.array(
            [self.mj_data.sensor(n).data[0] for n in self.joint_vel_sensor_names]
        )
        ddq_joints     = np.zeros(self.nu)
        tau_est_joints = self.mj_data.ctrl[: self.nu].copy()

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [qpos_joints, qvel_joints, ddq_joints, tau_est_joints]
        ).tolist()
        self.joint_state_pub.publish(msg)

    def _publish_goals(self):
        which_motion_msg      = Float64()
        which_motion_msg.data = float(self.motion_idx)
        self.which_motion_pub.publish(which_motion_msg)

        pelvis_pos  = self.mj_data.body("pelvis").xpos.astype(np.float32)
        pelvis_quat = self.mj_data.body("pelvis").xquat.astype(np.float32)
        R           = quat_to_rotation_matrix(pelvis_quat)

        # Box centre in pelvis frame (live physics position).
        box_center_w  = self.mj_data.body("box").xpos.astype(np.float32)
        box_in_pelvis = R.T @ (box_center_w - pelvis_pos)

        # Box orientation: rotate grasp offsets from box frame → world → pelvis frame.
        box_quat = self.mj_data.body("box").xquat.astype(np.float32)
        R_box    = quat_to_rotation_matrix(box_quat)

        # Goals 1 & 2: offsets are in box frame so they track both box translation and
        # rotation.  R_box rotates them to world frame; R.T brings them to pelvis frame.
        right_goal = (box_in_pelvis + R.T @ R_box @ self._right_grasp_offsets[self.motion_idx]).astype(np.float32)
        left_goal  = (box_in_pelvis + R.T @ R_box @ self._left_grasp_offsets[self.motion_idx]).astype(np.float32)

        # Goal 3 (otherhand): the per-episode offset in right-palm local frame.
        # Training (commands.py) emits _moving_target_offset_w directly for dynamic
        # targets — the constant sampled offset in the target body's local frame — with
        # NO frame transformation.  No live FK required; just pass the offset as-is.
        otherhand_goal = (self._otherhand_offsets[self.motion_idx]           # [0, 0.25, 0] nominal
                          if self._otherhand_wide
                          else np.array([0.0, 0.35, 0.0], dtype=np.float32))  # [0, 0.40, 0] wide

        msg      = Float32MultiArray()
        msg.data = np.concatenate([right_goal, left_goal, otherhand_goal]).tolist()
        self.goal_pub.publish(msg)

    #################################################################
    # VISUALIZATION
    #################################################################

    def _update_viz(self) -> None:
        """
        Draw overlay spheres showing the goal grasp targets.
        The box itself is rendered by the physics engine.

          geom[0] — orange sphere: right-palm target
          geom[1] — green  sphere: left-palm  target
        """
        box_center_w = self.mj_data.body("box").xpos.astype(np.float64)
        box_quat     = self.mj_data.body("box").xquat.astype(np.float32)
        R_box        = quat_to_rotation_matrix(box_quat).astype(np.float64)

        # Grasp offsets are in box frame; rotate to world frame for visualisation.
        r_off     = self._right_grasp_offsets[self.motion_idx].astype(np.float64)
        l_off     = self._left_grasp_offsets[self.motion_idx].astype(np.float64)
        right_tgt = box_center_w + R_box @ r_off
        left_tgt  = box_center_w + R_box @ l_off

        scn       = self.viewer.user_scn
        scn.ngeom = 0

        mujoco.mjv_initGeom(
            scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.04, 0.0, 0.0], dtype=np.float64),
            right_tgt,
            np.eye(3, dtype=np.float64).flatten(),
            np.array([1.0, 0.5, 0.0, 0.8], dtype=np.float64),   # orange
        )
        mujoco.mjv_initGeom(
            scn.geoms[1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.04, 0.0, 0.0], dtype=np.float64),
            left_tgt,
            np.eye(3, dtype=np.float64).flatten(),
            np.array([0.0, 0.9, 0.1, 0.8], dtype=np.float64),   # green
        )
        scn.ngeom = 2

    #################################################################
    # SIMULATION
    #################################################################

    def _apply_sensor_noise(self):
        for i in range(self.mj_model.nsensor):
            std = self.mj_model.sensor_noise[i]
            if std <= 0.0:
                continue
            adr = self.mj_model.sensor_adr[i]
            dim = self.mj_model.sensor_dim[i]
            self.mj_data.sensordata[adr : adr + dim] += np.random.normal(
                0.0, std, size=dim
            )

    def _compute_torque(self):
        qpos_joints = self.mj_data.qpos[7 : 7 + self.nu]
        qvel_joints = self.mj_data.qvel[6 : 6 + self.nu]
        return (
            self.Kp * (self.qpos_des - qpos_joints)
            + self.Kd * (self.qvel_des - qvel_joints)
            + self.tau_ff
        )

    def _step_simulation(self):
        # ------------------------------------------------------------------
        # Arrow-key box movement: override qpos + zero velocity so physics
        # doesn't fight the teleop input.  Once keys are released, the box
        # is governed by physics and can be picked up normally.
        #
        # Since the viewer callback only delivers the key code (no
        # press/release flag), we time out the velocity if no key event has
        # arrived in the last _KEY_TIMEOUT seconds.
        # ------------------------------------------------------------------
        if np.any(self._box_key_vel != 0.0):
            if time.perf_counter() - self._box_key_last_t > 0.20:
                self._box_key_vel[:] = 0.0
            else:
                curr = self.mj_data.qpos[
                    self._box_qpos_adr : self._box_qpos_adr + 3
                ].copy()
                new_pos = np.clip(
                    curr + self._box_key_vel * self.sim_dt,
                    [-_BOX_X_LIMIT, -_BOX_Y_LIMIT, _BOX_HALF_Z],
                    [ _BOX_X_LIMIT,  _BOX_Y_LIMIT, _BOX_Z_MAX],
                )
                self.mj_data.qpos[self._box_qpos_adr : self._box_qpos_adr + 3] = new_pos
                self.mj_data.qvel[self._box_dof_adr  : self._box_dof_adr  + 6] = 0.0

        if self.command_received:
            self.mj_data.ctrl[:] = self._compute_torque()
        else:
            self.mj_data.ctrl[:] = 0.0

        mujoco.mj_step(self.mj_model, self.mj_data)

        # Auto-select the nearest motion based on the live box position.
        # Compare box centre in pelvis frame against each motion's nominal box
        # centre (midpoint of right and left pre-grab targets from YAML).
        # Only runs when no pickup is in progress to avoid switching mid-motion.
        if not self._motion_in_progress and len(self._nominal_box_centers) > 1:
            _pelvis_pos = self.mj_data.body("pelvis").xpos.astype(np.float32)
            _R          = quat_to_rotation_matrix(
                              self.mj_data.body("pelvis").xquat.astype(np.float32))
            _box_w      = self.mj_data.body("box").xpos.astype(np.float32)
            _box_p      = _R.T @ (_box_w - _pelvis_pos)
            _dists      = np.linalg.norm(self._nominal_box_centers - _box_p, axis=1)
            _new_idx    = int(np.argmin(_dists))
            if _new_idx != self.motion_idx:
                self.motion_idx = _new_idx
                self._init_goals()

        if self.apply_noise:
            self._apply_sensor_noise()

        time_msg      = Float64()
        time_msg.data = self.mj_data.time
        self.simulation_time_pub.publish(time_msg)

        now = time.perf_counter()
        if (
            self.viewer.is_running()
            and (now - self._last_viewer_sync) >= 1.0 / self.viewer_render_hz
        ):
            self._update_viz()
            self.viewer.sync()
            real_elapsed = now - self._real_start_time
            box_pos      = self.mj_data.body("box").xpos
            self.viewer.set_texts(
                (
                    self._viewer_font_scale,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    (
                        f"Sim time:  {self.mj_data.time:.2f}s\n"
                        f"Real time: {real_elapsed:.2f}s\n"
                        f"Motion:    pickup_box\n"
                        f"Box  x={box_pos[0]:.2f}"
                        f"  y={box_pos[1]:.2f}"
                        f"  z={box_pos[2]:.2f}\n"
                        f"Grip width: {'nominal 0.25 [Y→0.40]' if self._otherhand_wide else 'wide 0.40 [Y→0.25]'}"
                    ),
                    "",
                )
            )
            self._last_viewer_sync = now

        remaining = self._next_step_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        self._next_step_deadline += self.sim_dt

    def destroy_node(self):
        if self.viewer.is_running():
            self.viewer.close()
        super().destroy_node()


############################################################################
# MAIN
############################################################################


def main(args=None):
    rclpy.init()

    parser = argparse.ArgumentParser(description="Box-pickup simulation node.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--noise",  action="store_true",    help="Enable sensor noise.")
    args = parser.parse_args()

    sim_node = SimulationNode(args.config, apply_noise=args.noise)

    try:
        while rclpy.ok() and sim_node.viewer.is_running():
            rclpy.spin_once(sim_node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("Simulation shutdown complete.")


if __name__ == "__main__":
    main()
