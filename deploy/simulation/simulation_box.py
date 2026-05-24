##
#
# Simulation node for the box-pickup task.
#
# The box is a real MuJoCo free-body that the robot can physically interact
# with.  Arrow keys move the box; B button on the joystick triggers the motion.
#
# Goal vector published each tick (6 floats):
#   [0:3]  right-palm target  = box_center_in_pelvis + [0, -GRASP_OFFSET, 0]
#   [3:6]  left-palm target   = right_palm_in_pelvis  + otherhand_offset   (FK)
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
sys.path.insert(0, str(REPO_ROOT / "submodules" / "deploy_robot"))
sys.path.insert(0, str(REPO_ROOT / "deploy" / "utils"))

from utils.math_utils import (
    quat_conjugate,
    quat_multiply,
    quat_to_rotation_matrix,
    quat_to_rpy,
)
from forward_kinematics import pickup_fk_goals

# ---------------------------------------------------------------------------
# Box geometry constants
# ---------------------------------------------------------------------------
_BOX_HALF_X = 0.10    # depth  half-extent (m)
_BOX_HALF_Y = 0.1   # width  half-extent (m) — = _GRASP_OFFSET
_BOX_HALF_Z = 0.2   # height half-extent (m)
_BOX_MASS   = 0.1     # kg

# Each palm target is _GRASP_OFFSET from the box centre along the pelvis Y-axis.
# Right hand:  box_in_pelvis + [0, -_GRASP_OFFSET, 0]   (robot's right)
# Left hand:   box_in_pelvis + [0, +_GRASP_OFFSET, 0]   (robot's left)
# Consistent with otherhand_offset=[0, 0.25, 0] since 2×_GRASP_OFFSET = 0.25.
_GRASP_OFFSET = _BOX_HALF_Y+0.1   # 0.125 m

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

    Goal vector (6 floats) published each control tick:
      [0:3]  right-palm target in pelvis frame
             = box_center_in_pelvis  +  [0, -_GRASP_OFFSET, 0]
      [3:6]  left-palm target in pelvis frame  (FK, not static)
             = right_palm_in_pelvis  +  otherhand_offset
    """

    def __init__(self, config_path: str, apply_noise: bool = False):
        super().__init__("simulation_node")

        self.config      = self._load_config(config_path)
        self.apply_noise = apply_noise

        self._init_params()
        self._init_simulation()   # loads model with box injected, launches viewer

        # ------------------------------------------------------------------
        # Place the box at its initial world position (derived from YAML
        # right-palm target rotated into world frame via default pelvis pose).
        # ------------------------------------------------------------------
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        _R0 = quat_to_rotation_matrix(self.mj_data.body("pelvis").xquat.astype(np.float32))
        _p0 = self.mj_data.body("pelvis").xpos.astype(np.float32)

        _box_goal = next(
            g for g in self.config.get("goals", [])
            if g["type"] == "position"
            and "otherhand" not in g["name"]
            and g["motion_index"] == 0
        )
        # YAML vector is the RIGHT-PALM target in pelvis frame.
        # Box centre is _GRASP_OFFSET to the robot's left of that.
        _right_target_pelvis = np.array(_box_goal["vector"], dtype=np.float32)
        _box_center_pelvis   = _right_target_pelvis + np.array([0.0, _GRASP_OFFSET, 0.0])
        _box_center_world    = _R0 @ _box_center_pelvis + _p0
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

        self.create_timer(0.0,         self._step_simulation)
        self.create_timer(self.sim_dt, self._publish_pelvis_imu)
        self.create_timer(self.sim_dt, self._publish_joint_state)
        self.create_timer(self.sim_dt, self._publish_goals)

        print("Simulation node initialized.")
        print("    Press [Tab] to toggle the left UI.")
        print("    Press [Shift + Tab] to toggle the right UI.")
        print("    Arrow keys : move box  (Up/Down = X,  Left/Right = Y,  PgUp/PgDn = Z)")
        print("    Joystick B : trigger pickup motion")
        print(f"    Box initial position (world): {_box_center_world.tolist()}")

    #################################################################
    # INITIALIZATION
    #################################################################

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.is_absolute():
            candidate = REPO_ROOT / "deploy" / "configs" / config_path
            if candidate.exists():
                path = candidate
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from [{path}].")
        return config

    def _init_params(self):
        self.default_base   = np.array(self.config["default_base_pos"])
        self.default_joints = np.array(self.config["default_joint_pos"])

    def _init_simulation(self):
        xml_path = REPO_ROOT / "robots" / self.config["xml_path"]

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
        Build per-goal metadata from the config.

        The ``"otherhand"`` position goal carries the grip-width offset
        (``[0, 0.25, 0]`` in pelvis frame) used for the FK computation.
        """
        goals_cfg   = self.config.get("goals", [])
        pelvis_quat = self.mj_data.body("pelvis").xquat.astype(np.float32)

        self._goal_types:  list[str]        = []
        self._goal_is_fk:  list[bool]       = []
        self._goal_vel_w:  list[np.ndarray] = []
        self._goal_quat_w: list[np.ndarray] = []

        for goal in [g for g in goals_cfg if g["motion_index"] == 0]:
            vec   = np.array(goal["vector"], dtype=np.float32)
            gtype = goal["type"]
            is_fk = "otherhand" in goal["name"]

            self._goal_types.append(gtype)
            self._goal_is_fk.append(is_fk)

            if gtype == "velocity":
                self._goal_vel_w.append(vec)
                self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif gtype == "orientation":
                self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
                self._goal_quat_w.append(quat_multiply(pelvis_quat, rpy_to_quat(vec)))
            else:  # position
                self._goal_vel_w.append(np.zeros(3, dtype=np.float32))
                self._goal_quat_w.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
                if is_fk:
                    # vec is the grip-width offset in pelvis/anchor frame.
                    # Matches target_pos_mean from motion_lib.py MotionGoalCfg.
                    # Build two presets: nominal ± 0.2 m on the Y (grip-width) axis.
                    _TOGGLE_DELTA = 0.3
                    nominal = vec.copy()
                    self._otherhand_offsets = [
                        nominal + np.array([0.0, -_TOGGLE_DELTA, 0.0], dtype=np.float32),
                        nominal + np.array([0.0,  _TOGGLE_DELTA, 0.0], dtype=np.float32),
                    ]
                    self._otherhand_idx    = 0   # Y button cycles through presets
                    self._otherhand_offset = self._otherhand_offsets[0].copy()
                    self._y_btn_prev       = False  # debounce state

        names = [g["name"] for g in goals_cfg if g["motion_index"] == 0]
        print(f"Goals initialized for pickup motion: {names}")
        print(f"    otherhand offset presets:")
        for i, off in enumerate(self._otherhand_offsets):
            marker = " <-- active" if i == self._otherhand_idx else ""
            print(f"      [{i}] {off.tolist()}{marker}")

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
        B button (data[4]) — trigger pickup motion.
        Y button (data[6]) — cycle otherhand offset preset (nominal ± 0.2 m).
        """
        if len(msg.data) > 4 and float(msg.data[4]) > 0.5:
            self._motion_in_progress = True

        # Y button: rising-edge toggle so one press = one switch.
        y_btn = len(msg.data) > 6 and float(msg.data[6]) > 0.5
        if y_btn and not self._y_btn_prev:
            self._otherhand_idx    = (self._otherhand_idx + 1) % len(self._otherhand_offsets)
            self._otherhand_offset = self._otherhand_offsets[self._otherhand_idx].copy()
            print(f"[Y] otherhand offset → preset {self._otherhand_idx}: "
                  f"{self._otherhand_offset.tolist()}")
        self._y_btn_prev = y_btn

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
        which_motion_msg.data = 0.0
        self.which_motion_pub.publish(which_motion_msg)

        if not self._goal_types:
            return

        pelvis_pos      = self.mj_data.body("pelvis").xpos.astype(np.float32)
        pelvis_quat     = self.mj_data.body("pelvis").xquat.astype(np.float32)
        R               = quat_to_rotation_matrix(pelvis_quat)
        pelvis_quat_inv = quat_conjugate(pelvis_quat)

        # Box centre in pelvis frame (live physics position)
        box_center_w  = self.mj_data.body("box").xpos.astype(np.float32)
        box_in_pelvis = R.T @ (box_center_w - pelvis_pos)

        goal_vecs = []
        for gtype, is_fk, vel_w, quat_w in zip(
            self._goal_types, self._goal_is_fk,
            self._goal_vel_w,  self._goal_quat_w,
        ):
            if gtype == "position" and not is_fk:
                # Right-palm target: _GRASP_OFFSET to the robot's right of box centre.
                # Pelvis-frame Y-negative = robot's right.
                right_target = box_in_pelvis + np.array(
                    [0.0, -_GRASP_OFFSET, 0.0], dtype=np.float32
                )
                goal_vecs.append(right_target)

            elif gtype == "position" and is_fk:
                # Left-palm target: right_palm_in_pelvis + otherhand_offset.
                # Matches training: target_pos_w = right_palm_w + R_pelvis @ offset
                #   → obs = right_palm_in_pelvis + offset  (commands.py command property)
                goal_vecs.append(
                    pickup_fk_goals(self.mj_data, R, pelvis_pos, self._otherhand_offset)
                )

            elif gtype == "velocity":
                goal_vecs.append(vel_w)

            elif gtype == "orientation":
                goal_vecs.append(quat_multiply(pelvis_quat_inv, quat_w))

        msg = Float32MultiArray()
        msg.data = np.concatenate(goal_vecs).tolist()
        self.goal_pub.publish(msg)

    #################################################################
    # VISUALIZATION
    #################################################################

    def _update_viz(self) -> None:
        """
        Draw overlay spheres showing the goal grasp targets.
        The box itself is rendered by the physics engine.

          geom[0] — orange sphere: right-palm target (right side of box)
          geom[1] — green  sphere: left-palm  target (left  side of box)
        """
        box_center_w = self.mj_data.body("box").xpos.astype(np.float64)
        pelvis_quat  = self.mj_data.body("pelvis").xquat.astype(np.float32)
        R            = quat_to_rotation_matrix(pelvis_quat).astype(np.float64)

        grasp_vec  = R @ np.array([0.0, _GRASP_OFFSET, 0.0])  # pelvis-Y → world
        right_tgt  = box_center_w - grasp_vec   # robot's right side
        left_tgt   = box_center_w + grasp_vec   # robot's left  side

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
                        f"  z={box_pos[2]:.2f}"
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
