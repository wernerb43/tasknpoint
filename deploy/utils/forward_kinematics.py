##
#
# Forward kinematics helpers for the G1 robot in MuJoCo.
#
# After mj_step (or an explicit mj_kinematics call) every body/site xpos and
# xmat are already up-to-date in MjData — these helpers just read those
# values and express them in convenient reference frames.
#
# NOTE: call mujoco.mj_kinematics(model, data) before using these during
# initialisation (before the first mj_step).  During the control loop
# mj_step keeps everything current automatically.
#
##

import numpy as np

# ---------------------------------------------------------------------------
# Named sites / bodies used as end-effectors
# ---------------------------------------------------------------------------

#: MuJoCo site names defined on the wrist-yaw links (see g1_27dof_deploy_tennis.xml)
SITE_LEFT_PALM  = "left_palm"
SITE_RIGHT_PALM = "right_palm"

#: Most-distal body on each arm (parent of the palm sites)
BODY_LEFT_WRIST  = "left_wrist_yaw_link"
BODY_RIGHT_WRIST = "right_wrist_yaw_link"


# ---------------------------------------------------------------------------
# World-frame accessors
# ---------------------------------------------------------------------------

def left_palm_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the left palm site.  Shape (3,), float32."""
    return mj_data.site(SITE_LEFT_PALM).xpos.astype(np.float32).copy()


def right_palm_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the right palm site.  Shape (3,), float32."""
    return mj_data.site(SITE_RIGHT_PALM).xpos.astype(np.float32).copy()


def left_wrist_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the left wrist-yaw body origin.  Shape (3,), float32."""
    return mj_data.body(BODY_LEFT_WRIST).xpos.astype(np.float32).copy()


def right_wrist_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the right wrist-yaw body origin.  Shape (3,), float32."""
    return mj_data.body(BODY_RIGHT_WRIST).xpos.astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Frame-relative accessors
# ---------------------------------------------------------------------------

def pos_in_frame(
    pos_world: np.ndarray,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """
    Express a world-frame position in a given reference frame.

    Parameters
    ----------
    pos_world : (3,) world-frame position
    R_frame   : (3, 3) rotation matrix of the reference frame (columns = frame axes)
    p_frame   : (3,) world-frame origin of the reference frame

    Returns
    -------
    (3,) position expressed in the reference frame, float32
    """
    return (R_frame.T @ (pos_world - p_frame)).astype(np.float32)


def left_palm_in_frame(
    mj_data,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """Left palm position expressed in the given reference frame.  Shape (3,)."""
    return pos_in_frame(left_palm_pos_world(mj_data), R_frame, p_frame)


def right_palm_in_frame(
    mj_data,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """Right palm position expressed in the given reference frame.  Shape (3,)."""
    return pos_in_frame(right_palm_pos_world(mj_data), R_frame, p_frame)


# ---------------------------------------------------------------------------
# Relative hand geometry
# ---------------------------------------------------------------------------

def left_wrt_right_world(mj_data) -> np.ndarray:
    """
    Vector from the right palm to the left palm in the world frame.

    This captures the relative bimanual configuration independent of
    pelvis pose, and is useful as a goal when expressed in a body frame.

    Returns
    -------
    (3,) float32
    """
    return left_palm_pos_world(mj_data) - right_palm_pos_world(mj_data)


def left_wrt_right_in_frame(
    mj_data,
    R_frame: np.ndarray,
) -> np.ndarray:
    """
    Vector from the right palm to the left palm, expressed in the given frame.

    Only the orientation of the reference frame matters (not its origin), so
    only R_frame is required.

    Parameters
    ----------
    R_frame : (3, 3) rotation matrix of the reference frame

    Returns
    -------
    (3,) float32
    """
    return (R_frame.T @ left_wrt_right_world(mj_data)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pelvis-frame convenience wrappers
# ---------------------------------------------------------------------------

def pelvis_frame(mj_data):
    """
    Return (R_pelvis, p_pelvis) — the current pelvis rotation matrix and
    world-frame origin — ready to pass to the frame helpers above.

    Uses the ``pelvis`` body which is always present in the G1 models.
    """
    from utils.math_utils import quat_to_rotation_matrix  # local import to avoid circular deps
    p = mj_data.body("pelvis").xpos.astype(np.float32).copy()
    R = quat_to_rotation_matrix(mj_data.body("pelvis").xquat.astype(np.float32))
    return R, p


def left_palm_in_pelvis(mj_data) -> np.ndarray:
    """Left palm position in the pelvis frame.  Shape (3,)."""
    R, p = pelvis_frame(mj_data)
    return left_palm_in_frame(mj_data, R, p)


def right_palm_in_pelvis(mj_data) -> np.ndarray:
    """Right palm position in the pelvis frame.  Shape (3,)."""
    R, p = pelvis_frame(mj_data)
    return right_palm_in_frame(mj_data, R, p)


def left_wrt_right_in_pelvis(mj_data) -> np.ndarray:
    """
    Vector from the right palm to the left palm, expressed in the pelvis frame.
    Shape (3,).
    """
    R, _ = pelvis_frame(mj_data)
    return left_wrt_right_in_frame(mj_data, R)


# ---------------------------------------------------------------------------
# Pickup-task helper  (both goals in one call)
# ---------------------------------------------------------------------------

def pickup_fk_goals(
    mj_data,
    R_pelvis: np.ndarray,
    p_pelvis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two FK-derived goal vectors for the box-pickup task.

    During pickup the policy receives two goal vectors:
      1. box_pos_in_pelvis  — where the box is (computed externally, passed in)
                              *not* computed here; see simulation_box.py
      2. left_wrt_right     — vector from right palm → left palm in pelvis frame
                              computed here via FK on the current robot state

    This function returns the *second* goal only (left_wrt_right_in_pelvis),
    since the box position is managed by the simulation node's joystick logic.

    Parameters
    ----------
    mj_data   : live MuJoCo MjData (xpos must be current, i.e. after mj_step)
    R_pelvis  : (3, 3) current pelvis rotation matrix
    p_pelvis  : (3,)  current pelvis world-frame position

    Returns
    -------
    left_palm_in_pelvis   : (3,) float32
        Absolute left-palm position in the pelvis frame (useful for debugging).
    left_wrt_right        : (3,) float32
        Vector right-palm → left-palm expressed in the pelvis frame.
        Pass this as the second entry in the published goals vector.
    """
    lp_w = left_palm_pos_world(mj_data)
    rp_w = right_palm_pos_world(mj_data)

    R_inv = R_pelvis.T
    left_abs   = (R_inv @ (lp_w - p_pelvis)).astype(np.float32)
    left_wrt_r = (R_inv @ (lp_w - rp_w)).astype(np.float32)

    return left_abs, left_wrt_r
