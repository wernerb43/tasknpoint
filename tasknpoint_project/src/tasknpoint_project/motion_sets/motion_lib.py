"""Motion library — canonical specs for every motion.

Add a new motion here and reference it by name in a motion_sets/*.toml file.
"""

import numpy as np

from tasknpoint_project.goal_cond_tracking.mdp import MotionCfg, MotionGoalCfg

vel_ori_window = 0.004 # phase window before and after position window
kick_window_x = 0.100
kick_window_y = 0.100
kick_window_z = 0.100

MOTION_LIB: dict[str, MotionCfg] = {
  "right_kick": MotionCfg(
    name="right_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.293)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.290,
        target_phase_end=0.297,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.285,
        target_phase_end=0.302,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.290,
        target_phase_end=0.297,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "left_kick": MotionCfg(
    name="left_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.344)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.0, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.340,
        target_phase_end=0.347,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.335,
        target_phase_end=0.352,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.340,
        target_phase_end=0.347,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "right_far_kick": MotionCfg(
    name="right_far_kick",
    sampling_weight=1.0,
    probe_points=[("right_foot", 0.408)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_foot",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.404,
        target_phase_end=0.411,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.399,
        target_phase_end=0.416,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="right_foot",
        source_type="site",
        target_phase_start=0.404,
        target_phase_end=0.411,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "left_far_kick": MotionCfg(
    name="left_far_kick",
    sampling_weight=1.0,
    probe_points=[("left_foot", 0.405)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_foot",
        source_type="site",
        target_pos_mean={"x": 1.0, "y": 0.6, "z": -0.70},
        target_pos_std={"x": 0.10 + kick_window_x, "y": 0.20 + kick_window_y, "z": 0.01 + kick_window_z},
        target_phase_start=0.401,
        target_phase_end=0.408,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.396,
        target_phase_end=0.413,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.0},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="left_foot",
        source_type="site",
        target_phase_start=0.401,
        target_phase_end=0.408,
        target_orientation_mean={"roll": 0.0, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "forehand": MotionCfg(
    name="forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.310)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.306,
        target_phase_end=0.313,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.300,
        target_phase_end=0.320,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.306,
        target_phase_end=0.313,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "backhand": MotionCfg(
    name="backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.372)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.368,
        target_phase_end=0.375,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.360,
        target_phase_end=0.380,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.368,
        target_phase_end=0.375,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_forehand": MotionCfg(
    name="two_step_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.399)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.394,
        target_phase_end=0.403,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.390,
        target_phase_end=0.410,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.394,
        target_phase_end=0.403,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_backhand": MotionCfg(
    name="two_step_backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.452)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.448,
        target_phase_end=0.455,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.440,
        target_phase_end=0.460,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448,
        target_phase_end=0.455,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_forehand": MotionCfg(
    name="stepback_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.337)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.333,
        target_phase_end=0.340,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.328,
        target_phase_end=0.345,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.333,
        target_phase_end=0.340,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_backhand": MotionCfg(
    name="stepback_backhand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.398,
        target_phase_end=0.405,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.390,
        target_phase_end=0.410,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398,
        target_phase_end=0.405,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  ################# new tennis motions with steps ##################
  "one_step_forehand": MotionCfg(
    name="one_step_forehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5589, "y": -0.4391, "z": 0.3129},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.398,
        target_phase_end=0.407,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.383,
        target_phase_end=0.422,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.383,
        target_phase_end=0.422,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "forehand_sidestep": MotionCfg(
    name="forehand_sidestep",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.429)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.4038, "y": 0.2925, "z": 0.3185},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.425,
        target_phase_end=0.433,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.410,
        target_phase_end=0.448,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.410,
        target_phase_end=0.448,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "forehand_two_sidestep": MotionCfg(
    name="forehand_two_sidestep",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.507)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.6079, "y": -1.4088, "z": -0.7081},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.503,
        target_phase_end=0.512,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.488,
        target_phase_end=0.527,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.488,
        target_phase_end=0.527,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "backhand_onehand": MotionCfg(
    name="backhand_onehand",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.321)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.74, "y": 0.03, "z": 0.2426},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.318,
        target_phase_end=0.324,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.303,
        target_phase_end=0.339,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.303,
        target_phase_end=0.339,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  "backhand_onehand_step": MotionCfg(
    name="backhand_onehand_step",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.447)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5105, "y": 0.7421, "z": 0.1194},
        target_pos_std={"x": 0.10, "y": 0.40, "z": 0.40},
        target_phase_start=0.443,
        target_phase_end=0.451,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.428,
        target_phase_end=0.466,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.428,
        target_phase_end=0.466,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  ########################################################## Fast motions ###############################################################

  "forehand_fast": MotionCfg(
    name="forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.386)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.381,
        target_phase_end=0.390,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.390 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.381 - vel_ori_window,
        target_phase_end=0.390 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "backhand_fast": MotionCfg(
    name="backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.397)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.6, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.393,
        target_phase_end=0.400,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.400 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.393 - vel_ori_window,
        target_phase_end=0.400 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_forehand_fast": MotionCfg(
    name="two_step_forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.427)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.423,
        target_phase_end=0.430,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.423 - vel_ori_window,
        target_phase_end=0.430 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.423 - vel_ori_window,
        target_phase_end=0.430 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "two_step_backhand_fast": MotionCfg(
    name="two_step_backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.452)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 1.2, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.448,
        target_phase_end=0.455,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448 - vel_ori_window,
        target_phase_end=0.455 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.448 - vel_ori_window,
        target_phase_end=0.455 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_forehand_fast": MotionCfg(
    name="stepback_forehand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.355)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": 0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.351,
        target_phase_end=0.358,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.351 - vel_ori_window,
        target_phase_end=0.358 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.351 - vel_ori_window,
        target_phase_end=0.358 + vel_ori_window,
        target_orientation_mean={"roll": np.pi / 8, "pitch": 0.0, "yaw": -np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),
  "stepback_backhand_fast": MotionCfg(
    name="stepback_backhand_fast",
    sampling_weight=1.0,
    probe_points=[("racket_contact", 0.402)],
    sub_targets=[
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="racket_contact",
        source_type="site",
        target_pos_mean={"x": 0.5, "y": -0.1, "z": 0.0},
        target_pos_std={"x": 0.10, "y": 0.30, "z": 0.30},
        target_phase_start=0.398,
        target_phase_end=0.405,
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398 - vel_ori_window,
        target_phase_end=0.405 + vel_ori_window,
        target_vel_mean={"x": 1.0, "y": 0.0, "z": 0.1},
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.2},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=10.0,
        source_link="racket_contact",
        source_type="site",
        target_phase_start=0.398 - vel_ori_window,
        target_phase_end=0.405 + vel_ori_window,
        target_orientation_mean={"roll": -np.pi / 8, "pitch": 0.0, "yaw": np.pi / 2},
        target_orientation_std={"roll": 0.1, "pitch": 0.0, "yaw": 0.0},
        orientation_axis="y",
      ),
    ],
  ),

  #########################################
  #                                       #
  #          box pickups to floor         #
  #                                       #
  #########################################


    "pickup_bench_to_floor": MotionCfg(
    name="pickup_bench_to_floor",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.412), ("left_palm", 0.412)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.412 - vel_ori_window,
        target_phase_end=0.412 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.762, "y": 0.422, "z": -0.282},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.412 - vel_ori_window,
        target_phase_end=0.412 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.412,   # from when the box is grabbed
        target_phase_end=0.904,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.405, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="left_palm",
        source_type="site",
        target_phase_start=0.405, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": -0.5, "z": 0.0}, # pushing on the box to the right (y negative)
        target_vel_std={"x": 0.0, "y": 0.0, "z": 0.0},
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=3.85,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.359, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_orientation_mean={"roll": -0.444, "pitch": 0.531, "yaw": 0.121},  # this is the init frame orientation at the box grab frame
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.1},    # small std dev
        orientation_axis="y",
      ),
      MotionGoalCfg(
        goal_type="orientation",
        goal_weight=3.85,
        source_link="left_palm",
        source_type="site",
        target_phase_start=0.359, # a bit before when the box was grabbed
        target_phase_end=0.489, # this is when the person stood up
        target_orientation_mean={"roll": -0.208, "pitch": 0.4220, "yaw": 0.232},  # this is the init frame orientation at the box grab frame
        target_orientation_std={"roll": 0.0, "pitch": 0.0, "yaw": 0.1},    # small std dev
        orientation_axis="y",
      ),
    ],
  ),

  "pickup_bench_to_floor_2": MotionCfg(
    name="pickup_bench_to_floor_2",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.338), ("left_palm", 0.338)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7732, "y": 0.1720, "z": 0.0325},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.6513, "y": 0.4940, "z": -0.0271},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.338,   # from when the box is grabbed
        target_phase_end=0.9008,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.338, # a bit before when the box was grabbed
        target_phase_end=0.9008, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
    ],
  ),

  "pickup_bench_to_floor_3": MotionCfg(
    name="pickup_bench_to_floor_3",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.338), ("left_palm", 0.338)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left hand when box is grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="left_palm",
        source_type="site",
        target_pos_mean={"x": 1.762, "y": 0.422, "z": -0.282},
        target_pos_std={"x": 0.05, "y": 0.05, "z": 0.05},
        target_phase_start=0.338 - vel_ori_window,
        target_phase_end=0.338 + vel_ori_window,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=1.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",
        target_pos_mean={"x": -0.026, "y": 0.286, "z": 0.046},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.02, "z": 0.02},
        target_phase_start=0.338,   # from when the box is grabbed
        target_phase_end=0.9008,     # until the box is set down
      ),
      MotionGoalCfg(
        goal_type="velocity",
        goal_weight=6.25,
        source_link="right_palm",
        source_type="site",
        target_phase_start=0.338, # a bit before when the box was grabbed
        target_phase_end=0.9008, # this is when the person stood up
        target_vel_mean={"x": 0.0, "y": 0.5, "z": 0.0}, # pushing on the box to the left (y positive)
        target_vel_std={"x": 0.0, "y": 0.2, "z": 0.0},
      ),
    ],
  ),

 "pickup_box_1": MotionCfg(
    name="pickup_box_1",
    sampling_weight=1.0,
    probe_points=[("right_palm", 0.412), ("right_palm", 0.867)],
    sub_targets=[
      # Right hand when box gets grabbed
      MotionGoalCfg(
        goal_type="position",
        goal_weight=100.0,
        source_link="right_palm",
        source_type="site",
        target_pos_mean={"x": 1.7876, "y": 0.1359, "z": -0.328},
        target_pos_std={"x": 0.2, "y": 0.2, "z": 0.05},
        target_phase_start=0.412,
        target_phase_end=0.430,
      ),
      # Left palm tracks right palm live during the hold phase, offset by the grip width.
      # This prevents the box from being "dropped" by the left hand drifting away.
      MotionGoalCfg(
        goal_type="position",
        goal_weight=2.0,
        source_link="left_palm",
        source_type="site",
        target_link="right_palm",   # dynamic: follows right palm's live position
        target_type="site",

        target_pos_mean={"x": 0.0, "y": 0.25, "z": 0.0},  # left - right offset, anchor frame
        target_pos_std={"x": 0.02, "y": 0.2, "z": 0.02},
        target_phase_start=0.420,   # from when the box is grabbed
        target_phase_end=0.867,     # until the box is set down
      ),

    ],
  ),


}
