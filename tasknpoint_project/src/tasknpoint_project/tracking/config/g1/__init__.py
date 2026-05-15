from mjlab.tasks.registry import register_mjlab_task
from tasknpoint_project.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_multi_target_tracking_env_cfg,
)
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-MultiTarget-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_multi_target_tracking_env_cfg(),
  play_env_cfg=unitree_g1_multi_target_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
