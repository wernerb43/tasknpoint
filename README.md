# Pipeline steps:

### step 1: retarget motions
```
# ilona: set rendering to stream
xvfb-run -a python retarget/construct_motion.py \
    --smplx_file /mnt/datasets/robo_results/retarget_inputs/backhand.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/backhand \
    --robot_xml robots/g1_27dof.xml \
    --rate_limit

# to visualize:
xvfb-run -a python retarget/construct_motion.py \
    --smplx_file /mnt/datasets/robo_results/retarget_inputs/backhand.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/backhand \
    --robot_xml robots/g1_27dof.xml \
    --rate_limit \
    --record_video
```

### step 2: convert to .npz format
```
MUJOCO_GL=egl uv run --directory tasknpoint_project python \
    /home/ilona/human_motion/robo/tasknpoint/tasknpoint_project/src/tasknpoint_project/scripts/csv_to_npz.py \
    --input-file ../retarget/retarget_outputs/backhand.csv \
    --output-name backhand \
    --input-fps 30 \
    --output-fps 50 \
    --render False
```

### step 3: launch training

All motion config lives in `src/tasknpoint_project/motion_sets/`:
- `motion_lib.py` — motion specs (positions, phases, weights)
- `motion_train_configs/*.toml` — train sets: which motions, registry prefixes, robot XML

Run from `tasknpoint_project/`:

```
# single motion (pass registry name directly):
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --registry-name demalenk-california-institute-of-technology-caltech/csv_to_npz/backhand \
    --env.scene.num-envs 4096

# motion set — all motions (TOML drives registry + robot XML):
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/all_motions.toml \
    --env.scene.num-envs 4096

# motion set — tennis only:
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml \
    --env.scene.num-envs 4096

# motion set — kicks only:
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --env.scene.num-envs 4096
```

To see which motions a set contains:
```
uv run motion-set src/tasknpoint_project/motion_sets/motion_train_configs/all_motions.toml --list
```

## Evals

Run from `tasknpoint_project/`:

```
uv run python -m tasknpoint_project.goal_cond_tracking.scripts.evaluate \
    Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/awxs697x \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --target-x-min 0.3 --target-x-max 0.7 \
    --target-y-min -0.8 --target-y-max 0.0 \
    --target-z-min -0.72 --target-z-max -0.71
```

The `--eval` flag uses the eval registry prefix (`wandb-registry-motions`).
Omitting it uses the train prefix (`csv_to_npz`).
