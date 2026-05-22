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
    --smplx_file /mnt/datasets/robo_results/retarget_inputs/03_03_2026_16_30_court6/backhand_Andrew_Zabelo_03_03_2026_16_32_26_000_6_E_01_17_41_03_481_19112_19254_pid21.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/robo_forehand \
    --robot_xml robots/g1_27dof.xml \
    --rate_limit \
    --record_video 

xvfb-run -a python retarget/construct_motion.py \
    --smplx_file /mnt/datasets/robo_results/retarget_inputs/pickup_bench_to_floor.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/pickup_bench_to_floor \
    --robot_xml robots/g1_27dof.xml \
    --rate_limit \
    --record_video

xvfb-run -a python retarget/construct_motion.py \
    --smplx_file /mnt/datasets/robo_results/retarget_inputs/pickup_bench_to_floor_2.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/pickup_bench_to_floor_2 \
    --robot_xml robots/g1_27dof.xml \
    --rate_limit \
    --record_video \
    --follow_camera

xvfb-run -a python retarget/construct_motion.py     --smplx_file /mnt/datasets/robo_results/retarget_inputs/pickup_bench_to_floor_3.npz     --robot unitree_g1     --save_path retarget/retarget_outputs/pickup_bench_to_floor_3     --robot_xml robots/g1_27dof.xml     --rate_limit     --record_video --follow_camera

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

MUJOCO_GL=egl uv run python src/tasknpoint_project/scripts/csv_to_npz.py \
    --input-file ../retarget/retarget_outputs/one_step_forehand.csv \
    --output-name one_step_forehand \
    --input-fps 30 \
    --output-fps 50 \
    --render False

MUJOCO_GL=egl uv run python src/tasknpoint_project/scripts/csv_to_npz.py \
    --input-file ../retarget/retarget_outputs/pickup_bench_to_floor_2.csv \
    --output-name pickup_bench_to_floor_2 \
    --input-fps 30 \
    --output-fps 50 \
    --render False \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/box_grab.toml

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

# motion set — tennis only expanded:
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only_expanded.toml \
    --env.scene.num-envs 4096

# motion set — kicks only:
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --env.scene.num-envs 4096

# box grab:
uv run train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/box_grab.toml \
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
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/r49k4cin \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --target-x-min 0.3 --target-x-max 0.7 \
    --target-y-min -0.8 --target-y-max 0.0 \
    --target-z-min -0.72 --target-z-max -0.71

uv run python -m tasknpoint_project.goal_cond_tracking.scripts.evaluate \
    Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/48cdla54 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml \
    --target-x-min 0.4 --target-x-max 0.6 \
    --target-y-min -2.0 --target-y-max 2.0 \
    --target-z-min -0.4 --target-z-max 0.40
```

The `--eval` flag uses the eval registry prefix (`wandb-registry-motions`).
Omitting it uses the train prefix (`csv_to_npz`).

## Play script:

```
uv run play Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/i4tr5j7v \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only_fast.toml

```

To run with VISER!:

```
 uv run play-viser Mjlab-MultiTarget-Tracking-Flat-Unitree-G1     --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/0ulgzbgg     --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/box_grab.toml
```

To also visualize blobs:
```
uv run play-viser-motion-ranges Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/48cdla54 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml
```