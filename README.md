<div align="center">
<h1> TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes </h1>
</div>

<div align="center">
  🌐 <a href="https://ilonadem.github.io/tasknpoint_website/">Project Page</a> |
  📚 <a href="https://arxiv.org/pdf/2606.26215">Paper</a>

</div>

## Installation

The repo uses two separate environments:

| Environment | Purpose | Manager |
|---|---|---|
| `tasknpoint` (uv) | Robot training / MuJoCo | `uv sync` |
| `phmr_pt2.6` (conda) | Video processing / PromptHMR | conda + pip |

### 1. Clone the repo with submodules

PromptHMR is included as a submodule — no separate clone needed.

```bash
git clone --recurse-submodules https://github.com/wernerb43/tasknpoint
# or, if already cloned:
git submodule update --init
```

### 2. Set up the tasknpoint environment

```bash
uv sync   # from repo root
```

The retargeting step (`retarget/construct_motion.py`) uses the GMR submodule
(`submodules/GMR`), which `uv sync` does not install. Install it and its
dependencies (`mink`, `smplx`, `qpsolvers`, etc.) into the `tasknpoint` env:

```bash
uv pip install -e submodules/GMR
```

GMR's interactive MuJoCo viewer needs an X display. On a machine **with** a
display (or admin access), install `xvfb` and run the retargeting commands under
`xvfb-run` (see Step 2 of the pipeline):

```bash
sudo apt install xvfb
```

**No sudo / no display?** Pass `--headless` to `construct_motion.py` instead. It
skips the interactive viewer entirely (no `xvfb-run`, no X display needed) and
renders any `--record_video` output offscreen via EGL. Drop the `xvfb-run`
prefix from the Step 2 commands and add `--headless`, e.g.:

```bash
uv run python retarget/construct_motion.py \
    --smplx_file retarget/retarget_inputs/boxing_test.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/boxing_test \
    --robot_xml robots/retargeting/g1_27dof.xml \
    --headless
```

All `construct_motion.py` invocations must run inside the `tasknpoint` uv env
(that's where GMR was installed above), hence `uv run python`. Bare `python`
uses whatever conda env is active and won't find GMR's dependencies.

Note: in headless mode the human joint-frame markers aren't drawn into recorded
video (they require the live viewer); the robot motion renders normally.

### 3. Set up the video processing environment

Run PromptHMR's provided install script, which creates the conda env and installs
torch, xformers, torch-scatter, chumpy, and all other deps:

```bash
cd submodules/PromptHMR
bash scripts/install.sh --pt_version=2.6 --world-video=true   # or --pt_version=2.4
conda activate phmr_pt2.6
cd ../..
```

Install the ffmpeg system binaries (needed for frame counting and video decoding):

```bash
conda install -c conda-forge ffmpeg -y
```

Then register PromptHMR and video_processing as importable packages.

PromptHMR's upstream repo ships no `pyproject.toml`, and since it's a git
submodule we can't track one inside it — so copy in the packaging file we keep
in this repo before installing:

```bash
cp video_processing/prompthmr_pyproject.toml submodules/PromptHMR/pyproject.toml

pip install -e submodules/PromptHMR --config-settings editable_mode=compat
pip install -e video_processing
```

### 4. Download model weights

From inside the PromptHMR submodule:

```bash
cd submodules/PromptHMR

# SMPL-X family body models (requires free account at smpl-x.is.tue.mpg.de)
bash scripts/fetch_smplx.sh

# PromptHMR checkpoints and annotations
bash scripts/fetch_data.sh

cd ../..
```

GMR also needs the SMPL-X body models, at `submodules/GMR/assets/body_models/`
(see [construct_motion.py](retarget/construct_motion.py)). Symlink the ones
fetched above so you don't download them twice:

```bash
mkdir -p submodules/GMR/assets/body_models
ln -s "$(pwd)/submodules/PromptHMR/data/body_models/smplx" \
      submodules/GMR/assets/body_models/smplx
```

Adjust the source path if `fetch_smplx.sh` placed the models elsewhere.

### 5. Make scripts executable (once)

```bash
chmod +x video_processing/run_prompthmr.sh \
         video_processing/run_prompthmr_single.sh \
         video_processing/fuse_results.sh \
         video_processing/extract_smpl.sh
```

---

# Pipeline steps:

### Step 1: Reconstruct human poses

Follow the steps in the [video_processing](video_processing/README.md) folder to reconstruct human poses. The scripts will by default store retargeting inputs in a `retarget/retarget_inputs` folder.

### Step 2: retarget motions

Next you will retarget the human motion to the unitree humanoid. You can save a video of the retargeted motion via the `--record_video` flag, which will save a visualization to `retarget/videos/`:

```
uv run python retarget/construct_motion.py \
    --smplx_file retarget/retarget_inputs/{motion}.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/{motion} \
    --robot_xml robots/retargeting/g1_27dof.xml
    --record_video 

```

If you are on a headless machine, and able to run xvfb-run, then use the command below. You can optionally set a ```--record_camera``` flag: 

```
xvfb-run -a uv run python retarget/construct_motion.py \
    --smplx_file retarget/retarget_inputs/{motion}.npz \
    --robot unitree_g1 \
    --save_path retarget/retarget_outputs/{motion} \
    --robot_xml robots/retargeting/g1_27dof.xml \
    --rate_limit
```

### step 3: convert to .npz format

First, create a new motion element titled after the motion that you just retargeted in [motion_lib.py](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py). We provide some examples to start with. Make sure to specify the ```probe_points``` at which the goal occurs. This should be a number between 0 and 1.

Next, convert the saved retargeted motion to the format expected by the training script:
```
MUJOCO_GL=egl uv run --directory tasknpoint_project python \
    src/tasknpoint_project/scripts/csv_to_npz.py \
    --input-file ../retarget/retarget_outputs/backhand_onehand.csv \
    --output-name backhand_onehand \
    --input-fps 30 \
    --output-fps 50 \
    --render False \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml

```

We provide different motion configurations in the [motion_train_configs](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_train_configs) folder, including tennis, soccer, and box pickup. By specifying which motion configuration to use, you will be able to print the relevant link coordinate that you will specify in [motion_lib.py](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py). The script will print something like:


```

============================================================
PROBE POINT POSITIONS
============================================================

  racket_contact (site)  phase=0.321  frame=56  t=1.120s
    world frame : x=0.4420  y=0.7021  z=1.0301
    init  frame : x=0.4415  y=0.7018  z=0.2426
    ori world   : roll=-0.3604  pitch=0.8088  yaw=1.1873  (rad)
    ori init    : roll=-0.3648  pitch=0.8064  yaw=1.1838  (rad)

============================================================
```

You will then set the orientation, position, and velocity goals accordingly. Please refer to [motion_lib.py](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py) for further detailed examples.

All motion configs are in the [motion_sets](tasknpoint_project/src/tasknpoint_project/motion_sets/) folder:
- [motion_lib.py](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_lib.py) — motion specs (positions, phases, weights)
- [motion_train_configs](tasknpoint_project/src/tasknpoint_project/motion_sets/motion_train_configs) train sets, e.g. which motions, registry prefixes, robot XML files to use

### step 4: launch training

To train TaskNPoint policies, first `cd` into the `tasknpoint_project/`. From here, you can launch training by running:

```
# single motion (pass registry name directly):
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --registry-name demalenk-california-institute-of-technology-caltech/csv_to_npz/backhand \
    --env.scene.num-envs 4096

# motion set — all motions (TOML drives registry + robot XML):
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/all_motions.toml \
    --env.scene.num-envs 4096

# motion set — tennis only:
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml \
    --env.scene.num-envs 4096

# motion set — tennis only expanded:
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only_expanded.toml \
    --env.scene.num-envs 4096

# motion set — kicks only:
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/kicks_only.toml \
    --env.scene.num-envs 4096

# box grab:
uv run tnp-train Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/box_grab.toml \
    --env.scene.num-envs 4096
```

To see which motions a set contains:
```
uv run tnp-motion-set src/tasknpoint_project/motion_sets/motion_train_configs/all_motions.toml --list
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
uv run tnp-play Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/i4tr5j7v \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only_fast.toml

```

To run with VISER!:

```
 uv run tnp-play-viser Mjlab-MultiTarget-Tracking-Flat-Unitree-G1     --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/0ulgzbgg     --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/box_grab.toml
```

To also visualize blobs:
```
uv run tnp-play-viser-motion-ranges Mjlab-MultiTarget-Tracking-Flat-Unitree-G1 \
    --wandb-run-path bwerner-california-institute-of-technology-caltech/mjlab/48cdla54 \
    --motion-config src/tasknpoint_project/motion_sets/motion_train_configs/tennis_only.toml
```

## License

This project is released under the [MIT License](LICENSE.md). Bundled
submodules (PromptHMR, GMR, unitree_sdk2_wrapper) retain their own licenses —
see each submodule for details.

## Citing TaskNPoint

```
@article{tasknpoint2026,
    author    = {Werner, Blake and Demler, Ilona and Perona, Pietro and Ames, Aaron D.},
    title     = {TaskNPoint: How to Teach Your Humanoid to Hit a Backhand in Minutes},
    journal   = {https://arxiv.org/pdf/2606.26215},
    year      = {2026},
}
```