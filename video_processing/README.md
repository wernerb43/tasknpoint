# Converting human demonstration videos into reference robot trajectories

Scripts for running the PromptHMR → robot retargeting pipeline on raw video files.

---

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
git submodule update --init --recursive
```

### 2. Set up the tasknpoint environment

```bash
uv sync   # from repo root
```

### 3. Set up the phmr environment

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

Then register PromptHMR and video_processing as importable packages:

```bash
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

### 5. Make scripts executable (once)

```bash
chmod +x video_processing/run_prompthmr.sh \
         video_processing/run_prompthmr_single.sh \
         video_processing/fuse_results.sh \
         video_processing/extract_smpl.sh
```

No `config.env` setup is needed — all paths are resolved automatically from the repo structure.

---

## Process human demonstration videos

All scripts activate the `phmr_pt2.6` conda env automatically via `config.env`.

### 1. Extract raw human pose estimates

Videos are broken into 500-frame chunks with a 250-frame overlap window.

```bash
# Single video
bash video_processing/run_prompthmr_single.sh <path/to/video.MOV>

# All videos in a session folder
bash video_processing/run_prompthmr.sh <path/to/session_folder>
```

Outputs are saved to `video_processing/human_pose_outputs/`.

### 2. Fuse chunks into a single world4d pkl (optional)

```bash
bash video_processing/fuse_results.sh <session_folder_name> <video_name> [--simple] [--force]
```

Saves `world4d_fused.pkl` alongside the chunk files.

### 3. Extract an action demonstration

Open the fused pkl in a viewer to identify start/end frames, then:

```bash
# Raw PromptHMR output:
bash video_processing/extract_smpl.sh \
    <path_to_results_pkl> <start_frame> <end_frame> <action_title>

# Fused world4d:
bash video_processing/extract_smpl.sh \
    <path_to_world4d_fused_pkl> <start_frame> <end_frame> <action_title> --fused

# With viser visualization:
bash video_processing/extract_smpl.sh \
    <path_to_results_pkl> <start_frame> <end_frame> <action_title> \
    --run-viser --video-path <path/to/original_video.MOV>
```

Saves retargeted actions to `retarget/<action_title>.npz`.

---

## Output format

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `pose_body` | T × 63 | Body joint axis-angles (joints 1–21) |
| `root_orient` | T × 3 | Root orientation axis-angle |
| `trans` | T × 3 | Root translation (MuJoCo frame, normalized to first frame) |
| `betas` | 1 × 16 | SMPL-X shape coefficients |
| `gender` | str | `"neutral"` |
| `mocap_frame_rate` | int | `30` |

Coordinate frame: Z-up (MuJoCo), with X and Y flipped from raw PromptHMR output via `R = diag([-1, -1, 1])`.
