# Converting human demonstration videos into reference robot trajectories

Scripts for running the PromptHMR → robot retargeting pipeline on raw video files.

---

## Installation + environment setup

### 1. Install PromptHMR

Clone and install the PromptHMR model repo in a separate directory:

```bash
git clone https://github.com/yufu-wang/PromptHMR
cd PromptHMR
bash scripts/install.sh --pt_version=2.6 --world-video=true
conda activate phmr_pt2.6
```

Then add a `pyproject.toml` at the PromptHMR repo root so it can be pip-installed:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "prompthmr"
version = "0.0.1"
description = "PromptHMR: Promptable Human Mesh Recovery"
readme = "README.md"
requires-python = ">=3.9"

[tool.setuptools.packages.find]
where = ["."]
include = ["prompt_hmr*"]
```

Ensure an `__init__.py` exists:

```bash
[ -f prompt_hmr/__init__.py ] || touch prompt_hmr/__init__.py
```

Install and verify:

```bash
pip install -e . --config-settings editable_mode=compat
python -c "import prompt_hmr; print('imported:', prompt_hmr.__name__)"
```

You will also need SMPL-X body model files and PromptHMR model weights. Create an account at https://smpl-x.is.tue.mpg.de/ and download the models by running:

```
# SMPLX family models
bash scripts/fetch_smplx.sh

# Checkpoints and annotations
bash scripts/fetch_data.sh
```

### 2. Install video_processing as a local package

Now `cd` back into the tasknpoint repo. From the `video_processing/` directory run:

```bash
pip install -e .
python -c "from utils import estimate_num_frames, get_smplx_path; print('ok')"
```

### 3. Set up `config.env`

Copy the template and fill in your machine's paths:

```bash
cp config.env.template config.env
```

Edit `config.env`:

| Variable | Description |
|---|---|
| `PHMR_CONDA_SH` | Path to `conda.sh` (leave empty to skip activation) |
| `PHMR_CONDA_ENV` | Conda environment name (default: `phmr_pt2.6`) |
| `PROMPTHMR_DATA_ROOT` | Directory containing `body_models/smplx/` and `body_models/smpl/` |
| `PROMPTHMR_PRETRAIN_ROOT` | PromptHMR pretrained models root (contains `pretrain/`) |
| `RESULTS_ROOT` | Where PromptHMR saves `results_*.pkl` chunks |
| `RETARGET_OUTPUTS_ROOT` | Where extracted `.npz` retargeting files are saved |

---

### 4. Set up executables

Make the scripts executable once:

```bash
chmod +x run_prompthmr.sh fuse_results.sh extract_smpl.sh
```

## Process human demonstration videos

### 1. Extract raw human pose estimates:

Videos are broken down into 500-frame subsequences. For longer videos, this script will save multiple 500-frame video chunks with a 250 frame overlap window.

To run on an individual folder:
```bash
bash run_prompthmr_single.sh {path to video folder}
```
Outputs will be saved to a ```human_pose_outputs/{video_name}``` folder.

To run on a folder of videos:
```bash
bash run_prompthmr.sh {path to video folder}
```

Outputs will be saved to a ```human_pose_outputs/{video_folder_name}/{video_name}``` folder.

### 2. Fuse chunks into a single world4d pkl (optional) 

```bash
bash fuse_results.sh {video_folder} {video_title}
```

Optional flags:

```bash
--simple    # skips PID matching, just concatenate frames
--force     # overwrites existing world4d_fused.pkl
```

Saves `world4d_fused.pkl` alongside the chunk files.

### 3. Extract action demonstration

If your video contains multiple action demonstrations, or is longer than the specified action duration, we also provide the option of cropping the video to the start and end time of the action. 

We also provide an optional script for identifying the start and end frame of the action. To use it, open the fused pkl (or a raw chunk) in a viewer (e.g., `run_viser_on_fused_robo_result.py`) to identify the start and end frame of the action you want. 


Next, extract the action sub-sequence by running either:
```bash
# fused world4d file:
bash extract_smpl.sh \
    {path_to_phmr_results_file} \
    {start_idx} {end_idx} {action_title} \
    --fused

# raw prompthmr output e.g. results_{startidx}_{endidx}.pkl:
bash extract_smpl.sh \
    {path_to_phmr_results_file} \
    {start_idx} {end_idx} {action_title}
```

You can also optionally run a viser visualization to view the cropped action sequence:

```bash
bash extract_smpl.sh \
    {path_to_phmr_results_file} \
    {start_idx} {end_idx} {action_title} \
    --run-viser \
    --video-path {path to original input video}
```

This saves retargeted actions to `retarget_outputs/{action_title}.npz`.

---

## Output format

Each `.npz` retargeting file contains:

| Key | Shape | Description |
|---|---|---|
| `pose_body` | T × 63 | Body joint axis-angles (joints 1–21) |
| `root_orient` | T × 3 | Root orientation axis-angle |
| `trans` | T × 3 | Root translation (MuJoCo frame, normalized to first frame) |
| `betas` | 1 × 16 | SMPL-X shape coefficients |
| `gender` | str | `"neutral"` |
| `mocap_frame_rate` | int | `30` |

Coordinate frame: Z-up (MuJoCo), with X and Y flipped from the raw PromptHMR output via `R = diag([-1, -1, 1])`.
