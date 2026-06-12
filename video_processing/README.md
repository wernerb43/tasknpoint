# video_processing

Scripts for running the PromptHMR → robot retargeting pipeline on raw video files.

## Pipeline overview

```
raw video
    │
    ▼  run_prompthmr.sh
results_{start}_{end}.pkl  (500-frame chunks in RESULTS_ROOT/)
    │
    ▼  fuse_results.sh
world4d_fused.pkl
    │
    ▼  extract_smpl.sh  (per action clip, with manually chosen start/end frames)
{action_title}.npz  →  RETARGET_OUTPUTS_ROOT/
```

The three shell scripts are thin wrappers that source `config.env` for paths and then call the corresponding Python scripts in the [caltech-tennis-benchmark](https://github.com/idemler/caltech-tennis-benchmark) repo.

---

## Installation

### 1. Install PromptHMR

Clone and install the PromptHMR model repo in a separate directory:

```bash
git clone https://github.com/yufu-wang/PromptHMR
cd PromptHMR
bash scripts/install.sh --pt_version=2.6
```

Then add a `pyproject.toml` at the repo root so it can be pip-installed:

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
pip install -e .
python -c "import prompt_hmr; print('imported:', prompt_hmr.__name__)"
```

You will also need SMPL-X body model files. Create an account at https://smpl-x.is.tue.mpg.de/ and download the models, then point `PROMPTHMR_DATA_ROOT` in `config.env` at the parent directory containing `body_models/smplx/`.

### 2. Install the caltech-tennis-benchmark repo

From the repo root:

```bash
pip install -e .
python -c "import prompthmr; print('ok')"
```

---

## Config setup

Copy the template and fill in your machine's paths:

```bash
cp config.env.template config.env
```

Edit `config.env`:

| Variable | Description |
|---|---|
| `PHMR_REPO` | Path to the caltech-tennis-benchmark repo root |
| `PHMR_CONDA_SH` | Path to `conda.sh` (leave empty to skip activation) |
| `PHMR_CONDA_ENV` | Conda environment name (default: `phmr_pt2.6`) |
| `PROMPTHMR_DATA_ROOT` | Directory containing `body_models/smplx/` and `body_models/smpl/` |
| `PROMPTHMR_PRETRAIN_ROOT` | PromptHMR pretrained models root (contains `pretrain/`) |
| `RESULTS_ROOT` | Where PromptHMR saves `results_*.pkl` chunks |
| `RETARGET_OUTPUTS_ROOT` | Where extracted `.npz` retargeting files are saved |

`config.env` is gitignored — `config.env.template` is the committed source of truth.

---

## Usage

Make the scripts executable once:

```bash
chmod +x run_prompthmr.sh fuse_results.sh extract_smpl.sh
```

### Step 1 — Run PromptHMR on a session folder

```bash
./run_prompthmr.sh /mnt/datasets/robodataset/05_23_2026_14_30_cast
```

Processes every `.mp4`/`.MOV`/`.mov` in the session folder (skips `overhead` files). Saves chunked `results_*.pkl` files to `$RESULTS_ROOT/<session_name>/<video_name>/`.

### Step 2 — Fuse chunks into a single world4d pkl

```bash
./fuse_results.sh 05_23_2026_14_30_cast 05_23_2026_14_31_02_000_court1_X_1_14_32_10_123
```

Optional flags forwarded to `fuse_results_robo.py`:

```bash
--simple    # skip PID matching, just concatenate frames
--force     # overwrite existing world4d_fused.pkl
```

Saves `world4d_fused.pkl` alongside the chunk files.

### Step 3 — Extract an action clip to .npz

Open the fused pkl (or a raw chunk) in a viewer (e.g., `run_viser_on_fused_robo_result.py` in the caltech repo) to identify the start and end frame of the action you want. Then:
```bash
./extract_smpl.sh \
    {path_to_phmr_results_file} \
    {start_idx} {end_idx} {action_title}
```


For example:
```bash
./extract_smpl.sh \
    /mnt/datasets/robo_results/prompthmr_results/05_23_2026_14_30_cast/VIDEO/results_0_499.pkl \
    85 157 one_hand_baseball_hit
```

With viser visualization and source video:

```bash
./extract_smpl.sh \
    /mnt/datasets/robo_results/prompthmr_results/05_23_2026_14_30_cast/VIDEO/results_0_499.pkl \
    85 157 one_hand_baseball_hit \
    --run-viser \
    --video-path /mnt/datasets/robodataset/05_23_2026_14_30_cast/VIDEO.MOV
```

Saves `$RETARGET_OUTPUTS_ROOT/one_hand_baseball_hit.npz`.

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
