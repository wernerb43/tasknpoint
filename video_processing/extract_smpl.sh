#!/usr/bin/env bash
# Step 3: Extract an action clip from PromptHMR results into a retargeting .npz.
#
# Usage:
#   ./extract_smpl.sh <results_pkl_path> <start_frame> <end_frame> <action_title> [extra args]
#
# The script always passes:
#   --is-og-pkl-file        (input is a raw PromptHMR results.pkl, not a fused world4d)
#   --out-folder <RETARGET_OUTPUTS_ROOT>
#   --no-run-viser          (viser off by default; pass --run-viser to override)
#
# Extra args are forwarded to extract_smpl_for_robo.py, e.g.:
#   --run-viser
#   --video-path /path/to/video.MOV
#   --save-stick-video
#
# Example (raw pkl, no viser):
#   ./extract_smpl.sh \
#       /mnt/datasets/robo_results/prompthmr_results/05_23_2026_14_30_cast/VIDEO/results_0_499.pkl \
#       85 157 one_hand_baseball_hit
#
# Example (raw pkl, with viser and source video):
#   ./extract_smpl.sh \
#       /mnt/datasets/robo_results/prompthmr_results/05_23_2026_14_30_cast/VIDEO/results_0_499.pkl \
#       85 157 one_hand_baseball_hit \
#       --run-viser \
#       --video-path /mnt/datasets/robodataset/05_23_2026_14_30_cast/VIDEO.MOV

set -euo pipefail

RESULTS_PKL="${1:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [--fused] [extra args]}"
START_FRAME="${2:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [--fused] [extra args]}"
END_FRAME="${3:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [--fused] [extra args]}"
ACTION_TITLE="${4:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [--fused] [extra args]}"
shift 4

# Resolve a possibly-relative path to absolute, before we cd elsewhere below.
abspath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

RESULTS_PKL="$(abspath "$RESULTS_PKL")"

# ── Parse --fused flag ────────────────────────────────────────────────────────
# By default the input is treated as a raw PromptHMR results pkl (--is-og-pkl-file).
# Pass --fused to treat it as a fused world4d pkl instead.
# Also resolve --video-path to an absolute path (relative paths break after cd).
PKL_TYPE_FLAG="--is-og-pkl-file"
REMAINING_ARGS=()
prev=""
for arg in "$@"; do
    if [[ "$arg" == "--fused" ]]; then
        PKL_TYPE_FLAG="--no-is-og-pkl-file"
    elif [[ "$prev" == "--video-path" ]]; then
        REMAINING_ARGS+=("$(abspath "$arg")")
    elif [[ "$arg" == --video-path=* ]]; then
        REMAINING_ARGS+=("--video-path=$(abspath "${arg#--video-path=}")")
    else
        REMAINING_ARGS+=("$arg")
    fi
    prev="$arg"
done
set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"

# ── Load config ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
set -o allexport
source "$REPO_ROOT/config.env"
set +o allexport

export PROMPTHMR_DATA_ROOT
export PROMPTHMR_PRETRAIN_ROOT

if [[ -n "${PHMR_CONDA_SH:-}" ]]; then
    source "$PHMR_CONDA_SH"
    conda activate "${PHMR_CONDA_ENV:-phmr_pt2.6}"
fi

# ── Change to PromptHMR repo root (same reason as run_prompthmr.sh) ──────────
cd "$(dirname "$PROMPTHMR_DATA_ROOT")"

# ── Run extraction ───────────────────────────────────────────────────────────
# Outputs go under retarget/retarget_inputs.
OUTPUT_FOLDER="$RETARGET_OUTPUTS_ROOT/retarget_inputs"

echo "Extracting '$ACTION_TITLE' (frames $START_FRAME–$END_FRAME) from:"
echo "  $RESULTS_PKL"
echo "Output folder: $OUTPUT_FOLDER"

python "$SCRIPT_DIR/extract_smpl_for_robo.py" \
    --results-pkl-path "$RESULTS_PKL" \
    --start-frame "$START_FRAME" \
    --end-frame   "$END_FRAME" \
    --action-title "$ACTION_TITLE" \
    --out-folder  "$OUTPUT_FOLDER" \
    "$PKL_TYPE_FLAG" \
    --no-run-viser \
    "$@"
