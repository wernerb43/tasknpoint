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

RESULTS_PKL="${1:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [extra args]}"
START_FRAME="${2:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [extra args]}"
END_FRAME="${3:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [extra args]}"
ACTION_TITLE="${4:?Usage: $0 <results_pkl_path> <start_frame> <end_frame> <action_title> [extra args]}"
shift 4

# ── Load config ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/config.env" ]]; then
    echo "ERROR: $SCRIPT_DIR/config.env not found."
    echo "Copy config.env.template to config.env and fill in your paths."
    exit 1
fi
set -o allexport
source "$SCRIPT_DIR/config.env"
set +o allexport

export PROMPTHMR_DATA_ROOT
export PROMPTHMR_PRETRAIN_ROOT

if [[ -n "${PHMR_CONDA_SH:-}" ]]; then
    source "$PHMR_CONDA_SH"
    conda activate "${PHMR_CONDA_ENV:-phmr_pt2.6}"
fi

# ── Run extraction ───────────────────────────────────────────────────────────
echo "Extracting '$ACTION_TITLE' (frames $START_FRAME–$END_FRAME) from:"
echo "  $RESULTS_PKL"
echo "Output folder: $RETARGET_OUTPUTS_ROOT"

python "$SCRIPT_DIR/extract_smpl_for_robo.py" \
    --results-pkl-path "$RESULTS_PKL" \
    --start-frame "$START_FRAME" \
    --end-frame   "$END_FRAME" \
    --action-title "$ACTION_TITLE" \
    --out-folder  "$RETARGET_OUTPUTS_ROOT" \
    --is-og-pkl-file \
    --no-run-viser \
    "$@"
