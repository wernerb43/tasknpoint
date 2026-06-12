#!/usr/bin/env bash
# Run PromptHMR on a single video file.
#
# Usage:
#   ./run_prompthmr_single.sh <video_path> [extra args]
#
# Extra args are forwarded to run_prompthmr_on_video.py, e.g.:
#   --static-camera
#   --start-frame 100 --end-frame 300
#   --chunk-size 250
#
# Example:
#   ./run_prompthmr_single.sh /mnt/datasets/robodataset/05_23_2026_14_30_cast/VIDEO.MOV
#
# Output goes to $RESULTS_ROOT/<parent_folder_name>/<video_stem>/
# matching the layout produced by run_prompthmr.sh.

set -euo pipefail

VIDEO_PATH="${1:?Usage: $0 <video_path> [extra args]}"
shift

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

# ── Activate conda env (optional) ───────────────────────────────────────────
if [[ -n "${PHMR_CONDA_SH:-}" ]]; then
    source "$PHMR_CONDA_SH"
    conda activate "${PHMR_CONDA_ENV:-phmr_pt2.6}"
fi

session_name="$(basename "$(dirname "$VIDEO_PATH")")"

echo "Video:      $VIDEO_PATH"
echo "Output dir: $RESULTS_ROOT/$session_name/$(basename "${VIDEO_PATH%.*}")/"
echo ""

# ── Change to PromptHMR repo root so relative paths resolve correctly ────────
cd "$(dirname "$PROMPTHMR_DATA_ROOT")"

# ── Run ──────────────────────────────────────────────────────────────────────
# run_prompthmr_on_video.py appends <session_name>/<video_stem> to output_dir itself.
python "$SCRIPT_DIR/run_prompthmr_on_video.py" \
    --input_video "$VIDEO_PATH" \
    --output_dir  "$RESULTS_ROOT" \
    "$@"
