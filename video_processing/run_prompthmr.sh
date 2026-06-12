#!/usr/bin/env bash
# Step 1: Run PromptHMR on all videos in a session folder.
#
# Usage:
#   ./run_prompthmr.sh <session_folder_path>
#
# Example:
#   ./run_prompthmr.sh /mnt/datasets/robodataset/05_23_2026_14_30_cast

set -euo pipefail

SESSION_FOLDER="${1:?Usage: $0 <session_folder_path>}"

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

# Export model paths so prompthmr/config.py picks them up via env vars
export PROMPTHMR_DATA_ROOT
export PROMPTHMR_PRETRAIN_ROOT

# ── Activate conda env (optional) ───────────────────────────────────────────
if [[ -n "${PHMR_CONDA_SH:-}" ]]; then
    source "$PHMR_CONDA_SH"
    conda activate "${PHMR_CONDA_ENV:-phmr_pt2.6}"
fi

# ── Derive session name from folder path ────────────────────────────────────
session_name="$(basename "$SESSION_FOLDER")"
output_dir="$RESULTS_ROOT/$session_name"

echo "Session:    $SESSION_FOLDER"
echo "Output dir: $output_dir"
echo ""

# ── Change to PromptHMR repo root ────────────────────────────────────────────
# PromptHMR code uses relative paths like "data/body_models/..." that must be
# resolved from the repo root. PROMPTHMR_DATA_ROOT is the data/ subdir, so its
# parent is the repo root. Python adds the script's directory to sys.path so
# imports from video_processing/ still work with an absolute script path.
PHMR_ROOT="$(dirname "$PROMPTHMR_DATA_ROOT")"
cd "$PHMR_ROOT"

# ── Process each video in the session folder ────────────────────────────────
for video_path in "$SESSION_FOLDER"/*.{mp4,MOV,mov}; do
    [[ -e "$video_path" ]] || continue
    [[ "$video_path" == *".DS_Store"* ]] && continue
    [[ "$video_path" == *"overhead"* ]] && continue

    video_id="$(basename "$video_path")"
    echo "Running PromptHMR on: $video_id"

    python "$SCRIPT_DIR/run_prompthmr_on_video.py" \
        --input_video "$video_path" \
        --output_dir "$output_dir"
done
