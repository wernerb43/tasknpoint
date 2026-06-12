#!/usr/bin/env bash
# Step 2: Fuse chunked PromptHMR results into a single world4d_fused.pkl.
#
# Usage:
#   ./fuse_results.sh <session_folder_name> <video_name> [extra args]
#
# Extra args are forwarded to fuse_results_robo.py, e.g.:
#   --simple     use simple fusion without PID matching
#   --force      overwrite existing world4d_fused.pkl
#
# Example:
#   ./fuse_results.sh 05_23_2026_14_30_cast 05_23_2026_14_31_02_000_court1_X_1_14_32_10_123
#   ./fuse_results.sh 05_23_2026_14_30_cast 05_23_2026_14_31_02_000_court1_X_1_14_32_10_123 --simple --force

set -euo pipefail

SESSION="${1:?Usage: $0 <session_folder_name> <video_name> [extra args]}"
VIDEO_NAME="${2:?Usage: $0 <session_folder_name> <video_name> [extra args]}"
shift 2  # remaining args forwarded to the Python script

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

# ── Run fusion ───────────────────────────────────────────────────────────────
echo "Fusing results for: $SESSION / $VIDEO_NAME"

python "$PHMR_REPO/prompthmr/robo/fuse_results_robo.py" \
    --folder_name "$RESULTS_ROOT/$SESSION" \
    --video_name  "$VIDEO_NAME" \
    "$@"
