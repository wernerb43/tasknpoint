#!/usr/bin/env bash
set -euo pipefail

INPUT_FILES=(
  "/home/blake/tasknpoint/retarget/retarget_outputs/backhand.csv"
  "/home/blake/tasknpoint/retarget/retarget_outputs/forehand.csv"
  "/home/blake/tasknpoint/retarget/retarget_outputs/stepback_backhand.csv"
  "/home/blake/tasknpoint/retarget/retarget_outputs/stepback_forehand.csv"
  "/home/blake/tasknpoint/retarget/retarget_outputs/two_step_backhand.csv"
  "/home/blake/tasknpoint/retarget/retarget_outputs/two_step_forehand.csv"
)

INPUT_FPS_VALUES=(10 20 40 50)

for input_file in "${INPUT_FILES[@]}"; do
  base=$(basename "$input_file" .csv)
  for fps in "${INPUT_FPS_VALUES[@]}"; do
    output_name="${base}_${fps}hz"
    echo "=== Running: $output_name ==="
    MUJOCO_GL=egl uv run python tasknpoint_project/src/tasknpoint_project/scripts/csv_to_npz.py \
      --input-file "$input_file" \
      --output-name "$output_name" \
      --input-fps "$fps" \
      --output-fps 50 \
      --render True
  done
done
