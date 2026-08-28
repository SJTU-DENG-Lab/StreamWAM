#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKBONE_PATH="${BACKBONE_PATH:?set BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-${LIBERO_HOME:-}}"
: "${LIBERO_HOME_PATH:?set LIBERO_HOME_PATH=/path/to/LIBERO or LIBERO_HOME=/path/to/LIBERO}"

cd "$REPO_ROOT"

"$PYTHON_BIN" examples/libero/multigpu_rollout.py \
  --gpus "$GPU_IDS" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/streamingwam_libero_fastwam_joint_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_joint_step_040000.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_joint_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 10 \
  --sampling-method euler \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video \
  "$@"
