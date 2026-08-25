#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKBONE_PATH="${BACKBONE_PATH:?set BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH=/path/to/ac_stream_checkpoint.pt}"
STATS_PATH="${STATS_PATH:?set STATS_PATH=/path/to/dataset_stats.json}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-${LIBERO_HOME:-}}"
: "${LIBERO_HOME_PATH:?set LIBERO_HOME_PATH=/path/to/LIBERO or LIBERO_HOME=/path/to/LIBERO}"

cd "$REPO_ROOT"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN is not available on PATH: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -c '
import torch
import triton
import streamwam
import examples.libero.multigpu_rollout
'

"$PYTHON_BIN" examples/libero/multigpu_rollout.py \
  --gpus "$GPU_IDS" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/streamwam_libero_ac_stream_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint "$CHECKPOINT_PATH" \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path "$STATS_PATH" \
  --libero-home "$LIBERO_HOME_PATH" \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --sampling-method ac-stream \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video \
  "$@"
