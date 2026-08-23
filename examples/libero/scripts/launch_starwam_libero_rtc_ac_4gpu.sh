#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BACKBONE_PATH="${BACKBONE_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/models/wan22_5b}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/evaluation/LIBERO}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
import platform
import sys
import torch
import triton
import starwam
import examples.libero.multigpu_rollout
print(
    "RTC-AC runtime: "
    f"python={sys.executable} "
    f"python_version={platform.python_version()} "
    f"torch={torch.__version__} "
    f"triton={triton.__version__} "
    f"cuda={torch.version.cuda}",
    flush=True,
)
'

"$PYTHON_BIN" examples/libero/multigpu_rollout.py \
  --gpus "$GPU_IDS" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/starwam_libero_rtc_ac_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_rtc_ac_step_005500.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_rtc_ac_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --sampling-method rtc_ac \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video \
  "$@"
