#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BACKBONE_PATH="${BACKBONE_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/models/wan22_5b}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/evaluation/LIBERO}"

cd "$REPO_ROOT"

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamwam_libero_rtc_ac_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_rtc_ac_step_005500.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_rtc_ac_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --sampling-method rtc_ac \
  --task-id 0 \
  --num-trials 1 \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --fixed-seed \
  --device cuda:0 \
  --mujoco-gl osmesa \
  --save-video \
  "$@"
