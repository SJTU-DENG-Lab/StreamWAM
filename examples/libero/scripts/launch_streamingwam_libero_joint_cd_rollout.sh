#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BACKBONE_PATH="${BACKBONE_PATH:?set BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-${LIBERO_HOME:-}}"
: "${LIBERO_HOME_PATH:?set LIBERO_HOME_PATH=/path/to/LIBERO or LIBERO_HOME=/path/to/LIBERO}"

cd "$REPO_ROOT"

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_joint_cd_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_joint_cd_step_003400.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_joint_cd_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --sampling-method consistency \
  --task-id 0 \
  --num-trials 1 \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --fixed-seed \
  --device cuda:0 \
  --mujoco-gl osmesa \
  --save-video
