#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BACKBONE_PATH="${BACKBONE_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/models/wan22_5b}"
LIBERO_HOME_PATH="${LIBERO_HOME_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/evaluation/LIBERO}"

cd "$REPO_ROOT"

python examples/libero/multigpu_rollout.py \
  --gpus "$GPU_IDS" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/streamwam_libero_joint_cd_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_joint_cd_step_003400.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_joint_cd_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --sampling-method consistency \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video \
  "$@"
