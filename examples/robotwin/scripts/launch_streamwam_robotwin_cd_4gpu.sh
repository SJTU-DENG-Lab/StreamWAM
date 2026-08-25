#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python}"
SIMULATOR_PYTHON="${SIMULATOR_PYTHON:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/envs/motus/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

"$PYTHON_BIN" examples/robotwin/multigpu_rollout.py \
  --config examples/robotwin/configs/recipes/streamwam_robotwin_mot_wan22_5b.yaml \
  --checkpoint /inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/outputs/starwam_rtc/starwam_cd_full_eval_20260821_161715/cd/checkpoint-2000 \
  --stats-path /inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/starwam/pretrained_models/starwam-robotwin/action_stats.json \
  --backbone-path "${BACKBONE_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/models/wan22_5b}" \
  --robotwin-home "${ROBOTWIN_HOME:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/fastwam-foresight/third_party/RoboTwin}" \
  --inference-python "$PYTHON_BIN" \
  --simulator-python "$SIMULATOR_PYTHON" \
  --inference-mode cd \
  --gpu-ids "$GPU_IDS" \
  "$@"
