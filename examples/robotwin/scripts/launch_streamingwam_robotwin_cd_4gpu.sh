#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SIMULATOR_PYTHON="${SIMULATOR_PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=/path/to/fastwam-cd-checkpoint}"
STATS_PATH="${STATS_PATH:?set STATS_PATH=/path/to/action_stats.json}"
BACKBONE_PATH="${BACKBONE_PATH:?set BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B}"
ROBOTWIN_HOME="${ROBOTWIN_HOME:?set ROBOTWIN_HOME=/path/to/RoboTwin}"
TEXT_CACHE_PATH="${TEXT_CACHE_PATH:?set TEXT_CACHE_PATH=/path/to/text-cache}"

"$PYTHON_BIN" examples/robotwin/multigpu_rollout.py \
  --config examples/robotwin/configs/recipes/streamingwam_robotwin_mot_wan22_5b.yaml \
  --checkpoint "$CHECKPOINT" \
  --stats-path "$STATS_PATH" \
  --backbone-path "$BACKBONE_PATH" \
  --text-cache-path "$TEXT_CACHE_PATH" \
  --robotwin-home "$ROBOTWIN_HOME" \
  --inference-python "$PYTHON_BIN" \
  --simulator-python "$SIMULATOR_PYTHON" \
  --inference-mode cd \
  --gpu-ids "$GPU_IDS" \
  "$@"
