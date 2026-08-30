#!/usr/bin/env bash
# Canonical RoboTwin benchmark: fixed trials, R16, per-task success/Chunk/Total Time.
# CHECKPOINT_FORMAT and MODE select released family defaults; explicit
# NUM_INFERENCE_STEPS/ACTION_NUM_INFERENCE_STEPS always take precedence.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"

MODE=${MODE:?set MODE=baseline|cd|ac-stream}
CKPT=${CKPT:?set CKPT}
CHECKPOINT_FORMAT=${CHECKPOINT_FORMAT:?set CHECKPOINT_FORMAT=fastwam|starwam|streamingwam}
CONFIG=${CONFIG:?set CONFIG to the RoboTwin recipe}
STATS_PATH=${STATS_PATH:?set STATS_PATH}
BACKBONE_PATH=${BACKBONE_PATH:?set BACKBONE_PATH}
ROBOTWIN_HOME=${ROBOTWIN_HOME:?set ROBOTWIN_HOME}
INFERENCE_PYTHON=${INFERENCE_PYTHON:?set INFERENCE_PYTHON}
SIMULATOR_PYTHON=${SIMULATOR_PYTHON:?set SIMULATOR_PYTHON}

GPU_TOPOLOGY=${GPU_TOPOLOGY:-colocated}
GPU_IDS=${GPU_IDS:-0,1,2,3}
NUM_TRIALS=${NUM_TRIALS:-100}
REPLAN_STEPS=${REPLAN_STEPS:-16}
MODEL_SEED=${MODEL_SEED:-${SEED:-42}}
EPISODE_SEED=${EPISODE_SEED:-$MODEL_SEED}
TEXT_CACHE_PATH=${TEXT_CACHE_PATH:-/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/starwam/cache/text_embeds_cache}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/robotwin_${MODE}_${CHECKPOINT_FORMAT}_r${REPLAN_STEPS}_${NUM_TRIALS}trials_$(date +%Y%m%d_%H%M%S)}

command=(
  "$INFERENCE_PYTHON" examples/robotwin/multigpu_rollout.py
  --config "$CONFIG"
  --checkpoint "$CKPT"
  --checkpoint-format "$CHECKPOINT_FORMAT"
  --stats-path "$STATS_PATH"
  --backbone-path "$BACKBONE_PATH"
  --text-cache-path "$TEXT_CACHE_PATH"
  --robotwin-home "$ROBOTWIN_HOME"
  --inference-python "$INFERENCE_PYTHON"
  --simulator-python "$SIMULATOR_PYTHON"
  --inference-mode "$MODE"
  --num-trials "$NUM_TRIALS"
  --replan-steps "$REPLAN_STEPS"
  --seed "$MODEL_SEED"
  --episode-seed "$EPISODE_SEED"
  --output-dir "$OUTPUT_DIR"
)

case "$GPU_TOPOLOGY" in
  colocated)
    # One inference server and one isolated simulator worker share each GPU.
    # D8 action execution suppresses redundant SAPIEN renders while inference
    # is in flight, so the four-GPU path keeps four jobs active in parallel.
    command+=(--gpu-ids "$GPU_IDS")
    ;;
  split)
    INFERENCE_GPU_IDS=${INFERENCE_GPU_IDS:-0,1}
    SIMULATOR_GPU_IDS=${SIMULATOR_GPU_IDS:-2,3}
    command+=(
      --inference-gpu-ids "$INFERENCE_GPU_IDS"
      --simulator-gpu-ids "$SIMULATOR_GPU_IDS"
      --require-gpu-isolation
    )
    ;;
  *)
    echo "GPU_TOPOLOGY must be colocated or split, got: $GPU_TOPOLOGY" >&2
    exit 2
    ;;
esac

if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  command+=(--num-inference-steps "$NUM_INFERENCE_STEPS")
fi
if [[ -n "${ACTION_NUM_INFERENCE_STEPS:-}" ]]; then
  command+=(--action-num-inference-steps "$ACTION_NUM_INFERENCE_STEPS")
fi
if [[ "$MODE" == "ac-stream" ]]; then
  if [[ "${AC_STREAM_BACKEND:-accelerated}" == "eager" ]]; then
    command+=(--ac-stream-eager)
  else
    command+=(--ac-stream-accelerated)
  fi
fi

exec "${command[@]}" "$@"
