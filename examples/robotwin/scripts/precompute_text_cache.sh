#!/bin/bash
# Precompute RoboTwin text-embedding cache across multiple GPUs.
#
# Splits the full instruction set (meta/tasks.jsonl) into NUM_SHARDS shards and
# runs one precompute process per GPU. Cache writes are atomic (tmp+rename) and
# keyed by prompt hash, so shards can safely share OUTPUT_DIR. Already-cached
# entries are skipped, so the job is resumable.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-}
PY=${PY:-python}
RECIPE=${RECIPE:-examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml}
BACKBONE=${BACKBONE:?set BACKBONE=/path/to/Wan2.2-TI2V-5B}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR=/path/to/text_embedding_cache}
NUM_SHARDS=${NUM_SHARDS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

cd "$REPO_DIR"
if [ -n "$CONDA_ENV" ] && [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi
mkdir -p "$OUTPUT_DIR/logs"

read -r -a GPU_ARR <<< "$GPUS"
if (( NUM_SHARDS < 1 || ${#GPU_ARR[@]} != NUM_SHARDS )); then
  echo "NUM_SHARDS ($NUM_SHARDS) must equal the number of GPUS (${#GPU_ARR[@]})" >&2
  exit 2
fi

PIDS=()
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  echo "[launch] shard $idx/$NUM_SHARDS on GPU $gpu -> $OUTPUT_DIR/logs/shard_${idx}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m starwam.tools.precompute_text_cache \
    --config "$RECIPE" \
    --pretrained-model-id "$BACKBONE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --dtype bf16 \
    --batch-size "$BATCH_SIZE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$idx" \
    > "$OUTPUT_DIR/logs/shard_${idx}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "[cache] shard $idx failed; inspect $OUTPUT_DIR/logs/shard_${idx}.log" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
echo "ALL SHARDS DONE"
