#!/bin/bash
# Shared-DiT StreamWAM LIBERO training on a single node with 8 GPUs.
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
PY=${PY:-python}

RECIPE=${RECIPE:-examples/libero/configs/recipes/streamwam_libero_shared_dit_wan22_5b.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/deepspeed_zero2.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29627}
TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}

cd "$REPO_DIR"

OUTPUT_DIR=$($PY - "$RECIPE" $TRAIN_OVERRIDES <<'PY'
import sys
from examples.libero.presets import validate_preset
from streamwam.config import load_config
from streamwam.utils.config_cli import apply_overrides

recipe = sys.argv[1]
overrides = sys.argv[2:]
cfg = load_config(recipe)
cfg = apply_overrides(cfg, overrides)
validate_preset(cfg)
print(cfg.training.output_dir)
PY
)

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

cat <<EOF
[launch] mode: Shared-DiT StreamWAM LIBERO single-node 8-GPU
[launch] recipe: $RECIPE
[launch] accelerate_config: $ACCELERATE_CONFIG
[launch] num_processes: $NUM_PROCESSES
[launch] num_machines: $NUM_MACHINES
[launch] machine_rank: $MACHINE_RANK
[launch] cuda_visible_devices: $CUDA_VISIBLE_DEVICES
[launch] output_dir: $OUTPUT_DIR
[launch] main_process_port: $MAIN_PROCESS_PORT
[launch] overrides: ${TRAIN_OVERRIDES:-<none>}
[launch] log_file: $LOG_FILE
EOF

EXTRA_ARGS=()
if [ -n "$TRAIN_OVERRIDES" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(--override $TRAIN_OVERRIDES)
fi

exec $PY -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --num_machines "$NUM_MACHINES" \
  --machine_rank "$MACHINE_RANK" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --module streamwam.training.train \
  --config "$RECIPE" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
