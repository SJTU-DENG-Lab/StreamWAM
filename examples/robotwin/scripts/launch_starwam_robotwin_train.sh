#!/bin/bash
# Recipe-driven StarWAM RoboTwin 2.0 training launcher.
#
# RECIPE may point to MoT or Shared-DiT. Keep the effective global batch at
# 1024; for example:
#   8 GPUs:  batch_size=8,  gradient_accumulation_steps=16
#   16 GPUs: batch_size=8,  gradient_accumulation_steps=8
#   16 GPUs: batch_size=16, gradient_accumulation_steps=4
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
CONDA_SH=${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-starwam}
PY=${PY:-python}

RECIPE=${RECIPE:-examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-configs/accelerate/deepspeed_zero2.yaml}
NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_MACHINES=${NUM_MACHINES:-1}
MACHINE_RANK=${MACHINE_RANK:-0}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29617}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-}   # required for multi-node (rank0 reachable IP)
TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}

if (( NUM_MACHINES > 1 )) && [ -z "$MAIN_PROCESS_IP" ]; then
  echo "MAIN_PROCESS_IP is required when NUM_MACHINES > 1" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}

cd "$REPO_DIR"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

OVERRIDE_VALUES=()
if [ -n "$TRAIN_OVERRIDES" ]; then
  # Values are whitespace-delimited key=value entries, matching --override.
  # shellcheck disable=SC2206
  OVERRIDE_VALUES=($TRAIN_OVERRIDES)
fi

OUTPUT_DIR=$("$PY" - "$RECIPE" "${OVERRIDE_VALUES[@]}" <<'PY'
import sys
from starwam.config import load_config
from starwam.utils.config_cli import apply_overrides

recipe = sys.argv[1]
overrides = sys.argv[2:]
cfg = load_config(recipe)
cfg = apply_overrides(cfg, overrides)
print(cfg.training.output_dir)
PY
)

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_rank${MACHINE_RANK}_$(date +%Y%m%d_%H%M%S).log"

cat <<EOF
[launch] mode: StarWAM RoboTwin recipe-driven training
[launch] recipe: $RECIPE
[launch] accelerate_config: $ACCELERATE_CONFIG
[launch] num_processes: $NUM_PROCESSES
[launch] num_machines: $NUM_MACHINES
[launch] machine_rank: $MACHINE_RANK
[launch] cuda_visible_devices: $CUDA_VISIBLE_DEVICES
[launch] output_dir: $OUTPUT_DIR
[launch] main_process_ip: ${MAIN_PROCESS_IP:-<local>}
[launch] main_process_port: $MAIN_PROCESS_PORT
[launch] overrides: ${TRAIN_OVERRIDES:-<none>}
[launch] log_file: $LOG_FILE
EOF

EXTRA_ARGS=()
if (( ${#OVERRIDE_VALUES[@]} > 0 )); then
  EXTRA_ARGS=(--override "${OVERRIDE_VALUES[@]}")
fi

LAUNCH_ARGS=(
  --config_file "$ACCELERATE_CONFIG"
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NUM_MACHINES"
  --machine_rank "$MACHINE_RANK"
  --main_process_port "$MAIN_PROCESS_PORT"
)
if [ -n "$MAIN_PROCESS_IP" ]; then
  LAUNCH_ARGS+=(--main_process_ip "$MAIN_PROCESS_IP")
fi

exec "$PY" -m accelerate.commands.launch \
  "${LAUNCH_ARGS[@]}" \
  --module starwam.training.train \
  --config "$RECIPE" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
