#!/bin/bash
# StarWAM RoboTwin 2.0 rollout driver (SAPIEN client side).
#
# Aligned to Fast-WAM: all 50 tasks, both demo_clean and demo_randomized,
# instruction_type=unseen, 100 episodes/task-setting (RoboTwin test_num=100).
#
# Runs inside the RoboTwin (SAPIEN) environment. The heavy StarWAM inference
# runs in separate policy servers (launch_starwam_robotwin_policy_server.sh);
# this driver only renders sims and forwards observations over a socket.
#
# It launches NWORKER RoboTwin worker processes. Worker i renders on
# CLIENT_GPUS[i % #gpus] and connects to server port SERVER_PORT_BASE + (i % NSERVERS).
# So a single-GPU client can still drive an 8-server backend (set CLIENT_GPUS to
# one id, NSERVERS/NWORKER to 8), and a co-located 8-GPU box can map 1:1.
#
# All machine-specific paths come from environment variables (no hardcoding):
#   ROBOTWIN=/path/to/RoboTwin \
#   PY=/path/to/robotwin-env/bin/python \
#   VK_ICD_FILENAMES=/path/to/nvidia_icd.json \        # if SAPIEN needs it
#   SERVER_HOST=<server ip> SERVER_PORT_BASE=8765 NSERVERS=8 \
#   CLIENT_GPUS="0" NWORKER=8 CKPT_TAG=checkpoint_tag \
#   bash examples/robotwin/scripts/launch_starwam_robotwin_rollout.sh
#
# The client protocol is model-family agnostic; MoT vs Shared-DiT is selected by
# the recipe/checkpoint used to start the inference servers.
#
# NOTE: There is NO wall-clock timeout around a task-setting. RoboTwin enforces
# a per-episode step limit itself, so a group of 100 episodes terminates on its
# own; wrapping it in `timeout` truncates episodes and biases the success rate.
set -euo pipefail

ROBOTWIN="${ROBOTWIN:?set ROBOTWIN=/path/to/RoboTwin}"
PY="${PY:-python}"
POLICY_NAME="${POLICY_NAME:-starwam_client}"
POLICY_CONFIG="${POLICY_CONFIG:-policy/$POLICY_NAME/deploy_policy_client.yml}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-8765}"
NSERVERS="${NSERVERS:-8}"
NWORKER="${NWORKER:-8}"
CLIENT_GPUS="${CLIENT_GPUS:-0 1 2 3 4 5 6 7}"   # sim GPUs, cycled across workers
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
SEED="${SEED:-0}"
CKPT_TAG="${CKPT_TAG:-starwam}"
LOGDIR="${LOGDIR:-$ROBOTWIN/rollout_logs/$CKPT_TAG}"
mkdir -p "$LOGDIR"
read -r -a GPU_ARR <<< "$CLIENT_GPUS"

if (( NWORKER < 1 || NSERVERS < 1 || ${#GPU_ARR[@]} < 1 )); then
  echo "NWORKER, NSERVERS, and CLIENT_GPUS must each specify at least one worker/server/GPU" >&2
  exit 2
fi

# 50 tasks, order from Fast-WAM _eval_step_limit.yml
TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic lift_pot move_can_pot move_playingcard_away move_stapler_pad hanging_mug open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_burger_fries place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand move_pillbottle_pad place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)
CONFIGS=(demo_clean demo_randomized)

# Build the (task, config) job list.
JOBS=()
for t in "${TASKS[@]}"; do for c in "${CONFIGS[@]}"; do JOBS+=("$t:$c"); done; done

run_worker() {
  local wid=$1
  local gpu="${GPU_ARR[$(( wid % ${#GPU_ARR[@]} ))]}"
  local port=$(( SERVER_PORT_BASE + (wid % NSERVERS) ))
  local i job task cfg
  for i in "${!JOBS[@]}"; do
    if (( i % NWORKER == wid )); then
      job="${JOBS[$i]}"; task="${job%%:*}"; cfg="${job##*:}"
      echo "[w$wid gpu$gpu p$port] === $task $cfg ==="
      CUDA_VISIBLE_DEVICES=$gpu "$PY" script/eval_policy.py \
        --config "$POLICY_CONFIG" \
        --overrides \
          policy_name "$POLICY_NAME" \
          task_name "$task" \
          task_config "$cfg" \
          instruction_type "$INSTRUCTION_TYPE" \
          ckpt_setting "$CKPT_TAG" \
          seed "$SEED" \
          policy_mode client \
          server_host "$SERVER_HOST" \
          server_port "$port"
    fi
  done
  echo "[w$wid] DONE"
}

cd "$ROBOTWIN" || exit 1
PIDS=()
for (( w=0; w<NWORKER; w++ )); do
  run_worker "$w" > "$LOGDIR/worker_${w}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[rollout] worker $i failed; inspect $LOGDIR/worker_${i}.log" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
echo "ALL ROLLOUTS DONE"
