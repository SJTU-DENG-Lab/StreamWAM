#!/usr/bin/env bash
set -euo pipefail

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --backbone-path /inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/models/wan22_5b \
  --stats-path checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --libero-home /inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/evaluation/LIBERO \
  --task-id 0 \
  --num-trials 1 \
  --num-steps-wait 30 \
  --replan-steps 10 \
  --num-inference-steps 10 \
  --fixed-seed \
  --device cuda:0 \
  --mujoco-gl osmesa \
  --save-video
