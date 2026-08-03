# RoboTwin 2.0 Examples

RoboTwin 2.0 workflow for StarWAM: setup, training, and rollout against the official RoboTwin harness. See root [README.md](../../README.md) and [../libero/LIBERO.md](../libero/LIBERO.md).

SAPIEN needs an NVIDIA Vulkan stack; StarWAM inference needs Torch/CUDA. Pick one of two modes depending on your environment:

- **local** (`local_policy.py`, `policy_mode: local`): when the Vulkan stack and Torch/StarWAM are available in the **same env** — inference runs in-process, simplest.
- **client/server** (`client_policy.py` + `policy_server.py`, `policy_mode: client`): when the two stacks **can't coexist** (e.g. the inference box has no Vulkan, or the render box has no suitable Torch) — the server runs inference in the Torch env, the client renders in the SAPIEN env, communicating over a socket.

## 1. Layout

```text
examples/robotwin/
  deploy_policy.py          # RoboTwin entry point; dispatches by policy_mode (local|client)
  local_policy.py           # in-process adapter (SAPIEN + Torch/StarWAM in one env)
  client_policy.py          # socket client adapter (SAPIEN-only env)
  policy_server.py          # StarWAM inference server (Torch/StarWAM env)
  deploy_policy.yml         # config for local mode
  deploy_policy_client.yml  # config for client mode
  configs/recipes/          # training recipes
  scripts/                  # launch scripts (train / server / rollout / text cache)
```

RoboTwin imports a policy by symlinking this directory into its `policy/` tree:

```bash
# local (single env):
ln -s /ABS/PATH/starWAM/examples/robotwin  RoboTwin/policy/starwam_policy
# client (split env):
ln -s /ABS/PATH/starWAM/examples/robotwin  RoboTwin/policy/starwam_client
```

## 2. Environment

### 2.1 Training / server (Torch/StarWAM)

```bash
conda create -n starwam python=3.11 -y
conda activate starwam
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### 2.2 Client (RoboTwin / SAPIEN)

Follow the official RoboTwin 2.0 install (SAPIEN, curobo, mplib, assets). The
StarWAM client adapter itself needs only `numpy` + the standard library. On
headless servers SAPIEN needs a valid NVIDIA Vulkan ICD; if `vulkaninfo` does
not list the GPU, point SAPIEN at one before running:

```bash
export VK_ICD_FILENAMES=/path/to/nvidia_icd.json
```

## 3. Recipes and results

| Recipe | Model family | Backbone | Initialization |
| --- | --- | --- | --- |
| `starwam_robotwin_mot_wan22_5b.yaml` | `mot_wam` | Wan2.2-TI2V-5B | Preprocessed ActionDiT payload required. |
| `starwam_robotwin_shared_dit_wan22_5b.yaml` | `shared_dit_wam` | Wan2.2-TI2V-5B | Backbone initialization only; no ActionDiT payload. |

Both results use 100 episodes per task-setting over all 50 tasks with
`instruction_type=unseen` and `replan_steps=24` (10,000 episodes per model).

| Model | Checkpoint | Inference steps | Clean | Randomized | Overall (micro) |
| --- | --- | ---: | ---: | ---: | ---: |
| MoT | checkpoint-27500 | 4 | 89.28% | 89.68% | **89.48% (8948/10000)** |
| Shared-DiT | checkpoint-45000 | 16 video / 16 action | 92.56% | 92.58% | **92.57% (9257/10000)** |

The complete comparison is in [Section 9](#9-per-task-results).

Key RoboTwin settings shared by both recipes:

- `framework.action_dim: 14`, `framework.proprio_dim: 14`: native dual-arm qpos.
- `framework.chunk_size: 32`.
- Three cameras with `concat_multi_camera: robotwin`: head 256x320 above left/right wrist 128x160 views, producing 384x320.
- `data.num_frames: 33`, `action_freq_ratio: 4`: 32 action steps and 9 sampled video frames.
- z-score normalization for actions and states.

## 4. Paths you must set

Release recipes intentionally keep placeholder paths. Leave the YAML reusable and provide machine paths through `--override` or launcher environment variables.

| Field | Required for | What to set |
| --- | --- | --- |
| `backbone.pretrained_model_id` | all | Local Wan2.2-TI2V-5B directory. |
| `framework.action_expert_init_from` | MoT training/eval only | Output of Section 6.1 (`preprocess_action_dit_init`); omit for Shared-DiT. |
| `training.output_dir` | training | Run output directory. |
| `data.dataset_dirs` | training | LeRobot-format RoboTwin 2.0 dataset dir(s). |
| `data.text_embedding_cache_dir` | training + eval | Text embedding cache dir. |
| `data.action_stats_path` | training + eval | z-score action stats JSON (created if missing). |
| `data.state_stats_path` | training + eval | State stats JSON (can share the action stats file). |

## 5. Data

RoboTwin 2.0 LeRobot v2.1 dataset (Fast-WAM preprocessed): `yuanty/robotwin2.0-fastwam`.
14-D dual-arm qpos state/action (left 6 joints + gripper, right 6 joints + gripper);
cameras `cam_high` / `cam_left_wrist` / `cam_right_wrist`.

Download the RoboTwin harness assets with the official script:

```bash
cd RoboTwin && bash script/_download_assets.sh
```

## 6. Preprocessing

### 6.1 ActionDiT initialization

```bash
python -m starwam.tools.preprocess_action_dit_init \
  --config examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml \
  --source-backbone wan22 \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output /path/to/preprocessed/starwam_action_dit_init_wan22.pt \
  --device cuda:0 --dtype bfloat16
```

Keep the released recipe unchanged and pass the generated path with `--override framework.action_expert_init_from=...`.

### 6.2 Text embedding cache (optional but recommended)

RoboTwin has ~921k unique frame-level instructions; precompute the T5 cache
across GPUs so training/eval don't re-encode on the fly:

```bash
RECIPE=examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml \
BACKBONE=/path/to/Wan2.2-TI2V-5B \
OUTPUT_DIR=/path/to/output/robotwin_run/text_embedding_cache \
bash examples/robotwin/scripts/precompute_text_cache.sh
```

## 7. Training

The generic launcher accepts either recipe through `RECIPE`. Both recipes use
lr 1e-4 and global batch 1024. MoT trains for 5 epochs (~29.7k optimizer steps);
Shared-DiT trains for 8 epochs (~47k steps). Run Section 6.1 only for MoT.

### 7.1 MoT, single node (8 GPUs)

```bash
cd /path/to/starWAM
export REPO_DIR=/path/to/starWAM
export CONDA_ENV=starwam
export RECIPE=examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml
export TRAIN_OVERRIDES='data.dataset_dirs=["/path/to/robotwin2.0"] backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B framework.action_expert_init_from=/path/to/preprocessed/starwam_action_dit_init_wan22.pt training.output_dir=/path/to/output/starwam_robotwin_mot_wan22_5b data.text_embedding_cache_dir=/path/to/output/starwam_robotwin_mot_wan22_5b/text_embedding_cache data.action_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json'

bash examples/robotwin/scripts/launch_starwam_robotwin_train.sh
```

The recipe defaults to `batch_size=8`, `gradient_accumulation_steps=16`, which
produces global batch 1024 on 8 GPUs.

### 7.2 Shared-DiT, two nodes (16 GPUs)

Run the following on both nodes, changing only `MACHINE_RANK` to `0` or `1`.
`MAIN_PROCESS_IP` must be the rank-0 address reachable from rank 1.

```bash
cd /path/to/starWAM
export REPO_DIR=/path/to/starWAM
export CONDA_ENV=starwam
export RECIPE=examples/robotwin/configs/recipes/starwam_robotwin_shared_dit_wan22_5b.yaml
export ACCELERATE_CONFIG=configs/accelerate/deepspeed_zero2_multinode.yaml
export NUM_MACHINES=2 NUM_PROCESSES=16 MACHINE_RANK=0  # use 1 on rank 1
export MAIN_PROCESS_IP=10.0.0.1 MAIN_PROCESS_PORT=29617
export TRAIN_OVERRIDES='data.dataset_dirs=["/path/to/robotwin2.0"] backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B training.output_dir=/path/to/output/starwam_robotwin_shared_dit_wan22_5b data.text_embedding_cache_dir=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/text_embedding_cache data.action_stats_path=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/action_stats.json training.batch_size=16 training.gradient_accumulation_steps=4 training.num_workers=12'

bash examples/robotwin/scripts/launch_starwam_robotwin_train.sh
```

This uses `16 per device x 4 accumulation x 16 GPUs = 1024` global batch and
does not set `framework.action_expert_init_from`.

## 8. Rollout / Evaluation

RoboTwin's `script/eval_policy.py` runs 100 episodes per (task, config). Both
reported evaluations use all 50 tasks, `demo_clean` and `demo_randomized`,
`instruction_type=unseen`, and `replan_steps=24`. MoT uses 4 inference steps;
Shared-DiT uses 16 video and 16 action inference steps.

Do NOT wrap a task-setting in a wall-clock `timeout`: RoboTwin already enforces a
per-episode step limit, so 100 episodes terminate on their own. A group-level
timeout truncates episodes and biases the success rate.

### 8.1 Split env (recommended): servers + client

Step 1: start one inference server per GPU in the Torch/StarWAM env. The recipe
provides the default inference steps; the explicit values below document the
reported settings.

MoT (4 steps):

```bash
REPO=/path/to/starWAM \
PY=/path/to/starwam-env/bin/python \
RECIPE=examples/robotwin/configs/recipes/starwam_robotwin_mot_wan22_5b.yaml \
CKPT=/path/to/output/starwam_robotwin_mot_wan22_5b/checkpoint-27500/pytorch_model \
BACKBONE=/path/to/Wan2.2-TI2V-5B \
ACTION_STATS=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json \
TEXTCACHE=/path/to/output/starwam_robotwin_mot_wan22_5b/eval_text_cache \
ACTION_INIT=/path/to/preprocessed/starwam_action_dit_init_wan22.pt \
NUM_INFERENCE_STEPS=4 ACTION_NUM_INFERENCE_STEPS=4 \
SERVER_BIND=0.0.0.0 SERVER_PORT_BASE=8765 NGPU=8 \
bash examples/robotwin/scripts/launch_starwam_robotwin_policy_server.sh
```

Shared-DiT (16 video / 16 action steps, no `ACTION_INIT`):

```bash
REPO=/path/to/starWAM \
PY=/path/to/starwam-env/bin/python \
RECIPE=examples/robotwin/configs/recipes/starwam_robotwin_shared_dit_wan22_5b.yaml \
CKPT=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/checkpoint-45000/pytorch_model \
BACKBONE=/path/to/Wan2.2-TI2V-5B \
ACTION_STATS=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/action_stats.json \
TEXTCACHE=/path/to/output/starwam_robotwin_shared_dit_wan22_5b/eval_text_cache \
NUM_INFERENCE_STEPS=16 ACTION_NUM_INFERENCE_STEPS=16 \
SERVER_BIND=0.0.0.0 SERVER_PORT_BASE=8765 NGPU=8 \
bash examples/robotwin/scripts/launch_starwam_robotwin_policy_server.sh
```

Wait until each `robotwin_server_logs/server_*.log` prints `model ready ... listening on`.

Step 2: symlink the model-family-agnostic client adapter and run all 100
task-settings in the RoboTwin env. Set `CKPT_TAG=starwam27500` for MoT or
`CKPT_TAG=sharedit45000` for Shared-DiT.

```bash
ln -s /path/to/starWAM/examples/robotwin RoboTwin/policy/starwam_client
cd RoboTwin

export VK_ICD_FILENAMES=/path/to/nvidia_icd.json   # if SAPIEN needs it
SERVER_HOST=10.0.0.1 SERVER_PORT_BASE=8765 NSERVERS=8 \
CLIENT_GPUS="0" NWORKER=8 CKPT_TAG=sharedit45000 \
bash /path/to/starWAM/examples/robotwin/scripts/launch_starwam_robotwin_rollout.sh
```

- A **single-GPU client** can drive all 8 servers: keep `CLIENT_GPUS="0"` and
  set `NWORKER`/`NSERVERS` to 8 (worker i → port `SERVER_PORT_BASE + i%8`).
- A **co-located 8-GPU box** maps 1:1: `CLIENT_GPUS="0 1 2 3 4 5 6 7"`.

### 8.2 Single command (single env)

If SAPIEN and Torch/StarWAM share one env, run the harness directly with
`policy_mode local` (no server needed):

```bash
ln -s /path/to/starWAM/examples/robotwin  RoboTwin/policy/starwam_policy
cd RoboTwin
python script/eval_policy.py \
  --config policy/starwam_policy/deploy_policy.yml \
  --overrides policy_name starwam_policy task_name adjust_bottle \
    task_config demo_clean instruction_type unseen ckpt_setting starwam27500 seed 0 \
    policy_mode local num_inference_steps 4 action_num_inference_steps 4 \
    checkpoint /path/to/output/starwam_robotwin_mot_wan22_5b/checkpoint-27500/pytorch_model \
    overrides "backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B data.action_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/starwam_robotwin_mot_wan22_5b/action_stats.json data.text_embedding_cache_dir=/path/to/output/starwam_robotwin_mot_wan22_5b/text_embedding_cache"
```

For Shared-DiT, also override `config_path` to the Shared-DiT recipe, use
checkpoint-45000, and set both inference-step values to 16.

### 8.3 Results

RoboTwin writes one result per (task, config) under
`RoboTwin/eval_result/<task>/<policy>/<config>/<ckpt_setting>/<time>/_result.txt`,
and the worker logs (`rollout_logs/<ckpt_tag>/worker_*.log`) print each `Success rate: X/Y`.
There is no built-in cross-task aggregation; sum successes across the 100
task-settings for the overall micro-average.

### 8.4 Released ModelScope checkpoint

The trained MoT and Shared-DiT checkpoints are released at
[`panshaohua/starwam`](https://www.modelscope.cn/models/panshaohua/starwam):

```text
starwam-robotwin/
  mot/starwam_wan225b_robotwin_mot.pt               # dual-arm MoT weights
  sharedit/starwam_wan225b_robotwin_sharedit.pt      # dual-arm Shared-DiT weights
  action_stats.json                                 # z-score action/state stats
```

Download and point `CKPT`/`ACTION_STATS` at it (server side); you still need the
Wan2.2 backbone locally:

```bash
pip install modelscope
modelscope download --model panshaohua/starwam --local_dir /path/to/starwam_ckpts
# MoT: CKPT=/path/to/starwam_ckpts/starwam-robotwin/mot/starwam_wan225b_robotwin_mot.pt
# Shared-DiT: CKPT=/path/to/starwam_ckpts/starwam-robotwin/sharedit/starwam_wan225b_robotwin_sharedit.pt
# ACTION_STATS=/path/to/starwam_ckpts/starwam-robotwin/action_stats.json
```

The released `.pt` is a plain model `state_dict`, so pass the file path directly
as `--checkpoint` (local mode) or `CKPT=` (server).

## 9. Per-task results

Each cell is the success rate over 100 episodes. Both models use
`instruction_type=unseen` and `replan_steps=24`; MoT checkpoint-27500 uses 4
inference steps, while Shared-DiT checkpoint-45000 uses 16/16 video/action steps.
Overall: **MoT 8948/10000 (89.48%)**; **Shared-DiT 9257/10000 (92.57%)**.

| Task | MoT clean | MoT rand. | Shared-DiT clean | Shared-DiT rand. |
|---|---:|---:|---:|---:|
| Adjust Bottle | 100% | 99% | 100% | 99% |
| Beat Block Hammer | 97% | 93% | 100% | 100% |
| Blocks Ranking Rgb | 100% | 100% | 100% | 100% |
| Blocks Ranking Size | 91% | 90% | 86% | 84% |
| Click Alarmclock | 100% | 100% | 100% | 100% |
| Click Bell | 100% | 100% | 100% | 100% |
| Dump Bin Bigbin | 93% | 97% | 96% | 97% |
| Grab Roller | 100% | 100% | 100% | 100% |
| Handover Block | 78% | 78% | 94% | 91% |
| Handover Mic | 99% | 100% | 99% | 100% |
| Lift Pot | 100% | 100% | 100% | 100% |
| Move Can Pot | 84% | 91% | 96% | 98% |
| Move Playingcard Away | 99% | 100% | 100% | 100% |
| Move Stapler Pad | 87% | 83% | 76% | 76% |
| Hanging Mug | 41% | 41% | 51% | 46% |
| Open Laptop | 93% | 96% | 95% | 100% |
| Open Microwave | 33% | 31% | 83% | 71% |
| Pick Diverse Bottles | 79% | 73% | 92% | 83% |
| Pick Dual Bottles | 85% | 83% | 98% | 92% |
| Place A2b Left | 99% | 97% | 99% | 97% |
| Place A2b Right | 98% | 98% | 96% | 97% |
| Place Bread Basket | 98% | 94% | 89% | 98% |
| Place Bread Skillet | 93% | 89% | 94% | 93% |
| Place Can Basket | 49% | 63% | 72% | 69% |
| Place Cans Plasticbox | 97% | 97% | 98% | 97% |
| Place Container Plate | 98% | 99% | 98% | 98% |
| Place Dual Shoes | 83% | 88% | 81% | 89% |
| Place Empty Cup | 100% | 100% | 100% | 100% |
| Place Fan | 93% | 89% | 100% | 94% |
| Place Burger Fries | 96% | 96% | 99% | 99% |
| Place Mouse Pad | 83% | 84% | 89% | 90% |
| Place Object Basket | 87% | 77% | 86% | 86% |
| Place Object Scale | 96% | 98% | 92% | 98% |
| Place Object Stand | 97% | 95% | 95% | 93% |
| Place Phone Stand | 89% | 96% | 96% | 99% |
| Move Pillbottle Pad | 99% | 100% | 100% | 99% |
| Place Shoe | 87% | 96% | 88% | 95% |
| Press Stapler | 94% | 93% | 90% | 90% |
| Put Bottles Dustbin | 81% | 86% | 96% | 98% |
| Put Object Cabinet | 94% | 92% | 92% | 93% |
| Rotate Qrcode | 89% | 86% | 89% | 87% |
| Scan Object | 92% | 92% | 93% | 88% |
| Shake Bottle | 100% | 100% | 100% | 100% |
| Shake Bottle Horizontally | 100% | 100% | 100% | 100% |
| Stack Blocks Three | 92% | 96% | 97% | 98% |
| Stack Blocks Two | 100% | 100% | 100% | 100% |
| Stack Bowls Three | 87% | 77% | 88% | 83% |
| Stack Bowls Two | 97% | 95% | 94% | 97% |
| Stamp Seal | 82% | 88% | 79% | 92% |
| Turn Switch | 55% | 68% | 72% | 75% |
| **Average** | **89.28%** | **89.68%** | **92.56%** | **92.58%** |

## 10. Troubleshooting

- SAPIEN `failed to find a rendering device`: NVIDIA Vulkan stack missing; check `vulkaninfo` lists the GPU or set `VK_ICD_FILENAMES`.
- Client can't reach servers: verify `SERVER_HOST`/ports and that servers bound `--host 0.0.0.0`.
- MoT missing ActionDiT init: run Section 6.1 and set `framework.action_expert_init_from`; Shared-DiT does not use it.
- curobo/mplib build errors on old glibc: use conda-forge or manylinux2014 wheels.
