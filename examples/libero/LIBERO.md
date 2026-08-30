# LIBERO Examples

This document describes the LIBERO workflow for Streaming-WAM: environment setup, recipe placeholders, preprocessing, training, and rollout/evaluation.

The generic Streaming-WAM package documentation is in the repository root [README.md](../../README.md).

## 1. Environment

The root `pyproject.toml` is the canonical environment definition. It selects
Python 3.10 and pins the PyTorch 2.7.1/cu128 and Triton 3.3.1 stack used by the
accelerated AC-Stream benchmark.

```bash
python -m pip install -U uv
uv sync --extra train
source .venv/bin/activate
```

The root README uses plain `uv sync` for AC-Stream rollout. This detailed guide
also covers training, so it installs the `train` extra. Activating `.venv`
keeps the `python` and `accelerate` commands below in the locked environment.

Install LIBERO from source into the same environment without allowing it to
replace the pinned runtime packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
uv pip install -e third_party/LIBERO --no-deps
```

Pass the checkout root with `--libero-home`, `LIBERO_HOME`, or
`LIBERO_HOME_PATH`. The expected layout is:

```text
LIBERO/
└── libero/
    ├── libero/
    │   ├── benchmark/
    │   ├── bddl_files/
    │   ├── init_files/
    │   └── assets/
    └── datasets/           # optional for rollout-only use
```

Environment notes:

- `--no-deps` prevents LIBERO from changing versions pinned in `pyproject.toml`.
- The PyTorch wheel carries the CUDA 12.8 runtime. A different host CUDA
  Toolkit version is acceptable when the NVIDIA driver supports that runtime.
- The accelerated path uses Inductor/Triton; TensorRT, FlashAttention, xFormers,
  diffusers, and DeepSpeed are not required for rollout.
- On headless servers, set `export MUJOCO_GL=egl` if rendering fails.

## 2. Recipes

Current LIBERO recipes are under:

```text
examples/libero/configs/recipes/
```

| Recipe | Model family | Backbone | Notes |
| --- | --- | --- | --- |
| `streamingwam_libero_mot_wan22_5b.yaml` | `mot_wam` | Wan2.2-TI2V-5B | Fast-WAM-aligned MoT recipe. Requires a preprocessed ActionDiT init payload. |
| `streamingwam_libero_mot_cosmos_predict2.yaml` | `mot_wam` | Cosmos-Predict2-2B-Video2World | MoT recipe with Cosmos-Predict2 backbone. Requires a preprocessed Cosmos ActionDiT init payload. |
| `streamingwam_libero_shared_dit_wan22_5b.yaml` | `shared_dit_wam` | Wan2.2-TI2V-5B | Shared-DiT/register-token recipe with decoupled video/action inference steps. |
| `streamingwam_libero_feature_conditioned_wan22_5b.yaml` | `feature_conditioned_action_model` | Wan2.2-TI2V-5B | Feature-conditioned action model: a single Wan DiT forward extracts observation tokens and a randomly initialized ActionDiT predicts actions. No preprocessed ActionDiT init required. |

### Wan2.2-TI2V-5B LIBERO results

Reported success rates (50 trials/task, 10 tasks/suite) for the three
Wan2.2-TI2V-5B recipes. Values are copied from each recipe header.

| Suite | MoT (`mot_wam`) | Shared-DiT (`shared_dit_wam`) | Feature-Conditioned (`feature_conditioned_action_model`) |
| --- | --- | --- | --- |
| libero_spatial | 97.8% | 98.8% | 90.8% |
| libero_object | 98.8% | 100.0% | 95.8% |
| libero_goal | 97.2% | 97.4% | 94.0% |
| libero_10 | 94.2% | 96.4% | 81.2% |
| Overall (micro) | 97.0% | 98.2% | 90.5% |
| Eval checkpoint | checkpoint-20000 | checkpoint-50000 | checkpoint-100000 |

### Download data and backbones

```bash
# LIBERO LeRobot datasets
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir /path/to/libero_spatial_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir /path/to/libero_object_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir /path/to/libero_goal_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir /path/to/libero_10_lerobot

# Backbones
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /path/to/Wan2.2-TI2V-5B
huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World --local-dir /path/to/Cosmos-Predict2-2B-Video2World
```

Set `data.dataset_dirs` to the four LIBERO dirs above, and set `backbone.pretrained_model_id` to the selected backbone dir.

## 3. Paths you must set

The release recipes use placeholder paths. Before running training or rollout, replace them in the YAML file or pass values through `--override`.

| Field | Required for | What to set |
| --- | --- | --- |
| `backbone.pretrained_model_id` | all recipes | Local Wan2.2 or Cosmos-Predict2 checkpoint directory. Download/prepare this yourself. |
| `framework.action_expert_init_from` | Wan2.2 MoT and Cosmos-Predict2 MoT | Output of Section 5.1 (`preprocess_action_dit_init`). Not needed for Shared-DiT or the feature-conditioned recipe (both leave it `null`). |
| `training.output_dir` | all recipes | Run output directory. Checkpoints, logs, stats, and caches are written here. |
| `data.dataset_dirs` | all real LIBERO runs | LeRobot-format LIBERO dataset dirs. Set in YAML or pass a quoted list through `--override`. |
| `data.text_embedding_cache_dir` | all real LIBERO runs | Text embedding cache dir. Training creates missing caches; Wan users may also precompute via Section 5.2. |
| `data.action_stats_path` | normalized-action recipes | Action stats JSON. Created from `data.dataset_dirs` if missing. |
| `data.state_stats_path` | normalized-state recipes | State/proprio stats JSON. Can share the same file as `data.action_stats_path`. |

The `--override` parser supports scalar `key=value` overrides and quoted Python/JSON-style lists.

Example overrides:

```bash
--override \
  'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
  backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
  framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_wan22.pt \
  training.output_dir=/path/to/output/streamingwam_libero_mot_wan22_5b \
  data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache \
  data.action_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json \
  data.state_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json \
  training.wandb_enabled=false
```

## 4. Data format

Streaming-WAM expects LIBERO training data in LeRobot-style episode format. Recipes use fields such as:

- RGB video observations, optionally from multiple cameras;
- low-dimensional actions;
- optional robot proprio/state;
- task language descriptions;
- precomputed T5 text embeddings.

Typical LIBERO settings:

```yaml
data:
  dataset_type: lerobot
  dataset_dirs:
    - /path/to/libero_spatial_lerobot
    - /path/to/libero_object_lerobot
    - /path/to/libero_goal_lerobot
    - /path/to/libero_10_lerobot
  video_keys:
    - observation.images.image
    - observation.images.wrist_image
  concat_multi_camera: horizontal
  action_key: action
  state_key: observation.state
  num_frames: 33
  action_freq_ratio: 4
  normalize_actions: true
  action_norm_mode: minmax
  normalize_states: true
  state_norm_mode: minmax
```

For code-only smoke tests, use `data.dataset_type: synthetic`. This uses dummy samples and is not for real training/evaluation.

## 5. Preprocessing

### 5.1 ActionDiT initialization for MoT WAM

For Wan2.2 and Cosmos-Predict2 MoT, `framework.action_expert_init_from` is not a downloaded checkpoint. It is generated from the selected video DiT weights once before training. Both recipes still use the generic token-action `ActionDiT`; Cosmos-Predict2 uses a best-effort structural mapping from Cosmos transformer weights rather than a separate `cosmos_action_dit` implementation.

Wan2.2:

```bash
python -m streamingwam.tools.preprocess_action_dit_init \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --source-backbone wan22 \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output /path/to/preprocessed/streamingwam_action_dit_init_wan22.pt \
  --device cuda:0 \
  --dtype bfloat16
```

Cosmos-Predict2:

```bash
python -m streamingwam.tools.preprocess_action_dit_init \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_cosmos_predict2.yaml \
  --source-backbone cosmos_predict2 \
  --pretrained-model-id /path/to/Cosmos-Predict2-2B-Video2World \
  --output /path/to/preprocessed/streamingwam_action_dit_init_cosmos_predict2_notimeproj.pt \
  --device cpu \
  --dtype bfloat16
```

Set the generated path in YAML or pass `--override framework.action_expert_init_from=/path/to/preprocessed/<payload>.pt`. Not needed for Shared-DiT or the feature-conditioned recipe (both leave it `null`).

### 5.2 Text embedding cache

Text embeddings are cache files generated from LIBERO task language. Training creates missing caches automatically. For Wan2.2 recipes, you can also precompute them explicitly:

```bash
python -m streamingwam.tools.precompute_text_cache \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
  --output-dir /path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache \
  --device cuda:0 \
  --dtype bf16 \
  --override 'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]'
```

Use separate cache dirs for Wan and Cosmos. Their `data.text_cache_encoder_id` values are already set in the recipes.

### 5.3 Action/state normalization stats

If `data.normalize_actions=true` or `data.normalize_states=true`, the training dataset builder loads stats from the configured JSON path. If the file does not exist, it computes the stats from `data.dataset_dirs` and writes the JSON file.

Recommended setup:

```yaml
data:
  action_stats_path: /path/to/output/<run_name>/action_stats.json
  state_stats_path: /path/to/output/<run_name>/action_stats.json
```

Using the same JSON for action and state stats is supported; the file stores separate `action` and `state` entries.

## 6. Training

### 6.1 Wan2.2 MoT WAM

Run the ActionDiT preprocessing in Section 5.1, then launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m streamingwam.training.train \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_wan22.pt \
    training.output_dir=/path/to/output/streamingwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.2 Cosmos-Predict2 MoT WAM

Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m streamingwam.training.train \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_cosmos_predict2.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Cosmos-Predict2-2B-Video2World \
    framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_cosmos_predict2_notimeproj.pt \
    training.output_dir=/path/to/output/streamingwam_libero_mot_cosmos_predict2 \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_cosmos_predict2/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_mot_cosmos_predict2/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_mot_cosmos_predict2/action_stats.json \
    training.wandb_enabled=false
```

### 6.3 Wan2.2 Shared-DiT WAM

Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m streamingwam.training.train \
  --config examples/libero/configs/recipes/streamingwam_libero_shared_dit_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.4 Wan2.2 Feature-Conditioned action model

The feature-conditioned recipe does not need the Section 5.1 ActionDiT init; the
action expert is randomly initialized (`framework.action_expert_init_from: null`).
Launch training:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  -m streamingwam.training.train \
  --config examples/libero/configs/recipes/streamingwam_libero_feature_conditioned_wan22_5b.yaml \
  --override \
    'data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"]' \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    training.wandb_enabled=false
```

### 6.5 Launch scripts

Convenience scripts are provided in:

```text
examples/libero/scripts/
```

Current scripts:

```text
launch_streamingwam_libero_mot_wan22_5b_8gpu.sh
launch_streamingwam_libero_shared_dit_wan22_5b_8gpu.sh
launch_streamingwam_libero_feature_conditioned_wan22_5b_8gpu.sh
launch_streamingwam_libero_mot_rollout.sh
```

Before running them, edit recipe paths or pass overrides through environment variables:

```bash
cd /path/to/Streaming-WAM

export REPO_DIR=/path/to/Streaming-WAM
export TRAIN_OVERRIDES='data.dataset_dirs=["/path/to/libero_spatial_lerobot","/path/to/libero_object_lerobot","/path/to/libero_goal_lerobot","/path/to/libero_10_lerobot"] backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_wan22.pt training.output_dir=/path/to/output/streamingwam_libero_mot_wan22_5b data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache data.action_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json data.state_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json training.wandb_enabled=false'

bash examples/libero/scripts/launch_streamingwam_libero_mot_wan22_5b_8gpu.sh
```

## 7. Checkpoints

Training writes checkpoints under:

```text
${training.output_dir}/checkpoint-<step>/
```

The rollout script can load:

- a checkpoint directory containing `model.pt`, `pytorch_model.bin`, or safetensors files;
- a direct checkpoint file;
- Accelerate/DeepSpeed checkpoint layouts handled by the loader.

If `--checkpoint` is omitted, rollout searches for the latest `checkpoint-*` under `training.output_dir`.

## 8. Rollout / Evaluation

### 8.1 Wan2.2 MoT rollout

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --checkpoint /path/to/output/streamingwam_libero_mot_wan22_5b/checkpoint-20000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_wan22.pt \
    training.output_dir=/path/to/output/streamingwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_mot_wan22_5b/action_stats.json
```

### 8.2 Cosmos-Predict2 MoT rollout

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_cosmos_predict2.yaml \
  --checkpoint /path/to/output/streamingwam_libero_mot_cosmos_predict2/checkpoint-20000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Cosmos-Predict2-2B-Video2World \
    framework.action_expert_init_from=/path/to/preprocessed/streamingwam_action_dit_init_cosmos_predict2_notimeproj.pt \
    training.output_dir=/path/to/output/streamingwam_libero_mot_cosmos_predict2 \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_cosmos_predict2/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_mot_cosmos_predict2/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_mot_cosmos_predict2/action_stats.json
```

### 8.3 Wan2.2 Shared-DiT rollout

Shared-DiT supports decoupled video/action denoising step counts. Pass both values explicitly:

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_shared_dit_wan22_5b.yaml \
  --checkpoint /path/to/output/streamingwam_libero_shared_dit_wan22_5b/checkpoint-50000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 16 \
  --action-num-inference-steps 16 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/action_stats.json
```

### 8.4 Wan2.2 Feature-Conditioned rollout

Feature-conditioned uses a single action denoising schedule (no decoupled steps),
so pass only `--num-inference-steps`.

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_feature_conditioned_wan22_5b.yaml \
  --checkpoint /path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/checkpoint-100000 \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-steps-wait 30 \
  --num-inference-steps 10 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/text_embedding_cache \
    data.action_stats_path=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/action_stats.json \
    data.state_stats_path=/path/to/output/streamingwam_libero_feature_conditioned_wan22_5b/action_stats.json
```

### 8.5 Rollout launcher and outputs

`examples/libero/scripts/launch_streamingwam_libero_mot_rollout.sh` is a readable
single-task FastWAM smoke-test command. After placing the release checkpoint
and stats under `checkpoints/fastwam_release/`, edit its explicit
`--backbone-path` and `--libero-home` arguments, then run:

```bash
bash examples/libero/scripts/launch_streamingwam_libero_mot_rollout.sh
```

For larger evaluations, invoke `examples/libero/rollout.py` separately with
the desired `--task-suite-name`, `--task-id`, and `--num-trials` values.

Useful rollout options:

```bash
--libero-home /path/to/LIBERO
--output-dir /path/to/rollout_outputs
--save-video
--fixed-seed
```

With `--save-video --output-dir /path/to/rollout_outputs`, videos are written to `/path/to/rollout_outputs/videos/`. Without `--output-dir`, videos are written under `${training.output_dir}/libero_rollout/<checkpoint-name>/<task-suite-name>/videos/`. Rollout videos save one frame per executed environment step after the initial wait period.

The rollout script loads the recipe/checkpoint, applies overrides, builds the model, and repeatedly calls `model.infer_action(...)` in LIBERO.

### 8.6 Results files and cross-suite summary

Each `rollout.py` run writes one `results.json` per suite:

```text
<training.output_dir>/libero_rollout/<checkpoint-name>/<task-suite-name>/results.json
```

Each `results.json` contains per-task success rates and the suite-level
micro-average (`total_successes`, `total_trials`, `success_rate`). It does
**not** aggregate across suites — each suite is a separate run/file.

To get the overall (cross-suite) success rate, aggregate the four
per-suite files with `examples/libero/summarize_results.py`:

```bash
# Aggregate every suite under one checkpoint directory:
python examples/libero/summarize_results.py \
  --rollout-dir /path/to/output/<recipe>/libero_rollout/<checkpoint-name>

# Or point at explicit results.json files:
python examples/libero/summarize_results.py \
  --results \
    /path/to/.../libero_spatial/results.json \
    /path/to/.../libero_object/results.json \
    /path/to/.../libero_goal/results.json \
    /path/to/.../libero_10/results.json \
  --output /path/to/summary.json
```

It prints a per-suite table plus the micro-average across all suites, and
writes `summary.json` (defaults to `<rollout-dir>/summary.json` when
`--rollout-dir` is used):

```text
Suite            Success  Trials  Success rate
---------------  -------  ------  ------------
libero_spatial   471      500     94.2%
libero_object    500      500     100.0%
libero_goal      484      500     96.8%
libero_10        481      500     96.2%
---------------  -------  ------  ------------
Overall (micro)  1936     2000    96.8%
```

The `Overall (micro)` value is `sum(total_successes) / sum(total_trials)`
across suites, matching the reported numbers in Section 2.

### 8.7 Rollout from released ModelScope checkpoints

Pretrained Wan2.2-TI2V-5B checkpoints remain in the pre-rename external
ModelScope namespace
[`panshaohua/starwam`](https://www.modelscope.cn/models/panshaohua/starwam).
Download them first:

```bash
uv tool run --from modelscope modelscope download \
  --model panshaohua/starwam \
  --local_dir /path/to/streamingwam_ckpts
```

After download, the LIBERO checkpoints are laid out as:

```text
/path/to/streamingwam_ckpts/starwam-libero/
  action_stats.json                          # shared action stats (both models)
  mot/starwam_wan225b_mot.pt                  # MoT WAM
  sharedit/starwam_wan225b_shareddit.pt       # Shared-DiT WAM
```

Point `--checkpoint` directly at the `.pt` file (these files use custom names,
so pass the file path, not the directory). You still need the Wan2.2 backbone
locally for `backbone.pretrained_model_id`.

Shared-DiT:

```bash
CKPT_ROOT=/path/to/streamingwam_ckpts/starwam-libero

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_shared_dit_wan22_5b.yaml \
  --checkpoint "$CKPT_ROOT/sharedit/starwam_wan225b_shareddit.pt" \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 16 \
  --action-num-inference-steps 16 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_shared_dit_wan22_5b/text_embedding_cache \
    data.action_stats_path="$CKPT_ROOT/action_stats.json" \
    data.state_stats_path="$CKPT_ROOT/action_stats.json"
```

MoT uses a single denoising schedule — pass only
`--num-inference-steps`:

```bash
CKPT_ROOT=/path/to/streamingwam_ckpts/starwam-libero

python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --checkpoint "$CKPT_ROOT/mot/starwam_wan225b_mot.pt" \
  --task-suite-name libero_spatial \
  --num-trials 50 \
  --num-inference-steps 8 \
  --replan-steps 10 \
  --device cuda:0 \
  --libero-home /path/to/LIBERO \
  --override \
    backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
    training.output_dir=/path/to/output/streamingwam_libero_mot_wan22_5b \
    data.text_embedding_cache_dir=/path/to/output/streamingwam_libero_mot_wan22_5b/text_embedding_cache \
    data.action_stats_path="$CKPT_ROOT/action_stats.json" \
    data.state_stats_path="$CKPT_ROOT/action_stats.json"
```

## 9. Decoupled action steps

MoT WAM uses a single denoising schedule for action rollout. For MoT recipes, `examples/libero/rollout.py` overrides `action_num_inference_steps` to match `num_inference_steps`, so `inference.action_num_inference_steps` is kept only for schema consistency.

Shared-DiT uses decoupled step counts, so rollout passes `--action-num-inference-steps` separately.

## 10. Direct FastWAM release checkpoint inference

Streaming-WAM can load the original FastWAM release checkpoint and statistics at
runtime without writing a converted checkpoint. Select the source format
explicitly with `--checkpoint-format fastwam`:

Before using the smoke-test launcher, place the two original FastWAM release
artifacts at these fixed local paths (they are ignored by Git and are not part
of a fresh clone):

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

After those files are present, only `--backbone-path` and `--libero-home` in
the launcher are machine-specific.

```bash
python examples/libero/rollout.py \
  --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --backbone-path /path/to/wan22_5b \
  --stats-path checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --libero-home /path/to/LIBERO \
  --task-id 0 \
  --num-trials 1 \
  --num-steps-wait 30 \
  --replan-steps 10 \
  --num-inference-steps 10 \
  --device cuda:0 \
  --mujoco-gl osmesa \
  --save-video
```

The Wan2.2 directory is still required for the VAE, T5 encoder/tokenizer, and
architecture metadata. FastWAM loading automatically ignores the recipe's
separate `framework.action_expert_init_from`, because the release checkpoint
contains the complete action expert. Video expert, action expert, and proprio
encoder keys and tensor shapes are validated before any weights are copied;
incompatible or partial checkpoints fail immediately.

The LIBERO launcher contains the same arguments directly. Edit its two
machine-specific paths and run:

```bash
bash examples/libero/scripts/launch_streamingwam_libero_mot_rollout.sh
```

## 11. Balanced multi-GPU evaluation

`multigpu_rollout.py` treats each `(suite, task_id, trial_id)` as one work
unit. It assigns all units across the selected physical GPUs with a difference
of at most one unit, then keeps one model process alive on each GPU. Worker
logs, videos, manifests, and results are isolated below the manager output
directory. The terminal receives only the merged success and timing summary.

The ready-to-run Joint CD launcher evaluates all 40 tasks once on GPUs
`0,1,2,3`:

```bash
bash examples/libero/scripts/launch_streamingwam_libero_joint_cd_4gpu.sh
```

Select different GPUs without editing Python:

```bash
GPU_IDS=2,3,6,7 bash examples/libero/scripts/launch_streamingwam_libero_joint_cd_4gpu.sh
```

AC-Stream has one launcher and one implementation. Set the four public asset
paths once, then run eager mode without an extra option or append
`--ac-stream-accelerated` for the compiler/cache path:

```bash
export BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B
export LIBERO_HOME_PATH=/path/to/LIBERO
export CHECKPOINT_PATH=/path/to/ac_stream_checkpoint.pt
export STATS_PATH=/path/to/dataset_stats.json

GPU_IDS=0,1,2,3 \
  bash examples/libero/scripts/launch_streamingwam_libero_ac_stream_4gpu.sh

GPU_IDS=0,1,2,3 \
  bash examples/libero/scripts/launch_streamingwam_libero_ac_stream_4gpu.sh \
  --ac-stream-accelerated
```

The accelerated run uses strict full-graph/static `torch.compile` with
Inductor CUDA Graph Trees, prompt cross-attention K/V caching, D0/D8 mask and
schedule caching, and one D0 plus one D8 prewarm per worker. Compilation and
prewarm are outside episode timing. A compilation or prewarm failure aborts
the worker instead of silently falling back to eager execution. The merged
`results.json` records the active acceleration backend and cache status.
Public AC-Stream Chunk Time is the sample-weighted mean of every recorded D8
model call after prewarming; D0 is excluded. The results also record Dynamo
graph/recompile counts, Inductor CUDA Graph skips, the first background-thread
D8 latency, and steady-state D8 mean/p50/p90 after separating the first runtime
D8 sample from each worker. For both synchronous and AC-Stream rollouts, Total
Time begins after reset and dummy stabilization, immediately before the first
policy observation, and ends when the terminal environment action returns.

For latency validation, use the locked environment from Section 1:

```bash
PYTHON_BIN=.venv/bin/python \
GPU_IDS=0,1,2,3 \
BACKBONE_PATH=/path/to/Wan2.2-TI2V-5B \
LIBERO_HOME_PATH=/path/to/LIBERO \
CHECKPOINT_PATH=/path/to/ac_stream_checkpoint.pt \
STATS_PATH=/path/to/dataset_stats.json \
  bash examples/libero/scripts/launch_streamingwam_libero_ac_stream_4gpu.sh \
  --ac-stream-accelerated
```

The validated H100 steady-state target is approximately 40–46 ms per D8 chunk.
The recorded steady-state reference reached 45.20 ms mean, 45.75 ms p50, and
46.50 ms p90, with one Dynamo graph, zero recompiles, zero CUDA Graph skips,
and zero of four deadline misses. These values remain diagnostics rather than
the all-D8 public mean.

To run 50 trials per task, change `--num-trials 1` to
`--num-trials 50` in the launcher. That creates 2000 episode units: 500 per
GPU with four GPUs, or 250 per GPU with eight GPUs. The complete assignment is
saved as `assignments.json`, per-worker diagnostics are in
`worker_gpu*/worker.log`, and the merged result is `results.json`.

## 12. Troubleshooting

- Placeholder paths: replace all `/path/to/...` values before running.
- Missing ActionDiT init for Streaming-WAM MoT checkpoints: run Section 5.1 and set
  `framework.action_expert_init_from`. Original FastWAM checkpoints selected
  with `--checkpoint-format fastwam` do not need this payload.
- LIBERO import error: install LIBERO or pass `--libero-home /path/to/LIBERO`.
- No checkpoint found: pass `--checkpoint` or check `training.output_dir`.
- Wrong action scale: check action/state stats paths and normalization settings.
- Multi-camera mismatch: check `video_keys`, `concat_multi_camera`, and `video_size`.
