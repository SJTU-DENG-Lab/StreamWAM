---
license: apache-2.0
library_name: pytorch
tags:
  - robotics
  - robot-learning
  - world-model
  - action-generation
  - libero
  - pytorch
---

<div align="center">
  <h1>StreamWAM</h1>
  <h3>Streaming World-Action Models for Robotic Manipulation</h3>

  <a href="https://github.com/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://github.com/SJTU-DENG-Lab/StreamWAM/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

StreamWAM is a research framework for streaming World-Action Models. It provides a unified testbed for studying and comparing efficient robot-control strategies.

StreamWAM uses the actions currently being executed by the robot to guide its next prediction. This allows action execution and model inference to proceed together, reducing the time required to complete a robot task while maintaining strong control performance.

This repository provides the released LIBERO checkpoints. The corresponding inference and evaluation code is available in the [StreamWAM GitHub repository](https://github.com/SJTU-DENG-Lab/StreamWAM).

## Released checkpoints

| Directory | Model | Description |
| --- | --- | --- |
| `joint-cd/` | FastWAM-Joint-CD | Fast one-step joint world-and-action prediction for LIBERO. |
| `ac-stream/` | StreamWAM | Recommended StreamWAM checkpoint for efficient LIBERO evaluation. |

Each directory contains:

```text
model.pt
dataset_stats.json
```

The checkpoints use the Wan2.2 TI2V 5B backbone. Wan2.2 model assets are not included here and must be downloaded separately.

## LIBERO results

We evaluate all methods on LIBERO-10, LIBERO-Spatial, LIBERO-Goal, and LIBERO-Object with 50 trials per task. `Chunk Time` is the average model inference time for one action chunk. `Episode Time` is the average end-to-end task duration for long- and short-horizon tasks.

| Method | LIBERO-10 | Spatial | Goal | Object | Average (%) ↑ | Chunk Time (ms) ↓ | Episode Time (s) ↓ Long / Short |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FastWAM | 96.20 | 96.20 | 94.20 | 96.20 | 95.70 | 493.0 | 16.31 / 8.25 |
| FastWAM-Joint-CD | 97.20 | 99.60 | 98.60 | 100.00 | 98.85 | 114.2 | 6.89 / 3.74 |
| **StreamWAM** | **96.60** | **98.80** | **97.40** | **100.00** | **98.20** | **41.0** | **5.36 / 3.15** |

## Installation

Clone the code repository and install the environment:

```bash
git clone https://github.com/SJTU-DENG-Lab/StreamWAM.git
cd StreamWAM

python -m pip install -U uv
uv sync
```

The reference environment uses Python 3.10, PyTorch 2.7.1 with CUDA 12.8, and Triton 3.3.1.

Prepare LIBERO and the Wan2.2 backbone:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
uv pip install -e third_party/LIBERO --no-deps

uv run huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

## Download checkpoints

Download both released checkpoints:

```bash
hf download SJTU-DENG-Lab/StreamWAM \
  --local-dir checkpoints/streamwam
```

The resulting layout is:

```text
checkpoints/streamwam/
├── joint-cd/
│   ├── model.pt
│   └── dataset_stats.json
└── ac-stream/
    ├── model.pt
    └── dataset_stats.json
```

## Run StreamWAM

Evaluate all 40 LIBERO tasks once on four GPUs:

```bash
PYTHON_BIN=.venv/bin/python \
GPU_IDS=0,1,2,3 \
BACKBONE_PATH="$PWD/checkpoints/Wan2.2-TI2V-5B" \
LIBERO_HOME_PATH="$PWD/third_party/LIBERO" \
CHECKPOINT_PATH="$PWD/checkpoints/streamwam/ac-stream/model.pt" \
STATS_PATH="$PWD/checkpoints/streamwam/ac-stream/dataset_stats.json" \
  bash examples/libero/scripts/launch_streamwam_libero_ac_stream_4gpu.sh \
  --ac-stream-accelerated
```

The launcher evaluates one trial for every task in `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`. GPU IDs can be changed through `GPU_IDS`.

## Run FastWAM-Joint-CD

```bash
python examples/libero/multigpu_rollout.py \
  --gpus 0,1,2,3 \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/streamwam_libero_joint_cd_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/streamwam/joint-cd/model.pt \
  --backbone-path checkpoints/Wan2.2-TI2V-5B \
  --stats-path checkpoints/streamwam/joint-cd/dataset_stats.json \
  --libero-home third_party/LIBERO \
  --num-steps-wait 30 \
  --replan-steps 16 \
  --num-inference-steps 1 \
  --sampling-method consistency \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video
```

For more evaluation options, see the [LIBERO guide](https://github.com/SJTU-DENG-Lab/StreamWAM/blob/main/examples/libero/LIBERO.md).

## License

Released under the [Apache License 2.0](https://github.com/SJTU-DENG-Lab/StreamWAM/blob/main/LICENSE).

## Acknowledgements

StreamWAM builds on ideas and open-source work from [FastWAM](https://github.com/yuantianyuan01/FastWAM), [StarWAM](https://github.com/shaohua-pan/StarWAM), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and [Wan2.2](https://github.com/Wan-Video/Wan2.2).
