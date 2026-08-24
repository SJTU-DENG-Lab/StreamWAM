<div align="center">
  <h1>StreamWAM</h1>
  <h3>Streaming World-Action Models for Robotic Manipulation</h3>

  <a href="https://github.com/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://huggingface.co/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-FFD21E" alt="Hugging Face Checkpoint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

StreamWAM is a research framework for **streaming World-Action Models (WAMs)**.
It provides a unified testbed for systematically studying and comparing
different streaming strategies for WAM-based robot control.

Building on this framework, we introduce **Action-Conditioned Streaming WAM
(AC-StreamWAM)**, a streaming formulation that feeds the prefix of actions
currently being executed by the robot back into the world model, conditioning
future video generation on ongoing actions. Rather than treating
inference–execution overlap merely as a systems optimization, AC-StreamWAM
couples the two processes: the actions being executed shape the predicted
visual future, while the model asynchronously infers the next world-action
chunk as the robot continues executing the current action chunk.

## Release status

| Asset | Status |
|---|---|
| StreamWAM inference and training code | ✅ Available in this repository |
| Accelerated AC-StreamWAM runtime | ✅ Available in this repository |
| LIBERO and RoboTwin recipes | ✅ Available in this repository |
| FastWAM-Joint-CD checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |
| AC-StreamWAM checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |
| Technical report | ⏳ Coming soon |

## Quick start: accelerated AC-StreamWAM on LIBERO

The reference environment uses Python 3.10, PyTorch 2.7.1/cu128, and Triton
3.3.1. `pyproject.toml` is the canonical dependency definition.

### 1. Install StreamWAM

```bash
git clone https://github.com/SJTU-DENG-Lab/StreamWAM.git
cd StreamWAM

python -m pip install -U uv
uv sync
```

`uv` installs PyTorch and torchvision from the official cu128 wheel index. A
compatible NVIDIA driver is required; the host CUDA Toolkit does not need to
match the wheel's bundled CUDA 12.8 runtime exactly.

### 2. Prepare LIBERO and Wan2.2

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
uv pip install -e third_party/LIBERO --no-deps

uv run huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

LIBERO is supplied as an external source checkout through `LIBERO_HOME_PATH`.
Its expected source tree contains `libero/libero/{benchmark,bddl_files,
init_files,assets}`. A `datasets/` directory is optional for rollout-only use.

### 3. Launch AC-StreamWAM

Place a compatible AC-StreamWAM checkpoint and its dataset statistics on disk,
then run:

```bash
PYTHON_BIN=.venv/bin/python \
GPU_IDS=0,1,2,3 \
BACKBONE_PATH="$PWD/checkpoints/Wan2.2-TI2V-5B" \
LIBERO_HOME_PATH="$PWD/third_party/LIBERO" \
CHECKPOINT_PATH=/path/to/rtc_ac_checkpoint.pt \
STATS_PATH=/path/to/dataset_stats.json \
  bash examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh \
  --rtc-ac-accelerated
```

The launcher defaults to one trial for every task in `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`. See the
[LIBERO guide](examples/libero/LIBERO.md) for checkpoint formats, training,
single-task rollout, and evaluation controls.

## Current results

We evaluate all methods on four LIBERO suites with 50 trials per task and 10
tasks per suite. Success rates are reported as percentages, and `Average` is
the arithmetic mean across LIBERO-10, LIBERO-Spatial, LIBERO-Goal, and
LIBERO-Object. `Chunk Time` measures the average inference latency per action
chunk, while `Episode Time` reports the average wall-clock duration for long-
and short-horizon tasks.

| Method | LIBERO-10 | Spatial | Goal | Object | Average (%) ↑ | Chunk Time (ms) ↓ | Episode Time (s) ↓ Long / Short |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastWAM | 96.20 | 96.20 | 94.20 | 96.20 | 95.70 | 493.0 | 16.31 / 8.25 |
| FastWAM-Joint-CD | 97.20 | 99.60 | 98.60 | 100.00 | 98.85 | 114.2 | 6.89 / 3.74 |
| FastWAM-RTC | 58.40 | 76.20 | 77.00 | 83.40 | 73.75 | 142.3 | 6.23 / 3.20 |
| **AC-StreamWAM (Ours)** | 96.60 | 98.80 | 97.40 | 100.00 | 98.20 | 41.0 | 5.36 / 3.15 |
| w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 | 35.1 | 5.20 / 2.92 |
| w/o Slot Encoder | 95.60 | 98.40 | 96.80 | 99.80 | 97.65 | 36.3 | 5.31 / 3.01 |

AC-StreamWAM achieves a 98.20% average success rate with a chunk latency of
41.0 ms, providing a strong balance between control performance and streaming
efficiency. It reduces chunk latency by approximately 12.0× compared with
FastWAM and 2.8× compared with FastWAM-Joint-CD. Removing action conditioning
decreases the average success rate by 1.95 percentage points, while removing
the slot encoder results in a 0.55-point drop.

## Accelerated runtime contract

The measured path uses PyTorch Inductor/Triton and does not require TensorRT,
DeepSpeed, FlashAttention, xFormers, or diffusers.

| Component | Validated value |
|---|---|
| Python | 3.10.20 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| Triton | 3.3.1 |
| GPU / dtype | NVIDIA H100 80 GB / BF16 |
| Compiler tools | GCC/G++ 11.4.0, ninja 1.13.0 |
| Input | batch 1, `[1, 3, 224, 448]` |
| Wan2.2 5B | hidden size 3072, 30 layers, 24 heads |
| AC-StreamWAM geometry | `H=32`, `stride=16`, `delay=8`, 9 video frames, 1 inference step |

A healthy accelerated run reports:

```text
compile_active: True
compile_fullgraph: True
compile_dynamic: False
cuda_graph_trees: True
dynamo_unique_graphs: 1
dynamo_recompiles: 0
inductor_cudagraph_skips: 0
prewarmed_d0: True
prewarmed_d8: True
```

Acceleration is intentionally strict: unsupported shapes, non-BF16 inputs, or
compiler/prewarm failures abort instead of silently falling back to eager
execution.

## Runtime layout

```text
streamwam/
├── backbone/              # Wan2.2 and Cosmos-Predict2 adapters
├── wam/                   # MoT, Shared-DiT, and AC-StreamWAM model wrappers
├── modules/               # DiT, ActionDiT, attention, and scheduler modules
├── inference/             # consistency sampling and AC-StreamWAM runtime
├── checkpointing/         # native and FastWAM checkpoint adapters
├── training/              # trainers, losses, and entrypoint
└── data/                  # dataset and text-cache utilities
examples/
├── libero/                # LIBERO recipes, rollout, and launchers
└── robotwin/              # RoboTwin recipes and deployment adapters
```

## Citation

A formal BibTeX entry will be added with the technical report. Until then,
please cite the repository URL in derived work.

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

StreamWAM builds on ideas and open-source work from
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[StarVLA](https://github.com/starVLA/starVLA),
[DreamZero](https://github.com/dreamzero0/dreamzero),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[Wan2.2](https://github.com/Wan-Video/Wan2.2), and
[Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).
