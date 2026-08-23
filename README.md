<div align="center">
  <h1>StreamWAM</h1>
  <h3>Asynchronous World-Action Models for Streaming Robot Control</h3>

  <a href="https://github.com/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://www.modelscope.cn/models/panshaohua/starwam"><img src="https://img.shields.io/badge/ModelScope-Checkpoints-624AFF" alt="ModelScope Checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

StreamWAM is a research framework for **streaming world-action models**. It
couples video/world-model backbones with action prediction and provides
**RTC-AC**, an action-conditioned runtime that infers the next action chunk
asynchronously while the robot executes the current chunk.

The accelerated RTC-AC path combines `torch.compile`, CUDA Graph Trees, prompt
K/V caching, fixed D0/D8 graphs, and asynchronous action execution. The
inference, training, LIBERO, and RoboTwin code in this repository is open
source.

## Release status

| Asset | Status |
|---|---|
| StreamWAM inference and training code | ✅ Available in this repository |
| Accelerated RTC-AC runtime | ✅ Available in this repository |
| LIBERO and RoboTwin recipes | ✅ Available in this repository |
| MoT and Shared-DiT checkpoints | ✅ [Available on ModelScope](https://www.modelscope.cn/models/panshaohua/starwam) |
| RTC-AC benchmark checkpoint | ⏳ Coming soon |
| Technical report | ⏳ Coming soon |

> [!NOTE]
> Public checkpoints retain their original `starwam` ModelScope paths so
> existing downloads remain valid. The repository and Python package are named
> `StreamWAM` and `streamwam`.

## Quick start: accelerated RTC-AC on LIBERO

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

### 3. Launch RTC-AC

Place a compatible RTC-AC checkpoint and its dataset statistics on disk, then
run:

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

The following RTC-AC measurements were collected with four NVIDIA H100 80 GB
GPUs. Steady-state D8 statistics exclude the first background D8 call from
each worker.

| Model | Runtime | Precision | Mean / chunk | p50 | p90 | Deadline misses |
|---|---|---|---:|---:|---:|---:|
| Wan2.2-TI2V-5B RTC-AC | Inductor + CUDA Graph Trees + K/V cache | BF16 | **45.20 ms** | 45.75 ms | 46.50 ms | 0 / 4 |

The recorded run produced one Dynamo graph, zero recompiles, and zero Inductor
CUDA Graph skips. A correctly reproduced H100 setup should normally reach
approximately **40–46 ms per steady-state D8 chunk**; exact latency depends on
hardware, clocks, drivers, and competing workloads.

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
| RTC geometry | `H=32`, `stride=16`, `delay=8`, 9 video frames, 1 inference step |

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
├── wam/                   # MoT, Shared-DiT, and RTC-AC model wrappers
├── modules/               # DiT, ActionDiT, attention, and scheduler modules
├── inference/             # consistency sampling and RTC-AC runtime
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
