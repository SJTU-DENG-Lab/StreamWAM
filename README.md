<div align="center">
  <h1>StreamWAM</h1>
  <h3>Streaming World-Action Models for Robotic Manipulation</h3>

  <a href="https://sjtu-deng-lab.github.io/StreamWAM/"><img src="https://img.shields.io/badge/Project-Page-087D70?logo=githubpages&logoColor=white" alt="Project Page"></a>
  <a href="https://github.com/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://huggingface.co/SJTU-DENG-Lab/StreamWAM"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-FFD21E" alt="Hugging Face Checkpoint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

StreamWAM is a research framework for **streaming World-Action Models (WAMs)**.
It provides a unified testbed for systematically studying and comparing
different streaming strategies for WAM-based robot control.

Building on this framework, we introduce **StreamWAM**, an
**action-conditioned streaming formulation** that feeds the prefix of actions
currently being executed by the robot back into the world model. This
explicitly conditions future video generation on ongoing robot actions.
Rather than treating inference–execution overlap merely as a systems
optimization, StreamWAM couples the two processes: the executed action prefix
shapes the predicted visual future, while the model asynchronously infers the
next world-action chunk as the robot continues executing the current chunk.

## Release status

| Asset | Status |
|---|---|
| StreamWAM inference and training code | ✅ Available in this repository |
| Accelerated StreamWAM runtime | ✅ Available in this repository |
| LIBERO and RoboTwin recipes | ✅ Available in this repository |
| FastWAM-Joint-CD checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |
| StreamWAM checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |
| Technical report | ⏳ Coming soon |

## Quick start: accelerated StreamWAM on LIBERO

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

### 3. Launch StreamWAM

Place a compatible StreamWAM checkpoint and its dataset statistics on disk,
then run:

```bash
PYTHON_BIN=.venv/bin/python \
GPU_IDS=0,1,2,3 \
BACKBONE_PATH="$PWD/checkpoints/Wan2.2-TI2V-5B" \
LIBERO_HOME_PATH="$PWD/third_party/LIBERO" \
CHECKPOINT_PATH=/path/to/ac_stream_checkpoint.pt \
STATS_PATH=/path/to/dataset_stats.json \
  bash examples/libero/scripts/launch_streamwam_libero_ac_stream_4gpu.sh \
  --ac-stream-accelerated
```

The launcher defaults to one trial for every task in `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`. See the
[LIBERO guide](examples/libero/LIBERO.md) for checkpoint formats, training,
single-task rollout, and evaluation controls.

## Current results

All StreamWAM models are initialized from FastWAM-Joint checkpoints and then
further trained. The RoboCasa implementation builds on X-WAM, while the
RoboTwin implementation builds on StarWAM.

### LIBERO

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
| StreamWAM | 96.60 | 98.80 | 97.40 | 100.00 | 98.20 | 41.0 | 5.36 / 3.15 |
| w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 | 35.1 | 5.20 / 2.92 |
| w/o Slot Encoder | 95.60 | 98.40 | 96.80 | 99.80 | 97.65 | 36.3 | 5.31 / 3.01 |

StreamWAM achieves a 98.20% average success rate with a chunk latency of
41.0 ms, providing a strong balance between control performance and streaming
efficiency. It reduces chunk latency by approximately 12.0× compared with
FastWAM and 2.8× compared with FastWAM-Joint-CD. Removing action conditioning
decreases the average success rate by 1.95 percentage points, while removing
the slot encoder results in a 0.55-point drop.

### RoboCasa

We evaluate on the standard RoboCasa protocol across 24 kitchen manipulation
tasks, with 50 trials per task, and report the average success rate.

| Method | Accuracy (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|
| X-WAM | 75.42 | 374.07 | 17.36 |
| X-WAM-CD | 75.83 | 134.37 | 13.04 |
| StreamWAM | 75.35 | 115.98 | 9.49 |

### RoboTwin

We evaluate 50 RoboTwin 2.0 tasks with 100 rollout episodes per task. `Clean`
reports the success rate under the easy setting, while `Random` reports the
success rate under the hard domain-randomization setting.

| Method | Clean (%) ↑ | Random (%) ↑ | Total (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|---:|---:|
| StarWAM | 84.8 | 86.0 | 85.4 | 189.3 | — |
| StarWAM-CD | 79.0 | 79.2 | 79.1 | 81.6 | — |
| StreamWAM | 87.2 | 88.8 | 87.6 | — | 112.2 |

## Runtime layout

```text
streamwam/
├── backbone/              # Wan2.2 and Cosmos-Predict2 adapters
├── wam/                   # MoT, Shared-DiT, and StreamWAM model wrappers
├── modules/               # DiT, ActionDiT, attention, and scheduler modules
├── inference/             # consistency sampling and StreamWAM runtime
├── checkpointing/         # native and FastWAM checkpoint adapters
├── training/              # trainers, losses, and entrypoint
└── data/                  # dataset and text-cache utilities
examples/
├── libero/                # LIBERO recipes, rollout, and launchers
└── robotwin/              # RoboTwin recipes and deployment adapters
```

## Citation

The arXiv entry is not public yet. For now, please cite the
[project page](https://sjtu-deng-lab.github.io/StreamWAM/):

```bibtex
@misc{denglab2026streamwam,
  title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/StreamWAM/}
}
```

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

StreamWAM builds on ideas and open-source work from
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[StarWAM](https://github.com/shaohua-pan/StarWAM),
[X-WAM](https://github.com/sharinka0715/X-WAM),
[StarVLA](https://github.com/starVLA/starVLA),
[DreamZero](https://github.com/dreamzero0/dreamzero),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[Wan2.2](https://github.com/Wan-Video/Wan2.2), and
[Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).
