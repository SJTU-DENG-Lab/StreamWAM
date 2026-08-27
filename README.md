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

Stream-WAM is further trained from FastWAM-Joint on LIBERO. To evaluate the
same streaming design across benchmarks and World Action Model families, the
RoboCasa study builds on X-WAM and the RoboTwin 2.0 study builds on StarWAM.

### Task performance

CD denotes one-step consistency distillation. We also report Stream-WAM
ablations without action conditioning and without the slot encoder. Best and
second-best results are shown in **bold** and <u>underlined</u>, respectively.

#### LIBERO

LIBERO evaluation covers four suites with 10 tasks per suite and 50 trials per
task. Success is averaged across Long, Spatial, Goal, and Object.

| Method | Long | Spatial | Goal | Object | Average ↑ |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 53.7 | 84.7 | 79.2 | 88.4 | 76.5 |
| π₀ | 85.2 | 96.8 | 95.8 | 98.8 | 94.1 |
| π₀.₅ | 92.4 | <u>98.8</u> | <u>98.0</u> | 98.2 | 96.9 |
| Motus | **97.6** | 96.8 | 96.6 | <u>99.8</u> | 97.7 |
| Fast-WAM | 95.2 | 98.2 | 97.0 | **100.0** | 97.6 |
| FastWAM-Joint-CD | <u>97.20</u> | **99.60** | **98.60** | **100.00** | **98.85** |
| FastWAM-RTC | 58.40 | 76.20 | 77.00 | 83.40 | 73.75 |
| Stream-WAM (Ours) | 96.60 | <u>98.80</u> | 97.40 | **100.00** | <u>98.20</u> |
| Stream-WAM w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 |
| Stream-WAM w/o Slot Encoder | 95.60 | 98.40 | 96.80 | <u>99.80</u> | 97.65 |

#### RoboTwin 2.0

RoboTwin 2.0 evaluates 50 tasks with 100 rollout episodes per task. Clean
reports the easy setting and Random reports the hard domain-randomization
setting.

| Method | Clean ↑ | Random ↑ | Total ↑ |
|---|---:|---:|---:|
| π₀ | 65.92 | 58.40 | 62.2 |
| π₀.₅ | 82.74 | 76.76 | 79.8 |
| Motus | **88.66** | 87.02 | **87.8** |
| Motus from WAN2.2 | 77.56 | 77.00 | 77.3 |
| FastWAM-Joint | <u>87.8</u> | <u>87.32</u> | 87.56 |
| StarWAM-Joint | 84.8 | 86.0 | 85.4 |
| StarWAM-CD | 79.0 | 79.2 | 79.1 |
| Stream-WAM (Ours) | 87.2 | **88.8** | <u>87.6</u> |

#### RoboCasa

RoboCasa evaluation follows the standard 24-task protocol, covering 24
kitchen manipulation tasks with 50 trials per task and reporting average
success.

| Method | Average Success ↑ |
|---|---:|
| π₀.₅ | 41.4% |
| π₀-FAST | 61.2% |
| π₀ | 62.5% |
| Cosmos Policy | 67.1% |
| X-WAM | **75.42%** |
| X-WAM-CD | 75.33% |
| Stream-WAM (Ours) | <u>75.35%</u> |

### Inference efficiency

Chunk Time measures the latency required to prepare the next action chunk.
Episode Time captures the accumulated cost of inference, execution, and
replanning over a complete rollout.

| Benchmark | Method | Chunk Time | Episode Time |
|---|---|---:|---:|
| LIBERO | FastWAM | 493.0 ms | 16.31 s Long / 8.25 s Short |
| LIBERO | FastWAM-Joint-CD | 114.2 ms | 6.89 s Long / 3.74 s Short |
| LIBERO | FastWAM-RTC | 142.3 ms | 6.23 s Long / 3.20 s Short |
| LIBERO | Stream-WAM | 41.0 ms | 5.36 s Long / 3.15 s Short |
| LIBERO | Stream-WAM w/o Action Conditioning | 35.1 ms | 5.20 s Long / 2.92 s Short |
| LIBERO | Stream-WAM w/o Slot Encoder | 36.3 ms | 5.31 s Long / 3.01 s Short |
| RoboTwin 2.0 | StarWAM-Joint | 190.17 ms | 110.22 s |
| RoboTwin 2.0 | StarWAM-CD | 81.21 ms | 102.59 s |
| RoboTwin 2.0 | Stream-WAM | 47.09 ms | 77.48 s |
| RoboCasa | X-WAM | 374.07 ms | 17.36 s |
| RoboCasa | X-WAM-CD | 134.37 ms | 13.04 s |
| RoboCasa | Stream-WAM | 115.98 ms | 9.49 s |

Relative to FastWAM on LIBERO, Stream-WAM reduces chunk latency from 493.0 ms
to 41.0 ms (12.0×); total time falls from 16.31 s to 5.36 s on Long tasks
(3.0×) and from 8.25 s to 3.15 s on Short tasks (2.6×). Relative to
StarWAM-Joint on RoboTwin 2.0, chunk latency decreases from 190.17 ms to
47.09 ms (4.0×) and total time from 110.22 s to 77.48 s (1.4×). Relative to
X-WAM on RoboCasa, chunk latency decreases from 374.07 ms to 115.98 ms (3.2×)
and total time from 17.36 s to 9.49 s (1.8×).

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
