<div align="center">
  <h1>Streaming-WAM</h1>
  <h3>Streaming Your World-Action Model for Real-Time Robot Manipulation.</h3>

  <a href="https://sjtu-deng-lab.github.io/Streaming-WAM/"><img src="https://img.shields.io/badge/Project-Page-087D70?logo=githubpages&logoColor=white" alt="Project Page"></a>
  <a href="https://github.com/SJTU-DENG-Lab/Streaming-WAM"><img src="https://img.shields.io/badge/GitHub-Code-111827?logo=github" alt="GitHub Code"></a>
  <a href="https://huggingface.co/SJTU-DENG-Lab/Streaming-WAM"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Checkpoint-FFD21E" alt="Hugging Face Checkpoint"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6B5BFF" alt="Apache 2.0 License"></a>
</div>

World Action Models (WAMs) jointly generate future visual observations and robot
actions, allowing policies to reason about how the scene may evolve under
interaction. Their iterative generation, however, is often slower than the robot
control cycle: synchronous execution leaves the robot idle during inference,
while naive asynchronous switching can create inconsistency between successive
predictions.

We introduce **Streaming-WAM**, an **action-conditioned streaming framework**
that overlaps WAM inference with robot execution. Shared actions across adjacent
chunks condition future video generation, aligning the predicted visual
trajectory with the motion underway; this action-conditioned future then guides
a consistent action continuation. Streaming-WAM therefore brings streaming into
world prediction rather than treating continuity only as an action space
constraint. We evaluate the method on LIBERO, RoboCasa, and RoboTwin 2.0. On
LIBERO, Streaming-WAM achieves 98.20% success with 41.0 ms chunk latency and
5.36/3.15 s total time on Long/Short tasks, yielding a 12.0× latency reduction
and 3.0×/2.6× total-time speedups over FastWAM.

## Release status

| Asset | Status |
|---|---|
| Streaming-WAM inference and training code | ✅ Available in this repository |
| Accelerated Streaming-WAM runtime | ✅ Available in this repository |
| LIBERO and RoboTwin recipes | ✅ Available in this repository |
| FastWAM-Joint-CD checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/Streaming-WAM) |
| Streaming-WAM checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/Streaming-WAM) |
| Technical report | ⏳ Coming soon |

## Quick start: accelerated Streaming-WAM on LIBERO

The reference environment uses Python 3.10, PyTorch 2.7.1/cu128, and Triton
3.3.1. `pyproject.toml` is the canonical dependency definition.

### 1. Install Streaming-WAM

```bash
git clone https://github.com/SJTU-DENG-Lab/Streaming-WAM.git
cd Streaming-WAM

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

### 3. Launch Streaming-WAM

Place a compatible Streaming-WAM checkpoint and its dataset statistics on disk,
then run:

```bash
PYTHON_BIN=.venv/bin/python \
GPU_IDS=0,1,2,3 \
BACKBONE_PATH="$PWD/checkpoints/Wan2.2-TI2V-5B" \
LIBERO_HOME_PATH="$PWD/third_party/LIBERO" \
CHECKPOINT_PATH=/path/to/ac_stream_checkpoint.pt \
STATS_PATH=/path/to/dataset_stats.json \
  bash examples/libero/scripts/launch_streamingwam_libero_ac_stream_4gpu.sh \
  --ac-stream-accelerated
```

The launcher defaults to one trial for every task in `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`. See the
[LIBERO guide](examples/libero/LIBERO.md) for checkpoint formats, training,
single-task rollout, and evaluation controls.

## Current results

### Task performance

We evaluate FastWAM-Joint and its streaming variant on LIBERO and RoboTwin 2.0, and apply the same streaming design to X-WAM on RoboCasa. All evaluations use four NVIDIA H100 GPUs.

We compare against general purpose robot policies and WAM baselines on task performance, and against WAM baselines on inference efficiency. CD denotes one-step consistency distillation. On LIBERO, we also ablate action conditioning and the slot encoder to assess each component. Best and second best task results are shown in **bold** and <u>underlined</u>, respectively.

#### LIBERO

LIBERO evaluation covers four suites: Long, Spatial, Goal, and Object, with 10 tasks per suite and 50 trials per task. We report average success across suites; Episode Time is reported separately for Long and Short tasks in the efficiency results.

| Method | Long | Spatial | Goal | Object | Average ↑ |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 53.7 | 84.7 | 79.2 | 88.4 | 76.5 |
| π₀ | 85.2 | 96.8 | 95.8 | 98.8 | 94.1 |
| π₀.₅ | 92.4 | <u>98.8</u> | <u>98.0</u> | 98.2 | 96.9 |
| Motus | **97.6** | 96.8 | 96.6 | <u>99.8</u> | 97.7 |
| Fast-WAM | 95.2 | 98.2 | 97.0 | **100.0** | 97.6 |
| FastWAM-Joint-CD | <u>97.20</u> | **99.60** | **98.60** | **100.00** | **98.85** |
| FastWAM-RTC | 58.40 | 76.20 | 77.00 | 83.40 | 73.75 |
| Streaming-WAM (Ours) | 96.60 | <u>98.80</u> | 97.40 | **100.00** | <u>98.20</u> |
| Streaming-WAM w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 |
| Streaming-WAM w/o Slot Encoder | 95.60 | 98.40 | 96.80 | <u>99.80</u> | 97.65 |

#### RoboTwin 2.0

RoboTwin 2.0 evaluates 50 tasks with 100 rollout episodes per task. Clean
reports the easy setting and Random reports the hard domain-randomization
setting.

| Method | Clean ↑ | Random ↑ | Total ↑ |
|---|---:|---:|---:|
| π₀ | 65.92 | 58.40 | 62.20 |
| π₀.₅ | 82.74 | 76.76 | 79.80 |
| Motus | <u>88.66</u> | 87.02 | <u>87.80</u> |
| Motus from WAN2.2 | 77.56 | 77.00 | 77.30 |
| FastWAM-Joint | 86.40 | <u>87.60</u> | 87.00 |
| FastWAM-CD | 86.20 | 85.80 | 86.00 |
| StreamingWAM (Ours) | **90.40** | **90.80** | **90.60** |

#### RoboCasa

RoboCasa follows the standard 24-task protocol, with 50 trials per kitchen manipulation task and average success reported across tasks.

| Method | Average Success ↑ |
|---|---:|
| π₀.₅ | 41.4% |
| π₀-FAST | 61.2% |
| π₀ | 62.5% |
| Cosmos Policy | 67.1% |
| X-WAM | **75.42%** |
| X-WAM-CD | 75.33% |
| Streaming-WAM (Ours) | <u>75.35%</u> |

#### Real robot evaluation

We evaluate standard Joint WAM inference, its distilled 1V10A and 1V2A variants, and Streaming-WAM on the same real robot manipulation task using a single NVIDIA GeForce RTX 5090 at a 25 Hz control frequency. Streaming-WAM reduces Chunk Time to 122.62 ms and completes the rollout in 38 s.

| Method | Chunk Time | Total Time |
|---|---:|---:|
| Joint WAM | 667.1 ms | 68 s |
| Distilled WAM (1V10A) | 402.7 ms | 60 s |
| Distilled WAM (1V2A) | 150.3 ms | 61 s |
| Streaming-WAM (Ours) | **122.62 ms** | **38 s** |

### Inference efficiency

Task success alone does not characterize runtime efficiency. We therefore report Chunk Time, the latency required to prepare the next action chunk, and Episode Time, the duration of a complete rollout, including inference, execution, and replanning.

| Benchmark | Method | Chunk Time | Episode Time |
|---|---|---:|---:|
| LIBERO | FastWAM | 493.0 ms | 16.31 s Long / 8.25 s Short |
| LIBERO | FastWAM-Joint-CD | 114.2 ms | 6.89 s Long / 3.74 s Short |
| LIBERO | FastWAM-RTC | 142.3 ms | 6.23 s Long / 3.20 s Short |
| LIBERO | Streaming-WAM | 41.0 ms | 5.36 s Long / 3.15 s Short |
| LIBERO | Streaming-WAM w/o Action Conditioning | 35.1 ms | 5.20 s Long / 2.92 s Short |
| LIBERO | Streaming-WAM w/o Slot Encoder | 36.3 ms | 5.31 s Long / 3.01 s Short |
| RoboTwin 2.0 | FastWAM-Joint | 652.1 ms | 32.97 s |
| RoboTwin 2.0 | FastWAM-CD | 165.2 ms | 25.21 s |
| RoboTwin 2.0 | StreamingWAM | 54.4 ms | 23.89 s |
| RoboCasa | X-WAM | 374.07 ms | 17.36 s |
| RoboCasa | X-WAM-CD | 134.37 ms | 13.04 s |
| RoboCasa | Streaming-WAM | 115.98 ms | 9.49 s |

Across all three benchmarks, Streaming-WAM reduces both runtime measures while maintaining comparable task success. On LIBERO, it achieves a 12.0× Chunk Time speedup over FastWAM and Episode Time speedups of 3.0× and 2.6× on Long and Short tasks, respectively, with 98.20% average success. On RoboTwin 2.0, relative to FastWAM-Joint, StreamingWAM reduces Chunk Time from 652.1 ms to 54.4 ms and Episode Time from 32.97 s to 23.89 s, while increasing overall success from 87.0 to 90.6. On RoboCasa, relative to X-WAM, Streaming-WAM achieves a 3.2× Chunk Time speedup and a 1.8× Episode Time speedup, with comparable average success (75.35% versus 75.42%).

## Runtime layout

```text
streamingwam/
├── backbone/              # Wan2.2 and Cosmos-Predict2 adapters
├── wam/                   # MoT, Shared-DiT, and Streaming-WAM model wrappers
├── modules/               # DiT, ActionDiT, attention, and scheduler modules
├── inference/             # consistency sampling and Streaming-WAM runtime
├── checkpointing/         # native and FastWAM checkpoint adapters
├── training/              # trainers, losses, and entrypoint
└── data/                  # dataset and text-cache utilities
examples/
├── libero/                # LIBERO recipes, rollout, and launchers
└── robotwin/              # RoboTwin recipes and deployment adapters
```

## Citation

The arXiv entry is not public yet. For now, please cite the
[project page](https://sjtu-deng-lab.github.io/Streaming-WAM/):

```bibtex
@misc{denglab2026streamingwam,
  title        = {Streaming-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/Streaming-WAM/}
}
```

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

Streaming-WAM builds on ideas and open-source work from
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[StarWAM](https://github.com/shaohua-pan/StarWAM),
[X-WAM](https://github.com/sharinka0715/X-WAM),
[StarVLA](https://github.com/starVLA/starVLA),
[DreamZero](https://github.com/dreamzero0/dreamzero),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[Wan2.2](https://github.com/Wan-Video/Wan2.2), and
[Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2).
