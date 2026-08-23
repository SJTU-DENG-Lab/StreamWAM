# StarWAM 仓库结构分析

## 1. 项目定位

StarWAM 是一个面向机器人 World-Action Model（WAM）的研究代码库。它将预训练视频生成/世界模型与动作预测模块组合起来，用于同时建模未来视觉变化和机器人动作。

仓库当前主要支持：

- Wan2.2 和 Cosmos-Predict2 视频骨干；
- MoT、Shared-DiT、Feature-Conditioned 三类 WAM；
- LIBERO 和 RoboTwin 2.0 两套机器人任务；
- LeRobot 格式数据、Flow Matching 训练、分布式训练和闭环动作推理。

该仓库规模约为 78 个 Git tracked 文件、1.23 万行 Python/Shell 代码。整体属于结构清楚、实验链路较完整的研究代码库，但尚未达到成熟通用框架的工程化程度。

## 2. 目录结构

```text
StarWAM/
├── starwam/                     # 核心 Python 包
│   ├── backbone/                # Wan2.2、Cosmos-Predict2 骨干适配
│   ├── wam/                     # 三类 World-Action Model 实现
│   ├── action_model/            # ActionDiT 动作专家构建和初始化
│   ├── modules/                 # MoT、DiT block、scheduler、register token
│   ├── data/                    # LeRobot 数据、视频解码、归一化、文本缓存
│   ├── training/                # Trainer、Flow Matching、loss、metrics
│   ├── eval/                    # 通用闭环策略推理
│   ├── tools/                   # 文本缓存和 ActionDiT 权重预处理工具
│   ├── config.py                # dataclass 配置系统
│   ├── taxonomy.py              # 模型家族和动作表示约束
│   └── builder.py               # 模型、数据集和 Trainer 总装入口
├── examples/
│   ├── libero/                  # LIBERO 配置、训练和 rollout
│   └── robotwin/                # RoboTwin 训练、部署和评测适配
├── configs/
│   ├── accelerate/              # 单机/多机 Accelerate 配置
│   └── deepspeed/               # DeepSpeed ZeRO-2 配置
├── README.md
└── pyproject.toml
```

## 3. 核心执行链路

```text
YAML Recipe
    ↓
load_config()
    ↓
taxonomy.model_family
    ↓
build_framework()
    ├── mot_wam
    ├── shared_dit_wam
    └── feature_conditioned_action_model
    ↓
build_backbone()
    ├── Wan2.2
    └── Cosmos-Predict2
    ↓
build_dataset()
    ├── LeRobotDataset
    └── LeRobotSyntheticDataset
    ↓
StarWAMTrainer
    ↓
Accelerate / DeepSpeed / WandB
    ↓
Checkpoint
    ↓
StarwamPolicy
    ↓
Action Chunk + Replan Queue
```

主要入口如下：

- 配置加载：[`starwam/config.py`](starwam/config.py)
- 模型和数据构建：[`starwam/builder.py`](starwam/builder.py)
- 训练入口：[`starwam/training/train.py`](starwam/training/train.py)
- 通用推理策略：[`starwam/eval/policy.py`](starwam/eval/policy.py)

典型训练命令：

```bash
python -m starwam.training.train \
  --config examples/libero/configs/recipes/starwam_libero_mot_wan22_5b.yaml \
  --override backbone.pretrained_model_id=/path/to/model
```

## 4. 配置系统

配置系统位于 [`starwam/config.py`](starwam/config.py)，采用 dataclass + YAML，没有依赖 Hydra。

顶层配置分为：

- `backbone`：骨干类型和预训练权重路径；
- `framework`：动作维度、chunk size、scheduler 和各模型家族参数；
- `training`：学习率、batch、训练策略、保存和评测参数；
- `data`：数据目录、相机、归一化、文本缓存；
- `inference`：视频和动作推理步数；
- `taxonomy`：模型家族、动作表示和条件方式。

配置加载时会拒绝未知字段，可以较早发现 YAML 拼写错误。命令行还支持 `dot.notation=value` 覆盖配置，例如：

```bash
--override \
  training.batch_size=8 \
  training.wandb_enabled=false \
  'data.dataset_dirs=["/path/to/dataset"]'
```

## 5. 三类 WAM 模型

| 模型家族 | 结构 | 视频损失 | 动作损失 | 主要特点 |
| --- | --- | ---: | ---: | --- |
| `mot_wam` | 视频 DiT 和 ActionDiT 两个 expert，每层混合 Q/K/V | 有 | 有 | 类似 Fast-WAM/Motus，是当前主力路线 |
| `shared_dit_wam` | 视频、动作和状态 token 进入同一个 DiT | 有 | 有 | 参数共享充分，可独立设置视频/动作去噪步数 |
| `feature_conditioned_action_model` | 视频 DiT 提取特征，ActionDiT 单独预测动作 | 无 | 有 | 结构较简单，动作依赖视频特征 |

### 5.1 MoT WAM

实现位于 [`starwam/wam/mot_wam.py`](starwam/wam/mot_wam.py)。

训练过程：

1. VAE 将视频编码成 latent；
2. 分别对视频 latent 和动作加入 Flow Matching noise；
3. 视频 DiT 和 ActionDiT 分别生成 token；
4. `MoT` 在每层联合处理两个 expert 的 Q/K/V；
5. 分别预测视频 velocity 和动作 velocity；
6. 视频、动作 Flow Matching loss 加权求和。

MoT 支持 `first_frame` 和 `full_video` 两种动作对视频的条件方式。实际训练一般需要先用工具从视频 DiT 权重生成 ActionDiT 初始化 payload。

### 5.2 Shared-DiT WAM

实现位于 [`starwam/wam/shared_dit_wam.py`](starwam/wam/shared_dit_wam.py)。

Shared-DiT 将以下信息放进统一 token 空间：

- clean video token；
- noisy video token；
- action register token；
- robot state register token；
- language context。

该路线不需要独立的 ActionDiT 初始化 payload，视频和动作可以采用解耦的去噪步数。从仓库公布的 LIBERO 和 RoboTwin 结果看，这是当前效果最好的路线。

### 5.3 Feature-Conditioned Action Model

实现位于 [`starwam/wam/feature_conditioned_action_model.py`](starwam/wam/feature_conditioned_action_model.py)。

视频骨干负责从 observation、ground-truth video 或 generated video 中提取特征，ActionDiT 根据这些特征预测动作。该路线只计算动作损失，联合世界建模程度低于 MoT 和 Shared-DiT，但结构更直接。

## 6. Backbone 设计

统一骨干接口位于 [`starwam/backbone/base.py`](starwam/backbone/base.py)，主要包括：

- `encode_video()`：视频编码为 VAE latent；
- `decode_latents()`：latent 解码回视频；
- `encode_text()`：语言编码；
- `get_dit()`：获取视频 DiT；
- `get_vae()`：获取 VAE；
- `build_shared_dit_core()`：构建骨干专用 Shared-DiT core。

当前支持的骨干包括：

- Wan2.2-TI2V-5B；
- Wan2.2 14B；
- Cosmos-Predict2-2B-Video2World。

其中 [`starwam/backbone/wan22.py`](starwam/backbone/wan22.py) 超过 2200 行，将 T5、VAE、DiT、checkpoint 加载和适配放在同一个文件中，是仓库当前最大的维护热点。

## 7. 数据层

数据层集中在 [`starwam/data/lerobot.py`](starwam/data/lerobot.py)，使用 LeRobot episode 格式，支持：

- 单相机和多相机输入；
- 水平、垂直相机拼接；
- RoboTwin 三相机专用布局；
- action/state 的 min-max 或 z-score 归一化；
- episode 级训练/验证划分；
- action、image 和 proprio padding mask；
- T5 文本 embedding 缓存；
- synthetic dataset 代码级 smoke test。

模型使用的 sample 大致为：

```python
{
    "video": [B, 3, T, H, W],
    "action": [B, action_horizon, action_dim],
    "proprio": [B, T, state_dim],
    "context": [B, text_len, text_dim],
    "context_mask": ...,
    "action_is_pad": ...,
    "image_is_pad": ...,
}
```

文本缓存和 action/state 统计量生成都使用了文件锁，可避免多卡进程并发写入同一文件。

## 8. 训练系统

训练器位于 [`starwam/training/trainer.py`](starwam/training/trainer.py)，支持：

- Hugging Face Accelerate；
- DeepSpeed ZeRO-2；
- bf16/fp16 mixed precision；
- gradient accumulation；
- full、LoRA、staged 三种训练策略；
- cosine、cosine with minimum LR 和 constant scheduler；
- checkpoint resume；
- model-only resume；
- checkpoint 数量限制和自动清理；
- WandB 日志；
- 小规模 validation。

训练目标统一采用 Flow Matching。MoT 和 Shared-DiT 同时计算视频、动作损失；Feature-Conditioned 模型只计算动作损失。

## 9. 推理与部署

通用推理封装在 [`starwam/eval/policy.py`](starwam/eval/policy.py) 的 `StarwamPolicy` 中，负责：

1. 加载 recipe 和 checkpoint；
2. 编码语言 instruction；
3. 归一化 proprio/state；
4. 生成 action chunk；
5. 对动作进行反归一化；
6. 通过内部队列按 `replan_steps` 执行动作。

LIBERO 的完整 rollout 位于 [`examples/libero/rollout.py`](examples/libero/rollout.py)。

RoboTwin 支持两种部署方式：

- Local：SAPIEN 和 StarWAM 在同一个环境中执行；
- Client/Server：渲染环境通过 socket 请求独立 GPU 推理服务器。

Client/Server 模式适合 SAPIEN/Vulkan 环境与 PyTorch/CUDA 推理环境无法共存的情况。

## 10. Benchmark Recipe

仓库包含 7 个主要 recipe：

### LIBERO

- Wan2.2 MoT；
- Cosmos-Predict2 MoT；
- Wan2.2 Shared-DiT；
- Cosmos-Predict2 Shared-DiT；
- Wan2.2 Feature-Conditioned。

### RoboTwin 2.0

- Wan2.2 MoT；
- Wan2.2 Shared-DiT。

仓库 README/recipe 中记录的代表性结果包括：

- LIBERO Wan2.2 MoT：总体约 97.0%；
- LIBERO Wan2.2 Shared-DiT：总体约 98.2%；
- RoboTwin MoT：总体约 89.48%；
- RoboTwin Shared-DiT：总体约 92.57%。

这些结果是仓库文档中报告的数据，本次分析未在当前机器重新训练或复现。

## 11. 工程质量评价

### 优点

1. 模型家族、骨干、数据和训练器的分层比较清晰；
2. LIBERO 和 RoboTwin 都提供了 recipe、预处理、训练和 rollout；
3. 配置字段严格检查，降低了实验配置静默出错的概率；
4. 对分布式 dataloader、DDP eval、缓存并发和 checkpoint resume 有专门处理；
5. MoT、Shared-DiT、Feature-Conditioned 三条路线可以在统一框架下比较；
6. RoboTwin 提供本地和 client/server 两种部署方式。

### 主要不足和风险

#### 11.1 缺少自动化测试和 CI

仓库没有 `tests/` 和 CI workflow。虽然 `pyproject.toml` 声明了 pytest，但当前没有配置加载、dataset、scheduler、checkpoint 或 synthetic training 的自动测试。

#### 11.2 安装依赖声明不完整

`pyproject.toml` 的核心依赖只有 Torch、PyYAML 和 einops，但实际数据路径还直接使用 NumPy、Pillow、PyArrow 等。较完整依赖只记录在 `examples/libero/requirements.txt` 中。

因此仅执行：

```bash
pip install -e .
```

不一定能覆盖真实训练和数据加载需要的全部依赖。

#### 11.3 Taxonomy 与 Builder 能力不完全一致

taxonomy 层接受 `action_head`、`token_action`、`latent_action` 等动作表示，但当前 builder 对 MoT 和 Shared-DiT 的训练实际上只允许 `token_action`。

#### 11.4 Backbone 文件过大

`wan22.py` 同时承担上游模型实现和 StarWAM adapter 职责。后续升级 Wan 版本或增加单元测试时，拆分成本较高。

#### 11.5 Cosmos 支持仍有边界

部分 Cosmos 路径要求 temporal patch size 为 1，部分 image-context cross-attention 尚未实现，因此不能认为所有 Wan recipe 都可以无条件替换为 Cosmos。

#### 11.6 Recipe 依赖人工路径配置

发布 recipe 中保留大量 `/path/to/...` 占位符。运行前必须设置：

- backbone checkpoint；
- dataset dirs；
- output dir；
- text embedding cache；
- action/state stats；
- MoT ActionDiT init payload。

## 12. 推荐阅读顺序

如果要快速理解或二次开发，建议按以下顺序阅读：

1. [`README.md`](README.md)：了解项目定位；
2. [`starwam/config.py`](starwam/config.py)：掌握完整配置面；
3. [`starwam/taxonomy.py`](starwam/taxonomy.py)：理解模型家族；
4. [`starwam/builder.py`](starwam/builder.py)：理解对象构建链路；
5. [`starwam/wam/mot_wam.py`](starwam/wam/mot_wam.py) 或 [`starwam/wam/shared_dit_wam.py`](starwam/wam/shared_dit_wam.py)：阅读模型主路径；
6. [`starwam/data/lerobot.py`](starwam/data/lerobot.py)：了解数据 sample；
7. [`starwam/training/trainer.py`](starwam/training/trainer.py)：了解训练和分布式行为；
8. [`starwam/eval/policy.py`](starwam/eval/policy.py)：了解部署推理接口；
9. 对应 benchmark 的 YAML recipe 和说明文档。

## 13. 后续改进建议

建议按优先级推进：

1. 增加配置、scheduler、dataset 和 synthetic model smoke test；
2. 添加基础 CI，至少执行 compile、lint、config load 和 CPU test；
3. 补齐 `pyproject.toml` 的数据和推理依赖，或提供明确 extras；
4. 统一 taxonomy 声明与 builder 实际支持范围；
5. 将 `wan22.py` 拆分为 T5、VAE、DiT、loader 和 adapter；
6. 为 Cosmos 的限制增加构建期显式校验；
7. 提供一个可直接执行的 synthetic 最小 recipe；
8. 将 benchmark adapter 与核心 policy 的图像预处理协议进一步标准化。

## 14. 本次静态验证

本次分析完成了以下只读检查：

- 所有 Python 文件通过 `python -m compileall`，未发现语法错误；
- 7 个 YAML recipe 均可被 PyYAML 正常解析；
- 仓库当前位于 `main`，检查前工作区干净且与 `origin/main` 同步；
- 当前 shell 环境没有安装 `torch`，因此没有完成包级 import、synthetic 训练或 GPU forward smoke test；
- 文档中报告的 benchmark 结果没有在本次分析中重新训练验证。

## 15. 总结

StarWAM 的核心价值在于：用统一配置、数据和训练系统组织多种 World-Action Model，并让相同模型家族能够复用不同视频世界模型骨干。

从研究使用角度看，仓库的 MoT 和 Shared-DiT 路线、LIBERO/RoboTwin recipe、分布式训练和闭环评测链路已经较完整；从长期工程维护角度看，当前最需要补充的是自动化测试、完整依赖声明、骨干模块拆分以及 taxonomy/实现能力的一致性。
