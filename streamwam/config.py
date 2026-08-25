"""StreamWAM configuration system. Dataclass-based, no Hydra."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class BackboneConfig:
    """Backbone config -- only user-facing choices. Internal dims auto-inferred."""

    type: str = "wan22_5b"  # wan22_5b | wan22_14b | cosmos_predict2
    pretrained_model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    load_text_encoder: bool = True
    tokenizer_max_len: int = 512


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    train_shift: float = 5.0
    infer_shift: float = 5.0


@dataclass
class FrameworkConfig:
    type: str = "mot"  # StreamWAM first functional path: mot
    variant: str = "standard"  # standard | ac-stream
    ac_stream_architecture: str = "fastwam_stage2_selfatt_z1"
    action_dim: int = 7
    action_gripper_dim: int = -1
    chunk_size: int = 16
    action_expert_type: str = "action_dit"
    action_expert_hidden_dim: int = 1536
    action_expert_num_layers: Optional[int] = None  # None = match backbone
    video_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    action_scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(train_shift=5.0, infer_shift=5.0)
    )
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0
    mot_checkpoint_mixed_attn: bool = True
    action_expert_use_gradient_checkpointing: bool = False
    proprio_dim: Optional[int] = None
    action_video_conditioning: str = "first_frame"  # first_frame | full_video
    # Optional path to a preprocessed ActionDiT init payload
    # (produced by `python -m streamwam.tools.preprocess_action_dit_init`).
    action_expert_init_from: Optional[str] = None
    # Action output head initialization policy. `random` matches Fast-WAM's
    # default payload behavior; `zero` starts the output head at zero; `payload`
    # loads `head_state_dict` from the preprocessing payload.
    action_expert_head_init: Literal["random", "zero", "payload"] = "random"

    # Feature-conditioned action-model settings.
    feature_condition_input: str = "observation"  # observation | ground_truth_video | generated_video
    feature_condition_noise: str = "none"  # none | random_flow_noise | scheduler
    feature_condition_layer: int = -1
    feature_condition_num_tokens: Optional[int] = None
    feature_condition_include_text: bool = True
    feature_condition_include_timestep: bool = True
    feature_condition_train_backbone: bool = False
    feature_condition_pin_first_latent_step: bool = True
    feature_condition_inference_video_steps: Optional[int] = None

    # Shared-DiT/register-token settings used by shared_dit_wam presets.
    num_frame_per_block: int = 2
    num_action_per_block: Optional[int] = None
    num_state_per_block: int = 1
    max_state_dim: Optional[int] = None
    shared_dit_clean_context: str = "full_video"  # none | full_video
    shared_dit_pin_first_latent_step: bool = True
    shared_dit_checkpoint_blocks: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 100
    max_steps: Optional[int] = None
    debug_stop_after_steps: Optional[int] = None  # stop early without changing scheduler horizon
    mixed_precision: str = "bf16"  # no | fp16 | bf16
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 500
    seed: int = 42
    lr_scheduler_type: str = "cosine"  # cosine | cosine_with_min_lr | constant
    min_lr: float = 0.0  # cosine uses learning_rate*0.01; cosine_with_min_lr uses this value
    warmup_steps: Optional[int] = None  # if set, overrides warmup_ratio
    save_total_limit: Optional[int] = None  # if set, keep at most N checkpoints
    resume: Optional[str] = None
    resume_model_only: bool = False  # load only model weights from resume and restart optimizer/scheduler
    resume_lr: Optional[float] = None  # override LR after restoring optimizer/scheduler state
    resume_lr_scheduler_type: Optional[str] = None  # optional scheduler replacement after resume
    strategy: str = "full"  # full | lora | staged
    # Staged strategy: train action heads only for the first
    # ``staged_warmup_steps`` global steps, then unfreeze the backbone /
    # MoT / action_expert and continue with full fine-tuning.
    staged_warmup_steps: int = 1000
    # LoRA hyper-parameters (only used when strategy == "lora").
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Linear layer suffixes to attach LoRA adapters to. Defaults match the
    # Wan2.2 DiT block (self_attn q/k/v/o + ffn.0/ffn.2 + cross_attn q/k/v/o).
    lora_target_modules: list = field(default_factory=lambda: [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        "ffn.0", "ffn.2",
    ])
    num_workers: int = 4
    eval_num_inference_steps: int = 20
    eval_action_num_inference_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    wandb_enabled: bool = False
    wandb_project: str = "streamwam"
    wandb_run_name: Optional[str] = None
    eval_max_samples: int = 4
    eval_compute_video_psnr: bool = False


@dataclass
class DataConfig:
    dataset_type: str = "synthetic"  # synthetic | lerobot
    root: Optional[str] = None  # required when dataset_type == "lerobot"
    dataset_dirs: list = field(default_factory=list)
    num_frames: int = 17
    video_size: list = field(default_factory=lambda: [256, 256])
    video_key: str = "observation.images.cam_high"  # single-camera video key
    video_keys: list = field(default_factory=list)  # multi-camera video keys; falls back to video_key when empty
    concat_multi_camera: str = "horizontal"  # horizontal | vertical | robotwin
    action_key: str = "action"
    state_key: str = "observation.state"
    action_freq_ratio: int = 1
    text_len: int = 512
    text_embedding_cache_dir: Optional[str] = None
    text_prompt_template: str = "A video recorded from a robot's point of view executing the following instruction: {task}"
    text_cache_encoder_id: str = "wan22ti2v5b"
    normalize_actions: bool = False
    action_norm_mode: str = "minmax"  # minmax | zscore
    action_stats_path: Optional[str] = None
    normalize_states: bool = False
    state_norm_mode: str = "minmax"  # minmax | zscore
    state_stats_path: Optional[str] = None
    delta_action_dim_mask: Optional[list] = None
    val_split: float = 0.0  # held-out fraction of episodes for validation
    val_split_seed: int = 42


@dataclass
class InferenceConfig:
    num_inference_steps: int = 20
    action_num_inference_steps: int = 10
    sampling_method: str = "euler"
    replan_steps: int = 5
    seed: int = 42
    ac_stream_horizon: int = 32
    ac_stream_stride: int = 16
    ac_stream_delay: int = 8
    ac_stream_launch_after_steps: int = 8
    ac_stream_block_on_miss: bool = True
    ac_stream_backend: str = "eager"


@dataclass
class TaxonomyConfig:
    package: str = "streamwam"
    model_family: str = "mot_wam"  # feature_conditioned_action_model | mot_wam | shared_dit_wam
    preset: Optional[str] = None
    action_representation: str = "token_action"  # token_action | latent_action | action_head
    conditioning: str = "first_frame"


@dataclass
class StreamWAMConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)


# ---------------------------------------------------------------------------
# YAML / dict loading
# ---------------------------------------------------------------------------

import dataclasses as _dc  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from typing import Any as _Any  # noqa: E402


def _from_dict(cls, raw: _Any):
    """Recursively materialize a dataclass instance from a (possibly nested)
    dict. Unknown keys are rejected to catch typos early."""
    if raw is None:
        return cls()
    if not _dc.is_dataclass(cls):
        return raw
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(raw)}")

    field_map = {f.name: f for f in _dc.fields(cls)}
    unknown = set(raw.keys()) - set(field_map.keys())
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(field_map.keys())}"
        )

    kwargs: dict = {}
    for name, fld in field_map.items():
        if name not in raw:
            continue
        val = raw[name]
        ftype = fld.type
        # Resolve string annotations lazily.
        if isinstance(ftype, str):
            ftype = eval(ftype, globals(), locals())  # noqa: S307
        # Nested dataclass.
        if _dc.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = _from_dict(ftype, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str | _Path) -> StreamWAMConfig:
    """Load a YAML recipe and return a fully-populated :class:`StreamWAMConfig`.

    - Unknown keys raise immediately (no silent typos).
    - Nested sub-configs (`backbone`, `framework`, `training`, `data`,
      `inference`, and the two `*_scheduler`s under framework) are resolved
      recursively.
    - Missing top-level sections fall back to dataclass defaults.

    Example:
        cfg = load_config("examples/libero/configs/recipes/streamwam_libero_mot_wan22_5b.yaml")
        model = build_framework(cfg)
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "load_config requires PyYAML. Install with: pip install pyyaml"
        ) from e

    p = _Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}")
    return _from_dict(StreamWAMConfig, raw)


def config_to_dict(cfg: StreamWAMConfig) -> dict:
    """Serialize a config back to a plain dict (useful for wandb/logging)."""
    return _dc.asdict(cfg)
