"""LIBERO preset validation for StreamWAM recipes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("examples.libero.presets")

MOT_LIBERO_FASTWAM_ALIGNED = "mot_libero_fastwam_aligned"
SHARED_DIT_LIBERO_WAN22_5B = "shared_dit_libero_wan22_5b"
SHARED_DIT_LIBERO_COSMOS_PREDICT2 = "shared_dit_libero_cosmos_predict2"


class PresetValidationError(ValueError):
    """Raised when a StreamWAM preset is configured inconsistently."""


def _taxonomy(config: Any) -> Any:
    return getattr(config, "taxonomy", None)


def _preset(config: Any) -> str | None:
    taxonomy = _taxonomy(config)
    return None if taxonomy is None else getattr(taxonomy, "preset", None)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PresetValidationError(message)


def _warn(condition: bool, message: str) -> None:
    if not condition:
        LOGGER.warning(message)


def validate_preset(config: Any) -> None:
    """Validate known StreamWAM presets.

    Unknown/empty presets are allowed so experimental recipes can still build.
    The Fast-WAM-aligned LIBERO preset is strict for settings that are known to
    affect reproduction quality, and warning-only for local path availability.
    """

    preset = _preset(config)
    if preset in (None, "", "none"):
        return
    if preset == MOT_LIBERO_FASTWAM_ALIGNED:
        validate_mot_libero_fastwam_aligned(config)
        return
    if preset == SHARED_DIT_LIBERO_WAN22_5B:
        validate_shared_dit_libero_wan22_5b(config)
        return
    if preset == SHARED_DIT_LIBERO_COSMOS_PREDICT2:
        validate_shared_dit_libero_cosmos_predict2(config)
        return
    LOGGER.warning("No StreamWAM preset validator registered for %r", preset)


def validate_shared_dit_libero_wan22_5b(config: Any) -> None:
    fw = config.framework
    data = config.data
    backbone = getattr(config, "backbone", None)
    taxonomy = getattr(config, "taxonomy", None)

    _require(fw.type == "shared_dit", "shared_dit_libero_wan22_5b requires framework.type='shared_dit'")
    _require(fw.action_dim == 7, "shared_dit_libero_wan22_5b requires framework.action_dim=7")
    _require(fw.chunk_size == 32, "shared_dit_libero_wan22_5b requires framework.chunk_size=32")
    _require(getattr(fw, "proprio_dim", None) == 8, "shared_dit_libero_wan22_5b requires framework.proprio_dim=8")
    _require(getattr(fw, "max_state_dim", None) == 8, "shared_dit_libero_wan22_5b requires framework.max_state_dim=8")
    _require(getattr(taxonomy, "action_representation", None) == "token_action", "shared_dit_libero_wan22_5b requires token_action")
    _require(getattr(taxonomy, "conditioning", None) == "full_video", "shared_dit_libero_wan22_5b requires taxonomy.conditioning='full_video'")
    _require(getattr(fw, "shared_dit_clean_context", "full_video") == "full_video", "shared_dit_libero_wan22_5b requires full-video clean context")
    _require(getattr(fw, "num_action_per_block", None) == 32, "shared_dit_libero_wan22_5b requires framework.num_action_per_block=32")
    _require(getattr(fw, "num_state_per_block", 1) == 1, "shared_dit_libero_wan22_5b requires framework.num_state_per_block=1")

    if data.dataset_type == "synthetic":
        LOGGER.warning(
            "Skipping LIBERO data-specific preset checks because data.dataset_type='synthetic'. "
            "This is intended only for smoke tests."
        )
    else:
        _require(data.dataset_type == "lerobot", "shared_dit_libero_wan22_5b requires data.dataset_type='lerobot'")
        _require(data.num_frames == 33, "shared_dit_libero_wan22_5b requires data.num_frames=33")
        _require(data.action_freq_ratio == 4, "shared_dit_libero_wan22_5b requires data.action_freq_ratio=4")
        _require(data.normalize_actions is True, "shared_dit_libero_wan22_5b requires data.normalize_actions=true")
        _require(data.action_norm_mode == "minmax", "shared_dit_libero_wan22_5b requires data.action_norm_mode='minmax'")
        _require(getattr(data, "normalize_states", False) is True, "shared_dit_libero_wan22_5b requires data.normalize_states=true")
        _require(getattr(data, "state_norm_mode", "minmax") == "minmax", "shared_dit_libero_wan22_5b requires data.state_norm_mode='minmax'")
        _require(list(data.video_size) == [160, 160], "Use per-camera data.video_size=[160, 160]; two horizontal cameras produce final 160x320")
        _require(data.concat_multi_camera == "horizontal", "shared_dit_libero_wan22_5b requires horizontal camera concatenation")
        _require(len(list(data.video_keys)) == 2, "shared_dit_libero_wan22_5b expects exactly two camera video_keys")

    for name, scheduler in (("video", fw.video_scheduler), ("action", fw.action_scheduler)):
        _require(scheduler.num_train_timesteps == 1000, f"{name} scheduler must use 1000 train timesteps")
        _require(float(scheduler.train_shift) == 5.0, f"{name} scheduler train_shift must be 5.0")
        _require(float(scheduler.infer_shift) == 5.0, f"{name} scheduler infer_shift must be 5.0")

    if backbone is not None:
        _warn(getattr(backbone, "type", None) == "wan22_5b", "shared_dit_libero_wan22_5b expects backbone.type='wan22_5b'")
        _warn(
            getattr(backbone, "load_text_encoder", False) is False,
            "Shared-DiT LIBERO training should use cached T5 embeddings and load_text_encoder=false.",
        )
    _warn(bool(getattr(data, "action_stats_path", None)), "Set data.action_stats_path so distributed runs reuse the same min/max stats.")


def validate_shared_dit_libero_cosmos_predict2(config: Any) -> None:
    fw = config.framework
    data = config.data
    backbone = getattr(config, "backbone", None)
    taxonomy = getattr(config, "taxonomy", None)

    _require(fw.type == "shared_dit", "shared_dit_libero_cosmos_predict2 requires framework.type='shared_dit'")
    _require(fw.action_dim == 7, "shared_dit_libero_cosmos_predict2 requires framework.action_dim=7")
    _require(fw.chunk_size == 32, "shared_dit_libero_cosmos_predict2 requires framework.chunk_size=32")
    _require(getattr(fw, "proprio_dim", None) == 8, "shared_dit_libero_cosmos_predict2 requires framework.proprio_dim=8")
    _require(getattr(fw, "max_state_dim", None) == 8, "shared_dit_libero_cosmos_predict2 requires framework.max_state_dim=8")
    _require(getattr(taxonomy, "action_representation", None) == "token_action", "shared_dit_libero_cosmos_predict2 requires token_action")
    _require(getattr(taxonomy, "conditioning", None) == "full_video", "shared_dit_libero_cosmos_predict2 requires taxonomy.conditioning='full_video'")
    _require(getattr(fw, "shared_dit_clean_context", "full_video") == "full_video", "shared_dit_libero_cosmos_predict2 requires full-video clean context")
    _require(getattr(fw, "num_frame_per_block", None) == 2, "shared_dit_libero_cosmos_predict2 requires framework.num_frame_per_block=2")
    _require(getattr(fw, "num_action_per_block", None) == 32, "shared_dit_libero_cosmos_predict2 requires framework.num_action_per_block=32")
    _require(getattr(fw, "num_state_per_block", 1) == 1, "shared_dit_libero_cosmos_predict2 requires framework.num_state_per_block=1")

    if data.dataset_type == "synthetic":
        LOGGER.warning(
            "Skipping LIBERO data-specific preset checks because data.dataset_type='synthetic'. "
            "This is intended only for smoke tests."
        )
    else:
        _require(data.dataset_type == "lerobot", "shared_dit_libero_cosmos_predict2 requires data.dataset_type='lerobot'")
        _require(data.num_frames == 33, "shared_dit_libero_cosmos_predict2 requires data.num_frames=33")
        _require(data.action_freq_ratio == 4, "shared_dit_libero_cosmos_predict2 requires data.action_freq_ratio=4")
        _require(data.normalize_actions is True, "shared_dit_libero_cosmos_predict2 requires data.normalize_actions=true")
        _require(data.action_norm_mode == "minmax", "shared_dit_libero_cosmos_predict2 requires data.action_norm_mode='minmax'")
        _require(getattr(data, "normalize_states", False) is True, "shared_dit_libero_cosmos_predict2 requires data.normalize_states=true")
        _require(getattr(data, "state_norm_mode", "minmax") == "minmax", "shared_dit_libero_cosmos_predict2 requires data.state_norm_mode='minmax'")
        _require(list(data.video_size) == [320, 320], "Use per-camera data.video_size=[320, 320]; two horizontal cameras produce final 320x640")
        _require(data.concat_multi_camera == "horizontal", "shared_dit_libero_cosmos_predict2 requires horizontal camera concatenation")
        _require(len(list(data.video_keys)) == 2, "shared_dit_libero_cosmos_predict2 expects exactly two camera video_keys")
        _require(getattr(data, "text_cache_encoder_id", None) == "cosmos_predict2_t5", "Cosmos Shared-DiT requires Cosmos T5 text caches")

    for name, scheduler in (("video", fw.video_scheduler), ("action", fw.action_scheduler)):
        _require(scheduler.num_train_timesteps == 1000, f"{name} scheduler must use 1000 train timesteps")
        _require(float(scheduler.train_shift) == 5.0, f"{name} scheduler train_shift must be 5.0")
        _require(float(scheduler.infer_shift) == 5.0, f"{name} scheduler infer_shift must be 5.0")

    if backbone is not None:
        _warn(getattr(backbone, "type", None) == "cosmos_predict2", "shared_dit_libero_cosmos_predict2 expects backbone.type='cosmos_predict2'")
        _warn(
            getattr(backbone, "load_text_encoder", False) is False,
            "Shared-DiT LIBERO training should use cached T5 embeddings and load_text_encoder=false.",
        )
    _warn(bool(getattr(data, "action_stats_path", None)), "Set data.action_stats_path so distributed runs reuse the same min/max stats.")


def validate_mot_libero_fastwam_aligned(config: Any) -> None:
    fw = config.framework
    data = config.data
    training = getattr(config, "training", None)
    backbone = getattr(config, "backbone", None)

    _require(fw.type == "mot", "mot_libero_fastwam_aligned requires framework.type='mot'")
    _require(fw.action_dim == 7, "mot_libero_fastwam_aligned requires framework.action_dim=7")
    _require(fw.chunk_size == 32, "mot_libero_fastwam_aligned requires framework.chunk_size=32")
    _require(getattr(fw, "proprio_dim", None) == 8, "mot_libero_fastwam_aligned requires framework.proprio_dim=8")
    action_video_conditioning = getattr(fw, "action_video_conditioning", "first_frame")
    taxonomy = getattr(config, "taxonomy", None)
    taxonomy_conditioning = getattr(taxonomy, "conditioning", action_video_conditioning)
    _require(
        action_video_conditioning in {"first_frame", "full_video"},
        "framework.action_video_conditioning must be 'first_frame' or 'full_video'",
    )
    _require(
        taxonomy_conditioning == action_video_conditioning,
        "taxonomy.conditioning must match framework.action_video_conditioning",
    )

    if data.dataset_type == "synthetic":
        LOGGER.warning(
            "Skipping LIBERO data-specific preset checks because data.dataset_type='synthetic'. "
            "This is intended only for smoke tests."
        )
    else:
        _require(data.dataset_type == "lerobot", "mot_libero_fastwam_aligned requires data.dataset_type='lerobot'")
        _require(data.num_frames == 33, "mot_libero_fastwam_aligned requires data.num_frames=33")
        _require(data.action_freq_ratio == 4, "mot_libero_fastwam_aligned requires data.action_freq_ratio=4")
        _require(data.normalize_actions is True, "mot_libero_fastwam_aligned requires data.normalize_actions=true")
        _require(data.action_norm_mode == "minmax", "mot_libero_fastwam_aligned requires data.action_norm_mode='minmax'")
        _require(getattr(data, "normalize_states", False) is True, "mot_libero_fastwam_aligned requires data.normalize_states=true")
        _require(getattr(data, "state_norm_mode", "minmax") == "minmax", "mot_libero_fastwam_aligned requires data.state_norm_mode='minmax'")
        _require(
            getattr(data, "state_key", "observation.state") == "observation.state",
            "mot_libero_fastwam_aligned expects proprio from data.state_key='observation.state'",
        )
        _require(list(data.video_size) == [224, 224], "Use per-camera data.video_size=[224, 224]; two horizontal cameras produce final 224x448")
        _require(data.concat_multi_camera == "horizontal", "mot_libero_fastwam_aligned requires horizontal camera concatenation")
        _require(len(list(data.video_keys)) == 2, "mot_libero_fastwam_aligned expects exactly two camera video_keys")

    for name, scheduler in (("video", fw.video_scheduler), ("action", fw.action_scheduler)):
        _require(scheduler.num_train_timesteps == 1000, f"{name} scheduler must use 1000 train timesteps")
        _require(float(scheduler.train_shift) == 5.0, f"{name} scheduler train_shift must be 5.0")
        _require(float(scheduler.infer_shift) == 5.0, f"{name} scheduler infer_shift must be 5.0")

    if training is not None:
        _warn(
            getattr(training, "strategy", "full") == "full",
            "Fast-WAM reference LIBERO training full-trains the DiT/MoT path; "
            "LoRA is useful for cheap smoke tests but is not fully reference-aligned.",
        )
        per_device_batch = int(getattr(training, "batch_size", 0))
        grad_accum = int(getattr(training, "gradient_accumulation_steps", 0))
        _warn(
            per_device_batch * grad_accum == 16,
            "Fast-WAM reference LIBERO recipe uses per-process batch_size=16 and grad_acc=1; "
            "keep batch_size*gradient_accumulation_steps=16 for the same 8-GPU global batch of 128.",
        )
        _warn(
            int(getattr(training, "num_workers", 0)) == 8,
            "Fast-WAM reference LIBERO recipe uses num_workers=8; lower values are safer but slower.",
        )
        _warn(
            getattr(training, "max_steps", None) is None,
            "Fast-WAM reference LIBERO recipe trains by num_epochs=10 with max_steps=null.",
        )
        _warn(
            int(getattr(training, "num_epochs", 0)) == 10,
            "Fast-WAM reference LIBERO recipe uses num_epochs=10.",
        )

    if backbone is not None:
        _warn(
            getattr(backbone, "load_text_encoder", False) is False,
            "Fast-WAM reference LIBERO training uses cached text embeddings and load_text_encoder=false; "
            "set true only if eval/runtime text encoding is required.",
        )
    _warn(
        getattr(fw, "mot_checkpoint_mixed_attn", True) is False,
        "Fast-WAM LIBERO joint 2cam task overrides mot_checkpoint_mixed_attn=false.",
    )

    action_init = getattr(fw, "action_expert_init_from", None)
    _warn(
        bool(action_init),
        "Fast-WAM-aligned LIBERO training should set framework.action_expert_init_from; "
        "random ActionDiT initialization often gives high action loss and low rollout success.",
    )
    if action_init:
        _warn(
            Path(action_init).is_file(),
            f"framework.action_expert_init_from does not exist: {action_init}",
        )

    _warn(
        bool(getattr(data, "action_stats_path", None)),
        "Fast-WAM-aligned LIBERO training should set data.action_stats_path so distributed runs reuse the same min/max stats.",
    )
    _warn(
        int(getattr(data, "text_len", 0)) in (128, 512),
        "Fast-WAM references use cached T5 embeddings with fixed context length; verify data.text_len and cache shape.",
    )

    if data.dataset_type != "synthetic":
        _warn(
            bool(getattr(data, "text_embedding_cache_dir", None)),
            "Set data.text_embedding_cache_dir so trusted Wan2.2 T5 caches can be reused or generated before training.",
        )
        roots = list(getattr(data, "dataset_dirs", None) or ([] if data.root is None else [data.root]))
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                LOGGER.warning("LIBERO dataset root does not exist, cannot verify text cache inputs: %s", root)
