"""StarWAM builders."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

from starwam.config import StarWAMConfig
from starwam.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    collect_lerobot_tasks,
    format_text_prompt,
    save_text_cache,
    text_cache_path,
)
from starwam.taxonomy import (
    FEATURE_CONDITIONED_ACTION_MODEL,
    MOT_WAM,
    SHARED_DIT_WAM,
    TOKEN_ACTION,
    validate_taxonomy,
)

LOGGER = logging.getLogger("starwam.builder")


def build_framework(config: Any, device: str = "cpu", dtype=None):
    """Build a StarWAM taxonomy model."""

    if dtype is None:
        dtype = torch.bfloat16

    model_family = validate_taxonomy(config)

    if model_family == MOT_WAM:
        if config.framework.type != "mot":
            raise ValueError("StarWAM mot_wam currently requires framework.type='mot'")
        action_representation = getattr(getattr(config, "taxonomy", None), "action_representation", TOKEN_ACTION)
        if action_representation != TOKEN_ACTION:
            raise NotImplementedError(
                "StarWAM mot_wam currently trains only taxonomy.action_representation='token_action'."
            )
        from starwam.backbone import build_backbone
        from starwam.wam import MoTWAM, RTCACWAM

        backbone = build_backbone(config.backbone, device=device, dtype=dtype)
        variant = str(getattr(config.framework, "variant", "standard")).strip().lower()
        if variant == "rtc_ac":
            LOGGER.info("Building StarWAM RTC-AC MoT WAM with backbone=%s", config.backbone.type)
            return RTCACWAM(backbone, config.framework, device=device, dtype=dtype)
        if variant != "standard":
            raise ValueError(
                f"Unsupported mot_wam framework.variant={variant!r}; "
                "expected 'standard' or 'rtc_ac'"
            )
        LOGGER.info("Building StarWAM MoT WAM with backbone=%s", config.backbone.type)
        return MoTWAM(backbone, config.framework, device=device, dtype=dtype)

    if model_family == FEATURE_CONDITIONED_ACTION_MODEL:
        if config.framework.type not in {"feature_conditioned", "feature_conditioned_action"}:
            raise ValueError(
                "StarWAM feature_conditioned_action_model requires "
                "framework.type='feature_conditioned'"
            )
        from starwam.backbone import build_backbone
        from starwam.wam import FeatureConditionedActionModel

        backbone = build_backbone(config.backbone, device=device, dtype=dtype)
        LOGGER.info("Building StarWAM feature-conditioned action model with backbone=%s", config.backbone.type)
        return FeatureConditionedActionModel(backbone, config.framework, device=device, dtype=dtype)

    if model_family == SHARED_DIT_WAM:
        if config.framework.type not in {"shared_dit", "shared_dit_wam"}:
            raise ValueError("StarWAM shared_dit_wam requires framework.type='shared_dit'")
        action_representation = getattr(getattr(config, "taxonomy", None), "action_representation", TOKEN_ACTION)
        if action_representation != TOKEN_ACTION:
            raise NotImplementedError(
                "StarWAM shared_dit_wam currently trains only taxonomy.action_representation='token_action'."
            )
        from starwam.backbone import build_backbone
        from starwam.wam import SharedDiTWAM

        backbone = build_backbone(config.backbone, device=device, dtype=dtype, load_dit=False)
        LOGGER.info("Building StarWAM shared-DiT WAM with backbone=%s", config.backbone.type)
        return SharedDiTWAM(backbone, config.framework, device=device, dtype=dtype)

    raise AssertionError(f"Unhandled StarWAM model family: {model_family}")


def _load_or_compute_lerobot_stats(
    stats_path: Path,
    roots: list[str],
    action_key: str,
    state_key: str,
    need_action: bool,
    need_state: bool,
):
    from starwam.data.lerobot import compute_action_stats, compute_state_stats, load_lerobot_stats, save_lerobot_stats

    if stats_path.is_file():
        stats = load_lerobot_stats(stats_path)
        if (not need_action or "action" in stats) and (not need_state or "state" in stats):
            return stats

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stats_path.with_suffix(stats_path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                stats = load_lerobot_stats(stats_path) if stats_path.is_file() else {}
                if need_action and "action" not in stats:
                    stats["action"] = compute_action_stats(roots, action_key=action_key)
                if need_state and "state" not in stats:
                    stats["state"] = compute_state_stats(roots, state_key=state_key)
                save_lerobot_stats(stats, stats_path)
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
            return load_lerobot_stats(stats_path)
        except FileExistsError:
            if stats_path.is_file():
                stats = load_lerobot_stats(stats_path)
                if (not need_action or "action" in stats) and (not need_state or "state" in stats):
                    return stats
            time.sleep(5)


def _ensure_text_caches(config: StarWAMConfig, roots: list[str]) -> None:
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    if not cache_dir:
        raise ValueError("data.text_embedding_cache_dir must be set for LeRobot training")
    text_len = int(getattr(config.data, "text_len", 128))
    prompt_template = getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    encoder_id = getattr(config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
    tasks = collect_lerobot_tasks(roots)
    if not tasks:
        raise ValueError(f"No tasks found in meta/tasks.jsonl under dataset roots: {roots}")
    missing = [task for task in tasks if not text_cache_path(cache_dir, task, text_len, prompt_template, encoder_id).is_file()]
    if not missing:
        return

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".starwam_text_cache.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                missing = [task for task in tasks if not text_cache_path(cache_dir, task, text_len, prompt_template, encoder_id).is_file()]
                if not missing:
                    return
                model_dir = Path(config.backbone.pretrained_model_id)
                device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
                encoder_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
                from starwam.backbone import build_text_encoder

                encoder = build_text_encoder(
                    config.backbone,
                    text_len=text_len,
                    device=device,
                    dtype=encoder_dtype,
                    model_dir=model_dir,
                )
                LOGGER.info("Generating %d missing StarWAM text caches in %s", len(missing), cache_dir)
                for start in range(0, len(missing), 4):
                    batch = missing[start:start + 4]
                    prompts = [format_text_prompt(task, prompt_template) for task in batch]
                    with torch.no_grad():
                        context, mask = encoder.encode(prompts)
                    for i, task in enumerate(batch):
                        save_text_cache(
                            text_cache_path(cache_dir, task, text_len, prompt_template, encoder_id),
                            context[i],
                            mask[i],
                            prompts[i],
                            task,
                        )
                del encoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
        except FileExistsError:
            if all(text_cache_path(cache_dir, task, text_len, prompt_template, encoder_id).is_file() for task in tasks):
                return
            time.sleep(5)


def build_dataset(config: StarWAMConfig, is_training: bool = True, text_dim: int = 4096):
    """Build the StarWAM dataset."""

    from torch.utils.data import ConcatDataset, Subset

    from starwam.data.lerobot import LeRobotDataset, LeRobotSyntheticDataset, compute_action_stats, compute_state_stats

    data_cfg = config.data
    if data_cfg.dataset_type == "synthetic":
        return LeRobotSyntheticDataset(
            config=data_cfg,
            action_dim=config.framework.action_dim,
            chunk_size=config.framework.chunk_size,
            text_dim=text_dim,
            is_training=is_training,
            proprio_dim=getattr(config.framework, "proprio_dim", None),
        )

    if data_cfg.dataset_type != "lerobot":
        raise ValueError(f"Unknown dataset_type: {data_cfg.dataset_type}")

    roots = list(data_cfg.dataset_dirs) if data_cfg.dataset_dirs else ([data_cfg.root] if data_cfg.root else [])
    roots = [str(root) for root in roots if root]
    if not roots:
        raise ValueError("data.root or data.dataset_dirs must be set for dataset_type='lerobot'")

    action_stats = None
    state_stats = None
    normalize_actions = bool(getattr(data_cfg, "normalize_actions", False))
    normalize_states = bool(getattr(data_cfg, "normalize_states", False))
    state_key = getattr(data_cfg, "state_key", "observation.state")
    if normalize_actions or normalize_states:
        shared_stats_path = getattr(data_cfg, "action_stats_path", None)
        state_stats_path = getattr(data_cfg, "state_stats_path", None) or shared_stats_path
        if shared_stats_path and state_stats_path == shared_stats_path:
            stats = _load_or_compute_lerobot_stats(
                Path(shared_stats_path),
                roots,
                data_cfg.action_key,
                state_key,
                normalize_actions,
                normalize_states,
            )
            action_stats = stats.get("action") if normalize_actions else None
            state_stats = stats.get("state") if normalize_states else None
        else:
            if normalize_actions:
                action_stats = (
                    _load_or_compute_lerobot_stats(Path(shared_stats_path), roots, data_cfg.action_key, state_key, True, False)["action"]
                    if shared_stats_path else compute_action_stats(roots, action_key=data_cfg.action_key)
                )
            if normalize_states:
                state_stats = (
                    _load_or_compute_lerobot_stats(Path(state_stats_path), roots, data_cfg.action_key, state_key, False, True)["state"]
                    if state_stats_path else compute_state_stats(roots, state_key=state_key)
                )

    subsets = []
    for root in roots:
        subsets.append(LeRobotDataset(
            root=root,
            video_key=data_cfg.video_key,
            video_keys=getattr(data_cfg, "video_keys", None),
            concat_multi_camera=data_cfg.concat_multi_camera,
            action_key=data_cfg.action_key,
            state_key=state_key,
            num_frames=data_cfg.num_frames,
            chunk_size=config.framework.chunk_size,
            video_size=tuple(data_cfg.video_size),
            text_len=data_cfg.text_len,
            text_dim=text_dim,
            action_freq_ratio=data_cfg.action_freq_ratio,
            normalize_action_stats=action_stats,
            action_norm_mode=data_cfg.action_norm_mode,
            normalize_state_stats=state_stats,
            state_norm_mode=getattr(data_cfg, "state_norm_mode", "minmax"),
            delta_action_dim_mask=getattr(data_cfg, "delta_action_dim_mask", None),
            proprio_dim=getattr(config.framework, "proprio_dim", None),
            text_embedding_cache_dir=getattr(data_cfg, "text_embedding_cache_dir", None),
            text_prompt_template=getattr(data_cfg, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT,
            text_cache_encoder_id=getattr(data_cfg, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID,
        ))

    dataset = subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)
    val_split = float(getattr(data_cfg, "val_split", 0.0) or 0.0)
    if val_split <= 0:
        return dataset

    generator = torch.Generator().manual_seed(int(getattr(data_cfg, "val_split_seed", 42)))
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    val_count = max(1, int(len(dataset) * val_split))
    indices = perm[val_count:] if is_training else perm[:val_count]
    return Subset(dataset, indices)


def build_trainer(model: Any, config: StarWAMConfig):
    """Build the StarWAM trainer."""

    from starwam.training.trainer import StarWAMTrainer

    text_dim = getattr(getattr(model, "backbone", None), "info", None).text_dim if hasattr(model, "backbone") else 4096
    train_dataset = build_dataset(config, is_training=True, text_dim=text_dim)
    val_dataset = None
    if float(getattr(config.data, "val_split", 0.0) or 0.0) > 0:
        val_dataset = build_dataset(config, is_training=False, text_dim=text_dim)
    if config.training.eval_action_num_inference_steps is None:
        config.training.eval_action_num_inference_steps = int(getattr(config.inference, "action_num_inference_steps", config.training.eval_num_inference_steps))
    return StarWAMTrainer(model, train_dataset, val_dataset, config.training)
