"""LIBERO environment rollout for Streaming-WAM policies."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streamingwam.config import load_config  # noqa: E402
from streamingwam.data.lerobot import (  # noqa: E402
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    format_text_prompt,
    iter_task_records,
    load_text_cache,
    save_text_cache,
    text_cache_path,
)
from streamingwam.utils.config_cli import apply_overrides  # noqa: E402
from streamingwam.inference import (  # noqa: E402
    ACStreamController,
    ACStreamPrediction,
    normalize_sampling_method,
    validate_ac_stream_geometry,
)
from examples.libero.timing import GlobalTimingSummary  # noqa: E402
from examples.libero.workload import load_worker_manifest  # noqa: E402
from streamingwam.checkpointing import (  # noqa: E402
    load_inference_checkpoint,
    load_inference_stats,
    prepare_inference_config,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("streamingwam.rollout_libero")

LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
LIBERO_ENV_RESOLUTION = 256
ContextMemoryCache = dict[
    tuple[str, str, str, str],
    tuple[torch.Tensor, torch.Tensor],
]


def _new_context_memory_cache(*, accelerated: bool) -> ContextMemoryCache | None:
    """Keep the GPU-resident prompt cache exclusive to accelerated AC-Stream."""

    return {} if accelerated else None


def _is_placeholder_path(value: Any) -> bool:
    return bool(value) and str(value).startswith("/path/to/")


def _configure_mujoco_runtime(mujoco_gl: str | None) -> None:
    """Configure rendering before importing LIBERO/robosuite/OpenGL."""

    if not mujoco_gl:
        return
    os.environ["MUJOCO_GL"] = mujoco_gl
    os.environ["PYOPENGL_PLATFORM"] = mujoco_gl


def _prepare_runtime_config(config: Any, args: argparse.Namespace) -> Any:
    """Apply explicit inference paths and replace recipe placeholders."""

    if args.backbone_path:
        config.backbone.pretrained_model_id = args.backbone_path
    if args.stats_path:
        config.data.action_stats_path = args.stats_path
        config.data.state_stats_path = args.stats_path

    if args.checkpoint_format == "fastwam":
        default_root = Path("outputs/fastwam_libero_eval")
        if _is_placeholder_path(config.training.output_dir):
            config.training.output_dir = str(default_root)
        cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
        if not cache_dir or _is_placeholder_path(cache_dir):
            config.data.text_embedding_cache_dir = str(default_root / "text_embedding_cache")
        config.data.dataset_dirs = [
            path for path in config.data.dataset_dirs if not _is_placeholder_path(path)
        ]
        if _is_placeholder_path(getattr(config.data, "root", None)):
            config.data.root = None
    return config


def _latest_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def _patch_torch_load_for_libero_init_states() -> None:
    original_load = torch.load
    if getattr(original_load, "_streamingwam_libero_compat", False):
        return

    def load_with_legacy_default(*args: Any, **kwargs: Any):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_default._streamingwam_libero_compat = True  # type: ignore[attr-defined]
    torch.load = load_with_legacy_default  # type: ignore[assignment]


def _add_libero_to_path(libero_home: str | None) -> None:
    if not libero_home:
        return
    path = Path(libero_home).expanduser().resolve()
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    default_config_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "streamingwam" / "libero_config"
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", str(default_config_dir)))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    benchmark_root = path / "libero" / "libero"
    if benchmark_root.is_dir():
        import yaml

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "benchmark_root": str(benchmark_root),
                "bddl_files": str(benchmark_root / "bddl_files"),
                "init_states": str(benchmark_root / "init_files"),
                "datasets": str(path / "libero" / "datasets"),
                "assets": str(benchmark_root / "assets"),
            }, f)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) * float(quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def _extract_proprio(obs: dict[str, Any], proprio_dim: int | None) -> torch.Tensor | None:
    if not proprio_dim or proprio_dim <= 0:
        return None
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    )
    if state.shape[0] < proprio_dim:
        raise ValueError(f"LIBERO proprio dim {state.shape[0]} is smaller than configured proprio_dim={proprio_dim}")
    return torch.as_tensor(state[:proprio_dim], dtype=torch.float32).view(1, proprio_dim)


def _resize_rgb(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return np.asarray(Image.fromarray(image).resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _obs_to_images(obs: dict[str, Any], config: Any) -> dict[str, np.ndarray]:
    primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    size_hw = tuple(int(x) for x in config.data.video_size)
    primary = _resize_rgb(primary, size_hw)
    wrist = _resize_rgb(wrist, size_hw)

    video_keys = list(getattr(config.data, "video_keys", []) or [getattr(config.data, "video_key", "observation.images.image")])
    if len(video_keys) >= 2:
        if getattr(config.data, "concat_multi_camera", "horizontal") == "vertical":
            image = np.concatenate([primary, wrist], axis=0)
        else:
            image = np.concatenate([primary, wrist], axis=1)
    else:
        image = primary
    return {"image": primary, "wrist_image": wrist, "concat": image}


def _obs_to_image(
    obs: dict[str, Any],
    config: Any,
    checkpoint_format: str = "streamingwam",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    images = _obs_to_images(obs, config)
    image = images["concat"]
    tensor = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0)
    if checkpoint_format == "fastwam":
        # FastWAM's released LIBERO evaluator casts uint8 pixels to the model
        # dtype before normalization. Preserve that order for checkpoint parity.
        tensor = tensor.to(
            device=device,
            dtype=dtype or torch.float32,
        )
        tensor = tensor * (2.0 / 255.0) - 1.0
    else:
        # Preserve Streaming-WAM's original float32 preprocessing behavior.
        tensor = tensor.to(dtype=torch.float32)
        tensor = tensor * (2.0 / 255.0) - 1.0
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype or tensor.dtype)
    return tensor, images


def _load_action_stats(config: Any, checkpoint_format: str = "streamingwam") -> dict[str, torch.Tensor] | None:
    if not getattr(config.data, "normalize_actions", False):
        return None
    stats_path = getattr(config.data, "action_stats_path", None)
    if not stats_path:
        raise ValueError("data.normalize_actions=true requires data.action_stats_path for rollout denormalization")
    stats = load_inference_stats(stats_path, checkpoint_format=checkpoint_format)
    if "action" not in stats:
        raise KeyError(f"No action stats found in {stats_path}")
    return stats["action"]


def _load_state_stats(config: Any, checkpoint_format: str = "streamingwam") -> dict[str, torch.Tensor] | None:
    if not getattr(config.data, "normalize_states", False):
        return None
    stats_path = getattr(config.data, "state_stats_path", None) or getattr(config.data, "action_stats_path", None)
    if not stats_path:
        raise ValueError("data.normalize_states=true requires data.state_stats_path or data.action_stats_path")
    stats = load_inference_stats(stats_path, checkpoint_format=checkpoint_format)
    if "state" not in stats:
        raise KeyError(f"No state stats found in {stats_path}")
    return stats["state"]


def _stat_tensor(stats: dict[str, torch.Tensor], key: str, dim: int, dtype: torch.dtype) -> torch.Tensor:
    value = stats[key].to(dtype)
    if value.numel() < dim:
        raise ValueError(f"state stats {key} dim {value.numel()} is smaller than proprio dim {dim}")
    return value[:dim]


def _normalize_state(proprio: torch.Tensor, config: Any, stats: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if stats is None:
        return proprio
    mode = getattr(config.data, "state_norm_mode", "minmax")
    dim = int(proprio.shape[-1])
    if mode == "zscore":
        mean = _stat_tensor(stats, "mean", dim, proprio.dtype)
        std = _stat_tensor(stats, "std", dim, proprio.dtype).clamp_min(1e-6)
        return ((proprio - mean) / std).clamp(-5.0, 5.0)
    if mode != "minmax":
        raise ValueError(f"Unsupported state_norm_mode={mode!r}")
    state_min = _stat_tensor(stats, "min", dim, proprio.dtype)
    state_max = _stat_tensor(stats, "max", dim, proprio.dtype)
    normalized = 2.0 * (proprio - state_min) / (state_max - state_min).clamp_min(1e-6) - 1.0
    return normalized.clamp(-5.0, 5.0)


def _denormalize_action(
    action: torch.Tensor,
    config: Any,
    stats: dict[str, torch.Tensor] | None,
    checkpoint_format: str = "streamingwam",
) -> np.ndarray:
    action = action.detach().float().cpu()
    if stats is None:
        denorm = action
    elif config.data.action_norm_mode == "zscore":
        denorm = action * stats["std"].clamp_min(1e-6) + stats["mean"]
    elif config.data.action_norm_mode == "minmax":
        action_min = stats["min"]
        action_max = stats["max"]
        normalized = action if checkpoint_format == "fastwam" else action.clamp(-1.0, 1.0)
        denorm = (normalized + 1.0) * 0.5 * (action_max - action_min).clamp_min(1e-6) + action_min
    else:
        raise ValueError(f"Unsupported action_norm_mode={config.data.action_norm_mode!r}")

    out = denorm.numpy()
    if out.ndim == 3:
        out = out[0]
    if out.shape[-1] != 7:
        raise ValueError(f"LIBERO rollout expects 7-D actions, got shape={out.shape}")

    gripper_open = out[..., -1] > 0.5
    out[..., -1] = np.where(gripper_open, -1.0, 1.0)
    return out.astype(np.float32)


def _build_task_cache_index(config: Any) -> dict[str, Path]:
    index: dict[str, Path] = {}
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    if not cache_dir:
        return index
    roots = list(config.data.dataset_dirs) if config.data.dataset_dirs else ([config.data.root] if config.data.root else [])
    prompt_template = getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    encoder_id = getattr(config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
    context_len = int(getattr(config.data, "text_len", 128))
    for root in roots:
        for record in iter_task_records(root):
            task = str(record["task"])
            cache = text_cache_path(cache_dir, task, context_len, prompt_template, encoder_id)
            if cache.is_file():
                index[task] = cache
    return index


def _prepare_context_for_checkpoint(
    context: torch.Tensor,
    mask: torch.Tensor,
    checkpoint_format: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if checkpoint_format != "fastwam":
        return context, mask

    # FastWAM.encode_prompt zeroes padded embeddings but deliberately exposes
    # all tokenizer positions to cross-attention. The projection MLP has bias,
    # so masking those zero rows is not numerically equivalent.
    context = context.clone()
    context[~mask] = 0
    return context, torch.ones_like(mask, dtype=torch.bool)


def _load_context(
    task_description: str,
    config: Any,
    task_cache: dict[str, Path],
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_format: str = "streamingwam",
    memory_cache: dict[
        tuple[str, str, str, str],
        tuple[torch.Tensor, torch.Tensor],
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory_key = (
        task_description,
        checkpoint_format,
        str(device),
        str(dtype),
    )
    if memory_cache is not None and memory_key in memory_cache:
        return memory_cache[memory_key]
    text_len = int(getattr(config.data, "text_len", 128))
    prompt_template = getattr(config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
    encoder_id = getattr(config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
    cache_dir = getattr(config.data, "text_embedding_cache_dir", None)
    cache = task_cache.get(task_description)
    if cache is None and cache_dir:
        cache = text_cache_path(cache_dir, task_description, text_len, prompt_template, encoder_id)
    text_dim = int(getattr(getattr(getattr(model, "backbone", None), "info", None), "text_dim", 4096))
    if cache is not None and cache.is_file():
        context, mask = load_text_cache(cache, text_len, text_dim)
        context, mask = _prepare_context_for_checkpoint(
            context,
            mask,
            checkpoint_format,
        )
        loaded = (
            context.unsqueeze(0).to(device=device, dtype=dtype),
            mask.unsqueeze(0).to(device=device),
        )
        if memory_cache is not None:
            memory_cache[memory_key] = loaded
        return loaded

    if not cache_dir:
        raise KeyError("data.text_embedding_cache_dir must be set for rollout text conditioning")
    from streamingwam.backbone.wan22 import Wan22TextEncoder

    model_dir = Path(config.backbone.pretrained_model_id)
    encoder = Wan22TextEncoder(
        ckpt_path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(model_dir / "google" / "umt5-xxl"),
        text_len=text_len,
        device=str(device),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    prompt = format_text_prompt(task_description, prompt_template)
    with torch.no_grad():
        context, mask = encoder.encode([prompt])
    cache = text_cache_path(cache_dir, task_description, text_len, prompt_template, encoder_id)
    save_text_cache(cache, context[0], mask[0], prompt, task_description)
    task_cache[task_description] = cache
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    context, mask = _prepare_context_for_checkpoint(
        context[0],
        mask[0],
        checkpoint_format,
    )
    context = context.unsqueeze(0)
    mask = mask.unsqueeze(0)
    loaded = (
        context.to(device=device, dtype=dtype),
        mask.to(device=device),
    )
    if memory_cache is not None:
        memory_cache[memory_key] = loaded
    return loaded


def _max_steps(task_suite_name: str) -> int:
    return {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }.get(task_suite_name, 400)


def _safe_task_name(task_description: str) -> str:
    return "_".join(task_description.lower().replace(".", " ").split())[:80]


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 30) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def _sampled_video_frame_count(config: Any) -> int:
    action_grid_frames = int(config.data.num_frames)
    action_freq_ratio = max(1, int(getattr(config.data, "action_freq_ratio", 1)))
    return len(range(0, action_grid_frames, action_freq_ratio))


def _uses_decoupled_action_steps(config: Any) -> bool:
    return str(getattr(config.framework, "type", "")) == "shared_dit"


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_inference_args(config: Any, args: argparse.Namespace) -> None:
    if args.num_inference_steps is None:
        args.num_inference_steps = int(getattr(config.inference, "num_inference_steps", 8))
    if args.action_num_inference_steps is None:
        args.action_num_inference_steps = int(
            getattr(config.inference, "action_num_inference_steps", args.num_inference_steps)
        )
    if args.sampling_method is None:
        args.sampling_method = str(getattr(config.inference, "sampling_method", "euler"))
    args.sampling_method = normalize_sampling_method(args.sampling_method)
    if args.replan_steps is None:
        args.replan_steps = int(getattr(config.inference, "replan_steps", 5))
    if args.ac_stream_accelerated and args.sampling_method != "ac-stream":
        raise ValueError(
            "--ac-stream-accelerated requires sampling_method='ac-stream'"
        )
    if args.ac_stream_accelerated and not str(args.device).startswith("cuda"):
        raise ValueError("--ac-stream-accelerated requires a CUDA device")
    if args.sampling_method == "consistency" and args.replan_steps != 16:
        raise ValueError(
            f"Joint CD consistency rollout requires replan_steps=16, got {args.replan_steps}"
        )
    if args.sampling_method == "ac-stream":
        if str(getattr(config.framework, "variant", "standard")) != "ac-stream":
            raise ValueError("sampling_method='ac-stream' requires framework.variant='ac-stream'")
        backend = str(getattr(config.inference, "ac_stream_backend", "eager")).strip().lower()
        if backend != "eager":
            raise ValueError(
                f"This AC-Stream phase supports ac_stream_backend='eager', got {backend!r}"
            )
        validate_ac_stream_geometry(
            action_horizon=int(config.framework.chunk_size),
            stride=int(getattr(config.inference, "ac_stream_stride", 16)),
            delay=int(getattr(config.inference, "ac_stream_delay", 8)),
            launch_after_steps=int(
                getattr(config.inference, "ac_stream_launch_after_steps", 8)
            ),
            num_video_frames=_sampled_video_frame_count(config),
            temporal_compress=4,
            num_inference_steps=args.num_inference_steps,
        )
        if args.replan_steps != 16:
            raise ValueError(
                f"AC-Stream requires replan_steps=16, got {args.replan_steps}"
            )
    if not _uses_decoupled_action_steps(config):
        args.action_num_inference_steps = args.num_inference_steps


def _predict_action_chunk(
    model: torch.nn.Module,
    obs: dict[str, Any],
    task_description: str,
    config: Any,
    task_cache: dict[str, Path],
    action_stats: dict[str, torch.Tensor] | None,
    state_stats: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
    num_inference_steps: int,
    action_num_inference_steps: int,
    sampling_method: str,
    checkpoint_format: str,
    seed: int | None,
    ac_stream_prev_action_chunk: torch.Tensor | None = None,
    ac_stream_inference_delay: int = 0,
    context_memory_cache: ContextMemoryCache | None = None,
) -> tuple[
    np.ndarray,
    torch.Tensor,
    dict[str, np.ndarray],
    float,
    float,
    int,
    int,
]:
    # Exclude model loading or other previously queued CUDA work from the
    # first chunk's communication measurement.
    _synchronize_device(device)
    communication_start = time.perf_counter_ns()
    image, images = _obs_to_image(
        obs,
        config,
        checkpoint_format=checkpoint_format,
        device=device,
        dtype=dtype,
    )
    context, context_mask = _load_context(
        task_description,
        config,
        task_cache,
        model,
        device,
        dtype,
        checkpoint_format=checkpoint_format,
        memory_cache=context_memory_cache,
    )
    proprio = _extract_proprio(obs, getattr(config.framework, "proprio_dim", None))
    if proprio is not None:
        proprio = _normalize_state(proprio, config, state_stats).to(device=device, dtype=dtype)
    _synchronize_device(device)
    communication_before_ms = (time.perf_counter_ns() - communication_start) / 1e6

    infer_kwargs: dict[str, Any] = {
        "proprio": proprio,
        "num_video_frames": _sampled_video_frame_count(config),
        "sampling_method": sampling_method,
    }
    if checkpoint_format == "fastwam":
        infer_kwargs["rand_device"] = "cpu"
    if _uses_decoupled_action_steps(config):
        infer_kwargs["action_num_inference_steps"] = action_num_inference_steps
    if sampling_method == "ac-stream":
        infer_kwargs.update(
            {
                "ac_stream_prev_action_chunk": ac_stream_prev_action_chunk,
                "ac_stream_inference_delay": int(ac_stream_inference_delay),
                "ac_stream_stride": int(getattr(config.inference, "ac_stream_stride", 16)),
                "ac_stream_delay": int(getattr(config.inference, "ac_stream_delay", 8)),
                "ac_stream_launch_after_steps": int(
                    getattr(config.inference, "ac_stream_launch_after_steps", 8)
                ),
                "ac_stream_context_key": task_description,
            }
        )

    inference_start = time.perf_counter_ns()
    pred = model.infer_action(
        input_image=image,
        context=context,
        context_mask=context_mask,
        action_horizon=int(config.framework.chunk_size),
        num_inference_steps=num_inference_steps,
        seed=seed,
        **infer_kwargs,
    )
    _synchronize_device(device)
    inference_completed = time.perf_counter_ns()
    inference_ms = (inference_completed - inference_start) / 1e6

    communication_after_start = time.perf_counter_ns()
    model_action = pred.detach().float().cpu()
    action = _denormalize_action(
        model_action,
        config,
        action_stats,
        checkpoint_format=checkpoint_format,
    )
    _synchronize_device(device)
    communication_after_ms = (time.perf_counter_ns() - communication_after_start) / 1e6
    if model_action.ndim == 3:
        model_action = model_action[0]
    return (
        action,
        model_action,
        images,
        communication_before_ms + communication_after_ms,
        inference_ms,
        inference_start,
        inference_completed,
    )


def _rollout_episode(
    env: Any,
    initial_state: Any,
    task_description: str,
    model: torch.nn.Module,
    config: Any,
    task_cache: dict[str, Path],
    context_memory_cache: ContextMemoryCache | None,
    prewarmed_tasks: set[str],
    action_stats: dict[str, torch.Tensor] | None,
    state_stats: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    episode_idx: int,
    timing: GlobalTimingSummary,
    task_suite_name: str,
) -> tuple[bool, list[np.ndarray]]:
    inference_seed = (
        args.seed
        if args.fixed_seed
        else (None if args.seed is None else args.seed + episode_idx)
    )
    task_key = f"{task_suite_name}/{task_description}"

    def predict(observation: Any):
        return _predict_action_chunk(
            model=model,
            obs=observation,
            task_description=task_description,
            config=config,
            task_cache=task_cache,
            action_stats=action_stats,
            state_stats=state_stats,
            device=device,
            dtype=dtype,
            num_inference_steps=args.num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            sampling_method=args.sampling_method,
            checkpoint_format=args.checkpoint_format,
            seed=inference_seed,
            context_memory_cache=context_memory_cache,
        )

    if args.checkpoint_format == "fastwam":
        _prewarm_sync_if_needed(
            task_key=task_key,
            prewarmed_tasks=prewarmed_tasks,
            env=env,
            initial_state=initial_state,
            num_steps_wait=args.num_steps_wait,
            predict=predict,
        )
    obs = _reset_and_stabilize(
        env,
        initial_state,
        num_steps_wait=args.num_steps_wait,
    )
    episode_start_ns = time.perf_counter_ns()
    pending_actions: list[list[float]] = []
    frames: list[np.ndarray] = []
    done = False
    active_chunk = None
    max_steps = int(args.max_steps or _max_steps(task_suite_name))
    episode_completed_ns = episode_start_ns

    for _ in range(max_steps):
        frames.append(_obs_to_images(obs, config)["concat"])

        if not pending_actions:
            (
                action_chunk,
                _,
                _,
                communication_ms,
                inference_ms,
                _,
                _,
            ) = predict(obs)
            active_chunk = timing.add_chunk(
                communication_ms=communication_ms,
                inference_ms=inference_ms,
            )
            pending_actions = action_chunk[: args.replan_steps].tolist()

        action_execution_start = time.perf_counter_ns()
        obs, _, done, _ = env.step(pending_actions.pop(0))
        action_execution_completed = time.perf_counter_ns()
        episode_completed_ns = action_execution_completed
        if active_chunk is None:
            raise RuntimeError("Action execution has no active generated chunk")
        active_chunk.add_action_execution(
            (action_execution_completed - action_execution_start) / 1e6
        )
        if done:
            break
    timing.add_episode_wall((episode_completed_ns - episode_start_ns) / 1e6)
    return bool(done), frames


def _reset_and_stabilize(
    env: Any,
    initial_state: Any,
    *,
    num_steps_wait: int,
) -> Any:
    """Restore an episode and complete simulator stabilization before timing."""

    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(int(num_steps_wait)):
        observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return observation


def _prewarm_sync_if_needed(
    *,
    task_key: str,
    prewarmed_tasks: set[str],
    env: Any,
    initial_state: Any,
    num_steps_wait: int,
    predict: Callable[[Any], Any],
) -> None:
    """Run one task-specific synchronous policy call outside evaluation timing."""

    if task_key in prewarmed_tasks:
        return
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(int(num_steps_wait)):
        observation, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            break
    predict(observation)
    prewarmed_tasks.add(task_key)


def _prewarm_ac_stream_if_needed(
    *,
    task_key: str,
    prewarmed_tasks: set[str],
    env: Any,
    initial_state: Any,
    num_steps_wait: int,
    model: torch.nn.Module,
    predict: Callable[[Any, torch.Tensor | None, int], ACStreamPrediction],
    accelerated: bool,
) -> None:
    """Prepare task-specific D0/D8 paths without entering evaluation timing."""

    if task_key in prewarmed_tasks:
        return
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(int(num_steps_wait)):
        observation, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            break
    d0 = predict(observation, None, 0)
    mark_prewarmed = accelerated and not bool(
        getattr(model, "ac_stream_prewarm_complete", False)
    )
    if mark_prewarmed:
        model.mark_ac_stream_prewarmed(0)
    predict(observation, d0.model_actions, 8)
    if mark_prewarmed:
        model.mark_ac_stream_prewarmed(8)
    prewarmed_tasks.add(task_key)


def _rollout_ac_stream_episode(
    env: Any,
    initial_state: Any,
    task_description: str,
    model: torch.nn.Module,
    config: Any,
    task_cache: dict[str, Path],
    context_memory_cache: ContextMemoryCache | None,
    prewarmed_tasks: set[str],
    action_stats: dict[str, torch.Tensor] | None,
    state_stats: dict[str, torch.Tensor] | None,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    episode_idx: int,
    timing: GlobalTimingSummary,
    task_suite_name: str,
) -> tuple[bool, list[np.ndarray]]:
    """Run the H32/s16/d8 AC-Stream controller against one LIBERO episode."""

    frames: list[np.ndarray] = []
    done = False
    active_chunk = None
    controller: ACStreamController | None = None
    max_steps = int(args.max_steps or _max_steps(task_suite_name))
    inference_seed = (
        args.seed
        if args.fixed_seed
        else (None if args.seed is None else args.seed + episode_idx)
    )

    def predict(
        observation: Any,
        previous_target: torch.Tensor | None,
        delay: int,
    ) -> ACStreamPrediction:
        (
            env_actions,
            model_actions,
            _,
            communication_ms,
            inference_ms,
            inference_started_ns,
            inference_completed_ns,
        ) = _predict_action_chunk(
            model=model,
            obs=observation,
            task_description=task_description,
            config=config,
            task_cache=task_cache,
            action_stats=action_stats,
            state_stats=state_stats,
            device=device,
            dtype=dtype,
            num_inference_steps=args.num_inference_steps,
            action_num_inference_steps=args.action_num_inference_steps,
            sampling_method="ac-stream",
            checkpoint_format=args.checkpoint_format,
            seed=inference_seed,
            ac_stream_prev_action_chunk=previous_target,
            ac_stream_inference_delay=delay,
            context_memory_cache=context_memory_cache,
        )
        return ACStreamPrediction(
            env_actions=env_actions,
            model_actions=model_actions,
            communication_ms=communication_ms,
            inference_ms=inference_ms,
            inference_started_ns=inference_started_ns,
            inference_completed_ns=inference_completed_ns,
        )

    if args.checkpoint_format == "fastwam":
        _prewarm_ac_stream_if_needed(
            task_key=f"{task_suite_name}/{task_description}",
            prewarmed_tasks=prewarmed_tasks,
            env=env,
            initial_state=initial_state,
            num_steps_wait=args.num_steps_wait,
            model=model,
            predict=predict,
            accelerated=args.ac_stream_accelerated,
        )
    obs = _reset_and_stabilize(
        env,
        initial_state,
        num_steps_wait=args.num_steps_wait,
    )
    episode_start_ns = time.perf_counter_ns()
    episode_completed_ns = episode_start_ns

    try:
        for _ in range(max_steps):
            frames.append(_obs_to_images(obs, config)["concat"])
            if controller is None:
                controller = ACStreamController(
                    predict,
                    block_on_miss=bool(
                        getattr(config.inference, "ac_stream_block_on_miss", True)
                    ),
                )
                controller.start_episode(obs)

            action = controller.next_action(obs)
            for record in controller.pop_overlap_records():
                timing.add_ac_stream_overlap(record)
            for prediction in controller.pop_installed_predictions():
                active_chunk = timing.add_chunk(
                    communication_ms=prediction.communication_ms,
                    inference_ms=prediction.inference_ms,
                )
            if active_chunk is None:
                raise RuntimeError("AC-Stream action execution has no installed chunk")
            action_execution_start = time.perf_counter_ns()
            obs, _, done, _ = env.step(action.tolist())
            action_execution_completed = time.perf_counter_ns()
            episode_completed_ns = action_execution_completed
            active_chunk.add_action_execution(
                (action_execution_completed - action_execution_start) / 1e6
            )
            controller.mark_action_executed(
                started_ns=action_execution_start,
                completed_ns=action_execution_completed,
            )
            if done:
                break
    finally:
        if controller is not None:
            pending_prediction = controller.close()
            if pending_prediction is not None:
                timing.add_chunk(
                    communication_ms=pending_prediction.communication_ms,
                    inference_ms=pending_prediction.inference_ms,
                )
            for record in controller.pop_overlap_records():
                timing.add_ac_stream_overlap(record)
    timing.add_episode_wall((episode_completed_ns - episode_start_ns) / 1e6)
    return bool(done), frames


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Streaming-WAM policy rollout in LIBERO environments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-format",
        choices=("streamingwam", "fastwam"),
        default="streamingwam",
        help="Source checkpoint/stats format (default: streamingwam).",
    )
    parser.add_argument("--backbone-path", default=None, help="Local Wan2.2 backbone directory.")
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Action and state statistics JSON used for inference.",
    )
    parser.add_argument(
        "--mujoco-gl",
        choices=("osmesa", "egl", "glfw"),
        default=None,
        help="Mujoco/OpenGL rendering backend configured before LIBERO import.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--libero-home", default=os.environ.get("LIBERO_HOME"))
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=None, help="Run one task id; default runs all tasks in suite")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--sampling-method",
        choices=("euler", "consistency", "ac-stream"),
        default=None,
        help="Denoising update rule; defaults to inference.sampling_method from the recipe.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-seed", action="store_true", help="Use the same diffusion seed for every episode")
    parser.add_argument(
        "--ac-stream-accelerated",
        action="store_true",
        help="Enable strict compiled/cached AC-Stream inference with D0/D8 prewarm.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--work-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--suppress-final-summary", action="store_true", help=argparse.SUPPRESS)
    return parser


def _build_evaluation_assignments(
    args: argparse.Namespace,
    benchmark_dict: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Resolve normal CLI selection or an explicit manager worker manifest."""

    if args.work_manifest:
        manifest = load_worker_manifest(args.work_manifest)
        return list(manifest["assignments"]), manifest

    task_suite = benchmark_dict[args.task_suite_name]()
    task_ids = [args.task_id] if args.task_id is not None else list(range(task_suite.n_tasks))
    assignments = [
        {
            "task_suite_name": args.task_suite_name,
            "task_id": task_id,
            "trial_ids": list(range(args.num_trials)),
        }
        for task_id in task_ids
    ]
    return assignments, None


def main() -> None:
    command_start = time.perf_counter_ns()
    args = _build_arg_parser().parse_args()
    timing = GlobalTimingSummary()

    _configure_mujoco_runtime(args.mujoco_gl)
    _add_libero_to_path(args.libero_home)
    _patch_torch_load_for_libero_init_states()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import LIBERO. Set LIBERO_HOME to the LIBERO source root and make sure its env deps are installed."
        ) from exc

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    config = _prepare_runtime_config(config, args)
    config = prepare_inference_config(config, args.checkpoint_format)

    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(config.training.output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.training.output_dir) / "libero_rollout" / checkpoint.name / args.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mp = (config.training.mixed_precision or "no").lower()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
    if device.type == "cpu":
        dtype = torch.float32

    _resolve_inference_args(config, args)
    if args.sampling_method == "ac-stream":
        timing.enable_ac_stream()

    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Checkpoint format: %s", args.checkpoint_format)
    logger.info("Sampling method: %s", args.sampling_method)
    logger.info("Output: %s", output_dir)
    logger.info("Task suite: %s", args.task_suite_name if not args.work_manifest else "manifest")
    logger.info(
        "Inference steps: num=%d action=%d decoupled_action_steps=%s sampled_video_frames=%d",
        args.num_inference_steps,
        args.action_num_inference_steps,
        _uses_decoupled_action_steps(config),
        _sampled_video_frame_count(config),
    )

    from streamingwam import build_framework

    model = build_framework(config, device=str(device), dtype=dtype).to(device)
    meta = load_inference_checkpoint(model, checkpoint, checkpoint_format=args.checkpoint_format)
    model.eval()
    logger.info("Loaded checkpoint metadata: %s", meta)
    if args.ac_stream_accelerated:
        enable_acceleration = getattr(model, "enable_ac_stream_acceleration", None)
        if not callable(enable_acceleration):
            raise TypeError("Selected model does not support AC-Stream acceleration")
        enable_acceleration()
        logger.info("AC-Stream acceleration enabled; D0/D8 prewarm pending")

    task_cache = _build_task_cache_index(config)
    context_memory_cache = _new_context_memory_cache(
        accelerated=args.ac_stream_accelerated,
    )
    prewarmed_tasks: set[str] = set()
    logger.info("Loaded %d task text embeddings from recipe dataset dirs", len(task_cache))
    action_stats = _load_action_stats(config, args.checkpoint_format)
    state_stats = _load_state_stats(config, args.checkpoint_format)

    benchmark_dict = benchmark.get_benchmark_dict()
    assignments, worker_manifest = _build_evaluation_assignments(args, benchmark_dict)
    suite_cache: dict[str, Any] = {}

    all_results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "task_suite_name": args.task_suite_name if worker_manifest is None else None,
        "num_trials": args.num_trials if worker_manifest is None else worker_manifest["workload_size"],
        "task_results": {},
    }
    if worker_manifest is not None:
        all_results["worker_id"] = args.worker_id or worker_manifest.get("worker_id")
        all_results["work_manifest"] = str(args.work_manifest)
    total_success = 0
    total_trials = 0

    for assignment in assignments:
        task_suite_name = str(assignment["task_suite_name"])
        task_id = int(assignment["task_id"])
        trial_ids = [int(trial_id) for trial_id in assignment["trial_ids"]]
        if task_suite_name not in suite_cache:
            suite_cache[task_suite_name] = benchmark_dict[task_suite_name]()
        task_suite = suite_cache[task_suite_name]
        timing.task_count += 1
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_success = 0
        task_records = []
        logger.info(
            "Task %s/%s/%s: %s",
            task_suite_name,
            task_id,
            task_suite.n_tasks - 1,
            task_description,
        )
        try:
            for trial_idx in trial_ids:
                if trial_idx >= len(initial_states):
                    raise IndexError(
                        f"Trial {trial_idx} is unavailable for {task_suite_name} task {task_id}; "
                        f"only {len(initial_states)} initial states exist"
                    )
                rollout_episode = (
                    _rollout_ac_stream_episode
                    if args.sampling_method == "ac-stream"
                    else _rollout_episode
                )
                episode_wall_count = len(timing.episode_wall_ms)
                success, frames = rollout_episode(
                    env=env,
                    initial_state=initial_states[trial_idx],
                    task_description=task_description,
                    model=model,
                    config=config,
                    task_cache=task_cache,
                    context_memory_cache=context_memory_cache,
                    prewarmed_tasks=prewarmed_tasks,
                    action_stats=action_stats,
                    state_stats=state_stats,
                    device=device,
                    dtype=dtype,
                    args=args,
                    episode_idx=trial_idx,
                    timing=timing,
                    task_suite_name=task_suite_name,
                )
                if len(timing.episode_wall_ms) != episode_wall_count + 1:
                    raise RuntimeError("Rollout did not record exactly one episode wall time")
                episode_wall_ms = timing.episode_wall_ms[-1]
                timing.trial_count += 1
                task_success += int(success)
                total_success += int(success)
                total_trials += 1
                record = {
                    "trial": trial_idx,
                    "success": bool(success),
                    "episode_wall_ms": episode_wall_ms,
                }
                task_records.append(record)
                logger.info("Task %d trial %d success=%s", task_id, trial_idx, success)
                if args.save_video:
                    suffix = "success" if success else "failure"
                    video_dir = output_dir / "videos"
                    if worker_manifest is not None:
                        video_dir = video_dir / task_suite_name
                    video_path = video_dir / f"task{task_id:02d}_trial{trial_idx:02d}_{suffix}_{_safe_task_name(task_description)}.mp4"
                    _save_video(video_path, frames)
        finally:
            env.close()

        task_rate = task_success / max(len(trial_ids), 1)
        result_key = str(task_id) if worker_manifest is None else f"{task_suite_name}/{task_id}"
        all_results["task_results"][result_key] = {
            "task_suite_name": task_suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "successes": task_success,
            "trials": len(trial_ids),
            "success_rate": task_rate,
            "episodes": task_records,
        }
        logger.info(
            "Task %s/%d success_rate=%.4f (%d/%d)",
            task_suite_name,
            task_id,
            task_rate,
            task_success,
            len(trial_ids),
        )

    all_results["total_successes"] = total_success
    all_results["total_trials"] = total_trials
    all_results["success_rate"] = total_success / max(total_trials, 1)
    if args.sampling_method == "ac-stream":
        status_fn = getattr(model, "ac_stream_acceleration_status", None)
        if not callable(status_fn):
            raise TypeError("AC-Stream model does not expose acceleration status")
        acceleration_status = status_fn()
        if args.ac_stream_accelerated and not (
            acceleration_status.get("compile_active")
            and acceleration_status.get("prewarmed_d0")
            and acceleration_status.get("prewarmed_d8")
        ):
            raise RuntimeError(
                "AC-Stream accelerated evaluation ended without active compile "
                f"and complete D0/D8 prewarm: {acceleration_status}"
            )
        all_results["ac_stream_acceleration"] = acceleration_status
    command_wall_ms = (time.perf_counter_ns() - command_start) / 1e6
    all_results["timing_summary"] = timing.as_dict(command_wall_ms=command_wall_ms)
    result_path = output_dir / "results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Total success_rate=%.4f (%d/%d)", all_results["success_rate"], total_success, total_trials)
    logger.info("Saved rollout results to %s", result_path)
    if not args.suppress_final_summary:
        if "ac_stream_acceleration" in all_results:
            logger.info(
                "AC-Stream acceleration: %s",
                all_results["ac_stream_acceleration"],
            )
        logger.info("\n%s", timing.format_summary(command_wall_ms=command_wall_ms))


if __name__ == "__main__":
    main()
