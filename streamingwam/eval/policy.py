"""Benchmark-agnostic Streaming-WAM policy wrapper for closed-loop evaluation.

This module is the shared, benchmark-neutral core used by evaluation adapters
(e.g. RoboTwin). It is intentionally independent of any specific benchmark and
of the LIBERO ``examples/libero/rollout.py`` script (which is left untouched):

    BenchmarkAdapter (per-benchmark)  ->  StreamingWAMPolicy (this file)  ->  model

An adapter is responsible for turning raw environment observations into
``(image_tensor, state_vector, instruction)`` and for turning predicted action
chunks back into environment actions. ``StreamingWAMPolicy`` owns everything generic:
recipe/checkpoint loading, text conditioning, (de)normalization, flow-matching
inference, and the replan action queue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
import torch

from streamingwam.config import load_config
from streamingwam.data.lerobot import (
    DEFAULT_TEXT_CACHE_ENCODER_ID,
    DEFAULT_TEXT_PROMPT,
    format_text_prompt,
    load_text_cache,
    save_text_cache,
    text_cache_path,
)
from streamingwam.checkpointing import (
    load_inference_checkpoint,
    load_inference_stats,
    prepare_inference_config,
)


# --------------------------------------------------------------------------- #
# Normalization (generic; no benchmark-specific gripper post-processing).
# --------------------------------------------------------------------------- #
def _normalize_vector(vec: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    dim = int(vec.shape[-1])
    if mode == "zscore":
        mean = stats["mean"][:dim].to(vec.dtype)
        std = stats["std"][:dim].to(vec.dtype).clamp_min(1e-6)
        return ((vec - mean) / std).clamp(-5.0, 5.0)
    if mode == "minmax":
        v_min = stats["min"][:dim].to(vec.dtype)
        v_max = stats["max"][:dim].to(vec.dtype)
        return (2.0 * (vec - v_min) / (v_max - v_min).clamp_min(1e-6) - 1.0).clamp(-5.0, 5.0)
    raise ValueError(f"Unsupported norm mode: {mode!r}")


def _denormalize_action(action: torch.Tensor, mode: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    if mode == "zscore":
        return action * stats["std"].clamp_min(1e-6) + stats["mean"]
    if mode == "minmax":
        a_min = stats["min"]
        a_max = stats["max"]
        return (action.clamp(-1.0, 1.0) + 1.0) * 0.5 * (a_max - a_min).clamp_min(1e-6) + a_min
    raise ValueError(f"Unsupported norm mode: {mode!r}")


@dataclass(frozen=True)
class PolicyChunkPrediction:
    env_actions: np.ndarray
    model_actions: torch.Tensor
    inference_ms: float


class StreamingWAMPolicy:
    """Loads a Streaming-WAM recipe/checkpoint and runs closed-loop action inference.

    The policy is benchmark-neutral: callers pass a preprocessed image tensor
    (already in the model's expected layout and ``[-1, 1]`` range), a raw state
    vector, and a language instruction. It returns denormalized action chunks
    and manages an internal replan queue.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint: str,
        *,
        overrides: Optional[list[str]] = None,
        device: str = "cuda:0",
        num_inference_steps: Optional[int] = None,
        action_num_inference_steps: Optional[int] = None,
        replan_steps: int = 8,
        seed: Optional[int] = None,
        checkpoint_format: str = "streamingwam",
        inference_mode: str = "baseline",
    ) -> None:
        from streamingwam import build_framework
        from streamingwam.utils.config_cli import apply_overrides

        config = load_config(config_path)
        if overrides:
            config = apply_overrides(config, overrides)
        config = prepare_inference_config(config, checkpoint_format)
        self.config = config
        self.checkpoint_format = checkpoint_format
        self.inference_mode = self._normalize_inference_mode(inference_mode)

        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        mp = (config.training.mixed_precision or "no").lower()
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
        if self.device.type == "cpu":
            self.dtype = torch.float32

        self.model = build_framework(config, device=str(self.device), dtype=self.dtype).to(self.device)
        report = load_inference_checkpoint(
            self.model,
            checkpoint,
            checkpoint_format=checkpoint_format,
            inference_mode=self.inference_mode,
        )
        print(f"[StreamingWAMPolicy] loaded checkpoint: {report}")
        self.model.eval()

        self.action_horizon = int(config.framework.chunk_size)
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else getattr(config.inference, "num_inference_steps", 8)
        )
        self.action_num_inference_steps = int(
            action_num_inference_steps
            if action_num_inference_steps is not None
            else getattr(config.inference, "action_num_inference_steps", self.num_inference_steps)
        )
        self.replan_steps = int(max(1, min(replan_steps, self.action_horizon)))
        self.seed = seed

        self._action_stats = self._load_stats("action")
        self._state_stats = self._load_stats("state")
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._text_encoder = None
        self.pending_actions: deque[np.ndarray] = deque()

    @staticmethod
    def _normalize_inference_mode(inference_mode: str) -> str:
        normalized = str(inference_mode).strip().lower()
        if normalized not in {"baseline", "cd", "ac-stream"}:
            raise ValueError(
                f"Unsupported inference mode {inference_mode!r}; expected "
                "'baseline', 'cd', or 'ac-stream'"
            )
        return normalized

    # -- stats -------------------------------------------------------------- #
    def _load_stats(self, key: str) -> Optional[dict[str, torch.Tensor]]:
        flag = "normalize_actions" if key == "action" else "normalize_states"
        if not getattr(self.config.data, flag, False):
            return None
        path_attr = "action_stats_path" if key == "action" else "state_stats_path"
        stats_path = getattr(self.config.data, path_attr, None) or getattr(self.config.data, "action_stats_path", None)
        if not stats_path:
            raise ValueError(f"{flag}=true requires data.{path_attr}")
        stats = load_inference_stats(stats_path, checkpoint_format=self.checkpoint_format)
        if key not in stats:
            raise KeyError(f"No {key} stats found in {stats_path}")
        return stats[key]

    # -- text conditioning -------------------------------------------------- #
    def _encode_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if instruction in self._text_cache:
            ctx, mask = self._text_cache[instruction]
            return ctx.to(self.device, self.dtype), mask.to(self.device)

        text_len = int(getattr(self.config.data, "text_len", 128))
        prompt_template = getattr(self.config.data, "text_prompt_template", None) or DEFAULT_TEXT_PROMPT
        encoder_id = getattr(self.config.data, "text_cache_encoder_id", None) or DEFAULT_TEXT_CACHE_ENCODER_ID
        cache_dir = getattr(self.config.data, "text_embedding_cache_dir", None)
        text_dim = int(getattr(getattr(getattr(self.model, "backbone", None), "info", None), "text_dim", 4096))

        cache = text_cache_path(cache_dir, instruction, text_len, prompt_template, encoder_id) if cache_dir else None
        if cache is not None and cache.is_file():
            ctx, mask = load_text_cache(cache, text_len, text_dim)
            ctx = ctx.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            if self._text_encoder is None:
                from streamingwam.backbone import build_text_encoder

                self._text_encoder = build_text_encoder(
                    self.config.backbone,
                    text_len=text_len,
                    device=str(self.device),
                    dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                )
            prompt = format_text_prompt(instruction, prompt_template)
            with torch.no_grad():
                ctx, mask = self._text_encoder.encode([prompt])
            if cache is not None:
                save_text_cache(cache, ctx[0], mask[0], prompt, instruction)

        self._text_cache[instruction] = (ctx.cpu(), mask.cpu())
        return ctx.to(self.device, self.dtype), mask.to(self.device)

    # -- inference ---------------------------------------------------------- #
    def _sampled_video_frame_count(self) -> int:
        num = int(self.config.data.num_frames)
        step = max(1, int(getattr(self.config.data, "action_freq_ratio", 1)))
        return len(range(0, num, step))

    @torch.no_grad()
    def predict_chunk_prediction(
        self, image: torch.Tensor, state: Optional[np.ndarray], instruction: str
    ) -> PolicyChunkPrediction:
        """Return a synchronous chunk and CUDA-synchronized model-core time."""
        image = image.to(device=self.device, dtype=self.dtype)
        context, context_mask = self._encode_context(instruction)

        proprio = None
        if getattr(self.config.framework, "proprio_dim", None) and state is not None:
            proprio_dim = int(self.config.framework.proprio_dim)
            proprio_t = torch.as_tensor(np.asarray(state, dtype=np.float32)[:proprio_dim]).view(1, proprio_dim)
            if self._state_stats is not None:
                proprio_t = _normalize_vector(proprio_t, self.config.data.state_norm_mode, self._state_stats)
            proprio = proprio_t.to(device=self.device, dtype=self.dtype)

        inference_mode = self._normalize_inference_mode(self.inference_mode)
        sampling_method = {
            "baseline": "euler",
            "cd": "consistency",
            "ac-stream": "ac-stream",
        }[inference_mode]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        pred = self.model.infer_action(
            input_image=image,
            context=context,
            context_mask=context_mask,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            action_num_inference_steps=self.action_num_inference_steps,
            seed=self.seed,
            proprio=proprio,
            num_video_frames=self._sampled_video_frame_count(),
            sampling_method=sampling_method,
            consistency_teacher_steps=int(
                getattr(self.config.inference, "consistency_teacher_steps", 4)
            ),
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter_ns() - started) / 1e6

        model_actions = pred.detach().float().cpu()
        if model_actions.ndim == 3:
            model_actions = model_actions[0]
        action = model_actions
        if self._action_stats is not None:
            action = _denormalize_action(action, self.config.data.action_norm_mode, self._action_stats)
        out = action.numpy()
        return PolicyChunkPrediction(
            env_actions=out.astype(np.float32),
            model_actions=model_actions,
            inference_ms=float(inference_ms),
        )

    @torch.no_grad()
    def predict_chunk(self, image: torch.Tensor, state: Optional[np.ndarray], instruction: str) -> np.ndarray:
        """Return a denormalized action chunk ``[T, action_dim]`` (numpy)."""
        return self.predict_chunk_prediction(image, state, instruction).env_actions

    @torch.no_grad()
    def predict_ac_stream_chunk(
        self,
        image: torch.Tensor,
        state: Optional[np.ndarray],
        instruction: str,
        *,
        previous_action_chunk: torch.Tensor | None,
        inference_delay: int,
    ) -> PolicyChunkPrediction:
        """Return normalized and environment AC-Stream chunks plus model time."""

        if self._normalize_inference_mode(self.inference_mode) != "ac-stream":
            raise ValueError("predict_ac_stream_chunk requires inference_mode='ac-stream'")
        image = image.to(device=self.device, dtype=self.dtype)
        context, context_mask = self._encode_context(instruction)
        proprio = None
        if getattr(self.config.framework, "proprio_dim", None) and state is not None:
            proprio_dim = int(self.config.framework.proprio_dim)
            proprio_t = torch.as_tensor(
                np.asarray(state, dtype=np.float32)[:proprio_dim]
            ).view(1, proprio_dim)
            if self._state_stats is not None:
                proprio_t = _normalize_vector(
                    proprio_t,
                    self.config.data.state_norm_mode,
                    self._state_stats,
                )
            proprio = proprio_t.to(device=self.device, dtype=self.dtype)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        pred = self.model.infer_action(
            input_image=image,
            context=context,
            context_mask=context_mask,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            action_num_inference_steps=self.action_num_inference_steps,
            seed=self.seed,
            proprio=proprio,
            num_video_frames=self._sampled_video_frame_count(),
            sampling_method="ac-stream",
            ac_stream_prev_action_chunk=previous_action_chunk,
            ac_stream_inference_delay=int(inference_delay),
            ac_stream_stride=int(getattr(self.config.inference, "ac_stream_stride", 16)),
            ac_stream_delay=int(getattr(self.config.inference, "ac_stream_delay", 8)),
            ac_stream_launch_after_steps=int(
                getattr(self.config.inference, "ac_stream_launch_after_steps", 8)
            ),
            ac_stream_context_key=instruction,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter_ns() - started) / 1e6

        model_actions = pred.detach().float().cpu()
        if model_actions.ndim == 3:
            model_actions = model_actions[0]
        env_actions = model_actions
        if self._action_stats is not None:
            env_actions = _denormalize_action(
                env_actions,
                self.config.data.action_norm_mode,
                self._action_stats,
            )
        return PolicyChunkPrediction(
            env_actions=env_actions.numpy().astype(np.float32),
            model_actions=model_actions,
            inference_ms=float(inference_ms),
        )

    # -- replan queue ------------------------------------------------------- #
    def needs_observation(self) -> bool:
        return not self.pending_actions

    def fill_queue(self, image: torch.Tensor, state: Optional[np.ndarray], instruction: str) -> None:
        chunk = self.predict_chunk(image, state, instruction)
        for i in range(min(self.replan_steps, chunk.shape[0])):
            self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))

    def pop_action(self) -> np.ndarray:
        return self.pending_actions.popleft()

    def reset(self) -> None:
        self.pending_actions.clear()
