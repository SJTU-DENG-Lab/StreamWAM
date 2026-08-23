"""D0/D8 asynchronous RTC action-chunk inference.

RTC-AC is the FastWAM Stage-2 selfatt-z1 inference contract: H32/s16/d8,
one-step joint consistency, a prefix-free D0 startup, and an eight-action
clean prefix for steady-state D8 predictions.  The eager controller and the
later accelerated backend intentionally share this module.
"""

from __future__ import annotations

import copy
import platform
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


RTC_AC_ACTION_HORIZON = 32
RTC_AC_STRIDE = 16
RTC_AC_DELAY = 8
RTC_AC_LAUNCH_AFTER_STEPS = 8
RTC_AC_VIDEO_FRAMES = 9
RTC_AC_TEMPORAL_COMPRESS = 4
RTC_AC_INFERENCE_STEPS = 1
RTC_AC_ACTION_DIM = 7
RTC_AC_CONDITION_SLOTS = 16


def _copy_tensor_in_place(
    destination: Tensor | None,
    source: Tensor,
) -> Tensor:
    if (
        destination is None
        or destination.shape != source.shape
        or destination.device != source.device
        or destination.dtype != source.dtype
    ):
        return source.detach()
    destination.copy_(source)
    return destination


def _read_compiler_counters() -> dict[str, int]:
    """Read stable Dynamo/Inductor counters without mutating global state."""

    try:
        from torch._dynamo.utils import counters
    except (ImportError, AttributeError):
        return {
            "dynamo_unique_graphs": 0,
            "inductor_cudagraph_skips": 0,
        }
    return {
        "dynamo_unique_graphs": int(
            counters.get("stats", {}).get("unique_graphs", 0)
        ),
        "inductor_cudagraph_skips": int(
            counters.get("inductor", {}).get("cudagraph_skips", 0)
        ),
    }


class RTCACAccelerationRuntime:
    """Strict fixed-shape compiler and inference caches for RTC-AC."""

    def __init__(self) -> None:
        self._attention_masks: dict[tuple[Any, ...], Tensor] = {}
        self._schedules: dict[
            tuple[Any, ...],
            tuple[Tensor, Tensor],
        ] = {}
        self._context_key: str | None = None
        self._static_projected_context: dict[str, Tensor] = {}
        self._static_cross_attention_kv: dict[
            str,
            tuple[tuple[Tensor, Tensor], ...],
        ] = {}
        self._static_context_length = 0
        self._compiled_mot: Callable[..., Any] | None = None
        self._compiled_mot_owner: int | None = None
        self._compile_active = False
        self._prewarmed_delays: set[int] = set()
        self._compiler_counter_baseline = _read_compiler_counters()

    @property
    def static_cross_attention_kv(
        self,
    ) -> dict[str, tuple[tuple[Tensor, Tensor], ...]]:
        if not self._static_cross_attention_kv:
            raise RuntimeError("RTC-AC static cross-attention cache is not prepared")
        return self._static_cross_attention_kv

    @property
    def static_context_length(self) -> int:
        return self._static_context_length

    def get_attention_mask(
        self,
        key: tuple[Any, ...],
        builder: Callable[[], Tensor],
    ) -> Tensor:
        if key not in self._attention_masks:
            self._attention_masks[key] = builder()
        return self._attention_masks[key]

    def get_schedule(
        self,
        key: tuple[Any, ...],
        builder: Callable[[], tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        if key not in self._schedules:
            self._schedules[key] = builder()
        return self._schedules[key]

    @staticmethod
    def _project_cross_attention_kv(
        expert: torch.nn.Module,
        context: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        return tuple(
            (
                block.cross_attn.norm_k(block.cross_attn.k(context)),
                block.cross_attn.v(context),
            )
            for block in expert.blocks
        )

    @staticmethod
    def _refresh_kv_in_place(
        old: tuple[tuple[Tensor, Tensor], ...] | None,
        new: tuple[tuple[Tensor, Tensor], ...],
    ) -> tuple[tuple[Tensor, Tensor], ...]:
        if old is None or len(old) != len(new):
            return tuple((key.detach(), value.detach()) for key, value in new)
        return tuple(
            (
                _copy_tensor_in_place(old_key, new_key),
                _copy_tensor_in_place(old_value, new_value),
            )
            for (old_key, old_value), (new_key, new_value) in zip(old, new)
        )

    @torch.no_grad()
    def prepare_contexts(
        self,
        *,
        context_key: str,
        static_context: Tensor,
        dynamic_context: Tensor,
        video_expert: torch.nn.Module,
        action_expert: torch.nn.Module,
    ) -> dict[str, Tensor]:
        """Cache task-static text work and project the dynamic proprio token."""

        if not context_key:
            raise ValueError("RTC-AC accelerated inference requires a context key")
        if static_context.ndim != 3 or dynamic_context.ndim != 3:
            raise ValueError("RTC-AC contexts must have shape [B,L,D]")
        if self._context_key != context_key:
            projected = {
                "video": video_expert.text_embedding(static_context),
                "action": action_expert.text_embedding(static_context),
            }
            for stream, value in projected.items():
                self._static_projected_context[stream] = _copy_tensor_in_place(
                    self._static_projected_context.get(stream),
                    value,
                )
            for stream, expert in (
                ("video", video_expert),
                ("action", action_expert),
            ):
                new_kv = self._project_cross_attention_kv(
                    expert,
                    self._static_projected_context[stream],
                )
                self._static_cross_attention_kv[stream] = self._refresh_kv_in_place(
                    self._static_cross_attention_kv.get(stream),
                    new_kv,
                )
            self._static_context_length = int(static_context.shape[1])
            self._context_key = str(context_key)

        dynamic_projected = {
            "video": video_expert.text_embedding(dynamic_context),
            "action": action_expert.text_embedding(dynamic_context),
        }
        return {
            stream: torch.cat(
                (self._static_projected_context[stream], dynamic_projected[stream]),
                dim=1,
            )
            for stream in ("video", "action")
        }

    def run_mot(self, mot: torch.nn.Module, **kwargs: Any) -> Any:
        owner = id(mot)
        if self._compiled_mot is None:
            self._compiled_mot = torch.compile(
                mot.forward_rtc_ac_accelerated,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )
            self._compiled_mot_owner = owner
        elif self._compiled_mot_owner != owner:
            raise RuntimeError("RTC-AC acceleration runtime cannot change MoT instances")
        result = self._compiled_mot(**kwargs)
        self._compile_active = True
        return result

    def mark_prewarmed(self, delay: int) -> None:
        delay = int(delay)
        if delay not in (0, RTC_AC_DELAY):
            raise ValueError(f"RTC-AC prewarm delay must be 0 or {RTC_AC_DELAY}")
        self._prewarmed_delays.add(delay)

    @property
    def prewarm_complete(self) -> bool:
        return self._prewarmed_delays == {0, RTC_AC_DELAY}

    def status(self) -> dict[str, Any]:
        import triton

        counters = _read_compiler_counters()
        unique_graphs = max(
            0,
            counters["dynamo_unique_graphs"]
            - self._compiler_counter_baseline["dynamo_unique_graphs"],
        )
        cudagraph_skips = max(
            0,
            counters["inductor_cudagraph_skips"]
            - self._compiler_counter_baseline["inductor_cudagraph_skips"],
        )
        gpu_name = None
        gpu_compute_capability = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            gpu_compute_capability = torch.cuda.get_device_capability(
                torch.cuda.current_device()
            )

        return {
            "backend": "accelerated",
            "compile_active": bool(self._compile_active),
            "compile_mode": "reduce-overhead",
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "cuda_graph_trees": True,
            "dynamo_unique_graphs": unique_graphs,
            "dynamo_recompiles": max(0, unique_graphs - 1),
            "inductor_cudagraph_skips": cudagraph_skips,
            "context_cache": bool(self._static_projected_context),
            "cross_attention_kv_cache": bool(self._static_cross_attention_kv),
            "attention_mask_cache_entries": len(self._attention_masks),
            "schedule_cache_entries": len(self._schedules),
            "prewarmed_d0": 0 in self._prewarmed_delays,
            "prewarmed_d8": RTC_AC_DELAY in self._prewarmed_delays,
            "runtime": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "triton_version": triton.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": gpu_name,
                "gpu_compute_capability": gpu_compute_capability,
            },
        }


def validate_rtc_ac_geometry(
    *,
    action_horizon: int,
    stride: int,
    delay: int,
    launch_after_steps: int,
    num_video_frames: int,
    temporal_compress: int,
    num_inference_steps: int,
) -> None:
    """Reject runtime geometry that differs from the step-5500 checkpoint."""

    expected = {
        "action_horizon": RTC_AC_ACTION_HORIZON,
        "stride": RTC_AC_STRIDE,
        "delay": RTC_AC_DELAY,
        "launch_after_steps": RTC_AC_LAUNCH_AFTER_STEPS,
        "num_video_frames": RTC_AC_VIDEO_FRAMES,
        "temporal_compress": RTC_AC_TEMPORAL_COMPRESS,
        "num_inference_steps": RTC_AC_INFERENCE_STEPS,
    }
    actual = {
        "action_horizon": int(action_horizon),
        "stride": int(stride),
        "delay": int(delay),
        "launch_after_steps": int(launch_after_steps),
        "num_video_frames": int(num_video_frames),
        "temporal_compress": int(temporal_compress),
        "num_inference_steps": int(num_inference_steps),
    }
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise ValueError(
                f"RTC-AC checkpoint requires {name}={expected_value}, "
                f"got {actual[name]}"
            )


def build_rtc_ac_prev_action_target(
    current_model_chunk: Tensor,
    cursor: int,
    *,
    action_horizon: int = RTC_AC_ACTION_HORIZON,
    execute_horizon: int = RTC_AC_STRIDE,
) -> Tensor:
    """Align the current normalized action chunk to a future D8 launch."""

    if current_model_chunk.ndim != 2:
        raise ValueError(
            "RTC-AC current action chunk must be [H,D], got "
            f"{tuple(current_model_chunk.shape)}"
        )
    if int(current_model_chunk.shape[0]) != int(action_horizon):
        raise ValueError(
            f"RTC-AC current chunk must have horizon {action_horizon}, got "
            f"{current_model_chunk.shape[0]}"
        )
    cursor = int(cursor)
    if cursor < 0 or cursor > int(action_horizon):
        raise ValueError(f"RTC-AC cursor must be in [0,{action_horizon}], got {cursor}")
    target = torch.zeros_like(current_model_chunk)
    valid_len = max(
        0,
        min(
            int(action_horizon) - int(execute_horizon),
            int(action_horizon) - cursor,
        ),
    )
    if valid_len:
        target[:valid_len] = current_model_chunk[cursor : cursor + valid_len]
    return target.detach()


def apply_rtc_ac_hard_prefix_(
    action_latents: Tensor,
    action_timesteps: Tensor,
    previous_action_chunk: Tensor,
) -> Tensor:
    """Clamp the D8 clean prefix and expose sigma=0 to the model/boundary."""

    expected = (action_latents.shape[0], RTC_AC_ACTION_HORIZON, RTC_AC_ACTION_DIM)
    if tuple(action_latents.shape) != expected:
        raise ValueError(
            f"RTC-AC action latents must be {expected}, got "
            f"{tuple(action_latents.shape)}"
        )
    if tuple(action_timesteps.shape) != expected[:2]:
        raise ValueError(
            f"RTC-AC action timesteps must be {expected[:2]}, got "
            f"{tuple(action_timesteps.shape)}"
        )
    if previous_action_chunk.ndim == 2:
        previous_action_chunk = previous_action_chunk.unsqueeze(0)
    if tuple(previous_action_chunk.shape) != expected:
        raise ValueError(
            f"RTC-AC previous action chunk must be {expected}, got "
            f"{tuple(previous_action_chunk.shape)}"
        )
    prefix = previous_action_chunk[:, :RTC_AC_DELAY].to(
        device=action_latents.device,
        dtype=action_latents.dtype,
    )
    action_latents[:, :RTC_AC_DELAY].copy_(prefix)
    action_timesteps[:, :RTC_AC_DELAY].zero_()
    return prefix.clone()


@dataclass(frozen=True)
class RTCACPrediction:
    """One normalized/model and denormalized/environment RTC-AC chunk."""

    env_actions: np.ndarray
    model_actions: Tensor
    communication_ms: float
    inference_ms: float
    inference_started_ns: int | None = None
    inference_completed_ns: int | None = None

    def __post_init__(self) -> None:
        if tuple(self.env_actions.shape) != (RTC_AC_ACTION_HORIZON, RTC_AC_ACTION_DIM):
            raise ValueError(
                "RTC-AC environment actions must be [32,7], got "
                f"{tuple(self.env_actions.shape)}"
            )
        if tuple(self.model_actions.shape) != (RTC_AC_ACTION_HORIZON, RTC_AC_ACTION_DIM):
            raise ValueError(
                "RTC-AC model actions must be [32,7], got "
                f"{tuple(self.model_actions.shape)}"
            )


RTCACPredictFn = Callable[[Any, Tensor | None, int], RTCACPrediction]


@dataclass(frozen=True)
class RTCACOverlapRecord:
    """Measured overlap of one asynchronous D8 inference with action execution."""

    inference_wall_ms: float
    action_overlap_ms: float
    boundary_wait_ms: float
    ready_before_boundary: bool | None
    episode_end_before_boundary: bool

    @property
    def hidden_inference_ratio(self) -> float:
        if self.inference_wall_ms <= 0.0:
            return 0.0
        return min(1.0, max(0.0, self.action_overlap_ms / self.inference_wall_ms))


def build_rtc_ac_overlap_record(
    *,
    inference_started_ns: int,
    inference_completed_ns: int,
    prediction_completed_ns: int,
    overlap_window_started_ns: int,
    action_execution_intervals_ns: list[tuple[int, int]] | None = None,
    boundary_ns: int | None = None,
    swap_ns: int | None = None,
    episode_end_ns: int | None = None,
) -> RTCACOverlapRecord:
    """Convert monotonic RTC-AC events into one D8 overlap measurement."""

    if (boundary_ns is None) == (episode_end_ns is None):
        raise ValueError("Provide exactly one of boundary_ns or episode_end_ns")
    interval_end_ns = boundary_ns if boundary_ns is not None else episode_end_ns
    assert interval_end_ns is not None
    inference_wall_ns = max(0, inference_completed_ns - inference_started_ns)
    overlap_start_ns = max(inference_started_ns, overlap_window_started_ns)
    overlap_end_ns = min(inference_completed_ns, interval_end_ns)
    if action_execution_intervals_ns is None:
        overlap_ns = max(0, overlap_end_ns - overlap_start_ns)
    else:
        overlap_ns = sum(
            max(0, min(inference_completed_ns, end_ns, interval_end_ns) - max(
                inference_started_ns,
                start_ns,
                overlap_window_started_ns,
            ))
            for start_ns, end_ns in action_execution_intervals_ns
        )
    if boundary_ns is None:
        wait_ns = 0
        ready = None
    else:
        wait_ns = max(0, prediction_completed_ns - boundary_ns)
        ready = prediction_completed_ns <= boundary_ns
    return RTCACOverlapRecord(
        inference_wall_ms=inference_wall_ns / 1e6,
        action_overlap_ms=overlap_ns / 1e6,
        boundary_wait_ms=wait_ns / 1e6,
        ready_before_boundary=ready,
        episode_end_before_boundary=boundary_ns is None,
    )


@dataclass(frozen=True)
class _RTCACFutureResult:
    prediction: RTCACPrediction
    prediction_completed_ns: int


class RTCACController:
    """Own the H32/s16/d8 asynchronous chunk state machine."""

    def __init__(
        self,
        predict: RTCACPredictFn,
        *,
        block_on_miss: bool = True,
    ) -> None:
        if not block_on_miss:
            raise ValueError(
                "RTC-AC H32/s16/d8 requires block_on_miss=True so the "
                "D8 prefix remains aligned at the stride boundary"
            )
        self._predict = predict
        self._block_on_miss = True
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="starwam_rtc_ac",
        )
        self._future: Future[_RTCACFutureResult] | None = None
        self._current: RTCACPrediction | None = None
        self._cursor = 0
        self._window_start_cursor = 0
        self._window_end_cursor = RTC_AC_STRIDE
        self._next_launch_cursor = RTC_AC_LAUNCH_AFTER_STEPS
        self._launch_cursor: int | None = None
        self._launch_ns: int | None = None
        self._pending_boundary_ns: int | None = None
        self._closed = False
        self._installed: list[RTCACPrediction] = []
        self._overlap_records: list[RTCACOverlapRecord] = []
        self._pending_action_intervals_ns: list[tuple[int, int]] = []

    @property
    def cursor(self) -> int:
        return self._cursor

    def start_episode(self, observation: Any) -> RTCACPrediction:
        if self._closed:
            raise RuntimeError("RTC-AC controller is closed")
        if self._current is not None:
            raise RuntimeError("RTC-AC episode is already started")
        prediction = self._predict(copy.deepcopy(observation), None, 0)
        self._install(prediction, new_cursor=0)
        return prediction

    def _install(self, prediction: RTCACPrediction, *, new_cursor: int) -> None:
        if not 0 <= int(new_cursor) < RTC_AC_ACTION_HORIZON:
            raise RuntimeError(f"RTC-AC swap cursor is invalid: {new_cursor}")
        self._current = prediction
        self._cursor = int(new_cursor)
        self._window_start_cursor = self._cursor
        self._window_end_cursor = min(
            self._window_start_cursor + RTC_AC_STRIDE,
            RTC_AC_ACTION_HORIZON,
        )
        self._next_launch_cursor = min(
            self._window_start_cursor + RTC_AC_LAUNCH_AFTER_STEPS,
            RTC_AC_ACTION_HORIZON,
        )
        self._installed.append(prediction)

    def _launch(self, observation: Any) -> None:
        if self._current is None or self._future is not None:
            raise RuntimeError("RTC-AC launch state is inconsistent")
        previous_target = build_rtc_ac_prev_action_target(
            self._current.model_actions,
            self._cursor,
            action_horizon=RTC_AC_ACTION_HORIZON,
            execute_horizon=RTC_AC_STRIDE,
        )
        observation_snapshot = copy.deepcopy(observation)
        self._launch_cursor = self._cursor
        self._launch_ns = time.perf_counter_ns()
        self._pending_boundary_ns = None
        self._pending_action_intervals_ns = []

        def predict_and_timestamp() -> _RTCACFutureResult:
            prediction = self._predict(
                observation_snapshot,
                previous_target,
                RTC_AC_DELAY,
            )
            return _RTCACFutureResult(prediction, time.perf_counter_ns())

        self._future = self._executor.submit(predict_and_timestamp)

    def _record_overlap(
        self,
        result: _RTCACFutureResult,
        *,
        boundary_ns: int | None = None,
        swap_ns: int | None = None,
        episode_end_ns: int | None = None,
    ) -> None:
        launch_ns = self._launch_ns
        if launch_ns is None:
            raise RuntimeError("RTC-AC pending prediction has no launch timestamp")
        prediction = result.prediction
        inference_completed_ns = prediction.inference_completed_ns
        if inference_completed_ns is None:
            inference_completed_ns = result.prediction_completed_ns
        inference_started_ns = prediction.inference_started_ns
        if inference_started_ns is None:
            inference_started_ns = inference_completed_ns - round(
                prediction.inference_ms * 1e6
            )
        self._overlap_records.append(
            build_rtc_ac_overlap_record(
                inference_started_ns=inference_started_ns,
                inference_completed_ns=inference_completed_ns,
                prediction_completed_ns=result.prediction_completed_ns,
                overlap_window_started_ns=launch_ns,
                action_execution_intervals_ns=self._pending_action_intervals_ns,
                boundary_ns=boundary_ns,
                swap_ns=swap_ns,
                episode_end_ns=episode_end_ns,
            )
        )

    def _complete_pending(self, *, boundary_ns: int | None = None) -> bool:
        if self._future is None:
            return False
        if not self._future.done() and not self._block_on_miss:
            return False
        future = self._future
        launch_cursor = self._launch_cursor
        self._future = None
        self._launch_cursor = None
        if launch_cursor is None:
            raise RuntimeError("RTC-AC pending prediction has no launch cursor")
        result = future.result()
        prediction = result.prediction
        elapsed_actions = self._cursor - launch_cursor
        if elapsed_actions >= RTC_AC_ACTION_HORIZON:
            raise RuntimeError(
                f"RTC-AC inference became stale after {elapsed_actions} actions"
            )
        self._install(prediction, new_cursor=elapsed_actions)
        swap_ns = time.perf_counter_ns()
        if boundary_ns is not None:
            self._record_overlap(result, boundary_ns=boundary_ns, swap_ns=swap_ns)
        self._launch_ns = None
        self._pending_action_intervals_ns = []
        self._pending_boundary_ns = None
        return True

    def next_action(self, observation: Any) -> np.ndarray:
        if self._closed:
            raise RuntimeError("RTC-AC controller is closed")
        if self._current is None:
            raise RuntimeError("Call start_episode before requesting RTC-AC actions")
        if self._cursor >= self._window_end_cursor and self._future is not None:
            self._complete_pending(
                boundary_ns=self._pending_boundary_ns or time.perf_counter_ns()
            )
        if (
            self._future is None
            and self._cursor >= self._next_launch_cursor
            and self._cursor < self._window_end_cursor
        ):
            self._launch(observation)
        if self._cursor >= RTC_AC_ACTION_HORIZON:
            if not self._complete_pending():
                raise RuntimeError("RTC-AC exhausted the current action chunk")
        return np.array(self._current.env_actions[self._cursor], copy=True)

    def mark_action_executed(
        self,
        *,
        started_ns: int | None = None,
        completed_ns: int | None = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("RTC-AC episode has not started")
        if (started_ns is None) != (completed_ns is None):
            raise ValueError("Provide both action execution timestamps or neither")
        if (
            self._future is not None
            and started_ns is not None
            and completed_ns is not None
        ):
            self._pending_action_intervals_ns.append((started_ns, completed_ns))
        if self._future is not None and self._cursor + 1 >= self._window_end_cursor:
            self._pending_boundary_ns = (
                completed_ns if completed_ns is not None else time.perf_counter_ns()
            )
        self._cursor += 1

    def pop_installed_predictions(self) -> list[RTCACPrediction]:
        installed, self._installed = self._installed, []
        return installed

    def pop_overlap_records(self) -> list[RTCACOverlapRecord]:
        records, self._overlap_records = self._overlap_records, []
        return records

    def close(self) -> RTCACPrediction | None:
        if self._closed:
            return None
        episode_end_ns = time.perf_counter_ns()
        pending_prediction = None
        future = self._future
        self._future = None
        self._launch_cursor = None
        try:
            if future is not None and not future.cancel():
                result = future.result()
                pending_prediction = result.prediction
                if self._pending_boundary_ns is None:
                    self._record_overlap(result, episode_end_ns=episode_end_ns)
                else:
                    self._record_overlap(
                        result,
                        boundary_ns=self._pending_boundary_ns,
                        swap_ns=time.perf_counter_ns(),
                    )
            return pending_prediction
        finally:
            self._launch_ns = None
            self._pending_action_intervals_ns = []
            self._pending_boundary_ns = None
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True
