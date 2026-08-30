"""RoboTwin policy adapter for Streaming-WAM (remote / socket client).

This is the SAPIEN-side counterpart to ``examples.robotwin.policy_server``. It
runs inside the RoboTwin environment and needs only ``numpy`` plus the Python
standard library. It forwards raw observations to the Streaming-WAM inference server
and executes the returned action chunk.

Use this instead of ``examples/robotwin/local_policy.py`` when SAPIEN and the
Torch/Streaming-WAM stack cannot share one environment.

RoboTwin harness entry points: get_model / eval / reset_model.
Camera order MUST match the recipe's ``data.video_keys`` = [head, left_wrist, right_wrist].
"""

from __future__ import annotations

import pickle
import socket
import struct
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from streamingwam.inference.ac_stream import (
    ACStreamController,
    ACStreamPrediction,
)


# Length-prefixed pickle framing (mirrors examples.robotwin.policy_server).
def _send_msg(conn: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


def _recv_exactly(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> Any:
    header = _recv_exactly(conn, 8)
    if header is None:
        raise ConnectionError("policy server closed the connection")
    (length,) = struct.unpack(">Q", header)
    body = _recv_exactly(conn, length)
    if body is None:
        raise ConnectionError("policy server closed the connection mid-message")
    return pickle.loads(body)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _take_environment_action(
    task_env: Any,
    action: np.ndarray,
    *,
    defer_render: bool,
) -> tuple[int, int]:
    """Execute one action while suppressing redundant renders during D8 overlap."""

    original_update_render = None
    original_get_obs = None
    if defer_render:
        original_update_render = task_env._update_render
        original_get_obs = task_env.get_obs

        def skip_update_render(*_args, **_kwargs):
            return None

        def get_obs_with_render(*args, **kwargs):
            task_env._update_render = original_update_render
            try:
                return original_get_obs(*args, **kwargs)
            finally:
                task_env._update_render = skip_update_render

        task_env._update_render = skip_update_render
        task_env.get_obs = get_obs_with_render
    started_ns = time.perf_counter_ns()
    try:
        task_env.take_action(action, action_type="qpos")
    finally:
        if original_update_render is not None:
            task_env._update_render = original_update_render
            task_env.get_obs = original_get_obs
    return started_ns, time.perf_counter_ns()


class RemoteStreamingWAMModel:
    """Talks to the Streaming-WAM inference server; manages the replan queue locally."""

    def __init__(
        self,
        host: str,
        port: int,
        replan_steps: int,
        connect_timeout: float = 600.0,
        inference_mode: str = "baseline",
        prewarm: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.replan_steps = int(max(1, replan_steps))
        self.inference_mode = str(inference_mode).strip().lower()
        if self.inference_mode not in {"baseline", "cd", "ac-stream"}:
            raise ValueError(f"Unsupported inference mode {inference_mode!r}")
        self.pending_actions: deque[np.ndarray] = deque()
        self._conn = self._connect(connect_timeout)
        self._request_id = 0
        self._ac_stream_controller: ACStreamController | None = None
        self.prewarm = bool(prewarm)
        self._prewarmed = False
        self.current_task = "unknown"
        self.current_config = "unknown"
        self._timing_records: list[dict[str, Any]] = []
        self._episode_records: list[dict[str, Any]] = []
        self._episode_started_ns: int | None = None
        self._terminal_action_completion_ns: int | None = None
        self._timing_hook_active = False
        self._active_timing_metadata: dict[str, Any] = {}
        self._episode_index = 0
        self._replan_index = 0

    @property
    def needs_prewarm(self) -> bool:
        return self.prewarm and not self._prewarmed

    def _connect(self, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                conn = socket.create_connection((self.host, self.port), timeout=30.0)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(None)
                print(f"[streamingwam_client] connected to policy server {self.host}:{self.port}", flush=True)
                return conn
            except OSError as err:
                last_err = err
                print(f"[streamingwam_client] waiting for policy server {self.host}:{self.port} ...", flush=True)
                time.sleep(3.0)
        raise ConnectionError(f"could not reach policy server {self.host}:{self.port}: {last_err}")

    def _infer(self, head, left, right, state, instruction, *, warmup: bool = False) -> np.ndarray:
        self._request_id += 1
        started_ns = time.perf_counter_ns()
        _send_msg(self._conn, {
            "cmd": "infer",
            "request_id": self._request_id,
            "head": np.ascontiguousarray(head),
            "left": np.ascontiguousarray(left),
            "right": np.ascontiguousarray(right),
            "state": np.asarray(state, dtype=np.float32),
            "instruction": str(instruction),
            "warmup": bool(warmup),
        })
        resp = _recv_msg(self._conn)
        completed_ns = time.perf_counter_ns()
        if "error" in resp:
            raise RuntimeError(f"policy server error:\n{resp['error']}")
        if resp.get("request_id") != self._request_id:
            raise RuntimeError(
                f"policy response ID mismatch: expected {self._request_id}, "
                f"got {resp.get('request_id')}"
            )
        self._record_inference(
            resp,
            started_ns=started_ns,
            completed_ns=completed_ns,
            warmup=warmup,
            regime="sync",
        )
        return np.asarray(resp["action"], dtype=np.float32)

    def _record_inference(
        self,
        response: dict,
        *,
        started_ns: int,
        completed_ns: int,
        warmup: bool,
        regime: str,
    ) -> None:
        self._timing_records.append({
            "record_type": "inference",
            "task": self.current_task,
            "config": self.current_config,
            "episode": self._episode_index,
            "replan_index": self._replan_index,
            "regime": regime,
            "model_inference_ms": float(response["model_inference_ms"]),
            "server_total_ms": float(response.get("server_total_ms", response["model_inference_ms"])),
            "request_roundtrip_ms": (completed_ns - started_ns) / 1e6,
            "warmup": bool(warmup or response.get("warmup", False)),
            "backend": response.get("backend", "eager"),
            "compile_active": bool(response.get("compile_active", False)),
            "runtime": response.get("runtime", {}),
        })
        self._replan_index += 1

    @staticmethod
    def _snapshot(task_env: Any, observation: Dict[str, Any]) -> dict[str, Any]:
        obs = observation["observation"]
        return {
            "head": np.ascontiguousarray(obs["head_camera"]["rgb"]),
            "left": np.ascontiguousarray(obs["left_camera"]["rgb"]),
            "right": np.ascontiguousarray(obs["right_camera"]["rgb"]),
            "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "instruction": str(task_env.get_instruction()),
        }

    def _predict_remote(
        self,
        snapshot: dict[str, Any],
        previous_action_chunk: torch.Tensor | None,
        inference_delay: int,
    ) -> ACStreamPrediction:
        self._request_id += 1
        request_id = self._request_id
        request = {
            "cmd": "infer",
            "request_id": request_id,
            **snapshot,
            "previous_action_chunk": (
                None
                if previous_action_chunk is None
                else previous_action_chunk.detach().cpu().numpy()
            ),
            "inference_delay": int(inference_delay),
        }
        started_ns = time.perf_counter_ns()
        _send_msg(self._conn, request)
        response = _recv_msg(self._conn)
        completed_ns = time.perf_counter_ns()
        if "error" in response:
            raise RuntimeError(f"policy server error:\n{response['error']}")
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"policy response ID mismatch: expected {request_id}, "
                f"got {response.get('request_id')}"
            )
        inference_ms = float(response["model_inference_ms"])
        communication_ms = max(0.0, (completed_ns - started_ns) / 1e6 - inference_ms)
        self._record_inference(
            response,
            started_ns=started_ns,
            completed_ns=completed_ns,
            warmup=bool(request.get("warmup", False)),
            regime="d0" if inference_delay == 0 else "d8",
        )
        return ACStreamPrediction(
            env_actions=np.asarray(response["action"], dtype=np.float32),
            model_actions=torch.as_tensor(
                np.asarray(response["model_action"], dtype=np.float32)
            ),
            communication_ms=communication_ms,
            inference_ms=inference_ms,
            inference_started_ns=started_ns,
            inference_completed_ns=completed_ns,
        )

    def _prewarm_once(self, task_env: Any, observation: Dict[str, Any] | None) -> None:
        if self._prewarmed or not self.prewarm:
            return
        if observation is None:
            raise ValueError("prewarm requires the first episode observation")
        snapshot = self._snapshot(task_env, observation)
        if self.inference_mode == "ac-stream":
            # Mark requests as warmup at the record layer; inference still uses
            # the exact D0/D8 public protocol and therefore compiles both graphs.
            d0 = self._predict_remote(snapshot, None, 0)
            self._timing_records[-1]["warmup"] = True
            self._predict_remote(snapshot, d0.model_actions, 8)
            self._timing_records[-1]["warmup"] = True
        else:
            self._infer(
                snapshot["head"], snapshot["left"], snapshot["right"],
                snapshot["state"], snapshot["instruction"], warmup=True,
            )
        self._prewarmed = True
        self._replan_index = 0

    def _start_episode(self, *, started_ns: int | None = None) -> None:
        if self._episode_started_ns is None:
            self._episode_started_ns = (
                time.perf_counter_ns() if started_ns is None else int(started_ns)
            )
            self._terminal_action_completion_ns = None
            self._replan_index = 0

    def finish_episode(
        self,
        success: bool | None = None,
        *,
        ended_ns: int | None = None,
    ) -> None:
        if self._episode_started_ns is None:
            return
        completed_ns = time.perf_counter_ns() if ended_ns is None else int(ended_ns)
        elapsed = (completed_ns - self._episode_started_ns) / 1e9
        self._episode_records.append({
            "record_type": "episode",
            "task": self.current_task,
            "config": self.current_config,
            "episode": self._episode_index,
            "success": success,
            "total_time_s": elapsed,
            **self._active_timing_metadata,
        })
        self._episode_started_ns = None
        self._terminal_action_completion_ns = None
        self._timing_hook_active = False
        self._active_timing_metadata = {}
        # Drain a terminal episode's final D8 request before the worker changes
        # task identity. The elapsed value above excludes this post-terminal
        # cleanup, while the launched chunk remains represented in timing.
        if self._ac_stream_controller is not None:
            self._ac_stream_controller.close()
            self._ac_stream_controller = None
        self._episode_index += 1

    def step(self, task_env: Any, observation: Optional[Dict[str, Any]]) -> None:
        self._prewarm_once(task_env, observation)
        self._start_episode()
        if self.inference_mode == "ac-stream":
            if self._ac_stream_controller is None:
                if observation is None:
                    raise ValueError("AC-Stream D0 requires an observation")
                snapshot = self._snapshot(task_env, observation)
                self._ac_stream_controller = ACStreamController(self._predict_remote)
                self._ac_stream_controller.start_episode(snapshot)
            elif self._ac_stream_controller.observation_required:
                if observation is None:
                    raise ValueError("AC-Stream D8 launch requires an observation")
                snapshot = self._snapshot(task_env, observation)
            else:
                snapshot = None
            action = self._ac_stream_controller.next_action(snapshot)
            started_ns, completed_ns = _take_environment_action(
                task_env,
                action,
                defer_render=self._ac_stream_controller.inference_in_flight,
            )
            self._ac_stream_controller.mark_action_executed(
                started_ns=started_ns,
                completed_ns=completed_ns,
            )
        else:
            if not self.pending_actions:
                if observation is None:
                    raise ValueError("Observation required on a replan step but got None.")
                obs = observation["observation"]
                chunk = self._infer(
                    obs["head_camera"]["rgb"],
                    obs["left_camera"]["rgb"],
                    obs["right_camera"]["rgb"],
                    observation["joint_action"]["vector"],
                    task_env.get_instruction(),
                )
                for i in range(min(self.replan_steps, chunk.shape[0])):
                    self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))
            if self.pending_actions:
                _, completed_ns = _take_environment_action(
                    task_env,
                    self.pending_actions.popleft(),
                    defer_render=False,
                )
        succeeded = bool(getattr(task_env, "eval_success", False))
        exhausted = int(getattr(task_env, "take_action_cnt", 0)) >= int(
            getattr(task_env, "step_lim", 1 << 30)
        )
        if succeeded or exhausted:
            self._terminal_action_completion_ns = completed_ns
            if not self._timing_hook_active:
                self.finish_episode(
                    success=succeeded,
                    ended_ns=self._terminal_action_completion_ns,
                )

    def should_request_observation(self) -> bool:
        if self.inference_mode != "ac-stream":
            return not self.pending_actions
        if self._ac_stream_controller is None:
            return True
        return self._ac_stream_controller.observation_required

    def prepare_instruction(self, instruction: str) -> None:
        self._request_id += 1
        _send_msg(
            self._conn,
            {
                "cmd": "prepare_instruction",
                "request_id": self._request_id,
                "instruction": str(instruction),
            },
        )
        response = _recv_msg(self._conn)
        if "error" in response:
            raise RuntimeError(f"policy server error:\n{response['error']}")
        if response.get("request_id") != self._request_id:
            raise RuntimeError(
                f"policy response ID mismatch: expected {self._request_id}, "
                f"got {response.get('request_id')}"
            )

    def prewarm_model(self, task_env: Any, observation: Dict[str, Any]) -> None:
        self._prewarm_once(task_env, observation)

    def begin_timing_trajectory(self, metadata: dict[str, Any]) -> None:
        if self._episode_started_ns is not None:
            raise RuntimeError("a timed RoboTwin trajectory is already active")
        self._timing_hook_active = True
        self._active_timing_metadata = dict(metadata)
        self._start_episode(started_ns=time.perf_counter_ns())

    def end_timing_trajectory(
        self,
        success: bool,
        metadata: dict[str, Any],
    ) -> None:
        if not self._timing_hook_active or self._episode_started_ns is None:
            raise RuntimeError("no timed RoboTwin trajectory is active")
        ended_ns = self._terminal_action_completion_ns
        if ended_ns is None:
            ended_ns = time.perf_counter_ns()
        self._active_timing_metadata.update(metadata)
        self.finish_episode(success=bool(success), ended_ns=ended_ns)

    def reset(self) -> None:
        self.finish_episode(success=None)
        self.pending_actions.clear()
        if self._ac_stream_controller is not None:
            self._ac_stream_controller.close()
            self._ac_stream_controller = None
        try:
            self._request_id += 1
            _send_msg(
                self._conn,
                {"cmd": "reset", "request_id": self._request_id},
            )
            _recv_msg(self._conn)
        except (AttributeError, OSError):
            pass

    def close(self) -> None:
        if self._ac_stream_controller is not None:
            self._ac_stream_controller.close()
            self._ac_stream_controller = None
        close = getattr(self._conn, "close", None)
        if callable(close):
            close()


def _get(usr_args: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = usr_args.get(key, default)
    if _is_none_like(value):
        return default
    return value


def get_model(usr_args: Dict[str, Any]) -> RemoteStreamingWAMModel:
    host = str(_get(usr_args, "server_host", "127.0.0.1"))
    port = int(_get(usr_args, "server_port", 8765))
    replan_steps = int(_get(usr_args, "replan_steps", 24))
    inference_mode = str(_get(usr_args, "inference_mode", "baseline"))
    prewarm = bool(_get(usr_args, "prewarm", True))
    return RemoteStreamingWAMModel(
        host=host,
        port=port,
        replan_steps=replan_steps,
        inference_mode=inference_mode,
        prewarm=prewarm,
    )


def eval(TASK_ENV: Any, model: RemoteStreamingWAMModel, observation: Optional[Dict[str, Any]]) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: RemoteStreamingWAMModel) -> None:
    model.reset()


def prepare_instruction(task_env: Any, model: RemoteStreamingWAMModel) -> None:
    model.prepare_instruction(task_env.get_instruction())


def prewarm_model(
    task_env: Any,
    model: RemoteStreamingWAMModel,
    observation: Dict[str, Any],
) -> None:
    model.prewarm_model(task_env, observation)


def begin_timing_trajectory(
    model: RemoteStreamingWAMModel,
    metadata: dict[str, Any],
) -> None:
    model.begin_timing_trajectory(metadata)


def end_timing_trajectory(
    model: RemoteStreamingWAMModel,
    success: bool,
    metadata: dict[str, Any],
) -> None:
    model.end_timing_trajectory(success, metadata)


def _finish_episode(model: RemoteStreamingWAMModel) -> None:
    model.finish_episode(success=None)
