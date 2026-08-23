"""RoboTwin policy adapter for StreamWAM (remote / socket client).

This is the SAPIEN-side counterpart to ``examples.robotwin.policy_server``. It
runs inside the RoboTwin environment and needs only ``numpy`` plus the Python
standard library. It forwards raw observations to the StreamWAM inference server
and executes the returned action chunk.

Use this instead of ``examples/robotwin/local_policy.py`` when SAPIEN and the
Torch/StreamWAM stack cannot share one environment.

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


class RemoteStreamWAMModel:
    """Talks to the StreamWAM inference server; manages the replan queue locally."""

    def __init__(self, host: str, port: int, replan_steps: int, connect_timeout: float = 600.0) -> None:
        self.host = host
        self.port = int(port)
        self.replan_steps = int(max(1, replan_steps))
        self.pending_actions: deque[np.ndarray] = deque()
        self._conn = self._connect(connect_timeout)

    def _connect(self, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                conn = socket.create_connection((self.host, self.port), timeout=30.0)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(None)
                print(f"[streamwam_client] connected to policy server {self.host}:{self.port}", flush=True)
                return conn
            except OSError as err:
                last_err = err
                print(f"[streamwam_client] waiting for policy server {self.host}:{self.port} ...", flush=True)
                time.sleep(3.0)
        raise ConnectionError(f"could not reach policy server {self.host}:{self.port}: {last_err}")

    def _infer(self, head, left, right, state, instruction) -> np.ndarray:
        _send_msg(self._conn, {
            "cmd": "infer",
            "head": np.ascontiguousarray(head),
            "left": np.ascontiguousarray(left),
            "right": np.ascontiguousarray(right),
            "state": np.asarray(state, dtype=np.float32),
            "instruction": str(instruction),
        })
        resp = _recv_msg(self._conn)
        if "error" in resp:
            raise RuntimeError(f"policy server error:\n{resp['error']}")
        return np.asarray(resp["action"], dtype=np.float32)

    def step(self, task_env: Any, observation: Optional[Dict[str, Any]]) -> None:
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
        if not self.pending_actions:
            return
        task_env.take_action(self.pending_actions.popleft(), action_type="qpos")

    def reset(self) -> None:
        self.pending_actions.clear()
        try:
            _send_msg(self._conn, {"cmd": "reset"})
            _recv_msg(self._conn)
        except OSError:
            pass


def _get(usr_args: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = usr_args.get(key, default)
    if _is_none_like(value):
        return default
    return value


def get_model(usr_args: Dict[str, Any]) -> RemoteStreamWAMModel:
    host = str(_get(usr_args, "server_host", "127.0.0.1"))
    port = int(_get(usr_args, "server_port", 8765))
    replan_steps = int(_get(usr_args, "replan_steps", 24))
    return RemoteStreamWAMModel(host=host, port=port, replan_steps=replan_steps)


def eval(TASK_ENV: Any, model: RemoteStreamWAMModel, observation: Optional[Dict[str, Any]]) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: RemoteStreamWAMModel) -> None:
    model.reset()
