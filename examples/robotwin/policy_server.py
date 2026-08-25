"""StreamWAM RoboTwin policy inference server (socket-based).

Runs in the Torch/StreamWAM environment and serves action-chunk inference over a
plain TCP socket (length-prefixed pickle, stdlib only) so the RoboTwin
simulation process can live in a separate SAPIEN environment.

The client sends raw RoboTwin observations (three camera RGB frames + the 14-D
state vector + instruction); the server composes the exact 384x320 training
grid, runs flow-matching inference, denormalizes, and returns the action chunk.

Run from the StreamWAM repo root, or with the repo root on PYTHONPATH:
    python -m examples.robotwin.policy_server \
        --config /path/to/recipe.yaml \
        --checkpoint /path/to/checkpoint-XXXX/pytorch_model \
        --override backbone.pretrained_model_id=/path/to/Wan2.2-TI2V-5B \
                   data.action_stats_path=/path/to/action_stats.json \
                   data.state_stats_path=/path/to/action_stats.json \
                   data.text_embedding_cache_dir=/path/to/text_embedding_cache \
        --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import pickle
import platform
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Make the StreamWAM package importable when launched from another working dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from streamwam.data.lerobot import _resize_frames  # noqa: E402
from streamwam.eval.policy import StreamWAMPolicy  # noqa: E402
from examples.robotwin.runtime import resolve_inference_runtime  # noqa: E402


# Length-prefixed pickle framing (stdlib only; mirrored by client_policy.py).
# Reject absurd frame sizes so a stray/non-protocol connection (e.g. a health
# probe sending HTTP bytes) can't be read as a huge length and OOM the server.
_MAX_MSG_BYTES = 512 * 1024 * 1024  # 512 MB hard cap; real frames are a few MB.


def send_msg(conn: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(conn: socket.socket):
    header = _recv_exactly(conn, 8)
    if header is None:
        return None
    (length,) = struct.unpack(">Q", header)
    if length == 0 or length > _MAX_MSG_BYTES:
        # Garbage/oversized length -> not our protocol; drop this connection.
        raise ValueError(f"invalid frame length {length} (max {_MAX_MSG_BYTES})")
    body = _recv_exactly(conn, length)
    if body is None:
        return None
    return pickle.loads(body)


def _build_robotwin_image(head, left, right, device, dtype) -> torch.Tensor:
    """Compose the RoboTwin 3-camera grid identical to training (384x320, [-1, 1])."""

    def chw(arr) -> torch.Tensor:
        a = np.ascontiguousarray(arr)
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)

    top = _resize_frames(chw(head), (256, 320)).float() / 255.0
    left_r = _resize_frames(chw(left), (128, 160)).float() / 255.0
    right_r = _resize_frames(chw(right), (128, 160)).float() / 255.0
    bottom = torch.cat([left_r, right_r], dim=-1)
    frame = torch.cat([top, bottom], dim=-2)
    frame = frame * 2.0 - 1.0
    return frame.to(device=device, dtype=dtype)


def _runtime_metadata(policy: StreamWAMPolicy) -> dict:
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    runtime = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton_version,
        "cuda_version": torch.version.cuda,
    }
    status = getattr(policy.model, "ac_stream_acceleration_status", None)
    if policy.inference_mode == "ac-stream" and callable(status):
        metadata = dict(status())
        metadata["runtime"] = {**runtime, **dict(metadata.get("runtime", {}))}
        return metadata
    return {
        "backend": "eager",
        "compile_active": False,
        "runtime": runtime,
    }


def handle_request(policy: StreamWAMPolicy, req: dict) -> dict:
    """Execute one validated protocol request and return a structured response."""

    request_id = req.get("request_id")
    cmd = req.get("cmd", "infer")
    if cmd == "reset":
        policy.reset()
        return {"ok": True, "request_id": request_id}
    if cmd != "infer":
        raise ValueError(f"unsupported policy command {cmd!r}")
    server_started_ns = time.perf_counter_ns()
    image = _build_robotwin_image(
        req["head"],
        req["left"],
        req["right"],
        policy.device,
        policy.dtype,
    )
    state = np.asarray(req["state"], dtype=np.float32)
    instruction = str(req["instruction"])
    if policy.inference_mode == "ac-stream":
        previous = req.get("previous_action_chunk")
        previous_tensor = (
            None
            if previous is None
            else torch.as_tensor(np.asarray(previous, dtype=np.float32))
        )
        prediction = policy.predict_ac_stream_chunk(
            image,
            state,
            instruction,
            previous_action_chunk=previous_tensor,
            inference_delay=int(req.get("inference_delay", 0)),
        )
        response = {
            "request_id": request_id,
            "action": np.asarray(prediction.env_actions, dtype=np.float32),
            "model_action": prediction.model_actions.detach().cpu().numpy().astype(np.float32),
            "model_inference_ms": float(prediction.inference_ms),
        }
    else:
        prediction = policy.predict_chunk_prediction(image, state, instruction)
        response = {
        "request_id": request_id,
        "action": np.asarray(prediction.env_actions, dtype=np.float32),
        "model_inference_ms": float(prediction.inference_ms),
        }
    response["server_total_ms"] = max(
        float(response["model_inference_ms"]),
        (time.perf_counter_ns() - server_started_ns) / 1e6,
    )
    response["warmup"] = bool(req.get("warmup", False))
    metadata = _runtime_metadata(policy)
    response["backend"] = metadata.get("backend", "eager")
    response["compile_active"] = bool(metadata.get("compile_active", False))
    response["runtime"] = metadata.get("runtime", {})
    return response


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StreamWAM RoboTwin policy inference server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-format",
        choices=("streamwam", "fastwam", "starwam"),
        default="streamwam",
        help="Source checkpoint/stats format (default: streamwam).",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--inference-mode",
        choices=("baseline", "cd", "ac-stream"),
        default="baseline",
    )
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--ac-stream-accelerated", action="store_true")
    backend.add_argument("--ac-stream-eager", action="store_true")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Video/joint flow-matching steps (default: recipe inference.num_inference_steps).",
    )
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=None,
        help="Shared-DiT action steps (default: recipe inference.action_num_inference_steps).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    runtime = resolve_inference_runtime(
        args.inference_mode,
        accelerated=args.ac_stream_accelerated,
        eager=args.ac_stream_eager,
    )

    policy = StreamWAMPolicy(
        config_path=args.config,
        checkpoint=args.checkpoint,
        overrides=list(args.override) if args.override else None,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
        seed=args.seed,
        checkpoint_format=args.checkpoint_format,
        inference_mode=runtime.inference_mode,
    )
    if runtime.accelerated:
        policy.model.enable_ac_stream_acceleration()
    print(f"[streamwam_robotwin_server] model ready on {args.device}; listening on {args.host}:{args.port}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    while True:
        try:
            conn, addr = srv.accept()
        except (ConnectionError, OSError):
            continue
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[streamwam_robotwin_server] client connected: {addr}", flush=True)
        try:
            while True:
                try:
                    req = recv_msg(conn)
                except (ConnectionError, OSError, ValueError, MemoryError, EOFError, pickle.UnpicklingError, struct.error):
                    # Bad/garbage/oversized frame or peer vanished mid-message.
                    # Drop this connection and go back to accept() instead of
                    # crashing the server.
                    break
                if req is None:
                    break
                try:
                    send_msg(conn, handle_request(policy, req))
                except (ConnectionError, OSError):
                    # Peer closed while we were replying; abandon this client.
                    break
                except Exception:  # noqa: BLE001
                    try:
                        send_msg(conn, {"error": traceback.format_exc()})
                    except (ConnectionError, OSError):
                        break
        except Exception:  # noqa: BLE001  keep the server alive across any per-connection failure
            traceback.print_exc()
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print(f"[streamwam_robotwin_server] client disconnected: {addr}", flush=True)


if __name__ == "__main__":
    main()
