"""StarWAM RoboTwin policy inference server (socket-based).

Runs in the Torch/StarWAM environment and serves action-chunk inference over a
plain TCP socket (length-prefixed pickle, stdlib only) so the RoboTwin
simulation process can live in a separate SAPIEN environment.

The client sends raw RoboTwin observations (three camera RGB frames + the 14-D
state vector + instruction); the server composes the exact 384x320 training
grid, runs flow-matching inference, denormalizes, and returns the action chunk.

Run from the StarWAM repo root, or with the repo root on PYTHONPATH:
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
import socket
import struct
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# Make the StarWAM package importable when launched from another working dir.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starwam.data.lerobot import _resize_frames  # noqa: E402
from starwam.eval.policy import StarwamPolicy  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="StarWAM RoboTwin policy inference server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--device", default="cuda:0")
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
    args = parser.parse_args()

    policy = StarwamPolicy(
        config_path=args.config,
        checkpoint=args.checkpoint,
        overrides=list(args.override) if args.override else None,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        action_num_inference_steps=args.action_num_inference_steps,
        seed=args.seed,
    )
    print(f"[starwam_robotwin_server] model ready on {args.device}; listening on {args.host}:{args.port}", flush=True)

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
        print(f"[starwam_robotwin_server] client connected: {addr}", flush=True)
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
                cmd = req.get("cmd", "infer")
                try:
                    if cmd == "reset":
                        policy.reset()
                        send_msg(conn, {"ok": True})
                        continue
                    image = _build_robotwin_image(
                        req["head"], req["left"], req["right"], policy.device, policy.dtype
                    )
                    state = np.asarray(req["state"], dtype=np.float32)
                    chunk = policy.predict_chunk(image, state, str(req["instruction"]))
                    send_msg(conn, {"action": np.asarray(chunk, dtype=np.float32)})
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
            print(f"[starwam_robotwin_server] client disconnected: {addr}", flush=True)


if __name__ == "__main__":
    main()
