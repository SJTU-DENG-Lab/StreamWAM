"""RoboTwin deploy-policy entry point for Streaming-WAM.

RoboTwin imports ``deploy_policy.py`` from ``RoboTwin/policy/<policy_name>`` and
calls get_model / eval / reset_model. This file stays lightweight so it can be
imported in a SAPIEN-only environment. The heavy Torch/Streaming-WAM stack is imported
only when ``policy_mode: local`` is selected.

Modes:
  * ``local``: run Streaming-WAM in the same process/env as RoboTwin.
  * ``client``: talk to ``examples.robotwin.policy_server`` over a socket.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_MODE_TO_MODULE = {
    "local": "local_policy",
    "inprocess": "local_policy",
    "in_process": "local_policy",
    "client": "client_policy",
    "remote": "client_policy",
    "socket": "client_policy",
}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _get(usr_args: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = usr_args.get(key, default)
    if _is_none_like(value):
        return default
    return value


def _infer_mode(usr_args: Dict[str, Any]) -> str:
    mode = _get(usr_args, "policy_mode")
    if mode is None:
        mode = "client" if ("server_host" in usr_args or "server_port" in usr_args) else "local"
    mode = str(mode).strip().lower()
    if mode not in _MODE_TO_MODULE:
        valid = ", ".join(sorted(_MODE_TO_MODULE))
        raise ValueError(f"Unknown Streaming-WAM RoboTwin policy_mode={mode!r}; valid modes: {valid}")
    return mode


def _load_adapter(mode: str):
    module_name = _MODE_TO_MODULE[mode]
    if __package__:
        return importlib.import_module(f".{module_name}", __package__)

    # Fallback for direct file loading outside a package context.
    policy_dir = Path(__file__).resolve().parent
    if str(policy_dir) not in sys.path:
        sys.path.insert(0, str(policy_dir))
    return importlib.import_module(module_name)


def get_model(usr_args: Dict[str, Any]):
    mode = _infer_mode(usr_args)
    return _load_adapter(mode).get_model(usr_args)


def eval(TASK_ENV: Any, model: Any, observation: Optional[Dict[str, Any]]) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: Any) -> None:
    model.reset()


def _finish_episode(model: Any) -> None:
    finish = getattr(model, "finish_episode", None)
    if callable(finish):
        finish(success=None)
