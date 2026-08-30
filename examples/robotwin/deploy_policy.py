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
    adapter = _load_adapter(mode)
    model = adapter.get_model(usr_args)
    # Keep the selected adapter on the model so every public deploy-policy
    # hook is forwarded through the same implementation.  RoboTwin imports
    # this module as the stable entry point; evaluators must not depend on a
    # concrete client/local adapter module.
    model._streamingwam_robotwin_adapter = adapter
    return model


def _adapter(model: Any):
    adapter = getattr(model, "_streamingwam_robotwin_adapter", None)
    if adapter is None:
        raise RuntimeError("model was not created by deploy_policy.get_model")
    return adapter


def _required_hook(model: Any, name: str, *args: Any) -> Any:
    hook = getattr(_adapter(model), name, None)
    if not callable(hook):
        raise AttributeError(
            f"RoboTwin policy adapter {_adapter(model).__name__!r} "
            f"does not implement required hook {name!r}"
        )
    return hook(*args)


def eval(TASK_ENV: Any, model: Any, observation: Optional[Dict[str, Any]]) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: Any) -> None:
    model.reset()


def prepare_instruction(task_env: Any, model: Any) -> None:
    _required_hook(model, "prepare_instruction", task_env, model)


def prewarm_model(
    task_env: Any,
    model: Any,
    observation: Dict[str, Any],
) -> None:
    _required_hook(model, "prewarm_model", task_env, model, observation)


def begin_timing_trajectory(model: Any, metadata: Dict[str, Any]) -> None:
    _required_hook(model, "begin_timing_trajectory", model, metadata)


def end_timing_trajectory(
    model: Any,
    success: bool,
    metadata: Dict[str, Any],
) -> None:
    _required_hook(model, "end_timing_trajectory", model, success, metadata)


def _finish_episode(model: Any) -> None:
    finish = getattr(model, "finish_episode", None)
    if callable(finish):
        finish(success=None)
