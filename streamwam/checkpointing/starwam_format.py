"""Original StarWAM checkpoint adaptation for inference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import torch


_KNOWN_PREFIXES = ("module.", "model.", "_orig_mod.")
_RTC_SLOT_KEY = "rtc_slot_state_embedding.weight"


def _resolve_starwam_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"StarWAM checkpoint path does not exist: {path}")
    for relative in (
        "mp_rank_00_model_states.pt",
        "pytorch_model/mp_rank_00_model_states.pt",
        "model.pt",
        "pytorch_model.bin",
    ):
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Unsupported StarWAM checkpoint directory {path}: expected "
        "pytorch_model/mp_rank_00_model_states.pt or a model checkpoint file"
    )


def _extract_state(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"StarWAM checkpoint payload must be a mapping, got {type(payload).__name__}"
        )
    for key in ("module", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise ValueError("StarWAM checkpoint does not contain a model state mapping")


def _strip_known_prefixes(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in _KNOWN_PREFIXES:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        if new_key in normalized:
            raise ValueError(f"StarWAM checkpoint has duplicate normalized key {new_key!r}")
        normalized[new_key] = value
    return normalized


def _validate_state(source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    source_keys = set(source)
    target_keys = set(target)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    if missing:
        raise ValueError(f"StarWAM checkpoint missing keys: {missing[:20]}")
    if unexpected:
        raise ValueError(f"StarWAM checkpoint unexpected keys: {unexpected[:20]}")
    shape_mismatches: list[str] = []
    for key in sorted(target_keys):
        value = source[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"StarWAM checkpoint key {key!r} must be a tensor, "
                f"got {type(value).__name__}"
            )
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(target[key].shape)}"
            )
    if shape_mismatches:
        raise ValueError(f"StarWAM checkpoint shape mismatch: {shape_mismatches[:20]}")


def _checkpoint_step(payload: Any, checkpoint_file: Path) -> int | None:
    if isinstance(payload, Mapping):
        for key in ("global_steps", "global_step", "step"):
            if key in payload:
                return int(payload[key])
    for parent in (checkpoint_file.parent, *checkpoint_file.parents):
        match = re.fullmatch(r"checkpoint-(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def load_starwam_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    inference_mode: str | None = None,
) -> dict[str, Any]:
    """Load an original StarWAM state without writing a converted copy."""

    mode = str(inference_mode or "baseline").strip().lower()
    if mode not in {"baseline", "cd", "ac-stream"}:
        raise ValueError(f"Unsupported StarWAM inference mode {mode!r}")
    if getattr(model, "taxonomy_model_family", None) != "mot_wam":
        raise ValueError("StarWAM checkpoint loading requires model family 'mot_wam'")
    if mode == "ac-stream" and getattr(model, "inference_variant", None) != "ac-stream":
        raise ValueError("StarWAM AC-Stream checkpoint requires an AC-Stream model variant")

    checkpoint_file = _resolve_starwam_checkpoint_file(path)
    payload = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
        mmap=checkpoint_file.is_file(),
    )
    state = _strip_known_prefixes(_extract_state(payload))
    if mode == "ac-stream" and _RTC_SLOT_KEY not in state:
        raise ValueError(f"StarWAM AC-Stream checkpoint missing {_RTC_SLOT_KEY!r}")

    target = model.state_dict()
    _validate_state(state, target)
    model.load_state_dict(state, strict=True)

    report: dict[str, Any] = {
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_format": "starwam",
        "inference_mode": mode,
        "model_tensors": len(state),
    }
    step = _checkpoint_step(payload, checkpoint_file)
    if step is not None:
        report["step"] = step
    return report


def prepare_starwam_config(config: Any) -> Any:
    """Disable partial initialization because original StarWAM states are complete."""

    taxonomy = getattr(config, "taxonomy", None)
    if getattr(taxonomy, "model_family", "mot_wam") != "mot_wam":
        raise ValueError("StarWAM checkpoint loading requires taxonomy.model_family='mot_wam'")
    framework = getattr(config, "framework", None)
    if framework is None:
        raise ValueError("StarWAM checkpoint loading requires config.framework")
    framework.action_expert_init_from = None
    return config
