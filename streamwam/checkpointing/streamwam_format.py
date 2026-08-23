"""Native StreamWAM inference checkpoint loading."""

from pathlib import Path
from typing import Any, Mapping

import torch


def _strip_known_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.", "_orig_mod.")
    output: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        output[new_key] = value
    return output


def _extract_inference_state(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    for key in ("model_state_dict", "module", "state_dict"):
        state = payload.get(key)
        if isinstance(state, dict):
            return state
    return payload


def _checkpoint_metadata(payload: Any, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "checkpoint_file": str(path),
        "checkpoint_format": "streamwam",
    }
    if isinstance(payload, dict):
        for key in ("global_steps", "global_step", "step", "ds_version", "torch_dtype"):
            if key in payload:
                metadata[key] = payload[key]
    return metadata


def _resolve_checkpoint_file(path: Path) -> tuple[Path | None, list[Path]]:
    if path.is_file():
        return path, []
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    for name in (
        "model.pt",
        "pytorch_model.bin",
        "model.bin",
        "mp_rank_00_model_states.pt",
        "pytorch_model/mp_rank_00_model_states.pt",
    ):
        candidate = path / name
        if candidate.is_file():
            return candidate, []
    safetensors_files = sorted(path.glob("*.safetensors"))
    if safetensors_files:
        return None, safetensors_files
    raise FileNotFoundError(
        f"Unsupported checkpoint directory {path}: expected model.pt, pytorch_model.bin, "
        "mp_rank_00_model_states.pt, or *.safetensors."
    )


def load_streamwam_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    """Load a native StreamWAM inference checkpoint non-strictly."""

    checkpoint_file, safetensors_files = _resolve_checkpoint_file(path)
    if safetensors_files:
        from safetensors.torch import load_file

        state: dict[str, Any] = {}
        for shard in safetensors_files:
            state.update(load_file(str(shard), device="cpu"))
        result = model.load_state_dict(_strip_known_prefixes(state), strict=False)
        return {
            "checkpoint_files": [str(shard) for shard in safetensors_files],
            "checkpoint_format": "streamwam",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    assert checkpoint_file is not None
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = _extract_inference_state(payload)
    if not isinstance(state, Mapping):
        raise ValueError(
            f"Checkpoint state at {checkpoint_file} must be a mapping, "
            f"got {type(state).__name__}"
        )
    result = model.load_state_dict(_strip_known_prefixes(state), strict=False)
    metadata = _checkpoint_metadata(payload, checkpoint_file)
    metadata.update(
        {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    )
    return metadata
