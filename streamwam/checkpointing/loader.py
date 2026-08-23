"""Public inference checkpoint dispatch by source format."""

from pathlib import Path
from typing import Any

import torch

from streamwam.checkpointing.fastwam_format import (
    load_fastwam_checkpoint,
    load_fastwam_stats,
    prepare_fastwam_config,
)
from streamwam.checkpointing.streamwam_format import load_streamwam_checkpoint


def _normalize_format(checkpoint_format: str) -> str:
    normalized = str(checkpoint_format).strip().lower()
    if normalized not in {"streamwam", "fastwam"}:
        raise ValueError(
            f"Unsupported checkpoint format {normalized!r}; expected 'streamwam' or 'fastwam'"
        )
    return normalized


def load_inference_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    checkpoint_format: str = "streamwam",
) -> dict[str, Any]:
    """Load an inference checkpoint in an explicitly selected source format."""

    normalized = _normalize_format(checkpoint_format)
    checkpoint_path = Path(path)
    if normalized == "streamwam":
        return load_streamwam_checkpoint(model, checkpoint_path)
    return load_fastwam_checkpoint(model, checkpoint_path)


def load_inference_stats(
    path: str | Path,
    checkpoint_format: str = "streamwam",
) -> dict[str, dict[str, torch.Tensor]]:
    """Load action/state statistics matching an inference checkpoint format."""

    normalized = _normalize_format(checkpoint_format)
    if normalized == "streamwam":
        from streamwam.data.lerobot import load_lerobot_stats

        return load_lerobot_stats(path)
    return load_fastwam_stats(path)


def prepare_inference_config(config: Any, checkpoint_format: str = "streamwam") -> Any:
    """Apply source-format requirements before constructing an inference model."""

    normalized = _normalize_format(checkpoint_format)
    if normalized == "streamwam":
        return config
    return prepare_fastwam_config(config)
