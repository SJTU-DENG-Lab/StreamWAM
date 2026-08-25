"""Inference-mode and backend contracts for RoboTwin evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceRuntime:
    inference_mode: str
    backend: str

    @property
    def accelerated(self) -> bool:
        return self.backend == "accelerated"


def resolve_inference_runtime(
    inference_mode: str,
    *,
    accelerated: bool = False,
    eager: bool = False,
) -> InferenceRuntime:
    """Validate a public RoboTwin mode/backend selection."""

    mode = str(inference_mode).strip().lower()
    if mode not in {"baseline", "cd", "ac-stream"}:
        raise ValueError(
            f"Unsupported inference mode {inference_mode!r}; expected "
            "'baseline', 'cd', or 'ac-stream'"
        )
    if accelerated and eager:
        raise ValueError("--ac-stream-accelerated and --ac-stream-eager are mutually exclusive")
    if mode != "ac-stream" and (accelerated or eager):
        raise ValueError(
            "AC-Stream backend flags are only valid for inference_mode='ac-stream'"
        )
    backend = "accelerated" if accelerated else "eager"
    return InferenceRuntime(inference_mode=mode, backend=backend)
