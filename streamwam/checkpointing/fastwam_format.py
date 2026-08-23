"""FastWAM release checkpoint and statistics adaptation."""

import json
from pathlib import Path
from typing import Any, Mapping

import torch


RTC_AC_PHASE = "stage2_1step_selfatt_z1"
RTC_AC_ARCHITECTURE = (
    "stage1_16slot_z1_only_fixed_residual_plus_d0d8_policy_stream_v1"
)
RTC_AC_SLOT_KEY = (
    "mixtures.video.rtc_1step_selfatt_z1_slot_encoder.state_embedding.weight"
)


def _validate_exact_state(
    label: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    source_keys = set(source)
    target_keys = set(target)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    if missing:
        raise ValueError(f"FastWAM {label} missing keys: {missing[:20]}")
    if unexpected:
        raise ValueError(f"FastWAM {label} unexpected keys: {unexpected[:20]}")

    shape_mismatches = []
    for key in sorted(target_keys):
        source_value = source[key]
        target_value = target[key]
        if not isinstance(source_value, torch.Tensor):
            raise ValueError(
                f"FastWAM {label} key {key!r} must be a tensor, "
                f"got {type(source_value).__name__}"
            )
        if tuple(source_value.shape) != tuple(target_value.shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(source_value.shape)} model={tuple(target_value.shape)}"
            )
    if shape_mismatches:
        raise ValueError(f"FastWAM {label} shape mismatch: {shape_mismatches[:20]}")


def load_fastwam_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    """Load an original FastWAM release checkpoint directly into StreamWAM MoT."""

    if not path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint must be a direct .pt file: {path}")
    if getattr(model, "taxonomy_model_family", None) != "mot_wam":
        raise ValueError("FastWAM checkpoint loading requires model family 'mot_wam'")
    mot_module = getattr(model, "mot", None)
    experts = getattr(mot_module, "experts", None)
    if experts is None or "video" not in experts or "action" not in experts:
        raise ValueError("FastWAM checkpoint loading requires model.mot.experts['video'/'action']")
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is None:
        raise ValueError("FastWAM checkpoint loading requires model.proprio_encoder")

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"FastWAM checkpoint payload must be a dict, got {type(payload).__name__}")
    for required_key in ("mot", "proprio_encoder", "step", "torch_dtype"):
        if required_key not in payload:
            raise ValueError(f"FastWAM checkpoint missing required key {required_key!r}")

    mot_state = payload["mot"]
    proprio_state = payload["proprio_encoder"]
    if not isinstance(mot_state, Mapping):
        raise ValueError("FastWAM checkpoint key 'mot' must contain a state-dict mapping")
    if not isinstance(proprio_state, Mapping):
        raise ValueError(
            "FastWAM checkpoint key 'proprio_encoder' must contain a state-dict mapping"
        )

    model_variant = getattr(model, "inference_variant", "standard")
    has_rtc_metadata = (
        "training_rtc_phase" in payload
        or "training_rtc_architecture" in payload
    )
    if has_rtc_metadata:
        if model_variant != "rtc_ac":
            raise ValueError(
                "FastWAM RTC-AC checkpoint requires an RTC-AC model variant"
            )
        phase = payload.get("training_rtc_phase")
        if phase != RTC_AC_PHASE:
            raise ValueError(
                f"FastWAM RTC-AC phase mismatch: expected {RTC_AC_PHASE!r}, "
                f"got {phase!r}"
            )
        architecture = payload.get("training_rtc_architecture")
        if architecture != RTC_AC_ARCHITECTURE:
            raise ValueError(
                "FastWAM RTC-AC architecture mismatch: expected "
                f"{RTC_AC_ARCHITECTURE!r}, got {architecture!r}"
            )
        slot_value = mot_state.get(RTC_AC_SLOT_KEY)
        if not isinstance(slot_value, torch.Tensor):
            raise ValueError(
                f"FastWAM RTC-AC checkpoint is missing tensor {RTC_AC_SLOT_KEY!r}"
            )
        if tuple(slot_value.shape) != (2, 1024):
            raise ValueError(
                f"FastWAM RTC-AC slot tensor must have shape (2, 1024), got "
                f"{tuple(slot_value.shape)}"
            )
    elif model_variant == "rtc_ac":
        raise ValueError(
            "RTC-AC model variant requires FastWAM RTC-AC checkpoint metadata"
        )

    video_prefix = "mixtures.video."
    action_prefix = "mixtures.action."
    unsupported = sorted(
        key
        for key in mot_state
        if not key.startswith(video_prefix) and not key.startswith(action_prefix)
    )
    if unsupported:
        raise ValueError(f"FastWAM checkpoint has unsupported mot keys: {unsupported[:20]}")

    video_state = {
        key[len(video_prefix):]: value
        for key, value in mot_state.items()
        if key.startswith(video_prefix)
    }
    action_state = {
        key[len(action_prefix):]: value
        for key, value in mot_state.items()
        if key.startswith(action_prefix)
    }

    video_target = experts["video"].state_dict()
    action_target = experts["action"].state_dict()
    proprio_target = proprio_encoder.state_dict()
    _validate_exact_state("video expert", video_state, video_target)
    _validate_exact_state("action expert", action_state, action_target)
    _validate_exact_state("proprio encoder", proprio_state, proprio_target)

    experts["video"].load_state_dict(video_state, strict=True)
    experts["action"].load_state_dict(action_state, strict=True)
    proprio_encoder.load_state_dict(proprio_state, strict=True)

    report = {
        "checkpoint_file": str(path),
        "checkpoint_format": "fastwam",
        "step": payload["step"],
        "torch_dtype": payload["torch_dtype"],
        "video_tensors": len(video_state),
        "action_tensors": len(action_state),
        "proprio_tensors": len(proprio_state),
    }
    if has_rtc_metadata:
        report.update(
            {
                "training_rtc_phase": payload["training_rtc_phase"],
                "training_rtc_architecture": payload["training_rtc_architecture"],
            }
        )
    return report


def load_fastwam_stats(path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """Map FastWAM release statistics to StreamWAM's canonical tensor fields."""

    stats_path = Path(path)
    with open(stats_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"FastWAM stats at {stats_path} must be a JSON object")

    field_map = {
        "global_min": "min",
        "global_max": "max",
        "global_mean": "mean",
        "global_std": "std",
    }
    canonical: dict[str, dict[str, torch.Tensor]] = {}
    for group_name in ("action", "state"):
        group = raw.get(group_name)
        default = group.get("default") if isinstance(group, dict) else None
        if not isinstance(default, dict):
            raise ValueError(f"FastWAM stats missing mapping {group_name}.default")
        missing = [source for source in field_map if source not in default]
        if missing:
            raise ValueError(f"FastWAM stats {group_name}.default missing fields: {missing}")
        canonical[group_name] = {
            target: torch.as_tensor(default[source], dtype=torch.float32)
            for source, target in field_map.items()
        }
    return canonical


def prepare_fastwam_config(config: Any) -> Any:
    """Prepare a MoT inference config for a complete FastWAM checkpoint."""

    taxonomy = getattr(config, "taxonomy", None)
    model_family = getattr(taxonomy, "model_family", "mot_wam")
    if model_family != "mot_wam":
        raise ValueError("FastWAM checkpoint loading requires taxonomy.model_family='mot_wam'")
    framework = getattr(config, "framework", None)
    if framework is None:
        raise ValueError("FastWAM checkpoint loading requires config.framework")
    framework.action_expert_init_from = None
    return config
