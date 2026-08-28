from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from streamingwam.checkpointing import load_inference_checkpoint, load_inference_stats
from streamingwam.modules.ac_stream import ACStreamSlotEncoder


class TinyExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)


class TinyMoT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleDict(
            {
                "video": TinyExpert(2, 3),
                "action": TinyExpert(4, 2),
            }
        )


class TinyWAM(nn.Module):
    taxonomy_model_family = "mot_wam"

    def __init__(self) -> None:
        super().__init__()
        self.mot = TinyMoT()
        self.proprio_encoder = nn.Linear(5, 2)


class TinyACStreamingWAM(TinyWAM):
    inference_variant = "ac-stream"

    def __init__(self) -> None:
        super().__init__()
        self.mot.experts["video"].rtc_1step_selfatt_z1_slot_encoder = ACStreamSlotEncoder(1024)


class TinyStarWAM(nn.Module):
    taxonomy_model_family = "mot_wam"

    def __init__(self, *, inference_variant: str = "standard") -> None:
        super().__init__()
        self.inference_variant = inference_variant
        self.linear = nn.Linear(2, 3)
        if inference_variant == "ac-stream":
            self.rtc_slot_state_embedding = nn.Embedding(2, 4)


def _filled_state(module: nn.Module, value: float) -> dict[str, torch.Tensor]:
    return {
        key: torch.full_like(tensor, value)
        for key, tensor in module.state_dict().items()
    }


def _fastwam_payload(model: TinyWAM) -> dict:
    video = _filled_state(model.mot.experts["video"], 1.25)
    action = _filled_state(model.mot.experts["action"], 2.5)
    proprio = _filled_state(model.proprio_encoder, 3.75)
    mot = {
        **{f"mixtures.video.{key}": value for key, value in video.items()},
        **{f"mixtures.action.{key}": value for key, value in action.items()},
    }
    return {
        "mot": mot,
        "proprio_encoder": proprio,
        "step": 21700,
        "torch_dtype": "torch.bfloat16",
    }


def _ac_stream_payload(model: TinyACStreamingWAM) -> dict:
    payload = _fastwam_payload(model)
    payload.update(
        {
            "step": 5500,
            "training_rtc_phase": "stage2_1step_selfatt_z1",
            "training_rtc_architecture": (
                "stage1_16slot_z1_only_fixed_residual_plus_d0d8_policy_stream_v1"
            ),
        }
    )
    return payload


def _save_payload(tmp_path: Path, payload: dict, name: str = "checkpoint.pt") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_state_unchanged(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key])


def test_fastwam_loads_video_action_and_proprio_strictly(tmp_path: Path) -> None:
    model = TinyWAM()
    path = _save_payload(tmp_path, _fastwam_payload(model))

    report = load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    for tensor in model.mot.experts["video"].state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 1.25))
    for tensor in model.mot.experts["action"].state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2.5))
    for tensor in model.proprio_encoder.state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 3.75))
    assert report == {
        "checkpoint_file": str(path),
        "checkpoint_format": "fastwam",
        "step": 21700,
        "torch_dtype": "torch.bfloat16",
        "video_tensors": 2,
        "action_tensors": 2,
        "proprio_tensors": 2,
    }


def test_fastwam_shape_mismatch_fails_before_any_parameter_changes(tmp_path: Path) -> None:
    model = TinyWAM()
    before = _snapshot_state(model)
    payload = _fastwam_payload(model)
    payload["mot"]["mixtures.action.linear.weight"] = torch.ones(3, 4)
    path = _save_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="shape mismatch"):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    _assert_state_unchanged(model, before)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("mot"), "missing required key.*mot"),
        (lambda payload: payload.pop("proprio_encoder"), "missing required key.*proprio_encoder"),
        (lambda payload: payload.pop("step"), "missing required key.*step"),
        (lambda payload: payload.pop("torch_dtype"), "missing required key.*torch_dtype"),
        (
            lambda payload: payload["mot"].__setitem__("mixtures.other.weight", torch.ones(1)),
            "unsupported mot keys",
        ),
        (
            lambda payload: payload["mot"].pop("mixtures.video.linear.bias"),
            "video expert.*missing keys",
        ),
        (
            lambda payload: payload["mot"].__setitem__("mixtures.video.extra", torch.ones(1)),
            "video expert.*unexpected keys",
        ),
        (
            lambda payload: payload["mot"].__setitem__("mixtures.video.linear.weight", "bad"),
            "video expert.*must be a tensor",
        ),
    ],
)
def test_fastwam_rejects_malformed_payloads(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    model = TinyWAM()
    before = _snapshot_state(model)
    payload = _fastwam_payload(model)
    mutation(payload)
    path = _save_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    _assert_state_unchanged(model, before)


def test_fastwam_requires_model_proprio_encoder_before_loading(tmp_path: Path) -> None:
    model = TinyWAM()
    payload = _fastwam_payload(model)
    path = _save_payload(tmp_path, payload)
    del model.proprio_encoder
    before = _snapshot_state(model)

    with pytest.raises(ValueError, match="requires model.proprio_encoder"):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    _assert_state_unchanged(model, before)


def test_fastwam_rejects_non_mot_model(tmp_path: Path) -> None:
    model = TinyWAM()
    path = _save_payload(tmp_path, _fastwam_payload(model))
    model.taxonomy_model_family = "shared_dit_wam"

    with pytest.raises(ValueError, match="requires model family 'mot_wam'"):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")


def test_ac_stream_checkpoint_rejects_standard_wam_before_mutation(tmp_path: Path) -> None:
    source = TinyACStreamingWAM()
    model = TinyWAM()
    before = _snapshot_state(model)
    path = _save_payload(tmp_path, _ac_stream_payload(source))

    with pytest.raises(ValueError, match="requires an AC-Stream model variant"):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    _assert_state_unchanged(model, before)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("training_rtc_phase", "wrong", "phase mismatch"),
        ("training_rtc_architecture", "wrong", "architecture mismatch"),
    ],
)
def test_ac_stream_checkpoint_metadata_mismatch_fails_before_mutation(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    model = TinyACStreamingWAM()
    before = _snapshot_state(model)
    payload = _ac_stream_payload(model)
    payload[field] = value
    path = _save_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    _assert_state_unchanged(model, before)


def test_ac_stream_checkpoint_loads_slot_encoder_and_reports_metadata(tmp_path: Path) -> None:
    model = TinyACStreamingWAM()
    payload = _ac_stream_payload(model)
    slot_key = (
        "mixtures.video.rtc_1step_selfatt_z1_slot_encoder.state_embedding.weight"
    )
    payload["mot"][slot_key] = torch.full((2, 1024), 9.0)
    path = _save_payload(tmp_path, payload)

    report = load_inference_checkpoint(model, path, checkpoint_format="fastwam")

    torch.testing.assert_close(
        model.mot.experts["video"].rtc_1step_selfatt_z1_slot_encoder.state_embedding.weight,
        torch.full((2, 1024), 9.0),
    )
    assert report["training_ac_stream_phase"] == "stage2_1step_selfatt_z1"
    assert report["training_ac_stream_architecture"].endswith("policy_stream_v1")


def test_standard_checkpoint_loading_remains_supported(tmp_path: Path) -> None:
    source = nn.Linear(2, 3)
    expected = _filled_state(source, 6.0)
    path = _save_payload(tmp_path, {"model_state_dict": expected, "step": 42})
    target = nn.Linear(2, 3)

    report = load_inference_checkpoint(target, path, checkpoint_format="streamingwam")

    for tensor in target.state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 6.0))
    assert report["step"] == 42
    assert report["checkpoint_format"] == "streamingwam"


def test_starwam_loads_original_top_level_state_dict_strictly(tmp_path: Path) -> None:
    source = TinyStarWAM()
    expected = _filled_state(source, 4.0)
    path = _save_payload(tmp_path, expected, name="starwam.pt")
    target = TinyStarWAM()

    report = load_inference_checkpoint(
        target,
        path,
        checkpoint_format="starwam",
        inference_mode="baseline",
    )

    for tensor in target.state_dict().values():
        torch.testing.assert_close(tensor, torch.full_like(tensor, 4.0))
    assert report["checkpoint_file"] == str(path)
    assert report["checkpoint_format"] == "starwam"
    assert report["inference_mode"] == "baseline"
    assert report["model_tensors"] == len(expected)


def test_starwam_resolves_deepspeed_checkpoint_directory(tmp_path: Path) -> None:
    source = TinyStarWAM()
    state = _filled_state(source, 5.0)
    checkpoint_dir = tmp_path / "checkpoint-2000"
    model_dir = checkpoint_dir / "pytorch_model"
    model_dir.mkdir(parents=True)
    checkpoint_file = model_dir / "mp_rank_00_model_states.pt"
    torch.save({"module": state, "global_steps": 2000}, checkpoint_file)
    target = TinyStarWAM()

    report = load_inference_checkpoint(
        target,
        checkpoint_dir,
        checkpoint_format="starwam",
        inference_mode="cd",
    )

    assert report["checkpoint_file"] == str(checkpoint_file)
    assert report["step"] == 2000
    assert report["inference_mode"] == "cd"


def test_starwam_ac_stream_requires_rtc_slot_before_mutation(tmp_path: Path) -> None:
    source = TinyStarWAM(inference_variant="ac-stream")
    state = _filled_state(source, 6.0)
    del state["rtc_slot_state_embedding.weight"]
    path = _save_payload(tmp_path, {"module": state}, name="rtc.pt")
    target = TinyStarWAM(inference_variant="ac-stream")
    before = _snapshot_state(target)

    with pytest.raises(ValueError, match="rtc_slot_state_embedding.weight"):
        load_inference_checkpoint(
            target,
            path,
            checkpoint_format="starwam",
            inference_mode="ac-stream",
        )

    _assert_state_unchanged(target, before)


def test_starwam_shape_mismatch_fails_before_mutation(tmp_path: Path) -> None:
    source = TinyStarWAM()
    state = _filled_state(source, 7.0)
    state["linear.weight"] = torch.ones(4, 2)
    path = _save_payload(tmp_path, state, name="bad.pt")
    target = TinyStarWAM()
    before = _snapshot_state(target)

    with pytest.raises(ValueError, match="shape mismatch"):
        load_inference_checkpoint(
            target,
            path,
            checkpoint_format="starwam",
            inference_mode="baseline",
        )

    _assert_state_unchanged(target, before)


def test_fastwam_stats_are_canonicalized_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "action": {
                    "default": {
                        "global_min": [-1.0, -2.0],
                        "global_max": [1.0, 2.0],
                        "global_mean": [0.1, 0.2],
                        "global_std": [0.3, 0.4],
                    }
                },
                "state": {
                    "default": {
                        "global_min": [-3.0],
                        "global_max": [3.0],
                        "global_mean": [0.5],
                        "global_std": [0.75],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    stats = load_inference_stats(path, checkpoint_format="fastwam")

    assert set(stats) == {"action", "state"}
    assert set(stats["action"]) == {"min", "max", "mean", "std"}
    torch.testing.assert_close(stats["action"]["min"], torch.tensor([-1.0, -2.0]))
    torch.testing.assert_close(stats["action"]["std"], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(stats["state"]["max"], torch.tensor([3.0]))
    torch.testing.assert_close(stats["state"]["mean"], torch.tensor([0.5]))


def test_fastwam_stats_require_default_global_fields(tmp_path: Path) -> None:
    path = tmp_path / "stats.json"
    path.write_text(
        json.dumps(
            {
                "action": {"default": {"global_min": [0.0]}},
                "state": {"default": {"global_min": [0.0]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action.default.*missing fields"):
        load_inference_stats(path, checkpoint_format="fastwam")


def test_unknown_checkpoint_format_is_rejected(tmp_path: Path) -> None:
    model = TinyWAM()
    path = _save_payload(tmp_path, _fastwam_payload(model))

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_inference_checkpoint(model, path, checkpoint_format="unknown")
