from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from streamwam.backbone.wan22 import Wan2_2_VAE, Wan22Backbone
from streamwam.config import BackboneConfig


class _DummyDit(nn.Module):
    def __init__(self, _info) -> None:
        super().__init__()


def test_wan22_backbone_builds_vae_in_model_compute_dtype(tmp_path) -> None:
    (tmp_path / "Wan2.2_VAE.pth").touch()
    config = BackboneConfig(
        type="wan22_5b",
        pretrained_model_id=str(tmp_path),
        load_text_encoder=False,
    )

    with (
        patch("streamwam.backbone.wan22.Wan22Dit", _DummyDit),
        patch(
            "streamwam.backbone.wan22._load_wan22_vae",
            return_value=SimpleNamespace(),
        ),
    ):
        backbone = Wan22Backbone(
            config,
            device="cpu",
            dtype=torch.bfloat16,
            load_vae=True,
            load_dit=False,
            load_text_encoder=False,
        )

    assert backbone.get_vae()._dtype == torch.bfloat16


def test_wan22_vae_weights_use_requested_compute_dtype() -> None:
    with patch(
        "streamwam.backbone.wan22._video_vae",
        return_value=nn.Linear(1, 1),
    ):
        vae = Wan2_2_VAE(
            vae_pth="unused.pt",
            dtype=torch.bfloat16,
            device="cpu",
        )

    assert next(vae.model.parameters()).dtype == torch.bfloat16


def test_wan22_vae_scale_matches_fastwam_float32_construction() -> None:
    with patch(
        "streamwam.backbone.wan22._video_vae",
        return_value=nn.Linear(1, 1),
    ):
        vae = Wan2_2_VAE(
            vae_pth="unused.pt",
            dtype=torch.bfloat16,
            device="cpu",
        )

    expected_std = torch.tensor(
        [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ],
        dtype=torch.float32,
    )

    assert torch.equal(vae.scale[1], expected_std.reciprocal())
