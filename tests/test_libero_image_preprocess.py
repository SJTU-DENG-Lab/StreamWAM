from types import SimpleNamespace

import numpy as np
import torch

from examples.libero.rollout import _obs_to_image


def _fixture_obs() -> dict[str, np.ndarray]:
    image = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3) * 21
    return {
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 1,
    }


def _fixture_config() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            video_size=[2, 2],
            video_keys=["image", "wrist_image"],
            concat_multi_camera="horizontal",
        )
    )


def test_fastwam_image_normalization_matches_reference_bfloat16_order() -> None:
    actual, images = _obs_to_image(
        _fixture_obs(),
        _fixture_config(),
        checkpoint_format="fastwam",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    expected = torch.as_tensor(images["concat"]).permute(2, 0, 1).unsqueeze(0)
    expected = expected.to(dtype=torch.bfloat16) * (2.0 / 255.0) - 1.0

    assert torch.equal(actual, expected)


def test_starwam_image_normalization_preserves_float32_path() -> None:
    actual, images = _obs_to_image(
        _fixture_obs(),
        _fixture_config(),
        checkpoint_format="starwam",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    expected = torch.as_tensor(images["concat"], dtype=torch.float32)
    expected = expected.permute(2, 0, 1).unsqueeze(0)
    expected = (expected * (2.0 / 255.0) - 1.0).to(torch.bfloat16)

    assert torch.equal(actual, expected)
