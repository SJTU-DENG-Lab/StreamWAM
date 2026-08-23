from types import SimpleNamespace

import torch

from examples.libero.rollout import _denormalize_action


def test_fastwam_action_denormalization_does_not_clip_model_output() -> None:
    action = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    stats = {
        "min": torch.zeros(7),
        "max": torch.full((7,), 10.0),
    }
    config = SimpleNamespace(
        data=SimpleNamespace(action_norm_mode="minmax"),
    )

    actual = _denormalize_action(
        action,
        config,
        stats,
        checkpoint_format="fastwam",
    )

    assert actual[0, 0] == 15.0


def test_starwam_action_denormalization_preserves_boundary_clipping() -> None:
    action = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    stats = {
        "min": torch.zeros(7),
        "max": torch.full((7,), 10.0),
    }
    config = SimpleNamespace(
        data=SimpleNamespace(action_norm_mode="minmax"),
    )

    actual = _denormalize_action(
        action,
        config,
        stats,
        checkpoint_format="starwam",
    )

    assert actual[0, 0] == 10.0
