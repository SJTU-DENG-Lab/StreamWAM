import pytest
import torch

from starwam.modules.action_dit import ActionDiT


def _model() -> ActionDiT:
    model = ActionDiT(
        hidden_dim=8,
        action_dim=3,
        ffn_dim=16,
        text_dim=6,
        freq_dim=4,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=1,
        max_seq_len=8,
    )
    return model.eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(2, 3, 3),
        torch.randn(2, 4, 6),
        torch.ones(2, 4, dtype=torch.bool),
    )


def test_action_dit_accepts_batch_timesteps() -> None:
    action, context, context_mask = _inputs()

    state = _model().pre_dit(action, torch.tensor([1000.0, 500.0]), context, context_mask)

    assert state["t_mod"].shape == (2, 6, 8)


def test_action_dit_accepts_tokenwise_timesteps() -> None:
    action, context, context_mask = _inputs()
    timestep = torch.tensor([[1000.0, 1000.0, 1000.0], [0.0, 500.0, 500.0]])

    state = _model().pre_dit(action, timestep, context, context_mask)

    assert state["t_mod"].shape == (2, 3, 6, 8)


def test_action_dit_rejects_wrong_tokenwise_horizon() -> None:
    action, context, context_mask = _inputs()

    with pytest.raises(ValueError, match="action token length"):
        _model().pre_dit(action, torch.ones(2, 2), context, context_mask)
