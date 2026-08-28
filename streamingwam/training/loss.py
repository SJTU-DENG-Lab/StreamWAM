"""Loss utilities for Streaming-WAM training."""

import torch
from torch import Tensor
from streamingwam.modules.scheduler import FlowMatchScheduler


def flow_matching_loss(
    pred: Tensor,
    target: Tensor,
    timesteps: Tensor,
    scheduler: FlowMatchScheduler,
    is_pad_mask: Tensor | None = None,
) -> Tensor:
    """Compute weighted MSE loss with flow-matching importance sampling.

    Args:
        pred: model prediction [B, ...]
        target: training target [B, ...]
        timesteps: timestep indices [B]
        scheduler: FlowMatchScheduler for computing weights
        is_pad_mask: optional bool mask [B, ...] where True = padded (ignore)
    Returns:
        scalar loss
    """
    # Per-element MSE
    mse = (pred - target) ** 2

    # Average over non-batch dims
    reduce_dims = list(range(1, mse.dim()))
    if is_pad_mask is not None:
        # Mask out padded positions. Broadcast the mask to the tensor layout.
        # Action tensors use [B, T, D], while video latents use [B, C, T, H, W].
        valid_mask = ~is_pad_mask
        if valid_mask.dim() == 2 and mse.dim() == 5:
            valid_mask = valid_mask[:, None, :, None, None]
        while valid_mask.dim() < mse.dim():
            valid_mask = valid_mask.unsqueeze(-1)
        valid_f = valid_mask.to(mse.dtype).expand_as(mse)
        mse = mse * valid_f
        denom = valid_f.sum(dim=reduce_dims).clamp(min=1.0)
        mse = mse.sum(dim=reduce_dims) / denom
    else:
        mse = mse.mean(dim=reduce_dims)

    # Apply per-sample importance weight
    weight = scheduler.training_weight(timesteps).to(mse.device, mse.dtype)
    weighted_mse = mse * weight

    return weighted_mse.mean()
