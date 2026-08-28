"""Evaluation metrics for Streaming-WAM frameworks.

Two metrics are exposed:

- ``action_mse``: per-dim mean squared error between predicted action chunk
  and ground-truth action chunk. Returns a ``float`` (mean over batch /
  time / dim) and an optional ``[T, D]`` per-step breakdown.
- ``video_psnr``: PSNR (dB) between predicted video pixels and ground-truth
  video pixels. Inputs are assumed to be in ``[-1, 1]`` (Wan2.2 convention)
  or ``[0, 1]`` — pass ``data_range`` accordingly.

Both metrics tolerate ``is_pad`` masks (broadcastable to the metric tensor
shape; ``True`` means *padded position, exclude from metric*).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _apply_pad_mask(diff_sq: Tensor, is_pad: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Multiply ``diff_sq`` by ``(1 - is_pad)`` and return (sum, count)."""
    if is_pad is None:
        return diff_sq.sum(), torch.tensor(
            float(diff_sq.numel()), device=diff_sq.device, dtype=diff_sq.dtype
        )
    keep = (~is_pad).to(diff_sq.dtype)
    while keep.dim() < diff_sq.dim():
        keep = keep.unsqueeze(-1)
    keep = keep.expand_as(diff_sq)
    return (diff_sq * keep).sum(), keep.sum().clamp_min(1.0)


@torch.no_grad()
def action_mse(
    pred: Tensor,
    target: Tensor,
    is_pad: Optional[Tensor] = None,
    per_step: bool = False,
) -> dict[str, Tensor]:
    """Compute action MSE.

    Args:
        pred:   ``[B, T, D]`` predicted actions.
        target: ``[B, T, D]`` ground-truth actions.
        is_pad: optional ``[B, T]`` boolean mask (True = padded).
        per_step: if True also returns the per-step ``[T]`` MSE.

    Returns:
        dict with keys: ``mse`` (scalar) and optionally ``mse_per_step``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"action_mse: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    total, count = _apply_pad_mask(diff_sq, is_pad)
    out = {"mse": total / count}
    if per_step:
        # Per-step MSE: average over batch and dim, masking out padded steps.
        if is_pad is None:
            out["mse_per_step"] = diff_sq.mean(dim=(0, 2))
        else:
            keep = (~is_pad).to(diff_sq.dtype).unsqueeze(-1).expand_as(diff_sq)
            num = (diff_sq * keep).sum(dim=(0, 2))
            den = keep.sum(dim=(0, 2)).clamp_min(1.0)
            out["mse_per_step"] = num / den
    return out


def action_dim_mse(
    pred: Tensor,
    target: Tensor,
    is_pad: Optional[Tensor] = None,
) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"action_dim_mse: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    if is_pad is None:
        return diff_sq.mean(dim=(0, 1))
    keep = (~is_pad).to(diff_sq.dtype).unsqueeze(-1).expand_as(diff_sq)
    num = (diff_sq * keep).sum(dim=(0, 1))
    den = keep.sum(dim=(0, 1)).clamp_min(1.0)
    return num / den


def action_monitor_metrics(
    pred: Tensor,
    target: Tensor,
    action: Tensor,
    is_pad: Optional[Tensor] = None,
    gripper_dim: int = -1,
) -> dict[str, float]:
    dim_mse = action_dim_mse(pred, target, is_pad=is_pad)
    gripper_idx = gripper_dim if gripper_dim >= 0 else dim_mse.shape[0] + gripper_dim
    if gripper_idx < 0 or gripper_idx >= dim_mse.shape[0]:
        raise ValueError(f"gripper_dim={gripper_dim} out of range for action dim={dim_mse.shape[0]}")

    eef_mask = torch.ones(dim_mse.shape[0], dtype=torch.bool, device=dim_mse.device)
    eef_mask[gripper_idx] = False
    action_f = action.float()
    if is_pad is None:
        gripper_values = action_f[..., gripper_idx]
    else:
        keep = ~is_pad
        gripper_values = action_f[..., gripper_idx][keep]
        if gripper_values.numel() == 0:
            gripper_values = action_f[..., gripper_idx].reshape(-1)

    return {
        "loss_action_eef": float(dim_mse[eef_mask].mean().detach().item()),
        "loss_action_gripper": float(dim_mse[gripper_idx].detach().item()),
        "action_target_gripper_mean": float(gripper_values.mean().detach().item()),
        "action_target_gripper_open_rate": float((gripper_values > 0).float().mean().detach().item()),
    }


@torch.no_grad()
def video_psnr(
    pred: Tensor,
    target: Tensor,
    data_range: float = 2.0,
    reduce: str = "mean",
) -> Tensor:
    """Compute PSNR (dB) between predicted and target video tensors.

    Args:
        pred / target: ``[B, C, T, H, W]`` (or any matching shape). Both are
            expected in the same range; defaults assume ``[-1, 1]`` so
            ``data_range=2.0``. Use ``data_range=1.0`` for ``[0, 1]`` data.
        reduce: ``mean`` returns a scalar; ``per_sample`` returns ``[B]``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"video_psnr: pred {tuple(pred.shape)} != target {tuple(target.shape)}"
        )
    diff_sq = (pred.float() - target.float()) ** 2
    if reduce == "per_sample":
        # Reduce over all non-batch dims.
        dims = tuple(range(1, diff_sq.dim()))
        mse = diff_sq.mean(dim=dims).clamp_min(1e-12)
    else:
        mse = diff_sq.mean().clamp_min(1e-12)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr


@torch.no_grad()
def evaluate_batch(
    model,
    batch: dict,
    *,
    action_horizon: Optional[int] = None,
    num_inference_steps: int = 10,
    compute_video: bool = False,
    num_video_frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict[str, float]:
    """Run ``infer_action`` (and optionally ``infer_joint``) on a batch and
    return a dict of metric name -> float.

    Expects ``batch`` to have keys: ``video``, ``action``, ``context``,
    ``context_mask``, optional ``action_is_pad``. The first frame of
    ``video`` is used as the input image for inference.

    Note: this runs inference per-sample to keep the API simple; for large
    eval sets prefer to batch by picking ``B=1`` slices in the eval loop.
    """
    metrics: dict[str, float] = {}
    video = batch["video"]
    action_gt = batch["action"]
    context = batch["context"]
    context_mask = batch.get("context_mask")
    is_pad = batch.get("action_is_pad")

    B = video.shape[0]
    T_a = action_horizon if action_horizon is not None else action_gt.shape[1]

    pred_actions = []
    for b in range(B):
        first_frame = video[b : b + 1, :, 0]  # [1, C, H, W]
        ctx = context[b : b + 1]
        cmask = context_mask[b : b + 1] if context_mask is not None else None
        pred_a = model.infer_action(
            first_frame, ctx, cmask,
            action_horizon=T_a,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        pred_actions.append(pred_a)
    pred_action = torch.cat(pred_actions, dim=0)
    am = action_mse(pred_action, action_gt[:, :T_a], is_pad=is_pad)
    metrics["action_mse"] = float(am["mse"].item())

    if compute_video:
        if num_video_frames is None:
            num_video_frames = video.shape[2]
        psnrs = []
        for b in range(B):
            first_frame = video[b : b + 1, :, 0]
            ctx = context[b : b + 1]
            cmask = context_mask[b : b + 1] if context_mask is not None else None
            out = model.infer_joint(
                first_frame, ctx, cmask,
                num_video_frames=num_video_frames,
                action_horizon=T_a,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            pred_video = out["video"]
            # Pixel ranges should align; use [-1, 1] default.
            psnrs.append(video_psnr(pred_video, video[b : b + 1, :, : pred_video.shape[2]]))
        metrics["video_psnr"] = float(torch.stack([p.unsqueeze(0) for p in psnrs]).mean().item())
    return metrics
