"""Consistency-distillation boundary functions.

These functions contain no scheduler or model state so the same equations can
be shared by direct CD and future RTC samplers.
"""

import torch
from torch import Tensor


def sample_joint_consistency_noise(
    video_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    *,
    seed: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Sample the two initial noises exactly as FastWAM Joint CD does."""

    video_generator = (
        None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    )
    action_generator = (
        None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
    )
    video = torch.randn(
        video_shape,
        generator=video_generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    action = torch.randn(
        action_shape,
        generator=action_generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    return video, action


def normalize_sampling_method(sampling_method: str) -> str:
    """Normalize the public Euler/consistency sampling vocabulary."""

    method = str(sampling_method).strip().lower()
    aliases = {
        "euler": "euler",
        "flow": "euler",
        "fm": "euler",
        "consistency": "consistency",
        "cd": "consistency",
        "lcm": "consistency",
        "rtc_ac": "rtc_ac",
        "rtc-ac": "rtc_ac",
    }
    if method not in aliases:
        raise ValueError(
            f"Unsupported sampling_method={sampling_method!r}; expected "
            "'euler', 'consistency', or 'rtc_ac'"
        )
    return aliases[method]


def _sigma_view(sigma: Tensor, sample: Tensor) -> Tensor:
    sigma = sigma.to(device=sample.device, dtype=sample.dtype)
    if sigma.ndim == 0:
        return sigma.view(*([1] * sample.ndim))
    if sample.ndim == 3:
        if sigma.ndim == 1:
            return sigma.view(-1, 1, 1)
        if sigma.ndim == 2:
            return sigma.unsqueeze(-1)
    if sample.ndim == 5:
        if sigma.ndim == 1:
            return sigma.view(-1, 1, 1, 1, 1)
        if sigma.ndim == 2:
            return sigma.view(sigma.shape[0], 1, sigma.shape[1], 1, 1)
    raise ValueError(
        f"Cannot broadcast sigma shape {tuple(sigma.shape)} to sample shape "
        f"{tuple(sample.shape)}"
    )


def action_consistency_boundary(
    sample: Tensor,
    velocity: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Recover a clean action prediction from a flow-velocity prediction."""

    sigma_view = _sigma_view(sigma, sample)
    return (sample - sigma_view * velocity).to(dtype=sample.dtype)


def video_consistency_boundary(
    sample: Tensor,
    velocity: Tensor,
    sigma: Tensor,
    sigma_data: float = 0.5,
) -> Tensor:
    """Apply the Karras consistency boundary used by FastWAM Joint CD."""

    if sigma_data <= 0:
        raise ValueError(f"sigma_data must be positive, got {sigma_data}")
    sigma_view = _sigma_view(sigma, sample)
    pred_x0 = sample - sigma_view * velocity
    sigma_float = sigma_view.float()
    sigma_data_sq = float(sigma_data) ** 2
    c_skip = sigma_data_sq / (sigma_float.square() + sigma_data_sq)
    c_out = sigma_float * float(sigma_data) / (
        sigma_float.square() + sigma_data_sq
    ).sqrt()
    return (
        c_skip.to(sample.dtype) * sample + c_out.to(sample.dtype) * pred_x0
    ).to(dtype=sample.dtype)
