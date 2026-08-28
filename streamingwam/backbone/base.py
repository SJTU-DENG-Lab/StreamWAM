"""Base backbone abstract class and BackboneInfo dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional


@dataclass
class BackboneInfo:
    """Architecture parameters auto-inferred from checkpoint. Never manually set by user."""

    hidden_dim: int
    num_layers: int
    num_heads: int
    attn_head_dim: int
    ffn_dim: int
    text_dim: int
    freq_dim: int
    eps: float
    patch_size: tuple
    in_channels: int  # VAE latent channels


class BaseBackbone(ABC, nn.Module):
    """Abstract backbone providing video generation DiT + VAE + text encoder.

    Subclasses (Wan22Backbone, etc.) load from pretrained checkpoints and
    auto-populate BackboneInfo from the loaded weights/config.
    """

    @property
    @abstractmethod
    def info(self) -> BackboneInfo:
        """Return auto-inferred architecture config from loaded weights."""
        ...

    @abstractmethod
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to latent space.

        Args:
            video: [B, 3, T, H, W] normalized to [-1, 1]
        Returns:
            latents: [B, C, T', H', W'] where C=in_channels
        """
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to video.

        Args:
            latents: [B, C, T', H', W']
        Returns:
            video: [B, 3, T, H, W]
        """
        ...

    @abstractmethod
    def encode_text(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts.

        Args:
            text: list of strings
        Returns:
            (context_emb [B, L, text_dim], context_mask [B, L])
        """
        ...

    @abstractmethod
    def get_dit(self) -> nn.Module:
        """Return the DiT (video expert) module for MoT composition.

        The returned module must have:
        - .blocks: nn.ModuleList of DiTBlock-compatible blocks
        - Each block must have .get_qkv() and .post_attention() methods
        """
        ...

    @abstractmethod
    def get_vae(self) -> nn.Module:
        """Return the VAE module."""
        ...

    def build_shared_dit_core(
        self,
        framework_config,
        *,
        state_dim: int,
        action_tokens_per_state: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Build a backbone-specific shared-DiT core for SharedDiTWAM.

        SharedDiTWAM owns the training objective and loss computation; each
        backbone owns how its video DiT is extended with action/state register
        tokens and how pretrained video weights are initialized.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement shared-DiT core construction")
