"""Shared-DiT Wan module with action/state register tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from streamwam.backbone.base import BackboneInfo
from streamwam.modules.shared_dit_components import (
    ActionRegisterEncoder,
    CategorySpecificMLP,
    build_shared_dit_attention_mask,
)
from streamwam.modules.wan_block import (
    DiTBlock,
    precompute_freqs_cis_1d,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)


class CausalWanHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        patch_volume = int(patch_size[0] * patch_size[1] * patch_size[2])
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, out_dim * patch_volume)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim ** 0.5)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        base = self.modulation.to(device=t.device, dtype=t.dtype).unsqueeze(0)
        shift, scale = (base + t.unsqueeze(2)).chunk(2, dim=2)
        x = self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        return self.head(x)


class CausalWanModel(nn.Module):
    """Wan shared-DiT that appends action/state registers to video tokens."""

    def __init__(
        self,
        info: BackboneInfo,
        action_dim: int,
        state_dim: int,
        action_horizon: int,
        action_tokens_per_state: int = 4,
        clean_context: str = "full_video",
        checkpoint_blocks: bool = True,
        num_categories: int = 1,
        num_frame_per_block: int = 1,
        num_action_per_block: Optional[int] = None,
        num_state_per_block: int = 1,
    ) -> None:
        super().__init__()
        if clean_context not in {"none", "full_video"}:
            raise ValueError(f"clean_context must be 'none' or 'full_video', got {clean_context!r}")
        self.info = info
        self.hidden_dim = info.hidden_dim
        self.num_heads = info.num_heads
        self.attn_head_dim = info.attn_head_dim
        self.freq_dim = info.freq_dim
        self.text_dim = info.text_dim
        self.patch_size = tuple(info.patch_size)
        self.in_channels = info.in_channels
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.action_tokens_per_state = max(1, int(action_tokens_per_state))
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.num_action_per_block = int(num_action_per_block or self.action_horizon)
        self.num_state_per_block = max(1, int(num_state_per_block))
        self.clean_context = clean_context
        self.checkpoint_blocks = bool(checkpoint_blocks)

        self.patch_embedding = nn.Conv3d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6 * self.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=self.hidden_dim,
                attn_head_dim=self.attn_head_dim,
                num_heads=self.num_heads,
                ffn_dim=info.ffn_dim,
                eps=info.eps,
            ) for _ in range(info.num_layers)
        ])
        self.head = CausalWanHead(self.hidden_dim, self.in_channels, self.patch_size, eps=info.eps)
        self.action_encoder = ActionRegisterEncoder(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            freq_dim=self.freq_dim,
            num_categories=num_categories,
        )
        self.state_encoder = CategorySpecificMLP(
            num_categories=num_categories,
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=num_categories,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
        )
        self._freqs_3d_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        self._freqs_action_cache: dict[str, Tensor] = {}
        self._freqs_state_cache: dict[str, Tensor] = {}

    def _get_3d_freqs(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        key = str(device)
        cached = self._freqs_3d_cache.get(key)
        if cached is None:
            f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(self.attn_head_dim)
            cached = (f_freqs.to(device), h_freqs.to(device), w_freqs.to(device))
            self._freqs_3d_cache[key] = cached
        return cached

    def _get_1d_freqs(self, cache: dict[str, Tensor], length: int, device: torch.device) -> Tensor:
        key = str(device)
        cached = cache.get(key)
        if cached is None or cached.shape[0] < length:
            cached = precompute_freqs_cis_1d(self.attn_head_dim, end=max(length, 1024)).to(device)
            cache[key] = cached
        return cached[:length]

    def _build_video_freqs(self, f: int, h: int, w: int, device: torch.device) -> Tensor:
        freqs_f, freqs_h, freqs_w = self._get_3d_freqs(device)
        ff = freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1)
        fh = freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1)
        fw = freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        return torch.cat([ff, fh, fw], dim=-1).reshape(f * h * w, 1, -1)

    def _build_time_mod(self, timesteps: Tensor, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        batch = timesteps.shape[0]
        if timesteps.dim() == 1:
            timestep_tokens = timesteps.to(dtype=dtype).view(batch, 1).expand(batch, seq_len)
        else:
            timestep_tokens = timesteps.to(dtype=dtype)
            if timestep_tokens.shape[1] != seq_len:
                timestep_tokens = timestep_tokens[:, :1].expand(batch, seq_len)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep_tokens.reshape(-1))
        t = self.time_embedding(t_emb).reshape(batch, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        return t, t_mod

    def _build_video_time_mod(self, timesteps: Tensor, f: int, tokens_per_frame: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        batch = timesteps.shape[0]
        timestep_tokens = timesteps.to(dtype=dtype).view(batch, 1, 1).expand(batch, f, tokens_per_frame).clone()
        timestep_tokens[:, 0] = 0
        timestep_tokens = timestep_tokens.reshape(batch, f * tokens_per_frame)
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep_tokens.reshape(-1))
        t = self.time_embedding(t_emb).reshape(batch, f * tokens_per_frame, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        return t, t_mod

    def _build_context_mask(self, context_mask: Optional[Tensor], batch: int, seq_len: int, device: torch.device) -> Optional[Tensor]:
        if context_mask is None:
            return None
        context_mask = context_mask.to(device=device)
        if context_mask.dim() == 2:
            return context_mask.unsqueeze(1).expand(batch, seq_len, -1)
        return context_mask

    def _build_self_attention_mask(
        self,
        *,
        clean_seq_len: int,
        noisy_seq_len: int,
        action_seq_len: int,
        state_seq_len: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> Tensor:
        return build_shared_dit_attention_mask(
            clean_seq_len=clean_seq_len,
            noisy_seq_len=noisy_seq_len,
            action_seq_len=action_seq_len,
            state_seq_len=state_seq_len,
            tokens_per_frame=tokens_per_frame,
            num_frame_per_block=self.num_frame_per_block,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            device=device,
        )

    def forward(
        self,
        noisy_video: Tensor,
        video_timestep: Tensor,
        context: Tensor,
        noisy_action: Tensor,
        action_timestep: Tensor,
        state: Tensor,
        context_mask: Optional[Tensor] = None,
        clean_video: Optional[Tensor] = None,
        state_is_pad: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        patch_param = self.patch_embedding.weight
        noisy_video = noisy_video.to(device=patch_param.device, dtype=patch_param.dtype)
        noisy_action = noisy_action.to(device=patch_param.device, dtype=patch_param.dtype)
        state = state.to(device=patch_param.device, dtype=patch_param.dtype)
        text_param = self.text_embedding[0].weight
        context = context.to(device=text_param.device, dtype=text_param.dtype)
        if clean_video is not None:
            clean_video = clean_video.to(device=patch_param.device, dtype=patch_param.dtype)
        if state_is_pad is not None:
            state_is_pad = state_is_pad.to(device=state.device)

        x_noisy = self.patch_embedding(noisy_video)
        _, _, f, h, w = x_noisy.shape
        batch = x_noisy.shape[0]
        tokens_per_frame = h * w
        noisy_seq_len = f * tokens_per_frame
        x_noisy = x_noisy.flatten(start_dim=2).transpose(1, 2)
        video_t, video_t_mod = self._build_video_time_mod(video_timestep, f, tokens_per_frame, x_noisy.dtype)
        video_freqs = self._build_video_freqs(f, h, w, x_noisy.device)

        if self.clean_context != "full_video" or clean_video is None:
            raise ValueError("DreamZero-aligned Shared-DiT requires clean_video with shared_dit_clean_context='full_video'")
        x_clean = self.patch_embedding(clean_video).flatten(start_dim=2).transpose(1, 2)
        if x_clean.shape[1] != noisy_seq_len:
            raise ValueError(
                f"clean/noisy token length mismatch: clean={x_clean.shape[1]}, noisy={noisy_seq_len}"
            )
        clean_t, clean_t_mod = self._build_time_mod(torch.zeros_like(video_timestep), noisy_seq_len, x_noisy.dtype)
        clean_seq_len = noisy_seq_len

        token_parts = [x_clean, x_noisy]
        t_mod_parts = [clean_t_mod, video_t_mod]
        freqs_parts = [video_freqs, video_freqs]

        category_ids = torch.zeros(batch, dtype=torch.long, device=x_noisy.device)
        action_seq_len = noisy_action.shape[1]
        if action_timestep.dim() == 1:
            action_timestep_tokens = action_timestep.view(batch, 1).expand(batch, action_seq_len)
        else:
            action_timestep_tokens = action_timestep
        action_tokens = self.action_encoder(noisy_action, action_timestep_tokens, category_ids)
        _, action_t_mod = self._build_time_mod(action_timestep_tokens, action_seq_len, x_noisy.dtype)
        action_freqs = self._get_1d_freqs(self._freqs_action_cache, action_seq_len, x_noisy.device).view(action_seq_len, 1, -1)
        token_parts.append(action_tokens)
        t_mod_parts.append(action_t_mod)
        freqs_parts.append(action_freqs)

        state_tokens = self.state_encoder(state, category_ids)
        if state_is_pad is not None:
            state_tokens = state_tokens.masked_fill(state_is_pad.unsqueeze(-1), 0)
        state_seq_len = state_tokens.shape[1]
        state_timestep = action_timestep_tokens[:, ::max(1, action_seq_len // max(state_seq_len, 1))]
        if state_timestep.shape[1] != state_seq_len:
            state_timestep = action_timestep_tokens[:, :1].expand(batch, state_seq_len)
        _, state_t_mod = self._build_time_mod(state_timestep, state_seq_len, x_noisy.dtype)
        state_freqs = self._get_1d_freqs(self._freqs_state_cache, state_seq_len, x_noisy.device).view(state_seq_len, 1, -1)
        token_parts.append(state_tokens)
        t_mod_parts.append(state_t_mod)
        freqs_parts.append(state_freqs)

        x = torch.cat(token_parts, dim=1)
        t_mod = torch.cat(t_mod_parts, dim=1)
        freqs = torch.cat(freqs_parts, dim=0)
        context = self.text_embedding(context)
        context_mask = self._build_context_mask(context_mask, batch, x.shape[1], x.device)
        self_attn_mask = self._build_self_attention_mask(
            clean_seq_len=clean_seq_len,
            noisy_seq_len=noisy_seq_len,
            action_seq_len=action_seq_len,
            state_seq_len=state_seq_len,
            tokens_per_frame=tokens_per_frame,
            device=x.device,
        )

        for block in self.blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask,
                    self_attn_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask, self_attn_mask)

        noisy_start = clean_seq_len
        video_tokens = x[:, noisy_start:noisy_start + noisy_seq_len]
        action_start = clean_seq_len + noisy_seq_len
        action_tokens = x[:, action_start:action_start + action_seq_len]
        video_pred = self.head(video_tokens, video_t)
        video_pred = self.unpatchify(video_pred, (f, h, w))
        action_pred = self.action_decoder(action_tokens, category_ids)
        return video_pred, action_pred

    def unpatchify(self, tokens_flat: Tensor, grid: tuple[int, int, int]) -> Tensor:
        from einops import rearrange

        f, h, w = grid
        p_t, p_h, p_w = self.patch_size
        return rearrange(
            tokens_flat,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w, x=p_t, y=p_h, z=p_w,
        )
