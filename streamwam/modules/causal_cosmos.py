"""Shared-DiT Cosmos-Predict2 core with action/state register tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from streamwam.backbone.base import BackboneInfo
from streamwam.modules.shared_dit_components import (
    ActionRegisterEncoder,
    CategorySpecificMLP,
    build_shared_dit_attention_mask,
)


def _config_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config[key]
    except (KeyError, TypeError):
        return getattr(config, key, default)


class CausalCosmosModel(nn.Module):
    """Cosmos shared-DiT that packs clean/noisy video plus action/state tokens."""

    def __init__(
        self,
        transformer: nn.Module,
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
        if clean_context != "full_video":
            raise ValueError("Cosmos Shared-DiT currently requires shared_dit_clean_context='full_video'")
        self.transformer = transformer
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

        self._validate_transformer()
        state_horizon = max(1, (self.action_horizon + self.action_tokens_per_state - 1) // self.action_tokens_per_state)
        self.action_pos_embed = nn.Parameter(torch.zeros(1, self.action_horizon, self.hidden_dim))
        self.state_pos_embed = nn.Parameter(torch.zeros(1, state_horizon, self.hidden_dim))
        nn.init.normal_(self.action_pos_embed, std=0.02)
        nn.init.normal_(self.state_pos_embed, std=0.02)

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
        self.action_out_norm = nn.LayerNorm(self.hidden_dim, eps=info.eps)
        self.action_decoder = CategorySpecificMLP(
            num_categories=num_categories,
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
        )
        nn.init.normal_(self.action_decoder.fc2.weight, std=1e-4)
        nn.init.zeros_(self.action_decoder.fc2.bias)

    def _validate_transformer(self) -> None:
        if self.patch_size[0] != 1:
            raise NotImplementedError("Cosmos Shared-DiT currently requires temporal patch_size=1")
        if getattr(self.transformer.config, "img_context_dim_in", None):
            raise NotImplementedError("Cosmos image-context cross-attention is not supported by Shared-DiT yet")
        if len(self.transformer.transformer_blocks) != self.info.num_layers:
            raise ValueError(
                f"Cosmos block count mismatch: {len(self.transformer.transformer_blocks)} != {self.info.num_layers}"
            )
        required = ("patch_embed", "time_embed", "rope", "transformer_blocks", "norm_out", "proj_out")
        missing = [name for name in required if not hasattr(self.transformer, name)]
        if missing:
            raise ValueError(f"Cosmos transformer is missing required modules: {missing}")

    def _prepare_condition_mask(self, latents: Tensor) -> Tensor:
        mask = latents.new_zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        if latents.shape[2] > 0:
            mask[:, :, 0:1] = 1.0
        return mask

    def _prepare_video_input(self, latents: Tensor) -> Tensor:
        batch, _, num_frames, height, width = latents.shape
        hidden_states = torch.cat([latents, self._prepare_condition_mask(latents)], dim=1)
        if _config_get(self.transformer.config, "concat_padding_mask", False):
            padding_mask = latents.new_zeros(batch, 1, height, width)
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)],
                dim=1,
            )
        return hidden_states

    def _patch_video(self, latents: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor], Optional[Tensor], dict[str, int]]:
        hidden_states = self._prepare_video_input(latents)
        batch, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = _config_get(self.transformer.config, "patch_size")
        if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
            raise ValueError(
                "Cosmos patch geometry mismatch: "
                f"video={(num_frames, height, width)}, patch={(p_t, p_h, p_w)}"
            )
        image_rotary_emb = self.transformer.rope(hidden_states, fps=None)
        extra_pos_emb = (
            self.transformer.learnable_pos_embed(hidden_states)
            if _config_get(self.transformer.config, "extra_pos_embed_type", None)
            else None
        )
        tokens = self.transformer.patch_embed(hidden_states).flatten(1, 3)
        meta = {
            "T": num_frames,
            "H": height,
            "W": width,
            "f": num_frames // p_t,
            "h": height // p_h,
            "w": width // p_w,
            "tokens_per_frame": (height // p_h) * (width // p_w),
        }
        return tokens, image_rotary_emb, extra_pos_emb, meta

    def _build_context_mask(self, context_mask: Optional[Tensor], device: torch.device) -> Optional[Tensor]:
        if context_mask is None:
            return None
        context_mask = context_mask.to(device=device)
        if context_mask.dim() == 2:
            return context_mask.unsqueeze(1).unsqueeze(1)
        if context_mask.dim() == 3:
            return context_mask[:, :1, :].unsqueeze(1)
        return context_mask

    def _build_time_embeddings(self, timestep_tokens: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        flat = timestep_tokens.reshape(-1)
        temb, embedded_timestep = self.transformer.time_embed(timestep_tokens, flat)
        batch, seq_len = timestep_tokens.shape
        temb = temb.reshape(batch, seq_len, -1).to(dtype=dtype)
        embedded_timestep = embedded_timestep.reshape(batch, seq_len, -1).to(dtype=dtype)
        return temb, embedded_timestep

    def _build_video_timesteps(self, timesteps: Tensor, f: int, tokens_per_frame: int, dtype: torch.dtype) -> Tensor:
        batch = timesteps.shape[0]
        timestep_tokens = timesteps.to(dtype=dtype).view(batch, 1, 1).expand(batch, f, tokens_per_frame).clone()
        timestep_tokens[:, 0] = 0
        return timestep_tokens.reshape(batch, f * tokens_per_frame)

    def _build_state_timesteps(self, action_timestep_tokens: Tensor, state_seq_len: int) -> Tensor:
        action_seq_len = action_timestep_tokens.shape[1]
        state_timestep = action_timestep_tokens[:, ::max(1, action_seq_len // max(state_seq_len, 1))]
        if state_timestep.shape[1] != state_seq_len:
            state_timestep = action_timestep_tokens[:, :1].expand(action_timestep_tokens.shape[0], state_seq_len)
        return state_timestep

    def _project_text_context(self, context: Tensor) -> Tensor:
        if _config_get(self.transformer.config, "use_crossattn_projection", False):
            return self.transformer.crossattn_proj(context)
        return context

    def _apply_video_rope(
        self,
        tensor: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
        clean_seq_len: int,
        noisy_seq_len: int,
    ) -> Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        clean = apply_rotary_emb(
            tensor[:, :, :clean_seq_len, :], image_rotary_emb, use_real=True, use_real_unbind_dim=-2
        )
        noisy_start = clean_seq_len
        noisy_end = noisy_start + noisy_seq_len
        noisy = apply_rotary_emb(
            tensor[:, :, noisy_start:noisy_end, :], image_rotary_emb, use_real=True, use_real_unbind_dim=-2
        )
        rest = tensor[:, :, noisy_end:, :]
        return torch.cat([clean, noisy, rest], dim=2)

    def _cosmos_block_forward(
        self,
        block: nn.Module,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        embedded_timestep: Tensor,
        temb: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor],
        extra_pos_emb: Optional[Tensor],
        clean_seq_len: int,
        noisy_seq_len: int,
        attention_mask: Optional[Tensor],
        self_attn_mask: Tensor,
    ) -> Tensor:
        block_input = hidden_states
        if extra_pos_emb is not None:
            zeros = hidden_states.new_zeros(hidden_states.shape[0], hidden_states.shape[1] - 2 * clean_seq_len, hidden_states.shape[2])
            block_input = hidden_states + torch.cat([extra_pos_emb, extra_pos_emb, zeros], dim=1)

        norm_hidden_states, gate = block.norm1(block_input, embedded_timestep, temb)
        attn = block.attn1
        query = attn.to_q(norm_hidden_states)
        key = attn.to_k(norm_hidden_states)
        value = attn.to_v(norm_hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = self._apply_video_rope(query, image_rotary_emb, clean_seq_len, noisy_seq_len)
        key = self._apply_video_rope(key, image_rotary_emb, clean_seq_len, noisy_seq_len)

        query_idx = query.size(3)
        key_idx = key.size(3)
        value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=self_attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).flatten(2, 3).type_as(query)
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = block.norm2(hidden_states, embedded_timestep, temb)
        attn_output = block.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = block.norm3(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * block.ff(norm_hidden_states)
        return hidden_states

    def _unpatchify(self, tokens: Tensor, meta: dict[str, int]) -> Tensor:
        p_t, p_h, p_w = _config_get(self.transformer.config, "patch_size")
        hidden_states = tokens.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (meta["f"], meta["h"], meta["w"]))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

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
        patch_param = self.transformer.patch_embed.proj.weight
        noisy_video = noisy_video.to(device=patch_param.device, dtype=patch_param.dtype)
        noisy_action = noisy_action.to(device=patch_param.device, dtype=patch_param.dtype)
        state = state.to(device=patch_param.device, dtype=patch_param.dtype)
        context = context.to(device=patch_param.device, dtype=patch_param.dtype)
        if clean_video is None:
            raise ValueError("Cosmos Shared-DiT requires clean_video with shared_dit_clean_context='full_video'")
        clean_video = clean_video.to(device=patch_param.device, dtype=patch_param.dtype)
        if state_is_pad is not None:
            state_is_pad = state_is_pad.to(device=state.device)

        x_noisy, image_rotary_emb, extra_pos_emb, meta = self._patch_video(noisy_video)
        x_clean, clean_rotary_emb, clean_extra_pos_emb, clean_meta = self._patch_video(clean_video)
        if x_clean.shape[1] != x_noisy.shape[1]:
            raise ValueError(f"clean/noisy token length mismatch: clean={x_clean.shape[1]}, noisy={x_noisy.shape[1]}")
        if clean_meta != meta:
            raise ValueError(f"clean/noisy video geometry mismatch: clean={clean_meta}, noisy={meta}")
        if any(a.shape != b.shape for a, b in zip(image_rotary_emb, clean_rotary_emb)):
            raise ValueError("clean/noisy Cosmos rotary embedding shape mismatch")
        if extra_pos_emb is None:
            extra_pos = None
        else:
            extra_pos = extra_pos_emb
            if clean_extra_pos_emb is not None and clean_extra_pos_emb.shape != extra_pos_emb.shape:
                raise ValueError("clean/noisy Cosmos extra position embeddings do not match")

        batch = x_noisy.shape[0]
        tokens_per_frame = meta["tokens_per_frame"]
        clean_seq_len = x_clean.shape[1]
        noisy_seq_len = x_noisy.shape[1]
        video_timestep_tokens = self._build_video_timesteps(video_timestep, meta["f"], tokens_per_frame, x_noisy.dtype)
        clean_timestep_tokens = torch.zeros_like(video_timestep_tokens)
        video_temb, video_embedded_timestep = self._build_time_embeddings(video_timestep_tokens, x_noisy.dtype)
        clean_temb, clean_embedded_timestep = self._build_time_embeddings(clean_timestep_tokens, x_noisy.dtype)

        category_ids = torch.zeros(batch, dtype=torch.long, device=x_noisy.device)
        action_seq_len = noisy_action.shape[1]
        if action_seq_len > self.action_pos_embed.shape[1]:
            raise ValueError(f"action length {action_seq_len} exceeds configured action_horizon={self.action_pos_embed.shape[1]}")
        if action_timestep.dim() == 1:
            action_timestep_tokens = action_timestep.view(batch, 1).expand(batch, action_seq_len)
        else:
            action_timestep_tokens = action_timestep
        action_tokens = self.action_encoder(noisy_action, action_timestep_tokens, category_ids)
        action_tokens = action_tokens + self.action_pos_embed[:, :action_seq_len].to(action_tokens.dtype)
        action_temb, action_embedded_timestep = self._build_time_embeddings(action_timestep_tokens.to(dtype=x_noisy.dtype), x_noisy.dtype)

        state_tokens = self.state_encoder(state, category_ids)
        if state_is_pad is not None:
            state_tokens = state_tokens.masked_fill(state_is_pad.unsqueeze(-1), 0)
        state_seq_len = state_tokens.shape[1]
        if state_seq_len > self.state_pos_embed.shape[1]:
            raise ValueError(f"state length {state_seq_len} exceeds configured state_horizon={self.state_pos_embed.shape[1]}")
        state_tokens = state_tokens + self.state_pos_embed[:, :state_seq_len].to(state_tokens.dtype)
        state_timestep_tokens = self._build_state_timesteps(action_timestep_tokens, state_seq_len)
        state_temb, state_embedded_timestep = self._build_time_embeddings(state_timestep_tokens.to(dtype=x_noisy.dtype), x_noisy.dtype)

        hidden_states = torch.cat([x_clean, x_noisy, action_tokens, state_tokens], dim=1)
        temb = torch.cat([clean_temb, video_temb, action_temb, state_temb], dim=1)
        embedded_timestep = torch.cat(
            [clean_embedded_timestep, video_embedded_timestep, action_embedded_timestep, state_embedded_timestep],
            dim=1,
        )
        context = self._project_text_context(context)
        context_mask = self._build_context_mask(context_mask, hidden_states.device)
        self_attn_mask = build_shared_dit_attention_mask(
            clean_seq_len=clean_seq_len,
            noisy_seq_len=noisy_seq_len,
            action_seq_len=action_seq_len,
            state_seq_len=state_seq_len,
            tokens_per_frame=tokens_per_frame,
            num_frame_per_block=self.num_frame_per_block,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            device=hidden_states.device,
        )

        for block in self.transformer.transformer_blocks:
            if self.training and self.checkpoint_blocks and torch.is_grad_enabled():
                hidden_states = checkpoint(
                    self._cosmos_block_forward,
                    block,
                    hidden_states,
                    context,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos,
                    clean_seq_len,
                    noisy_seq_len,
                    context_mask,
                    self_attn_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._cosmos_block_forward(
                    block,
                    hidden_states,
                    context,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos,
                    clean_seq_len,
                    noisy_seq_len,
                    context_mask,
                    self_attn_mask,
                )

        noisy_start = clean_seq_len
        video_tokens = hidden_states[:, noisy_start:noisy_start + noisy_seq_len]
        action_start = clean_seq_len + noisy_seq_len
        action_tokens = hidden_states[:, action_start:action_start + action_seq_len]
        video_tokens = self.transformer.norm_out(video_tokens, video_embedded_timestep, video_temb)
        video_tokens = self.transformer.proj_out(video_tokens)
        video_pred = self._unpatchify(video_tokens, meta)
        action_tokens = self.action_out_norm(action_tokens)
        action_pred = self.action_decoder(action_tokens, category_ids)
        return video_pred, action_pred
