"""Cosmos-Predict2 backbone adapter for Streaming-WAM MoT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamingwam.backbone.base import BaseBackbone, BackboneInfo
from streamingwam.config import BackboneConfig


def _patch_numpy_compat() -> None:
    import sys
    import types

    import numpy as np

    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if "soxr" not in sys.modules:
        sys.modules["soxr"] = types.ModuleType("soxr")


def _load_cosmos_transformer(model_dir: Path, dtype: torch.dtype):
    _patch_numpy_compat()
    from diffusers import CosmosTransformer3DModel

    return CosmosTransformer3DModel.from_pretrained(
        str(model_dir), subfolder="transformer", torch_dtype=dtype
    )


def _load_cosmos_vae(model_dir: Path, dtype: torch.dtype):
    _patch_numpy_compat()
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler

    vae = AutoencoderKLWan.from_pretrained(str(model_dir), subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(model_dir), subfolder="scheduler")
    return vae, scheduler


def _load_cosmos_t5(model_dir: Path, dtype: torch.dtype):
    _patch_numpy_compat()
    from transformers import T5EncoderModel, T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(str(model_dir), subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        str(model_dir), subfolder="text_encoder", torch_dtype=dtype
    )
    return tokenizer, text_encoder


def _config_get(config, key: str, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config[key]
    except (KeyError, TypeError):
        return getattr(config, key, default)


def _infer_cosmos_info(model_dir: Path) -> BackboneInfo:
    config_path = model_dir / "transformer" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cosmos transformer config not found: {config_path}")
    cfg = json.loads(config_path.read_text())
    num_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["attention_head_dim"])
    hidden_dim = num_heads * head_dim
    mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
    return BackboneInfo(
        hidden_dim=hidden_dim,
        num_layers=int(cfg["num_layers"]),
        num_heads=num_heads,
        attn_head_dim=head_dim,
        ffn_dim=int(hidden_dim * mlp_ratio),
        text_dim=int(cfg.get("text_embed_dim", 1024)),
        freq_dim=hidden_dim,
        eps=1e-6,
        patch_size=tuple(cfg.get("patch_size", [1, 2, 2])),
        in_channels=int(cfg.get("out_channels", 16)),
    )


class CosmosPredict2BlockAdapter(nn.Module):
    """Expose a native Cosmos block through the MoT expert block contract."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.num_heads = int(block.attn1.heads)
        self.attn_head_dim = int(block.attn1.to_q.out_features // block.attn1.heads)

    def _project_self_qkv(self, hidden_states: torch.Tensor, image_rotary_emb):
        attn = self.block.attn1
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        query_idx = query.size(3)
        key_idx = key.size(3)
        value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        query = query.transpose(1, 2).flatten(2, 3)
        key = key.transpose(1, 2).flatten(2, 3)
        value = value.transpose(1, 2).flatten(2, 3)
        return query, key, value

    def get_qkv(self, x: torch.Tensor, t_mod: dict, freqs):
        embedded_timestep = t_mod["embedded_timestep"]
        temb = t_mod["temb"]
        extra_pos_emb = t_mod.get("extra_pos_emb")
        hidden_states = x
        before_proj = getattr(self.block, "before_proj", None)
        if before_proj is not None:
            hidden_states = before_proj(hidden_states) + t_mod["latents"]
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        norm_hidden_states, gate = self.block.norm1(hidden_states, embedded_timestep, temb)
        q, k, v = self._project_self_qkv(norm_hidden_states, freqs)
        self._cache = {
            "x_id": id(x),
            "hidden_states": hidden_states,
            "gate_msa": gate,
            "embedded_timestep": embedded_timestep,
            "temb": temb,
        }
        return q, k, v

    def _context_mask(self, context_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context_mask is None:
            return None
        if context_mask.dim() == 2:
            return context_mask.unsqueeze(1).unsqueeze(1)
        if context_mask.dim() == 3:
            return context_mask[:, :1, :].unsqueeze(1)
        return context_mask

    def post_attention(
        self,
        x: torch.Tensor,
        attn_output: torch.Tensor,
        t_mod: dict,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cache = getattr(self, "_cache", None)
        if cache is not None and cache.get("x_id") == id(x):
            hidden_states = cache["hidden_states"]
            gate_msa = cache["gate_msa"]
            embedded_timestep = cache["embedded_timestep"]
            temb = cache["temb"]
        else:
            hidden_states = x
            embedded_timestep = t_mod["embedded_timestep"]
            temb = t_mod["temb"]
            _, gate_msa = self.block.norm1(hidden_states, embedded_timestep, temb)

        attn = self.block.attn1
        attn_output = attn.to_out[0](attn_output)
        attn_output = attn.to_out[1](attn_output)
        hidden_states = hidden_states + gate_msa * attn_output

        norm_hidden_states, gate = self.block.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.block.attn2(
            norm_hidden_states,
            encoder_hidden_states=context,
            attention_mask=self._context_mask(context_mask),
        )
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = self.block.norm3(hidden_states, embedded_timestep, temb)
        hidden_states = hidden_states + gate * self.block.ff(norm_hidden_states)
        after_proj = getattr(self.block, "after_proj", None)
        if after_proj is not None:
            hidden_states = after_proj(hidden_states)
        return hidden_states


class CosmosPredict2Dit(nn.Module):
    """MoT-compatible adapter around diffusers ``CosmosTransformer3DModel``."""

    def __init__(self, transformer: nn.Module, info: BackboneInfo):
        super().__init__()
        self.transformer = transformer
        self.info = info
        self.hidden_dim = info.hidden_dim
        self.num_heads = info.num_heads
        self.attn_head_dim = info.attn_head_dim
        self.num_layers = info.num_layers
        self.patch_size = tuple(info.patch_size)
        self.in_channels = info.in_channels
        self.blocks = nn.ModuleList([CosmosPredict2BlockAdapter(block) for block in transformer.transformer_blocks])
        self._validate_transformer()

    def _validate_transformer(self) -> None:
        if len(self.blocks) != self.info.num_layers:
            raise ValueError(f"Cosmos block count mismatch: {len(self.blocks)} != {self.info.num_layers}")
        if getattr(self.transformer.config, "img_context_dim_in", None):
            raise NotImplementedError("Cosmos image-context cross-attention is not supported by the MoT adapter yet")
        if self.patch_size[0] != 1:
            raise NotImplementedError("Cosmos MoT adapter currently requires temporal patch_size=1")

    def _prepare_condition_mask(self, latents: torch.Tensor) -> torch.Tensor:
        mask = latents.new_zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
        if latents.shape[2] > 0:
            mask[:, :, 0:1] = 1.0
        return mask

    def pre_dit(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        patch_param = self.transformer.patch_embed.proj.weight
        latents = latents.to(device=patch_param.device, dtype=patch_param.dtype)
        context = context.to(device=patch_param.device, dtype=patch_param.dtype)
        if context_mask is not None:
            context_mask = context_mask.to(device=latents.device)

        batch_size, _, num_frames, height, width = latents.shape
        hidden_states = torch.cat([latents, self._prepare_condition_mask(latents)], dim=1)
        if _config_get(self.transformer.config, "concat_padding_mask", False):
            padding_mask = latents.new_zeros(batch_size, 1, height, width)
            hidden_states = torch.cat([hidden_states, padding_mask.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)], dim=1)

        image_rotary_emb = self.transformer.rope(hidden_states, fps=None)
        extra_pos_emb = (
            self.transformer.learnable_pos_embed(hidden_states)
            if _config_get(self.transformer.config, "extra_pos_embed_type", None)
            else None
        )

        p_t, p_h, p_w = _config_get(self.transformer.config, "patch_size")
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        tokens = self.transformer.patch_embed(hidden_states).flatten(1, 3)

        timestep = timestep.reshape(batch_size).to(device=latents.device)
        temb, embedded_timestep = self.transformer.time_embed(tokens, timestep)

        text_context = context
        if _config_get(self.transformer.config, "use_crossattn_projection", False):
            text_context = self.transformer.crossattn_proj(text_context)

        return {
            "tokens": tokens,
            "freqs": image_rotary_emb,
            "t_mod": {
                "embedded_timestep": embedded_timestep,
                "temb": temb,
                "extra_pos_emb": extra_pos_emb,
            },
            "t": {"embedded_timestep": embedded_timestep, "temb": temb},
            "context": text_context,
            "context_mask": context_mask,
            "meta": {
                "T": num_frames,
                "H": height,
                "W": width,
                "f": post_patch_num_frames,
                "h": post_patch_height,
                "w": post_patch_width,
                "tokens_per_frame": post_patch_height * post_patch_width,
            },
        }

    def post_dit(self, tokens: torch.Tensor, meta: dict, t: dict) -> torch.Tensor:
        hidden_states = self.transformer.norm_out(tokens, t["embedded_timestep"], t["temb"])
        hidden_states = self.transformer.proj_out(hidden_states)
        p_t, p_h, p_w = _config_get(self.transformer.config, "patch_size")
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (meta["f"], meta["h"], meta["w"]))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return hidden_states

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = self.pre_dit(latents, timestep, context, context_mask)
        tokens = state["tokens"]
        for block in self.blocks:
            q, k, v = block.get_qkv(tokens, state["t_mod"], state["freqs"])
            bsz, seq_len, _ = q.shape
            q = q.view(bsz, seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_heads, self.attn_head_dim).transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.attn_head_dim)
            tokens = block.post_attention(tokens, attn_output, state["t_mod"], state["context"], state["context_mask"])
        return self.post_dit(tokens, state["meta"], state["t"])


class CosmosPredict2VAE(nn.Module):
    temporal_compress = 4
    spatial_compress = 8

    def __init__(self, model_dir: Path, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self._device = device
        self._dtype = dtype
        self.vae, self.scheduler = _load_cosmos_vae(model_dir, dtype=dtype)
        self.vae.to(device=device, dtype=dtype)
        self.sigma_data = float(getattr(self.scheduler.config, "sigma_data", 1.0))
        self._latents_mean = None
        self._latents_std = None
        if getattr(self.vae.config, "latents_mean", None) is not None:
            self._latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1)
            self._latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1)
        print(f"[CosmosPredict2VAE] Loaded VAE from {model_dir / 'vae'}")

    def _norm_stats(self, tensor: torch.Tensor):
        if self._latents_mean is None or self._latents_std is None:
            return None, None
        return (
            self._latents_mean.to(device=tensor.device, dtype=tensor.dtype),
            self._latents_std.to(device=tensor.device, dtype=tensor.dtype),
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        x = video.to(device=self._device, dtype=self._dtype)
        encoded = self.vae.encode(x)
        latents = encoded.latent_dist.sample()
        latents_mean, latents_std = self._norm_stats(latents)
        if latents_mean is not None:
            latents = (latents - latents_mean) / latents_std * self.sigma_data
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents.to(device=self._device, dtype=self._dtype)
        latents_mean, latents_std = self._norm_stats(z)
        if latents_mean is not None:
            z = z / self.sigma_data * latents_std + latents_mean
        decoded = self.vae.decode(z)
        sample = decoded.sample if hasattr(decoded, "sample") else decoded[0]
        return sample.float().clamp_(-1, 1)


class CosmosPredict2TextEncoder:
    def __init__(
        self,
        model_dir: Path,
        text_len: int = 512,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.text_len = text_len
        self.device = device
        self.dtype = dtype
        self.tokenizer, self.text_encoder = _load_cosmos_t5(model_dir, dtype=dtype)
        self.text_encoder.to(device=device, dtype=dtype)
        self.text_encoder.eval()
        print(f"[CosmosPredict2TextEncoder] Loaded T5 from {model_dir / 'text_encoder'}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.text_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.to(dtype=self.dtype), attention_mask.bool()


class CosmosPredict2Backbone(BaseBackbone):
    """Cosmos-Predict2-2B Video2World backbone for Streaming-WAM MoT."""

    def __init__(
        self,
        config: BackboneConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        load_vae: bool = True,
        load_dit: bool = True,
        load_text_encoder: bool = False,
    ):
        super().__init__()
        self._config = config
        self._dtype = dtype
        self._device = device
        self.model_dir = Path(config.pretrained_model_id)
        self._info = _infer_cosmos_info(self.model_dir)

        self.dit: Optional[CosmosPredict2Dit] = None
        if load_dit:
            transformer = _load_cosmos_transformer(self.model_dir, dtype=dtype)
            self.dit = CosmosPredict2Dit(transformer, self._info)
            print(f"[CosmosPredict2Backbone] Loaded DiT from {self.model_dir / 'transformer'}")

        self.vae: Optional[CosmosPredict2VAE] = None
        if load_vae:
            self.vae = CosmosPredict2VAE(self.model_dir, device=device, dtype=dtype)

        self.text_encoder: Optional[CosmosPredict2TextEncoder] = None
        if load_text_encoder:
            self.text_encoder = CosmosPredict2TextEncoder(
                self.model_dir,
                text_len=int(getattr(config, "tokenizer_max_len", 512)),
                device=device,
                dtype=dtype,
            )

    @property
    def info(self) -> BackboneInfo:
        return self._info

    def has_vae(self) -> bool:
        return self.vae is not None

    def has_text_encoder(self) -> bool:
        return self.text_encoder is not None

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("CosmosPredict2Backbone: VAE not loaded.")
        latents = self.vae.encode(video)
        return latents.to(device=video.device, dtype=self._dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("CosmosPredict2Backbone: VAE not loaded.")
        return self.vae.decode(latents)

    def encode_text(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            batch = len(text)
            length = int(getattr(self._config, "tokenizer_max_len", 512))
            device = torch.device(self._device)
            context = torch.randn(batch, length, self._info.text_dim, device=device, dtype=self._dtype)
            mask = torch.ones(batch, length, device=device, dtype=torch.bool)
            return context, mask
        return self.text_encoder.encode(text)

    def get_dit(self) -> nn.Module:
        if self.dit is None:
            raise RuntimeError("CosmosPredict2Backbone: DiT not loaded.")
        return self.dit

    def get_vae(self) -> Optional[nn.Module]:
        return self.vae

    def build_shared_dit_core(
        self,
        framework_config,
        *,
        state_dim: int,
        action_tokens_per_state: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        from streamingwam.modules.causal_cosmos import CausalCosmosModel

        clean_context = getattr(framework_config, "shared_dit_clean_context", "full_video")
        if clean_context != "full_video":
            raise ValueError("Cosmos Shared-DiT requires framework.shared_dit_clean_context='full_video'")
        transformer = _load_cosmos_transformer(self.model_dir, dtype=dtype)
        core = CausalCosmosModel(
            transformer=transformer,
            info=self.info,
            action_dim=int(framework_config.action_dim),
            state_dim=int(state_dim),
            action_horizon=int(framework_config.chunk_size),
            action_tokens_per_state=int(action_tokens_per_state),
            clean_context=clean_context,
            checkpoint_blocks=getattr(framework_config, "shared_dit_checkpoint_blocks", True),
            num_frame_per_block=int(getattr(framework_config, "num_frame_per_block", 1)),
            num_action_per_block=getattr(framework_config, "num_action_per_block", None),
            num_state_per_block=int(getattr(framework_config, "num_state_per_block", 1)),
        )
        return core.to(device=device, dtype=dtype)
