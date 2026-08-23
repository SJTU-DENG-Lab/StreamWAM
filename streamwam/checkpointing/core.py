"""Generic checkpoint helpers shared by training and model construction."""

import json
import os
from pathlib import Path
from typing import Optional

import torch

from streamwam.backbone.base import BackboneInfo


def infer_backbone_info(model_path: str) -> BackboneInfo:
    """Auto-infer backbone architecture parameters from a checkpoint config."""

    model_path = Path(model_path)
    candidate_configs = [
        model_path / "config.json",
        model_path / "transformer" / "config.json",
    ]
    for cfg_path in candidate_configs:
        if not cfg_path.exists():
            continue
        with open(cfg_path) as handle:
            cfg = json.load(handle)
        if not any(
            key in cfg
            for key in (
                "dim", "hidden_size", "hidden_dim", "num_heads",
                "num_attention_heads", "num_layers", "num_hidden_layers",
            )
        ):
            continue
        num_heads = int(cfg.get("num_heads", cfg.get("num_attention_heads", 24)))
        attn_head_dim = int(cfg.get("attention_head_dim", cfg.get("head_dim", 0)))
        hidden_dim = int(
            cfg.get(
                "dim",
                cfg.get(
                    "hidden_size",
                    cfg.get("hidden_dim", num_heads * attn_head_dim if attn_head_dim else 3072),
                ),
            )
        )
        if attn_head_dim <= 0:
            attn_head_dim = hidden_dim // num_heads
        mlp_ratio = float(cfg.get("mlp_ratio", 0.0) or 0.0)
        ffn_dim = int(
            cfg.get(
                "ffn_dim",
                cfg.get("intermediate_size", hidden_dim * mlp_ratio if mlp_ratio else 14336),
            )
        )
        return BackboneInfo(
            hidden_dim=hidden_dim,
            num_layers=int(cfg.get("num_layers", cfg.get("num_hidden_layers", 30))),
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            ffn_dim=ffn_dim,
            text_dim=int(cfg.get("text_dim", cfg.get("text_embed_dim", 4096))),
            freq_dim=int(
                cfg.get(
                    "freq_dim",
                    hidden_dim if cfg.get("_class_name") == "CosmosTransformer3DModel" else 256,
                )
            ),
            eps=float(cfg.get("eps", cfg.get("layer_norm_eps", 1e-6))),
            patch_size=tuple(cfg.get("patch_size", [1, 2, 2])),
            in_channels=int(
                cfg.get("in_dim", cfg.get("in_channels", cfg.get("out_channels", 16)))
            ),
        )

    return BackboneInfo(
        hidden_dim=3072,
        num_layers=30,
        num_heads=24,
        attn_head_dim=128,
        ffn_dim=14336,
        text_dim=4096,
        freq_dim=256,
        eps=1e-6,
        patch_size=(1, 2, 2),
        in_channels=48,
    )


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    step: int,
    extra: Optional[dict] = None,
):
    """Save a training checkpoint."""

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"checkpoint-{step}.pt")
    state = {"model_state_dict": model.state_dict(), "step": step}
    if extra:
        state.update(extra)
    torch.save(state, save_path)


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = False) -> dict:
    """Load a training checkpoint and return its metadata."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    return {key: value for key, value in state.items() if key != "model_state_dict"}


def load_action_dit_backbone_init(
    action_expert: torch.nn.Module,
    path: str,
    head_init: str = "random",
) -> dict:
    """Load a preprocessed video-DiT initialization into ActionDiT."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "backbone_state_dict" not in payload:
        raise ValueError(
            f"Payload at {path} missing `backbone_state_dict`; "
            "did you generate it via `python -m streamwam.tools.preprocess_action_dit_init`?"
        )

    head_init = str(head_init).strip().lower()
    if head_init not in {"random", "zero", "payload"}:
        raise ValueError(f"Unsupported action head init policy: {head_init!r}")

    state = dict(payload["backbone_state_dict"])
    if head_init == "zero":
        state["head.weight"] = torch.zeros_like(action_expert.head.weight, device="cpu")
        state["head.bias"] = torch.zeros_like(action_expert.head.bias, device="cpu")
    elif head_init == "payload":
        head_state = payload.get("head_state_dict")
        if not isinstance(head_state, dict):
            raise ValueError(
                f"Payload at {path} missing `head_state_dict`; "
                "regenerate it with a head init policy that writes head weights."
            )
        state.update(head_state)

    result = action_expert.load_state_dict(state, strict=False)
    return {
        "meta": payload.get("meta", {}),
        "policy": payload.get("policy", {}),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "num_loaded": len(state),
        "head_init": head_init,
    }
