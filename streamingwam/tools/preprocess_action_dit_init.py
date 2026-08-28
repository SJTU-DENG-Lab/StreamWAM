"""Preprocess Streaming-WAM ActionDiT initialization from video DiT weights.

The output payload is consumed by ``framework.action_expert_init_from``. Wan2.2
uses direct Wan-compatible key copying/interpolation. Cosmos-Predict2 uses a
best-effort structural mapping into the generic ActionDiT: attention, text, time,
and FFN weights are copied where shapes are compatible, while action-specific and
non-isomorphic AdaLN/gating parameters keep ActionDiT defaults. Use ``--head-init
zero`` or ``--head-init payload`` to make output-head initialization explicit in
the payload and pair it with ``framework.action_expert_head_init``.

Examples:
    python -m streamingwam.tools.preprocess_action_dit_init \
        --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
        --pretrained-model-id /path/to/Wan2.2-TI2V-5B \
        --output /path/to/preprocessed/streamingwam_action_dit_init.pt \
        --device cuda --dtype bfloat16

    python -m streamingwam.tools.preprocess_action_dit_init \
        --config examples/libero/configs/recipes/streamingwam_libero_mot_wan22_5b.yaml \
        --video-state-dict /path/to/wan22_dit_state_dict.pt \
        --output /path/to/preprocessed/streamingwam_action_dit_init.pt

    python -m streamingwam.tools.preprocess_action_dit_init \
        --config examples/libero/configs/recipes/streamingwam_libero_mot_cosmos_predict2.yaml \
        --source-backbone cosmos_predict2 \
        --pretrained-model-id /path/to/Cosmos-Predict2-2B-Video2World \
        --output /path/to/preprocessed/streamingwam_action_dit_init_cosmos_predict2.pt \
        --device cpu --dtype bfloat16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)


def _resize_to_shape(src: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == target_shape:
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(f"Cannot reduce rank: src={tuple(src.shape)}, target={target_shape}")
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        if out.shape[dim] == new_size:
            continue
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    if tuple(out.shape) != target_shape:
        raise ValueError(
            f"Resize produced wrong shape: src={tuple(src.shape)}, target={target_shape}, got={tuple(out.shape)}"
        )
    return out.to(dtype=src.dtype)


def _convert_tensor(
    src: torch.Tensor,
    target: torch.Tensor,
    apply_alpha_scaling: bool,
) -> tuple[torch.Tensor, bool]:
    target_shape = tuple(target.shape)
    if tuple(src.shape) == target_shape:
        value = src
        resized = False
    else:
        value = _resize_to_shape(src, target_shape)
        if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
            alpha = (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
            value = value.to(torch.float32) * alpha
        resized = True
    return value.detach().to(dtype=target.dtype, device="cpu").contiguous(), resized


def _load_video_state_from_file(path: str) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if isinstance(raw, dict) and "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
        return raw["model_state_dict"]
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unsupported source format: {type(raw)}")


def _build_action_dit_from_config(
    config_path: str,
    device: str,
    dtype: torch.dtype,
    pretrained_model_id: str | None = None,
):
    from streamingwam.action_model import build_action_dit
    from streamingwam.config import load_config
    from streamingwam.checkpointing import infer_backbone_info

    cfg = load_config(config_path)
    if pretrained_model_id:
        cfg.backbone.pretrained_model_id = pretrained_model_id
    backbone_info = infer_backbone_info(cfg.backbone.pretrained_model_id)
    action_dit = build_action_dit(backbone_info, cfg.framework).to(device=device, dtype=dtype)
    return cfg, backbone_info, action_dit


def _load_wan22_video_state(
    config_path: str,
    dtype: torch.dtype,
    pretrained_model_id: str | None = None,
) -> dict[str, torch.Tensor]:
    from streamingwam.config import load_config
    from streamingwam.backbone.wan22 import Wan22Dit
    from streamingwam.checkpointing import infer_backbone_info

    cfg = load_config(config_path)
    if pretrained_model_id:
        cfg.backbone.pretrained_model_id = pretrained_model_id
    info = infer_backbone_info(cfg.backbone.pretrained_model_id)
    dit = Wan22Dit(info)
    dit.load_pretrained(cfg.backbone.pretrained_model_id, dtype=dtype)
    return dit.state_dict()


def _load_cosmos_video_state(
    config_path: str,
    dtype: torch.dtype,
    pretrained_model_id: str | None = None,
) -> dict[str, torch.Tensor]:
    from streamingwam.config import load_config
    from safetensors.torch import load_file

    cfg = load_config(config_path)
    if pretrained_model_id:
        cfg.backbone.pretrained_model_id = pretrained_model_id
    model_dir = Path(cfg.backbone.pretrained_model_id)
    single_file = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    if single_file.exists():
        state = load_file(str(single_file), device="cpu")
        return {key: value.to(dtype=dtype) for key, value in state.items()}

    index_file = model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Cosmos transformer safetensors not found under {model_dir / 'transformer'}")

    import json

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})
    state: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(str(index_file.parent / shard_name), device="cpu")
        for key, value in shard.items():
            state[key] = value.to(dtype=dtype)
    return state


def _put_mapped(
    out: dict[str, torch.Tensor],
    target_key: str,
    target_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    source_key: str,
    apply_alpha_scaling: bool,
    stats: dict[str, int],
) -> None:
    if target_key not in target_state:
        stats["target_missing"] += 1
        return
    if source_key not in source_state:
        if target_key.endswith(".bias"):
            out[target_key] = torch.zeros_like(target_state[target_key], device="cpu")
            stats["zero"] += 1
        else:
            stats["source_missing"] += 1
        return
    value, resized = _convert_tensor(source_state[source_key], target_state[target_key], apply_alpha_scaling)
    out[target_key] = value
    stats["interpolated" if resized else "copied"] += 1


def _init_linear_identity_like(weight: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(weight, device="cpu")
    diag = min(out.shape[0], out.shape[1])
    out[:diag, :diag] = torch.eye(diag, dtype=out.dtype)
    if out.shape[0] > out.shape[1]:
        repeats = (out.shape[0] + out.shape[1] - 1) // out.shape[1]
        tiled = torch.eye(out.shape[1], dtype=out.dtype).repeat(repeats, 1)[: out.shape[0]]
        out.copy_(tiled / repeats ** 0.5)
    return out


def _put_identity_linear(
    out: dict[str, torch.Tensor],
    prefix: str,
    target_state: dict[str, torch.Tensor],
    stats: dict[str, int],
) -> None:
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"
    if weight_key in target_state:
        out[weight_key] = _init_linear_identity_like(target_state[weight_key])
        stats["identity"] += 1
    if bias_key in target_state:
        out[bias_key] = torch.zeros_like(target_state[bias_key], device="cpu")
        stats["zero"] += 1


def _map_cosmos_to_action_state(
    video_state: dict[str, torch.Tensor],
    action_state: dict[str, torch.Tensor],
    apply_alpha_scaling: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    stats = {"copied": 0, "interpolated": 0, "source_missing": 0, "target_missing": 0, "identity": 0, "zero": 0}
    mapped: dict[str, torch.Tensor] = {}

    _put_identity_linear(mapped, "text_embedding.0", action_state, stats)
    _put_identity_linear(mapped, "text_embedding.2", action_state, stats)
    _put_identity_linear(mapped, "time_embedding.2", action_state, stats)
    _put_mapped(
        mapped,
        "time_embedding.0.weight",
        action_state,
        video_state,
        "time_embed.t_embedder.linear_1.weight",
        apply_alpha_scaling,
        stats,
    )
    # Keep ActionDiT's Wan-style AdaLN/gating projection at its default init.
    # Cosmos time_embed.t_embedder.linear_2 is not isomorphic to ActionDiT's
    # hidden -> 6*hidden modulation projection and can produce very large action
    # outputs if copied/interpolated into time_projection.1.

    block_map = {
        "self_attn.q.weight": "attn1.to_q.weight",
        "self_attn.q.bias": "attn1.to_q.bias",
        "self_attn.k.weight": "attn1.to_k.weight",
        "self_attn.k.bias": "attn1.to_k.bias",
        "self_attn.v.weight": "attn1.to_v.weight",
        "self_attn.v.bias": "attn1.to_v.bias",
        "self_attn.o.weight": "attn1.to_out.0.weight",
        "self_attn.o.bias": "attn1.to_out.0.bias",
        "self_attn.norm_q.weight": "attn1.norm_q.weight",
        "self_attn.norm_k.weight": "attn1.norm_k.weight",
        "cross_attn.q.weight": "attn2.to_q.weight",
        "cross_attn.q.bias": "attn2.to_q.bias",
        "cross_attn.k.weight": "attn2.to_k.weight",
        "cross_attn.k.bias": "attn2.to_k.bias",
        "cross_attn.v.weight": "attn2.to_v.weight",
        "cross_attn.v.bias": "attn2.to_v.bias",
        "cross_attn.o.weight": "attn2.to_out.0.weight",
        "cross_attn.o.bias": "attn2.to_out.0.bias",
        "cross_attn.norm_q.weight": "attn2.norm_q.weight",
        "cross_attn.norm_k.weight": "attn2.norm_k.weight",
        "ffn.0.weight": "ff.net.0.proj.weight",
        "ffn.0.bias": "ff.net.0.proj.bias",
        "ffn.2.weight": "ff.net.2.weight",
        "ffn.2.bias": "ff.net.2.bias",
    }
    num_blocks = max(
        [int(key.split(".")[1]) for key in action_state if key.startswith("blocks.") and key.split(".")[1].isdigit()],
        default=-1,
    ) + 1
    for i in range(num_blocks):
        for target_suffix, source_suffix in block_map.items():
            _put_mapped(
                mapped,
                f"blocks.{i}.{target_suffix}",
                action_state,
                video_state,
                f"transformer_blocks.{i}.{source_suffix}",
                apply_alpha_scaling,
                stats,
            )
    return mapped, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Streaming-WAM YAML recipe used to infer ActionDiT shape.")
    parser.add_argument("--output", required=True, help="Output .pt payload path.")
    parser.add_argument(
        "--video-state-dict",
        default=None,
        help="Optional torch state_dict for the source video DiT. If omitted, load from recipe backbone.",
    )
    parser.add_argument(
        "--pretrained-model-id",
        default=None,
        help="Local source backbone checkpoint directory. Overrides backbone.pretrained_model_id from the recipe.",
    )
    parser.add_argument(
        "--source-backbone",
        default="auto",
        choices=["auto", "wan22", "cosmos_predict2"],
        help="Source video DiT family used to build the ActionDiT init payload.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--head-init", default="random", choices=["random", "zero", "payload"])
    parser.add_argument("--no-alpha-scaling", dest="apply_alpha_scaling", action="store_false")
    parser.set_defaults(apply_alpha_scaling=True)
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    cfg, backbone_info, action_dit = _build_action_dit_from_config(
        args.config,
        args.device,
        dtype,
        pretrained_model_id=args.pretrained_model_id,
    )
    action_state = action_dit.state_dict()

    source_backbone = args.source_backbone
    if source_backbone == "auto":
        backbone_type = str(getattr(cfg.backbone, "type", "")).lower()
        source_backbone = "cosmos_predict2" if backbone_type == "cosmos_predict2" else "wan22"

    if args.video_state_dict:
        print(f"[INFO] Loading source video DiT state_dict from {args.video_state_dict}")
        video_state = _load_video_state_from_file(args.video_state_dict)
    elif source_backbone == "cosmos_predict2":
        print(f"[INFO] Loading source Cosmos-Predict2 DiT from {cfg.backbone.pretrained_model_id}")
        video_state = _load_cosmos_video_state(args.config, dtype, pretrained_model_id=args.pretrained_model_id)
    else:
        print(f"[INFO] Loading source Wan2.2 DiT from {cfg.backbone.pretrained_model_id}")
        video_state = _load_wan22_video_state(args.config, dtype, pretrained_model_id=args.pretrained_model_id)

    source_stats: dict[str, int] = {}
    if source_backbone == "cosmos_predict2":
        backbone_state, source_stats = _map_cosmos_to_action_state(video_state, action_state, args.apply_alpha_scaling)
        copied = source_stats["copied"]
        interpolated = source_stats["interpolated"]
        if not backbone_state:
            raise ValueError("Cosmos-Predict2 source mapping produced no ActionDiT weights")
    else:
        backbone_keys = action_dit.backbone_key_set(action_state.keys())
        backbone_state = {}
        copied = 0
        interpolated = 0
        missing: list[str] = []
        for key in sorted(backbone_keys):
            if key not in video_state:
                missing.append(key)
                continue
            value, resized = _convert_tensor(video_state[key], action_state[key], args.apply_alpha_scaling)
            backbone_state[key] = value
            if resized:
                interpolated += 1
            else:
                copied += 1

        if missing:
            raise ValueError(f"{len(missing)} keys missing in source video DiT state_dict: {missing[:10]}")

    payload: dict[str, Any] = {
        "policy": {
            "alpha_scaling": bool(args.apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "action_backbone_skip_prefixes": list(action_dit.ACTION_BACKBONE_SKIP_PREFIXES),
            "head_init": args.head_init,
            "source": "video_state_dict" if args.video_state_dict else "streamingwam_recipe_backbone",
            "source_backbone": source_backbone,
        },
        "backbone_state_dict": backbone_state,
        "meta": {
            "hidden_dim": int(cfg.framework.action_expert_hidden_dim),
            "ffn_dim": int(cfg.framework.action_expert_hidden_dim) * 4,
            "num_layers": int(cfg.framework.action_expert_num_layers or backbone_info.num_layers),
            "num_heads": int(backbone_info.num_heads),
            "attn_head_dim": int(backbone_info.attn_head_dim),
            "text_dim": int(backbone_info.text_dim),
            "freq_dim": int(backbone_info.freq_dim),
            "eps": float(backbone_info.eps),
            "action_dim": int(cfg.framework.action_dim),
            "source_mapping_stats": source_stats,
        },
    }

    if args.head_init == "zero":
        payload["head_state_dict"] = {
            "head.weight": torch.zeros_like(action_dit.head.weight, device="cpu"),
            "head.bias": torch.zeros_like(action_dit.head.bias, device="cpu"),
        }
    elif args.head_init == "payload":
        payload["head_state_dict"] = {
            "head.weight": action_dit.head.weight.detach().to(device="cpu").contiguous(),
            "head.bias": action_dit.head.bias.detach().to(device="cpu").contiguous(),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    extra = ""
    if source_stats:
        extra = f", identity={source_stats.get('identity', 0)}, zero={source_stats.get('zero', 0)}, source_missing={source_stats.get('source_missing', 0)}"
    print(
        f"[INFO] Saved Streaming-WAM ActionDiT init to {out_path} "
        f"(source_backbone={source_backbone}, copied={copied}, interpolated={interpolated}{extra}, "
        f"total={len(backbone_state)}, head_init={args.head_init})."
    )
    print(f"[INFO] Set framework.action_expert_init_from: {out_path}")
    print(f"[INFO] Set framework.action_expert_head_init: {args.head_init}")


if __name__ == "__main__":
    main()
