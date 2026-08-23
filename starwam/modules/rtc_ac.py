"""Checkpoint-specific modules for the FastWAM RTC-AC MoT variant."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from starwam.modules.mot import MoT
from starwam.inference.rtc_ac import (
    RTC_AC_ACTION_HORIZON,
    RTC_AC_CONDITION_SLOTS,
    RTC_AC_DELAY,
)


def _build_video_mask(
    video_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
) -> Tensor:
    first_frame = min(max(int(video_tokens_per_frame), 1), int(video_seq_len))
    mask = torch.ones(
        (int(video_seq_len), int(video_seq_len)),
        dtype=torch.bool,
        device=device,
    )
    mask[:first_frame, first_frame:] = False
    return mask


def build_rtc_ac_policy_mask(
    *,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    known_prefix_length: int,
    device: torch.device,
) -> Tensor:
    """Build the checkpoint's directed D0 or D8 video/policy mask."""

    if int(action_seq_len) != RTC_AC_ACTION_HORIZON:
        raise ValueError(
            f"RTC-AC policy stream requires {RTC_AC_ACTION_HORIZON} actions, "
            f"got {action_seq_len}"
        )
    if int(known_prefix_length) not in (0, RTC_AC_DELAY):
        raise ValueError(f"RTC-AC prefix length must be 0 or {RTC_AC_DELAY}")
    video_seq_len = int(video_seq_len)
    action_seq_len = int(action_seq_len)
    action_start = video_seq_len
    prefix_end = action_start + int(known_prefix_length)
    total = video_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = _build_video_mask(
        video_seq_len,
        video_tokens_per_frame,
        device,
    )
    if known_prefix_length:
        first_frame = int(video_tokens_per_frame)
        mask[action_start:prefix_end, :first_frame] = True
        mask[action_start:prefix_end, action_start:prefix_end] = True
    mask[prefix_end:, :] = True
    return mask


def build_rtc_ac_condition_mask(
    *,
    video_seq_len: int,
    condition_seq_len: int,
    video_tokens_per_frame: int,
    device: torch.device,
) -> Tensor:
    """Build the fixed 16-slot condition graph used by D0 and D8."""

    if int(condition_seq_len) != RTC_AC_CONDITION_SLOTS:
        raise ValueError(
            f"RTC-AC condition stream requires {RTC_AC_CONDITION_SLOTS} slots, "
            f"got {condition_seq_len}"
        )
    video_seq_len = int(video_seq_len)
    condition_seq_len = int(condition_seq_len)
    tokens_per_frame = int(video_tokens_per_frame)
    if video_seq_len != 3 * tokens_per_frame:
        raise ValueError(
            "RTC-AC video tokens must contain exactly z0/z1/z2; got "
            f"video_seq_len={video_seq_len}, tokens_per_frame={tokens_per_frame}"
        )
    condition_start = video_seq_len
    prefix_end = condition_start + RTC_AC_DELAY
    total = video_seq_len + condition_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:video_seq_len, :video_seq_len] = _build_video_mask(
        video_seq_len,
        tokens_per_frame,
        device,
    )
    mask[tokens_per_frame : 2 * tokens_per_frame, condition_start:] = True
    mask[condition_start:prefix_end, :tokens_per_frame] = True
    mask[condition_start:prefix_end, condition_start:prefix_end] = True
    mask[prefix_end:, :tokens_per_frame] = True
    mask[prefix_end:, condition_start:] = True
    return mask


class RTCACSlotEncoder(nn.Module):
    """Represent known clean actions and structure-only unknown slots."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.state_embedding = nn.Embedding(2, self.hidden_dim)
        nn.init.normal_(self.state_embedding.weight, std=self.hidden_dim**-0.5)

    def forward(self, action_tokens: Tensor, known_mask: Tensor) -> Tensor:
        if tuple(action_tokens.shape[1:]) != (
            RTC_AC_CONDITION_SLOTS,
            self.hidden_dim,
        ):
            raise ValueError(
                "RTC-AC action slots must have shape "
                f"[B,{RTC_AC_CONDITION_SLOTS},{self.hidden_dim}], got "
                f"{tuple(action_tokens.shape)}"
            )
        if tuple(known_mask.shape) != tuple(action_tokens.shape[:2]):
            raise ValueError(
                f"RTC-AC known mask must be {tuple(action_tokens.shape[:2])}, "
                f"got {tuple(known_mask.shape)}"
            )
        known_mask = known_mask.to(device=action_tokens.device, dtype=torch.bool)
        content = torch.where(
            known_mask.unsqueeze(-1),
            action_tokens,
            torch.zeros_like(action_tokens),
        )
        state = self.state_embedding(known_mask.to(dtype=torch.long))
        return content + state.to(dtype=action_tokens.dtype)


class RTCACMoT(MoT):
    """Three-stream Stage-2 MoT sharing one ActionDiT across policy/condition."""

    @staticmethod
    def _directed_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block: nn.Module,
        mask: Tensor,
    ) -> Tensor:
        batch, query_len, _ = q.shape
        key_len = k.shape[1]
        num_heads = int(block.num_heads)
        head_dim = int(block.attn_head_dim)
        q = q.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, key_len, num_heads, head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return output.transpose(1, 2).reshape(batch, query_len, num_heads * head_dim)

    def forward_rtc_ac(
        self,
        expert_states: dict[str, dict[str, Tensor]],
        *,
        policy_attention_mask: Tensor,
        condition_attention_mask: Tensor,
        video_tokens_per_frame: int,
        action_condition_active: Tensor,
    ) -> dict[str, Tensor]:
        if self.expert_order != ["video", "action"]:
            raise ValueError("RTC-AC requires video/action expert order")
        if set(expert_states) != {"video", "action", "condition"}:
            raise ValueError("RTC-AC requires video, action, and condition streams")

        video_expert = self.experts["video"]
        action_expert = self.experts["action"]
        tokens = {
            name: expert_states[name]["tokens"]
            for name in ("video", "action", "condition")
        }
        video_len = int(tokens["video"].shape[1])
        action_len = int(tokens["action"].shape[1])
        condition_len = int(tokens["condition"].shape[1])
        tokens_per_frame = int(video_tokens_per_frame)
        if video_len != 3 * tokens_per_frame:
            raise ValueError("RTC-AC requires z0/z1/z2 video tokens")
        if action_len != RTC_AC_ACTION_HORIZON:
            raise ValueError(f"RTC-AC requires {RTC_AC_ACTION_HORIZON} policy actions")
        if condition_len != RTC_AC_CONDITION_SLOTS:
            raise ValueError(f"RTC-AC requires {RTC_AC_CONDITION_SLOTS} condition slots")
        if tuple(action_condition_active.shape) != (tokens["video"].shape[0],):
            raise ValueError("RTC-AC action_condition_active must be [B]")

        active = action_condition_active.to(
            device=tokens["video"].device,
            dtype=tokens["video"].dtype,
        ).view(-1, 1, 1)
        z1 = slice(tokens_per_frame, 2 * tokens_per_frame)

        for layer_index in range(self.num_layers):
            video_block = video_expert.blocks[layer_index]
            action_block = action_expert.blocks[layer_index]
            projected: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
            for name, block in (
                ("video", video_block),
                ("action", action_block),
                ("condition", action_block),
            ):
                state = expert_states[name]
                projected[name] = block.get_qkv(
                    tokens[name],
                    state["t_mod"],
                    state["freqs"],
                )

            q_video, k_video, v_video = projected["video"]
            q_action, k_action, v_action = projected["action"]
            q_condition, k_condition, v_condition = projected["condition"]
            video_output = self._directed_attention(
                q_video,
                k_video,
                v_video,
                video_block,
                policy_attention_mask[:video_len, :video_len],
            )
            action_output = self._directed_attention(
                q_action,
                torch.cat((k_video, k_action), dim=1),
                torch.cat((v_video, v_action), dim=1),
                action_block,
                policy_attention_mask[video_len:, :],
            )
            condition_output = self._directed_attention(
                q_condition,
                torch.cat((k_video, k_condition), dim=1),
                torch.cat((v_video, v_condition), dim=1),
                action_block,
                condition_attention_mask[video_len:, :],
            )
            z1_condition = self._directed_attention(
                q_video[:, z1],
                k_condition,
                v_condition,
                video_block,
                condition_attention_mask[z1, video_len:],
            )
            video_output = torch.cat(
                (
                    video_output[:, :tokens_per_frame],
                    video_output[:, z1] + active * z1_condition,
                    video_output[:, 2 * tokens_per_frame :],
                ),
                dim=1,
            )

            for name, block, attention_output in (
                ("video", video_block, video_output),
                ("action", action_block, action_output),
                ("condition", action_block, condition_output),
            ):
                state = expert_states[name]
                tokens[name] = block.post_attention(
                    tokens[name],
                    attention_output,
                    state["t_mod"],
                    state["context"],
                    state["context_mask"],
                )

        return {"video": tokens["video"], "action": tokens["action"]}

    @staticmethod
    def _cross_attention_kv(
        block: nn.Module,
        context: Tensor,
        static_kv: tuple[Tensor, Tensor],
        static_context_length: int,
    ) -> tuple[Tensor, Tensor]:
        static_key, static_value = static_kv
        if int(context.shape[1]) == int(static_context_length):
            return static_key, static_value
        dynamic_context = context[:, int(static_context_length) :]
        dynamic_key = block.cross_attn.norm_k(
            block.cross_attn.k(dynamic_context)
        )
        dynamic_value = block.cross_attn.v(dynamic_context)
        return (
            torch.cat((static_key, dynamic_key), dim=1),
            torch.cat((static_value, dynamic_value), dim=1),
        )

    def forward_rtc_ac_accelerated(
        self,
        *,
        tokens_all: dict[str, Tensor],
        freqs_all: dict[str, Tensor],
        context_all: dict[str, dict[str, object]],
        t_mod_all: dict[str, Tensor],
        policy_attention_mask: Tensor,
        condition_attention_mask: Tensor,
        video_tokens_per_frame: int,
        action_condition_active: Tensor,
    ) -> dict[str, Tensor]:
        """Reference-shaped fixed-geometry RTC core used by TorchInductor."""

        video_expert = self.experts["video"]
        action_expert = self.experts["action"]
        tokens = tokens_all
        video_len = int(tokens["video"].shape[1])
        tokens_per_frame = int(video_tokens_per_frame)
        active = action_condition_active.to(
            device=tokens["video"].device,
            dtype=tokens["video"].dtype,
        ).view(-1, 1, 1)
        z1 = slice(tokens_per_frame, 2 * tokens_per_frame)

        for layer_index in range(self.num_layers):
            video_block = video_expert.blocks[layer_index]
            action_block = action_expert.blocks[layer_index]
            projected = {}
            block_states = {}
            for name, block in (
                ("video", video_block),
                ("action", action_block),
                ("condition", action_block),
            ):
                q, k, v, block_state = block.get_qkv_functional(
                    tokens[name],
                    t_mod_all[name],
                    freqs_all[name],
                )
                projected[name] = (q, k, v)
                block_states[name] = block_state

            q_video, k_video, v_video = projected["video"]
            q_action, k_action, v_action = projected["action"]
            q_condition, k_condition, v_condition = projected["condition"]
            video_output = self._directed_attention(
                q_video,
                k_video,
                v_video,
                video_block,
                policy_attention_mask[:video_len, :video_len],
            )
            action_output = self._directed_attention(
                q_action,
                torch.cat((k_video, k_action), dim=1),
                torch.cat((v_video, v_action), dim=1),
                action_block,
                policy_attention_mask[video_len:, :],
            )
            condition_output = self._directed_attention(
                q_condition,
                torch.cat((k_video, k_condition), dim=1),
                torch.cat((v_video, v_condition), dim=1),
                action_block,
                condition_attention_mask[video_len:, :],
            )
            z1_condition = self._directed_attention(
                q_video[:, z1],
                k_condition,
                v_condition,
                video_block,
                condition_attention_mask[z1, video_len:],
            )
            video_output = torch.cat(
                (
                    video_output[:, :tokens_per_frame],
                    video_output[:, z1] + active * z1_condition,
                    video_output[:, 2 * tokens_per_frame :],
                ),
                dim=1,
            )

            video_cross_kv = self._cross_attention_kv(
                video_block,
                context_all["video"]["context"],
                context_all["video"]["static_cross_attention_kv"][layer_index],
                int(context_all["video"]["static_context_length"]),
            )
            action_cross_kv = self._cross_attention_kv(
                action_block,
                context_all["action"]["context"],
                context_all["action"]["static_cross_attention_kv"][layer_index],
                int(context_all["action"]["static_context_length"]),
            )
            tokens["video"] = video_block.post_attention_with_kv(
                block_states["video"],
                video_output,
                video_cross_kv[0],
                video_cross_kv[1],
                context_all["video"]["mask"],
            )
            tokens["action"] = action_block.post_attention_with_kv(
                block_states["action"],
                action_output,
                action_cross_kv[0],
                action_cross_kv[1],
                context_all["action"]["mask"],
            )
            tokens["condition"] = action_block.post_attention_with_kv(
                block_states["condition"],
                condition_output,
                action_cross_kv[0],
                action_cross_kv[1],
                context_all["condition"]["mask"],
            )

        return {"video": tokens["video"], "action": tokens["action"]}
