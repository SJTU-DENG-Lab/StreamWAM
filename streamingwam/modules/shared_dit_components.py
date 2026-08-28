"""Reusable Shared-DiT action/state token components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from streamingwam.modules.wan_block import sinusoidal_embedding_1d


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_categories, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(num_categories, output_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor, category_ids: Tensor) -> Tensor:
        weight = self.weight[category_ids]
        bias = self.bias[category_ids]
        return torch.bmm(x, weight) + bias.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.fc1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.fc2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: Tensor, category_ids: Tensor) -> Tensor:
        return self.fc2(F.silu(self.fc1(x, category_ids)), category_ids)


class ActionRegisterEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, freq_dim: int, num_categories: int = 1) -> None:
        super().__init__()
        self.action_in = CategorySpecificLinear(num_categories, action_dim, hidden_dim)
        self.fuse = CategorySpecificLinear(num_categories, hidden_dim + freq_dim, hidden_dim)
        self.out = CategorySpecificLinear(num_categories, hidden_dim, hidden_dim)
        self.freq_dim = freq_dim

    def forward(self, action: Tensor, timesteps: Tensor, category_ids: Tensor) -> Tensor:
        action_features = self.action_in(action, category_ids)
        if timesteps.dim() == 1:
            timesteps = timesteps.view(action.shape[0], 1).expand(action.shape[0], action.shape[1])
        elif timesteps.shape[1] != action.shape[1]:
            timesteps = timesteps[:, :1].expand(action.shape[0], action.shape[1])
        timestep_features = sinusoidal_embedding_1d(self.freq_dim, timesteps.reshape(-1)).reshape(
            action.shape[0], action.shape[1], self.freq_dim
        ).to(device=action_features.device, dtype=action_features.dtype)
        x = torch.cat([action_features, timestep_features], dim=-1)
        x = F.silu(self.fuse(x, category_ids))
        return self.out(x, category_ids)


def build_shared_dit_attention_mask(
    *,
    clean_seq_len: int,
    noisy_seq_len: int,
    action_seq_len: int,
    state_seq_len: int,
    tokens_per_frame: int,
    num_frame_per_block: int,
    num_action_per_block: int,
    num_state_per_block: int,
    device: torch.device,
) -> Tensor:
    """DreamZero-style blockwise causal mask for [clean][noisy][action][state]."""
    if clean_seq_len != noisy_seq_len:
        raise ValueError(
            "Shared-DiT DreamZero mask requires clean and noisy video to have the same token length; "
            f"got clean_seq_len={clean_seq_len}, noisy_seq_len={noisy_seq_len}"
        )
    if tokens_per_frame <= 0 or clean_seq_len % tokens_per_frame != 0:
        raise ValueError(
            f"Invalid video token geometry: clean_seq_len={clean_seq_len}, tokens_per_frame={tokens_per_frame}"
        )

    num_frame_per_block = max(1, int(num_frame_per_block))
    num_action_per_block = int(num_action_per_block)
    num_state_per_block = max(1, int(num_state_per_block))
    num_frames = clean_seq_len // tokens_per_frame
    if (num_frames - 1) % num_frame_per_block != 0:
        raise ValueError(
            "DreamZero Shared-DiT requires future video frames to form complete blocks: "
            f"num_frames={num_frames}, num_frame_per_block={num_frame_per_block}"
        )
    num_blocks = (num_frames - 1) // num_frame_per_block
    if num_blocks <= 0:
        raise ValueError(
            "DreamZero Shared-DiT needs at least one future video block: "
            f"num_frames={num_frames}, num_frame_per_block={num_frame_per_block}"
        )
    if action_seq_len != num_blocks * num_action_per_block:
        raise ValueError(
            "Action tokens must align with DreamZero video blocks: "
            f"action_seq_len={action_seq_len}, expected={num_blocks * num_action_per_block} "
            f"(num_blocks={num_blocks}, num_action_per_block={num_action_per_block})"
        )
    if state_seq_len != num_blocks * num_state_per_block:
        raise ValueError(
            "State tokens must align with DreamZero video blocks: "
            f"state_seq_len={state_seq_len}, expected={num_blocks * num_state_per_block} "
            f"(num_blocks={num_blocks}, num_state_per_block={num_state_per_block})"
        )

    total = clean_seq_len + noisy_seq_len + action_seq_len + state_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)

    clean_start = 0
    clean_end = clean_seq_len
    noisy_start = clean_end
    noisy_end = noisy_start + noisy_seq_len
    action_start = noisy_end
    state_start = action_start + action_seq_len

    first_clean_start = clean_start
    first_clean_end = clean_start + tokens_per_frame
    block_video_tokens = num_frame_per_block * tokens_per_frame

    mask[first_clean_start:first_clean_end, first_clean_start:first_clean_end] = True
    for block_idx in range(num_blocks):
        clean_block_start = first_clean_end + block_idx * block_video_tokens
        clean_block_end = clean_block_start + block_video_tokens
        noisy_block_start = noisy_start + tokens_per_frame + block_idx * block_video_tokens
        noisy_block_end = noisy_block_start + block_video_tokens
        action_block_start = action_start + block_idx * num_action_per_block
        action_block_end = action_block_start + num_action_per_block
        state_block_start = state_start + block_idx * num_state_per_block
        state_block_end = state_block_start + num_state_per_block

        clean_context_end = clean_block_start
        clean_query = slice(clean_block_start, clean_block_end)
        clean_context = slice(clean_start, clean_block_end)
        mask[clean_query, clean_context] = True

        noisy_query = slice(noisy_block_start, noisy_block_end)
        mask[noisy_query, clean_start:clean_context_end] = True
        mask[noisy_query, noisy_block_start:noisy_block_end] = True
        mask[noisy_query, action_block_start:action_block_end] = True
        mask[noisy_query, state_block_start:state_block_end] = True

        action_query = slice(action_block_start, action_block_end)
        mask[action_query, clean_start:clean_context_end] = True
        mask[action_query, noisy_block_start:noisy_block_end] = True
        mask[action_query, action_block_start:action_block_end] = True
        mask[action_query, state_block_start:state_block_end] = True

        state_query = slice(state_block_start, state_block_end)
        mask[state_query, state_block_start:state_block_end] = True

    noisy_first = slice(noisy_start, noisy_start + tokens_per_frame)
    mask[noisy_first, noisy_first] = True
    return mask
