from __future__ import annotations

import platform
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import triton

from streamwam.inference import rtc_ac as rtc_ac_inference
from streamwam.backbone.base import BackboneInfo
from streamwam.backbone.wan22 import Wan22Dit
from streamwam.inference.rtc_ac import RTCACAccelerationRuntime
from streamwam.modules.action_dit import ActionDiT
from streamwam.modules.rtc_ac import (
    RTCACMoT,
    build_rtc_ac_condition_mask,
    build_rtc_ac_policy_mask,
)
from streamwam.modules.wan_block import DiTBlock, precompute_freqs_cis_1d
from streamwam.wam import rtc_ac_wam


def _self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, sequence, _ = q.shape
    q = q.view(batch, sequence, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, sequence, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, sequence, num_heads, head_dim).transpose(1, 2)
    output = F.scaled_dot_product_attention(q, k, v)
    return output.transpose(1, 2).reshape(batch, sequence, num_heads * head_dim)


def test_functional_wan_block_matches_eager_post_attention() -> None:
    torch.manual_seed(17)
    block = DiTBlock(
        hidden_dim=8,
        attn_head_dim=4,
        num_heads=2,
        ffn_dim=16,
    ).eval()
    tokens = torch.randn(1, 3, 8)
    timestep_modulation = torch.randn(1, 3, 6, 8)
    frequencies = precompute_freqs_cis_1d(4, end=3).unsqueeze(1)
    context = torch.randn(1, 4, 8)
    context_mask = torch.ones(1, 3, 4, dtype=torch.bool)

    eager_q, eager_k, eager_v = block.get_qkv(
        tokens,
        timestep_modulation,
        frequencies,
    )
    eager_attention = _self_attention(
        eager_q,
        eager_k,
        eager_v,
        num_heads=2,
        head_dim=4,
    )
    eager = block.post_attention(
        tokens,
        eager_attention,
        timestep_modulation,
        context,
        context_mask,
    )

    q, k, v, state = block.get_qkv_functional(
        tokens,
        timestep_modulation,
        frequencies,
    )
    attention = _self_attention(q, k, v, num_heads=2, head_dim=4)
    cross_key = block.cross_attn.norm_k(block.cross_attn.k(context))
    cross_value = block.cross_attn.v(context)
    actual = block.post_attention_with_kv(
        state,
        attention,
        cross_key,
        cross_value,
        context_mask,
    )

    torch.testing.assert_close(actual, eager, rtol=1e-5, atol=1e-6)


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=8,
                    attn_head_dim=4,
                    num_heads=2,
                    ffn_dim=16,
                )
            ]
        )
        self.text_embedding = nn.Linear(8, 8)


def _wan22_5b_contract_expert(dtype: torch.dtype = torch.bfloat16) -> Wan22Dit:
    expert = Wan22Dit.__new__(Wan22Dit)
    nn.Module.__init__(expert)
    expert.hidden_dim = 3072
    expert.num_layers = 30
    expert.num_heads = 24
    expert.register_parameter("contract_parameter", nn.Parameter(torch.zeros((), dtype=dtype)))
    return expert


def test_accelerated_contract_accepts_wan22_5b_bf16_batch_one() -> None:
    rtc_ac_wam.validate_rtc_ac_accelerated_contract(
        input_image=torch.zeros((1, 3, 224, 448), dtype=torch.bfloat16),
        video_expert=_wan22_5b_contract_expert(),
        action_expert=_TinyExpert().to(dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    ("input_image", "video_expert", "action_expert", "message"),
    [
        (
            torch.zeros((1, 3, 224, 448), dtype=torch.float32),
            _wan22_5b_contract_expert(),
            _TinyExpert().to(dtype=torch.bfloat16),
            "BF16",
        ),
        (
            torch.zeros((2, 3, 224, 448), dtype=torch.bfloat16),
            _wan22_5b_contract_expert(),
            _TinyExpert().to(dtype=torch.bfloat16),
            "batch size 1",
        ),
        (
            torch.zeros((1, 3, 224, 448), dtype=torch.bfloat16),
            _TinyExpert(),
            _TinyExpert().to(dtype=torch.bfloat16),
            "Wan2.2 5B",
        ),
        (
            torch.zeros((1, 3, 224, 448), dtype=torch.bfloat16),
            _wan22_5b_contract_expert(torch.float32),
            _TinyExpert().to(dtype=torch.bfloat16),
            "BF16 parameters",
        ),
        (
            torch.zeros((1, 3, 224, 224), dtype=torch.bfloat16),
            _wan22_5b_contract_expert(),
            _TinyExpert().to(dtype=torch.bfloat16),
            "224x448",
        ),
        (
            torch.zeros((1, 3, 224, 448), dtype=torch.bfloat16),
            _wan22_5b_contract_expert(),
            _TinyExpert(),
            "ActionDiT BF16 parameters",
        ),
    ],
)
def test_accelerated_contract_rejects_unsupported_runtime(
    input_image: torch.Tensor,
    video_expert: nn.Module,
    action_expert: nn.Module,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rtc_ac_wam.validate_rtc_ac_accelerated_contract(
            input_image=input_image,
            video_expert=video_expert,
            action_expert=action_expert,
        )


def _expert_state(sequence: int, context: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randn(1, sequence, 8),
        "freqs": precompute_freqs_cis_1d(4, end=sequence).unsqueeze(1),
        "t_mod": torch.randn(1, sequence, 6, 8),
        "context": context,
        "context_mask": torch.ones(1, sequence, context.shape[1], dtype=torch.bool),
    }


@pytest.mark.parametrize("known_prefix_length", [0, 8])
def test_reference_shaped_accelerated_rtc_mot_matches_eager(
    known_prefix_length: int,
) -> None:
    torch.manual_seed(23)
    video = _TinyExpert().eval()
    action = _TinyExpert().eval()
    mot = RTCACMoT(
        experts={"video": video, "action": action},
        checkpoint_mixed_attn=False,
    ).eval()
    video_context = torch.randn(1, 5, 8)
    action_context = torch.randn(1, 5, 8)
    states = {
        "video": _expert_state(6, video_context),
        "action": _expert_state(32, action_context),
        "condition": _expert_state(16, action_context),
    }
    policy_mask = build_rtc_ac_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=known_prefix_length,
        device=torch.device("cpu"),
    )
    condition_mask = build_rtc_ac_condition_mask(
        video_seq_len=6,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
    )
    active = torch.tensor([known_prefix_length > 0])
    eager_states = {
        name: {key: value.clone() for key, value in state.items()}
        for name, state in states.items()
    }
    eager = mot.forward_rtc_ac(
        eager_states,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=active,
    )

    static_kv = {
        "video": tuple(
                (
                    block.cross_attn.norm_k(block.cross_attn.k(video_context[:, :4])),
                    block.cross_attn.v(video_context[:, :4]),
            )
            for block in video.blocks
        ),
        "action": tuple(
                (
                    block.cross_attn.norm_k(block.cross_attn.k(action_context[:, :4])),
                    block.cross_attn.v(action_context[:, :4]),
            )
            for block in action.blocks
        ),
    }
    accelerated = mot.forward_rtc_ac_accelerated(
        tokens_all={name: state["tokens"] for name, state in states.items()},
        freqs_all={name: state["freqs"] for name, state in states.items()},
        t_mod_all={name: state["t_mod"] for name, state in states.items()},
        context_all={
            name: {
                "context": state["context"],
                "mask": state["context_mask"],
                "static_cross_attention_kv": static_kv[
                    "video" if name == "video" else "action"
                ],
                "static_context_length": 4,
            }
            for name, state in states.items()
        },
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=active,
    )

    torch.testing.assert_close(accelerated["video"], eager["video"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(accelerated["action"], eager["action"], rtol=1e-5, atol=1e-6)


def test_acceleration_runtime_reuses_masks_and_schedules() -> None:
    runtime = RTCACAccelerationRuntime()
    mask_builds = 0
    schedule_builds = 0

    def build_mask() -> torch.Tensor:
        nonlocal mask_builds
        mask_builds += 1
        return torch.ones(3, 3, dtype=torch.bool)

    def build_schedule() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal schedule_builds
        schedule_builds += 1
        return torch.tensor([1.0]), torch.tensor([-1.0])

    first_mask = runtime.get_attention_mask(("d0", "cpu"), build_mask)
    second_mask = runtime.get_attention_mask(("d0", "cpu"), build_mask)
    first_schedule = runtime.get_schedule(("video", "cpu"), build_schedule)
    second_schedule = runtime.get_schedule(("video", "cpu"), build_schedule)

    assert first_mask is second_mask
    assert first_schedule[0] is second_schedule[0]
    assert first_schedule[1] is second_schedule[1]
    assert mask_builds == 1
    assert schedule_builds == 1


def test_acceleration_status_reports_actual_runtime_identity() -> None:
    runtime = RTCACAccelerationRuntime()

    status = runtime.status()

    assert status["runtime"] == {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_compute_capability": None,
    }


def test_acceleration_status_reports_scoped_compiler_counter_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        [
            {"dynamo_unique_graphs": 10, "inductor_cudagraph_skips": 2},
            {"dynamo_unique_graphs": 12, "inductor_cudagraph_skips": 5},
        ]
    )
    monkeypatch.setattr(
        rtc_ac_inference,
        "_read_compiler_counters",
        lambda: next(snapshots),
        raising=False,
    )
    runtime = RTCACAccelerationRuntime()
    runtime._compile_active = True

    status = runtime.status()

    assert status["dynamo_unique_graphs"] == 2
    assert status["dynamo_recompiles"] == 1
    assert status["inductor_cudagraph_skips"] == 3


def test_acceleration_runtime_refreshes_static_kv_in_place() -> None:
    torch.manual_seed(31)
    runtime = RTCACAccelerationRuntime()
    video = _TinyExpert().eval()
    action = _TinyExpert().eval()
    first_text = torch.randn(1, 5, 8)
    second_text = torch.randn(1, 5, 8)
    first_dynamic = torch.randn(1, 1, 8)
    second_dynamic = torch.randn(1, 1, 8)

    first_contexts = runtime.prepare_contexts(
        context_key="first",
        static_context=first_text,
        dynamic_context=first_dynamic,
        video_expert=video,
        action_expert=action,
    )
    first_key = runtime.static_cross_attention_kv["video"][0][0]
    first_pointer = first_key.data_ptr()
    first_value = first_key.clone()
    second_contexts = runtime.prepare_contexts(
        context_key="second",
        static_context=second_text,
        dynamic_context=second_dynamic,
        video_expert=video,
        action_expert=action,
    )
    second_key = runtime.static_cross_attention_kv["video"][0][0]

    assert second_key.data_ptr() == first_pointer
    assert not torch.equal(second_key, first_value)
    assert not torch.equal(first_contexts["video"][:, -1], second_contexts["video"][:, -1])


def test_acceleration_runtime_compiles_fullgraph_static(monkeypatch) -> None:
    runtime = RTCACAccelerationRuntime()
    calls = []

    class _Mot:
        @staticmethod
        def forward_rtc_ac_accelerated(*, value: torch.Tensor) -> torch.Tensor:
            return value + 1

    def fake_compile(function, **kwargs):
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    value = torch.tensor([2.0])
    mot = _Mot()

    first = runtime.run_mot(mot, value=value)
    second = runtime.run_mot(mot, value=value)

    torch.testing.assert_close(first, torch.tensor([3.0]))
    torch.testing.assert_close(second, torch.tensor([3.0]))
    assert len(calls) == 1
    assert calls[0][1] == {
        "mode": "reduce-overhead",
        "fullgraph": True,
        "dynamic": False,
    }


def test_acceleration_runtime_does_not_fallback_after_compile_error(monkeypatch) -> None:
    runtime = RTCACAccelerationRuntime()

    class _Mot:
        @staticmethod
        def forward_rtc_ac_accelerated(*, value: torch.Tensor) -> torch.Tensor:
            return value

    def fake_compile(function, **kwargs):
        del function, kwargs

        def fail(**call_kwargs):
            del call_kwargs
            raise RuntimeError("fullgraph failed")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.raises(RuntimeError, match="fullgraph failed"):
        runtime.run_mot(_Mot(), value=torch.tensor([1.0]))


def test_action_and_video_pre_dit_accept_cached_projected_context() -> None:
    torch.manual_seed(41)
    action = ActionDiT(
        hidden_dim=8,
        action_dim=7,
        ffn_dim=16,
        text_dim=6,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=1,
    ).eval()
    video = Wan22Dit(
        BackboneInfo(
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            attn_head_dim=4,
            ffn_dim=16,
            text_dim=6,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 1, 1),
            in_channels=2,
        )
    ).eval()
    context = torch.randn(1, 5, 6)
    mask = torch.ones(1, 5, dtype=torch.bool)
    action_projected = action.text_embedding(context)
    video_projected = video.text_embedding(context)

    action_state = action.pre_dit(
        torch.randn(1, 32, 7),
        torch.ones(1, 32),
        context,
        mask,
        projected_context=action_projected,
    )
    video_state = video.pre_dit(
        torch.randn(1, 2, 3, 2, 2),
        torch.ones(1),
        context,
        mask,
        projected_context=video_projected,
    )

    torch.testing.assert_close(action_state["context"], action_projected)
    torch.testing.assert_close(video_state["context"], video_projected)
