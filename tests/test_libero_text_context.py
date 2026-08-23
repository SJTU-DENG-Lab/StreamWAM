import torch
from types import SimpleNamespace

import examples.libero.rollout as rollout
from examples.libero.rollout import (
    _load_context,
    _new_context_memory_cache,
    _predict_action_chunk,
    _prepare_context_for_checkpoint,
)


def test_runtime_context_memory_cache_is_accelerated_only() -> None:
    assert _new_context_memory_cache(accelerated=False) is None
    assert _new_context_memory_cache(accelerated=True) == {}


def test_fastwam_context_exposes_zero_padding_like_reference_encode_prompt() -> None:
    context = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [9.0, 9.0],
            [8.0, 8.0],
        ]
    )
    mask = torch.tensor([True, True, False, False])

    actual_context, actual_mask = _prepare_context_for_checkpoint(
        context,
        mask,
        checkpoint_format="fastwam",
    )

    assert torch.equal(
        actual_context,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert torch.equal(actual_mask, torch.ones(4, dtype=torch.bool))


def test_starwam_context_preserves_padding_mask_and_values() -> None:
    context = torch.tensor([[1.0, 2.0], [9.0, 9.0]])
    mask = torch.tensor([True, False])

    actual_context, actual_mask = _prepare_context_for_checkpoint(
        context,
        mask,
        checkpoint_format="starwam",
    )

    assert torch.equal(actual_context, context)
    assert torch.equal(actual_mask, mask)


def test_runtime_context_memory_cache_avoids_reloading_task_tensor(
    tmp_path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "task.pt"
    cache_path.touch()
    loads = 0

    def fake_load_text_cache(path, text_len, text_dim):
        nonlocal loads
        del path, text_len, text_dim
        loads += 1
        return torch.ones(4, 2), torch.tensor([True, True, False, False])

    monkeypatch.setattr(rollout, "load_text_cache", fake_load_text_cache)
    config = SimpleNamespace(
        data=SimpleNamespace(
            text_len=4,
            text_prompt_template=None,
            text_cache_encoder_id=None,
            text_embedding_cache_dir=None,
        )
    )
    model = SimpleNamespace(
        backbone=SimpleNamespace(info=SimpleNamespace(text_dim=2))
    )
    memory_cache = {}

    first = _load_context(
        "task",
        config,
        {"task": cache_path},
        model,
        torch.device("cpu"),
        torch.float32,
        checkpoint_format="fastwam",
        memory_cache=memory_cache,
    )
    second = _load_context(
        "task",
        config,
        {"task": cache_path},
        model,
        torch.device("cpu"),
        torch.float32,
        checkpoint_format="fastwam",
        memory_cache=memory_cache,
    )

    assert loads == 1
    assert first[0].data_ptr() == second[0].data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr()


def test_predict_action_chunk_forwards_memory_cache_to_context_loader(
    monkeypatch,
) -> None:
    memory_cache = {}
    received = []
    inference_kwargs = []

    def fake_obs_to_image(
        obs,
        config,
        checkpoint_format="starwam",
        device=None,
        dtype=None,
    ):
        del obs, config, checkpoint_format, device, dtype
        return torch.zeros(1, 3, 2, 2), {"concat": torch.zeros(2, 2, 3).numpy()}

    def fake_load_context(*args, memory_cache=None, **kwargs):
        del args, kwargs
        received.append(memory_cache)
        return torch.zeros(1, 4, 2), torch.ones(1, 4, dtype=torch.bool)

    class FakeModel:
        def infer_action(self, **kwargs):
            inference_kwargs.append(kwargs)
            return torch.zeros(1, 32, 7)

    config = SimpleNamespace(
        data=SimpleNamespace(num_frames=9, action_freq_ratio=1),
        framework=SimpleNamespace(chunk_size=32, proprio_dim=None, type="mot"),
        inference=SimpleNamespace(),
    )
    monkeypatch.setattr(rollout, "_obs_to_image", fake_obs_to_image)
    monkeypatch.setattr(rollout, "_load_context", fake_load_context)

    _predict_action_chunk(
        model=FakeModel(),
        obs={},
        task_description="task",
        config=config,
        task_cache={},
        action_stats=None,
        state_stats=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_inference_steps=1,
        action_num_inference_steps=1,
        sampling_method="rtc_ac",
        checkpoint_format="fastwam",
        seed=0,
        context_memory_cache=memory_cache,
    )

    assert received == [memory_cache]
    assert inference_kwargs[0]["rand_device"] == "cpu"
