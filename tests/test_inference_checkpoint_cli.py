from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from examples.libero.rollout import (
    _add_libero_to_path,
    _build_arg_parser as build_libero_parser,
    _configure_mujoco_runtime,
    _prepare_runtime_config,
    _resolve_inference_args,
)
from examples.robotwin.policy_server import _build_arg_parser as build_robotwin_server_parser
from streamwam.config import StreamWAMConfig
from streamwam.checkpointing import prepare_inference_config


def test_fastwam_config_skips_separate_action_init() -> None:
    config = StreamWAMConfig()
    config.framework.action_expert_init_from = "/unused/action-init.pt"

    prepared = prepare_inference_config(config, checkpoint_format="fastwam")

    assert prepared is config
    assert prepared.framework.action_expert_init_from is None


def test_streamwam_config_keeps_action_init() -> None:
    config = StreamWAMConfig()
    config.framework.action_expert_init_from = "/needed/action-init.pt"

    prepare_inference_config(config, checkpoint_format="streamwam")

    assert config.framework.action_expert_init_from == "/needed/action-init.pt"


def test_prepare_config_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        prepare_inference_config(StreamWAMConfig(), checkpoint_format="unknown")


def test_libero_checkpoint_format_defaults_to_streamwam() -> None:
    args = build_libero_parser().parse_args(["--config", "recipe.yaml"])

    assert args.checkpoint_format == "streamwam"


def test_libero_checkpoint_format_accepts_fastwam() -> None:
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--checkpoint-format", "fastwam"]
    )

    assert args.checkpoint_format == "fastwam"


def test_libero_parser_accepts_consistency_sampling() -> None:
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--sampling-method", "consistency"]
    )

    assert args.sampling_method == "consistency"


def test_libero_parser_accepts_ac_stream_sampling() -> None:
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--sampling-method", "ac-stream"]
    )

    assert args.sampling_method == "ac-stream"


def test_libero_parser_accepts_ac_stream_acceleration() -> None:
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--ac-stream-accelerated"]
    )

    assert args.ac_stream_accelerated is True


def test_ac_stream_acceleration_rejects_non_ac_stream_sampling() -> None:
    config = StreamWAMConfig()
    config.inference.sampling_method = "euler"
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--ac-stream-accelerated"]
    )

    with pytest.raises(ValueError, match="requires sampling_method='ac-stream'"):
        _resolve_inference_args(config, args)


def test_ac_stream_acceleration_requires_cuda_device() -> None:
    config = StreamWAMConfig()
    config.framework.variant = "ac-stream"
    config.framework.chunk_size = 32
    config.data.num_frames = 33
    config.data.action_freq_ratio = 4
    config.inference.sampling_method = "ac-stream"
    config.inference.num_inference_steps = 1
    config.inference.replan_steps = 16
    args = build_libero_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--ac-stream-accelerated",
            "--device",
            "cpu",
        ]
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        _resolve_inference_args(config, args)


def test_ac_stream_recipe_resolves_eager_h32_s16_d8() -> None:
    config = StreamWAMConfig()
    config.framework.variant = "ac-stream"
    config.framework.chunk_size = 32
    config.data.num_frames = 33
    config.data.action_freq_ratio = 4
    config.inference.sampling_method = "ac-stream"
    config.inference.num_inference_steps = 1
    config.inference.replan_steps = 16
    args = build_libero_parser().parse_args(["--config", "recipe.yaml"])

    _resolve_inference_args(config, args)

    assert args.sampling_method == "ac-stream"
    assert args.replan_steps == 16
    assert args.num_inference_steps == 1


def test_joint_cd_recipe_resolves_replan_16() -> None:
    config = StreamWAMConfig()
    config.inference.sampling_method = "consistency"
    config.inference.num_inference_steps = 1
    config.inference.replan_steps = 16
    args = build_libero_parser().parse_args(["--config", "recipe.yaml"])

    _resolve_inference_args(config, args)

    assert args.replan_steps == 16
    assert args.num_inference_steps == 1
    assert args.sampling_method == "consistency"


def test_joint_cd_rejects_replan_other_than_16() -> None:
    config = StreamWAMConfig()
    config.inference.sampling_method = "consistency"
    args = build_libero_parser().parse_args(
        ["--config", "recipe.yaml", "--replan-steps", "5"]
    )

    with pytest.raises(ValueError, match="replan_steps=16"):
        _resolve_inference_args(config, args)


@pytest.mark.parametrize("alias", ["cd", "lcm", "Consistency"])
def test_joint_cd_aliases_still_require_replan_16(alias: str) -> None:
    config = StreamWAMConfig()
    config.inference.sampling_method = alias
    config.inference.replan_steps = 5
    args = build_libero_parser().parse_args(["--config", "recipe.yaml"])

    with pytest.raises(ValueError, match="replan_steps=16"):
        _resolve_inference_args(config, args)


def test_libero_parser_accepts_explicit_runtime_paths_and_renderer() -> None:
    args = build_libero_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--backbone-path",
            "/models/wan22_5b",
            "--stats-path",
            "/models/fastwam/stats.json",
            "--mujoco-gl",
            "osmesa",
        ]
    )

    assert args.backbone_path == "/models/wan22_5b"
    assert args.stats_path == "/models/fastwam/stats.json"
    assert args.mujoco_gl == "osmesa"


def test_libero_runtime_paths_replace_recipe_placeholders() -> None:
    config = StreamWAMConfig()
    config.training.output_dir = "/path/to/output/streamwam_libero"
    config.data.text_embedding_cache_dir = "/path/to/output/streamwam_libero/text_embedding_cache"
    config.data.dataset_dirs = ["/path/to/libero_spatial_lerobot"]
    args = build_libero_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--checkpoint-format",
            "fastwam",
            "--backbone-path",
            "/models/wan22_5b",
            "--stats-path",
            "/models/fastwam/stats.json",
        ]
    )

    prepared = _prepare_runtime_config(config, args)

    assert prepared is config
    assert config.backbone.pretrained_model_id == "/models/wan22_5b"
    assert config.data.action_stats_path == "/models/fastwam/stats.json"
    assert config.data.state_stats_path == "/models/fastwam/stats.json"
    assert config.training.output_dir == "outputs/fastwam_libero_eval"
    assert config.data.text_embedding_cache_dir == "outputs/fastwam_libero_eval/text_embedding_cache"
    assert config.data.dataset_dirs == []


def test_libero_mujoco_renderer_is_configured_in_process(monkeypatch) -> None:
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)

    _configure_mujoco_runtime("egl")

    assert os.environ["MUJOCO_GL"] == "egl"
    assert os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_osmesa_configuration_does_not_load_a_second_cpp_runtime() -> None:
    script = """
from examples.libero.rollout import _configure_mujoco_runtime

def loaded_cpp_runtimes():
    with open('/proc/self/maps', encoding='utf-8') as handle:
        return {line.split()[-1] for line in handle if 'libstdc++.so' in line}

before = loaded_cpp_runtimes()
_configure_mujoco_runtime('osmesa')
after = loaded_cpp_runtimes()
assert after == before, (before, after)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_explicit_libero_home_refreshes_cached_config(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    first_home = tmp_path / "first_libero"
    second_home = tmp_path / "second_libero"
    (first_home / "libero" / "libero").mkdir(parents=True)
    (second_home / "libero" / "libero").mkdir(parents=True)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(sys, "path", list(sys.path))

    _add_libero_to_path(str(first_home))
    _add_libero_to_path(str(second_home))

    config = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["benchmark_root"] == str(second_home / "libero" / "libero")
    assert config["datasets"] == str(second_home / "libero" / "datasets")


def test_robotwin_server_checkpoint_format_defaults_to_streamwam() -> None:
    args = build_robotwin_server_parser().parse_args(
        ["--config", "recipe.yaml", "--checkpoint", "model.pt"]
    )

    assert args.checkpoint_format == "streamwam"


def test_robotwin_server_checkpoint_format_accepts_fastwam() -> None:
    args = build_robotwin_server_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--checkpoint-format",
            "fastwam",
        ]
    )

    assert args.checkpoint_format == "fastwam"


def test_robotwin_server_checkpoint_format_accepts_starwam() -> None:
    args = build_robotwin_server_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--checkpoint-format",
            "starwam",
        ]
    )

    assert args.checkpoint_format == "starwam"


def test_robotwin_server_accepts_three_inference_modes() -> None:
    for mode in ("baseline", "cd", "ac-stream"):
        args = build_robotwin_server_parser().parse_args(
            [
                "--config",
                "recipe.yaml",
                "--checkpoint",
                "model.pt",
                "--inference-mode",
                mode,
            ]
        )
        assert args.inference_mode == mode


def test_robotwin_server_accepts_ac_stream_backend_flags() -> None:
    accelerated = build_robotwin_server_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--inference-mode",
            "ac-stream",
            "--ac-stream-accelerated",
        ]
    )
    eager = build_robotwin_server_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--inference-mode",
            "ac-stream",
            "--ac-stream-eager",
        ]
    )

    assert accelerated.ac_stream_accelerated is True
    assert eager.ac_stream_eager is True
