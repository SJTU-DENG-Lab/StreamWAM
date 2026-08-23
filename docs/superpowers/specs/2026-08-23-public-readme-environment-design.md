# StreamWAM Public README and Environment Design

**Date:** 2026-08-23

## Goal

Present StreamWAM as a clean open-source research repository and provide one reproducible environment for the validated RTC-AC accelerated path. The public entry point should follow the compact organization used by WaveForcing without copying project-specific claims or assets.

## README Structure

The root README will be organized as follows:

1. Centered project name, one-line positioning, and only valid public badges.
2. A concise description of StreamWAM, RTC-AC, and asynchronous world-action inference.
3. A release-status table that distinguishes available code from unreleased assets.
4. A short Quick Start covering environment installation, LIBERO preparation, model/config preparation, and the four-GPU RTC-AC launch command.
5. A current-results table reporting the validated H100 baseline: 45.20 ms/chunk mean, 45.75 ms p50, 46.50 ms p90, zero recompiles, zero CUDA Graph skips, and zero deadline misses in the recorded four-chunk steady-state window.
6. The accelerated runtime contract and expected diagnostic values.
7. A compact repository tree.
8. Links to the detailed LIBERO guide, followed by citation, license, and acknowledgements.

The README will not expose machine-specific paths, imply that unreleased checkpoints are public, or list unsupported performance backends.

## Environment Interface

The root `pyproject.toml` will be the canonical environment definition, matching WaveForcing's single-entry approach. A root `.python-version` will record Python 3.10.20 as the exact reproduced interpreter while project metadata accepts compatible Python 3.10 patch releases. The environment definition will:

- constrain Python to the validated 3.10 series;
- pin the validated PyTorch 2.7.1/cu128, torchvision 0.22.1/cu128, Triton 3.3.1, and supporting packages;
- define the PyTorch cu128 package index for `uv`;
- keep the environment installable as the `streamwam` package;
- separate nonessential training dependencies from the validated LIBERO inference stack when separation remains useful.

The duplicate `examples/libero/requirements.txt` will be removed once all unique dependencies have been represented in `pyproject.toml`. The public install command will use `uv`; the detailed LIBERO guide will point back to that canonical command instead of maintaining a second dependency list.

No environment filename will include internal labels such as `rtc-ac-cu128`.

## Runtime and Compatibility Notes

The documentation will state that the measured path uses PyTorch Inductor/Triton rather than TensorRT. It will distinguish the CUDA 12.8 runtime bundled with the PyTorch wheel from the host CUDA Toolkit and explain that a compatible NVIDIA driver is the governing requirement for this tested configuration.

LIBERO remains an external source checkout supplied with `--libero-home`. The documented directory contract will include `libero/libero/{benchmark,bddl_files,init_files,assets}` and note that `datasets/` is optional for rollout-only use.

The accelerated mode contract is CUDA, BF16, batch size one, image tensor `[1, 3, 224, 448]`, Wan2.2 5B geometry, and RTC parameters `H=32`, `stride=16`, `delay=8`, nine video frames, and one inference step.

## Verification

Before completion:

- build/install metadata will be validated from `pyproject.toml`;
- stale README commands and old package names will be searched for;
- referenced scripts and paths will be checked for existence;
- the existing test suite will be run;
- the working-tree diff will be reviewed for unrelated changes and machine-specific paths.

Performance numbers are documentation of the supplied validated run, not a benchmark rerun requirement for this documentation change.
