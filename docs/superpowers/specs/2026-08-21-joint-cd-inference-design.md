# Joint CD 1-Step Inference Design

## Scope

Add direct synchronous inference for the FastWAM Joint CD step-3400 checkpoint. The policy jointly predicts world and action with one consistency step, an action horizon of 32, and synchronous replanning every 16 environment steps. RTC and accelerated RTC are out of scope.

## Command contract

The existing `--checkpoint-format fastwam` continues to describe the checkpoint payload. A separate `--sampling-method consistency` option selects CD inference semantics. Existing commands retain `euler` as their default and are unaffected.

The supported Joint CD geometry is:

- `sampling_method=consistency`
- `num_inference_steps=1`
- `action_horizon=32`
- `action_video_conditioning=full_video`
- LIBERO `replan_steps=16`

## Architecture

Pure consistency-boundary math lives in `streamwam/inference/consistency.py`. `MoTWAM` selects either the existing Euler update or the new consistency update without duplicating its inference loop. `ActionDiT.pre_dit` accepts both batch-level `[B]` and token-wise `[B,H]` timesteps, matching the original FastWAM model contract while preserving existing `[B]` behavior.

The existing FastWAM checkpoint loader remains unchanged because the checkpoint contains the already-supported `mot`, `proprio_encoder`, `step`, and `torch_dtype` fields.

## Configuration and rollout

A dedicated LIBERO recipe selects `full_video`, horizon 32, one inference step, and consistency sampling. The rollout CLI exposes `--sampling-method {euler,consistency}` and forwards it to the model. A separate flat Bash launcher supplies the Joint CD checkpoint, statistics, backbone, LIBERO path, one-step sampling, and replanning interval.

The source checkpoint is exposed under StreamWAM's `checkpoints/` directory with a symbolic link; checkpoint contents are not copied or converted.

## Validation

Tests cover consistency-boundary numerics, timestep broadcasting and validation, ActionDiT batch/token timestep behavior, sampler dispatch and invalid Joint CD geometry, CLI forwarding, and the checkpoint symlink target. Existing Euler tests must continue to pass. When GPU assets are available, a fixed-input/fixed-seed comparison against the source repository is the final parity check.

## Non-goals

This change does not add RTC state, asynchronous execution, CUDA Graphs, `torch.compile`, prompt K/V caching, checkpoint conversion, or Git commits.
