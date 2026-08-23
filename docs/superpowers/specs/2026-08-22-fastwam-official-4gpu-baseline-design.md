# FastWAM Official Four-GPU Baseline Design

## Goal

Add a clean four-GPU, one-trial-per-task LIBERO launcher for the released
FastWAM checkpoint. Its output must use the same aggregation and timing
definitions as the existing Joint CD and RTC-AC four-GPU launchers.

## Evaluation contract

- Checkpoint: `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
- Statistics: `checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json`
- Sampling: synchronous 10-step Euler
- Action horizon: 32 actions
- Replanning: execute 10 actions per generated chunk
- Workload: all 40 LIBERO tasks, trial 0 only
- GPUs: configurable through `GPU_IDS`, default `0,1,2,3`
- Seed: fixed seed 42
- Rendering: EGL
- Engineering acceleration: none; no RTC, `torch.compile`, or CUDA Graph

## Architecture

Create only `examples/libero/scripts/launch_streamwam_libero_fastwam_4gpu.sh`.
The launcher delegates workload balancing, persistent workers, result merging,
and timing to the existing `examples/libero/multigpu_rollout.py`. Backbone and
LIBERO paths remain environment-overridable, matching the CD and RTC launchers.

No model, sampler, rollout, checkpoint loader, or timing implementation changes
are needed.

## Comparison metrics

The four-mode primary comparison uses:

- chunk time: `average_inference_ms_per_chunk`, the synchronized model call;
- total time: `average_episode_wall_ms`, measured after initial state setup until
  episode success or timeout.

`command_wall_ms` is retained for operational reporting but excluded from the
model comparison because it includes process startup, model loading, environment
initialization, and video output.

## Validation

An integration-style test executes the launcher with a fake `python` executable
and asserts the emitted manager command uses all four suites, one trial, the
released FastWAM checkpoint, 10 Euler steps, replan 10, fixed seed, EGL, and the
caller-provided GPU/path overrides. Shell syntax is checked with `bash -n`.

## Constraints

- Do not create Git commits.
- Preserve all existing working-tree changes.
- Do not change the behavior of Joint CD, RTC-AC, or ordinary rollout code.
