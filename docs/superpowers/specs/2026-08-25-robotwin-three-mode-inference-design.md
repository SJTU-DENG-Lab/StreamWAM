# RoboTwin Three-Mode Inference Design

## Goal

Add three reproducible RoboTwin 2.0 inference modes to StreamWAM without
converting or duplicating the original StarWAM checkpoints:

1. `baseline`: official StarWAM MoT first-frame action inference, four Euler
   steps, and `replan_steps=32`.
2. `cd`: StarWAM first-frame one-step action consistency inference using the
   yzy CD checkpoint, with `replan_steps=32`.
3. `ac-stream`: yzy's three-stream RTC inference with `H=32`, `s=16`, and
   `d=8`. The same implementation supports eager and accelerated backends;
   formal evaluation defaults to the accelerated backend.

The target one-trial evaluation contains all 50 RoboTwin tasks in both
`demo_clean` and `demo_randomized`, for 100 episodes distributed across four
user-selected GPUs.

## Terminology and Algorithm Contracts

The released StarWAM RoboTwin MoT checkpoint is jointly trained over world and
action streams. Its official evaluation path is nevertheless first-frame
conditioned action inference rather than full-video world/action sampling.
The implementation must preserve this distinction.

### Baseline

- Model: StarWAM MoT RoboTwin checkpoint.
- Conditioning: `action_video_conditioning=first_frame`.
- Sampling: four-step Euler action denoising.
- Model output used by the policy: 14-D action chunk.
- Horizon and execution: predict the configured chunk and execute 32 actions
  before replanning.

### CD

- Model: yzy StarWAM CD checkpoint.
- Conditioning: `action_video_conditioning=first_frame`.
- Sampling: exactly one action consistency boundary.
- The first-frame video tokens are encoded and cached; no future video is
  sampled during evaluation.
- This mode is distinct from the existing LIBERO full-video Joint-CD path. Both
  variants must remain available without silently changing one into the other.
- Horizon and execution: use the checkpoint's configured horizon and execute
  32 actions before replanning.

### AC-Stream

- Model: yzy Stage-2 RTC checkpoint.
- Model streams: noisy future video, clean action-prefix condition, and policy
  action.
- Geometry: `H=32`, stride `s=16`, inference delay `d=8`.
- D0 obtains the initial chunk without a clean action prefix.
- At the D8 launch point, inference for the next chunk starts in a background
  worker while the simulator continues executing the current chunk.
- At the stride boundary, the controller installs the completed prediction or
  waits only for its unfinished tail.
- `--ac-stream-accelerated` changes only the execution backend. It must not
  change inputs, masks, schedules, checkpoint weights, or outputs.

## Architecture

Use a common RoboTwin evaluation shell around two model implementations and
three sampling strategies:

```text
RoboTwin simulator process (motus environment)
    -> RoboTwin observation adapter
    -> remote policy / AC-Stream controller
    -> socket RPC
    -> inference server (wyx PyTorch environment)
         baseline  -> MoTWAM first-frame Euler
         cd        -> MoTWAM first-frame consistency
         ac-stream -> ACStreamWAM RTC (eager or accelerated)
```

The simulator and inference server remain separate processes because the wyx
inference environment provides the required PyTorch/Triton acceleration stack
but does not provide RoboTwin's SAPIEN/MPLib runtime.

## Public CLI

The inference server and multi-GPU manager expose one mode selector:

```text
--inference-mode baseline|cd|ac-stream
```

AC-Stream additionally accepts:

```text
--ac-stream-accelerated
--ac-stream-eager
```

The two backend flags are mutually exclusive and are invalid for `baseline`
and `cd`. The AC-Stream multi-GPU manager defaults to accelerated; passing
`--ac-stream-eager` selects the reference backend explicitly. The inference
server receives the resolved backend and uses `--ac-stream-accelerated` only
when acceleration is active.

Checkpoint loading uses:

```text
--checkpoint-format starwam
```

The loader accepts original checkpoint files or DeepSpeed checkpoint
directories and never writes a converted checkpoint.

## Model Layer

### `streamwam/wam/mot_wam.py`

Retain the existing Euler action path and full-video Joint-CD path. Add a
first-frame consistency action branch selected when sampling is consistency and
`action_video_conditioning=first_frame`.

The branch must:

1. encode the observation frame;
2. prefill the video cache;
3. sample action noise using the checkpoint-compatible RNG contract;
4. evaluate the action velocity once at the consistency timestep;
5. apply the yzy action consistency boundary;
6. return only the action chunk.

The existing full-video validation remains local to full-video Joint-CD and
must not reject the RoboTwin first-frame CD mode.

### `streamwam/wam/ac_stream_wam.py`

Preserve the three-stream RTC mathematics while making tensor geometry derive
from configuration and actual inputs. Remove LIBERO-only model constraints:

- action dimension comes from `framework.action_dim`;
- proprio dimension comes from `framework.proprio_dim`;
- image and latent geometry come from the input and VAE;
- horizon, stride, and delay come from the inference configuration.

RoboTwin uses a 14-D action/proprio representation and a `384x320` composed RGB
frame. LIBERO continues to use its existing 7-D and `224x448` contract.

The accelerated path must share the eager path's model parameters and masks.
Static buffers and compiled callables may be specialized per validated input
contract, but the specialization must be explicit and reported.

## Checkpoint Loading

Add a StarWAM source-format adapter under `streamwam/checkpointing/` and register
it in the existing loader registry.

The adapter must:

- extract model state from original StarWAM `.pt` payloads;
- resolve DeepSpeed checkpoint directories to their model-state file;
- normalize only known wrapper prefixes;
- load baseline and CD states into `MoTWAM`;
- load RTC state into `ACStreamWAM`, including
  `rtc_slot_state_embedding.weight`;
- validate model variant, action dimension, proprio dimension, and required RTC
  keys before mutating the target model;
- report missing and unexpected keys as an error unless explicitly documented
  compatibility keys are involved;
- return metadata describing source path, mode, step when available, and tensor
  counts;
- never save converted weights.

Dataset statistics remain external JSON files and are loaded without copying.

## RoboTwin Policy and Transport

The existing camera adapter remains authoritative:

- head camera resized to `256x320`;
- left and right wrist cameras resized to `128x160`;
- wrist views concatenated horizontally beneath the head view;
- final RGB input shape `3x384x320` in `[-1, 1]`;
- state and action remain in the native 14-D RoboTwin qpos order.

Baseline and CD use synchronous request/response at each replan boundary.

AC-Stream uses the existing transport-neutral AC-Stream controller with a
remote predictor callable. A single background worker owns inference requests
so socket access is serialized. The simulator thread continues calling
`task_env.take_action` while the background request is outstanding. Requests
carry the previous action chunk, launch cursor, instruction, observation, and
state required to reproduce D0/D8 semantics. Episode reset drains or cancels
the outstanding prediction and clears all controller state.

Protocol responses include the action chunk, server-side model inference time,
backend metadata, and any structured error. Client-side communication and
action execution timing use monotonic wall clocks.

## Recipes and Launchers

Use two recipes:

- `examples/robotwin/configs/recipes/streamwam_robotwin_mot_wan22_5b.yaml`
  for baseline and CD;
- `examples/robotwin/configs/recipes/streamwam_robotwin_ac_stream_wan22_5b.yaml`
  for AC-Stream.

Provide three launchers:

- `launch_streamwam_robotwin_baseline_4gpu.sh`;
- `launch_streamwam_robotwin_cd_4gpu.sh`;
- `launch_streamwam_robotwin_ac_stream_4gpu.sh`.

Shell scripts contain only stable defaults and a Python invocation. Environment
preflight, path validation, GPU allocation, worker lifecycle, and result merging
belong in Python.

The launchers accept arbitrary `GPU_IDS`, checkpoint, statistics, backbone,
RoboTwin home, number of trials, task/config filters, and output directory.
Their default trial count is one. A complete official-scale run can set the
trial count to 50 without changing code.

## Multi-GPU Evaluation

For each selected GPU, start one inference server in the wyx environment and
one RoboTwin simulator worker in the motus environment. Allocate task/config
jobs dynamically so faster workers receive additional jobs and all GPUs remain
approximately balanced.

One trial means:

```text
50 tasks x 2 configurations x 1 episode = 100 episodes
```

The manager must fail the evaluation if a worker exits unexpectedly, a result
is missing, or duplicate task/config/trial identities appear. Partial results
remain on disk for diagnosis but are not reported as a complete score.

## Timing and Output

The final console summary prints once after all workers finish:

```text
=== Multi-GPU RoboTwin Evaluation Summary ===
Mode: ac-stream accelerated
GPUs: 0,1,2,3
Tasks/configurations: 100
Trials: 100
Success: 88/100 (88.00%)
Chunk Time: 4x.xx ms
Total Time / Episode: x.xx s
Results: /absolute/path/results.json
```

Definitions:

- `Chunk Time`: arithmetic mean of server-side model inference durations for
  measured chunks. Accelerated AC-Stream excludes documented compile/prewarm
  work and uses the same warmup exclusion rule as the wyx reference.
- `Total Time / Episode`: arithmetic mean of simulator episode wall time from
  episode reset completion until success/failure termination. Model loading,
  compilation, server startup, and task asset loading are excluded.

## Per-Job Supervision Amendment

The simulator side must not keep one RoboTwin process alive across many jobs.
Each `(task, config, trial)` runs in a fresh process while the per-GPU inference
server remains resident. The manager dynamically assigns the next pending job
to each free GPU.

Each job writes an atomic sidecar after completion. A parent-side watchdog,
configured by `--job-timeout-seconds` and defaulting to 1200 seconds, supervises
the entire process group. On timeout the manager captures the job identity and
phase diagnostics, terminates the full simulator process group, records
`skipped_timeout`, and immediately schedules the next job. Timed-out jobs are
never retried automatically.

Skipped jobs are excluded from the success-rate denominator and from Chunk
Time and Total Time. A run containing skips is reported as `INCOMPLETE` with
planned, completed, and skipped counts. It still writes all partial results but
returns a nonzero exit status. Resume mode reuses completed or skipped sidecars
without silently rerunning them.

The timing boundary is unchanged: measured chunks exclude prewarm, and episode
time starts at the first real policy step and ends at success or the environment
step limit. Process startup, expert seed checking, asset setup, watchdog time,
and process cleanup remain outside reported model metrics.

The JSON result additionally retains per-episode identities and raw timing
totals so aggregation can be audited, while the console remains concise.

## Error Handling

Reject before evaluation:

- checkpoint/model-mode mismatch;
- missing RTC-specific tensors;
- incompatible action/proprio dimensions;
- invalid AC-Stream geometry;
- acceleration requested without CUDA or with an unsupported runtime;
- duplicate GPU IDs or unavailable devices;
- inaccessible backbone, checkpoint, statistics, or RoboTwin paths.

Worker errors include the GPU, task, configuration, trial, server log, and
simulator log paths. Socket failures propagate as evaluation failures rather
than silently retrying an action from stale state.

## Verification

### Unit tests

- CLI mode selection and invalid flag combinations.
- StarWAM checkpoint file and DeepSpeed-directory resolution.
- Pre-mutation validation for baseline, CD, and RTC checkpoints.
- First-frame CD boundary against the yzy reference on deterministic tensors.
- Dynamic AC-Stream geometry for both LIBERO and RoboTwin dimensions.
- Eager/accelerated AC-Stream numerical agreement within the established
  bfloat16 tolerance.
- D0/D8 controller launch, overlap, boundary wait, reset, and worker failure.
- Multi-GPU workload uniqueness, completeness, and timing aggregation.

### Reference parity tests

For one fixed RoboTwin observation and seed, compare StreamWAM against yzy for:

- baseline action chunk;
- CD action chunk;
- RTC D0 action chunk;
- RTC D8 action chunk with the same clean prefix.

Compare intermediate masks, timesteps, normalized inputs, and pre-denormalized
actions before relying on end-to-end success rate.

### Evaluation acceptance

Run four-card, one-trial evaluations for all three modes. Verify:

- 100 unique episodes complete per mode;
- success-rate trends are compatible with the 10-trial yzy reference, while
  acknowledging one-trial sampling noise;
- eager and accelerated AC-Stream use identical checkpoint and algorithm
  inputs;
- accelerated steady-state chunk time approaches the wyx reference environment;
- AC-Stream inference overlaps action execution and has no unexplained stale or
  geometry misses.

## Non-Goals

- Retraining or redistilling baseline/CD/RTC checkpoints.
- Converting checkpoints into StreamWAM-native files.
- Claiming the official baseline or yzy CD performs full-video joint sampling.
- Adding dataset-specific branches to the generic checkpoint loader.
- Changing existing LIBERO inference semantics or launch commands.
