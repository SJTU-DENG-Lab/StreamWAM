# RTC-AC Optional Acceleration Design

## Goal

Add an opt-in accelerated backend to the existing StarWAM RTC-AC inference
path. A single `--rtc-ac-accelerated` flag enables the wyx Stage-2 z1
acceleration contract without creating another inference module or changing
the checkpoint, RTC mathematics, D0/D8 scheduling, action normalization, or
success behavior.

## Alternatives considered

### Full wyx-equivalent acceleration in the existing RTC-AC path — selected

Compile the fixed-shape MoT core, cache static prompt-derived work, reuse D0/D8
masks and schedules, and prewarm both phases. This provides a meaningful fourth
benchmark group and isolates engineering acceleration from the RTC algorithm.

### Compile-only acceleration

Wrapping the current `forward_rtc_ac` with `torch.compile` is smaller, but the
current block API stores Python-side mutable `_cache` state and repeats all text
projection and cross-attention K/V work. Fullgraph compilation is therefore
fragile and the expected speed is materially below the wyx result.

### Cache-only acceleration

Mask, schedule, and K/V caching is robust but cannot deliver the compiled CUDA
Graph path requested for the fourth group. It is useful as an internal layer of
the selected solution, not as the public accelerated mode.

## Public interface

Both LIBERO frontends accept one boolean flag:

```text
--rtc-ac-accelerated
```

The multi-GPU manager forwards it unchanged to every worker. Existing RTC
launchers append their own positional arguments to the Python command, so the
same launcher supports both modes:

```bash
# Group 3: eager RTC-AC
GPU_IDS=0,1,2,3 \
  bash examples/libero/scripts/launch_starwam_libero_rtc_ac_4gpu.sh

# Group 4: accelerated RTC-AC
GPU_IDS=0,1,2,3 \
  bash examples/libero/scripts/launch_starwam_libero_rtc_ac_4gpu.sh \
  --rtc-ac-accelerated
```

The flag is valid only with `sampling_method=rtc_ac`, `framework.variant=rtc_ac`,
CUDA, and the fixed H32/s16/d8 one-step geometry. Invalid combinations fail
before evaluation begins.

## Architecture

### `starwam/inference/rtc_ac.py`

This remains the only RTC-AC inference runtime module. Add a focused
`RTCACAccelerationRuntime` that owns:

- acceleration enablement and strict status;
- `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)` for
  the fixed-shape MoT core;
- stable-address static text projection and per-layer cross-attention K/V
  caches, updated in place when the task prompt changes;
- separate D0 and D8 policy-mask cache entries and the shared condition mask;
- one-step video/action schedule tensors;
- D0/D8 prewarm completion state and a serializable status report.

Compilation uses PyTorch Inductor's reduce-overhead CUDA Graph Trees. No manual
`torch.cuda.CUDAGraph` wrapper is added because it would duplicate Inductor's
graph management and introduce a second replay lifecycle around the async
controller.

The runtime is absent/disabled on the eager path. It must not mutate global
PyTorch compiler settings.

### `starwam/modules/wan_block.py`

Expose a functional block path that returns the AdaLN/residual state explicitly
instead of writing `block._cache`. Add a matching post-attention operation that
accepts already-projected cross-attention K/V. Existing `get_qkv` and
`post_attention` behavior remains unchanged for every eager caller.

This removes Python mutation from the compiled graph while sharing the same
parameters, modulation, attention, residual, and FFN formulas as eager.

### `starwam/modules/rtc_ac.py`

Keep `RTCACMoT` as the only RTC MoT class. Add one compile-friendly accelerated
forward method in the same class. It consumes fixed-shape expert states,
D0/D8 masks, and cached static cross-attention K/V; it computes K/V only for the
dynamic proprio token and reuses the ActionDiT result for the action and
condition streams.

The existing `forward_rtc_ac` remains the eager reference implementation.

### `starwam/wam/rtc_ac_wam.py`

`RTCACWAM` owns an optional `RTCACAccelerationRuntime`. Its public
`enable_rtc_ac_acceleration()` method is called after checkpoint loading and
before the first episode. `infer_action` dispatches only the MoT execution and
cache preparation through the runtime; noise generation, first-frame pinning,
D0/D8 hard prefix, consistency boundaries, and returned actions stay shared.

Static text tokens are separated from the appended proprio token. Text
projections and their per-layer K/V values are cached per current task;
proprio projection and K/V remain dynamic on every chunk. Cache tensors are
updated in place when the task changes so compiled/CUDA-graph addresses remain
stable.

### LIBERO rollout and multi-GPU manager

`examples/libero/rollout.py`:

- parses and validates `--rtc-ac-accelerated`;
- enables acceleration after checkpoint loading;
- keeps an in-memory raw text-context cache per worker so repeated chunks do
  not reload the same task tensor from disk;
- before the first measured RTC episode, runs one real-shaped D0 call followed
  by one D8 call using the same task observation and normalized action prefix;
- resets the environment after prewarm and only then starts episode timing;
- excludes both prewarm calls from chunk, overlap, episode, and workload
  metrics;
- records acceleration status once in `results.json` and the final summary.

`examples/libero/multigpu_rollout.py` forwards the flag and requires every
worker to report the same active acceleration status before merging results.

Both RTC shell launchers append `"$@"` to their Python command. No shell branch
or second accelerated launcher is introduced.

## Timing and comparison contract

Group 3 and Group 4 use the identical checkpoint, seed, H32/s16/d8 scheduler,
task/trial assignments, and EGL configuration.

- raw chunk time remains `average_inference_ms_per_chunk`;
- total time remains `average_episode_wall_ms`;
- compilation and D0/D8 prewarm are excluded from both metrics;
- complete command wall time still includes initialization and prewarm and is
  not used as the model-speed comparison;
- RTC overlap metrics retain their current definitions.

The final log identifies the backend as either `rtc_ac/eager` or
`rtc_ac/accelerated`, reports compile/prewarm/cache status, and prints only once
per complete command.

## Failure behavior

Acceleration is strict because the flag asserts a benchmark mode:

- compilation, fullgraph capture, D0 prewarm, or D8 prewarm failure aborts the
  worker with the original exception;
- no automatic fallback to eager is allowed;
- the multi-GPU manager terminates sibling workers and points to the failed
  worker log, matching existing worker-failure behavior;
- eager RTC behavior is unchanged when the flag is absent.

## Correctness and regression testing

Tests cover:

1. CLI parsing, validation, and multi-GPU forwarding.
2. The eager functional block output versus the cache-free reference formulas.
3. Cached and uncached RTC MoT outputs for both D0 and D8 using fixed seeds.
4. Static text cache refresh with stable tensor addresses and dynamic proprio
   K/V changes.
5. Mask and schedule cache reuse.
6. `torch.compile` construction arguments and strict compile failure.
7. Exactly one unmeasured D0/D8 prewarm per worker and environment reset before
   episode timing.
8. Acceleration metadata consistency during worker result merge.
9. Existing eager RTC, overlap, CD, and FastWAM tests remaining green.

A real-GPU smoke check runs one task before the 40-task benchmark and requires
successful D0/D8 prewarm, active compilation, identical eager/accelerated
action output within BF16 tolerance for a fixed input, and no eager fallback.

## Scope constraints

- Do not add another file under `starwam/inference/`.
- Do not create a separate accelerated WAM or MoT model variant.
- Do not modify checkpoint weights or create converted checkpoint files.
- Do not change eager RTC, Joint CD, or standard FastWAM behavior.
- Do not create Git commits; the user will inspect and commit manually.
