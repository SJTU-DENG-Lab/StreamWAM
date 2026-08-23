# RTC-AC 40 ms Performance Parity Design

## Objective

Make StreamWAM's opt-in accelerated RTC-AC path reproduce the established wyx
Stage-2 selfatt-z1 steady-state inference latency while preserving StreamWAM's
existing architecture and all non-accelerated behavior.

The performance target is measured after excluding the first asynchronous D8
call in a process:

- steady-state D8 mean: 40-45 ms per model inference chunk;
- D8 p50: at most 45 ms;
- no correctness regression on the FastWAM step-5500 checkpoint;
- no behavior change to eager RTC-AC, Joint CD, standard FastWAM joint, or
  ordinary StreamWAM Euler inference.

The wyx reference reproduced this target on GPU 0 with PyTorch 2.7.1+cu128 and
Triton 3.3.1. Its measured D8 calls were 114.14, 45.38, 40.52, 41.86, and
34.63 ms; excluding the first background-thread D8 gives a 40.60 ms mean.

## Scope

Acceleration remains opt-in through `--rtc-ac-accelerated`. The optimized
path is intentionally checkpoint- and geometry-specific:

- FastWAM Stage-2 selfatt-z1 step-5500 architecture;
- Wan2.2 5B video expert;
- batch size 1;
- BF16 model inputs and weights;
- action horizon 32;
- stride 16;
- asynchronous delay 8 and launch after 8 executed actions;
- nine input video frames, producing three latent video frames;
- one consistency inference step;
- 16 condition slots.

Requests outside this contract fail before compilation with a precise error.
The eager RTC-AC path retains its existing behavior and is not made dependent
on the accelerated contract.

## Architecture

### Reference-shaped compiled core

`streamwam/modules/rtc_ac.py` keeps the existing eager `forward_rtc_ac` method.
Only the accelerated method is replaced with a reference-shaped fixed-geometry
core whose inputs and block execution order mirror wyx
`Stage2ThreeStreamMoT1StepSelfAttZ1Accelerated.forward_stage2`:

1. Receive separate `tokens_all`, `freqs_all`, `context_all`, and `t_mod_all`
   mappings for video, action, and condition streams.
2. For each MoT layer, construct expert Q/K/V and explicit post-block state in
   video, action, condition order.
3. Resolve cached static cross-attention K/V plus the dynamic proprio token.
4. Execute the four directed attentions using the same masks and slices as the
   wyx z1 core.
5. Apply post-attention, cross-attention, and FFN work in the same stream order.
6. Return only video and action tokens.

The implementation uses StreamWAM modules and parameter ownership; it does not
import runtime code from the wyx repository. State-dict names and checkpoint
loading remain unchanged.

### Inference entry

`RTCACWAM.infer_action` uses `torch.inference_mode()` rather than
`torch.no_grad()`, matching the wyx public accelerated inference entry. It
continues to prepare VAE latents, noise, schedules, fixed masks, prompt cache,
proprio token, and consistency boundaries in the existing StreamWAM WAM layer.

For accelerated calls, `rtc_ac_wam.py` converts the prepared states into the
reference-shaped compiled payload. Eager calls continue using the existing
`expert_states` payload and `forward_rtc_ac` method.

### Compilation and caches

`RTCACAccelerationRuntime` continues to own compilation and inference-only
caches. Its compilation contract stays identical to wyx:

```python
torch.compile(
    mot.forward_rtc_ac_accelerated,
    mode="reduce-overhead",
    fullgraph=True,
    dynamic=False,
)
```

Prompt projection, cross-attention K/V, attention masks, and scheduler tensors
remain cached. Task changes refresh reusable static tensors in place so they do
not create new input addresses or compile variants.

The D0 and D8 prewarm calls remain outside evaluation timing. The first actual
background-thread D8 call is reported separately as a thread warmup sample and
is excluded only from the new steady-state diagnostic metric; the existing raw
average remains available and is not silently redefined.

## Acceleration diagnostics

The final `rtc_ac_acceleration` result records evidence rather than a requested
configuration claim:

- compile active;
- compile mode, fullgraph, and dynamic settings;
- Dynamo unique graph and recompile counters when available;
- Inductor CUDA Graph skip count when available;
- D0 and D8 prewarm completion;
- Python, PyTorch, Triton, CUDA, and GPU identity;
- asynchronous D8 total samples;
- first-background-D8 latency;
- steady-state D8 mean, p50, p90, and sample count.

Counter fields are nullable when a supported PyTorch API does not expose them.
Missing counters do not imply successful CUDA Graph capture. Any positive
CUDA Graph skip or recompile count is printed prominently in the final summary.

No per-chunk diagnostic is printed during evaluation.

## Validation

### Automated tests

Tests must establish the following before production code changes:

1. Accelerated inference executes under inference mode.
2. Eager RTC-AC remains callable and preserves its existing payload.
3. Accelerated calls reject non-BF16, batch size other than one, non-Wan2.2,
   and geometry outside the fixed step-5500 contract before compilation.
4. The accelerated MoT accepts the reference-shaped three-stream payload and
   returns video/action tokens.
5. Runtime status exposes compiler counters without failing when counters are
   unavailable.
6. Timing aggregation retains raw D8 averages and additionally excludes
   exactly the first asynchronous D8 sample from steady-state statistics.
7. Existing checkpoint, consistency, RTC controller, LIBERO timing, and CLI
   regression tests remain green.

### GPU acceptance run

The acceptance run uses the same environment that reproduced wyx:

```text
Python 3.10.20
PyTorch 2.7.1+cu128
Triton 3.3.1
CUDA 12.8
GPU 0
```

First run one LIBERO task/trial and inspect the raw and steady-state D8 metrics.
If steady-state mean remains above 50 ms, use compiler counters and a gated
stage profiler to locate the remaining time before changing more code. Do not
declare parity from compile activation alone.

After the short run reaches the target, run the existing four-GPU, one-trial,
40-task accelerated evaluation and verify success rate, deadline misses, chunk
latency, and episode wall time.

## Failure Handling

- Unsupported accelerated geometry raises before the first compiled call.
- Compilation failure remains fatal for accelerated mode; it never silently
  falls back to eager inference.
- A CUDA Graph skip or recompile does not abort evaluation, but is persisted in
  results and highlighted in the final summary.
- Diagnostic collection must never alter model tensors, RNG state, execution
  order, or synchronization boundaries.

## Non-goals

- Generalizing accelerated RTC-AC to arbitrary backbones or checkpoint shapes.
- Compiling the VAE or the entire rollout loop.
- Adding another inference module or a second RTC-AC implementation.
- Changing asynchronous scheduling, action execution, normalization, or RTC
  correctness semantics.
- Modifying timing definitions to make reported latency appear lower.
