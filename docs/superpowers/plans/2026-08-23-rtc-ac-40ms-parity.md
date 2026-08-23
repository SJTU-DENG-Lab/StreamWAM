# RTC-AC 40 ms Performance Parity Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with red-green-refactor. This repository must remain uncommitted for user review.

**Goal:** Align StarWAM's opt-in accelerated RTC-AC compiled core with the wyx Stage-2 selfatt-z1 reference and expose enough evidence to validate 40-45 ms steady-state D8 latency.

**Architecture:** Preserve the existing eager `RTCACMoT.forward_rtc_ac` path. Replace only the accelerated payload/core with the reference-shaped three-stream interface, keep compilation and caches in `RTCACAccelerationRuntime`, and add final-only compiler and steady-state latency diagnostics.

**Tech Stack:** Python 3.10, PyTorch 2.7.1/2.9 compatibility, TorchDynamo/TorchInductor, pytest, LIBERO rollout manager.

## Global Constraints

- Do not create a Git commit; leave every change visible as an uncommitted diff.
- Do not change eager RTC-AC, Joint CD, standard FastWAM joint, or Euler semantics.
- Accelerated RTC-AC supports only Wan2.2 5B, BF16, batch 1, H32/s16/d8/T9, one inference step, and 16 condition slots.
- Compilation remains `mode="reduce-overhead", fullgraph=True, dynamic=False` with no eager fallback.
- Do not print per-chunk diagnostics; report aggregates once at command completion.

---

### Task 1: Reference-shaped accelerated MoT core

**Files:**
- Modify: `tests/test_rtc_ac_acceleration.py`
- Modify: `starwam/modules/rtc_ac.py`
- Modify: `starwam/wam/rtc_ac_wam.py`

**Interfaces:**
- Consumes: existing eager expert states from video/action `pre_dit` and cached K/V from `RTCACAccelerationRuntime`.
- Produces: `RTCACMoT.forward_rtc_ac_accelerated(*, tokens_all, freqs_all, context_all, t_mod_all, policy_attention_mask, condition_attention_mask, video_tokens_per_frame, action_condition_active) -> dict[str, Tensor]`.

- [ ] Add a failing test that invokes the accelerated core with separate reference-shaped mappings and compares its video/action outputs with the eager core on a tiny deterministic MoT.
- [ ] Run `pytest -q tests/test_rtc_ac_acceleration.py -k reference_shaped` and confirm failure because the current accelerated signature requires `expert_states` and explicit static K/V arguments.
- [ ] Implement the reference-shaped fixed-geometry loop in `RTCACMoT` using the wyx Q/K/V construction, attention slices, cached K/V resolution, and stream order.
- [ ] Adapt only the accelerated branch in `RTCACWAM.infer_action` to construct `tokens_all`, `freqs_all`, `context_all`, and `t_mod_all`; retain the eager payload unchanged.
- [ ] Run the focused accelerated/eager equivalence tests and then all `tests/test_rtc_ac_acceleration.py` tests.

### Task 2: Strict accelerated contract and inference mode

**Files:**
- Modify: `tests/test_rtc_ac_acceleration.py`
- Modify: `tests/test_rtc_ac.py`
- Modify: `starwam/wam/rtc_ac_wam.py`
- Modify: `starwam/inference/rtc_ac.py`

**Interfaces:**
- Produces: `validate_rtc_ac_accelerated_contract(...) -> None`, called before context preparation, VAE encode, or compilation.

- [ ] Add failing tests for non-BF16, batch size not equal to one, non-Wan2.2 expert, and existing invalid geometry.
- [ ] Add a failing test proving the public RTC-AC inference entry runs with `torch.is_inference_mode_enabled()` true.
- [ ] Run the focused tests and confirm the new accelerated cases fail while eager tests remain green.
- [ ] Implement the strict accelerated-only contract and change the RTC-AC inference decorator to `@torch.inference_mode()`.
- [ ] Run `tests/test_rtc_ac.py` and `tests/test_rtc_ac_acceleration.py` together.

### Task 3: Compiler and CUDA Graph evidence

**Files:**
- Modify: `tests/test_rtc_ac_acceleration.py`
- Modify: `starwam/inference/rtc_ac.py`
- Modify: `examples/libero/multigpu_rollout.py`

**Interfaces:**
- Produces nullable status fields `dynamo_unique_graphs`, `dynamo_recompiles`, and `inductor_cudagraph_skips`, plus GPU identity under `runtime`.

- [ ] Add failing tests that inject Dynamo/Inductor counters and assert status reports them, then clear counters and assert status safely returns null/zero-compatible values.
- [ ] Add a failing summary-format test that highlights positive CUDA Graph skips or recompiles without printing warnings for zero values.
- [ ] Implement defensive counter lookup without mutating or clearing global counters.
- [ ] Add CUDA device name/capability when CUDA is available, while keeping CPU test environments safe.
- [ ] Update the final manager summary and run focused acceleration/manager tests.

### Task 4: First-background-D8 and steady-state metrics

**Files:**
- Modify: `tests/test_libero_timing.py`
- Modify: `examples/libero/timing.py`
- Modify: `examples/libero/multigpu_rollout.py`

**Interfaces:**
- Extends `timing_summary.rtc_ac_overlap` with `first_background_d8_inference_ms`, `steady_state_d8_count`, `steady_state_d8_mean_ms`, `steady_state_d8_p50_ms`, and `steady_state_d8_p90_ms`.

- [ ] Add failing aggregation tests with D8 samples `[114.14, 45.38, 40.52, 41.86, 34.63]`, asserting the raw average remains unchanged and steady-state mean is approximately 40.60 ms after excluding exactly the first sample.
- [ ] Add failing multi-worker merge tests proving each worker excludes its own first background D8 rather than dropping only one global sample.
- [ ] Store D8 inference samples only in final aggregation state and calculate quantiles without intermediate logging.
- [ ] Merge per-worker steady-state sums/counts and samples deterministically, then print one final steady-state summary.
- [ ] Run timing and multi-GPU manager tests.

### Task 5: Regression validation and GPU handoff

**Files:**
- Modify: `examples/libero/LIBERO.md`

- [ ] Run the complete CPU test suite in the active LIBERO environment.
- [ ] Run focused acceleration tests with the wyx Python 3.10/PyTorch 2.7.1 environment.
- [ ] Run `git diff --check` and inspect only task-related diffs; do not stage or commit files.
- [ ] Document the one-GPU, one-task acceptance command using the wyx Python environment and `--rtc-ac-accelerated`.
- [ ] Hand off acceptance criteria: first background D8 reported separately, steady-state D8 mean 40-45 ms, p50 at most 45 ms, zero unexpected recompiles/graph skips, and unchanged task success.
