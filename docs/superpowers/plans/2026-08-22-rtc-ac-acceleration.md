# RTC-AC Optional Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict opt-in wyx-equivalent compilation, CUDA Graph Tree, context/KV/mask/schedule caching, and unmeasured D0/D8 prewarm to the existing RTC-AC path.

**Architecture:** Keep one `RTCACWAM`, one `RTCACMoT`, and the existing `starwam/inference/rtc_ac.py`. Add a small acceleration runtime there, a functional compile-safe Wan block API, an accelerated method on the existing MoT, and CLI/prewarm wiring in the existing LIBERO frontend.

**Tech Stack:** Python 3.10, PyTorch 2.9, TorchInductor/Triton, CUDA, pytest, Bash

## Global Constraints

- Public opt-in is exactly `--rtc-ac-accelerated`.
- Acceleration requires CUDA and strict `torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)` success.
- D0 and D8 prewarm occur before episode timing and are never included in chunk/overlap/workload metrics.
- Absence of the flag preserves eager RTC behavior.
- Do not add another file under `starwam/inference/` or another WAM/MoT variant.
- Do not create Git commits.

---

### Task 1: CLI contract and forwarding

**Files:**
- Modify: `examples/libero/rollout.py`
- Modify: `examples/libero/multigpu_rollout.py`
- Modify: `examples/libero/scripts/launch_starwam_libero_rtc_ac_rollout.sh`
- Modify: `examples/libero/scripts/launch_starwam_libero_rtc_ac_4gpu.sh`
- Test: `tests/test_inference_checkpoint_cli.py`
- Test: `tests/test_libero_multigpu_manager.py`

**Interfaces:**
- Consumes: existing RTC `sampling_method=rtc_ac` CLI
- Produces: `args.rtc_ac_accelerated: bool` forwarded to every worker

- [ ] Add failing tests that parse the flag, reject it outside RTC/CUDA, verify worker forwarding, and run the launcher through a recording Python executable.
- [ ] Run the focused tests and verify failures are caused by the missing flag.
- [ ] Add the boolean option, validation, worker forwarding, and `"$@"` launcher forwarding.
- [ ] Run the focused tests and `bash -n` until green.

### Task 2: Compile-safe Wan block operations

**Files:**
- Modify: `starwam/modules/wan_block.py`
- Test: `tests/test_rtc_ac_acceleration.py`

**Interfaces:**
- Produces: `DiTBlock.get_qkv_functional(...) -> (q, k, v, state)` and `DiTBlock.post_attention_with_kv(state, attention_output, cross_key, cross_value, context_mask) -> Tensor`

- [ ] Add a failing numerical test comparing the functional path with existing `get_qkv`/`post_attention` on fixed CPU tensors.
- [ ] Verify the test fails because the functional methods are absent.
- [ ] Implement explicit AdaLN state and cached-K/V cross-attention without module mutation.
- [ ] Verify BF16/FP32 numerical equivalence and existing module tests.

### Task 3: Existing RTC MoT accelerated core

**Files:**
- Modify: `starwam/modules/rtc_ac.py`
- Test: `tests/test_rtc_ac_acceleration.py`

**Interfaces:**
- Consumes: functional Wan block operations and per-layer static video/action K/V tuples
- Produces: `RTCACMoT.forward_rtc_ac_accelerated(...) -> {"video": Tensor, "action": Tensor}`

- [ ] Add failing D0 and D8 tests comparing eager and accelerated core outputs on a tiny real RTC MoT.
- [ ] Implement the same directed policy/condition/z1 attention graph using functional block state; compute only dynamic proprio K/V and reuse ActionDiT K/V for action/condition.
- [ ] Verify fixed-seed numerical equivalence for D0 and D8.

### Task 4: Acceleration runtime and RTCACWAM integration

**Files:**
- Modify: `starwam/inference/rtc_ac.py`
- Modify: `starwam/backbone/wan22.py`
- Modify: `starwam/modules/action_dit.py`
- Modify: `starwam/wam/rtc_ac_wam.py`
- Test: `tests/test_rtc_ac_acceleration.py`

**Interfaces:**
- Produces: `RTCACAccelerationRuntime`, `RTCACWAM.enable_rtc_ac_acceleration()`, `RTCACWAM.rtc_ac_acceleration_status()`

- [ ] Add failing tests for mask/schedule identity reuse, stable-address task-cache refresh, dynamic proprio K/V, exact compile kwargs, and strict compile errors.
- [ ] Add optional already-projected context inputs to video/action `pre_dit` while preserving defaults.
- [ ] Implement runtime caches, compiled MoT dispatch, task-key refresh, schedule reuse, phase status, and strict error propagation.
- [ ] Integrate the runtime into the existing shared RTC noise/boundary path.
- [ ] Run acceleration and eager RTC regression tests.

### Task 5: Unmeasured D0/D8 prewarm and reporting

**Files:**
- Modify: `examples/libero/rollout.py`
- Modify: `examples/libero/multigpu_rollout.py`
- Modify: `examples/libero/timing.py`
- Test: `tests/test_libero_timing.py`
- Test: `tests/test_libero_multigpu_manager.py`

**Interfaces:**
- Produces: one per-worker D0/D8 prewarm before the first episode and one merged `rtc_ac_acceleration` status block

- [ ] Add failing tests proving prewarm occurs once, is excluded from timing, resets the environment, and worker status mismatches are rejected.
- [ ] Add per-worker GPU context memory caching, prewarm orchestration, final-only status output, JSON metadata, and merge validation.
- [ ] Run focused rollout/timing/manager tests.

### Task 6: Full verification

**Files:**
- Verify all files above; no new production files

- [ ] Run all `tests/` in the LIBERO environment.
- [ ] Run `py_compile`, `bash -n`, and `git diff --check`.
- [ ] Run a one-task real-GPU accelerated smoke evaluation and verify strict D0/D8 prewarm plus active compile status.
- [ ] Compare the one-task accelerated output with eager under fixed seed and BF16 tolerance before handing off the 40-task command.
