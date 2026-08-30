# Align LIBERO and RoboTwin Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LIBERO and RoboTwin use the same Chunk Time selection, episode timer boundary, and all-completed-episode Total Time aggregation.

**Architecture:** Keep raw timing diagnostics intact and expose canonical public fields. `chunk_time_ms` selects all non-warmup D8 calls for AC Streaming and synchronized inference calls otherwise; Total Time directly averages every completed episode in both evaluators. LIBERO stabilization moves into a shared pre-timing phase used by synchronous and AC Streaming rollouts.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest

## Global Constraints

- A zero-D8 AC run reports no Chunk Time instead of falling back to D0.
- Public Total Time includes successful and failed completed episodes.
- Global Total Time is a direct episode average, matching LIBERO's workload sum
  divided by completed trial count.
- Compilation, prewarming, reset, dummy stabilization, video encoding, and post-terminal asynchronous cleanup remain outside Total Time.
- Initial D0 is excluded from AC Streaming Chunk Time; every recorded non-warmup D8 remains included.
- Existing detailed timing and overlap fields remain available.

---

### Task 1: Canonical LIBERO Chunk Time

**Files:**
- Modify: `examples/libero/timing.py:131-185`
- Modify: `examples/libero/multigpu_rollout.py:385-461`
- Test: `tests/test_libero_timing.py`

**Interfaces:**
- Consumes: `GlobalTimingSummary.chunks` and `GlobalTimingSummary.ac_stream_overlap_records`
- Produces: `timing_summary["chunk_time_ms"]: float | None`

- [ ] **Step 1: Write failing single-worker tests**

Add tests proving synchronous `chunk_time_ms` equals the synchronized inference mean and AC Streaming `chunk_time_ms` equals the mean of all recorded D8 overlap samples, rather than the mixed D0/D8 chunk mean.

```python
assert sync_summary["chunk_time_ms"] == 15.0
assert ac_summary["average_inference_ms_per_chunk"] == 150.0
assert ac_summary["chunk_time_ms"] == 60.0
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_libero_timing.py -k 'canonical_chunk_time'`

Expected: FAIL because `chunk_time_ms` does not exist.

- [ ] **Step 3: Implement the single-worker field**

In `GlobalTimingSummary.as_dict`, retain `average_inference_ms_per_chunk` as the raw all-chunk diagnostic. Set `chunk_time_ms` to that mean for non-AC runs. For AC runs, compute the overlap summary once and use `average_inference_wall_ms` when at least one D8 record exists.

- [ ] **Step 4: Add a failing multi-GPU merge test**

Create two minimal worker result JSON fixtures with different D8 counts and assert the merged canonical value is the sample-weighted mean of every D8 sample, not an equal worker mean and not the mixed inference mean.

```python
assert merged["timing_summary"]["chunk_time_ms"] == pytest.approx(50.0)
assert merged["timing_summary"]["readme_aligned"]["chunk_time_ms_mean"] == pytest.approx(50.0)
```

- [ ] **Step 5: Run the merge test and verify RED**

Run: `pytest -q tests/test_libero_timing.py -k 'multigpu_uses_all_d8'`

Expected: FAIL because the merged summary has no canonical field and README output uses the mixed mean.

- [ ] **Step 6: Implement multi-GPU selection and rendering**

Set merged `chunk_time_ms` from `ac_stream_inference_wall_sum_ms / ac_stream_async_count` for AC runs and fall back to `average_inference_ms_per_chunk` otherwise. Make the CLI `Chunk Time` and `readme_aligned.chunk_time_ms_mean` consume this canonical field.

- [ ] **Step 7: Run Task 1 tests**

Run: `pytest -q tests/test_libero_timing.py`

Expected: PASS.

### Task 2: Exclude LIBERO Stabilization from Total Time

**Files:**
- Modify: `examples/libero/rollout.py:627-721`
- Modify: `examples/libero/rollout.py:781-917`
- Test: `tests/test_libero_timing.py`

**Interfaces:**
- Consumes: `env`, `initial_state`, and `num_steps_wait`
- Produces: observation after reset and dummy stabilization; episode wall time beginning immediately afterward

- [ ] **Step 1: Write failing rollout boundary tests**

Use a deterministic fake clock and environment whose dummy steps advance 10 ms and policy action advances 100 ms. Exercise the real synchronous rollout and AC Streaming rollout with model inference/controller boundaries replaced only at the external model layer.

```python
assert timing.episode_wall_ms == [100.0]
```

- [ ] **Step 2: Run boundary tests and verify RED**

Run: `pytest -q tests/test_libero_timing.py -k 'excludes_stabilization'`

Expected: FAIL with an episode wall value that includes dummy-step time.

- [ ] **Step 3: Implement a shared stabilization phase**

Add a small `_reset_and_stabilize` helper that resets the environment, restores the initial state, executes up to `num_steps_wait` dummy actions, and returns the latest observation. Call it before `episode_start_ns` in both rollout modes, then iterate only over policy-control steps.

- [ ] **Step 4: Run boundary tests and verify GREEN**

Run: `pytest -q tests/test_libero_timing.py -k 'excludes_stabilization'`

Expected: PASS for synchronous and AC Streaming rollouts.

- [ ] **Step 5: Run focused timing regression tests**

Run: `pytest -q tests/test_libero_timing.py tests/test_robotwin_inference_modes.py`

Expected: PASS with RoboTwin behavior unchanged.

### Task 3: All-Episode Total Time

**Files:**
- Modify: `examples/libero/multigpu_rollout.py`
- Modify: `examples/robotwin/timing.py`
- Test: `tests/test_libero_timing.py`
- Test: `tests/test_robotwin_inference_modes.py`

**Interfaces:**
- Consumes: completed LIBERO episode records and RoboTwin `record_type=episode` records
- Produces: direct all-episode Total Time globally and in public breakdowns

- [ ] **Step 1: Write failing aggregation tests**

Use successful and failed episodes with intentionally different durations.
Assert that both evaluators include every duration and that RoboTwin global,
setting, task, and config summaries agree on the all-episode input set.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_libero_timing.py tests/test_robotwin_inference_modes.py -k 'all_completed'`

Expected: FAIL because LIBERO's public Long/Short fields and RoboTwin's expanded
aggregator filter failed episodes.

- [ ] **Step 3: Implement the minimal aggregation change**

Replace successful-only timing collections with all completed episode timing
collections. Emit `total_time_s` and `timed_episodes` in breakdowns and retain a
separate `successful_episode_time_s` diagnostic only at the global level.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_libero_timing.py tests/test_robotwin_inference_modes.py -k 'all_completed'`

Expected: PASS.

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `examples/libero/LIBERO.md:686-705`
- Modify: `examples/robotwin/RoboTwin.md`

**Interfaces:**
- Consumes: canonical behavior implemented by Tasks 1 and 2
- Produces: benchmark documentation matching the emitted metrics

- [ ] **Step 1: Update the protocol description**

Document that public AC Streaming Chunk Time is the mean of all non-warmup D8 calls, while first-background and steady-state values are diagnostics. State that reset and dummy stabilization are outside episode Total Time and every completed episode contributes regardless of success.

- [ ] **Step 2: Run format and targeted checks**

Run: `git diff --check`

Run: `pytest -q tests/test_libero_timing.py tests/test_robotwin_inference_modes.py tests/test_ac_stream.py tests/test_ac_stream_acceleration.py`

Expected: all tests pass and `git diff --check` is clean.

- [ ] **Step 3: Review the final diff**

Confirm both evaluators use all completed episodes for Total Time and that the diff does not alter rollout actions or success evaluation.

- [ ] **Step 4: Commit**

```bash
git add examples/libero/rollout.py examples/libero/timing.py examples/libero/multigpu_rollout.py examples/libero/LIBERO.md tests/test_libero_timing.py docs/superpowers/specs/2026-08-30-align-libero-robotwin-timing-design.md docs/superpowers/plans/2026-08-30-align-libero-robotwin-timing.md
git commit -m "fix(libero): align timing protocol with RoboTwin"
```
