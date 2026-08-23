# RTC-AC Overlap Timing Implementation Plan

> Implement inline without Git commits; the user will review and commit manually.

**Goal:** Add final-only metrics that directly quantify how much D8 model inference overlaps action execution, while preserving RTC-AC scheduling and all non-RTC output.

**Architecture:** Capture exact synchronized model-inference start/end timestamps in the existing prediction path. The RTC-AC controller combines those timestamps with D8 launch, action-window boundary, swap, and episode-end timestamps to produce immutable overlap records. The existing global LIBERO timing collector aggregates those records and emits one optional RTC-only JSON object and one final-only text block.

**Tech Stack:** Python, `concurrent.futures`, `time.perf_counter_ns`, pytest.

---

### Task 1: Lock the overlap math with failing tests

**Files:**
- Modify: `tests/test_rtc_ac.py`
- Modify: `tests/test_libero_timing.py`

Add deterministic timestamp tests for an on-time D8 prediction, a deadline miss, and an episode-end drain. Add aggregation tests for hidden ratio, boundary wait, effective critical-path time, JSON shape, RTC-only formatting, and non-RTC regression behavior. Run the focused tests and confirm they fail because the new API is absent.

### Task 2: Add controller event records

**Files:**
- Modify: `streamwam/inference/rtc_ac.py`
- Modify: `streamwam/inference/__init__.py`

Extend `RTCACPrediction` with optional synchronized inference timestamps. Add an immutable D8 overlap record and a pure timestamp-to-record helper. Record D8 launch/completion/boundary/swap or episode-end events in `RTCACController`, expose a drain-once record API, and retain deterministic executor cleanup and exception behavior.

### Task 3: Aggregate and format final-only RTC metrics

**Files:**
- Modify: `examples/libero/timing.py`

Allow `GlobalTimingSummary` to opt into RTC mode and accept overlap records. Add an optional `rtc_ac_overlap` object to `as_dict`, calculate exact aggregate counts/averages plus effective per-chunk critical-path time, and append exactly one RTC section from `format_summary`. Leave normal Euler/CD dictionaries and text unchanged.

### Task 4: Wire exact timestamps through LIBERO rollout

**Files:**
- Modify: `examples/libero/rollout.py`

Return exact inference start/end timestamps from `_predict_action_chunk`, attach them to each RTC prediction, enable RTC aggregation in the RTC episode path, and drain controller overlap records after swaps and during final cleanup. Keep all timing output at command finalization.

### Task 5: Verify behavior and review

Run focused tests, the complete relevant test suite, syntax checks, `bash -n` for launchers, and `git diff --check`. Review the diff for non-RTC behavior changes and request an independent read-only code review. Do not create a Git commit.
