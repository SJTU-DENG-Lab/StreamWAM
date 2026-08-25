# AC-Stream Runtime Rename Implementation Plan

> **For agentic workers:** Implement inline with test-driven development; do not create Git commits.

**Goal:** Rename the complete active RTC-AC runtime and public interface to AC-Stream without changing inference math or checkpoint bytes.

**Architecture:** Perform one coherent symbol/module/config/schema migration. Public strings use `ac-stream`; Python identifiers and filenames use `ac_stream`; checkpoint tensor names remain untouched.

**Tech Stack:** Python, PyTorch, argparse, YAML, pytest, Bash, Hugging Face Hub.

## Global Constraints

- No compatibility alias for the old public name.
- No checkpoint conversion or model duplication.
- No behavior changes to Joint CD, Euler, or standard FastWAM.
- No Git commit.

### Task 1: Lock the new public contract with failing tests

**Files:** Modify `tests/test_rtc_ac.py`, `tests/test_rtc_ac_acceleration.py`, `tests/test_libero_multigpu_manager.py`, and `tests/test_libero_timing.py` before production files.

- [ ] Rename expected imports and public strings to `ac_stream`/`ac-stream`.
- [ ] Assert the parser accepts `--sampling-method ac-stream` and `--ac-stream-accelerated`.
- [ ] Assert merged JSON uses `ac_stream_overlap` and `ac_stream_acceleration`.
- [ ] Run the focused tests and confirm failure on missing modules/new CLI.

### Task 2: Rename model and inference modules

**Files:** Rename and modify `streamwam/inference/rtc_ac.py`, `streamwam/modules/rtc_ac.py`, `streamwam/wam/rtc_ac_wam.py`, package exports, builder, config, and checkpoint loader.

- [ ] Rename files to `ac_stream.py`/`ac_stream_wam.py`.
- [ ] Rename Python symbols and active user-facing messages.
- [ ] Preserve state-dict attribute names required by raw checkpoint loading.
- [ ] Run focused model/checkpoint tests to green.

### Task 3: Rename LIBERO frontend and result schema

**Files:** Modify `examples/libero/rollout.py`, `multigpu_rollout.py`, `timing.py`, recipes, launchers, and related tests.

- [ ] Rename CLI, config fields, runtime routing, metrics, and final summaries.
- [ ] Rename recipe and launcher filenames.
- [ ] Run LIBERO timing/manager/CLI tests to green.

### Task 4: Rename active documentation and publish model card

**Files:** Modify `README.md`, `examples/libero/LIBERO.md`, and `docs/huggingface/README.md`.

- [ ] Replace public RTC-AC names and all active example commands.
- [ ] Keep `ac-stream/` Hugging Face paths.
- [ ] Upload the verified model card to `SJTU-DENG-Lab/StreamWAM`.

### Task 5: Final verification

- [ ] Run focused tests and full pytest.
- [ ] Run Bash syntax checks for renamed launchers.
- [ ] Scan active code/docs for forbidden old names.
- [ ] Run `git diff --check` and verify remote README.
