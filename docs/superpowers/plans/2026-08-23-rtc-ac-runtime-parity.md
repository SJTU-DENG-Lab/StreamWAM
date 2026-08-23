# RTC-AC Runtime Parity Implementation Plan

> **For agentic workers:** Execute inline with strict red-green-refactor cycles.

**Goal:** Make the accelerated RTC-AC launcher use an explicitly selected wyx-compatible Python runtime and report verifiable runtime provenance.

**Architecture:** The shell launcher owns interpreter selection and preflight. The Python acceleration runtime owns version provenance. The existing multi-GPU manager continues to launch workers with `sys.executable` and aggregates the worker-reported backend status.

**Tech Stack:** Bash, Python 3.10, PyTorch, Triton, pytest.

## Global Constraints

- Do not modify either Conda/venv environment.
- Do not change RTC-AC math, checkpoint loading, task allocation or timing definitions.
- Existing launcher behavior remains the default when `PYTHON_BIN` is unset.
- Do not create a Git commit.

---

### Task 1: Interpreter-selectable launcher

**Files:**
- Modify: `examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh`
- Test: `tests/test_libero_launchers.py`

**Interfaces:**
- Consumes: optional environment variable `PYTHON_BIN`
- Produces: manager process launched by the selected executable

- [ ] Write a subprocess test using a temporary executable shim that records its argv and exits during preflight.
- [ ] Run the test and verify it fails because the launcher still invokes literal `python`.
- [ ] Add `PYTHON_BIN="${PYTHON_BIN:-python}"`, executable validation and import/version preflight, then invoke the manager with `"$PYTHON_BIN"`.
- [ ] Run the launcher test and Bash syntax checks.

### Task 2: Runtime provenance

**Files:**
- Modify: `streamwam/inference/rtc_ac.py`
- Modify: `examples/libero/multigpu_rollout.py`
- Test: `tests/test_rtc_ac_acceleration.py`
- Test: `tests/test_libero_multigpu_manager.py`

**Interfaces:**
- Produces: acceleration status field `runtime` containing executable, Python, PyTorch, Triton and CUDA versions
- Consumes: identical runtime metadata from every worker

- [ ] Write failing tests for literal runtime metadata keys and rejection of mixed worker runtime identities.
- [ ] Run the tests and confirm failures are caused by missing provenance.
- [ ] Implement runtime collection in `RTCACAccelerationRuntime.status()` and manager aggregation/printing.
- [ ] Run the focused tests.

### Task 3: Regression and real-run handoff

**Files:**
- Modify only if a regression is found in Task 1 or Task 2.

**Interfaces:**
- Consumes: completed launcher and runtime status
- Produces: verified command for the user's GPU run

- [ ] Run all tests in the `libero` environment.
- [ ] Run Python compilation, Bash syntax and `git diff --check`.
- [ ] Verify the wyx Python can import both StreamWAM rollout entrypoints.
- [ ] Provide the exact single-GPU accelerated RTC-AC command; do not create a commit.
