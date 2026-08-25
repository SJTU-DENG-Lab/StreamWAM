# RoboTwin Per-Job Supervision Implementation Plan

> **For agentic workers:** Implement inline with test-first development. Do not create Git commits; the user performs manual review and commit.

**Goal:** Prevent one hung RoboTwin task from blocking a GPU or leaving renderer processes while preserving the established success and timing definitions.

**Architecture:** Keep one inference server resident per GPU. Run each task/config/trial in a fresh simulator process group, dynamically schedule pending jobs, atomically persist every result, and let the parent manager enforce a hard watchdog.

**Tech Stack:** Python 3.10, subprocess process groups, JSON sidecars, pytest.

## Global Constraints

- Never reload the model between jobs.
- Never retry a timed-out job automatically.
- Exclude timed-out jobs from accuracy and timing denominators.
- Preserve model chunk and episode timing boundaries.
- Preserve all LIBERO behavior.
- Do not create Git commits.

### Task 1: Single-Job Runner

**Files:**
- Modify: `examples/robotwin/robotwin_worker.py`
- Test: `tests/test_robotwin_job_runner.py`

- [ ] Add failing tests for exactly one job, atomic output, phase sidecar, prewarm flag, and terminal result schema.
- [ ] Add a one-job CLI contract and write status before environment setup.
- [ ] Persist result/timing atomically before process exit.
- [ ] Verify focused runner tests.

### Task 2: Dynamic Supervisor and Watchdog

**Files:**
- Modify: `examples/robotwin/multigpu_rollout.py`
- Test: `tests/test_robotwin_multigpu_manager.py`

- [ ] Add failing tests for dynamic refill, timeout skip without retry, process-group termination, and continued scheduling.
- [ ] Add `--job-timeout-seconds` with default 1200.
- [ ] Launch each simulator with a new session/process group.
- [ ] Poll active jobs and atomically record `skipped_timeout` after termination.
- [ ] Verify focused manager tests.

### Task 3: Resume and Honest Aggregation

**Files:**
- Modify: `examples/robotwin/multigpu_rollout.py`
- Modify: `examples/robotwin/timing.py`
- Test: `tests/test_robotwin_timing.py`
- Test: `tests/test_robotwin_multigpu_manager.py`

- [ ] Add failing tests that completed/skipped identities resume without rerun.
- [ ] Add failing tests for completed-only accuracy/timing and `INCOMPLETE` output.
- [ ] Merge per-job sidecars and return nonzero when skips or infrastructure errors exist.
- [ ] Run the complete test suite, syntax checks, and `git diff --check`.
