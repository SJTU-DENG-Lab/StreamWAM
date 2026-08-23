# LIBERO Single-Trial CD/RTC Comparison Implementation Plan

> **For agentic workers:** Execute inline with test-driven development. Do not
> create Git commits; the user will inspect and commit manually.

**Goal:** Add directly comparable four-GPU, 40-task, one-trial launch paths for Joint CD and RTC-AC.

**Architecture:** Extend the existing worker timing payload with episode wall measurements, then make the existing multi-GPU merger combine both common timing and optional RTC overlap data. Keep launch policy in two small shell entrypoints that pass all behavior to Python CLI arguments.

**Tech Stack:** Python, argparse, JSON, subprocess workers, pytest, bash.

## Global Constraints

- Exactly 40 tasks and one trial per task by default.
- EGL, fixed seed 42, 30 wait steps, H32/replan16, one inference step.
- Final-only aggregate output; no per-action timing persistence.
- No model inference or RTC scheduling changes.
- No Git commits.

---

### Task 1: Common episode timing

**Files:**
- Modify: `examples/libero/timing.py`
- Modify: `examples/libero/rollout.py`
- Test: `tests/test_libero_timing.py`

- [ ] Add a failing test proving episode wall values aggregate independently of chunk counts.
- [ ] Add `episode_wall_ms` storage, average episode wall, and workload-sum fields.
- [ ] Capture the interval after `set_init_state` through rollout return for both CD and RTC paths.
- [ ] Run focused tests.

### Task 2: Multi-GPU RTC aggregation

**Files:**
- Modify: `examples/libero/multigpu_rollout.py`
- Test: `tests/test_libero_multigpu_manager.py`

- [ ] Add a failing parser test for `--sampling-method rtc_ac`.
- [ ] Add failing literal-fixture tests for weighted common episode timing and RTC count/average reconstruction.
- [ ] Accept `rtc_ac`, merge optional overlap payloads, and append its final summary only for RTC workers.
- [ ] Reject a mixture of RTC and non-RTC worker timing payloads.
- [ ] Run focused tests.

### Task 3: Comparable launchers

**Files:**
- Modify: `examples/libero/scripts/launch_starwam_libero_joint_cd_4gpu.sh`
- Create: `examples/libero/scripts/launch_starwam_libero_rtc_ac_4gpu.sh`

- [ ] Make the CD launcher default to one trial and EGL.
- [ ] Add the matching RTC launcher with identical suites, trial count, seed, wait, GPU selection, and video setting.
- [ ] Validate both scripts with `bash -n`.

### Task 4: Verification and review

- [ ] Run the complete test suite, Python compilation, launcher syntax checks, and `git diff --check`.
- [ ] Request a read-only review focused on aggregation denominators, one-trial contract, and CD/RTC parity.
- [ ] Report both commands and the wyx comparison baseline without creating a commit.
