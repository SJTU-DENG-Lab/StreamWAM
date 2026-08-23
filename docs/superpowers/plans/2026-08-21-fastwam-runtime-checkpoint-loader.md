# FastWAM Runtime Checkpoint Loader Implementation Plan

> **Superseded placement:** This plan records the initial implementation. The
> completed follow-up structure is defined in
> `docs/superpowers/plans/2026-08-21-checkpointing-package-refactor.md`, which
> replaces `starwam/utils/checkpoint.py` with `starwam/checkpointing/`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load original FastWAM LIBERO and RoboTwin release checkpoints and dataset statistics directly at StarWAM inference time via `--checkpoint-format fastwam`, without writing converted copies.

**Architecture:** Extend the existing `starwam/utils/checkpoint.py` as the single inference checkpoint-format dispatcher. The FastWAM branch memory-maps the payload, validates exact expert/proprio keys and shapes before mutation, and strictly loads the three shared submodules; both inference frontends reuse the same model and stats functions.

**Tech Stack:** Python 3.10+, PyTorch, pytest, argparse, JSON, existing StarWAM dataclass recipes.

## Global Constraints

- Do not create a `starwam/checkpointing/` package.
- Do not write or retain a converted checkpoint.
- `checkpoint_format="starwam"` remains the default and preserves existing inference behavior.
- FastWAM loading is explicit through `checkpoint_format="fastwam"`; do not auto-detect.
- FastWAM video, action, and proprio state must validate exactly before any target parameter is mutated.
- FastWAM stats conversion occurs only in memory.
- Training and resume checkpoint behavior is out of scope.

---

### Task 1: Shared inference checkpoint and stats loaders

**Files:**
- Modify: `starwam/utils/checkpoint.py`
- Modify: `starwam/utils/__init__.py`
- Create: `tests/test_checkpoint_formats.py`

**Interfaces:**
- Produces: `load_inference_checkpoint(model, path, checkpoint_format="starwam") -> dict`
- Produces: `load_inference_stats(path, checkpoint_format="starwam") -> dict[str, dict[str, torch.Tensor]]`
- Consumes: a `mot_wam` model exposing `mot.experts["video"]`, `mot.experts["action"]`, and `proprio_encoder`.

- [ ] **Step 1: Write failing tests for FastWAM exact loading**

Create tiny video/action/proprio modules, save a payload with `mixtures.video.*`, `mixtures.action.*`, and `proprio_encoder`, call `load_inference_checkpoint(..., checkpoint_format="fastwam")`, and assert all target parameters and returned metadata match.

- [ ] **Step 2: Write failing tests for pre-mutation validation**

Cover missing top-level groups, unknown `mixtures.other.*` keys, missing tensors, unexpected tensors, shape mismatch, wrong model family, and missing proprio encoder. Snapshot target tensors and assert shape/key validation failures leave all targets unchanged.

- [ ] **Step 3: Write failing tests for stats conversion**

Save a FastWAM stats JSON containing `action.default` and `state.default`. Assert `global_min/max/mean/std` become canonical `min/max/mean/std` tensors and malformed groups fail.

- [ ] **Step 4: Run tests and verify the new API is absent**

Run: `pytest -q tests/test_checkpoint_formats.py`

Expected: collection/import failure for missing `load_inference_checkpoint` and `load_inference_stats`.

- [ ] **Step 5: Implement the shared format dispatcher**

Add standard-file/directory loading equivalent to the current inference loaders. Add FastWAM payload parsing with `torch.load(..., map_location="cpu", weights_only=True, mmap=True)`, prefix splitting, complete key/shape validation, strict submodule loading, metadata reporting, and reference cleanup.

- [ ] **Step 6: Implement FastWAM stats canonicalization**

Map each `default` group from `global_min/max/mean/std` to `min/max/mean/std` tensors. Keep the StarWAM branch delegated to `load_lerobot_stats` via a local import to avoid import cycles.

- [ ] **Step 7: Run focused tests**

Run: `pytest -q tests/test_checkpoint_formats.py`

Expected: all tests pass.

---

### Task 2: LIBERO inference integration

**Files:**
- Modify: `examples/libero/rollout.py`
- Modify: `examples/libero/scripts/launch_starwam_libero_mot_rollout.sh`
- Modify: `examples/libero/LIBERO.md`
- Create: `tests/test_libero_checkpoint_format.py`

**Interfaces:**
- Consumes: Task 1 `load_inference_checkpoint` and `load_inference_stats`.
- Produces: LIBERO CLI flag `--checkpoint-format {starwam,fastwam}`.
- Produces: launcher environment variable `CHECKPOINT_FORMAT`, default `starwam`.

- [ ] **Step 1: Write failing source-level CLI and wiring tests**

Assert the LIBERO parser exposes `--checkpoint-format`, defaults to `starwam`, and passes the selected value to both shared checkpoint and stats loaders. Assert FastWAM selection clears `framework.action_expert_init_from` before `build_framework`.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q tests/test_libero_checkpoint_format.py`

Expected: failures because the flag and shared-loader wiring do not exist.

- [ ] **Step 3: Replace duplicated checkpoint parsing**

Remove the local state extraction/loading implementation from `rollout.py`, import the shared functions, add the CLI flag, clear ActionDiT init only for FastWAM, and route action/state stats through the selected format.

- [ ] **Step 4: Update the LIBERO launcher and documentation**

Pass `--checkpoint-format "$CHECKPOINT_FORMAT"` from the shell launcher and document the direct FastWAM release command, required Wan2.2 base assets, stats file, `num_steps_wait=30`, `replan_steps=10`, and strict validation logs.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_libero_checkpoint_format.py tests/test_checkpoint_formats.py`

Expected: all tests pass.

---

### Task 3: Shared policy and RoboTwin integration

**Files:**
- Modify: `starwam/eval/policy.py`
- Modify: `examples/robotwin/policy_server.py`
- Modify: `examples/robotwin/scripts/launch_starwam_robotwin_policy_server.sh`
- Modify: `examples/robotwin/RoboTwin.md`
- Create: `tests/test_policy_checkpoint_format.py`

**Interfaces:**
- Consumes: Task 1 shared checkpoint/stats functions.
- Produces: `StarwamPolicy(..., checkpoint_format="starwam")`.
- Produces: RoboTwin server flag `--checkpoint-format {starwam,fastwam}` and launcher `CHECKPOINT_FORMAT` environment variable.

- [ ] **Step 1: Write failing policy-format wiring tests**

Assert `StarwamPolicy` accepts the format parameter, clears ActionDiT init for FastWAM before model construction, and passes the format to model/stats loading. Assert the RoboTwin server forwards its CLI value.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `pytest -q tests/test_policy_checkpoint_format.py`

Expected: failures because the policy and server do not accept the new format.

- [ ] **Step 3: Centralize policy loading**

Remove duplicate checkpoint parsing from `starwam/eval/policy.py`, use the Task 1 loader, add `checkpoint_format`, and load both action/state stats through the shared stats function.

- [ ] **Step 4: Wire RoboTwin CLI and launcher**

Add the server flag, pass it into `StarwamPolicy`, add `CHECKPOINT_FORMAT` to the launcher, and document direct use of `robotwin_uncond_3cam_384.pt` and its z-score stats.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_policy_checkpoint_format.py tests/test_checkpoint_formats.py`

Expected: all tests pass.

---

### Task 4: Full verification and handoff

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a verified direct-loading implementation and an exact launch command.

- [ ] **Step 1: Run the complete available test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run static verification**

Run: `python -m compileall -q starwam examples tests`

Expected: exit code 0.

- [ ] **Step 3: Inspect diffs and whitespace**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; only planned implementation, tests, and docs are modified, while user-owned untracked files remain untouched.

- [ ] **Step 4: Review strictness and compatibility**

Confirm the standard format retains current behavior, FastWAM never partially loads, no converted checkpoint is written, and both LIBERO and RoboTwin use the shared functions.

- [ ] **Step 5: Report the implementation**

List each modified file, test results, environment limitations, and the final LIBERO direct FastWAM command.
