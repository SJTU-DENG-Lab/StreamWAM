# Checkpointing Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the oversized `streamwam/utils/checkpoint.py` module with a focused `streamwam/checkpointing/` package without changing checkpoint behavior.

**Architecture:** `core.py` owns the pre-existing generic checkpoint helpers, `streamwam_format.py` owns native inference loading, `fastwam_format.py` owns FastWAM validation/loading/stats/config adaptation, and `loader.py` is the only format dispatcher. The package `__init__.py` exposes the public API; the old utils module is deleted and every repository import is migrated.

**Tech Stack:** Python, PyTorch, pytest

## Global Constraints

- Delete `streamwam/utils/checkpoint.py`; do not retain a compatibility shim.
- Preserve all runtime behavior and public function signatures.
- Keep all changes unstaged and uncommitted for manual review.
- Do not write converted checkpoint files.

---

### Task 1: Lock the new package boundary with a failing import test

**Files:**
- Modify: `tests/test_checkpoint_formats.py`
- Create: `streamwam/checkpointing/__init__.py`

**Interfaces:**
- Produces: public imports `load_inference_checkpoint`, `load_inference_stats`, and `prepare_inference_config` from `streamwam.checkpointing`.

- [ ] Add a test importing the inference APIs from `streamwam.checkpointing` and exercising unknown-format rejection.
- [ ] Run the focused test and verify collection fails because the package does not exist.
- [ ] Create the package public interface during Tasks 2-4.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Move generic and native StreamWAM checkpoint responsibilities

**Files:**
- Create: `streamwam/checkpointing/core.py`
- Create: `streamwam/checkpointing/streamwam_format.py`

**Interfaces:**
- `core.py` produces `infer_backbone_info`, `save_checkpoint`, `load_checkpoint`, and `load_action_dit_backbone_init`.
- `streamwam_format.py` produces `load_streamwam_checkpoint(model, path)`.

- [ ] Move the four generic helpers unchanged into `core.py`.
- [ ] Move native inference file resolution, prefix stripping, metadata extraction, and non-strict loading into `streamwam_format.py`.
- [ ] Run existing standard-format tests and verify identical results.

### Task 3: Isolate FastWAM format behavior

**Files:**
- Create: `streamwam/checkpointing/fastwam_format.py`

**Interfaces:**
- Produces `load_fastwam_checkpoint(model, path)`, `load_fastwam_stats(path)`, and `prepare_fastwam_config(config)`.

- [ ] Move strict pre-load validation and direct mmap loading into `fastwam_format.py`.
- [ ] Move in-memory statistics mapping and config preparation into the same format adapter.
- [ ] Run all FastWAM malformed, success, and non-mutation tests.

### Task 4: Add dispatch and migrate all callers

**Files:**
- Create: `streamwam/checkpointing/loader.py`
- Create: `streamwam/checkpointing/__init__.py`
- Delete: `streamwam/utils/checkpoint.py`
- Modify: all Python files importing `streamwam.utils.checkpoint`
- Modify: `streamwam/utils/__init__.py`

**Interfaces:**
- `loader.py` produces the existing signatures `load_inference_checkpoint`, `load_inference_stats`, and `prepare_inference_config`.
- `__init__.py` re-exports all seven public checkpoint helpers.

- [ ] Implement explicit `streamwam`/`fastwam` dispatch with unchanged defaults and error messages.
- [ ] Replace every repository import with `from streamwam.checkpointing import ...`.
- [ ] Remove checkpoint exports from `streamwam.utils` and delete the old module.
- [ ] Verify `rg 'streamwam\.utils\.checkpoint'` returns no results.

### Task 5: Verify the refactor

**Files:**
- Modify: checkpoint design documentation where it still states that no package exists.

- [ ] Run the complete pytest suite.
- [ ] Run Python compile checks for `streamwam`, `examples`, and `tests`.
- [ ] Run shell syntax checks for the LIBERO and RoboTwin launchers.
- [ ] Run `git diff --check` and leave all changes uncommitted.
