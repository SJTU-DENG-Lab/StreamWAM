# RoboTwin Joint/CD Replan-32 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change RoboTwin synchronous Joint and CD evaluation from replan=24 to replan=32 without changing AC-Stream H32/s16/d8.

**Architecture:** Keep the mode-to-worker-command mapping centralized in `examples/robotwin/multigpu_rollout.py`. Exercise the real dynamic job launcher with a fake completed simulator process and inspect its generated command, so the test covers the value actually passed to RoboTwin.

**Tech Stack:** Python 3.10, pytest, argparse, subprocess-based RoboTwin evaluation manager.

## Global Constraints

- Baseline uses `replan_steps=32`.
- CD uses `replan_steps=32`.
- AC-Stream remains fixed at H32/s16/d8 and receives `replan_steps=16`.
- Timing formulas, checkpoint loading, timeout behavior, and launch scripts remain unchanged.

---

### Task 1: Update and verify the mode-to-replan mapping

**Files:**
- Modify: `tests/test_robotwin_multigpu_manager.py`
- Modify: `examples/robotwin/multigpu_rollout.py:301`

**Interfaces:**
- Consumes: `_run_job_queue(args, jobs, server_group, output_dir, *, popen=...)`
- Produces: worker commands containing `--replan-steps 32` for `baseline` and `cd`, and `--replan-steps 16` for `ac-stream`.

- [x] **Step 1: Write the failing parameterized test**

Add a pytest parameterization over `("baseline", "32")`, `("cd", "32")`, and `("ac-stream", "16")`. Run one completed fake job through `_run_job_queue`, capture the subprocess command, and assert the value immediately following `--replan-steps`.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONPATH="$PWD" /inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python -m pytest -q tests/test_robotwin_multigpu_manager.py::test_worker_replan_steps_follow_inference_mode
```

Expected: baseline and CD cases fail because the command currently contains `24`; AC-Stream passes with `16`.

- [x] **Step 3: Implement the minimal mapping change**

Replace the synchronous fallback in the worker command construction:

```python
replan_steps = 16 if args.inference_mode == "ac-stream" else 32
```

- [x] **Step 4: Run focused and complete verification**

Run the new test, all RoboTwin manager/launcher tests, `py_compile`, launcher `bash -n`, `git diff --check`, and the full pytest suite. Expected: all pass.

- [x] **Step 5: Commit the implementation**

```bash
git add examples/robotwin/multigpu_rollout.py tests/test_robotwin_multigpu_manager.py docs/superpowers/plans/2026-08-25-robotwin-replan-32.md
git commit -m "eval: use replan 32 for RoboTwin Joint and CD"
```
