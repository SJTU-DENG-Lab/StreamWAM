# Simple LIBERO FastWAM Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one FastWAM LIBERO smoke test launch through a readable flat Python CLI with no Bash-side runtime branches.

**Architecture:** `examples/libero/rollout.py` owns CLI-to-config path mapping and pre-LIBERO rendering setup. The Bash launcher becomes one explicit `python examples/libero/rollout.py ...` command intended for manual editing.

**Tech Stack:** Python, argparse, PyTorch, LIBERO, Bash, pytest

## Global Constraints

- Preserve `--checkpoint-format fastwam` direct checkpoint loading.
- The launcher targets one suite/task/process only.
- Keep all changes unstaged and uncommitted.

---

### Task 1: Add the runtime CLI contract

- [ ] Extend parser tests with literal values for `--backbone-path`, `--stats-path`, and `--mujoco-gl`.
- [ ] Run the focused test and observe unknown-argument failure.
- [ ] Add the three arguments to `_build_arg_parser()`.
- [ ] Re-run the focused test.

### Task 2: Move runtime preparation into Python

- [ ] Add a test showing explicit backbone/stats paths override config and FastWAM placeholder output/cache paths become local paths.
- [ ] Add a test showing the selected Mujoco backend is installed in the process environment before LIBERO import.
- [ ] Implement focused config and runtime preparation helpers.
- [ ] Keep missing dataset placeholder directories out of inference cache indexing.
- [ ] Run rollout-focused tests.

### Task 3: Replace the Bash launcher

- [ ] Replace the launcher with `set -euo pipefail` plus one explicit Python command.
- [ ] Include the exact one-task/one-trial FastWAM arguments approved by the user.
- [ ] Run `bash -n` and CLI `--help`.

### Task 4: Verify

- [ ] Run the complete pytest suite and compile checks.
- [ ] Run `git diff --check`.
- [ ] Confirm no files are staged and no commit exists.
