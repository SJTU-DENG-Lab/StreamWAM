# LIBERO Configurable Multi-GPU Evaluation Implementation Plan

> Execute inline with test-driven development. Do not commit; the user will inspect and commit manually.

**Goal:** Evaluate arbitrary LIBERO suites/trial counts across a configurable GPU list with balanced trial workloads, one model load per GPU, isolated workers, and one merged result.

**Architecture:** `workload.py` owns deterministic partitioning and manifests. `rollout.py` gains an internal manifest worker mode while retaining normal CLI behavior. `multigpu_rollout.py` owns subprocess lifecycle and merging. A flat Bash launcher supplies the current four-GPU/40-task/one-trial command.

**Tech Stack:** Python, subprocess, JSON, PyTorch, pytest, Bash.

## Constraints

- Workload difference between workers is at most one trial.
- One worker/model instance per selected GPU.
- Worker stdout goes to per-worker logs; only manager prints the merged summary.
- Worker directories and video names cannot collide.
- No Git commit.

### Task 1: Workload partitioning

- Write failing coverage/balance/manifest tests.
- Implement suite/task/trial expansion and contiguous balanced partitioning.
- Run focused tests.

### Task 2: Manifest worker mode

- Write failing parser/manifest tests.
- Extend rollout to evaluate assignments across suites and explicit trial IDs.
- Suppress worker timing display and use suite-qualified result/video keys.
- Run focused and single-GPU regression tests.

### Task 3: Manager and merging

- Write failing worker-command and weighted-merge tests.
- Implement process launch, failure handling, result merge, one final summary, and assignments.json.
- Add the four-GPU launcher with all four suites and one trial.
- Run full tests, compile checks, Bash syntax, and `git diff --check`.
