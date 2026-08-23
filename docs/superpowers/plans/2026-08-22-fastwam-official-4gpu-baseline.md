# FastWAM Official Four-GPU Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a four-GPU launcher for the released FastWAM 10-step Euler LIBERO baseline with directly comparable chunk and episode timing.

**Architecture:** Add one launcher that calls the existing multi-GPU manager. Test the launcher's observable command by placing a recording `python` executable first on `PATH`; do not modify inference or timing code.

**Tech Stack:** Bash, Python, pytest, existing LIBERO multi-GPU manager

## Global Constraints

- Use the released FastWAM checkpoint and statistics without conversion.
- Run all four LIBERO suites with one trial per task.
- Use 10-step Euler, replan 10, fixed seed, and EGL.
- Keep `GPU_IDS`, `BACKBONE_PATH`, and `LIBERO_HOME_PATH` configurable.
- Do not create Git commits.
- Preserve all existing working-tree changes.

---

### Task 1: Official FastWAM four-GPU launcher

**Files:**
- Create: `examples/libero/scripts/launch_streamwam_libero_fastwam_4gpu.sh`
- Modify: `tests/test_libero_multigpu_manager.py`

**Interfaces:**
- Consumes: `examples/libero/multigpu_rollout.py` CLI
- Produces: a Bash launcher configurable through `GPU_IDS`, `BACKBONE_PATH`, and `LIBERO_HOME_PATH`

- [ ] **Step 1: Write the failing launcher integration test**

Add a test that runs the not-yet-created launcher with a temporary recording
`python` executable and verifies its printed arguments include the official
FastWAM checkpoint, statistics, four suites, one trial, Euler, 10 inference
steps, replan 10, fixed seed, EGL, and environment overrides.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest -q tests/test_libero_multigpu_manager.py::test_fastwam_official_launcher_emits_baseline_protocol
```

Expected: failure because
`examples/libero/scripts/launch_streamwam_libero_fastwam_4gpu.sh` does not exist.

- [ ] **Step 3: Add the minimal launcher**

Create a launcher following the existing CD/RTC script structure. Resolve the
repository root from `BASH_SOURCE`, default GPUs to `0,1,2,3`, allow path
overrides, and invoke:

```bash
python examples/libero/multigpu_rollout.py \
  --gpus "$GPU_IDS" \
  --suites libero_spatial,libero_object,libero_goal,libero_10 \
  --num-trials 1 \
  --config examples/libero/configs/recipes/streamwam_libero_mot_wan22_5b.yaml \
  --checkpoint-format fastwam \
  --checkpoint checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --backbone-path "$BACKBONE_PATH" \
  --stats-path checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --libero-home "$LIBERO_HOME_PATH" \
  --num-steps-wait 30 \
  --replan-steps 10 \
  --num-inference-steps 10 \
  --sampling-method euler \
  --fixed-seed \
  --mujoco-gl egl \
  --save-video
```

- [ ] **Step 4: Verify the test and shell syntax**

Run:

```bash
pytest -q tests/test_libero_multigpu_manager.py::test_fastwam_official_launcher_emits_baseline_protocol
bash -n examples/libero/scripts/launch_streamwam_libero_fastwam_4gpu.sh
```

Expected: both commands succeed.

- [ ] **Step 5: Run scoped regression tests**

Run:

```bash
pytest -q tests/test_libero_multigpu_manager.py tests/test_libero_multigpu_workload.py tests/test_libero_timing.py
git diff --check
```

Expected: all tests pass and the diff check emits no errors.
