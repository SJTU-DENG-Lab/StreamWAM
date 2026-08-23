# LIBERO Global Chunk Timing Implementation Plan

> Execute inline with test-driven development. Do not commit; the user will inspect and commit manually.

**Goal:** Save LIBERO videos at 30 FPS and print exactly one global average timing summary after all selected evaluation tasks complete.

**Architecture:** Put timing definitions and deterministic aggregation in `examples/libero/timing.py`. Keep measurement boundaries in `rollout.py`, synchronize CUDA only around communication/inference boundaries, attach action execution wall time to the active generated chunk, and serialize only aggregate statistics.

**Tech Stack:** Python, PyTorch, pytest, imageio.

## Global Constraints

- No video FPS CLI option.
- No per-chunk, per-trial, or per-task timing output or timing records.
- No model or checkpoint-loader changes.
- No Git commit.

### Task 1: Timing aggregation

- Create failing deterministic tests for chunk averages, empty summaries, and one-block formatting.
- Implement `ChunkTiming` and `GlobalTimingSummary` in `examples/libero/timing.py`.
- Run focused tests.

### Task 2: Rollout measurement and serialization

- Add failing tests for the 30 FPS save default and result summary shape.
- Change `_save_video` default to 30.
- Measure communication and synchronized inference in `_predict_action_chunk`.
- Accumulate `env.step(action)` time on the active chunk.
- Add one global summary to `results.json` and emit it once after all tasks.
- Run focused and complete tests plus `git diff --check`.
