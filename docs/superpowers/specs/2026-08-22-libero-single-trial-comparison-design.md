# LIBERO Single-Trial CD/RTC Comparison Design

## Goal

Provide two directly comparable four-GPU LIBERO evaluations: Joint CD and
RTC-AC, each covering the same 40 tasks with exactly one trial per task.

## Evaluation Contract

- Suites, in order: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Ten tasks per suite, trial ID 0 only: 40 episodes total.
- Default GPUs: `0,1,2,3`, overridable through `GPU_IDS`.
- Seed 42 with fixed diffusion noise, 30 reset wait steps, H=32, one inference
  step, and replan/stride 16.
- EGL is the default rendering backend, matching the wyx evaluation. It remains
  an explicit Python CLI argument rather than shell-side renderer logic.
- Videos are saved, matching the reference evaluation's task-duration scope.
- Joint CD uses step-3400 and `sampling_method=consistency`.
- RTC-AC uses step-5500 and `sampling_method=rtc_ac` with D0/D8 eager async.

## Timing Contract

The same aggregate timing keys are produced for CD and RTC:

- synchronized model inference per generated chunk;
- communication per generated chunk;
- action execution per generated chunk;
- arithmetic total per generated chunk;
- average episode evaluation wall time;
- aggregate evaluation workload time, defined as the sum of episode wall
  times and therefore independent of four-worker scheduling;
- four-worker command wall time, including model loading.

Episode wall time begins immediately after `env.set_init_state` returns and
ends when rollout succeeds or times out. Video encoding is reported separately
through command wall and is not allowed to distort robot trajectory time. This
matches wyx's `robot_trajectory_total_s` definition more closely than the old
command wall metric.

RTC additionally preserves the final-only overlap summary. The multi-GPU
manager reconstructs exact global sums from each worker's counts and weighted
averages. It does not retain per-action timing arrays.

## Output

Each launcher writes a new timestamped output directory containing worker
logs/results, assignments, and one merged `results.json`. The terminal prints
one final global summary. RTC prints one additional overlap block. No per-task
timing lines are added.

## Non-goals

- This does not add the wyx accelerated backend (`torch.compile`, CUDA Graph,
  KV cache, mask cache, or D0/D8 prewarm).
- It does not attempt to reproduce the 2000-episode 98.2% estimate from a
  40-episode sample.
- It does not change model math, checkpoint loading, task initialization, or
  RTC scheduling.
