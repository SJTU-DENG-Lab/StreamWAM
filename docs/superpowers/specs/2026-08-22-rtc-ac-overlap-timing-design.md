# RTC-AC Overlap Timing Design

## Goal

Make the final LIBERO timing summary show whether RTC-AC inference actually ran while actions were executing, how much inference time was hidden, and whether the controller waited at a chunk boundary. Keep the existing rule that timing is printed exactly once after the complete evaluation command.

## Scope

This change adds observability only. It does not change RTC-AC sampling, D0/D8 geometry, action scheduling, checkpoint loading, success evaluation, or synchronous CD behavior.

## Event Model

For every steady-state D8 inference, the controller records monotonic timestamps:

- `launch`: the prediction Future is submitted at the configured action cursor.
- `inference_start` / `inference_complete`: the exact synchronized model call
  interval inside the background prediction callback.
- `prediction_complete`: the full background callback, including communication,
  returns and is ready to install.
- `boundary`: the current 16-action execution window reaches its swap boundary.
- `swap`: the new prediction is installed. This remains an internal scheduling
  event and is not used as model-wait time because the main-loop gap between
  `boundary` and `swap` can contain rendering and Python overhead.
- `episode_end`: used instead of `boundary` when an outstanding prediction is drained before its action window reaches the boundary.

D0 is excluded from asynchronous overlap statistics because no actions are available to overlap its initial blocking inference.

## Per-D8 Metrics

- `inference_wall_ms = inference_complete - inference_start`
- `action_overlap_ms` is the sum of intersections between the model-inference
  interval and the measured `env.step(action)` intervals before `boundary` (or
  `episode_end`). Small Python/rendering gaps are therefore not called overlap.
- `boundary_wait_ms = max(0, prediction_complete - boundary)`
- `hidden_inference_ratio = action_overlap_ms / inference_wall_ms`, clamped to `[0, 1]`
- `ready_before_boundary = prediction_complete <= boundary`
- `deadline_miss = not ready_before_boundary`

Because RTC-AC requires `block_on_miss=true`, every deadline miss produces measurable boundary waiting instead of allowing the clean-prefix alignment to drift.

## Aggregation

`GlobalTimingSummary` receives one RTC-AC async record per completed D8 prediction, including a prediction drained at episode end. Across all tasks and trials it reports:

- asynchronous D8 inference count;
- ready-before-boundary count and percentage;
- average inference wall time;
- average action overlap time;
- average boundary wait time;
- average hidden-inference percentage;
- deadline miss count and percentage.

The existing arithmetic `average_total_ms_per_chunk` remains for backward compatibility and is explicitly labelled arithmetic because it double-counts overlapped inference.

RTC-AC additionally reports an effective critical-path estimate over all generated chunks:

`effective_total_ms = arithmetic_total_ms - sum(D8 action_overlap_ms)`

The effective per-chunk average divides this total by the existing generated-chunk count. It subtracts only measured D8 inference overlap. D0 blocking inference, communication, unhidden inference represented by boundary wait, and action execution therefore remain in the estimate.

## Output

The normal Euler/CD summary remains unchanged except for clarifying the arithmetic total label if needed. RTC-AC appends one final-only section:

```text
========== RTC-AC Async Overlap ==========
async D8 inferences             : 4
ready before chunk boundary     : 4/4 (100.00%)
average inference wall time     : 288.71 ms
average action overlap time     : 288.71 ms
average boundary wait time      : 0.00 ms
average hidden inference ratio  : 100.00%
deadline misses                 : 0/4 (0.00%)
average effective time/chunk    : 2265.67 ms
inference hidden by actions     : 288.71 ms/chunk
==========================================
```

No per-task, per-trial, or per-chunk timing lines are printed.

## Persistence

The single top-level `timing_summary` in `results.json` gains an optional `rtc_ac_overlap` object. It is present only for RTC-AC runs. Existing keys remain unchanged so the multi-GPU result reader remains compatible.

## Failure and Episode-End Handling

- A failed background prediction propagates through the existing controller error path and does not create a successful overlap record.
- A prediction completed and drained at episode end is counted as an inference. Its overlap ends at `min(inference_complete, episode_end)`; it is marked `episode_end_before_boundary` and excluded from boundary-ready and boundary-wait denominators.
- An installed D8 prediction is recorded exactly once; draining the controller cannot duplicate it.
- Empty RTC runs produce zero counts and zero-valued averages without division errors.

## Tests

- deterministic controller test where inference completes before boundary: 100% hidden, zero wait;
- delayed predictor test where boundary waits: partial/no hidden ratio as derived from timestamps and positive wait;
- episode-end drain test: inference counted once, boundary metrics excluded;
- predictor exception test: no overlap record and executor cleanup remains deterministic;
- aggregation/formatting test: exactly one final section and stable JSON shape;
- regression tests: Euler and synchronous consistency summaries do not gain RTC-only output.
