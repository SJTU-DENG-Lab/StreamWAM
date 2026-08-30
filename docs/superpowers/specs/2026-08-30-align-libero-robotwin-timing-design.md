# Align LIBERO and RoboTwin Timing

## Goal

Align the LIBERO and RoboTwin timing protocols except for the intentionally
different Total Time aggregation rule. RoboTwin's normal aggregation remains
unchanged; its zero-D8 AC edge case is corrected to avoid reporting D0 as
Chunk Time.

## Considered approaches

1. **Align LIBERO to RoboTwin at reporting and timer boundaries (selected).**
   Start LIBERO episode timing after its dummy stabilization steps and report
   AC Streaming Chunk Time from non-warmup D8 inference. This is the smallest
   change and preserves the existing rollout behavior and raw diagnostics.
2. Align RoboTwin to LIBERO. This would count initialization-like simulator
   work in RoboTwin and mix D0 with D8, weakening the benchmark definition.
3. Add a second compatibility report while leaving existing summaries intact.
   This avoids changing old fields but creates two competing public metrics.

## Design

LIBERO synchronous and AC Streaming rollouts will reset the environment, run
all configured dummy stabilization actions, and then start the episode wall
timer immediately before the first policy observation and prediction. The
timer will continue to end when the terminal environment action returns.
Compilation, model prewarming, reset, stabilization, video encoding, and
post-terminal asynchronous cleanup remain outside Total Time.

For synchronous and consistency-distilled evaluation, Chunk Time remains the
mean CUDA-synchronized model inference call. For AC Streaming evaluation, the
primary Chunk Time field will use all non-warmup D8 inference samples,
matching RoboTwin. Initial D0 inference does not define the public AC
Streaming Chunk Time. The first-background and steady-state D8 breakdowns
remain available as additional diagnostics; the first runtime D8 remains part
of the primary mean because model prewarming is already outside timing.

The LIBERO Total Time aggregation is deliberately unchanged. RoboTwin may
continue to use successful-episode task/config macro averaging while LIBERO
continues to expose its existing aggregate and Long/Short successful-episode
fields.

## Compatibility

Existing detailed timing fields remain available. The semantic change is
limited to the LIBERO AC Streaming primary Chunk Time value and the LIBERO
episode timer boundary, plus a nullable Chunk Time for zero-D8 AC episodes in
both benchmarks. Results generated before and after this change should record
the source commit because the Total Time values are not directly
interchangeable.

## Testing

Regression tests will verify that:

- LIBERO stabilization actions occur before the episode timer starts.
- AC Streaming primary Chunk Time selects all non-warmup D8 samples rather
  than mixing D0 and D8.
- Synchronous Chunk Time remains unchanged.
- Detailed D0/D8 diagnostics and the intentionally different Total Time
  aggregation remain intact.
