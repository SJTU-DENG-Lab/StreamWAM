# RoboCasa Latency Refresh Design

## Goal

Replace the obsolete RoboCasa runtime measurements everywhere they appear, regenerate both latency figures, and update the reported speedups from the new authoritative values.

## Authoritative Data

Keep the public method names while mapping the supplied internal stages as follows:

| Public method | Supplied stage | Chunk Time | Episode Time |
|---|---|---:|---:|
| X-WAM | normal X-WAM teacher | 374.07 ms | 17.36 s |
| X-WAM-CD | Stage1 CD | 134.37 ms | 13.04 s |
| Stream-WAM | current RTC step2500 | 115.98 ms | 9.49 s |

Do not expose `normal X-WAM teacher`, `Stage1 CD`, `RTC`, or `step2500` in public-facing chart labels or tables.

## Speedup Wording

Compute RoboCasa speedups using X-WAM as the baseline and Stream-WAM as the accelerated method:

- Chunk latency: `374.07 / 115.98 = 3.225...`, reported as `3.2×`.
- End-to-end episode time: `17.36 / 9.49 = 1.829...`, reported as `1.8×`.

Replace the old `3.7×` RoboCasa chunk-latency claim and old `3.2×` RoboCasa end-to-end claim. Do not change the LIBERO or RoboTwin speedups.

## Figure Regeneration

- Update `ROBOCASA_CHUNK` to `(374.07, 134.37, 115.98)`.
- Update `ROBOCASA_EPISODE` to `(17.36, 13.04, 9.49)`.
- Retain method labels `X-WAM`, `X-WAM CD`, and `Stream-WAM` in the chart.
- Retain the existing chart dimensions, typography, colors, annotations, benchmark order, and Stream-WAM teal highlight.
- Tighten only the RoboCasa panel ceilings:
  - Chunk Time: `410 ms`
  - Episode Time: `20 s`
- Regenerate `docs/assets/stream-wam-chunk-time.png` and `docs/assets/stream-wam-episode-time.png` from the checked-in generator.

## Synchronized Surfaces

Update all current public data surfaces:

1. `docs/generate_latency_figure.py` source tuples and RoboCasa ceilings.
2. The hidden accessible latency table in `docs/index.html`.
3. The RoboCasa speedup sentence in `docs/index.html`.
4. The RoboCasa runtime columns in `README.md`.
5. Both checked-in PNG figure assets.
6. Regression tests for source tuples, accessible table values, speedup text, and regenerated-image equality.

Historical design and plan documents preserve the values that were current when those documents were written and are not rewritten.

## Verification

- Add failing tests for the exact new tuples, visible speedup wording, accessible data table, and README values.
- Regenerate figures into a temporary directory and assert they are byte-identical to the checked-in assets.
- Run the complete academic project-page test module.
- Inspect both regenerated images and confirm the updated RoboCasa bars and annotations are readable.
- Review the final diff to ensure LIBERO, RoboTwin, and task-success results remain unchanged.
