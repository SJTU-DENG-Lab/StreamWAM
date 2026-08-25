# Stream-WAM Static Latency Figure Design

## Scope

Refine the academic project page results presentation without changing any reported values:

- Order benchmark sections as LIBERO, RoboTwin 2.0, then RoboCasa.
- Remove the LingBot-VA and LingBot-VA from WAN2.2 comparison rows.
- Replace the HTML latency bars with one Python-generated static figure.
- Include LIBERO, RoboTwin 2.0, and RoboCasa in the latency figure.
- Rename Total Time to Episode Time everywhere in the figure and surrounding copy.

## Figure design

The static figure is a 2×2 grid of vertical bar charts:

1. LIBERO Chunk Time (ms)
2. LIBERO Episode Time (s), with Long and Short shown as paired bars
3. RoboTwin 2.0 and RoboCasa Chunk Time (ms), grouped by benchmark
4. RoboTwin 2.0 and RoboCasa Episode Time (s), grouped by benchmark

All currently reported latency methods remain visible. Stream-WAM uses the page's saturated teal, complete baselines use muted warm brown and gray, and the two LIBERO ablations use light hatched bars inside a subtly labeled ablation region. Every bar has its exact value printed above it and every panel states that lower is better. All panels use continuous axes; the LIBERO chunk panel spans 0–520 ms without an axis break.

The generated image uses the page's warm paper background and restrained typography. It is exported as a high-resolution PNG with a transparent or matching paper surround, embedded responsively, and linked to itself so mobile readers can open the original resolution.

## Reproducibility

A repository-local Python script owns both the authoritative latency constants and rendering. It uses matplotlib and can be rerun to reproduce the committed PNG. The site remains dependency-free at runtime because GitHub Pages serves only the generated image.

## Verification

- Page tests assert benchmark order, the absence of LingBot-VA rows, the static image reference, accessible alternative text, and the absence of legacy HTML latency bars.
- Generator tests run the script and verify that it produces a non-empty PNG with the expected signature.
- Desktop and mobile browser checks verify image scaling, no page-level horizontal overflow, and readable surrounding copy.
