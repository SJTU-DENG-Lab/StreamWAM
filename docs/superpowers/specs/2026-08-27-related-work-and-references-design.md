# Related Work and References Design

## Goal

Replace the first two editorial sections of the Stream-WAM project page with a shorter, source-grounded account of the real-time deployment problem and existing asynchronous strategies. Align all Stream-WAM prose on the page with the `shared actions` terminology used by the current Streaming overview, and add a standard numbered References section immediately above Citation.

## Scope

- Rewrite the first editorial section at `#act-wam`.
- Rewrite the second editorial section at `#act-async`.
- Keep the third Streaming overview section and method figure structurally unchanged.
- Replace `committed action prefix` with `shared actions` throughout the project page. When the input representation matters, use `shared action slots`, `unknown action slots`, and `condition slots`.
- Do not use `shared prefix`; the method figure and implementation do not use that term.
- Add References as a sibling section immediately before the existing Citation section.
- Preserve all task results, latency results, figures, Citation BibTeX, and open-source buttons.

## Section 1

### Title

`World Action Models and the Real-Time Gap`

### Copy

> World Action Models (WAMs) jointly predict future visual representations and action chunks, allowing robot policies to connect action generation with object motion, physical interaction, and task progress [1]. This coupling has produced strong manipulation performance, but its iterative generation is often much slower than the robot control cycle.
>
> Under synchronous deployment, inference and execution are serialized: the robot finishes its current chunk, waits for the next prediction, and only then resumes motion. Model latency therefore becomes idle time or repeated stale commands, while recent environmental changes cannot affect the policy until the next update.
>
> Longer action horizons reduce the frequency of model calls but commit the robot to a staler open-loop plan. Faster inference shortens the wait, yet any remaining latency still lies on the critical path. Real-time WAM deployment therefore requires prediction to overlap ongoing motion without losing consistency between the predicted world evolution and the actions actually executed.

This section contains 128 words, approximately half the length of the supplied draft while retaining the WAM definition, synchronous bottleneck, horizon trade-off, and required deployment shift.

## Section 2

### Title

`Asynchronous Strategies and the Missing World Coupling`

### Copy

> Asynchronous execution starts the next prediction before the current action chunk is exhausted. This creates a temporal overlap in which the same actions are shared across adjacent chunks, and the incoming prediction must remain consistent with the motion already underway.
>
> Real-time chunking (RTC) handles this boundary at inference time through inpainting-style guidance of the denoising trajectory [2,3]. It requires no retraining, but adds guidance computation at every denoising step and did not reliably constrain the delay region in the WAM deployment studied in [2]. Prefix-conditioned methods instead provide the delay-region actions as clean context during training. Training-Time RTC is a representative method in this family: it learns to generate the remaining action continuation without additional inference-time guidance, but requires model retraining [2,4].
>
> These methods primarily address continuity in action space. LingBot-VA 2.0 additionally introduces FDM grounding, which uses recent visual feedback and the executing action to roll the visual cache forward before the standard video and action prediction [5]. This additional future-state prediction increases the computation required by each update and can limit efficiency at high control rates.
>
> Stream-WAM couples action continuity with world prediction inside a joint WAM. It reuses shared actions across adjacent chunks and routes them both directly into each Stream Update and into shared action slots. Together with the unknown action slots, the resulting condition slots guide action-conditioned visual generation while robot execution continues.

This section contains 204 words. It presents the methods in a causal sequence: asynchronous overlap, inference-time RTC, prefix-conditioned Training-Time RTC, FDM grounding, and the remaining world-coupling gap addressed by Stream-WAM.

## Technical Accuracy Decisions

### Prefix-Conditioned Methods

`Prefix-Conditioned Methods` is a family in *World Action Models in Real Time*, not a separately named Motubrain algorithm. Training-Time RTC is the representative implementation. The copy therefore names the family and its representative without inventing another method.

### FDM Grounding

LingBot-VA 2.0 Figure 6 shows an FDM future-state prediction before the standard video and action prediction. The prose uses `additional future-state prediction` rather than `predicts the future twice` because the figure and the text surrounding Equation (29) differ slightly in how the later action decode is described. The efficiency statement is framed as a consequence of additional per-update computation, not as a reported FDM-specific ablation.

### Shared Actions

The Streaming overview defines `A₀[8:16] = A₁[0:8]` as `shared actions`. They describe the same temporal interval in adjacent chunks and are not replayed after the switch. Stream-WAM routes these shared actions directly into Stream Update and through shared action slots; shared action slots and unknown action slots form the condition slots used for action-conditioned visual generation.

## Inline Citations

Use bracketed reference links in the text:

```html
<a class="reference-cite" href="#ref-fast-wam" aria-label="Reference 1">[1]</a>
```

Each link points to its numbered entry in the References section. External paper links open in a new tab with `rel="noopener noreferrer"`.

## References Section

Add a sibling section before the existing `#resources` Citation section:

```html
<section class="reference-section" id="references" aria-labelledby="references-title">
  <h2 id="references-title">References</h2>
  <ol class="reference-list">
    <!-- five numbered entries -->
  </ol>
</section>
```

The section uses the same reading-column width as Discussion. It is not nested inside `#resources`, so the existing Citation card and its tests remain structurally stable.

### Entries

1. T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-WAM: Do World Action Models Need Test-Time Future Imagination?” arXiv preprint arXiv:2603.16666, 2026. https://arxiv.org/abs/2603.16666
2. Motubrain Team, “World Action Models in Real Time: An Empirical Study of Smooth Execution via Asynchronous Deployment,” arXiv preprint arXiv:2608.01880, 2026. https://arxiv.org/abs/2608.01880
3. K. Black, M. Y. Galliker, and S. Levine, “Real-Time Execution of Action Chunking Flow Policies,” arXiv preprint arXiv:2506.07339, 2025. https://arxiv.org/abs/2506.07339
4. K. Black, A. Z. Ren, M. Equi, and S. Levine, “Training-Time Action Conditioning for Efficient Real-Time Chunking,” arXiv preprint arXiv:2512.05964, 2025. https://arxiv.org/abs/2512.05964
5. Q. Zhang et al., “Native Video-Action Pretraining for Generalizable Robot Control,” arXiv preprint arXiv:2607.08639, 2026. https://arxiv.org/abs/2607.08639

## Presentation

- References use a restrained numbered list rather than citation cards.
- The heading matches the existing serif section-heading style.
- Entries use compact body text, hanging indentation, and visible but understated arXiv links.
- References sit close to the concluding Discussion copy; Citation follows with its existing larger BibTeX card and open-source buttons.
- No new colored background region is introduced.

## Verification

- Update semantic page tests for the new section headings and core method terminology.
- Add assertions for References-before-Citation ordering, five stable reference IDs, inline back-links, and the five official arXiv URLs.
- Assert that `committed action prefix` and `shared prefix` do not appear in visible project-page copy.
- Run the complete project-page test module and inspect the local page at desktop and mobile widths.
