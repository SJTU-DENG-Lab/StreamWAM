# Stream-WAM Interwoven Method Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three-card method summary with a readable, original interwoven timeline that accurately shows Stream-WAM predicting during robot execution and conditioning the next visual future on the committed action prefix.

**Architecture:** Keep the method artwork in one standalone, self-describing SVG and embed it through a responsive HTML figure shell. The SVG owns the complete static explanation plus non-essential CSS animation; page CSS owns breakout sizing, the contained mobile viewport, caption typography, and focus behavior. Existing page tests are extended to parse both HTML and SVG so the asset, semantic groups, wording, insertion order, and reduced-motion fallback cannot silently regress.

**Tech Stack:** Semantic HTML, responsive CSS, standalone SVG/CSS animation, Python `pytest`, `xml.etree.ElementTree`.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-08-26-streamwam-interwoven-method-figure-design.md`.
- Preserve the complete explanation when animation is disabled.
- Show one directed action-prefix path into the next visual-future group, not unrestricted cross-chunk attention.
- Keep robot execution visually continuous through handoff and show prediction overlapping current execution.
- Do not use `VM`, `IDM`, `FDM`, cache terminology, exact latency values, or the reference figure's stacked-table composition.
- Keep mobile overflow local to the labelled artwork viewport.

---

## Task 1: Lock the method-figure contract with failing tests

**Files:**

- Modify: `tests/test_academic_project_page.py`
- Test: `tests/test_academic_project_page.py`

- [ ] Add `xml.etree.ElementTree` and a `METHOD_FIGURE_PATH` constant.
- [ ] Add an HTML integration test that requires the standalone SVG, exact caption, screen-reader description, and placement between the mechanism prose and the loop-repetition prose.
- [ ] Add an SVG semantics test that requires the execution rail, startup/streaming prediction groups, prefix bridge, future-visual target, condition slots, and handoff while rejecting cache/VM/IDM/FDM labels.
- [ ] Add a motion/accessibility test that requires an SVG title/description, visible static base state, named method animations, and an internal `prefers-reduced-motion: reduce` override.
- [ ] Add a responsive CSS test that requires a horizontally contained method viewport and readable minimum artwork width on narrow screens.
- [ ] Run `pytest -q tests/test_academic_project_page.py -k 'interwoven or method_svg or method_figure'` and confirm the new tests fail because the new asset and markup do not exist yet.

## Task 2: Build and embed the standalone interwoven timeline

**Files:**

- Create: `docs/assets/stream-wam-method.svg`
- Modify: `docs/index.html`
- Modify: `docs/styles.css`
- Test: `tests/test_academic_project_page.py`

- [ ] Create a `1600 × 760` SVG with a light editorial panel, conceptual time spine, two staggered joint world-action prediction cards, and one gapless teal robot-execution rail.
- [ ] Draw the committed-prefix bracket and curved bridge so it ends only at the next visual-future target; include a compact visual/action/condition-slot cutaway.
- [ ] Bend the prepared next-action strip into the execution rail at the handoff and fade the rail toward a repeated continuation.
- [ ] Add subtle six-to-eight-second flow, reveal, prefix, future, and handoff animations while retaining the complete final state as the unanimated base.
- [ ] Add an internal reduced-motion rule that disables every method animation without hiding any explanatory element.
- [ ] Replace the old three-card HTML with a labelled viewport, the standalone SVG image, hidden extended description, and the approved visible caption.
- [ ] Replace the obsolete `.method-figure` grid rules with a responsive figure shell, caption styling, focus styling, and a narrow-screen minimum artwork width.
- [ ] Bump the shared CSS/JS cache-busting version in `docs/index.html`.
- [ ] Run the targeted tests until they pass.

## Task 3: Verify behavior, rendering, and regression safety

**Files:**

- Verify: `docs/assets/stream-wam-method.svg`
- Verify: `docs/index.html`
- Verify: `docs/styles.css`
- Verify: `tests/test_academic_project_page.py`

- [ ] Parse the SVG as XML and render it with an available local SVG-capable renderer.
- [ ] Inspect the rendered figure at desktop scale for readable labels, continuous execution, correct overlap, and directed conditioning.
- [ ] Inspect the page at desktop and narrow widths; confirm no document-level horizontal overflow and that figure-only overflow remains inside its labelled viewport.
- [ ] Run `pytest -q tests/test_academic_project_page.py`.
- [ ] Run `git diff --check` and inspect the scoped diff.
- [ ] Request a focused code/design review and address any high-confidence findings.
- [ ] Commit the implementation only after verification passes.
