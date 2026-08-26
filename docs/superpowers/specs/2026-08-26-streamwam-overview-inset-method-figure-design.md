# Stream-WAM Overview + Inset Method Figure Design

## Goal

Replace the oversized 32-cell diagram with one compact academic figure that
works at two reading scales:

1. a Figure-6-style global spacetime overview explains the repeated streaming
   pipeline across several model updates and robot-execution windows;
2. one small inset inside unused overview space magnifies a single overlap and
   explains observation timing, action-prefix continuity, handoff, and visual
   action conditioning.

The figure must communicate the method without requiring readers to count
individual actions.

## Overall Composition

Use one SVG, not two separate figures. The canvas remains 1600 px wide and is
reduced from 980 px to approximately 720–760 px high.

The global overview occupies the upper and left portions of the canvas. The
inset sits inside natural whitespace in the lower-right portion of the same
canvas. A restrained dashed callout links one selected overlap in the overview
to the inset.

The inset must not cover the overview timeline, labels, arrows, or outputs.

## Global Overview

The overview follows a horizontal `t₀ → t₃` spacetime axis and shows three
successive stages:

- cold start: `O₀ → Joint WAM → V₀ + A₀`;
- while `A₀` executes, Stream-WAM observes `O₁` and predicts `V₁ + A₁`;
- while `A₁` executes, Stream-WAM observes `O₂` and predicts `V₂ + A₂`.

Use long continuous bars rather than action cells:

- green bars denote robot execution;
- warm neutral model boxes or bars denote model prediction;
- blue blocks denote predicted visual futures;
- gold paths denote action-conditioned visual generation.

Each model prediction begins during the current execution window and ends at or
before the next handoff. Execution remains continuous after the initial cold
start. The second streaming cycle is a lighter repetition so the first cycle
remains the visual focus.

The overview explains the repeated asynchronous pipeline only. It does not show
action indices, slot counts, or per-action cells.

## Magnified Inset

The inset magnifies one selected `t → t+1` overlap from the first streaming
cycle. It uses three compact layers.

### Timeline layer

Draw `A₀` as three continuous segments:

- execution before the new observation;
- a gold overlap segment that continues executing while inference runs;
- a pale look-ahead suffix that is replaced at handoff.

At the first boundary, mark `Observe O₁` and the start of `AC-Stream inference`.
Do not assign inference completion to a fixed action index.

### Continuity layer

Draw `A₁` below `A₀`, shifted so its gold shared prefix aligns vertically with
the gold overlap segment of `A₀`. A short label states that they are the same
physical actions. At the right boundary of the shared segment, a handoff arrow
enters the green continuation of `A₁`.

Do not draw 32 cells or enumerate every action. If numeric notation is needed,
use only the compact range relationship `A₀[8:16] = A₁[0:8]`.

### Conditioning layer

Copy the shared prefix into a two-part condition strip:

- `known action context` in gold;
- `unknown future slots` as a pale dashed block.

The strip and `O₁` enter the same `AC-Stream update`. A gold directed path ends
at `V₁`, while a neutral output path ends at `A₁`.

The inset explains both action continuity and action-conditioned visual future
generation without becoming a second full diagram.

## Visual Language

- Static SVG only; no cursor, animation, shimmer, or keyframes.
- White background, flat fills, thin strokes, no gradients, filters, shadows,
  glow, or cartoon illustration.
- Use the existing muted green, gold, blue, gray, and warm-neutral palette.
- Use a thin neutral inset border and a dashed magnification connector.
- Keep all primary labels readable when the artwork is displayed at 1100 px.
- Preserve the existing publication-style typography and compact monospaced
  symbols for `O`, `V`, and `A` tokens.

## Page Integration

Keep the figure in its current position under “Stream-WAM conditions the visual
future on the action already underway.” Update the SVG intrinsic height, cache
key, hidden description, and caption to describe the two reading scales.

The desktop page continues using the current breakout width. The mobile figure
remains horizontally scrollable with a minimum artwork width of 1100 px.

## Verification

- The SVG has one global `t₀ → t₃` overview and exactly one magnified inset.
- The overview shows continuous execution with prediction overlapped inside the
  current execution window.
- The overview contains no per-action cells or detailed slot construction.
- The inset contains no 32-cell ribbon.
- The inset aligns the shared `A₀` segment with the `A₁` prefix.
- Handoff enters the continuation of `A₁`, not the start of its prefix.
- Observation and action context enter the same update.
- The gold conditioning path terminates at `V₁`.
- The inset remains inside overview whitespace without covering the main flow.
- No animation or forbidden completion-time copy remains.
- The 1600 px and 1100 px browser renders remain legible.
- Page tests, XML validation, and `git diff --check` pass.
