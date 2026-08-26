# Stream-WAM Interwoven Method Figure Design

## Goal

Replace the three-card method diagram in the action-conditioned streaming
section with a full-width, publication-ready method figure. The figure must
explain the LIBERO AC-Stream mechanism without copying the cache-centric visual
grammar of Figure 6 in arXiv:2607.08639.

The same vector source must work in two forms:

- a complete static SVG suitable for screenshots, slides, and paper export;
- a lightly animated project-page figure whose motion reinforces, but does not
  carry, the method explanation.

## Source of Truth

The figure follows the repository implementation rather than presenting a
generic asynchronous pipeline:

- The controller uses an initial D0 prediction followed by asynchronous D8
  predictions under the H32/s16/d8 rollout contract.
- A new prediction launches after execution advances into the current action
  chunk and overlaps the remaining execution window.
- The controller aligns a target from the active model action chunk and clamps
  its known prefix into the next prediction.
- The condition stream represents known action slots separately from unknown
  slots.
- Directed attention carries the executing action prefix into the next
  future-video `z1` tokens. It does not create unconstrained cross-chunk
  attention into every visual and action token.
- At the stride boundary, the prepared prediction is installed at the aligned
  cursor; the controller waits only if the asynchronous result is not ready.

Exact action counts are part of the implementation but are omitted from the
visible artwork. The figure communicates launch, overlap, prefix alignment,
and handoff without turning the main method image into a timing specification.

## Distinction from the Reference Figure

The reference Figure 6 uses stacked time slices, observation caches, VM/IDM
boxes, and forward-dynamics cache updates. Stream-WAM's figure must not reuse
those components or its table-like composition.

Instead, the new figure uses one horizontally interwoven flow:

- a continuous execution rail;
- staggered world-action prediction cards;
- a curved action-prefix bridge from active execution into the next visual
  future;
- a compact model cutaway that exposes the directed condition path;
- a seamless handoff marker where the next prediction joins the execution
  rail.

This visual metaphor makes the method recognizable as Stream-WAM rather than a
redrawn cache schedule.

## Composition

The SVG uses a wide `1600 × 760` view box and a light editorial background that
matches the article section.

### 1. Time spine

A thin horizontal time arrow crosses the composition from left to right. Three
soft vertical landmarks label the conceptual phases `Observe`, `Think while
acting`, and `Handoff`, without numbered timesteps.

### 2. Continuous execution rail

A teal rail occupies the lower third of the figure. It begins after the startup
prediction and continues across both chunk windows without a visual gap. Small
action capsules move along it in the animated version. Two adjacent rail
segments use subtly different teal shades to make the chunk handoff legible
without implying a pause.

The segment between asynchronous launch and handoff is overlaid by a gold
bracket labelled `Committed action prefix`. This is the intervention underway
while the next prediction is computed.

### 3. World-action prediction cards

Two staggered cards occupy the upper half:

- The startup card receives the current observation and produces the first
  visual/action chunk.
- The streaming card begins above the active execution rail and finishes before
  the handoff marker.

Each card contains a compact paired representation:

- a blue filmstrip for `Current observation` and `Visual future`;
- a violet/gold strip for `Next action chunk`.

The cards use one shared enclosing shape to emphasize joint world-action
prediction rather than separate VM and policy modules.

### 4. Action-conditioned bridge

A gold-to-violet curved ribbon rises from the committed-prefix bracket into the
streaming prediction card. It terminates only at the first future-video group,
which receives a blue-violet glow. A concise label reads `Action-conditioned
visual future`.

Inside the streaming card, a miniature three-stream cutaway shows:

- visual tokens;
- policy action tokens;
- known/unknown condition slots.

Only one directed bridge runs from the known action slots to the future visual
group. The cutaway must not resemble a dense attention matrix and must remain
readable at the article's responsive width.

### 5. Handoff and repetition

At the handoff marker, the prepared action strip bends into the continuous
execution rail. A short `Next chunk ready` label marks the transfer. A faded
continuation at the right edge indicates that the same loop repeats without
adding a third full card.

## Visual Language

- Teal: real robot execution and continuity.
- Blue: observations and visual futures.
- Violet: generated world-action prediction.
- Gold: committed actions and the directed conditioning path.
- Charcoal: text, axes, and structural outlines.
- Pale gray: inactive or future structure.

Use sentence case throughout. Avoid all-caps module names, exact latency
numbers, `VM`, `IDM`, `FDM`, cache terminology, and large blocks of explanatory
text inside the figure.

The visible labels are limited to:

- `Current observation`
- `World-action prediction`
- `Robot execution`
- `Committed action prefix`
- `Action-conditioned visual future`
- `Next action chunk`
- `Next chunk ready`
- `Continuous control`

## Animation

The animation is a subtle six-to-eight-second loop:

1. The startup card resolves and the first execution segment becomes active.
2. Teal action capsules advance continuously along the execution rail.
3. At the launch landmark, the streaming prediction card begins to resolve
   left-to-right.
4. The committed-prefix bracket and curved bridge pulse once.
5. The future-video group lights up, followed by the next action strip.
6. At handoff, the next strip joins the execution rail without a gap.

No element disappears in a way that changes the explanation. The base SVG
already shows the final method state. Animation changes opacity, glow, and
local reveal masks only. `prefers-reduced-motion: reduce` disables every loop
and displays the static final state.

## Project-Page Integration

- Replace the existing `.method-figure` three-card markup after the first four
  paragraphs of `#act-streamwam`.
- Keep the figure between the mechanism explanation and the paragraphs that
  describe loop repetition and framework scope.
- Embed the standalone SVG from `docs/assets/stream-wam-method.svg` in a
  responsive figure shell.
- Keep a visible academic-style caption under the artwork and a complete
  screen-reader description in HTML.
- At desktop widths, the figure may extend to the existing breakout width. At
  tablet and mobile widths, preserve the complete composition and allow a
  labelled horizontal viewport rather than shrinking text below readability.

## Caption

`Stream-WAM predicts the next world-action chunk while the robot executes the
current one. The committed action prefix is aligned with the new prediction and
conditions its visual future, allowing the prepared action continuation to
enter the control stream at the next handoff.`

## Verification

- The standalone SVG remains understandable with animation disabled.
- The action-prefix bridge terminates at the future visual group, not at every
  output group.
- Execution is visually continuous across the handoff.
- The asynchronous prediction begins during the current execution window and
  ends no later than the conceptual handoff in the illustrated successful
  case.
- The figure contains none of the reference paper's VM/IDM/FDM or cache labels.
- Text remains readable at 1440 px and 1024 px page widths.
- The 390 px and 320 px layouts have no document-level horizontal overflow;
  any figure-only overflow is contained in its labelled viewport.
- Reduced-motion mode has no animated transforms, opacity loops, or pulsing
  effects.
- Automated page tests assert the asset, caption, accessible description, and
  insertion order before the experiment section.
