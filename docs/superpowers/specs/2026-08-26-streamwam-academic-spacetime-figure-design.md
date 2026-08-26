# Stream-WAM Academic Spacetime Figure Design

## Goal

Replace the current illustrated interwoven timeline with a publication-style
asynchronous spacetime diagram. The new figure should communicate the complete
AC-Stream loop at a glance: cold-start prediction, action execution, prediction
during execution, aligned action-prefix conditioning of the next visual future,
and a gapless handoff to the next action chunk.

The composition may borrow the two-dimensional time/cycle grammar of Figure 6
in arXiv:2607.08639, but it must use Stream-WAM's own model states, data flow,
and visual identity rather than copying that paper's cache, VM, IDM, or FDM
components.

## Source of Truth

The diagram follows the repository's AC-Stream implementation:

- The episode begins with one synchronous cold-start world-action prediction.
- Robot execution advances through the current action window without a planned
  pause.
- Partway through that execution window, the controller snapshots the current
  observation and launches the next prediction asynchronously.
- The active model action chunk is temporally aligned into a previous-action
  target. Its known prefix is clamped into the next prediction.
- The known action condition contributes directed attention to the next visual
  future, while the next action continuation is generated jointly with that
  future.
- At the action-window boundary, the prepared prediction is installed at its
  elapsed cursor and execution continues. The controller waits only if the
  asynchronous prediction misses the boundary.

Exact action counts, delay values, and latency numbers are intentionally omitted
from the artwork. The figure explains the mechanism, not one evaluation setting.

## Composition

The standalone SVG uses a `1600 × 860` view box, a white background, and a
left-to-right time axis across the top.

### Time structure

- Four vertical landmarks are labelled `t₀`, `t₁`, `t₂`, and `t₃`.
- Thin vertical rules descend from the landmarks.
- Thin dashed horizontal rules separate `Cold start`, `Streaming update 1`, and
  `Streaming update 2`.
- The layout is a two-dimensional spacetime chart rather than a collection of
  floating cards.

### Cold start

The first row shows:

`Observation O₀ → Joint WAM → Predicted video V₀ + Action chunk A₀`

The predicted action connects directly to an `Executing A₀` bar spanning the
next time interval.

### Streaming update

While `A₀` is executing, the next row shows two inputs entering one compact
`AC-Stream update` block:

- the current `Observation O₁`;
- an `Aligned action prefix` extracted from the active execution bar.

The update produces a paired output:

- `Action-conditioned video V₁`;
- `Next action chunk A₁`.

A single gold directed path from the aligned prefix terminates at `V₁`. It must
not point to the whole output pair or imply unrestricted cross-chunk attention.
The `A₁` output connects forward to an `Executing A₁` bar beginning at the next
handoff landmark with no visual gap.

### Repetition

A third, lighter row repeats the same structure for `O₂`, `V₂`, and `A₂`. It
establishes that Stream-WAM is a recurring controller rather than a one-off
transition, without adding another full explanation.

## Visual Language

The figure should resemble a camera-ready method diagram:

- pure white or near-white background;
- 1–1.5 px charcoal axes and connectors;
- square or very slightly rounded modules, with no shadows or glow;
- compact serif/sans headings and monospaced mathematical labels;
- pale gray observations, muted blue video predictions, muted green action
  chunks and execution bars, and muted gold action-prefix conditioning;
- no robot illustration, landscape thumbnails, capsules, gradients, large
  checkmarks, decorative arrows, or oversized prose.

Every connector should have a causal meaning. Labels should be short enough to
remain readable at the page's desktop breakout width and its mobile minimum
artwork width.

## Animation

The static SVG contains the entire explanation. The only animation is a thin,
low-opacity vertical time cursor that moves from `t₀` to `t₃` over approximately
eight seconds.

The cursor does not reveal, hide, scale, pulse, or recolor any method component.
`prefers-reduced-motion: reduce` removes the cursor animation and leaves the
cursor at `t₀` or hides it without changing the diagram.

## Page Integration

- Replace `docs/assets/stream-wam-method.svg` in place so the existing method
  section and caption remain stable.
- Retain the labelled horizontal viewport for narrow screens.
- Update the HTML extended description and visible caption only where necessary
  to match the new figure vocabulary.
- Preserve the figure's location between the mechanism explanation and the
  paragraphs describing repeated operation.

## Verification

- The SVG parses as valid XML and rasterizes at 1600 px and 1100 px.
- The time landmarks, update rows, prediction blocks, execution bars, aligned
  prefix, conditioned visual outputs, and forward handoffs are present.
- Every asynchronous prediction block begins during its current execution bar
  and finishes before or at the next handoff.
- The gold prefix path ends at the next video output only.
- Execution bars meet their handoff landmarks without a gap.
- The artwork contains no `VM`, `IDM`, `FDM`, or cache terminology.
- No content other than the time cursor is animated.
- The mobile viewport contains horizontal overflow without widening the document.
- Page-specific automated tests and `git diff --check` pass.
