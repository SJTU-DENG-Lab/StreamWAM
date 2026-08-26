# Stream-WAM Action Overlap Detail Design

## Goal

Refine the academic spacetime figure so the first streaming update explains
AC-Stream through concrete action cells rather than abstract `Observation` and
`Aligned action prefix` blocks. A reader should be able to identify when the
new observation is captured, when inference starts and ends, which actions are
executed during inference, how `A₀` overlaps `A₁`, and how the 16 condition
slots are constructed.

## Authoritative Timing

The figure follows the fixed LIBERO AC-Stream schedule:

- Every model action chunk contains 32 actions.
- The controller executes a 16-action window before installing the next
  prediction.
- After `A₀[0:8]` has executed, the controller captures `O₁` and launches the
  asynchronous prediction.
- `A₀[8:16]` continues executing while that prediction runs.
- The eight actions `A₀[8:16]` are clamped into `A₁[0:8]`; these are the same
  actions at the same physical times, not two separately executed segments.
- At the handoff after `A₀[15]`, the controller installs `A₁` at cursor 8 and
  continues from `A₁[8]`.
- `A₀[16:32]` is unused look-ahead after this handoff.
- The next action window executes `A₁[8:24]`. Its latter eight actions become
  the overlap with the following chunk.

Inference completion has no fixed action index. The figure places an unlabeled
end tick before the action-16 handoff to illustrate the successful overlap case
without presenting that location as an algorithm parameter.

## Figure Changes

### Remove motion

Delete the animated time cursor and every related keyframe and reduced-motion
rule. The method figure is completely static.

### A₀ action ribbon

Replace the single `Executing A₀` rectangle with a 32-cell horizontal ribbon.
Every cell represents one action and the ribbon is divided visually into:

- cells `0–7`: actions executed before the new observation;
- cells `8–15`: actions executed while asynchronous inference runs;
- cells `16–31`: unused look-ahead after the handoff.

Use green for executed cells, a gold outline or top band for the eight actions
executed during inference, and pale gray for unused look-ahead. Brackets label
the three ranges without numbering every cell individually.

### Observation and inference markers

At the boundary between cells 7 and 8, draw a compact camera-frame snapshot
labelled `Observe O₁`. The same vertical marker begins a thin `AC-Stream
inference` bar above cells `8–15`. The inference bar ends before the action-16
handoff with an unlabeled end tick.

### A₀/A₁ overlap

Draw a second 32-cell ribbon for `A₁`, shifted so `A₁[0:8]` is vertically
aligned with `A₀[8:16]`. Use matching gold fills and one pair of vertical
connectors to state:

`A₀[8:16] = A₁[0:8]`

The overlap is not executed twice. A handoff arrow at the action-16 boundary
points directly to `A₁[8]`, and `A₁[8:24]` is marked as the next executed
window. `A₁[24:32]` is pale look-ahead.

### Condition-slot construction

Below the overlap, draw a 16-cell condition strip:

- slots `0–7`: gold known slots filled from `A₀[8:16]`;
- slots `8–15`: pale unknown slots with a restrained empty-slot mark.

A bracket reads `16 condition slots`, with subordinate labels `8 known action
slots` and `8 unknown slots`. A direct connector from the overlap copies the
eight shared actions into the known half. The complete slot strip enters the
AC-Stream update, and a gold directed path from that update terminates at the
next visual future `V₁`.

The observation snapshot enters the same update separately. The output retains
the paired `V₁` and `A₁` semantics, but the action ribbon itself explains the
hard prefix clamp and continuation more clearly than a model-box annotation.

## Visual Language

- Preserve the current white background, thin time rules, monospaced action
  labels, flat muted palette, and publication-style geometry.
- Use square or slightly rounded cells with no gradients, shadows, glow, or
  cartoon illustration.
- The camera snapshot is a compact monochrome frame glyph, not a decorative
  robot scene.
- Keep exact numeric labels only where they explain the algorithm: 32 actions,
  indices `0–7`, `8–15`, `16–31`, and 16 condition slots.
- Do not show latency numbers, an exact inference-completion index, or the text
  `Next chunk ready` or `completion time varies`.

## Layout

Retain the cold-start row and the top `t₀–t₃` spacetime axis. Expand the first
streaming-update row vertically so the detailed `A₀`, `A₁`, and condition-slot
ribbons remain readable. Reduce the second streaming update to a light repeated
continuation rather than repeating the complete detail.

The SVG may increase its height from 860 to 980 if needed. Preserve the page's
contained mobile viewport and a minimum artwork width of 1100 px.

## Verification

- No animation or `@keyframes` remains in the SVG.
- `A₀` and `A₁` each contain exactly 32 visible action cells.
- Observation and inference begin at the `A₀[7]/A₀[8]` boundary.
- The inference end tick is before the action-16 handoff.
- `A₀[8:16]` and `A₁[0:8]` are aligned and visibly identified as one shared
  eight-action segment.
- The handoff resumes at `A₁[8]` rather than `A₁[0]`.
- The condition strip contains exactly eight known and eight unknown slots.
- The known slots derive from the shared action segment.
- The observation and the complete condition strip enter the same update.
- The directed action-conditioning path ends at `V₁`, not the entire output.
- The 1600 px and 1100 px rasterizations remain readable.
- Page tests and `git diff --check` pass.
