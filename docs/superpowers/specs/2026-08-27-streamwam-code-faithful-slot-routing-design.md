# Stream-WAM Code-Faithful Slot Routing Design

## Goal

Correct the lower method inset so it reflects the implemented LIBERO AC-Stream data flow instead of implying that newly predicted actions create unknown condition slots.

## Verified semantics

- A₀ contains 32 actions.
- O₁ is sampled after A₀[0:8] has executed.
- A₀[8:16] is reused in two parallel roles:
  - it is hard-clamped as the action prefix A₁[0:8];
  - it supplies the values of eight shared-action condition slots.
- Eight structure-only unknown slots are appended to those eight shared-action slots, producing 16 condition slots for future-video generation.
- A₁[8:32] is predicted by the model and does not point to or create the unknown slots.

## Visual design

Keep the existing two-scale academic figure. In the temporal inset, place O₁ at the boundary between executed A₀[0:8] and shared A₀[8:16]. Align A₀[8:16] with A₁[0:8] and label both blocks `shared actions`.

The A₁ `shared actions` block has two visible outgoing paths:

- a direct path into `shared action slots`;
- a separate direct input path into `Stream Update` for the hard action prefix.

Place `shared action slots` and `unknown action slots` next to each other on one line per label. Their combined condition path also enters `Stream Update`. O₁ remains a third direct input.

The lower O₁ input box must begin at the same x-coordinate as the O₁ observation boundary on the A₀ timeline. Its connector begins at the moved box's right edge and continues directly to `Stream Update`.

Do not show count suffixes such as `· 8`, `· 16`, or `· 24`. Use `executed actions`, `remaining actions`, `predicted actions`, and `condition slots` without numeric suffixes.

Remove `known action context`, `unknown future slots`, `shared prefix`, `look-ahead`, the false new-action-to-slot arrow, the decorative orange output arrows after the overview updates, and the attention legend phrase `to next visual future`.

The visible caption and screen-reader description must state the same three-input semantics: O₁, the direct shared-action prefix, and the condition slots.
