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

Keep the existing two-scale academic figure. In the temporal inset, place O₁ at the boundary between executed A₀[0:8] and shared A₀[8:16]. Align A₀[8:16] with A₁[0:8], label both as shared actions, and show the reused segment taking two explicit routes: one to the A₁ action prefix and one to a compact condition-slot strip made from `shared action slots · 8` plus `unknown slots · 8`. Label the combined strip `16 condition slots`.

Remove `known action context`, `unknown future slots`, `shared prefix`, `look-ahead`, the false new-action-to-slot arrow, the decorative orange output arrows after the overview updates, and the attention legend phrase `to next visual future`.

The visible caption and screen-reader description must state the same two-branch semantics.

