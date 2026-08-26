# Stream-WAM method figure alignment design

## Scope

Refine the existing static Stream-WAM method SVG without changing its attention-mask semantics or its current temporal-segment labels.

## Layout

- Align the lower `Temporal overlap` and `Action-conditioned attention` frame with the overview's `Cold start` left boundary at `x = 58`.
- Preserve the lower frame's right boundary at `x = 1520`, then rebalance the temporal and attention columns inside that wider frame.
- Keep the attention mask, token order, and two gold cross-chunk cells unchanged.

## Temporal inset

- Preserve the current A₀ labels: `execute`, `shared prefix`, and `look-ahead`.
- Preserve the current A₁ labels: `shared prefix` and `new actions`.
- Place `known action context` and `unknown future slots` below the A₁ row, aligned respectively with the A₁ shared-prefix and continuation columns.
- Remove the `Observe O₁` annotation while retaining the temporal boundary marker.
- Remove `predict A₁ while A₀ continues` and its decorative gold rule.

## Overview cleanup

- Remove the bent arrow from the A₁ output back to `Robot execution A₁`; continuous execution already communicates the transition.
- Preserve the rest of the overview and the inset callout.

## Verification

- Automated tests assert frame alignment, removed annotations and arrow, conditioning-slot placement, and unchanged attention connectivity.
- Browser rendering is checked at desktop and narrow widths before pushing to `main`.
