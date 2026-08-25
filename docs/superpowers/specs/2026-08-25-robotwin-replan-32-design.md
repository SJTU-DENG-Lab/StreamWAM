# RoboTwin Joint/CD Replan-32 Design

## Goal

Make the RoboTwin synchronous Joint and CD evaluations execute the complete
32-action prediction horizon before replanning. Keep AC-Stream's checkpoint
contract unchanged at H32/s16/d8.

## Runtime behavior

- `baseline`: `replan_steps=32`
- `cd`: `replan_steps=32`
- `ac-stream`: fixed H32/s16/d8; the controller continues to switch chunks
  every 16 executed actions and launches D8 inference after 8 actions.

This comparison aligns Joint/CD with the model's full H=32 output horizon. It
does not claim that Joint/CD and AC-Stream have the same replanning frequency.

## Code boundary

The mode-to-replan mapping remains centralized in the RoboTwin multi-GPU
manager when constructing each one-job simulator command. Launch scripts,
model code, checkpoint loading, timeout supervision, and timing formulas remain
unchanged.

## Verification

- Add a manager test asserting that baseline and CD worker commands receive
  `--replan-steps 32`.
- Assert that AC-Stream worker commands still receive `--replan-steps 16`.
- Run the focused RoboTwin tests and the full repository test suite.

## Evaluation interpretation

The rerun should report success rate, chunk time, and total time per episode.
Because replan=32 changes the executed policy trajectory, both success rate and
the number of actions required for success may change relative to the earlier
replan=24 results.
