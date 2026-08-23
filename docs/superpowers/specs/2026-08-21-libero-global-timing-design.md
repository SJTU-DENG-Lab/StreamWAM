# LIBERO Global Chunk Timing Design

LIBERO video output uses a fixed default of 30 FPS with no new CLI option. Timing is collected at chunk granularity and aggregated across every task and trial selected by one command. No timing is printed for individual chunks, trials, or tasks. A single summary is logged after all requested evaluation work finishes and stored once under `results.json["timing_summary"]`.

The only measured chunk components are communication, model inference, and action execution. Communication covers observation/context/proprio preparation, CPU-to-device inputs, device-to-CPU action output, denormalization, and NumPy conversion. Inference covers the synchronized `model.infer_action` call. Action execution is the sum of wall time spent in `env.step(action)` for actions consumed from that generated chunk. Chunk total is the sum of those three values. Small overhead is not itemized.

The final summary contains task, trial, and chunk counts; average communication, inference, action execution, and total milliseconds per generated chunk; and command wall time. Model internals and checkpoint loading are unchanged.
