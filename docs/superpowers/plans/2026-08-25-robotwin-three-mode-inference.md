# RoboTwin Three-Mode Inference Implementation Plan

> **For agentic workers:** Implement task-by-task with test-first development and review after every independently testable task. Do not create Git commits; the user will inspect and commit manually.

**Goal:** Add baseline, first-frame CD, and eager/accelerated AC-Stream inference for RoboTwin 2.0, accepting original StarWAM checkpoints and supporting balanced four-GPU one-trial evaluation.

**Architecture:** Keep RoboTwin simulation in the motus environment and model inference in the wyx PyTorch environment. Reuse `MoTWAM` for baseline/CD, generalize `ACStreamWAM` for RoboTwin RTC, and connect them through the existing remote policy protocol plus a Python multi-GPU manager.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, Triton 3.3.1, CUDA 12.8, pytest, TCP length-prefixed pickle protocol, RoboTwin 2.0/SAPIEN/MPLib.

## Global Constraints

- Preserve existing LIBERO behavior and commands.
- Do not convert or duplicate model checkpoints.
- Do not create Git commits; the user performs manual review and commit.
- Baseline is first-frame four-step Euler with `replan_steps=32`.
- CD is first-frame one-step action consistency with `replan_steps=32`.
- AC-Stream uses the yzy RTC mathematics with `H=32`, `s=16`, `d=8`.
- Eager and accelerated AC-Stream share one mathematical implementation.
- Formal AC-Stream evaluation defaults to accelerated and supports explicit eager selection.
- One trial is 50 tasks times two configurations, totaling 100 episodes.
- Console output reports success, chunk time, total time per episode, and results path once after completion.

---

### Task 1: StarWAM Checkpoint Source Adapter

**Files:**
- Create: `streamwam/checkpointing/starwam_format.py`
- Modify: `streamwam/checkpointing/core.py`
- Modify: `streamwam/checkpointing/loader.py`
- Modify: `streamwam/checkpointing/__init__.py`
- Modify: `tests/test_checkpoint_formats.py`
- Modify: `tests/test_inference_checkpoint_cli.py`

**Interfaces:**
- Produces: `StarWAMCheckpointAdapter`, registered under format name `starwam`.
- Produces: resolution of either a `.pt` file or DeepSpeed checkpoint directory to one model-state file.
- Produces: metadata fields `checkpoint_file`, `checkpoint_format`, `inference_mode`, `step`, and tensor counts.
- Consumes: existing `CheckpointAdapter` registry and `load_inference_checkpoint()` entry point.

- [ ] Add failing tests proving `--checkpoint-format starwam` is accepted by RoboTwin server/manager parsers.
- [ ] Add failing tests for raw `.pt` payload extraction, DeepSpeed directory resolution, known prefix normalization, and missing-file errors.
- [ ] Add failing tests that baseline/CD load ordinary MoT states and AC-Stream requires and loads `rtc_slot_state_embedding.weight`.
- [ ] Add a failing test proving validation happens before the target model is mutated on dimension/mode mismatch.
- [ ] Run the focused tests and confirm they fail for missing StarWAM support.
- [ ] Implement `StarWAMCheckpointAdapter` with read-only source resolution and strict pre-mutation validation.
- [ ] Register `starwam` without changing `streamwam` or `fastwam` adapter behavior.
- [ ] Run focused checkpoint/CLI tests and confirm they pass.

### Task 2: First-Frame CD in `MoTWAM`

**Files:**
- Modify: `streamwam/wam/mot_wam.py`
- Modify: `streamwam/eval/policy.py`
- Modify: `tests/test_consistency_sampling.py`
- Create: `tests/test_robotwin_inference_modes.py`

**Interfaces:**
- Consumes: `sampling_method="consistency"` and `framework.action_video_conditioning`.
- Produces: first-frame consistency action inference when conditioning is `first_frame`.
- Preserves: existing full-video Joint-CD when conditioning is `full_video`.
- Produces: policy-level `inference_mode` validation for `baseline` and `cd`.

- [ ] Add a deterministic failing test matching yzy's first-frame CD boundary for one action tensor and fixed seed.
- [ ] Add failing tests that first-frame CD accepts exactly one step, rejects other step counts, and does not call `infer_joint()`.
- [ ] Add a regression test proving full-video consistency still calls the existing joint path and retains its geometry checks.
- [ ] Add failing policy tests mapping `baseline` to Euler and `cd` to first-frame consistency without changing normalization or denormalization.
- [ ] Run the focused tests and confirm the new cases fail.
- [ ] Split consistency validation into first-frame and full-video contracts.
- [ ] Implement the first-frame branch: encode frame, prefill video KV cache, sample compatible action noise, evaluate one velocity, apply `action_consistency_boundary`, return action.
- [ ] Thread `inference_mode`/sampling arguments through `StreamWAMPolicy.predict_chunk()`.
- [ ] Run consistency and policy tests and confirm all pass.

### Task 3: Generalize AC-Stream and Match the RoboTwin RTC Checkpoint

**Files:**
- Modify: `streamwam/inference/ac_stream.py`
- Modify: `streamwam/modules/ac_stream.py`
- Modify: `streamwam/wam/ac_stream_wam.py`
- Modify: `streamwam/config.py`
- Modify: `streamwam/builder.py`
- Modify: `tests/test_ac_stream.py`
- Modify: `tests/test_ac_stream_acceleration.py`
- Modify: `tests/test_robotwin_inference_modes.py`

**Interfaces:**
- Produces: dynamic AC-Stream contract using configured action/proprio dimensions and input-derived latent geometry.
- Produces: an RTC slot-state parameter compatible with `rtc_slot_state_embedding.weight`.
- Preserves: LIBERO `7-D`, `224x448`, `H32/s16/d8` behavior.
- Adds: RoboTwin `14-D`, `384x320`, `H32/s16/d8` behavior.
- Preserves: one eager forward used as the source of truth for accelerated compilation.

- [ ] Add failing tests for 14-D actions/proprio and RoboTwin image geometry while retaining the existing LIBERO contract tests.
- [ ] Add a failing state-dict compatibility test against the yzy RTC key name and shape.
- [ ] Add deterministic D0 and D8 reference tests comparing masks, slot states, prefix clamping, and outputs to yzy.
- [ ] Add failing eager/accelerated parity tests for both LIBERO and RoboTwin contracts.
- [ ] Run focused AC-Stream tests and confirm failures are limited to the new dynamic contract.
- [ ] Replace module constants for action dimension and image shape with validated per-instance configuration.
- [ ] Generalize static acceleration buffers to a cache keyed by the explicit input contract.
- [ ] Align the slot-state embedding and three-stream routing with yzy while retaining known FastWAM checkpoint compatibility through the loader adapter.
- [ ] Run AC-Stream tests and verify eager/accelerated parity.

### Task 4: RoboTwin Mode-Aware Inference Server

**Files:**
- Modify: `examples/robotwin/policy_server.py`
- Modify: `examples/robotwin/local_policy.py`
- Create: `examples/robotwin/runtime.py`
- Modify: `tests/test_inference_checkpoint_cli.py`
- Create: `tests/test_robotwin_policy_server.py`

**Interfaces:**
- Produces CLI: `--inference-mode baseline|cd|ac-stream`.
- Produces mutually exclusive CLI: `--ac-stream-accelerated`, `--ac-stream-eager`.
- Produces protocol commands: `reset`, `infer`, and mode-aware D0/D8 inference.
- Response fields: `action`, `model_inference_ms`, `backend`, `request_id`, or structured `error`.
- Consumes: `StreamWAMPolicy` and the configured checkpoint adapter.

- [ ] Add failing parser tests for defaults, valid modes, invalid backend/mode combinations, and `starwam` checkpoint format.
- [ ] Add failing protocol tests for baseline/CD synchronous inference and AC-Stream D0/D8 requests.
- [ ] Add failing tests that CUDA timing is synchronized around model inference while compile/prewarm is excluded.
- [ ] Add failing reset/error tests proving stale predictions cannot cross episode boundaries.
- [ ] Run focused tests and confirm failures.
- [ ] Add `runtime.py` for mode validation, runtime preflight, and backend resolution.
- [ ] Build the correct model/policy for each mode and return structured timing metadata.
- [ ] Prewarm accelerated AC-Stream D0 and D8 after load and before measured inference.
- [ ] Run server/parser tests and confirm they pass.

### Task 5: Truly Asynchronous RoboTwin AC-Stream Client

**Files:**
- Modify: `examples/robotwin/client_policy.py`
- Modify: `examples/robotwin/deploy_policy_client.yml`
- Create: `tests/test_robotwin_client_policy.py`

**Interfaces:**
- Consumes: the existing transport-neutral `ACStreamController` with a remote predictor callable.
- Produces: baseline/CD synchronous action queues.
- Produces: AC-Stream D0/D8 action scheduling with model inference overlapping `task_env.take_action()`.
- Produces per-episode totals: success identity, chunk inference totals/count, and episode wall duration.

- [ ] Add a fake-server failing test showing D8 starts after eight executed actions and finishes while actions 8-15 continue.
- [ ] Add a failing boundary-miss test proving the client waits only at the stride boundary.
- [ ] Add failing tests for reset, episode end with an outstanding request, socket failure, and request ID mismatch.
- [ ] Add a regression test for baseline/CD replan queues at 32 actions.
- [ ] Run focused client tests and confirm failures.
- [ ] Refactor connection access behind a serialized remote predictor.
- [ ] Integrate `ACStreamController` for AC-Stream and keep the synchronous queue for baseline/CD.
- [ ] Record monotonic episode and communication timing without printing per episode.
- [ ] Run client tests and confirm they pass.

### Task 6: Recipes, Four-GPU Manager, and Launchers

**Files:**
- Modify: `examples/robotwin/configs/recipes/streamwam_robotwin_mot_wan22_5b.yaml`
- Create: `examples/robotwin/configs/recipes/streamwam_robotwin_ac_stream_wan22_5b.yaml`
- Create: `examples/robotwin/multigpu_rollout.py`
- Create: `examples/robotwin/workload.py`
- Create: `examples/robotwin/timing.py`
- Create: `examples/robotwin/scripts/launch_streamwam_robotwin_baseline_4gpu.sh`
- Create: `examples/robotwin/scripts/launch_streamwam_robotwin_cd_4gpu.sh`
- Create: `examples/robotwin/scripts/launch_streamwam_robotwin_ac_stream_4gpu.sh`
- Create: `tests/test_robotwin_multigpu_workload.py`
- Create: `tests/test_robotwin_multigpu_manager.py`
- Create: `tests/test_robotwin_timing.py`

**Interfaces:**
- Produces 100 unique one-trial jobs from 50 tasks times two configurations.
- Consumes arbitrary comma-separated GPU IDs and optional task/config filters.
- Starts one wyx inference server and one motus simulator worker per selected GPU.
- Produces `results.json` with episode identities, success totals, timing totals, worker logs, and runtime metadata.
- Prints one final summary with mode, GPUs, workload, success, chunk time, episode time, and results path.

- [ ] Add failing workload tests for uniqueness, complete 50x2 coverage, filters, deterministic ordering, and balanced/dynamic assignment.
- [ ] Add failing manager tests for separate Python binaries, ports, GPU propagation, worker failure, missing/duplicate results, and cleanup.
- [ ] Add failing timing tests for arithmetic chunk/episode means and compile/prewarm exclusion.
- [ ] Add launcher tests proving shell scripts contain only defaults plus one Python invocation and forward user arguments.
- [ ] Run focused manager/workload/timing tests and confirm failures.
- [ ] Implement workload construction, worker orchestration, result validation, aggregation, and concise output.
- [ ] Add MoT and AC-Stream recipes with exact dimensions and inference contracts.
- [ ] Add three launchers with one-trial defaults; AC-Stream defaults accelerated and accepts `--ac-stream-eager`.
- [ ] Run the complete RoboTwin unit-test subset and confirm it passes.

### Task 7: Reference Parity and End-to-End Smoke Verification

**Files:**
- Create: `tests/reference/robotwin_yzy_parity.py`
- Modify only if a confirmed mismatch is found: files from Tasks 1-6.

**Interfaces:**
- Consumes one fixed normalized observation, text context, proprio vector, seed, and the three original checkpoints.
- Produces comparison records for baseline, CD, RTC D0, and RTC D8 before action denormalization.

- [ ] Run static import/compile checks for every new Python module.
- [ ] Run all focused tests from Tasks 1-6.
- [ ] Run the existing full test suite and record unrelated pre-existing failures separately.
- [ ] Run baseline parity against yzy and compare normalized inputs, timesteps, masks, and action output.
- [ ] Run CD parity against yzy using exactly one consistency boundary.
- [ ] Run AC-Stream eager D0/D8 parity against yzy.
- [ ] Run accelerated D0/D8 parity against eager and confirm one graph with no unexplained recompiles or CUDA Graph skips.
- [ ] Run a single-task RoboTwin smoke evaluation for each mode before presenting four-card commands.
- [ ] Inspect `git diff --check` and `git status --short`; do not commit.

## Final Handoff

After all automated and smoke checks pass, report:

- every modified/created file grouped by responsibility;
- exact four-GPU, one-trial commands for baseline, CD, AC-Stream eager, and AC-Stream accelerated;
- which checks ran and their results;
- any remaining environmental limitation preventing a full simulator run;
- no Git commit hash, because all changes remain for manual review.
