# Joint CD 1-Step Inference Implementation Plan

> **For agentic workers:** Execute inline, task by task, using test-driven development. Do not commit; the user will inspect and commit manually.

**Goal:** Run the original FastWAM Joint CD step-3400 checkpoint in StarWAM with synchronous one-step consistency inference and LIBERO replanning every 16 steps.

**Architecture:** Keep checkpoint format and sampling semantics separate. Reuse `MoTWAM` and its full-video joint attention path, add focused consistency-boundary functions, extend ActionDiT's public timestep contract, and expose the sampler through the existing rollout interface.

**Tech Stack:** Python, PyTorch, pytest, YAML, Bash.

## Global Constraints

- Existing Euler inference remains the default and retains its behavior.
- No checkpoint conversion or copied checkpoint weights.
- No RTC, asynchronous controller, compilation, CUDA Graph, or cache implementation.
- Do not create a Git commit.

---

### Task 1: Consistency boundary functions

**Files:**
- Create: `tests/test_consistency_sampling.py`
- Create: `starwam/inference/__init__.py`
- Create: `starwam/inference/consistency.py`

**Interfaces:**
- Produce: `action_consistency_boundary(sample, velocity, sigma) -> Tensor`
- Produce: `video_consistency_boundary(sample, velocity, sigma, sigma_data=0.5) -> Tensor`

- [ ] Write literal-value tests for action and video boundaries, scalar and token-wise sigma broadcasting, and dtype preservation.
- [ ] Run `pytest -q tests/test_consistency_sampling.py` and verify import failure.
- [ ] Implement broadcasting and both boundary equations without scheduler state.
- [ ] Run the focused test and verify it passes.

### Task 2: Token-wise ActionDiT timestep support

**Files:**
- Create: `tests/test_action_dit_timesteps.py`
- Modify: `starwam/modules/action_dit.py`

**Interfaces:**
- Consume: `ActionDiT.pre_dit(action_tokens, timestep, context, context_mask)`
- Produce: support for timestep shapes `[B]`, `[1]`, `[B,H]`, and inference-only `[1,H]` expansion.

- [ ] Write tests using a small real ActionDiT that assert batch timestep produces `[B,6,D]`, token timestep produces `[B,H,6,D]`, and malformed shapes raise `ValueError`.
- [ ] Run the focused test and verify the token-wise cases fail against current reshape behavior.
- [ ] Add explicit shape validation and the two embedding branches while preserving the current batch branch.
- [ ] Run the focused test and existing action tests.

### Task 3: MoTWAM consistency dispatch

**Files:**
- Modify: `tests/test_consistency_sampling.py`
- Modify: `starwam/wam/mot_wam.py`

**Interfaces:**
- Consume: consistency boundary functions from Task 1.
- Produce: `sampling_method` support in `infer_action` and `infer_joint`, defaulting to `euler`.

- [ ] Add focused tests that exercise a minimal MoTWAM inference loop and distinguish Euler output from consistency output using hand-derived values.
- [ ] Add validation tests for an unknown method, non-one-step consistency, horizon other than 32, and conditioning other than `full_video`.
- [ ] Run the tests and verify each new behavior fails for the intended missing branch.
- [ ] Add a private method normalizer and geometry validator, build token-wise consistency action timesteps, and dispatch only the latent update through the new boundary functions.
- [ ] Keep the existing Euler statements unchanged and run the focused tests.

### Task 4: LIBERO command surface and dedicated recipe

**Files:**
- Modify: `tests/test_inference_checkpoint_cli.py`
- Modify: `examples/libero/rollout.py`
- Create: `examples/libero/configs/recipes/starwam_libero_joint_cd_wan22_5b.yaml`

**Interfaces:**
- Produce: `--sampling-method {euler,consistency}`, default `euler`.
- Produce: recipe with `chunk_size=32`, `action_video_conditioning=full_video`, `num_inference_steps=1`, and `sampling_method=consistency`.

- [ ] Add parser/default and forwarding tests, then verify failure before production changes.
- [ ] Add the CLI option, resolve it from CLI or recipe, log it, and pass it through `_predict_action_chunk`.
- [ ] Create the dedicated recipe by copying only the required existing FastWAM-aligned fields and changing the Joint CD inference fields.
- [ ] Run CLI and config-loading tests.

### Task 5: Checkpoint link and clean launcher

**Files:**
- Create symlink: `checkpoints/fastwam_joint_cd_step_003400.pt`
- Create: `examples/libero/scripts/launch_starwam_libero_joint_cd_rollout.sh`

**Interfaces:**
- Symlink target: `/inspire/hdd/project/qproject-fundationmodel/yangyi-253108120173/wyx/workspace/Dual-Streaming-World-Action-Model/outputs/cd_fastwam_joint_1step_bs1_ga8_20260807_080431/checkpoints/weights/step_003400.pt`
- Launcher: a flat `python examples/libero/rollout.py ...` command with consistency, one step, horizon from recipe, and `replan_steps=16`.

- [ ] Resolve the real source checkpoint and ensure the target is a regular file.
- [ ] Create the relative or absolute symbolic link without copying bytes.
- [ ] Add the standalone launcher using the existing locally configured backbone and LIBERO roots; use the Joint CD run's `dataset_stats.json` via a second symlink if needed by the command.
- [ ] Run `bash -n` and `readlink -f`; run `rollout.py --help` to validate the new option.

### Task 6: Regression and parity validation

**Files:**
- Modify only tests if a real uncovered regression is found.

- [ ] Run all focused tests for checkpoint formats, CLI, consistency, and ActionDiT.
- [ ] Run the complete `pytest -q` suite in the LIBERO environment.
- [ ] Run `git diff --check`, inspect `git diff --stat`, and confirm nothing is staged or committed.
- [ ] If GPU time permits, run one Joint CD LIBERO trial; otherwise report the exact unexecuted GPU validation command.
