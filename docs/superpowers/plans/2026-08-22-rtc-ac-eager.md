# RTC-AC Eager Inference Implementation Plan

> **For agentic workers:** implement this plan inline with test-driven development. Do not create Git commits; the repository owner will inspect and commit manually.

**Goal:** Add the original FastWAM Stage-2 selfatt-z1 step-5500 D0/D8 asynchronous RTC inference path to StreamWAM under the single public name `rtc_ac`, first with an eager backend.

**Architecture:** Keep standard `MoTWAM` and synchronous `consistency` unchanged. Add the checkpoint-specific three-stream model as `RTCACMoT`/`RTCACWAM`, place all D0/D8 sampling and asynchronous chunk scheduling in one `streamwam/inference/rtc_ac.py`, and let LIBERO select the path through `sampling_method=rtc_ac`.

**Tech Stack:** Python 3.10+, PyTorch, `ThreadPoolExecutor`, StreamWAM MoT/Wan2.2, pytest, LIBERO.

## Global Constraints

- Public inference name is exactly `rtc_ac` in config, modules, WAM, logs, and scripts.
- Keep `--checkpoint-format fastwam`; do not convert or copy checkpoint tensors.
- Exact checkpoint geometry is H=32, video frames=9, temporal compression=4, stride=16, delay=8, and launch-after=8.
- D0 has no clean action prefix; D8 has exactly eight clean prefix actions.
- Existing Euler and synchronous consistency behavior must remain unchanged.
- The first implementation is eager only; compile/CUDA Graph/KV-cache acceleration follows after accuracy validation.
- Do not make Git commits.

---

### Task 1: RTC-AC configuration and public mode

**Files:**
- Modify: `streamwam/config.py`
- Modify: `streamwam/inference/consistency.py`
- Create: `streamwam/inference/rtc_ac.py`
- Modify: `streamwam/inference/__init__.py`
- Test: `tests/test_rtc_ac.py`

**Interfaces:**
- Produces `validate_rtc_ac_geometry(...)`, `build_rtc_ac_prev_action_target(...)`, and normalized `sampling_method='rtc_ac'`.

- [ ] Write failing tests proving aliases do not disturb Euler/consistency and RTC-AC rejects any geometry other than H32/s16/d8/launch8/T9/compress4.
- [ ] Run `pytest -q tests/test_rtc_ac.py` and confirm failure because `rtc_ac` is absent.
- [ ] Add `framework.variant`, eager RTC-AC inference fields, normalization, constants, and validation.
- [ ] Run the focused tests and keep existing consistency tests green.

### Task 2: RTC-AC checkpoint-specific model structure

**Files:**
- Create: `streamwam/modules/rtc_ac.py`
- Modify: `streamwam/modules/__init__.py`
- Create: `streamwam/wam/rtc_ac_wam.py`
- Modify: `streamwam/wam/__init__.py`
- Modify: `streamwam/builder.py`
- Test: `tests/test_rtc_ac.py`

**Interfaces:**
- Produces `RTCACSlotEncoder`, `RTCACMoT`, and `RTCACWAM`.
- `RTCACWAM.infer_action(..., rtc_prev_action_chunk=None, rtc_inference_delay=0)` returns normalized `[1,32,7]` actions.

- [ ] Write failing slot-encoder tests with literal known/unknown masks and builder variant tests.
- [ ] Verify the tests fail on missing RTC-AC classes.
- [ ] Implement the 16-slot encoder, exact D0/D8 policy/condition masks, shared ActionDiT three-stream forward, and RTC-AC WAM construction.
- [ ] Implement one-step D0/D8 joint inference using the existing consistency boundaries and exact CPU RNG convention.
- [ ] Run focused tests, synchronous consistency tests, and action-timestep tests.

### Task 3: Strict Stage-2 FastWAM checkpoint loading

**Files:**
- Modify: `streamwam/checkpointing/fastwam_format.py`
- Test: `tests/test_checkpoint_formats.py`

**Interfaces:**
- `load_fastwam_checkpoint` detects RTC-AC payload metadata only to validate compatibility; source format stays `fastwam`.

- [ ] Write failing tests for phase mismatch, architecture mismatch, standard-model/RTC-checkpoint mismatch, and no-preload mutation.
- [ ] Verify failures occur before any target tensor changes.
- [ ] Validate exact phase, architecture, required slot tensor shape `[2,1024]`, and RTC-AC model type before strict state loading.
- [ ] Run checkpoint format tests.

### Task 4: D0/D8 asynchronous controller

**Files:**
- Modify: `streamwam/inference/rtc_ac.py`
- Test: `tests/test_rtc_ac.py`

**Interfaces:**
- Produces `RTCACController` that accepts a prediction callback and owns one executor/future.
- Controller exposes current action, launch/completion transitions, and deterministic cleanup.

- [ ] Write a real fake-predictor state-machine test: D0 blocks, actions 0-7 execute, D8 launches at cursor 8 with the aligned old-chunk target, and swaps at cursor 16.
- [ ] Write tests for block-on-miss, predictor exception propagation, and episode-close cleanup.
- [ ] Verify failures before implementing controller behavior.
- [ ] Implement the minimal thread/future state machine in `rtc_ac.py`.
- [ ] Run focused tests repeatedly to catch thread-lifecycle flakiness.

### Task 5: LIBERO RTC-AC integration

**Files:**
- Modify: `examples/libero/rollout.py`
- Create: `examples/libero/configs/recipes/streamwam_libero_rtc_ac_wan22_5b.yaml`
- Create: `examples/libero/scripts/launch_streamwam_libero_rtc_ac_rollout.sh`
- Test: `tests/test_rtc_ac.py`
- Test: `tests/test_inference_checkpoint_cli.py`

**Interfaces:**
- `sampling_method=rtc_ac` selects `_rollout_rtc_ac_episode`; all other methods retain `_rollout_episode`.
- Chunk prediction accepts optional normalized `rtc_prev_action_chunk` and returns both normalized model actions and denormalized environment actions.

- [ ] Write failing rollout tests proving the RTC-AC branch uses normalized actions for D8 prefixes while the environment executes denormalized actions.
- [ ] Add parser/config resolution and the dedicated RTC-AC episode loop.
- [ ] Preserve final-only global timing output and 30-FPS video behavior.
- [ ] Add the readable single-GPU launcher and validate it with `bash -n`.

### Task 6: Original checkpoint links and verification

**Files:**
- Create symlink: `checkpoints/fastwam_rtc_ac_step_005500.pt`
- Create symlink: exact dataset stats used by the reference evaluation, after resolving it from the reference launcher/config.

- [ ] Resolve and validate the reference checkpoint and stats targets with read-only checks.
- [ ] Create repository-local symlinks without copying model data.
- [ ] Inspect payload metadata and run a strict CPU load validation.
- [ ] Run scoped pytest, `git diff --check`, `bash -n`, and one-task launcher smoke test where the environment permits.
