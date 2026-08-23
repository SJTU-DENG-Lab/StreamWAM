# StreamWAM Public README and Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a compact WaveForcing-style StreamWAM landing page and one reproducible Python 3.10/cu128 environment for the validated RTC-AC path.

**Architecture:** The root `pyproject.toml` is the canonical dependency source, with `uv` selecting PyTorch's cu128 index and `.python-version` recording the reproduced interpreter. The root README exposes the shortest public workflow and measured result, while `examples/libero/LIBERO.md` retains detailed benchmark guidance. RTC-AC launcher inputs become explicit environment variables so no private filesystem layout is embedded in public code.

**Tech Stack:** Python 3.10.20, uv, PyTorch 2.7.1/cu128, Triton 3.3.1, Bash, pytest, LIBERO.

## Global Constraints

- Do not create an environment filename containing `rtc-ac-cu128` or another internal benchmark suffix.
- Select Python 3.10 in `.python-version`, accept only Python 3.10 in package metadata, and record 3.10.20 as the benchmark interpreter in the README.
- Pin PyTorch 2.7.1 from the cu128 index, torchvision 0.22.1 from the cu128 index, and Triton 3.3.1.
- Keep TensorRT, diffusers, DeepSpeed, FlashAttention, and xFormers out of the default accelerated inference environment.
- Do not publish `/inspire/...`, the validated `wyx/FastWAM/.venv`, or another machine-specific absolute path.
- Keep legacy external checkpoint identifiers unchanged when they are the real public asset names.
- Report the supplied H100 result as a measured baseline, not as a universal guarantee.

---

### Task 1: Canonical Python Environment

**Files:**
- Create: `.python-version`
- Modify: `pyproject.toml`
- Delete: `examples/libero/requirements.txt`
- Create: `uv.lock`

**Interfaces:**
- Consumes: validated dependency versions in the approved design.
- Produces: `uv sync` as the single public environment installation command.

- [ ] **Step 1: Record the interpreter and dependency set**

Set `.python-version` to `3.10`. Update `pyproject.toml` to use `requires-python = ">=3.10,<3.11"`, exact validated default dependencies, and explicit `pytorch-cu128` sources for torch and torchvision. Preserve nonessential training tools only in a separate `train` extra.

- [ ] **Step 2: Remove the duplicate dependency source**

Delete `examples/libero/requirements.txt` after every dependency required for the validated inference environment is represented in `pyproject.toml`.

- [ ] **Step 3: Resolve and validate the environment metadata**

Run:

```bash
uv lock
uv lock --check
uv run --no-sync --with tomli python -c 'import tomli; tomli.load(open("pyproject.toml", "rb"))'
```

Expected: dependency resolution succeeds, the lock is current, and TOML parsing exits zero.

- [ ] **Step 4: Commit**

```bash
git add .python-version pyproject.toml uv.lock examples/libero/requirements.txt
git commit -m "build: pin the public StreamWAM environment"
```

### Task 2: Portable RTC-AC Launcher

**Files:**
- Modify: `tests/test_rtc_ac_acceleration.py`
- Modify: `examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh`

**Interfaces:**
- Consumes: `BACKBONE_PATH`, `LIBERO_HOME_PATH` or `LIBERO_HOME`, `CHECKPOINT_PATH`, `STATS_PATH`, optional `PYTHON_BIN`, and optional `GPU_IDS`.
- Produces: an early actionable error when required public paths are absent; otherwise preserves the existing multigpu rollout command and forwards user CLI arguments.

- [ ] **Step 1: Write the failing launcher-contract test**

Add a subprocess test that executes the launcher with all required path variables removed and asserts a nonzero exit plus an error requesting `BACKBONE_PATH`. This catches any reintroduction of a private default path before Python/model startup.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_rtc_ac_acceleration.py -k launcher_requires_public_paths -v
```

Expected: FAIL because the current launcher silently supplies a private default.

- [ ] **Step 3: Implement explicit public path inputs**

Replace private defaults with Bash required-variable checks. Resolve `LIBERO_HOME_PATH` from `LIBERO_HOME` when only the standard variable is set. Replace hard-coded checkpoint and stats arguments with `CHECKPOINT_PATH` and `STATS_PATH`, leaving `GPU_IDS=0,1,2,3` and `PYTHON_BIN=python` as portable defaults.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the focused pytest command again. Expected: PASS, with no model import or GPU initialization.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rtc_ac_acceleration.py examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh
git commit -m "fix: make the RTC-AC launcher portable"
```

### Task 3: Public Documentation

**Files:**
- Modify: `README.md`
- Modify: `examples/libero/LIBERO.md`

**Interfaces:**
- Consumes: `uv sync`, the launcher variables from Task 2, existing public model identifiers, and the validated result supplied by the user.
- Produces: a short public landing page plus a consistent detailed LIBERO setup guide.

- [ ] **Step 1: Rewrite the root README**

Use the approved order: centered title/badges, concise introduction, release status, Quick Start, current results, accelerated runtime contract, runtime layout, citation, license, and acknowledgements. Include the exact four-GPU launch command using public placeholder variables and the 45.20/45.75/46.50 ms measurements.

- [ ] **Step 2: Align the detailed LIBERO guide**

Replace the Python 3.11/cu124/FlashAttention environment section with `uv sync`, document the external LIBERO source layout, replace the private `wyx` validation command with the public launcher contract, update the acceptance range to the supplied 40–46 ms baseline, and remove references to the deleted requirements file.

- [ ] **Step 3: Validate links, paths, and public naming**

Run:

```bash
test -f examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh
test -f examples/libero/configs/recipes/streamwam_libero_rtc_ac_wan22_5b.yaml
test ! -e examples/libero/requirements.txt
! rg '/inspire/|wyx/FastWAM/.venv|launch_starwam|requirements\.txt' README.md examples/libero/LIBERO.md examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh
```

Expected: every command exits zero.

- [ ] **Step 4: Commit**

```bash
git add README.md examples/libero/LIBERO.md
git commit -m "docs: publish the StreamWAM quick start"
```

### Task 4: Repository Verification and Review

**Files:**
- Verify: all files changed by Tasks 1–3.

**Interfaces:**
- Consumes: completed environment, launcher, and documentation commits.
- Produces: a clean, reviewed branch ready to push.

- [ ] **Step 1: Run automated verification**

```bash
uv lock --check
pytest -q
bash -n examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh
git diff --check origin/main..HEAD
```

Expected: all commands pass.

- [ ] **Step 2: Review the final repository delta**

Inspect `git diff --stat origin/main..HEAD`, `git diff origin/main..HEAD`, and `git status --short --branch`. Confirm only the approved design, plan, environment, launcher, tests, lockfile, and docs changed.

- [ ] **Step 3: Request independent code review**

Provide the reviewer with the approved spec, this plan, the `origin/main` base SHA, and final HEAD SHA. Fix every Critical or Important issue and rerun Step 1.

- [ ] **Step 4: Report completion**

Summarize files changed, validation evidence, commits created, and whether the branch is ahead of `origin/main`. Do not push without an explicit push request.
