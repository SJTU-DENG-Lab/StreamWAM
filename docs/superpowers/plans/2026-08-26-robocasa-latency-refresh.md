# RoboCasa Latency Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the new RoboCasa runtime measurements across the project page, README, source generator, generated charts, speedup copy, and regression tests.

**Architecture:** Keep `docs/generate_latency_figure.py` as the authoritative chart source. Tests first encode the new tuples and public text, then the generator and text surfaces are updated, the PNGs are regenerated mechanically, and byte equality confirms the committed assets match their source.

**Tech Stack:** Python, matplotlib, NumPy, Pillow, static HTML, Markdown, pytest.

## Global Constraints

- X-WAM: `374.07 ms`, `17.36 s`.
- X-WAM-CD: `134.37 ms`, `13.04 s`.
- Stream-WAM: `115.98 ms`, `9.49 s`.
- Public method names remain X-WAM, X-WAM-CD, and Stream-WAM.
- RoboCasa speedups are `3.2×` for chunk latency and `1.8×` for end-to-end time.
- RoboCasa chart ceilings are `410 ms` and `20 s`.
- LIBERO, RoboTwin, task-success results, chart layout, colors, and dimensions remain unchanged.
- Commit and push the completed refresh to `origin/main`.

---

### Task 1: Refresh RoboCasa runtime data and generated figures

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/generate_latency_figure.py`
- Modify: `docs/index.html`
- Modify: `README.md`
- Modify: `docs/assets/stream-wam-chunk-time.png`
- Modify: `docs/assets/stream-wam-episode-time.png`

**Interfaces:**
- Consumes: `ROBOCASA_CHUNK`, `ROBOCASA_EPISODE`, the generator's three-method panel, the hidden latency table, and the results speedup paragraph.
- Produces: New authoritative runtime tuples and checked-in 2400×900 figures generated from them.

- [ ] **Step 1: Write failing data, prose, README, and asset tests**

Update literal expectations to require:

```python
assert module.ROBOCASA_CHUNK == (374.07, 134.37, 115.98)
assert module.ROBOCASA_EPISODE == (17.36, 13.04, 9.49)
```

Require the accessible latency table and README to expose the same three rows, require `3.2× on RoboCasa` in the chunk claim and `1.8× on RoboCasa` in the end-to-end claim, and reject obsolete values `504.00`, `37.31`, `135.21`, `33.60`, `136.76`, and `11.76` in current runtime surfaces.

Extend the temporary-render test so each generated PNG must be byte-identical to its corresponding checked-in asset.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'latency or benchmark_results or narrative or readme'
```

Expected: FAIL on the old tuples, old accessible table, old README runtime cells, and old RoboCasa speedup claims.

- [ ] **Step 3: Update authoritative source and text surfaces**

In `docs/generate_latency_figure.py`, set:

```python
ROBOCASA_CHUNK = (374.07, 134.37, 115.98)
ROBOCASA_EPISODE = (17.36, 13.04, 9.49)
```

Set only the RoboCasa chunk panel ceiling to `410` and episode panel ceiling to `20`. Update the `docs/index.html` hidden rows and speedup sentence plus the three README runtime rows. Do not change task-success values.

- [ ] **Step 4: Regenerate checked-in PNGs**

Run:

```bash
python docs/generate_latency_figure.py --output-dir docs/assets
```

Expected: both 2400×900 PNG files change and contain the new RoboCasa annotations.

- [ ] **Step 5: Verify focused and full tests**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'latency or benchmark_results or narrative or readme'
pytest -q tests/test_academic_project_page.py
git diff --check
```

Expected: all commands pass.

- [ ] **Step 6: Inspect figures, review, commit, and push**

Visually inspect both generated PNGs, request a read-only review, and confirm no LIBERO, RoboTwin, or task-success data changed. Then commit and push:

```bash
git add README.md docs/index.html docs/generate_latency_figure.py docs/assets/stream-wam-chunk-time.png docs/assets/stream-wam-episode-time.png tests/test_academic_project_page.py
git commit -m "fix: refresh RoboCasa latency results"
git push origin main
```
