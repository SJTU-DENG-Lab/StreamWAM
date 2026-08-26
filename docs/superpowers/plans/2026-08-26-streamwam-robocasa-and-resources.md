# Stream-WAM RoboCasa and Resources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the RoboCasa evaluation presentation, identify Stream-WAM as the authors' method in visible performance tables, and replace the final project-page boilerplate with concise open-source and citation content.

**Architecture:** Keep the existing static-page structure and benchmark parser tests. Update the HTML and README as the source of truth, encode the exact visible table rows and resource semantics in focused regression tests, and make only the CSS additions needed for inline resource links and the BibTeX panel.

**Tech Stack:** Static HTML, CSS, Markdown, Python `pytest`, standard-library `html.parser`.

## Global Constraints

- RoboCasa uses the standard 24-task protocol with 50 trials per task and reports average success.
- The RoboCasa rows and values must exactly match the approved seven-row table.
- `Stream-WAM (Ours)` appears only in the three visible task-performance tables, not in hidden latency data, figures, navigation, prose, metadata, or resource names.
- The hardware sentence must say `Our evaluations use four NVIDIA H100 GPUs`.
- The Resources section contains only the approved open-source sentence and exact BibTeX citation; no resource strip, model-lineage paragraph, acknowledgements, copy button, or JavaScript.
- Preserve unrelated title and hero work from the other agent.
- Do not push these changes until the user asks for the combined push.

---

### Task 1: Correct benchmark protocol and task-performance tables

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html`
- Modify: `README.md`

**Interfaces:**
- Consumes: The existing `parse_benchmarks()` test helper and static benchmark table markup.
- Produces: Exact visible LIBERO, RoboTwin 2.0, and RoboCasa rows; corrected 24-task protocol copy in HTML and README.

- [ ] **Step 1: Write failing benchmark regression tests**

Update the expected benchmark content to require:

```python
assert benchmarks["benchmark-robocasa"]["headers"] == ["Method", "Average Success ↑"]
assert benchmarks["benchmark-robocasa"]["rows"] == [
    ["π₀.₅", "41.4%"],
    ["π₀-FAST", "61.2%"],
    ["π₀", "62.5%"],
    ["Cosmos Policy", "67.1%"],
    ["X-WAM", "75.42%"],
    ["X-WAM-CD", "75.33%"],
    ["Stream-WAM (Ours)", "75.35%"],
]
```

Require `Stream-WAM (Ours)` in the visible LIBERO and RoboTwin row expectations, require `24 kitchen manipulation tasks`, `50 trials per task`, and `average success` in the RoboCasa section, and require the README to describe the same protocol. Update the caption expectation to `RoboCasa 24-task average success results` and the hardware wording expectation to `Our evaluations use four NVIDIA H100 GPUs`.

- [ ] **Step 2: Run focused tests to verify they fail**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'benchmark or research_story'
```

Expected: FAIL because the page still has the old 50-task copy, three-row RoboCasa table, old caption/header, and unmarked Stream-WAM rows.

- [ ] **Step 3: Implement the benchmark and protocol changes**

In `docs/index.html`:

- Change the hardware sentence to `Our evaluations use four NVIDIA H100 GPUs.`
- Rename the Stream-WAM method cell in the three visible task-performance tables to `Stream-WAM (Ours)`.
- Describe RoboCasa as 24 kitchen manipulation tasks with 50 trials per task and average success, linking the supplied SOTA2 leaderboard for the external published references.
- Change the caption to `RoboCasa 24-task average success results` and the value header to `Average Success ↑`.
- Insert the approved seven rows in their exact order and mark X-WAM as best plus Stream-WAM as second best.
- Leave all hidden latency-table method names unchanged.

In `README.md`, replace the 50-target-task protocol with the standard 24 kitchen manipulation tasks, 50 trials per task, and average success.

- [ ] **Step 4: Run focused tests to verify they pass**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'benchmark or research_story'
```

Expected: PASS.

- [ ] **Step 5: Commit the benchmark correction**

```bash
git add tests/test_academic_project_page.py docs/index.html README.md
git commit -m "fix: correct RoboCasa benchmark presentation"
```

---

### Task 2: Simplify resources and add the citation block

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: Existing `#resources` section and its reading-column layout.
- Produces: One linked open-source sentence and one semantic, scrollable BibTeX citation block.

- [ ] **Step 1: Write a failing resources regression test**

Add a test that scopes assertions to the Resources section and requires:

```python
assert "Open source." in resources_html
assert ">Citation</h3>" in resources_html
assert 'href="https://github.com/SJTU-DENG-Lab/StreamWAM"' in resources_html
assert "huggingface.co" in resources_html
assert "@misc{denglab2026streamwam," in resources_html
assert "howpublished = {Project page}" in resources_html
assert "organization = {Shanghai Jiao Tong University}" in resources_html
assert 'url          = {https://sjtu-deng-lab.github.io/StreamWAM/}' in resources_html
```

Also assert that the scoped markup no longer contains `resource-links`, `Model lineage.`, `Acknowledgements.`, `Paper · Coming Soon`, or `Rollout film · Coming Soon`; assert the stylesheet no longer contains `.resource-links` or `.acknowledgements` and does contain citation-block overflow styling.

- [ ] **Step 2: Run the resources test to verify it fails**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k resources
```

Expected: FAIL because the old link strip, lineage, and acknowledgements are still present and there is no Citation block.

- [ ] **Step 3: Implement concise resources markup and styles**

Replace the section body with:

```html
<p class="lead"><strong>Open source.</strong> Stream-WAM training and inference code and evaluation recipes are available on <a href="https://github.com/SJTU-DENG-Lab/StreamWAM">GitHub</a>, with released checkpoints hosted on <a href="https://huggingface.co/SJTU-DENG-Lab">Hugging Face</a>.</p>
<div class="citation-block" aria-labelledby="citation-title">
  <h3 id="citation-title">Citation</h3>
  <pre><code>@misc{denglab2026streamwam,
  title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/StreamWAM/}
}</code></pre>
</div>
```

Retain `.resources` background styling, remove `.resource-links` and `.acknowledgements`, style inline links like existing editorial links, give the Citation heading compact spacing, and style `pre` with a paper background, border, radius, readable monospace font, and `overflow-x: auto`.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k resources
pytest -q tests/test_academic_project_page.py
```

Expected: Both commands PASS.

- [ ] **Step 5: Review scope and commit**

Run:

```bash
git diff --check
git diff -- docs/index.html docs/styles.css README.md tests/test_academic_project_page.py
```

Confirm the diff contains no unrelated hero/title changes, then commit:

```bash
git add tests/test_academic_project_page.py docs/index.html docs/styles.css
git commit -m "refactor: simplify resources and add citation"
```
