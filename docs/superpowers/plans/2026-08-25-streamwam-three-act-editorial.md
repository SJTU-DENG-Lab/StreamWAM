# StreamWAM Three-Act Editorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the body’s large chapter hierarchy with a detailed three-act argument covering WAM capability and latency, asynchronous continuity methods, and StreamWAM’s action-conditioned visual generation.

**Architecture:** Preserve the masthead and complete benchmark tables. Rewrite the pre-experiment body as three ordered semantic sections with small labels, original prose, one inline paper reference, and a single subordinate method figure; simplify CSS so the acts read as one article rather than separate display regions.

**Tech Stack:** Semantic HTML5, responsive CSS, vanilla JavaScript, Python `pytest`, GitHub Pages.

## Global Constraints

- Preserve every benchmark value, protocol, model-lineage statement, public link, and release status.
- Use `StreamWAM` as the public method name and keep internal runtime names absent.
- Attribute the inference-time RTC platform result to arXiv:2608.01880 rather than generalizing it.
- Describe prefix conditioning as effective for action continuation while distinguishing explicit action-conditioned future-video generation.
- Map `w/o Action Conditioning` to the page’s ablation accurately without claiming equivalence to every prefix-conditioned method.
- Use original prose and no verbatim quotation from the reference paper.
- Keep the masthead, three visible benchmark tables, Paper `Coming Soon`, and rollout-film `Coming Soon`.
- Add no framework, build step, CDN, analytics, backend, or remote font.

---

### Task 1: Lock the Three-Act Narrative Contract

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `PageParser`, `parse_page()`, and the static article.
- Produces: regression checks for act order, prose density, citation, and body-heading scale.

- [ ] **Step 1: Add a failing act-order and prose-density test**

```python
def test_article_opens_with_three_detailed_editorial_acts() -> None:
    parser, html = parse_page()
    act_ids = [
        attrs["id"] for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id", "").startswith("act-")
    ]
    assert act_ids == ["act-wam", "act-async", "act-streamwam"]
    assert parser.pre_experiment_paragraph_count >= 18
    assert 'href="https://arxiv.org/abs/2608.01880"' in html
```

- [ ] **Step 2: Add a failing hierarchy test**

Require the three acts to use `.act-label` and `.act-opening` rather than display-sized `h2` elements, while keeping the experiments heading and benchmark `h3` elements.

- [ ] **Step 3: Run the focused suite and observe the expected failure**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: the new test fails because the current article has seven large pre-experiment chapters and no `act-*` sections.

---

### Task 2: Rewrite the Body into Three Detailed Acts

**Files:**
- Modify: `academic_project_page/index.html`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: current masthead, method figure, experiment tables, closing resources, and arXiv source framing.
- Produces: `#act-wam`, `#act-async`, and `#act-streamwam` followed by the existing experiments.

- [ ] **Step 1: Keep the masthead unchanged**

Preserve its title, summaries, links, rollout frame, and three headline results exactly.

- [ ] **Step 2: Write Act I in six substantial paragraphs**

Explain visually grounded WAM prediction, joint video–action generation, iterative denoising cost, full deployment-pipeline latency, synchronous pauses or stale actions, and why useful execution should overlap inference.

- [ ] **Step 3: Write Act II in seven substantial paragraphs**

Explain asynchronous overlap, inter-chunk disagreement, hard-switch discontinuity, temporal alignment, inference-time RTC, prefix-conditioned continuation, and the remaining missing condition on generated video. Link arXiv:2608.01880 at the first detailed strategy comparison.

- [ ] **Step 4: Write Act III in six substantial paragraphs**

Explain how StreamWAM retains asynchronous and prefix-conditioned continuation, feeds the executing action prefix into the world-model path, conditions future-video generation, and jointly prepares the next world-action chunk. Explain the `w/o Action Conditioning` ablation without overgeneralizing it.

- [ ] **Step 5: Retain one subordinate method figure**

Place the existing three-stage action-prefix → conditioned-video → next-chunk figure after Act III. Remove the separate execution timeline and common-testbed display chapter; integrate their useful explanations into the three acts.

- [ ] **Step 6: Preserve experiments and closing content**

Keep all three benchmark sections, tables, protocols, interpretations, discussion, resources, lineage, and acknowledgements unchanged except for transitions required by the new article flow.

---

### Task 3: Reduce Visual Segmentation

**Files:**
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/README.md`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `.editorial-act`, `.act-label`, `.act-opening`, and `.method-figure` from Task 2.
- Produces: one continuous reading surface with modest labels and no display-scale act headings.

- [ ] **Step 1: Style acts as continuous prose**

Use a shared paper background, 760–820 px measure, 17–19 px body copy, 1.75–1.85 line height, and 70–95 px transitions. Give `.act-label` metadata scale and `.act-opening` a modest serif opening sentence rather than a display heading.

- [ ] **Step 2: Remove obsolete chapter presentation**

Delete or stop using styles for the previous motivation, overlap, execution timeline, testbed chapter, blockquote, and strongly separated pre-experiment sections.

- [ ] **Step 3: Update maintenance documentation**

Describe the three-act order, the inline arXiv attribution, and the requirement that the pre-experiment argument remain prose-led.

- [ ] **Step 4: Run verification**

Run `pytest -q tests/test_academic_project_page.py`, `node --check academic_project_page/script.js`, and `git diff --check`; all must exit successfully.

---

### Task 4: Source Review, Browser QA, and Deployment

**Files:**
- Modify if required: `academic_project_page/index.html`
- Modify if required: `academic_project_page/styles.css`
- Modify: `academic_project_page/assets/streamwam-social-preview.jpg` only if the masthead rendering changes.

**Interfaces:**
- Consumes: the completed three-act article.
- Produces: a source-faithful, responsive, reviewed, and deployed public page.

- [ ] **Step 1: Verify source fidelity**

Compare every RTC and prefix-conditioning statement against arXiv:2608.01880v2. Confirm platform-specific findings are attributed and no source sentence is reproduced verbatim.

- [ ] **Step 2: Browser-test three widths**

Inspect 1440 px, 1100 px, and 390 px. Confirm the three acts read continuously, paragraph measure is comfortable, act labels are subordinate, all tables remain visible, and there is no page overflow or console error.

- [ ] **Step 3: Request independent review**

Ask the reviewer to check scientific scope, paper attribution, the prefix/action-conditioned-video distinction, text density, exact benchmarks, accessibility, and dirty-worktree boundaries. Fix every Critical or Important issue.

- [ ] **Step 4: Commit only focused files**

```bash
git add academic_project_page tests/test_academic_project_page.py
git commit -m "feat: rewrite project story as three-act editorial"
```

- [ ] **Step 5: Push and verify Pages**

Push `main`, wait for the Pages workflow to complete successfully, then verify the live act order, arXiv link, visible tables, theme, and browser console.
