# StreamWAM Text-First Research Blog Redesign

## Goal

Turn the current StreamWAM project page into a text-first research blog. The opening may retain the
existing editorial masthead, one real rollout visual, public links, and three headline results. After
that opening, the page becomes a sustained research narrative in which prose carries the argument and
figures support it.

This iteration changes the page hierarchy and draft prose. It does not change benchmark values,
evaluation protocols, model lineage, public links, release status, or the public StreamWAM method name.

## Editorial model

The page should read like a research group explaining a new idea to technically literate readers,
rather than like a product landing page or a visual project gallery. It uses a problem-driven story:

1. Establish why latency and waiting matter for WAM-based robot control.
2. Explain how sequential WAM control works and why streaming is attractive.
3. Show why inference/execution overlap alone leaves a modeling mismatch.
4. Introduce action-conditioned future-video generation as the central StreamWAM idea.
5. Explain how the model generates the next world-action chunk while the robot keeps acting.
6. Position the repository as a common testbed for streaming strategies.
7. Interpret the current evidence from LIBERO, RoboCasa, and RoboTwin without overstating it.

The voice is accessible academic English: concrete, explanatory, and technically accurate. Paragraphs
should normally contain a complete idea rather than acting as labels for surrounding cards.

## Page hierarchy

### Opening masthead

The opening remains visually engaging and may keep:

- `StreamWAM` and `Streaming World-Action Models for Robotic Manipulation`.
- A concise framework and action-conditioned streaming summary.
- Code, Models, and Paper `Coming Soon` status.
- One real successful rollout visual with a descriptive caption.
- Three headline numbers: LIBERO average success `98.20%`, LIBERO chunk time `41.0 ms`, and
  RoboCasa total time `11.76 s`.

The masthead is the only part of the page that behaves like a showcase. It should end cleanly and lead
into an article introduction.

### Long-form article

Everything after the masthead uses a centered reading column of approximately 760–820 px. The article
contains the following sections:

#### 1. Why Streaming World-Action Models?

Introduce the control setting and the basic tension between a model that needs time to generate a
future and a robot that should keep moving. Explain sequential infer-then-execute behavior in prose.
A single small timeline may illustrate waiting, but it must not replace the explanation.

#### 2. From Execution Overlap to a Modeling Problem

Explain that asynchronous scheduling can reduce visible waiting, but the world model may generate its
future without knowing which part of the action chunk is already being executed. Frame this as the
motivation for model-level coupling. Avoid claims that depend on an unreleased technical report.

#### 3. Action-Conditioned Future Generation

Present the core idea in several paragraphs: the currently executing action prefix is fed back to the
world model; future video generation is conditioned on that prefix; the predicted visual future and
next action chunk therefore reflect robot motion already underway. Keep one wide three-stage method
figure after the prose.

#### 4. Streaming While the Robot Acts

Explain the execution sequence in chronological prose, including the current chunk, executed prefix,
conditioned future, next chunk, and handoff. Retain one compact execution timeline. Remove latency and
video placeholder cards from the middle of the article; unreleased artifacts are mentioned once near
the end.

#### 5. A Common Testbed for Streaming Strategies

Describe StreamWAM as a research framework for comparing sequential execution, overlap-only streaming,
and action-conditioned streaming under common control and evaluation interfaces. Express the strategy
comparison primarily in paragraphs or a compact inline table, not three large feature cards.

#### 6. What the Current Results Show

Introduce model lineage and evaluation scope, then present LIBERO, RoboCasa, and RoboTwin in sequence.
Each benchmark receives:

- A short contextual paragraph explaining the protocol.
- The unchanged authoritative result table.
- One restrained interpretation derived directly from the reported values.

The result panels are no longer hidden behind tabs. All three tables remain visible in the document so
the article is complete without interaction or JavaScript.

#### 7. Where This Leaves Us

Close with prose covering three provisional conclusions: action conditioning connects ongoing execution
to visual prediction; streaming latency and control success should be evaluated together; expanded
analysis, limitations, and failure cases will accompany the technical report. Follow with compact Code,
Models, Paper, and rollout-film status plus acknowledgements.

## Text-to-visual balance

Prose should account for approximately 75–80 percent of the reading experience. The body retains only:

- One method figure.
- One compact execution timeline.
- Three benchmark tables.
- One optional three-image rollout strip after the quantitative results.

Remove or substantially simplify the current metric band repetition, sticky chapter rail, large strategy
cards, paired comparison cards, future-artifact placeholder cards, resource-card grid, and other blocks
that fragment the text. Small pull quotes, section numbers, and margin notes are allowed when they
highlight an argument already explained in prose.

## Navigation and responsive behavior

The slim sticky header remains. A compact table of contents may appear near the article introduction,
but it should not stay beside the reader or compete with the text. Desktop, tablet, and mobile all use
the same linear article order.

On mobile, prose remains single-column and tables scroll horizontally. The article must remain fully
readable with JavaScript disabled. JavaScript is limited to the mobile header menu and optional current-
section highlighting; no scientific content may depend on it.

## Content constraints

- Public method naming is always `StreamWAM`; do not publish `RTC-AC`, `AC-StreamWAM`, or `Ours`.
- Preserve every current benchmark value and protocol exactly.
- Preserve the stated lineage: StreamWAM starts from FastWAM-Joint and is further trained; RoboCasa
  builds on X-WAM; RoboTwin builds on StarWAM.
- Do not invent authors, affiliations, citation, venue, limitations, failure cases, ablations, or results.
- Paper and final rollout film remain `Coming Soon` and have no fake controls.
- Use complete draft prose; do not publish TODO, TBD, Lorem Ipsum, or editorial instructions.
- Keep all dependencies local; add no framework, build step, CDN, analytics, or remote font.

## Implementation boundaries

- `academic_project_page/index.html` owns the linear semantic article and public copy.
- `academic_project_page/styles.css` owns the editorial reading system and responsive tables.
- `academic_project_page/script.js` owns only progressive navigation behavior.
- Existing real rollout posters and the local social-preview image remain the only media assets.
- `.github/workflows/pages.yml` continues deploying only `academic_project_page/`.
- Training, inference, evaluation, checkpoint, and repository-root README code are out of scope.

## Verification

Before deployment:

- Add tests for the linear article order, visible benchmark sections, long-form prose, naming policy,
  exact benchmark data, local resources, no-JavaScript readability, and contrast.
- Validate JavaScript syntax and local references.
- Test desktop, intermediate, and mobile widths in a real browser.
- Confirm all three result tables are visible without JavaScript.
- Request independent review for scientific restraint, text-first hierarchy, accessibility, and scope.
- Commit only the academic page and its focused tests, push to `main`, and verify GitHub Pages online.

## Success criteria

The redesign succeeds when a reader can understand the motivation, distinction from overlap-only
streaming, action-conditioned generation mechanism, asynchronous execution sequence, evaluation setup,
and current evidence primarily by reading the article. Removing the figures should make the page less
illustrated but not conceptually incomplete.
