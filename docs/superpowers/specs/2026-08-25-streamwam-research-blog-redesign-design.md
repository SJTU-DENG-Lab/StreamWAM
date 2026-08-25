# StreamWAM Research Blog Redesign

## Goal

Redesign the existing StreamWAM academic project page as a bright, long-form research blog. The page should be capable of carrying substantial technical writing later, while the current preview remains concise, truthful, and useful before the paper and final rollout videos are ready.

The redesign changes presentation and narrative structure, not benchmark values, public links, method naming, or release status.

## Editorial direction

The page follows a research-story structure rather than a product landing page or a paper rendered as HTML. It leads readers through the motivation, framework, method, execution model, evidence, and open questions.

The visual tone is a bright academic editorial:

- Warm white paper background with subtle cool-gray section surfaces.
- Charcoal typography for long-form reading.
- Teal is the primary method and navigation accent.
- Muted orange is reserved for observations and `Coming Soon` states.
- System fonts keep the site self-contained and fast.
- Images are brighter and more natural than the current desaturated cinematic treatment.
- Fine rules, figure numbers, margin notes, and caption typography create the feeling of a carefully edited research article.

Dark backgrounds are limited to small technical diagrams or image overlays where they materially improve contrast. There is no full-width dark section.

## Reading model

The document has three coordinated navigation layers:

1. A slim sticky header provides project identity and Code, Models, and Paper status.
2. A compact chapter rail appears beside the article on wide screens and tracks the current chapter. On narrow screens it becomes a horizontal chapter index below the hero.
3. Each major section starts with a numbered chapter label and a one-sentence conclusion, so readers can skim the full argument without reading every paragraph.

The main prose measure is approximately 760–820 px. Figures, timelines, galleries, and result tables may break out to a wider 1180–1240 px canvas. Anchor links remain stable for `motivation`, `testbed`, `method`, `execution`, `experiments`, `discussion`, and `resources`.

## Narrative architecture

### 00 — Abstract

The opening uses a restrained editorial masthead rather than an application-style hero. It contains:

- StreamWAM title and `Streaming World-Action Models for Robotic Manipulation` subtitle.
- The existing framework and action-conditioned streaming summary.
- Active Code and Models links plus Paper `Coming Soon`.
- One wide real rollout image with a descriptive caption and `Rollout film · Coming Soon` status.
- Three compact headline results: LIBERO average success `98.20%`, LIBERO chunk time `41.0 ms`, and RoboCasa total time `11.76 s`.

### 01 — Why Streaming WAM?

This chapter explains the research problem in provisional preview language:

- World-action models must predict sufficiently far ahead to support control.
- Sequential inference followed by execution introduces avoidable waiting.
- Merely overlapping the two processes does not ensure that the predicted visual future reflects actions already underway.

A two-lane timeline contrasts sequential inference/execution with streaming overlap. The text is a concise research explanation, not a claim of paper completeness.

### 02 — A Unified Streaming Testbed

This chapter positions StreamWAM as a framework for systematically comparing streaming strategies under common evaluation protocols. It uses a small comparison matrix for three conceptual strategy families:

- Sequential WAM.
- Execution overlap without action-conditioned visual generation.
- StreamWAM with action-conditioned streaming.

The matrix communicates information flow and execution behavior without inventing unreported numerical results.

### 03 — Action-Conditioned Streaming

This is the visual and conceptual center of the article. A wide three-stage figure shows:

1. The action prefix currently being executed.
2. Future video generation explicitly conditioned on that action prefix.
3. Asynchronous prediction of the next world-action chunk.

The chapter explicitly distinguishes model-level coupling from a systems-only optimization. A highlighted `Key idea` note contains the shortest accurate explanation of the method for skimming and future citation.

### 04 — Streaming While Acting

A full-width execution timeline explains that the robot continues executing the current chunk while the model prepares the next one. The existing D0/D8 or internal runtime terminology is not published. The visualization uses generic `current chunk`, `executed prefix`, `conditioned future`, and `next chunk` labels.

This chapter reserves a figure slot for a future quantitative latency breakdown and a video slot for the final rollout film. Both are marked `Coming Soon` without fake controls or fabricated data.

### 05 — Experiments

The current result tables remain authoritative and unchanged. The chapter begins with evaluation context and model lineage, then provides accessible benchmark tabs for:

- LIBERO: four suites, 10 tasks per suite, 50 trials per task.
- RoboCasa: 50 target tasks and 50 trials per task.
- RoboTwin 2.0: 50 tasks and 100 rollout episodes per task, reporting Clean and Random.

Each benchmark panel opens with a short conclusion derived only from existing values. StreamWAM rows retain the same typography and background as other rows. The gallery of three real LIBERO posters follows the quantitative results as qualitative evidence.

### 06 — Discussion

The preview includes three explicitly provisional notes:

- Action conditioning links ongoing execution to visual prediction.
- Streaming performance must be considered jointly with control success.
- Broader analysis, limitations, and failure cases will accompany the technical report.

The chapter does not manufacture limitations or unpublished claims. Paper and expanded analysis remain `Coming Soon`.

### 07 — Resources

The closing contains Code, Models, Paper status, rollout-film status, model-lineage notes, and the existing acknowledgements. Citation and authors remain excluded until the paper release.

## Draft-content policy

The first blog version uses complete sentences rather than placeholder Latin text. Every provisional section must be visibly useful on its own but remain easy to replace:

- No `TODO`, `TBD`, or Lorem Ipsum appears publicly.
- Unreleased artifacts use `Coming Soon`.
- Unfinished scientific discussion is described as preview context, not as a final paper claim.
- No invented author list, citation, venue, method result, limitation, or comparison is added.
- Internal names such as `RTC-AC` and `AC-StreamWAM` remain absent; the method is always `StreamWAM`, with action conditioning emphasized in prose.

## Responsive behavior

Desktop uses a reading column plus a sticky chapter rail. Tablet collapses the rail into a horizontal index. Mobile presents a single-column article with full-width figures, compact metadata, horizontally scrollable tables, and an accessible menu.

The article remains readable without JavaScript. JavaScript progressively enhances chapter tracking, compact navigation, and result tabs. Reduced-motion settings disable animated scrolling and nonessential transitions.

## Implementation boundaries

The existing static architecture remains:

- `academic_project_page/index.html` owns semantic article structure and copy.
- `academic_project_page/styles.css` owns the bright editorial system and responsive layout.
- `academic_project_page/script.js` owns progressive navigation, chapter tracking, and result tabs.
- Existing poster and social-preview assets remain local; the social preview is regenerated to match the light visual identity.
- `.github/workflows/pages.yml` continues deploying only `academic_project_page/`.

No framework, build step, CDN, remote font, analytics system, backend, or content management system is introduced.

## Verification

Before deployment:

- Update automated tests for the new chapter anchors and public draft-content policy.
- Preserve all benchmark, protocol, public-link, deployment-scope, contrast, and no-JavaScript assertions.
- Validate JavaScript syntax and every local reference.
- Serve the page over HTTP and test desktop and mobile layouts in a real browser.
- Verify chapter navigation, result tabs, reduced-motion behavior, and absence of browser errors.
- Request independent review for content accuracy, accessibility, responsive behavior, and deployment scope.
- Stage only redesign-related files in the dirty worktree, push to `main`, and monitor GitHub Pages through successful deployment and live-page smoke testing.

## Out of scope

- Final paper prose, authors, affiliations, citation, BibTeX, venue, or PDF.
- Final rollout videos, latency-breakdown figure, failure-case study, or new experimental values.
- Changes to training, inference, evaluation, checkpoints, or repository README.
- A theme switcher or retained dark-mode variant in this first redesign.
