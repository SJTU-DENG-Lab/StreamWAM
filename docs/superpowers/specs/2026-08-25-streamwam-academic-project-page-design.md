# StreamWAM Academic Project Page Design

## Purpose

Create a public preview page for StreamWAM that explains the project before the technical report and final videos are released. The page should feel like a robotics research project rather than a product landing page, foreground action-conditioned streaming, and make the current code, checkpoints, and benchmark results easy to reach.

## Audience and launch state

The primary audience is robotics and world-model researchers evaluating the method, results, and released artifacts. This is a preview launch:

- Code and checkpoints are available.
- The paper is explicitly marked **Coming Soon**.
- Final rollout films are explicitly marked **Coming Soon**.
- Real rollout frames are used as honest preview posters; the page never shows a non-functional play control.
- Authors, affiliations, citation text, and analytics are excluded until the paper release.

## Visual direction

Use a restrained “robotics cinema” direction:

- Near-black navy background with cool mint/cyan accents.
- Warm orange is reserved for limited release-state labels such as **Coming Soon**.
- Large editorial typography and generous whitespace establish hierarchy.
- Real LIBERO rollout frames form a cinematic hero montage and qualitative gallery.
- Subtle motion may reveal content and animate the streaming diagram, but the page must honor `prefers-reduced-motion`.
- StreamWAM result rows use the same typography and background as other rows; the method is identified by its name, not visual highlighting.

The hero headline is **“Think ahead. Act now.”**, followed by **“Streaming World-Action Models for Robotic Manipulation.”**

## Information architecture

The page is a single responsive document with five navigation destinations:

1. **Overview** — project identity, action-conditioned positioning, artifact buttons, and the coming-soon rollout poster.
2. **Method** — a three-stage visual explanation: executing action prefix → action-conditioned visual future → asynchronously inferred next world-action chunk. A compact comparison explains that the framework systematically evaluates streaming strategies rather than presenting overlap as a systems-only optimization.
3. **Results** — a summary metric ribbon followed by LIBERO, RoboCasa, and RoboTwin tables. Values and evaluation protocols match the repository README.
4. **Gallery** — three real LIBERO rollout posters with task labels and a clear future-video state.
5. **Resources** — active Code and Models links, inactive Paper and Rollout Film entries marked Coming Soon, followed by acknowledgements.

The mobile layout keeps the same reading order, turns the navigation into a compact menu, stacks the hero montage, and makes wide result tables horizontally scrollable with an accessible hint.

## Content requirements

The overview must state that StreamWAM is a research framework and unified testbed for systematically studying streaming strategies in WAM-based robot control. It must then introduce StreamWAM itself as an action-conditioned streaming formulation: the prefix of actions currently being executed is fed back to the world model, conditioning future video generation while the next world-action chunk is inferred asynchronously.

Public links:

- Code: `https://github.com/SJTU-DENG-Lab/StreamWAM`
- Models: `https://huggingface.co/SJTU-DENG-Lab/StreamWAM`
- Paper: Coming Soon, without a dead link

Top-level summary metrics:

- LIBERO average success: `98.20%`
- LIBERO chunk time: `41.0 ms`
- RoboCasa total time: `11.76 s`

All detailed result values, evaluation protocols, model-lineage notes, and acknowledgements must be copied from the current public README. The public method name is always **StreamWAM**; internal runtime labels such as RTC-AC or AC-StreamWAM must not appear.

## Technical architecture

The site lives entirely under `academic_project_page/`:

- `index.html` owns semantic structure and public copy.
- `styles.css` owns the responsive visual system and motion preferences.
- `script.js` owns progressive enhancements only: compact navigation, result tabs, and viewport reveals. All essential content remains readable without JavaScript.
- `assets/` contains optimized local poster images and social preview artwork.
- `README.md` documents local preview, content editing, video replacement, and deployment.

No framework, build step, CDN, web font, remote image, cookie, or analytics dependency is introduced. Relative asset paths make the page work both locally and at the GitHub project-site path `/StreamWAM/`.

GitHub Pages deployment uses `.github/workflows/pages.yml` and uploads only `academic_project_page/`. The workflow must not package repository outputs, checkpoints, or training data. Repository administrators may need to select **GitHub Actions** once under Settings → Pages before the first deployment.

## Future video interface

Initial gallery cards use poster images and text-only coming-soon badges. Each card is structured so a future local `<video controls preload="metadata" poster="…">` can replace the poster container without changing the surrounding layout. The editing guide names the expected video location and replacement markup. No placeholder video files are committed.

## Accessibility and failure behavior

- Semantic headings, landmarks, tables, and link text support assistive navigation.
- Keyboard focus is visible, contrast remains readable, and controls have accessible labels.
- JavaScript-enhanced tab controls preserve all result content if JavaScript fails.
- Missing poster images leave meaningful alt text and task captions.
- External links use safe new-tab attributes where appropriate.
- A reduced-motion media query disables non-essential transitions and animated scrolling.

## Verification

Before push:

- Assert required copy, links, benchmark values, landmarks, and asset references with a repository-local test.
- Verify every local URL resolves when the directory is served over HTTP.
- Check desktop and mobile layout through browser screenshots when a compatible browser runner is available; otherwise document the unavailable runner and complete structural and HTTP smoke tests.
- Validate that only academic-page, workflow, spec, and plan files are staged in the dirty worktree.
- Request an independent code review and resolve any blocking findings.

## Out of scope

- Paper PDF, citation, BibTeX, author list, affiliation block, and publication venue.
- Final rollout videos or fabricated demo playback.
- Backend services, telemetry, CMS, search, or user accounts.
- Changes to training, inference, evaluation, or README content outside the page.
