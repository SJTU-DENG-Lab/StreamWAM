# StreamWAM Three-Act Editorial Redesign

## Goal

Refine the text-first StreamWAM research blog into a continuous three-act argument. Preserve the
current masthead, which already communicates the project and headline results well. Replace the body’s
large chapter headings and strongly separated regions with denser, connected prose that develops the
research motivation before introducing the method.

This revision changes body copy, hierarchy, and visual rhythm. It does not change benchmark values,
protocols, model lineage, public links, release status, or the public StreamWAM method name.

## Source boundary

The discussion of asynchronous WAM deployment may draw factual framing from *World Action Models in
Real Time: An Empirical Study of Smooth Execution via Asynchronous Deployment* (arXiv:2608.01880v2),
including its description of end-to-end WAM latency, inter-chunk disagreement, inference-time RTC, and
prefix-conditioned generation.

The public blog must paraphrase and reorganize this material rather than reproduce the paper’s prose.
Claims attributed to the paper remain scoped to its empirical platform. In particular, the blog must
not claim that prefix conditioning is generally ineffective: the paper reports a favorable overall
precision–smoothness balance for that family. StreamWAM’s distinction is instead described as an
extension from action-continuation consistency to action-conditioned visual-future generation.

The page should link the paper inline using `https://arxiv.org/abs/2608.01880`.

## Editorial structure

The masthead remains unchanged except for any minor spacing required to connect it to the article.
After the masthead, the core article follows three acts. These acts may have small uppercase labels or
margin numbers, but no large display headings. Each act contains four to six substantial paragraphs,
and transitions explicitly into the next act.

### Act I: WAM capability and the real deployment problem

The first act establishes why WAMs are useful before discussing their cost:

- WAMs connect predicted visual futures with robot action chunks instead of treating perception and
  control as unrelated outputs.
- Future-video generation provides a structured account of what an action sequence is intended to do
  to the scene.
- Joint video–action generation and iterative denoising are computationally expensive.
- Real deployment latency also includes camera capture, preprocessing or transmission, inference,
  command dispatch, and interpolation; it is not represented fully by an isolated model timing.
- Under synchronous infer-then-execute deployment, the robot may pause or replay stale actions while
  waiting, which appears as visibly hesitant or stop-and-go behavior on hardware.
- Chunked execution amortizes inference cost but does not eliminate the exposed wait at boundaries.

The act ends by motivating overlap: if inference cannot yet finish inside one control tick, the robot
should continue acting while the model works.

### Act II: Asynchrony and the continuity problem

The second act explains why overlap is useful and why it creates a new problem:

- Asynchronous deployment launches the next inference call while the current action chunk is still
  executing, hiding part of the model latency behind useful robot motion.
- Consecutive chunks are predicted from observations captured at different times, so their overlap
  region can disagree.
- A hard switch can create a discontinuity even when both chunks are individually plausible.
- Accurate temporal alignment is a prerequisite: the incoming chunk must be indexed to the robot’s
  actual progress through the outgoing chunk.

The prose then introduces two representative reconciliation families:

- Inference-time RTC guides the incoming denoising trajectory toward consistency with the outgoing
  chunk. The cited paper reports that velocity-level guidance did not reliably constrain its delay
  region on the evaluated platform, and it also adds work to inference.
- Prefix-conditioned generation supplies the committed action prefix as a clean condition and trains
  the model to produce a consistent continuation. It requires training support and assumes the prior
  action commitment remains useful. Its scope is action-chunk continuation: by itself, it does not
  explicitly ensure that the WAM’s generated video future is conditioned on the action currently being
  executed.

The blog maps this latter distinction to the existing `w/o Action Conditioning` row: this ablation
retains prefix-conditioned continuation while removing action conditioning from future-video
generation. The wording must not imply that the ablation is identical to every prefix-conditioned
method in the literature.

### Act III: StreamWAM

The third act introduces StreamWAM as the response to the remaining WAM-specific mismatch:

- StreamWAM retains asynchronous execution and prefix-conditioned continuation.
- It feeds the action prefix currently being executed into the world-model path.
- The action context conditions future-video generation, so the generated visual future describes what
  should happen under the robot motion already underway.
- The next action chunk is generated together with a visual future aligned to ongoing execution.
- Inference and execution are therefore coupled at both the runtime and modeling levels.
- The method should be presented as addressing visual–action consistency during streaming, not as a
  universal solution to abrupt environment changes or every source of deployment latency.

One compact method figure follows the prose. It supports the argument but does not replace it. The
article then transitions directly into the current benchmark evidence.

## Remaining article structure

After the three acts:

- Keep a short common-testbed passage, but integrate it as ordinary prose rather than a separate large
  chapter.
- Keep all three benchmark tables visible in sequence, with their current protocols and values.
- Keep restrained table interpretations derived directly from reported values.
- Keep the closing discussion, resources, model lineage, and acknowledgements.
- Keep Paper and rollout film as non-interactive `Coming Soon` statuses.

## Visual hierarchy

- The masthead remains the only display-scale section.
- Body act labels use approximately 11–13 px uppercase metadata styling.
- Core body prose remains approximately 17–19 px with a 760–820 px reading measure and generous line
  height.
- Remove display-sized body headings. The article may use a modest introductory title once, followed by
  small act labels and typographic opening sentences.
- Remove strong alternating section backgrounds and excessive top/bottom separation around the three
  acts. Use whitespace, a thin rule, or a small label to mark transitions.
- Keep only one method figure before the experiments. The execution explanation should be expressed in
  prose unless a compact inline timeline remains clearly subordinate to the text.
- Experiment tables may retain a subtle surface because they need scanning structure, but the text
  before them should remain visually continuous with the article.

## Copy and citation rules

- Use original explanatory prose; do not imitate the cited paper sentence by sentence.
- Do not quote the paper verbatim.
- Link the paper at the first detailed discussion of existing asynchronous strategies.
- Attribute platform-specific RTC findings to the cited study rather than stating them as universal.
- Describe prefix conditioning as effective for action continuation while identifying the missing
  explicit condition on future-video generation.
- Preserve the exact public mapping of `w/o Action Conditioning` without renaming the table row.
- Do not add authors, affiliations, venue, citation section, unreported results, or claimed superiority.
- Keep internal runtime names absent from public prose.

## Verification

- Add focused tests for the three act labels and their order.
- Require the body to contain at least 18 substantive paragraphs before the experiment section.
- Require the arXiv paper link and prohibit long verbatim text copied from the source.
- Preserve the complete benchmark table and protocol regression tests.
- Confirm the body contains no display-sized `h2` headings for the three acts.
- Test desktop, intermediate, and mobile widths for readable paragraph measure and reduced visual
  segmentation.
- Request independent review of source fidelity, claim scope, action-conditioning distinction, and
  public-release quality before push.

## Success criteria

A reader should be able to follow one continuous argument: WAMs provide visually grounded action
prediction but are slow on hardware; asynchronous execution hides latency but creates inter-chunk
continuity problems; existing RTC and prefix-conditioned methods reconcile action chunks in different
ways; StreamWAM additionally conditions the generated visual future on the action currently being
executed. The argument should remain understandable without relying on large headings or diagrams.
