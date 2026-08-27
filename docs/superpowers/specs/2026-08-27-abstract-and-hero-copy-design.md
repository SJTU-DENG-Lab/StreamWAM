# Abstract and Hero Copy Design

## Scope

Revise only the project-page hero description and Abstract. Preserve the page structure, figures, metrics, and all unrelated working-tree changes.

## Copy

The hero should state the method in three compact moves: Stream-WAM introduces action-conditioned streaming, the committed action prefix aligns future video generation with ongoing motion, and the robot continues acting while the next prediction is prepared.

The Abstract should:

1. Define World Action Models as jointly generating future visual observations and robot actions.
2. Motivate streaming through synchronous idle time and inconsistency under naive asynchronous switching.
3. Explain that committed actions condition future video generation and guide a consistent action continuation.
4. State that streaming is introduced within world prediction, then summarize evaluation scope and the headline LIBERO result.

## Terminology

- Use `World Action Models (WAMs)`, not `world-action models`.
- Use `future visual observations`, `future video generation`, `action chunk`, and `action continuation`.
- Avoid `video-model priors`, `future-video`, `world-action chunk`, `action postfix`, and `state–prediction mismatch`.
- Retain `action-conditioned` because it names the central mechanism.

## Verification and Change Isolation

Add a focused static-page copy test that checks the rendered hero and Abstract text. Stage the relevant `docs/index.html` hunks selectively so the in-progress method-figure edits remain uncommitted and are not pushed.
