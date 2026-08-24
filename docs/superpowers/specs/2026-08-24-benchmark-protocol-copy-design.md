# Benchmark Protocol Copy Design

## Goal

Add the missing evaluation protocol directly before the RoboCasa and RoboTwin
result tables in the root README so readers can interpret the reported success
rates without changing the approved results layout.

## RoboCasa copy

Insert this paragraph immediately after `### RoboCasa` and before its table:

> We evaluate on the RoboCasa target benchmark across 50 target tasks, with 50
> trials per task, and report the average task success rate.

This defines the evaluation scope, sample count, and reported accuracy metric.

## RoboTwin copy

Insert this paragraph immediately after `### RoboTwin` and before its table:

> We evaluate 50 RoboTwin 2.0 tasks with 100 rollout episodes per task. `Clean`
> reports the success rate under the easy setting, while `Random` reports the
> success rate under the hard domain-randomization setting.

Keep the existing `Clean` and `Random` table headers unchanged.

## Scope constraints

- Do not change any result-table value, unit, row label, or emphasis.
- Do not change the LIBERO protocol or results.
- Do not modify any section outside the two benchmark protocol additions.
- Preserve the untracked `docs/huggingface/README.md` file without staging or
  editing it.

## Validation

- Confirm each paragraph appears exactly once under its matching subsection and
  before the corresponding table.
- Confirm the RoboCasa and RoboTwin tables are byte-for-byte unchanged.
- Run `git diff --check` and the package-identity test in the validated FastWAM
  virtual environment.
