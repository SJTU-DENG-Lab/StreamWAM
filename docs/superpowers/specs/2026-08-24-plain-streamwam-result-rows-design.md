# Plain StreamWAM Result Rows Design

## Goal

Remove the visual highlighting from the StreamWAM rows in all three result
tables while preserving the approved public naming and benchmark content.

## Presentation

Use ordinary Markdown table cells for the complete StreamWAM row. Do not use
`<mark>`, bold markup, an `(Ours)` suffix, icons, badges, inline HTML, or custom
styling.

The exact rows are:

```markdown
| StreamWAM | 96.60 | 98.80 | 97.40 | 100.00 | 98.20 | 41.0 | 5.36 / 3.15 |
| StreamWAM | 75.35 | 136.76 | 11.76 |
| StreamWAM | 87.2 | 88.8 | 87.6 | — | 112.2 |
```

## Scope

- Change only the three StreamWAM result rows in the published `README.md`.
- Preserve all method names, result values, units, protocols, table order, and
  surrounding prose.
- Preserve the committed `rtc_ac` executable literals until the concurrent
  runtime rename is committed with its implementation.
- Preserve and exclude every concurrent source, script, test, model-card, and
  working-tree README interface change.

## Validation

- Confirm exactly three result rows start with `| StreamWAM |`.
- Confirm `README.md` contains no `<mark>`, `(Ours)`, or bold StreamWAM result
  cells.
- Confirm the three rows match the exact approved values above.
- Confirm the staged diff changes only `README.md` and only removes `<mark>`
  tags from those rows.
- Run `git diff --check` and the package-identity tests in the validated
  environment.
