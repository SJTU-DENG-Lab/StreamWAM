# README Results Synchronization Design

## Goal

Make the GitHub README report the same current task-performance and inference-efficiency results as the Stream-WAM project page.

## Scope

- Replace the README's three outdated combined result tables with task-performance tables matching the project page for LIBERO, RoboTwin 2.0, and RoboCasa.
- Include all baselines and Stream-WAM ablations currently shown on the project page.
- Use `Stream-WAM (Ours)` consistently in visible performance tables.
- Preserve the project page's bold best-result and underlined second-best styling.
- Present inference efficiency separately from task success, using the exact current chunk and episode times from the project page's accessible data table.
- Keep the benchmark protocol descriptions concise and aligned with the project page.
- Do not change installation, runtime, citation, license, or acknowledgement content.

## Verification

- Extend the page test suite to parse the README result tables and compare their rows with literal expected values.
- Assert that stale RoboTwin runtime values and the incorrect RoboCasa `75.83` value are absent from the current-results section.
- Run the complete project-page test module and `git diff --check` before pushing.
