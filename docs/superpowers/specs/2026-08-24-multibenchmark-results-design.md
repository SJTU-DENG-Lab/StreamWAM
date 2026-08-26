# Multi-Benchmark README Results Design

## Scope

Update the public README to present StreamWAM results on LIBERO, RoboCasa, and
RoboTwin in a compact benchmark-oriented layout. Remove the temporary runtime
contract and citation sections, and expand the acknowledgements to name the
upstream codebases used by the benchmark implementations.

## Results organization

Keep one `## Current results` section with three subsections in this order:

1. `### LIBERO`
2. `### RoboCasa`
3. `### RoboTwin`

The existing LIBERO evaluation protocol, table, and short result analysis stay
under the LIBERO subsection.

Before the benchmark-specific results, state the model lineage clearly:

- StreamWAM models are initialized from FastWAM-Joint checkpoints and then
  further trained.
- The RoboCasa implementation builds on X-WAM.
- The RoboTwin implementation builds on StarWAM.

This wording must distinguish checkpoint initialization from implementation
lineage and must not imply that every benchmark implementation is based on both
X-WAM and StarWAM.

## RoboCasa table

Use units in the headers and mark higher accuracy and lower time as preferable.
Bold the StreamWAM row and label it as the proposed method.

| Method | Accuracy (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|
| X-WAM | 75.42 | 504.00 | 37.31 |
| X-WAM-CD | 75.33 | 135.21 | 33.60 |
| **StreamWAM (Ours)** | **75.35** | **136.76** | **11.76** |

## RoboTwin table

Use units in the headers. A missing measurement is represented by an em dash,
not by zero or an empty cell. Bold the StreamWAM row and label it as the
proposed method.

| Method | Clean (%) ↑ | Random (%) ↑ | Total (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|---:|---:|
| StarWAM | 84.8 | 86.0 | 85.4 | 189.3 | — |
| StarWAM-CD | 79.0 | 79.2 | 79.1 | 81.6 | — |
| **StreamWAM (Ours)** | **87.2** | **88.8** | **87.6** | — | **112.2** |

## Removed sections

Delete the complete `## Accelerated runtime contract` section, including its
environment table, diagnostic output, and strict-fallback paragraph.

Delete the complete `## Citation` section. It will be restored after the
technical report and formal BibTeX entry are available.

## Acknowledgements

Retain the existing acknowledgements and add direct links to:

- [StarWAM](https://github.com/shaohua-pan/StarWAM)
- [X-WAM](https://github.com/sharinka0715/X-WAM)

## Validation

- Confirm that the README contains exactly one `## Current results` section and
  the three benchmark subsections in the approved order.
- Confirm each Markdown table has the correct number of columns in every row.
- Preserve all supplied values and express units only in column headers.
- Confirm the runtime-contract and citation headings no longer exist.
- Confirm both new acknowledgement links are present.
- Run Markdown whitespace checks and the repository's documentation/package
  identity test in the previously validated environment.
