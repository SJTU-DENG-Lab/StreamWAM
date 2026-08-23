# RTC-AC Runtime Parity Design

## Goal

Run StarWAM accelerated RTC-AC with the same Python acceleration stack used by
the wyx reference evaluation, without modifying the user's current `libero`
environment or changing RTC inference semantics.

## Reference runtime

- Python: `/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python`
- PyTorch: `2.7.1+cu128`
- Triton: `3.3.1`
- CUDA runtime reported by PyTorch: `12.8`

## Design

The RTC-AC launcher accepts `PYTHON_BIN`, defaulting to `python`. It invokes the
multi-GPU manager with that executable. The manager already uses
`sys.executable` for every worker, so manager and workers remain in one runtime.

Accelerated evaluation records executable, Python, PyTorch, Triton and CUDA
versions in its acceleration status. The final manager summary prints this
runtime identity next to compile/cache status. This makes runtime parity
observable rather than inferred from the activated shell.

Before launching a costly evaluation, the shell launcher runs a lightweight
import/version preflight with the selected executable. A missing executable or
incompatible StarWAM import fails before model loading.

## Compatibility

- Eager RTC-AC and all model math remain unchanged.
- Existing commands without `PYTHON_BIN` continue to use the active environment.
- The wyx environment is selected explicitly; the current `libero` environment
  is not downgraded or mutated.
- Checkpoint, stats, task allocation and timing definitions remain unchanged.

## Verification

Automated tests execute a real temporary Python shim through the launcher and
verify that it is selected, while manager tests verify that workers inherit
`sys.executable` and runtime metadata is aggregated without being invented.
The existing accelerated/eager numerical equivalence tests remain required.

The real GPU validation uses one GPU and one trial over all 40 LIBERO tasks.
The primary comparison is accelerated model inference per chunk against the
previous StarWAM result (`127.86 ms`) and the wyx runtime result (about
`40.16 ms` on its recorded benchmark). Exact latency is hardware/load dependent;
runtime and backend identity must match before interpreting the number.

## Repository policy

No Git commit is created. All changes remain visible in the working tree for
manual review and submission.
