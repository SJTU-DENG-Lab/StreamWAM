# StreamWAM Repository Migration Design

Date: 2026-08-23

## Objective

Replace the contents of
`SJTU-DENG-Lab/Dual-Streaming-World-Action-Model` with the current local
codebase and rename the public project to `StreamWAM`.

The rename is intentionally complete. The source package, distribution name,
imports, commands, recipes, launchers, documentation, log names, and public
repository identity will use `streamwam` or `StreamWAM`. No compatibility
package for `import starwam` will remain. After migration, all supported code
paths must continue to execute under the new package name.

## Chosen History Strategy

Preserve the current local repository history and add explicit migration
commits. Do not merge the unrelated history of the target repository and do
not squash the current codebase into a new root commit.

The target repository's existing history will be backed up before any remote
branch is replaced. Once the backup is verified, the target `main` branch will
be replaced by the current repository history and obsolete target branches
will be removed.

## Safety and Backup

Before changing the target remote:

1. Mirror-clone every target ref, including `main`, `hxy/streaming`, and any
   tags discovered at execution time.
2. Create a full Git bundle outside the working repository under
   `WAM/repository-backups/`.
3. Record the target refs and SHA-256 checksum alongside the bundle.
4. Verify the bundle and perform a restoration test from it.
5. Do not force-push or delete a target ref unless all preceding checks pass.

Before the mechanical rename, inspect the current dirty working tree for
secrets, generated outputs, checkpoints, and unexpectedly large files. Commit
the intended RTC-AC, Joint-CD, asynchronous rollout, documentation, and test
changes as a pre-rename baseline. Create a local annotated backup tag at that
commit.

## Complete Rename Scope

The migration changes all repository-owned project identities:

- GitHub repository: `StreamWAM`
- Python distribution: `streamwam`
- Python source package: `starwam/` to `streamwam/`
- Imports: `starwam.*` to `streamwam.*`
- Module commands: `python -m starwam...` to `python -m streamwam...`
- Recipe and launcher filenames containing `starwam`
- YAML presets and repository-owned output/log defaults
- README, examples, technical documentation, and code comments
- Logger names, user-facing errors, and environment variables owned by this
  project

The migration will not invent new locations for external assets. References
to legacy externally published checkpoint or model identifiers may retain
their actual external names when changing them would make the asset
unresolvable. Such occurrences must be identified as external compatibility
references rather than active package names.

No `starwam` compatibility package or import alias will be created. A clean
environment must accept `import streamwam` and reject `import starwam`.

## Runtime Compatibility

The rename must preserve model behavior and supported checkpoint loading.
State-dict tensor keys will not be renamed merely because the Python package
changes. Existing state-dict checkpoint adapters remain responsible for their
current formats. Any serialized artifact that embeds Python module paths will
be explicitly identified during testing; it will not be silently claimed as
compatible.

The following paths are in scope for runtime verification:

- typed YAML configuration loading and CLI overrides;
- model and checkpoint format dispatch;
- LIBERO preprocessing, rollout, timing, and multi-GPU workload management;
- direct Joint-CD consistency inference;
- action-conditioned RTC-AC inference and asynchronous chunk control;
- RTC-AC acceleration contracts;
- repository CLI entry points and documented module commands.

## Validation Gates

Local validation must pass before any target branch replacement:

1. Compile all tracked Python sources.
2. Confirm `import streamwam` succeeds from a clean interpreter.
3. Confirm `import starwam` fails because no compatibility package exists.
4. Run the complete repository pytest suite.
5. Run `--help` smoke checks for principal training, rollout, preprocessing,
   and evaluation entry points.
6. Search tracked source and documentation for stale repository-owned
   `starwam`/`StarWAM` names and classify any intentional external references.
7. When suitable GPU resources and assets are available, run a minimal
   single-task inference smoke test. If unavailable, report that limitation
   explicitly rather than weakening the preceding gates.

After pushing, verify the renamed GitHub repository, default branch, remote
refs, and a fresh clone or fetch from the new URL.

## Remote Migration

After local validation and backup verification:

1. Replace target `main` using a lease tied to the previously recorded target
   commit. Do not use an unconstrained force push.
2. Remove obsolete target branches only after confirming their commits exist
   in the verified backup bundle.
3. Rename the GitHub repository from
   `Dual-Streaming-World-Action-Model` to `StreamWAM`.
4. Set the local `origin` URL to
   `https://github.com/SJTU-DENG-Lab/StreamWAM.git`.
5. Verify GitHub redirects, the new clone URL, default branch, and pushed
   commit.

The current checkout directory may remain named `StarWAM` during the active
workspace session because moving the workspace root can invalidate editor and
tool paths. New clones will naturally use the directory name `StreamWAM`.

## Failure Handling

- If backup creation or restoration verification fails, stop before remote
  mutation.
- If current-tree validation fails, fix the local rename and repeat all local
  gates before pushing.
- If repository rename permission is unavailable, leave the validated code
  unpushed or pushed only under the explicitly approved existing repository
  identity; do not create an unapproved alternate repository.
- If branch protection rejects replacement, report the exact rule and request
  the required administrative change instead of bypassing it.
- If post-push verification fails, preserve the remote state and restore from
  the verified backup only when the failed state and restoration target have
  been explicitly resolved.

## Completion Criteria

The migration is complete only when:

- the old target repository is recoverable from a verified full backup;
- the local codebase is committed under the complete StreamWAM identity;
- all available validation gates pass;
- `SJTU-DENG-Lab/StreamWAM` serves the migrated `main` branch;
- obsolete target branches are removed after backup verification;
- the local `origin` points to the new repository URL; and
- the final report records backup paths, checksums, commit IDs, test results,
  and any intentionally retained external legacy names.
