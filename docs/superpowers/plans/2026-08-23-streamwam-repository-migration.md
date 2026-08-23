# StreamWAM Repository Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fully rename the current codebase from StarWAM/`starwam` to StreamWAM/`streamwam`, preserve runtime behavior, back up the old SJTU-DENG-Lab repository, and replace and rename that GitHub repository safely.

**Architecture:** Preserve the current repository history and make the migration in reviewable commits. Treat the verified Git bundle and the passing local validation suite as independent hard gates before changing the target remote; use force-with-lease tied to recorded commit IDs rather than an unconstrained force push.

**Tech Stack:** Git, Git bundle, Bash/Zsh, Python 3.10+, setuptools/`pyproject.toml`, pytest, GitHub REST API.

## Global Constraints

- The public repository and project name must be `StreamWAM`.
- The Python distribution and source package must be `streamwam`.
- No compatibility package or alias for `import starwam` may remain.
- Repository-owned imports, commands, recipes, launchers, defaults, logs, and documentation must use the new identity.
- Existing supported runtime behavior and state-dict checkpoint loading must continue working.
- External legacy asset identifiers may retain their real published names only when changing them would make those assets unresolvable.
- The target repository must not be mutated until its complete refs are stored in a verified and restoration-tested bundle.
- The target `main` replacement must use force-with-lease tied to the recorded old commit.
- Current user changes must be preserved; generated outputs, checkpoints, credentials, and accidental large files must not be committed.

---

## File Structure and Migration Boundaries

- Rename: `starwam/` to `streamwam/` — complete Python package identity.
- Modify: `pyproject.toml` — distribution metadata and setuptools package discovery.
- Rename/modify: `examples/**/*.py`, `examples/**/*.sh`, and `examples/**/*.yaml` — imports, module commands, recipe names, launchers, output defaults, and logs.
- Modify: `README.md`, `REPOSITORY_ANALYSIS.md`, `examples/**/*.md`, and `docs/**/*.md` — public identity and commands.
- Modify: `tests/**/*.py` — package imports while preserving behavioral assertions.
- Create: `tests/test_package_identity.py` — explicit hard-rename contract.
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/` — mirror, bundle, refs, checksum, and restoration check.
- Modify local Git metadata: `origin` URL — final SJTU-DENG-Lab StreamWAM repository.

### Task 1: Audit and Back Up the Target Repository

**Files:**
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/mirror.git`
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/Dual-Streaming-World-Action-Model-20260823.bundle`
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/refs-before-migration.txt`
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/SHA256SUMS`
- Create outside repository: `../repository-backups/dual-streaming-world-action-model-20260823/restore-check`

**Interfaces:**
- Consumes: `https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git`.
- Produces: a verified full-ref bundle, recorded target ref hashes, and exact old `main`/`hxy/streaming` hashes for later leases.

- [ ] **Step 1: Record the current local and target state**

Run:

```bash
git status --short --branch
git remote -v
git ls-remote --symref https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git HEAD
git ls-remote --heads --tags https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git
```

Expected: target `HEAD` resolves to `refs/heads/main`; all target refs are visible and recorded before mutation.

- [ ] **Step 2: Create a mirror and full bundle outside the repository**

Run from the current repository root:

```bash
backup_root=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/dual-streaming-world-action-model-20260823
mkdir -p "$backup_root"
git clone --mirror https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git "$backup_root/mirror.git"
git --git-dir="$backup_root/mirror.git" for-each-ref --format='%(objectname) %(refname)' | sort > "$backup_root/refs-before-migration.txt"
git --git-dir="$backup_root/mirror.git" bundle create "$backup_root/Dual-Streaming-World-Action-Model-20260823.bundle" --all
sha256sum "$backup_root/Dual-Streaming-World-Action-Model-20260823.bundle" > "$backup_root/SHA256SUMS"
```

Expected: the bundle and manifest exist outside the working repository and include every mirrored ref.

- [ ] **Step 3: Verify bundle integrity and restoration**

Run:

```bash
backup_root=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/dual-streaming-world-action-model-20260823
sha256sum --check "$backup_root/SHA256SUMS"
git bundle verify "$backup_root/Dual-Streaming-World-Action-Model-20260823.bundle"
git clone --mirror "$backup_root/Dual-Streaming-World-Action-Model-20260823.bundle" "$backup_root/restore-check"
git --git-dir="$backup_root/restore-check" for-each-ref --format='%(objectname) %(refname)' | sort > "$backup_root/refs-restored.txt"
diff -u "$backup_root/refs-before-migration.txt" "$backup_root/refs-restored.txt"
```

Expected: checksum and bundle verification pass and `diff` prints no differences. Stop the migration if any command fails.

### Task 2: Validate and Commit the Current Pre-Rename Work

**Files:**
- Modify: all currently modified tracked files intentionally belonging to RTC-AC, Joint-CD, rollout, documentation, and tests.
- Add: all currently untracked source, configuration, documentation, and test files that pass the audit.
- Exclude: `outputs/`, checkpoints, caches, credentials, environment files, and accidental large artifacts.

**Interfaces:**
- Consumes: the current dirty working tree based on commit `0a9e58e`.
- Produces: a clean pre-rename baseline commit and annotated tag `backup/pre-streamwam-rename-20260823`.

- [ ] **Step 1: Audit every pending path and size**

Run:

```bash
git status --short
git diff --stat
git diff --check
find . -path './.git' -prune -o -type f -size +5M -print
git ls-files --others --exclude-standard -z | xargs -0 -r file
```

Expected: every pending file is classified; no model weights, outputs, credential files, or unexpected binary assets are selected for commit.

- [ ] **Step 2: Scan pending content for credential patterns**

Run:

```bash
git diff -- . ':!docs/superpowers/plans/2026-08-23-streamwam-repository-migration.md' | rg -n 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|github_pat_|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{16}' || true
git ls-files --others --exclude-standard -z | xargs -0 -r rg -n 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|github_pat_|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{16}' || true
```

Expected: no credential material is found. Investigate any match before staging.

- [ ] **Step 3: Run the pre-rename behavioral suite**

Run:

```bash
python -m compileall -q starwam examples tests
python -m pytest -q
```

Expected: compilation and all existing tests pass. If a test fails, use systematic debugging and fix the current implementation before creating the baseline commit.

- [ ] **Step 4: Stage only audited project files and review the index**

Run:

```bash
git add -u
git add REPOSITORY_ANALYSIS.md docs examples starwam tests
git diff --cached --check
git diff --cached --stat
git status --short
```

Expected: the index contains intended source/docs/tests only; backup files and runtime outputs remain outside the index.

- [ ] **Step 5: Commit and tag the pre-rename baseline**

Run:

```bash
git commit -m "Add asynchronous RTC-AC and joint consistency rollout"
git tag -a backup/pre-streamwam-rename-20260823 -m "Pre-StreamWAM rename baseline"
```

Expected: the current RTC-AC work is recoverable by commit and annotated tag.

### Task 3: Establish the Hard Package-Rename Contract

**Files:**
- Create: `tests/test_package_identity.py`
- Rename: `starwam/` to `streamwam/`
- Modify: `pyproject.toml`
- Modify: all Python imports under `streamwam/`, `examples/`, and `tests/`.

**Interfaces:**
- Consumes: Python import machinery and `pyproject.toml`.
- Produces: importable `streamwam`, absent `starwam`, and distribution metadata named `streamwam`.

- [ ] **Step 1: Write the failing package-identity test**

Create `tests/test_package_identity.py` with:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_streamwam_is_the_only_package_identity() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "streamwam"
    assert importlib.util.find_spec("streamwam") is not None
    assert importlib.util.find_spec("starwam") is None
```

- [ ] **Step 2: Run the identity test and verify it fails before rename**

Run:

```bash
python -m pytest tests/test_package_identity.py -q
```

Expected: FAIL because the distribution and source package are still named `starwam`.

- [ ] **Step 3: Rename the package and perform the mechanical identity rewrite**

Run:

```bash
git mv starwam streamwam
rg -l -0 'StarWAM|starwam|STARWAM' pyproject.toml streamwam examples tests README.md REPOSITORY_ANALYSIS.md docs \
  --glob '!docs/superpowers/specs/2026-08-23-streamwam-repository-migration-design.md' \
  --glob '!docs/superpowers/plans/2026-08-23-streamwam-repository-migration.md' \
  | xargs -0 -r perl -pi -e 's/STARWAM/STREAMWAM/g; s/StarWAM/StreamWAM/g; s/starwam/streamwam/g'
```

Expected: package directory, imports, distribution metadata, module commands, and repository-owned identity strings use StreamWAM naming.

- [ ] **Step 4: Rename repository-owned files containing the old name**

Preview the exact set:

```bash
git ls-files | rg '(^|/)StarWAM|(^|/)starwam|STARWAM'
```

Then collision-check and rename it:

```bash
git ls-files -z \
  | while IFS= read -r -d '' old_path; do
      new_path=${old_path//STARWAM/STREAMWAM}
      new_path=${new_path//StarWAM/StreamWAM}
      new_path=${new_path//starwam/streamwam}
      if [ "$new_path" != "$old_path" ]; then
        if [ -e "$new_path" ]; then
          echo "rename collision: $old_path -> $new_path" >&2
          exit 1
        fi
        mkdir -p "$(dirname "$new_path")"
        git mv "$old_path" "$new_path"
      fi
    done
```

Expected: recipe, launcher, and repository documentation paths are renamed with no destination collisions; `git status` represents them as renames where similarity permits.

- [ ] **Step 5: Run the focused hard-rename test**

Run:

```bash
python -m pytest tests/test_package_identity.py -q
```

Expected: PASS; `streamwam` resolves and `starwam` does not.

### Task 4: Review Semantic Branding and Preserve External Compatibility

**Files:**
- Modify: `README.md`
- Modify: `REPOSITORY_ANALYSIS.md`
- Modify: `examples/libero/LIBERO.md`
- Modify: `examples/robotwin/RoboTwin.md`
- Modify: renamed YAML recipes and shell launchers under `examples/`.
- Modify: renamed source files under `streamwam/` where user-facing names or environment variables remain.

**Interfaces:**
- Consumes: the mechanical rewrite from Task 3.
- Produces: coherent StreamWAM positioning and a classified list of intentional external legacy identifiers.

- [ ] **Step 1: Rewrite the public project positioning**

Update the README title and opening description to identify StreamWAM as a general World-Action Model framework with streaming asynchronous action-conditioned inference, while retaining the documented MoT, Shared-DiT, feature-conditioned, Joint-CD, and RTC-AC families actually present in the repository.

Expected opening identity:

```markdown
# StreamWAM: Streaming World-Action Models for Robot Control

StreamWAM is a research framework for World-Action Models with synchronous and asynchronous robot-policy inference, including action-conditioned RTC-AC execution.
```

- [ ] **Step 2: Inspect every residual old-name occurrence**

Run:

```bash
rg -n -i 'starwam' --glob '!outputs/**' --glob '!.git/**' .
```

Expected: no repository-owned old package/import/command/default remains. Each retained match must be an actual external published URL, model identifier, historical citation, migration design record, or Git commit reference and must not affect imports or execution.

- [ ] **Step 3: Check renamed commands and paths for broken references**

Run:

```bash
rg -n 'python -m |--config |bash examples/' README.md examples docs --glob '*.md' --glob '*.sh'
```

Expected: documented module commands use `streamwam`; referenced recipe and launcher paths exist after filename renames.

- [ ] **Step 4: Commit the complete local rename**

Run:

```bash
git add -A
git diff --cached --check
git diff --cached --stat
git commit -m "Rename project and package to StreamWAM"
```

Expected: one reviewable rename commit containing no runtime outputs or backup artifacts.

### Task 5: Run Full Local Validation and Fix Rename Regressions

**Files:**
- Modify only files demonstrated by validation failures to contain incorrect renamed imports, paths, metadata, or CLI references.
- Test: `tests/test_package_identity.py`
- Test: all existing `tests/**/*.py`.

**Interfaces:**
- Consumes: the fully renamed local codebase.
- Produces: evidence that all available runtime gates pass before remote replacement.

- [ ] **Step 1: Compile all Python sources**

Run:

```bash
python -m compileall -q streamwam examples tests
```

Expected: exit code 0 with no syntax or import-path compilation errors.

- [ ] **Step 2: Verify clean interpreter import behavior**

Run:

```bash
python -c 'import streamwam; print(streamwam.__file__)'
python -c 'import importlib.util; assert importlib.util.find_spec("starwam") is None'
```

Expected: the first command prints `streamwam/__init__.py`; the second exits 0.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
python -m pytest -q
```

Expected: every test passes. For any failure, use systematic debugging, make the smallest rename-specific correction, rerun the failing test, and then rerun the complete suite.

- [ ] **Step 4: Smoke-test principal CLI entry points**

Run:

```bash
python -m streamwam.training.train --help
python -m streamwam.tools.precompute_text_cache --help
python examples/libero/rollout.py --help
python examples/libero/multigpu_rollout.py --help
python examples/robotwin/policy_server.py --help
```

Expected: each command exits 0 and prints usage without importing the removed package.

- [ ] **Step 5: Validate tracked paths and repository cleanliness**

Run:

```bash
git ls-files | rg '(^|/)starwam($|/)|starwam' || true
git grep -n -E '(^|[^A-Za-z])import starwam|from starwam|python -m starwam' || true
git status --short --branch
```

Expected: no old package path/import/command remains; only explicitly classified legacy external/documentation references may remain; working tree is clean after committing any validation fix.

- [ ] **Step 6: Run an asset-backed GPU smoke test when resources are available**

Use the renamed single-task LIBERO launcher with one task and one trial, existing local Wan2.2 assets, matching checkpoint/stats, and the active `libero` environment.

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash examples/libero/scripts/launch_streamwam_libero_rtc_ac_rollout.sh \
  --task-suite-name libero_spatial \
  --task-id 0 \
  --num-trials 1 \
  --output-dir outputs/streamwam_migration_gpu_smoke
```

Expected: model and checkpoint load, one environment episode executes, and `results.json` is written. If GPU/assets are unavailable, record the exact unavailable prerequisite in the final report; do not skip Tasks 5.1–5.5.

### Task 6: Replace and Rename the GitHub Repository

**Files:**
- Modify local Git remote metadata: `origin`.
- Mutate remote repository only after Tasks 1–5 pass.

**Interfaces:**
- Consumes: verified backup bundle, recorded target ref hashes, clean tested local `main`, and `${GITHUB_TOKEN}` with repository administration permission.
- Produces: `https://github.com/SJTU-DENG-Lab/StreamWAM.git` serving the migrated `main` with obsolete target branches removed.

- [ ] **Step 1: Reverify backup and capture lease hashes immediately before push**

Run:

```bash
backup_root=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/dual-streaming-world-action-model-20260823
sha256sum --check "$backup_root/SHA256SUMS"
git bundle verify "$backup_root/Dual-Streaming-World-Action-Model-20260823.bundle"
git ls-remote --heads --tags https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git
git status --short --branch
```

Expected: bundle remains valid, remote refs still match `refs-before-migration.txt`, and local worktree is clean. Stop if the target changed since backup and refresh the backup/designated leases.

- [ ] **Step 2: Add the target remote and replace `main` with an exact lease**

Read the old target `main` hash from `refs-before-migration.txt`, validate that
it is non-empty, then run:

```bash
backup_root=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/dual-streaming-world-action-model-20260823
old_main_hash=$(awk '$2 == "refs/heads/main" {print $1}' "$backup_root/refs-before-migration.txt")
test -n "$old_main_hash"
git remote add streamwam-target https://github.com/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model.git
git push --force-with-lease=refs/heads/main:"$old_main_hash" streamwam-target main:main
```

Expected: push succeeds only if target `main` still equals the recorded backed-up commit; never omit the lease.

- [ ] **Step 3: Delete backed-up obsolete target branches with exact leases**

For `hxy/streaming` and every other non-`main` target branch recorded in the backup manifest, run:

```bash
backup_root=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/dual-streaming-world-action-model-20260823
old_streaming_hash=$(awk '$2 == "refs/heads/hxy/streaming" {print $1}' "$backup_root/refs-before-migration.txt")
test -n "$old_streaming_hash"
git push --force-with-lease=refs/heads/hxy/streaming:"$old_streaming_hash" streamwam-target :refs/heads/hxy/streaming
```

Expected: obsolete branches are removed only when their current commits equal their backed-up hashes. Keep tags only if explicitly classified as belonging to the new project; otherwise remove them with the same recorded-ref discipline.

- [ ] **Step 4: Rename the repository through the GitHub API**

Run without printing the token:

```bash
test -n "${GITHUB_TOKEN:-}"
curl --fail-with-body --silent --show-error \
  -X PATCH \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/SJTU-DENG-Lab/Dual-Streaming-World-Action-Model \
  -d '{"name":"StreamWAM"}'
```

Expected: response reports `full_name` as `SJTU-DENG-Lab/StreamWAM` and default branch `main`. If permission is denied, stop and report it without creating another repository.

- [ ] **Step 5: Point local origin to the new repository and verify it**

Run:

```bash
git remote remove streamwam-target
git remote set-url origin https://github.com/SJTU-DENG-Lab/StreamWAM.git
git fetch origin
git branch --set-upstream-to=origin/main main
git remote -v
git ls-remote --symref origin HEAD
git ls-remote --heads --tags origin
```

Expected: `origin` uses the new URL, `HEAD` resolves to `main`, and remote `main` equals local `HEAD`.

- [ ] **Step 6: Perform a fresh-clone verification**

Run into a new, explicit verification directory outside the working tree:

```bash
verify_dir=/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/hxy/WAM/repository-backups/streamwam-fresh-clone-20260823
git clone https://github.com/SJTU-DENG-Lab/StreamWAM.git "$verify_dir"
git -C "$verify_dir" rev-parse HEAD
git -C "$verify_dir" status --short --branch
```

Expected: fresh clone `HEAD` equals migrated local `HEAD` and its working tree is clean.

### Task 7: Final Migration Record

**Files:**
- Modify: `docs/superpowers/plans/2026-08-23-streamwam-repository-migration.md` only to check completed steps if plan tracking is retained.
- No runtime source changes.

**Interfaces:**
- Consumes: all backup, test, commit, and remote-verification evidence.
- Produces: a final report containing recovery and verification facts.

- [ ] **Step 1: Record completion evidence**

Report:

```text
Backup bundle absolute path
Backup SHA-256
Backed-up old main and branch commit IDs
Pre-rename baseline commit and backup tag
StreamWAM rename/fix commit IDs
pytest result count
CLI smoke-test results
GPU smoke-test result or exact unavailable prerequisite
Final GitHub URL
Final remote main commit ID
Fresh-clone verification commit ID
Intentional retained external legacy identifiers, if any
```

Expected: the user can restore the old repository, identify every migration commit, and verify the new repository without relying on hidden session state.
