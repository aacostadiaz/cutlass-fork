
# Release Process Guide

This document outlines the branching, syncing, and release workflow for the repository.

---

## Branching Strategy

| Branch           | Purpose                                       |
|------------------|-----------------------------------------------|
| `main`           | Mirrors the upstream repository.              |
| `sycl_develop`   | Active development of sycl specific features. |
| `release/*`      | Frozen branches for released versions.        |

### Key Rules

- All new development is done in `sycl_develop`.
- Changes from upstream are first synced into `main`, then merged into `sycl_develop`.
- Releases are cut from `sycl_develop` and should remain frozen.
- Patch fixes are applied on new release branches and back-merged as needed.

---

## Syncing with Upstream

1. Fetch and merge the latest changes from the **upstream repo** into the `main` branch.
   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

2. Open a pull request from `main` to `sycl_develop`.

3. **Merge this PR using a *merge commit*** to preserve upstream commit history.
    - Do **not** squash or rebase this PR.

---

## Creating a Release

1. From the current `sycl_develop`, create a new release branch:
   ```bash
   git checkout sycl_develop
   git pull
   git checkout -b release/v1.2.0
   ```

2. Tag the version:
   ```bash
   git tag v1.2.0
   git push origin release/v1.2.0
   git push origin v1.2.0
   ```

3. The branch `release/v1.2.0` is now considered **frozen**.

---

## Post-Release Fixes (Patch Versions)

When a bug is found in a released version:

1. Create a new patch branch from the release tag:
   ```bash
   git checkout release/v1.2.0
   git checkout -b release/v1.2.1
   ```

2. Apply and commit the fix:
   ```bash
   git commit -am "Fix: [description]"
   ```

3. Tag the patch release:
   ```bash
   git tag v1.2.1
   git push origin release/v1.2.1
   git push origin v1.2.1
   ```

4. Merge the fix back into:
    - `sycl_develop` (to propagate the fix to ongoing development).

   ```bash
   git checkout sycl_develop
   git merge release/v1.2.1
   ```
