# Public release checklist

This checklist separates the research workspace from the public code release.
It is intentionally not part of the anonymous conference PDF package.

## Publish

- `src/worldcache/`
- `scripts/` after removing machine-specific helper assumptions
- `tests/`
- `paper/` and the selected lightweight diagnostics
- `README.md`, `DATASETS.md`, `environment.yml`, `pyproject.toml`, `LICENSE`
- `third_party/` with its existing SPDX notices

## Curate before the first public git push

These files contain session history, prompts, or machine-specific research
paths and should not be copied into a public repository without manual review:

- `SESSION_STATE.md`
- `SESSION_HANDOFF.md`
- `logs/`
- `external_prompts/`
- `WORLD_CACHE_FINDINGS.md`
- `TWO_WEEK_EXPLORATION_PLAN.md`
- `outputs/reports/TWO_WEEK_FINDINGS.md`
- environment snapshots under `outputs/reports/`

The `.gitignore` protects new local copies, but it cannot untrack files that
were already committed. Use a deliberate allowlist or an explicit review of
`git ls-files` before publishing.

## Verified in this workspace

- Main paper and supplement compile to five pages each.
- Submission consistency verifier passes.
- The anonymous PDF package contains only the two PDFs.
- The reproducibility bundle contains the manifest-locked files.
- The full regression suite passes 38 tests.
- The hosted CPU CI workflow is syntactically valid and its local equivalent
  passes the 36 tests that do not require CUDA/Habitat-Sim.
- Release-facing text contains no detected credential markers or local data
  paths after the `DATASETS.md` cleanup.
