# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims for [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed (breaking)

- **Package rename**: PyPI distribution and Python import name change
  from `ml4h-latentverse` / `ml4h_latentverse` to `latentverse`.
  - Old: `pip install ml4h-latentverse` then `from ml4h_latentverse import ...`
  - New: `pip install latentverse` then `from latentverse import ...`
  - The previous PyPI project (`ml4h-latentverse`, last release 0.1.2)
    is orphaned under an inaccessible account; releases continue under
    the new name.
- **Submodule rename**: `ml4h_latentverse.tests` → `latentverse.evaluations`.
  The old name was misleading (these are public ML evaluation entrypoints,
  not unit tests of the library) and collided with the pytest convention.
  Old `from ml4h_latentverse.tests.clustering import run_clustering` becomes
  `from latentverse.evaluations.clustering import run_clustering`.

### Changed (non-breaking)

- Repository moved to **src-layout** (`src/latentverse/`); prevents the
  working tree from shadowing an installed copy during dev.
- `requirements.txt` removed; dependencies are declared once in
  `pyproject.toml` (`[project.dependencies]` for runtime,
  `[project.optional-dependencies].dev` for `pytest` + `ruff`).
- Build artifacts (`dist/`) and OS metadata (`.DS_Store`) removed from
  the tree and added to `.gitignore`.
- Pytest now pinned to top-level `tests/` via
  `[tool.pytest.ini_options]`; will not accidentally collect the
  package's own evaluations submodule.
- GitHub Actions CI added: ruff lint + pytest on push and PR.

## [0.3.1] - 2026-07

Scale-hardening follow-up: bound the evaluations' internal parallelism and
cap robustness peak memory so many-label web-app runs don't oversubscribe a
Cloud Run instance.

### Changed (non-breaking)

- **Bounded internal parallelism.** The probing, expressiveness, and
  robustness evaluations no longer hard-code joblib `n_jobs=-1` ("grab every
  core"). They now read `LATENTVERSE_N_JOBS` (default `2`) via the new
  `latentverse.utils.get_n_jobs` helper. This prevents CPU/BLAS thread
  oversubscription when a host (e.g. the web app) runs many label columns
  concurrently and each column's evaluation would otherwise fan out across
  all cores. Set `LATENTVERSE_N_JOBS=-1` to restore the old all-cores
  behaviour. Bounding joblib workers does not cap the BLAS threads each fit
  spawns — use `OMP_NUM_THREADS` for that second layer.
- **Robustness streams noise one level at a time.** `run_robustness` used to
  pre-allocate a noise matrix for *every* noise level up front
  (`len(noise_levels) × N × D × 8` bytes resident before any work). It now
  generates each level's noise inside the worker from an independent,
  deterministically-seeded RNG (`default_rng([random_state, index])`), so
  peak memory is bounded by the worker count and results stay reproducible
  and thread-safe. Absolute noise values differ from 0.3.0 (new per-level
  RNG); metric *shapes* are unchanged.

## [0.3.0] - 2025-04

- MultiLoReFT multimodal decoupling module integrated.
- Reproducibility: every public entrypoint accepts `random_state`.
- `num_clusters` override and `standardize=True` available where
  applicable (clusterability, robustness-clustering).
- Smoke tests pinning return-shape contract.

## [0.2.0]

- Internal: refactored test entrypoints; documentation pass.

## [0.1.x]

- Initial public releases on PyPI under the legacy `ml4h-latentverse`
  distribution name (now orphaned).
