# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims for [Semantic Versioning](https://semver.org/).

## [0.3.5]

### Fixed

- **Disentanglement informativeness is now deterministic.** The binary branch of
  `compute_informativeness_score` fit `fit_logistic` (elastic-net
  LogisticRegression, `solver="saga"`) without `random_state`, so SAGA's
  per-epoch shuffling drew from the global numpy RNG. On fits that stop short of
  convergence the AUROC moved at reported precision with the ambient global
  state (measured 0.46822 / 0.46814 / 0.46818 across three seeds), with no
  `ConvergenceWarning`. Threads the already-in-scope `random_state` into the
  solver; converged fits are bit-identical, so well-behaved data is unchanged.
  **Metric-value change for non-converged fits (a bug fix — the old values were
  non-reproducible draws).**

- **CSV embeddings with empty cells are now imputed, not rejected.** The
  pyarrow-backed reader surfaces empty cells as `pd.NA`; `astype(float64)` raised
  `TypeError` on `pd.NA` before the documented mean-imputation could run, so such
  uploads were rejected with a misleading "non-numeric values" error. Fixed in
  both `pipeline._sanitize_features` and the web app's `TestRunner`.

### Added

- **Fired row caps are disclosed via a `"Rows evaluated"` annotation.** The
  supervised paths silently downsample above the 5k/10k caps; the pipeline now
  reports `"N of M (subsampled for speed)"` on a fired cap, matching the web
  app's key/format (closing a latent CLI↔web parity gap).

## [Unreleased]

### Added

- **`latentverse` command-line interface.** A `latentverse` console script
  (equivalently `python -m latentverse`) runs the five core evaluations
  (`clusterability`, `disentanglement`, `expressiveness`, `robustness`,
  `probing`) and the multimodal decoupling over a representations file
  (`.csv`/`.tsv`/`.npy`), with optional labels. It reproduces the LatentVerse
  web application's numbers by driving a duplicated orchestration pipeline
  (`latentverse.pipeline`) — the same ID-column merge, label coercion,
  cluster-count heuristic, sanitisation, standardisation, deterministic
  subsample, row caps, and overridden expressiveness/robustness sweeps the web
  tier applies — and calling the same `latentverse.evaluations.*` metric
  functions. A cross-parity test (`tests/test_cross_parity.py`) asserts the CLI
  and the web app agree to `rtol=1e-9`. See `latentverse --help`.
  - Reproducibility note: pin `LATENTVERSE_N_JOBS=1` (and single-threaded BLAS)
    for bit-stable `expressiveness` — its fold RNG is not thread-safe at
    `n_jobs > 1`. Multimodal runs use a canonical CPU reference config.

### Changed (breaking)

- **Import rename**: the Python import name is now `latentverse`
  (was `ml4h_latentverse`).
  - Old: `from ml4h_latentverse import ...`
  - New: `from latentverse import ...`
  - **Installation is from source** — the name `latentverse` is **not yet
    published to PyPI**, so `pip install latentverse` does not work today.
    Install from a checkout (`pip install .`, or `pip install -e ".[dev]"` for
    development) or directly from the repository
    (`pip install "git+https://github.com/broadinstitute/ml4h-latentverse@<ref>"`).
    The previous PyPI project (`ml4h-latentverse`, last release 0.1.2) is
    orphaned under an inaccessible account; a release under the new name is
    planned but has not happened yet.
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

## [0.3.2] - 2026-07

Correctness fix for categorical disentanglement factors + a loss-free probing
speedup.

### Fixed

- **Disentanglement now picks its estimator family by task type, not by a raw
  "&gt;2 distinct values" test.** `run_disentanglement` used
  `is_continuous = len(np.unique(labels)) > 2`, so a nominal multiclass factor
  (e.g. a 4-class label a host encoded as 0/1/2/3) was fed to the *regression*
  estimators (`mutual_info_regression`, per-dim `LinearRegression` R² for SAP,
  `MLPRegressor` pseudo-R² for Informativeness). That imposes an arbitrary
  ordinal spacing on the class codes, making DCI-Informativeness and SAP depend
  on the meaningless encoding order (empirically ±0.23 for Informativeness under
  a relabel of the same 4 classes). It now uses
  `detect_task_type(labels) == "regression"` — the same detector probing and
  expressiveness already use — so binary/multiclass factors route to the
  classification estimators (`mutual_info_classif`, classifier-accuracy SAP,
  classifier Informativeness), matching the DCI/MIG/SAP definitions. Binary and
  genuinely-continuous factors are unaffected. Categorical-factor DCI-Disentanglement/
  Completeness/MIG remain near-0 on a single factor — that is single-factor
  structural degeneracy, not the estimator.
- **Multiclass Informativeness no longer crashes.** The classification branch of
  `compute_informativeness_score` called the binary-only `fit_logistic`
  (`predict_proba[:, 1]` + binary `roc_auc`), which raised on &gt;2 classes. It
  now uses a matched-capacity `MLPClassifier` + macro one-vs-rest AUROC for
  multiclass (reduces to standard AUROC at 2 classes; accuracy fallback when a
  rare class is absent from the test split). Binary keeps the established
  `fit_logistic` path unchanged.

### Changed (non-breaking)

- **Probing scores every metric in one CV pass per model.** `run_probing`
  issued a separate `cross_val_score` per scorer (binary: `roc_auc`+`accuracy`;
  multiclass: `accuracy`+`f1_macro`), and each call refit all models on all
  folds — training every model twice on those paths. It now uses a single
  `cross_validate(scoring=[...])`, which scores all metrics on the same fitted
  folds. Results are **byte-identical** (same splitter + seed ⇒ identical folds
  and fits); measured ~2× faster on the 2-scorer paths (regression, single
  scorer, is unchanged).

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
