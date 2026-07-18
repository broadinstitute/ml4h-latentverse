"""Standalone evaluation pipeline — a faithful DUPLICATE of the LatentVerse
web application's number-affecting orchestration.

Why this module exists
-----------------------
Every metric the LatentVerse paper reports is produced not by a raw
``latentverse.evaluations.run_*`` call but by a layer of web-app orchestration
that sits on top of it: an ID-column merge, a NaN-label drop, label
encoding/coercion, a ``num_clusters`` heuristic, non-finite sanitisation,
optional standardisation, a deterministic pre-run subsample, per-metric row
caps, and two *overridden* sweep defaults (expressiveness percent list and
robustness noise levels). A CLI that called ``run_*`` directly would therefore
produce **different numbers** than the web UI.

This module reproduces that orchestration inside the library so the CLI matches
the web numbers, and a cross-parity test (``tests/test_cross_parity.py``) asserts
the two stay byte-for-byte in agreement. The duplication is deliberate
(architecture "Option (c)"): the production web app is *not* modified; drift is
*caught* by the parity test rather than *prevented* by single-sourcing.

Provenance — every transformation below mirrors a specific web-app site:
    app/comparison_service.py  (process_comparison / _process_multimodal)
    app/test_runner.py         (TestRunner._run_single_label_test / _execute_test)
    app/ml_adapter.py          (MLAdapter.run_*)
    app/data_processor.py      (DataProcessor.load_* / merge_dataframes / encode_*)
    app/config.py              (Config.*_SAMPLE_THRESHOLD, MULTILOREFT_*)

What is deliberately NOT duplicated (none of it touches the five numbers):
    async orchestration, plot generation, progress callbacks, provenance blocks,
    the multimodal artifact cache, label-alignment warnings, and the 4-decimal
    UI string formatting. This module returns RAW numeric metrics.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — read from the SAME environment variables the web app's Config reads,
# with identical defaults, so a run under a pinned environment matches the web
# tier. Read lazily (per call) rather than at import so a test can set the env
# and get the same value the web app's import-time read would.
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def pin_deterministic_defaults() -> None:
    """Default this process to REPRODUCIBLE evaluation.

    HISTORY: through 0.3.3, ``run_expressiveness`` seeded and used the GLOBAL
    numpy RNG inside ``Parallel(backend="threading")`` folds, so at
    ``LATENTVERSE_N_JOBS > 1`` the per-fold shuffles (and the stochastic SAGA
    solver) raced and the reported numbers were NOT reproducible. Fixed in
    0.3.4: each fold owns a private ``RandomState`` that also drives its
    solver, so results are deterministic — and identical to the legacy
    ``N_JOBS=1`` values — at any ``LATENTVERSE_N_JOBS``.

    We keep the ``N_JOBS=1`` default anyway as a conservative baseline (the
    n_jobs-invariance is asserted by tests on the expressiveness path, not
    proven for every future metric). ``setdefault`` respects an explicit
    override, so callers can raise it for speed without losing determinism.
    """
    os.environ.setdefault("LATENTVERSE_N_JOBS", "1")


def _supervised_sample_threshold() -> int:
    # app/config.py: Config.SUPERVISED_SAMPLE_THRESHOLD (disentanglement,
    # expressiveness). Default 5000.
    return _env_int("SUPERVISED_SAMPLE_THRESHOLD", 5000)


def _probing_sample_threshold() -> int:
    # app/config.py: Config.PROBING_FAST_SAMPLE_THRESHOLD. Default 10000.
    return _env_int("PROBING_FAST_SAMPLE_THRESHOLD", 10000)


def _probing_full_cv_folds() -> int:
    # app/config.py: Config.PROBING_FULL_CV_FOLDS. Default 3.
    return _env_int("PROBING_FULL_CV_FOLDS", 3)


# app/comparison_service.py: _MAX_NUM_CLUSTERS (Finding #6 clamp). Default 1000.
def _max_num_clusters() -> int:
    return _env_int("MAX_NUM_CLUSTERS", 1000)


# app/ml_adapter.py:177 — the web app OVERRIDES the library's expressiveness
# default ([0, 5, 10, 20]) with this sweep.
DEFAULT_PERCENT_REMOVED: List[int] = [0, 10, 20, 30, 40, 50]

# app/ml_adapter.py:199 — the library's run_robustness has NO default for
# noise_levels (a required positional); the web app always passes this.
DEFAULT_NOISE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]

_ROW_ID = "__row_id__"

CORE_TESTS = (
    "clusterability",
    "disentanglement",
    "expressiveness",
    "robustness",
    "probing",
)


# ---------------------------------------------------------------------------
# Data loading — duplicates app/data_processor.py. For files below the web app's
# 50 MB polars threshold (the only regime a CLI fixture hits) the web app reads
# via pandas with ``dtype_backend="pyarrow"``; we mirror that so dtypes — and
# therefore the merge / NaN handling — line up exactly.
# ---------------------------------------------------------------------------
try:  # pyarrow is a hard dep of the web app's reader; mirror its presence.
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False


def _detect_delimiter(first_line: str) -> str:
    """app/data_processor.py:_detect_delimiter (single source of truth)."""
    tab, comma, semi = (
        first_line.count("\t"),
        first_line.count(","),
        first_line.count(";"),
    )
    if tab > comma and tab > semi:
        return "\t"
    if comma > tab and comma > semi:
        return ","
    if semi > 0:
        return ";"
    return ","


def _is_headerless_csv(file_path: str) -> bool:
    """app/data_processor.py:_is_headerless_csv (first row all-numeric)."""
    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
    except OSError:
        return False
    if not first_line:
        return False
    delimiter = _detect_delimiter(first_line)
    for val in first_line.split(delimiter):
        val = val.strip()
        if not val:
            continue
        try:
            float(val)
        except ValueError:
            if val.lower() != "nan":
                return False
    return True


def _read_csv(file_path: str) -> pd.DataFrame:
    """Mirror app/fast_csv_reader.read_csv_fast's small-file pandas path plus
    app/data_processor.load_csv's headerless synthesis + __row_id__ insert."""
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    delimiter = _detect_delimiter(first_line)
    headerless = _is_headerless_csv(file_path)
    read_kwargs: Dict[str, Any] = {
        "sep": delimiter,
        "engine": "c",
        "low_memory": False,
        "header": None if headerless else 0,
    }
    if _HAS_PYARROW:
        read_kwargs["dtype_backend"] = "pyarrow"
    df = pd.read_csv(file_path, **read_kwargs)
    if headerless:
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    if _ROW_ID not in df.columns:
        df.insert(0, _ROW_ID, range(len(df)))
    return df


def _load_npy(file_path: str) -> pd.DataFrame:
    """app/data_processor.load_npy (allow_pickle=False, col_i names, __row_id__)."""
    arr = np.asarray(np.load(file_path, allow_pickle=False))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"NPY embeddings must be 1D or 2D; got shape {tuple(arr.shape)}.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"NPY embeddings must be numeric; got dtype '{arr.dtype}'.")
    df = pd.DataFrame(np.ascontiguousarray(arr), columns=[f"col_{i}" for i in range(arr.shape[1])])
    df.insert(0, _ROW_ID, range(len(df)))
    return df


def load_representation_file(file_path: str, id_column: Optional[str] = None) -> pd.DataFrame:
    """app/data_processor.load_representation_file (dispatch on extension)."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        return _load_npy(file_path)
    return _read_csv(file_path)


def _normalize_id_series(s: pd.Series) -> pd.Series:
    """app/data_processor.normalize_id_series (merge-safe string key)."""
    text = s.astype(str).str.strip()
    return text.str.replace(r"^([+-]?\d+)\.0+$", r"\1", regex=True)


def _merge_dataframes(
    rep_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    rep_id_col: str,
    labels_id_col: str,
    how: str = "inner",
) -> pd.DataFrame:
    """app/data_processor.merge_dataframes — latent_* rename + normalized join.

    Operates on copies (the web app renames in place; we avoid mutating the
    caller's frames, which is behaviourally identical for the resulting merge)."""
    rep_df = rep_df.copy()
    labels_df = labels_df.copy()
    latent_cols = [c for c in rep_df.columns if c not in {rep_id_col, _ROW_ID}]
    rep_df.rename(columns={c: f"latent_{i}" for i, c in enumerate(latent_cols)}, inplace=True)
    rep_df[rep_id_col] = _normalize_id_series(rep_df[rep_id_col])
    labels_df[labels_id_col] = _normalize_id_series(labels_df[labels_id_col])
    return pd.merge(
        rep_df,
        labels_df,
        left_on=rep_id_col,
        right_on=labels_id_col,
        how=how,
        copy=False,
    )


def _prepare_representation_only_df(rep_df: pd.DataFrame, rep_id_col: Optional[str]) -> pd.DataFrame:
    """app/comparison_service._prepare_representation_only_df."""
    id_like = set()
    if rep_id_col:
        id_like.add(rep_id_col)
    if _ROW_ID in rep_df.columns:
        id_like.add(_ROW_ID)
    feature_cols = [c for c in rep_df.columns if c not in id_like]
    if not feature_cols:
        raise ValueError("No representation feature columns available after ID removal.")
    rename = {c: f"latent_{i}" for i, c in enumerate(feature_cols)}
    return rep_df.rename(columns=rename).copy()


def _extract_latent_and_labels(merged_df: pd.DataFrame, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """app/data_processor.extract_latent_and_labels (drops NaN-label rows)."""
    representations = merged_df.filter(regex="^latent_").to_numpy()
    labels = merged_df[label_column].to_numpy()
    mask = ~pd.isna(labels)
    if hasattr(mask, "to_numpy"):
        mask = mask.to_numpy()
    return representations[mask], labels[mask]


# ---------------------------------------------------------------------------
# Preprocessing — duplicates app/test_runner.py static helpers.
# ---------------------------------------------------------------------------
def _estimate_num_clusters(representations: np.ndarray, labels: Optional[np.ndarray] = None) -> int:
    """app/test_runner.TestRunner._estimate_num_clusters."""
    n_samples = max(2, int(representations.shape[0]))
    if labels is not None:
        valid = labels[~pd.isna(labels)]
        if len(valid) >= 2:
            n_unique = int(len(np.unique(valid)))
            if n_unique <= 2:
                return 2
            if n_unique <= 20:
                return n_unique
            return 4
    return int(max(2, min(10, round(np.sqrt(n_samples)))))


def _encode_optional_labels(labels: np.ndarray) -> np.ndarray:
    """app/test_runner.TestRunner._encode_optional_labels (NaN preserved)."""
    labels = np.asarray(labels)
    encoded = np.full(labels.shape[0], np.nan, dtype=np.float64)
    valid_mask = ~pd.isna(labels)
    if valid_mask.any():
        _, inverse = np.unique(labels[valid_mask], return_inverse=True)
        encoded[valid_mask] = inverse.astype(np.float64)
    return encoded


def _to_numeric_labels(labels: np.ndarray) -> np.ndarray:
    """app/test_runner.TestRunner._to_numeric_labels (supervised coercion)."""
    labels = np.asarray(labels)
    if labels.dtype != object and np.issubdtype(labels.dtype, np.number):
        return labels.astype(np.float64)
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=np.float64)
    original_missing = pd.isna(labels)
    failed_parse = np.isnan(numeric) & ~np.asarray(original_missing)
    if not failed_parse.any():
        return numeric
    return _encode_optional_labels(labels)


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """app/data_processor.encode_labels (factorise; require >= 2 classes)."""
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    num_unique = len(unique_labels)
    if num_unique < 2:
        raise ValueError(
            f"Label column has only {num_unique} unique value(s); at least 2 "
            "distinct classes are required for label-based metrics."
        )
    return inverse, num_unique


def _sanitize_features(representations: np.ndarray) -> np.ndarray:
    """app/test_runner.TestRunner._sanitize_features (finite float64, mean-impute).

    Returns the clean matrix only (the web app also returns a user-facing note,
    which does not affect any metric value)."""
    representations = np.asarray(representations)
    if representations.dtype == object or not np.issubdtype(representations.dtype, np.floating):
        try:
            representations = representations.astype(np.float64)
        except (ValueError, TypeError):
            raise ValueError("The representation/embedding matrix contains non-numeric values.")
    non_finite = ~np.isfinite(representations)
    if int(non_finite.sum()):
        representations = representations.copy()
        representations[non_finite] = np.nan
        col_mean = np.nanmean(representations, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nan_rows, nan_cols = np.where(np.isnan(representations))
        representations[nan_rows, nan_cols] = np.take(col_mean, nan_cols)
    return representations


def _maybe_standardize(representations: np.ndarray, enabled: bool) -> np.ndarray:
    """app/test_runner.TestRunner._maybe_standardize."""
    if not enabled:
        return representations
    if representations.dtype == object or not np.issubdtype(representations.dtype, np.floating):
        representations = np.asarray(representations, dtype=np.float64)
    mean = representations.mean(axis=0, keepdims=True)
    std = representations.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (representations - mean) / std


def _subsample_rows(
    representations: np.ndarray, labels: np.ndarray, threshold: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """app/test_runner.TestRunner._subsample_rows (seeded, sorted indices)."""
    n = int(representations.shape[0])
    if threshold and threshold > 0 and n > threshold:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=threshold, replace=False)
        idx.sort()
        labels = np.asarray(labels)
        return representations[idx], labels[idx], n, True
    return representations, labels, n, False


def _unwrap_metrics(raw: Any) -> Dict[str, Any]:
    """app/ml_adapter.MLAdapter._unwrap_metrics (tolerate results/metrics)."""
    if not isinstance(raw, dict):
        return {}
    if "results" in raw and isinstance(raw["results"], dict):
        return raw["results"]
    if "metrics" in raw and isinstance(raw["metrics"], dict):
        return raw["metrics"]
    return raw


# ---------------------------------------------------------------------------
# Per-test execution — duplicates the adapter calls (app/ml_adapter.py) plus the
# runner's per-test wrapping (app/test_runner.py). Returns RAW metric dicts.
# ---------------------------------------------------------------------------
def _run_clusterability(
    representations: np.ndarray,
    labels: Optional[np.ndarray],
    num_clusters: int,
    random_state: int,
) -> Dict[str, Any]:
    from latentverse.evaluations.clustering import run_clustering

    raw = run_clustering(
        representations=representations,
        labels=labels,
        num_clusters=num_clusters,
        plots=False,
        random_state=random_state,
    )
    metrics = dict(_unwrap_metrics(raw))
    # app/test_runner._run_clusterability: single-class / unlabeled → intrinsic
    # metrics only (the library still returns the NMI/CL slots as None).
    valid_labels = labels[~pd.isna(labels)] if labels is not None else np.asarray([])
    if labels is None or len(valid_labels) == 0 or len(np.unique(valid_labels)) < 2:
        metrics = {
            "Silhouette Score": metrics.get("Silhouette Score"),
            "Davies-Bouldin Index": metrics.get("Davies-Bouldin Index"),
        }
    metrics["Clusters (k)"] = int(num_clusters)
    return metrics


def _run_disentanglement(representations: np.ndarray, labels: np.ndarray, random_state: int) -> Dict[str, Any]:
    from latentverse.evaluations.disentanglement import run_disentanglement

    representations, labels, _, _ = _subsample_rows(
        representations, labels, _supervised_sample_threshold(), random_state
    )
    # app/ml_adapter.run_disentanglement: labels.reshape(-1, 1).
    raw = run_disentanglement(representations, np.asarray(labels).reshape(-1, 1), random_state=random_state)
    return dict(_unwrap_metrics(raw))


def _run_disentanglement_multifactor(
    representations: np.ndarray, factors: np.ndarray, random_state: int
) -> Dict[str, Any]:
    from latentverse.evaluations.disentanglement import run_disentanglement

    factors = np.asarray(factors)
    if factors.ndim != 2 or factors.shape[1] < 2:
        raise ValueError("Multi-factor disentanglement requires at least two label columns.")
    raw = run_disentanglement(representations, factors, random_state=random_state)
    return dict(_unwrap_metrics(raw))


def _run_expressiveness(
    representations: np.ndarray,
    labels: np.ndarray,
    percent_removed: List[int],
    random_state: int,
) -> Dict[str, Any]:
    from latentverse.evaluations.expressiveness import run_expressiveness

    representations, labels, _, _ = _subsample_rows(
        representations, labels, _supervised_sample_threshold(), random_state
    )
    raw = run_expressiveness(
        representations=representations,
        labels=np.asarray(labels).reshape(-1, 1),
        percent_to_remove_list=percent_removed,
        plots=False,
        random_state=random_state,
    )
    return dict(_unwrap_metrics(raw))


def _run_robustness(
    representations: np.ndarray,
    labels: Optional[np.ndarray],
    metric: str,
    num_clusters: Optional[int],
    noise_levels: List[float],
    random_state: int,
) -> Dict[str, Any]:
    from latentverse.evaluations.robustness import run_robustness

    if labels is None and metric != "clustering":
        raise ValueError("Labels are required for robustness probing mode.")
    raw = run_robustness(
        representations=representations,
        labels=labels,
        noise_levels=noise_levels,
        metric=metric,
        plots=False,
        num_clusters=num_clusters,
        random_state=random_state,
    )
    metrics = dict(_unwrap_metrics(raw))
    if isinstance(metrics, dict) and len(metrics) == 0:
        raise ValueError("Robustness produced no scores — every noise level failed to evaluate.")
    return metrics


def _run_probing(representations: np.ndarray, labels: np.ndarray, random_state: int) -> Dict[str, Any]:
    from latentverse.evaluations.probing import run_probing

    n_samples = int(representations.shape[0])
    threshold = _probing_sample_threshold()
    if threshold > 0 and n_samples > threshold:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=threshold, replace=False)
        idx.sort()
        representations = representations[idx]
        labels = labels[idx]
    raw = run_probing(
        representations,
        np.asarray(labels).ravel(),
        n_folds=_probing_full_cv_folds(),
        random_state=random_state,
    )
    return dict(_unwrap_metrics(raw))


# ---------------------------------------------------------------------------
# Runner-level orchestration — duplicates app/test_runner._run_single_label_test
# + _execute_test for one label column.
# ---------------------------------------------------------------------------
def _run_single_label(
    merged_df: pd.DataFrame,
    test_type: str,
    label_col: str,
    robustness_metric: Optional[str],
    random_state: int,
    standardize: bool,
    num_clusters_override: Optional[int],
    percent_removed: List[int],
    noise_levels: List[float],
) -> Dict[str, Any]:
    is_clustering_branch = test_type == "clusterability" or (
        test_type == "robustness" and robustness_metric == "clustering"
    )

    if is_clustering_branch:
        representations = _sanitize_features(np.asarray(merged_df.filter(regex="^latent_").to_numpy()))
        representations = _maybe_standardize(representations, standardize)
        labels = np.asarray(merged_df[label_col].to_numpy())
        labels_encoded = _encode_optional_labels(labels)
        if num_clusters_override is not None:
            num_clusters = num_clusters_override
        else:
            num_clusters = _estimate_num_clusters(representations, labels=labels)

        if test_type == "clusterability":
            return _run_clusterability(representations, labels_encoded, num_clusters, random_state)
        return _run_robustness(
            representations,
            labels_encoded,
            "clustering",
            num_clusters,
            noise_levels,
            random_state,
        )

    # Supervised branch (disentanglement / expressiveness / probing /
    # robustness-probing): NaN-label drop, sanitize, standardize, numeric coerce.
    representations, labels = _extract_latent_and_labels(merged_df, label_col)
    representations = np.asarray(representations)
    representations = _sanitize_features(representations)
    representations = _maybe_standardize(representations, standardize)
    labels = _to_numeric_labels(np.asarray(labels))
    # app/test_runner:796 encode_labels is called (raises on < 2 classes) even
    # though only `labels` is passed to the supervised metric.
    _encode_labels(labels)

    if test_type == "disentanglement":
        return _run_disentanglement(representations, labels, random_state)
    if test_type == "expressiveness":
        return _run_expressiveness(representations, labels, percent_removed, random_state)
    if test_type == "robustness":
        return _run_robustness(representations, labels, robustness_metric, None, noise_levels, random_state)
    if test_type == "probing":
        return _run_probing(representations, labels, random_state)
    raise ValueError(f"Unknown test type: {test_type}")


def _run_unlabeled(
    merged_df: pd.DataFrame,
    test_type: str,
    robustness_metric: Optional[str],
    random_state: int,
    standardize: bool,
    num_clusters_override: Optional[int],
    noise_levels: List[float],
) -> Dict[str, Any]:
    """app/test_runner._run_unlabeled_test (intrinsic clustering only)."""
    representations = _sanitize_features(np.asarray(merged_df.filter(regex="^latent_").to_numpy()))
    if representations.shape[0] < 2:
        raise ValueError(f"Need at least 2 representation rows for clustering, got {representations.shape[0]}.")
    representations = _maybe_standardize(representations, standardize)
    num_clusters = (
        num_clusters_override if num_clusters_override is not None else _estimate_num_clusters(representations)
    )
    if test_type == "clusterability":
        return _run_clusterability(representations, None, num_clusters, random_state)
    if test_type == "robustness" and robustness_metric == "clustering":
        return _run_robustness(representations, None, "clustering", num_clusters, noise_levels, random_state)
    raise ValueError("Unsupported unlabeled mode.")


# ---------------------------------------------------------------------------
# Top-level configuration + entrypoint — duplicates the number-affecting part of
# app/comparison_service.process_comparison.
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Everything the web app derives from a comparison request that changes the
    numbers. Field names mirror the web app's ``comp_data`` keys where useful."""

    test_type: str
    representations_path: str
    labels_path: Optional[str] = None
    rep_id_col: Optional[str] = None
    labels_id_col: Optional[str] = None
    label_cols: List[str] = field(default_factory=list)
    robustness_metric: Optional[str] = None
    random_seed: int = 42
    standardize: bool = False
    subsample_rows: Optional[int] = None
    num_clusters_override: Optional[int] = None
    percent_removed: List[int] = field(default_factory=lambda: list(DEFAULT_PERCENT_REMOVED))
    noise_levels: List[float] = field(default_factory=lambda: list(DEFAULT_NOISE_LEVELS))
    # Multimodal
    representations2_path: Optional[str] = None
    rep2_id_col: Optional[str] = None

    def __post_init__(self) -> None:
        # app/comparison_service.py:809-823 — sub-2 subsample dropped; k < 2
        # dropped; k clamped to _MAX_NUM_CLUSTERS.
        if self.subsample_rows is not None and self.subsample_rows < 2:
            self.subsample_rows = None
        if self.num_clusters_override is not None:
            if self.num_clusters_override < 2:
                self.num_clusters_override = None
            else:
                self.num_clusters_override = min(self.num_clusters_override, _max_num_clusters())


def _missing_column(kind: str, name: str, df: pd.DataFrame) -> ValueError:
    available = ", ".join(repr(str(c)) for c in list(df.columns)[:25])
    return ValueError(
        f"{kind} column '{name}' not found. Available columns: {available}."
    )


def _load_and_merge(cfg: PipelineConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Reproduce process_comparison's load → merge → deterministic subsample."""
    rep_df = load_representation_file(cfg.representations_path, cfg.rep_id_col)
    selected_labels = list(cfg.label_cols)

    labels_required = _labels_required(cfg.test_type, cfg.robustness_metric)
    has_labels = bool(cfg.labels_path and selected_labels)
    if labels_required and not has_labels:
        raise ValueError(f"Test '{cfg.test_type}' requires labels: pass --labels and --label-cols.")

    # Validate user-supplied column names up front: the web app's UI offers
    # only existing columns in dropdowns, but a CLI user can typo one — turn
    # the would-be KeyError deep in the merge into an actionable message.
    if cfg.rep_id_col and cfg.rep_id_col not in rep_df.columns:
        raise _missing_column("Representation id (--id-col)", cfg.rep_id_col, rep_df)

    if has_labels:
        if not cfg.rep_id_col:
            raise ValueError("A representation id column (--id-col) is required to merge labels.")
        labels_df = _read_csv(cfg.labels_path)
        labels_id_col = cfg.labels_id_col or cfg.rep_id_col
        if labels_id_col not in labels_df.columns:
            raise _missing_column("Labels id (--labels-id-col)", labels_id_col, labels_df)
        for col in selected_labels:
            if col not in labels_df.columns:
                raise _missing_column("Label (--label-cols)", col, labels_df)
        # merge_how: left for clustering modes (keep all rep rows), inner else.
        merge_how = (
            "left"
            if cfg.test_type == "clusterability"
            or (cfg.test_type == "robustness" and cfg.robustness_metric == "clustering")
            else "inner"
        )
        merged_df = _merge_dataframes(rep_df, labels_df, cfg.rep_id_col, labels_id_col, how=merge_how)
        if merge_how == "inner" and len(merged_df) == 0:
            raise ValueError(
                "No ids matched when merging representations with labels — the id columns share no values."
            )
    else:
        merged_df = _prepare_representation_only_df(rep_df, cfg.rep_id_col)
        selected_labels = []

    # process_comparison:827 — deterministic pre-run subsample (seeded).
    if cfg.subsample_rows is not None and len(merged_df) > cfg.subsample_rows:
        merged_df = merged_df.sample(n=cfg.subsample_rows, random_state=cfg.random_seed).reset_index(drop=True)

    return merged_df, selected_labels


def _labels_required(test_type: str, robustness_metric: Optional[str]) -> bool:
    """app/comparison_service._labels_required_for_test."""
    if test_type in {"disentanglement", "expressiveness", "probing"}:
        return True
    if test_type == "robustness":
        return robustness_metric != "clustering"
    return False


def run_pipeline(cfg: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    """Run one core test and return ``{label_or_mode: raw_metrics_dict}``.

    Mirrors the shape of the web app's per-label results, but with RAW numeric
    metric values (no 4-decimal UI formatting)."""
    pin_deterministic_defaults()
    if cfg.test_type == "multimodal":
        raise ValueError("Use run_multimodal_pipeline for the multimodal test.")
    if cfg.test_type not in CORE_TESTS:
        raise ValueError(f"Unknown test '{cfg.test_type}'. Choose one of {CORE_TESTS}.")

    merged_df, selected_labels = _load_and_merge(cfg)

    if not selected_labels:
        # Unlabeled (intrinsic) mode — only clusterability / robustness-clustering.
        result = _run_unlabeled(
            merged_df,
            cfg.test_type,
            cfg.robustness_metric,
            cfg.random_seed,
            cfg.standardize,
            cfg.num_clusters_override,
            cfg.noise_levels,
        )
        return {"Intrinsic (No Labels)": result}

    # Multi-factor disentanglement (>= 2 label columns) — app/test_runner:304.
    if cfg.test_type == "disentanglement" and len(selected_labels) > 1:
        representations = _sanitize_features(np.asarray(merged_df.filter(regex="^latent_").to_numpy()))
        representations = _maybe_standardize(representations, cfg.standardize)
        factors = np.column_stack([_encode_optional_labels(merged_df[c].to_numpy()) for c in selected_labels])
        combined = " + ".join(selected_labels)
        return {combined: _run_disentanglement_multifactor(representations, factors, cfg.random_seed)}

    results: Dict[str, Dict[str, Any]] = {}
    for label_col in selected_labels:
        results[label_col] = _run_single_label(
            merged_df,
            cfg.test_type,
            label_col,
            cfg.robustness_metric,
            cfg.random_seed,
            cfg.standardize,
            cfg.num_clusters_override,
            cfg.percent_removed,
            cfg.noise_levels,
        )
    return results
