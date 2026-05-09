"""
Minimal smoke tests for the five public entrypoints + multiloreft.

These pin the *return shape*, not the exact metric values — value
assertions are dataset-dependent and would make CI flaky. Anyone
adding a new metric should update the asserted key set here so
downstream consumers (notably the web app) get a clear signal when
the contract changes.

Run:
    pytest tests/test_smoke.py -v
"""

import numpy as np
import pytest

from latentverse import (
    run_clustering,
    run_disentanglement,
    run_expressiveness,
    run_probing,
    run_robustness,
)


def _make_separable(n=200, dim=12, k=3, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    reps = centers[labels] + rng.normal(scale=1.0, size=(n, dim))
    return reps.astype(np.float32), labels.astype(np.float64)


# ---------------------------------------------------------------------------
# Clusterability
#
# Contract:
#   run_clustering returns {
#       "results": {"Silhouette Score", "Davies-Bouldin Index",
#                   "Normalized Mutual Information", "Cluster Learnability"},
#       "plot_url": str | None,
#       "representations_2d": np.ndarray (N, 2),
#       "pca_available": bool,
#       "cluster_labels": np.ndarray (N,),
#   }
# ---------------------------------------------------------------------------
CLUSTERING_METRICS = {
    "Silhouette Score",
    "Davies-Bouldin Index",
    "Normalized Mutual Information",
    "Cluster Learnability",
}


def test_run_clustering_with_labels_returns_four_metrics():
    reps, labels = _make_separable(seed=1)
    out = run_clustering(reps, labels, random_state=42)
    assert "results" in out
    metrics = out["results"]
    assert CLUSTERING_METRICS.issubset(metrics.keys()), (
        f"missing metric(s): {CLUSTERING_METRICS - set(metrics.keys())}"
    )
    assert isinstance(metrics["Silhouette Score"], (float, np.floating))


def test_run_clustering_without_labels_falls_back_to_intrinsic_only():
    reps, _ = _make_separable(seed=2)
    out = run_clustering(reps, labels=None, num_clusters=3, random_state=42)
    metrics = out["results"]
    assert isinstance(metrics["Silhouette Score"], (float, np.floating))
    # Label-dependent metrics should be None when no labels are given.
    assert metrics.get("Normalized Mutual Information") is None


def test_run_clustering_random_state_is_reproducible():
    reps, labels = _make_separable(seed=3)
    a = run_clustering(reps, labels, random_state=7)["results"]
    b = run_clustering(reps, labels, random_state=7)["results"]
    for k in ("Davies-Bouldin Index",):
        assert a[k] == pytest.approx(b[k], abs=1e-9), (
            f"non-reproducible metric: {k} ({a[k]} vs {b[k]})"
        )


def test_run_clustering_num_clusters_override_is_honoured():
    reps, labels = _make_separable(n=300, dim=8, k=3, seed=4)
    out = run_clustering(reps, labels, num_clusters=7, random_state=42)
    cluster_labels = out["cluster_labels"]
    assert len(set(cluster_labels.tolist())) == 7


# ---------------------------------------------------------------------------
# Disentanglement
#
# Contract:
#   run_disentanglement returns {
#       "metrics": {"DCI": {...}, "MIG": float, "SAP": float, "TC": float},
#       "plot_data": {...},
#   }
# ---------------------------------------------------------------------------
def test_run_disentanglement_returns_dci_mig_tc_sap():
    reps, labels = _make_separable(n=400, dim=8, k=4, seed=4)
    out = run_disentanglement(reps, labels, random_state=42)
    assert "metrics" in out
    metrics = out["metrics"]
    assert "DCI" in metrics
    assert "MIG" in metrics
    assert "TC" in metrics
    assert "SAP" in metrics


# ---------------------------------------------------------------------------
# Expressiveness
#
# Contract:
#   run_expressiveness returns {"metrics": {...}, ...}
# ---------------------------------------------------------------------------
def test_run_expressiveness_runs_without_plots():
    reps, labels = _make_separable(n=300, dim=8, k=3, seed=5)
    out = run_expressiveness(
        reps,
        labels,
        percent_to_remove_list=[0, 20],
        plots=False,
        random_state=42,
    )
    assert "metrics" in out


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def test_run_probing_returns_metrics_and_plot_data():
    reps, labels = _make_separable(n=300, dim=8, k=2, seed=6)
    out = run_probing(reps, labels, random_state=42)
    if isinstance(out, tuple):
        metrics, _ = out
        assert isinstance(metrics, dict)
    else:
        assert isinstance(out, dict)
        # Webapp consumes metrics under either top-level keys or "metrics".
        assert any(k in out for k in ("metrics", "Accuracy", "R2"))


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_run_robustness_clustering_metric():
    reps, labels = _make_separable(n=200, dim=8, k=3, seed=7)
    out = run_robustness(
        reps,
        labels,
        noise_levels=[0.1, 0.5],
        metric="clustering",
        plots=False,
        random_state=42,
    )
    assert "metrics" in out


# ---------------------------------------------------------------------------
# Multimodal (multiloreft) — light-weight import check; full training is
# expensive enough that we don't run it in CI by default.
# ---------------------------------------------------------------------------
def test_multiloreft_importable():
    from latentverse.multiloreft import MultiLoReFT  # noqa: F401
