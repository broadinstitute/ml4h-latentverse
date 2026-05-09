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

from ml4h_latentverse import (
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
# ---------------------------------------------------------------------------
def test_run_clustering_with_labels_returns_four_metrics():
    reps, labels = _make_separable(seed=1)
    out = run_clustering(reps, labels, random_state=42)
    # The library returns a flat dict of metrics. The web-app adapter is
    # the only place that wraps this further.
    expected = {
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Normalized Mutual Information",
        "Cluster Learnability",
    }
    assert expected.issubset(out.keys()), f"missing metric(s): {expected - set(out.keys())}"
    assert isinstance(out["Silhouette Score"], (float, np.floating))


def test_run_clustering_without_labels_falls_back_to_intrinsic_only():
    reps, _ = _make_separable(seed=2)
    out = run_clustering(reps, labels=None, num_clusters=3, random_state=42)
    assert isinstance(out["Silhouette Score"], (float, np.floating))
    # Label-dependent metrics may be None / absent when no labels are given.
    assert out.get("Normalized Mutual Information") in (None, *(np.nan,))


def test_run_clustering_random_state_is_reproducible():
    reps, labels = _make_separable(seed=3)
    a = run_clustering(reps, labels, random_state=7)
    b = run_clustering(reps, labels, random_state=7)
    for k in ("Silhouette Score", "Davies-Bouldin Index"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), (
            f"non-reproducible metric: {k} ({a[k]} vs {b[k]})"
        )


# ---------------------------------------------------------------------------
# Disentanglement
# ---------------------------------------------------------------------------
def test_run_disentanglement_returns_dci_mig_tc():
    reps, labels = _make_separable(n=400, dim=8, k=4, seed=4)
    out = run_disentanglement(reps, labels, random_state=42)
    assert "DCI" in out
    assert "MIG" in out
    assert "TC" in out


# ---------------------------------------------------------------------------
# Expressiveness
# ---------------------------------------------------------------------------
def test_run_expressiveness_runs_without_plots():
    reps, labels = _make_separable(n=300, dim=8, k=3, seed=5)
    out = run_expressiveness(
        reps, labels, percent_to_remove_list=[0, 20], plots=False, random_state=42
    )
    assert "metrics" in out


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def test_run_probing_returns_metrics_and_plot_data():
    reps, labels = _make_separable(n=300, dim=8, k=2, seed=6)
    out = run_probing(reps, labels, random_state=42)
    # Library returns a tuple (metrics, plot_data) per the new probing rewrite.
    # The web app adapter unpacks this into its JSON response.
    if isinstance(out, tuple):
        metrics, plot_data = out
        assert isinstance(metrics, dict)
        assert "Accuracy" in metrics
    else:
        assert isinstance(out, dict)
        assert "Accuracy" in out or "metrics" in out


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
    from ml4h_latentverse.multiloreft import MultiLoReFT  # noqa: F401
