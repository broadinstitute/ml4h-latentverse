"""Metric-contract tests for run_clustering — Cluster Learnability fixed points.

These exist so the metric's SEMANTICS can never be redefined silently again
(that happened once: the label-aware probe became a KMeans-id probe with no
test objecting, and the KMeans variant scored ~1.0 on pure noise).

Cluster Learnability contract (paper, Appendix C): balanced accuracy of a
logistic-regression probe (stratified 80/20, class_weight="balanced")
recovering GROUND-TRUTH labels. Fixed points asserted here:

  * structureless input + random labels  -> ~1/k (chance)
  * structureless input + imbalanced labels -> ~1/k (balanced accuracy must
    remove the majority-class prevalence artefact; plain accuracy would sit
    near the majority frequency instead)
  * well-separated class structure       -> ~1.0
  * no labels                            -> None (label-free runs rely on the
    intrinsic metrics)
  * noise must score clearly BELOW structure (dynamic range — the property
    the KMeans-id variant lacked)
"""

import numpy as np
import pytest

from latentverse.evaluations.clustering import run_clustering


def _noise(n=300, dim=8, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim))


def _blobs(n=300, dim=8, k=3, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, sep, (k, dim))
    labels = rng.integers(0, k, n)
    reps = centers[labels] + rng.normal(0, 0.3, (n, dim))
    return reps, labels


def _cl(reps, labels, k=3):
    out = run_clustering(reps, labels, num_clusters=k, random_state=42)
    return out["results"]["Cluster Learnability"]


def test_cl_is_chance_on_noise_with_random_labels():
    rng = np.random.default_rng(0)
    reps = _noise(seed=0)
    labels = rng.integers(0, 3, len(reps))
    cl = _cl(reps, labels, k=3)
    assert cl == pytest.approx(1.0 / 3.0, abs=0.15), (
        f"no signal => balanced accuracy must sit at ~1/k chance, got {cl}"
    )


def test_cl_not_inflated_by_class_imbalance():
    """The prevalence artefact: plain accuracy on a no-signal 80/20 label
    split sits near 0.8 (majority-class rate). Balanced accuracy must not."""
    rng = np.random.default_rng(0)
    reps = _noise(n=400, seed=1)
    labels = (rng.random(len(reps)) < 0.2).astype(int)  # ~80/20, no signal
    cl = _cl(reps, labels, k=2)
    assert cl == pytest.approx(0.5, abs=0.15), (
        f"imbalanced no-signal labels => chance 0.5 for balanced accuracy, got {cl} "
        "(a value near 0.8 means the majority-class prevalence artefact is back)"
    )


def test_cl_is_high_on_separable_class_structure():
    reps, labels = _blobs(sep=4.0)
    cl = _cl(reps, labels, k=3)
    assert cl >= 0.9, f"well-separated classes must be trivially recoverable, got {cl}"


def test_cl_separates_noise_from_structure():
    """Dynamic range — the property whose absence broke the KMeans-id variant."""
    rng = np.random.default_rng(0)
    reps_noise = _noise(seed=0)
    cl_noise = _cl(reps_noise, rng.integers(0, 3, len(reps_noise)), k=3)
    reps_blobs, labels_blobs = _blobs(sep=4.0)
    cl_blobs = _cl(reps_blobs, labels_blobs, k=3)
    assert cl_noise < cl_blobs - 0.2, (
        f"metric must put clear daylight between noise ({cl_noise}) and "
        f"structure ({cl_blobs})"
    )


def test_cl_is_none_without_labels():
    out = run_clustering(_noise(seed=0), labels=None, num_clusters=3, random_state=42)
    assert out["results"]["Cluster Learnability"] is None


def test_cl_deterministic_for_fixed_seed():
    reps, labels = _blobs(sep=1.0, seed=3)
    assert _cl(reps, labels) == _cl(reps, labels)
