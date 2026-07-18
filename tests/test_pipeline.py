"""Unit tests for the duplicated pipeline (``latentverse.pipeline``).

These pin the number-affecting transformations the pipeline duplicates from the
web app, INDEPENDENTLY of the cross-parity test (which needs the web app
checkout). They cover branches the small fixture keeps under caps — notably the
row-cap subsample RNG — by asserting against the exact seeded-RNG algorithm the
web app uses.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from latentverse import pipeline as pl

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
REP = os.path.join(FIXTURES, "rep.csv")
REP2 = os.path.join(FIXTURES, "rep2.csv")
LABELS = os.path.join(FIXTURES, "labels.csv")


def _base_cfg(test_type, **kw):
    return pl.PipelineConfig(
        test_type=test_type,
        representations_path=REP,
        labels_path=LABELS,
        rep_id_col="sample_id",
        labels_id_col="sample_id",
        label_cols=["group"],
        **kw,
    )


# ---------------------------------------------------------------------------
# Row-cap subsample RNG — the paper numbers came from CAPPED runs, but the
# fixture sits under every cap, so pin the exact seeded-RNG algorithm here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,threshold,seed", [(100, 40, 42), (5001, 5000, 7), (500, 250, 0)])
def test_subsample_rows_matches_seeded_rng(n, threshold, seed):
    reps = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    labels = np.arange(n)
    out_r, out_l, n_orig, subsampled = pl._subsample_rows(reps, labels, threshold, seed)

    expected_idx = np.random.default_rng(seed).choice(n, size=threshold, replace=False)
    expected_idx.sort()

    assert subsampled is True
    assert n_orig == n
    assert out_r.shape[0] == threshold
    np.testing.assert_array_equal(out_r, reps[expected_idx])
    np.testing.assert_array_equal(out_l, labels[expected_idx])


def test_subsample_rows_noop_below_threshold():
    reps = np.zeros((100, 2))
    labels = np.zeros(100)
    out_r, out_l, n_orig, subsampled = pl._subsample_rows(reps, labels, 5000, 42)
    assert subsampled is False and n_orig == 100 and out_r.shape[0] == 100


def test_probing_cap_triggers_and_is_deterministic(monkeypatch):
    """Force the probing cap below the fixture size and confirm the capped path
    runs and is reproducible (the cap is otherwise a no-op under the fixture)."""
    monkeypatch.setenv("PROBING_FAST_SAMPLE_THRESHOLD", "60")
    monkeypatch.setenv("LATENTVERSE_N_JOBS", "1")
    a = pl.run_pipeline(_base_cfg("probing"))["group"]
    b = pl.run_pipeline(_base_cfg("probing"))["group"]
    assert a["AUROC"] == b["AUROC"]  # deterministic under the fired cap


# ---------------------------------------------------------------------------
# num_clusters heuristic (app/test_runner._estimate_num_clusters).
# ---------------------------------------------------------------------------
def test_estimate_num_clusters_label_aware():
    reps = np.zeros((200, 4))
    assert pl._estimate_num_clusters(reps, labels=np.array([0, 1] * 100)) == 2
    assert pl._estimate_num_clusters(reps, labels=np.repeat(np.arange(5), 40)) == 5
    # > 20 unique (continuous) -> quartiles (4)
    assert pl._estimate_num_clusters(reps, labels=np.arange(200).astype(float)) == 4


def test_estimate_num_clusters_unlabeled_sqrt_bounded():
    assert pl._estimate_num_clusters(np.zeros((9, 2))) == 3  # sqrt(9)
    assert pl._estimate_num_clusters(np.zeros((10000, 2))) == 10  # capped at 10


# ---------------------------------------------------------------------------
# Label coercion / encoding (app/test_runner._to_numeric_labels / _encode_*).
# ---------------------------------------------------------------------------
def test_to_numeric_labels_passthrough_numeric():
    out = pl._to_numeric_labels(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(out, np.array([0.0, 1.0, 2.0]))


def test_to_numeric_labels_coerces_numeric_strings():
    out = pl._to_numeric_labels(np.array(["1", "0", "2"], dtype=object))
    np.testing.assert_array_equal(out, np.array([1.0, 0.0, 2.0]))


def test_to_numeric_labels_factorises_categoricals():
    out = pl._to_numeric_labels(np.array(["case", "control", "case"], dtype=object))
    assert set(np.unique(out)) == {0.0, 1.0}
    assert out[0] == out[2] and out[0] != out[1]


def test_encode_optional_labels_preserves_nan():
    out = pl._encode_optional_labels(np.array(["a", None, "b"], dtype=object))
    assert np.isnan(out[1])
    assert out[0] != out[2]


def test_encode_labels_rejects_single_class():
    with pytest.raises(ValueError, match="at least 2"):
        pl._encode_labels(np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Merge + NaN-label drop (app/data_processor.merge_dataframes / extract_*).
# ---------------------------------------------------------------------------
def test_inner_merge_drops_unmatched_then_nan_labels():
    # Supervised path: inner merge (110 matched of 120) then NaN-label drop (3).
    cfg = _base_cfg("probing")
    merged, labels = pl._load_and_merge(cfg)
    assert labels == ["group"]
    assert len(merged) == 110  # ids 110..119 have no label row
    reps, y = pl._extract_latent_and_labels(merged, "group")
    assert reps.shape[0] == 107  # minus 3 explicit NaN labels
    assert not pd.isna(y).any()


def test_left_merge_keeps_all_rows_for_clustering():
    cfg = _base_cfg("clusterability")
    merged, labels = pl._load_and_merge(cfg)
    assert len(merged) == 120  # left join keeps every representation row


# ---------------------------------------------------------------------------
# PipelineConfig clamps (app/comparison_service.py:809-823).
# ---------------------------------------------------------------------------
def test_config_drops_subsample_below_two():
    assert _base_cfg("clusterability", subsample_rows=1).subsample_rows is None


def test_config_drops_k_below_two_and_clamps_high(monkeypatch):
    assert _base_cfg("clusterability", num_clusters_override=1).num_clusters_override is None
    monkeypatch.setenv("MAX_NUM_CLUSTERS", "1000")
    cfg = _base_cfg("clusterability", num_clusters_override=99999)
    assert cfg.num_clusters_override == 1000


# ---------------------------------------------------------------------------
# End-to-end shape + overridden sweeps.
# ---------------------------------------------------------------------------
def test_all_core_tests_return_expected_keys(monkeypatch):
    monkeypatch.setenv("LATENTVERSE_N_JOBS", "1")
    r = pl.run_pipeline(_base_cfg("clusterability"))["group"]
    assert {"Silhouette Score", "Cluster Learnability"} <= set(r)

    r = pl.run_pipeline(_base_cfg("disentanglement"))["group"]
    assert {"DCI", "MIG", "SAP", "TC"} <= set(r)

    r = pl.run_pipeline(_base_cfg("expressiveness"))["group"]
    # Expressiveness nests one level deeper (library labels each factor column).
    inner = r[next(iter(r))]
    # Overridden expressiveness sweep [0,10,20,30,40,50] -> a "50% Removed" key.
    assert "50% Removed" in inner

    r = pl.run_pipeline(_base_cfg("probing"))["group"]
    assert "AUROC" in r and isinstance(r["AUROC"], list)

    r = pl.run_pipeline(_base_cfg("robustness", robustness_metric="probing"))["group"]
    # Overridden robustness noise levels [0.1..0.5] -> 5 scores per metric.
    first_list = next(v for v in r.values() if isinstance(v, list))
    assert len(first_list) == 5


def test_unlabeled_clusterability_intrinsic_only():
    cfg = pl.PipelineConfig(test_type="clusterability", representations_path=REP, rep_id_col="sample_id")
    r = pl.run_pipeline(cfg)
    assert "Intrinsic (No Labels)" in r
    metrics = r["Intrinsic (No Labels)"]
    assert "Silhouette Score" in metrics
    assert "Normalized Mutual Information" not in metrics  # no labels


def test_labels_required_raises_without_labels():
    cfg = pl.PipelineConfig(test_type="probing", representations_path=REP, rep_id_col="sample_id")
    with pytest.raises(ValueError, match="requires labels"):
        pl.run_pipeline(cfg)
