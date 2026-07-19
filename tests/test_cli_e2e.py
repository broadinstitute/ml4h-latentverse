"""END-TO-END tests of the standalone ``latentverse`` CLI surface.

Contract under test (the whole point of this suite):

  * the CLI NEVER crashes with a raw Python traceback — every error path
    exits non-zero with a user-actionable ``error: ...`` line on stderr;
  * it never emits a silently-wrong number — outputs are deterministic for a
    given ``--seed`` (including across ``LATENTVERSE_N_JOBS``), row caps
    change results only when they actually fire, and no-op flags are no-ops;
  * stdout is machine-readable: strict JSON (no bare ``NaN``/``Infinity``
    tokens — non-finite values are ``null``) or parseable CSV; progress
    chatter and warnings go to stderr;
  * both entry points — the ``latentverse`` console script and
    ``python -m latentverse`` — produce byte-identical output.

Everything here runs the REAL CLI in a subprocess (no direct function
calls), on top of the committed parity fixtures plus synthetic edge files
built in a session tmpdir.

Runtime notes: multimodal cases set ``MULTILOREFT_MAX_EPOCHS=2`` to keep the
MultiLoReFT trainings short — numeric parity for multimodal is asserted
elsewhere (``test_cross_parity.py``) under the canonical config; here we test
the CLI surface (exit codes, output shape, error text).
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
REP = os.path.join(FIXTURES, "rep.csv")
REP2 = os.path.join(FIXTURES, "rep2.csv")
LABELS = os.path.join(FIXTURES, "labels.csv")

CONSOLE_SCRIPT = os.path.join(os.path.dirname(sys.executable), "latentverse")

CORE_TESTS = ["clusterability", "disentanglement", "expressiveness", "robustness", "probing"]

# Deterministic runtime: BLAS single-threaded; N_JOBS default (1) applies.
PINNED = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
}

TIMEOUT = 600


def run_cli(*args: str, entry: str = "module", env: dict | None = None) -> subprocess.CompletedProcess:
    if entry == "module":
        cmd = [sys.executable, "-m", "latentverse", *args]
    elif entry == "script":
        cmd = [CONSOLE_SCRIPT, *args]
    else:  # pragma: no cover
        raise AssertionError(entry)
    full_env = dict(os.environ, **PINNED, **(env or {}))
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=TIMEOUT)


def _reject_constant(token: str):
    raise AssertionError(f"stdout JSON contains bare non-finite token {token!r}")


def parse_stdout_json(proc: subprocess.CompletedProcess) -> dict:
    """Strict-parse stdout: it must be EXACTLY one JSON document, with no
    stray warnings and no bare NaN/Infinity tokens."""
    assert proc.stdout.strip(), f"empty stdout; stderr:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout, parse_constant=_reject_constant)


def _find_rows_evaluated(payload: dict):
    """The 'Rows evaluated' cap annotation, wherever it lands in results
    (it's set per-label alongside the metric values). None when no cap fired."""
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    for metrics in results.values():
        if isinstance(metrics, dict) and "Rows evaluated" in metrics:
            return metrics["Rows evaluated"]
    return None


def assert_clean_error(proc: subprocess.CompletedProcess, *needles: str) -> None:
    assert proc.returncode != 0, f"expected failure, got exit 0; stdout:\n{proc.stdout[:500]}"
    assert "Traceback" not in proc.stderr, f"raw traceback leaked:\n{proc.stderr[-2000:]}"
    assert "error:" in proc.stderr, f"no 'error:' line on stderr:\n{proc.stderr[-2000:]}"
    for needle in needles:
        assert needle in proc.stderr, f"expected {needle!r} in stderr:\n{proc.stderr[-2000:]}"


def assert_finite_or_null(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            assert_finite_or_null(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            assert_finite_or_null(v, f"{path}[{i}]")
    elif isinstance(value, float):
        assert np.isfinite(value), f"{path}: non-finite float {value} leaked into output"


def labeled_args(test: str, label: str = "group", rep: str = REP) -> list[str]:
    return [test, "--representations", rep, "--labels", LABELS, "--id-col", "sample_id", "--label-cols", label]


# ---------------------------------------------------------------------------
# Synthetic edge-case files
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def edge(tmp_path_factory):
    d = tmp_path_factory.mktemp("edge")

    def w(name: str, text: str) -> str:
        p = d / name
        p.write_text(text)
        return str(p)

    files = {}
    files["empty"] = w("empty.csv", "")
    files["one_row"] = w("one_row.csv", "sample_id,f1,f2\n0,1.0,2.0\n")
    rows8 = "".join(f"{i},{1.0 + i * 0.3}\n" for i in range(8))
    files["one_feature"] = w("one_feature.csv", "sample_id,f1\n" + rows8)
    files["one_class_labels"] = w("one_class.csv", "sample_id,label\n" + "".join(f"{i},A\n" for i in range(8)))
    files["all_nan_labels"] = w("all_nan.csv", "sample_id,label\n" + "".join(f"{i},\n" for i in range(8)))
    some_nan = "".join(f"{i},{'' if i in (3, 77) else ('case' if i % 2 else 'control')}\n" for i in range(110))
    files["some_nan_labels"] = w("some_nan.csv", "sample_id,label\n" + some_nan)
    nan_feat = "".join(f"{i},{'' if i == 2 else 1.0 + i * 0.2},{2.0 - i * 0.1}\n" for i in range(30))
    files["nan_features"] = w("nan_features.csv", "sample_id,f1,f2\n" + nan_feat)
    inf_feat = "".join(f"{i},{'inf' if i == 2 else 1.0 + i * 0.2},{'-inf' if i == 5 else 2.0 - i * 0.1}\n" for i in range(30))
    files["inf_features"] = w("inf_features.csv", "sample_id,f1,f2\n" + inf_feat)
    files["non_numeric"] = w("non_numeric.csv", "sample_id,f1,f2\n0,1.0,hello\n1,2.0,2.0\n2,1.5,x\n3,2.5,1.0\n")
    files["no_overlap_labels"] = w("no_overlap.csv", "sample_id,label\n" + "".join(f"X{i},{'A' if i % 2 else 'B'}\n" for i in range(8)))
    dup = "sample_id,label\n" + "".join(f"{i},case\n{i},control\n" for i in range(0, 30))
    files["dup_id_labels"] = w("dup_ids.csv", dup)
    files["high_card_labels"] = w(
        "high_card.csv", "sample_id,label\n" + "".join(f"{i},c{i % 45}\n" for i in range(110))
    )
    files["continuous_labels"] = w(
        "continuous.csv", "sample_id,label\n" + "".join(f"{i},{0.37 * i + 0.11}\n" for i in range(110))
    )
    # Headerless: first row is all-numeric → parsed as DATA with synthesised
    # col_i names (the "header row parsed as data" case, inverted: a numeric
    # first row must NOT be swallowed as a header).
    rng = np.random.default_rng(3)
    headerless = "\n".join(",".join(f"{v:.6f}" for v in row) for row in rng.normal(size=(40, 5))) + "\n"
    files["headerless"] = w("headerless.csv", headerless)
    semi = "sample_id;f1;f2\n" + "".join(f"{i};{1.0 + i * 0.2};{2.0 - i * 0.1}\n" for i in range(30))
    files["semicolon"] = w("semicolon.csv", semi)
    files["garbage_csv"] = str(d / "garbage.csv")
    (d / "garbage.csv").write_bytes(bytes(range(200, 256)) * 4)
    files["garbage_npy"] = str(d / "garbage.npy")
    (d / "garbage.npy").write_bytes(b"not a numpy file at all")

    np.save(d / "rep3d.npy", np.zeros((4, 3, 2)))
    files["npy_3d"] = str(d / "rep3d.npy")
    nan_arr = np.arange(60, dtype=np.float64).reshape(20, 3)
    nan_arr[2, 1] = np.nan
    np.save(d / "nan.npy", nan_arr)
    files["npy_nan"] = str(d / "nan.npy")
    files["disjoint_id_rep2"] = w(
        "rep2_disjoint.csv", "sid,f1,f2\n" + "".join(f"X{i},{1.0 + i * 0.2},{2.0 - i * 0.1}\n" for i in range(6))
    )
    files["tiny_rep"] = w("tiny_rep.csv", "f1,f2\n" + "".join(f"{1.0 + i * 0.2},{2.0 - i * 0.1}\n" for i in range(8)))
    np.save(d / "vec.npy", np.linspace(0.0, 5.0, 24))
    files["npy_1d"] = str(d / "vec.npy")
    import pandas as pd

    rep_df = pd.read_csv(REP)
    np.save(d / "rep.npy", rep_df.drop(columns=["sample_id"]).to_numpy(dtype=np.float64))
    files["rep_npy"] = str(d / "rep.npy")
    rep_df.to_csv(d / "rep.tsv", sep="\t", index=False)
    files["rep_tsv"] = str(d / "rep.tsv")

    files["dir"] = str(d)
    return files


# ---------------------------------------------------------------------------
# Happy path — every subcommand, strict-JSON stdout, in-range metrics, exit 0
# ---------------------------------------------------------------------------
class TestHappyPath:
    @pytest.mark.parametrize("test", CORE_TESTS)
    def test_core_labeled_json(self, test):
        proc = run_cli(*labeled_args(test))
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = parse_stdout_json(proc)
        assert payload["test"] == test
        assert "group" in payload["results"]
        assert_finite_or_null(payload)

    def test_clusterability_metrics_in_range(self):
        payload = parse_stdout_json(run_cli(*labeled_args("clusterability")))
        m = payload["results"]["group"]
        assert -1.0 <= m["Silhouette Score"] <= 1.0
        assert m["Davies-Bouldin Index"] >= 0.0
        assert 0.0 <= m["Normalized Mutual Information"] <= 1.0
        assert 0.0 <= m["Cluster Learnability"] <= 1.0
        assert m["Clusters (k)"] == 2  # binary label → heuristic picks 2

    def test_expressiveness_metrics_in_range(self):
        payload = parse_stdout_json(run_cli(*labeled_args("expressiveness")))
        # run_expressiveness nests one internal "Label N" layer per label
        # column passed to it; the pipeline passes exactly one, and the web
        # app returns the same nesting (asserted by the parity suite).
        m = payload["results"]["group"]["Label 1"]
        assert m["Metric Type"] == "AUROC"
        assert 0.0 <= m["Baseline (0% removed)"] <= 1.0
        assert 0.0 <= m["Compactness"] <= 1.0
        assert m["Random Baseline"] == 0.5

    def test_probing_metrics_shape(self):
        payload = parse_stdout_json(run_cli(*labeled_args("probing")))
        m = payload["results"]["group"]
        assert isinstance(m["Model Complexity"], list) and len(m["Model Complexity"]) >= 2
        assert all(0.0 <= v <= 1.0 for v in m["AUROC"])

    def test_robustness_clustering_unlabeled(self):
        proc = run_cli("robustness", "--representations", REP, "--id-col", "sample_id",
                       "--robustness-metric", "clustering")
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = parse_stdout_json(proc)
        assert "Intrinsic (No Labels)" in payload["results"]

    def test_clusterability_unlabeled(self):
        proc = run_cli("clusterability", "--representations", REP, "--id-col", "sample_id")
        assert proc.returncode == 0, proc.stderr[-2000:]
        m = parse_stdout_json(proc)["results"]["Intrinsic (No Labels)"]
        assert set(m) == {"Silhouette Score", "Davies-Bouldin Index", "Clusters (k)"}

    def test_multimodal_happy(self):
        proc = run_cli(
            "multimodal", "--representations", REP, "--representations2", REP2,
            "--id-col", "sample_id", "--rep2-id-col", "sample_id",
            "--labels", LABELS, "--label-cols", "group", "--test-type", "clusterability",
            env={"MULTILOREFT_MAX_EPOCHS": "2"},
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = parse_stdout_json(proc)
        assert payload["multimodal"] is True
        assert set(payload["subspaceResults"]) >= {"shared"}
        assert_finite_or_null(payload)

    def test_version(self):
        proc = run_cli("--version")
        assert proc.returncode == 0
        import latentverse

        assert proc.stdout.strip() == latentverse.__version__

    def test_top_level_help_lists_every_subcommand(self):
        proc = run_cli("--help")
        assert proc.returncode == 0
        for sub in [*CORE_TESTS, "multimodal"]:
            assert sub in proc.stdout, f"--help does not mention subcommand {sub!r}"
        assert "--version" in proc.stdout

    @pytest.mark.parametrize("sub", [*CORE_TESTS, "multimodal"])
    def test_subcommand_help_documents_every_flag(self, sub):
        common = [
            "--representations", "--labels", "--id-col", "--label-cols",
            "--labels-id-col", "--seed", "--standardize", "--subsample",
            "--n-clusters", "--noise-levels", "--percent-removed",
            "--out", "--format",
        ]
        extra = {
            "robustness": ["--robustness-metric"],
            "multimodal": ["--representations2", "--rep2-id-col", "--test-type", "--robustness-metric"],
        }
        proc = run_cli(sub, "--help")
        assert proc.returncode == 0
        for flag in common + extra.get(sub, []):
            assert flag in proc.stdout, f"{sub} --help does not document {flag}"

    def test_no_subcommand_is_usage_error(self):
        proc = run_cli()
        assert proc.returncode == 2
        assert "usage" in proc.stderr.lower() or "usage" in proc.stdout.lower()

    def test_unknown_flag_is_usage_error(self):
        proc = run_cli("clusterability", "--representations", REP, "--frobnicate")
        assert proc.returncode == 2
        assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# Both entry points — byte-identical output
# ---------------------------------------------------------------------------
class TestEntryPoints:
    @pytest.mark.parametrize("test", CORE_TESTS)
    def test_console_script_matches_module(self, test):
        a = run_cli(*labeled_args(test), entry="module")
        b = run_cli(*labeled_args(test), entry="script")
        assert a.returncode == b.returncode == 0, (a.stderr[-800:], b.stderr[-800:])
        assert a.stdout == b.stdout

    def test_console_script_multimodal_dispatch(self):
        # Full multimodal training is covered once above; here prove the
        # console script wires the multimodal parser identically (usage error).
        proc = run_cli("multimodal", "--representations", REP, entry="script")
        assert proc.returncode == 2
        assert "representations2" in proc.stderr


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
class TestFlags:
    def test_out_json_and_csv(self, tmp_path):
        for fmt, parser in (("json", json.loads), ("csv", None)):
            out = tmp_path / f"r.{fmt}"
            proc = run_cli(*labeled_args("clusterability"), "--out", str(out), "--format", fmt)
            assert proc.returncode == 0, proc.stderr[-800:]
            assert proc.stdout.strip() == "", "with --out, stdout must stay empty"
            text = out.read_text()
            if fmt == "json":
                assert_finite_or_null(json.loads(text, parse_constant=_reject_constant))
            else:
                rows = list(csv.DictReader(io.StringIO(text)))
                assert rows and set(rows[0]) == {"metric", "value"}
                assert any(r["metric"].endswith("Silhouette Score") for r in rows)

    def test_format_csv_stdout_parses(self):
        proc = run_cli(*labeled_args("probing"), "--format", "csv")
        assert proc.returncode == 0
        rows = list(csv.DictReader(io.StringIO(proc.stdout)))
        assert rows and set(rows[0]) == {"metric", "value"}

    def test_tsv_representations(self, edge):
        proc = run_cli(*labeled_args("clusterability", rep=edge["rep_tsv"]))
        assert proc.returncode == 0, proc.stderr[-800:]
        # identical numbers to the CSV twin — the delimiter must not matter
        assert parse_stdout_json(proc) == parse_stdout_json(run_cli(*labeled_args("clusterability")))

    def test_npy_representations_row_id(self, edge):
        proc = run_cli(
            "probing", "--representations", edge["rep_npy"], "--labels", LABELS,
            "--id-col", "__row_id__", "--labels-id-col", "sample_id", "--label-cols", "group",
        )
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_multi_label_cols_run_per_label(self):
        payload = parse_stdout_json(run_cli(*labeled_args("probing", label="group,cohort")))
        assert set(payload["results"]) == {"group", "cohort"}

    def test_multi_label_disentanglement_combined(self):
        payload = parse_stdout_json(run_cli(*labeled_args("disentanglement", label="group,cohort")))
        assert list(payload["results"]) == ["group + cohort"]

    def test_seed_changes_are_respected(self):
        a = parse_stdout_json(run_cli(*labeled_args("clusterability"), "--seed", "42", "--subsample", "60"))
        b = parse_stdout_json(run_cli(*labeled_args("clusterability"), "--seed", "43", "--subsample", "60"))
        assert a != b, "different seeds must select a different subsample"

    def test_standardize_changes_numbers(self):
        a = parse_stdout_json(run_cli(*labeled_args("clusterability")))
        b = parse_stdout_json(run_cli(*labeled_args("clusterability"), "--standardize"))
        assert a != b

    def test_n_clusters_override(self):
        payload = parse_stdout_json(run_cli(*labeled_args("clusterability"), "--n-clusters", "7"))
        assert payload["results"]["group"]["Clusters (k)"] == 7

    def test_n_clusters_below_two_is_dropped(self):
        a = parse_stdout_json(run_cli(*labeled_args("clusterability"), "--n-clusters", "1"))
        b = parse_stdout_json(run_cli(*labeled_args("clusterability")))
        assert a == b
        assert a["results"]["group"]["Clusters (k)"] == 2

    def test_subsample_larger_than_dataset_is_noop(self):
        a = run_cli(*labeled_args("probing"), "--subsample", "100000")
        b = run_cli(*labeled_args("probing"))
        assert a.returncode == b.returncode == 0
        assert a.stdout == b.stdout

    def test_subsample_below_two_is_dropped(self):
        a = run_cli(*labeled_args("probing"), "--subsample", "1")
        b = run_cli(*labeled_args("probing"))
        assert a.stdout == b.stdout

    def test_noise_levels_flag(self):
        payload = parse_stdout_json(
            run_cli(*labeled_args("robustness"), "--robustness-metric", "probing",
                    "--noise-levels", "0.1,0.3")
        )
        m = payload["results"]["group"]
        lists = [v for v in m.values() if isinstance(v, list)]
        assert lists and all(len(v) in (2, 3) for v in lists), m  # noise sweep (± a baseline entry)

    def test_percent_removed_flag(self):
        payload = parse_stdout_json(run_cli(*labeled_args("expressiveness"), "--percent-removed", "0,25"))
        m = payload["results"]["group"]["Label 1"]
        assert "Baseline (0% removed)" in m and "25% Removed" in m
        assert "10% Removed" not in m

    def test_robustness_metric_defaults(self):
        # labels given → probing; no labels → clustering.
        with_labels = parse_stdout_json(run_cli(*labeled_args("robustness")))
        assert "group" in with_labels["results"]
        unlabeled = parse_stdout_json(
            run_cli("robustness", "--representations", REP, "--id-col", "sample_id",
                    "--robustness-metric", "clustering")
        )
        assert "Intrinsic (No Labels)" in unlabeled["results"]


# ---------------------------------------------------------------------------
# Determinism — the Issue-1 guarantee, at the CLI surface
# ---------------------------------------------------------------------------
class TestDeterminism:
    @pytest.mark.parametrize("test", CORE_TESTS)
    def test_same_seed_byte_identical(self, test):
        a = run_cli(*labeled_args(test), "--seed", "42")
        b = run_cli(*labeled_args(test), "--seed", "42")
        assert a.returncode == b.returncode == 0
        assert a.stdout == b.stdout

    @pytest.mark.parametrize("n_jobs", ["1", "2", "4"])
    def test_expressiveness_invariant_to_n_jobs(self, n_jobs):
        base = run_cli(*labeled_args("expressiveness"), env={"LATENTVERSE_N_JOBS": "1"})
        got = run_cli(*labeled_args("expressiveness"), env={"LATENTVERSE_N_JOBS": n_jobs})
        assert base.returncode == got.returncode == 0
        assert got.stdout == base.stdout, f"expressiveness diverged at LATENTVERSE_N_JOBS={n_jobs}"


# ---------------------------------------------------------------------------
# Row caps — env-tunable thresholds actually fire and are deterministic
# ---------------------------------------------------------------------------
class TestRowCaps:
    @pytest.mark.parametrize(
        "test,env_name",
        [
            ("expressiveness", "SUPERVISED_SAMPLE_THRESHOLD"),
            ("disentanglement", "SUPERVISED_SAMPLE_THRESHOLD"),
            ("probing", "PROBING_FAST_SAMPLE_THRESHOLD"),
        ],
    )
    def test_cap_fires_and_is_deterministic(self, test, env_name):
        # 107 valid labelled rows > 50 → the cap fires and must change the
        # numbers; two capped runs must agree byte-for-byte.
        capped1 = run_cli(*labeled_args(test), env={env_name: "50"})
        capped2 = run_cli(*labeled_args(test), env={env_name: "50"})
        uncapped = run_cli(*labeled_args(test))
        assert capped1.returncode == uncapped.returncode == 0
        assert capped1.stdout == capped2.stdout
        assert capped1.stdout != uncapped.stdout, f"{env_name}=50 did not change {test} output"
        # A fired cap is disclosed with the web app's exact "Rows evaluated"
        # key/format, so a capped CLI run does not report a downsampled metric
        # as if it used the whole file. Absent (not empty/false) when uncapped.
        capped_metrics = parse_stdout_json(capped1)
        assert _find_rows_evaluated(capped_metrics) == "50 of 107 (subsampled for speed)"
        assert _find_rows_evaluated(parse_stdout_json(uncapped)) is None

    def test_rows_evaluated_absent_when_no_cap_fires(self):
        # A run under the cap must NOT carry the annotation (it would falsely
        # imply a downsample), matching the web app's conditional disclosure.
        proc = run_cli(*labeled_args("probing"), env={"PROBING_FAST_SAMPLE_THRESHOLD": "5000"})
        assert proc.returncode == 0
        assert _find_rows_evaluated(parse_stdout_json(proc)) is None

    def test_cap_above_n_is_noop(self):
        capped = run_cli(*labeled_args("expressiveness"), env={"SUPERVISED_SAMPLE_THRESHOLD": "5000"})
        default = run_cli(*labeled_args("expressiveness"))
        assert capped.stdout == default.stdout


# ---------------------------------------------------------------------------
# Edge cases — clean error or correct result; never a traceback
# ---------------------------------------------------------------------------
class TestEdgeErrors:
    def test_empty_file(self, edge):
        assert_clean_error(run_cli("clusterability", "--representations", edge["empty"]))

    def test_single_row(self, edge):
        assert_clean_error(
            run_cli("clusterability", "--representations", edge["one_row"], "--id-col", "sample_id"),
            "at least 2",
        )

    def test_missing_file(self, edge):
        assert_clean_error(
            run_cli("clusterability", "--representations", os.path.join(edge["dir"], "nope.csv")),
            "file not found",
        )

    def test_unreadable_file(self, edge, tmp_path):
        p = tmp_path / "noperm.csv"
        p.write_text("sample_id,f1\n0,1.0\n")
        p.chmod(0o000)
        try:
            assert_clean_error(run_cli("clusterability", "--representations", str(p)))
        finally:
            p.chmod(0o644)

    def test_garbage_csv(self, edge):
        assert_clean_error(run_cli("clusterability", "--representations", edge["garbage_csv"]))

    def test_garbage_npy(self, edge):
        assert_clean_error(run_cli("clusterability", "--representations", edge["garbage_npy"]))

    def test_npy_3d(self, edge):
        assert_clean_error(run_cli("clusterability", "--representations", edge["npy_3d"]), "1D or 2D")

    def test_non_numeric_cells(self, edge):
        assert_clean_error(
            run_cli("clusterability", "--representations", edge["non_numeric"], "--id-col", "sample_id"),
            "non-numeric",
        )

    def test_multimodal_disjoint_ids(self, edge):
        assert_clean_error(
            run_cli("multimodal", "--representations", REP, "--representations2",
                    edge["disjoint_id_rep2"], "--id-col", "sample_id", "--rep2-id-col", "sid",
                    env={"MULTILOREFT_MAX_EPOCHS": "2"}),
            "No rows remain",
        )

    def test_single_class_labels(self, edge):
        assert_clean_error(
            run_cli("probing", "--representations", edge["one_feature"], "--labels",
                    edge["one_class_labels"], "--id-col", "sample_id", "--label-cols", "label"),
            "distinct classes",
        )

    def test_all_nan_labels(self, edge):
        assert_clean_error(
            run_cli("probing", "--representations", edge["one_feature"], "--labels",
                    edge["all_nan_labels"], "--id-col", "sample_id", "--label-cols", "label"),
        )

    def test_zero_id_overlap(self, edge):
        assert_clean_error(
            run_cli("probing", "--representations", REP, "--labels", edge["no_overlap_labels"],
                    "--id-col", "sample_id", "--label-cols", "label"),
            "No ids matched",
        )

    def test_unknown_label_column(self):
        assert_clean_error(run_cli(*labeled_args("probing", label="not_a_col")), "not_a_col", "Available columns")

    def test_unknown_id_column(self):
        assert_clean_error(
            run_cli("probing", "--representations", REP, "--labels", LABELS,
                    "--id-col", "not_a_col", "--label-cols", "group"),
            "not_a_col",
        )

    def test_unknown_labels_id_column(self):
        assert_clean_error(
            run_cli("probing", "--representations", REP, "--labels", LABELS, "--id-col", "sample_id",
                    "--labels-id-col", "nope", "--label-cols", "group"),
            "nope",
        )

    def test_labels_required_but_missing(self):
        assert_clean_error(run_cli("probing", "--representations", REP), "requires labels")

    def test_n_clusters_exceeding_samples(self):
        # 5000 is clamped to MAX_NUM_CLUSTERS=1000, still > n=120 → KMeans's
        # own actionable message, cleanly wrapped.
        assert_clean_error(
            run_cli("clusterability", "--representations", REP, "--id-col", "sample_id",
                    "--n-clusters", "5000"),
            "n_samples",
        )

    def test_n_lt_k(self, edge):
        assert_clean_error(
            run_cli("clusterability", "--representations", edge["one_feature"], "--id-col",
                    "sample_id", "--n-clusters", "10"),
        )

    def test_out_to_missing_directory(self, edge):
        assert_clean_error(
            run_cli(*labeled_args("clusterability"), "--out", os.path.join(edge["dir"], "no", "dir", "r.json")),
        )


class TestEdgeCorrectResults:
    def test_nan_feature_cells_in_csv_imputed_like_webapp(self, edge):
        # The pyarrow-backed CSV reader yields pd.NA for empty cells;
        # _sanitize_features maps them to float NaN so the documented
        # mean-imputation runs instead of a misleading non-numeric rejection.
        # The web app applies the same fix (app/test_runner.py) — the two
        # sides were flipped together.
        proc = run_cli("clusterability", "--representations", edge["nan_features"], "--id-col", "sample_id")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_single_feature_column(self, edge):
        proc = run_cli("clusterability", "--representations", edge["one_feature"], "--id-col", "sample_id")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_npy_1d_vector(self, edge):
        proc = run_cli("clusterability", "--representations", edge["npy_1d"])
        assert proc.returncode == 0, proc.stderr[-800:]

    def test_some_nan_labels_dropped(self, edge):
        proc = run_cli("probing", "--representations", REP, "--labels", edge["some_nan_labels"],
                       "--id-col", "sample_id", "--label-cols", "label")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_nan_features_imputed_via_npy(self, edge):
        # .npy carries real float NaN (no pyarrow NA wrapper), so the
        # documented mean-imputation path actually runs.
        proc = run_cli("clusterability", "--representations", edge["npy_nan"])
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_inf_feature_strings_imputed(self, edge):
        # pyarrow parses 'inf'/'-inf' CSV cells as float ±inf → treated as
        # missing and mean-imputed (correct result, not an error).
        proc = run_cli("clusterability", "--representations", edge["inf_features"], "--id-col", "sample_id")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_multimodal_row_count_mismatch_aligned_by_row_id(self, edge):
        # 8-row rep1 vs 120-row rep2 joined on __row_id__: the id merge takes
        # the intersection (web-app semantics) instead of erroring.
        proc = run_cli("multimodal", "--representations", edge["tiny_rep"], "--representations2", REP2,
                       "--id-col", "__row_id__", "--rep2-id-col", "__row_id__",
                       env={"MULTILOREFT_MAX_EPOCHS": "2"})
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_headerless_numeric_first_row_is_data(self, edge):
        proc = run_cli("clusterability", "--representations", edge["headerless"])
        assert proc.returncode == 0, proc.stderr[-800:]

    def test_semicolon_delimiter(self, edge):
        proc = run_cli("clusterability", "--representations", edge["semicolon"], "--id-col", "sample_id")
        assert proc.returncode == 0, proc.stderr[-800:]

    def test_duplicate_ids_follow_webapp_merge_semantics(self, edge):
        # Duplicated label ids fan out in the inner merge (web-app semantics);
        # must run to completion, not crash.
        proc = run_cli("probing", "--representations", REP, "--labels", edge["dup_id_labels"],
                       "--id-col", "sample_id", "--label-cols", "label")
        assert proc.returncode == 0, proc.stderr[-800:]

    def test_high_cardinality_label(self, edge):
        proc = run_cli("probing", "--representations", REP, "--labels", edge["high_card_labels"],
                       "--id-col", "sample_id", "--label-cols", "label")
        assert proc.returncode == 0, proc.stderr[-800:]
        assert_finite_or_null(parse_stdout_json(proc))

    def test_continuous_label_in_clusterability(self, edge):
        # Continuous where categorical is expected: factorised to n classes,
        # heuristic k=4 — the web app's behaviour; must not crash.
        proc = run_cli("clusterability", "--representations", REP, "--labels", edge["continuous_labels"],
                       "--id-col", "sample_id", "--label-cols", "label")
        assert proc.returncode == 0, proc.stderr[-800:]

    def test_categorical_label_in_regression_slot(self):
        # Categorical where continuous might be expected: coerced to codes →
        # binary AUROC path; in-range result.
        payload = parse_stdout_json(run_cli(*labeled_args("expressiveness", label="group")))
        assert payload["results"]["group"]["Label 1"]["Metric Type"] == "AUROC"

    def test_multimodal_identical_modalities(self, edge):
        # Fully-shared signal: the decoupler may find no modality-specific
        # subspace. Either a clean per-subspace result or a clean error.
        proc = run_cli("multimodal", "--representations", REP, "--representations2", REP,
                       "--id-col", "sample_id", "--rep2-id-col", "sample_id",
                       "--labels", LABELS, "--label-cols", "group",
                       env={"MULTILOREFT_MAX_EPOCHS": "2"})
        assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
        if proc.returncode == 0:
            assert_finite_or_null(parse_stdout_json(proc))
        else:
            assert "error:" in proc.stderr


# ---------------------------------------------------------------------------
# Output hygiene — stdout is ONLY the payload
# ---------------------------------------------------------------------------
class TestOutputHygiene:
    @pytest.mark.parametrize("test", CORE_TESTS)
    def test_stdout_is_pure_json(self, test):
        proc = run_cli(*labeled_args(test))
        assert proc.returncode == 0
        parse_stdout_json(proc)  # would raise on any stray warning line

    def test_progress_chatter_goes_to_stderr(self):
        # The library prints progress lines; they must land on stderr.
        proc = run_cli(*labeled_args("robustness"))
        assert proc.returncode == 0
        parse_stdout_json(proc)
