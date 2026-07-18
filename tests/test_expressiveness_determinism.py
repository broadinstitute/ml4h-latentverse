"""Regression test for the 0.3.4 expressiveness RNG fix.

Through 0.3.3, ``run_expressiveness``'s ``process_fold`` seeded and consumed
the GLOBAL numpy RNG inside ``Parallel(backend="threading")`` folds, so at
``LATENTVERSE_N_JOBS > 1`` the fold train/test split (and the stochastic SAGA
solver on the binary path) raced across threads: the reported numbers were a
single non-reproducible draw. The production web app runs at N_JOBS=2, so this
was live.

The fix gives each fold a private ``np.random.RandomState(random_state +
fold_idx)`` that drives both the shuffle and the solver. These tests assert
the two properties the fix guarantees:

  1. Determinism: identical results across repeated runs at every n_jobs.
  2. n_jobs-invariance: n_jobs ∈ {1, 2, 4} all yield the *same* results —
     which are the legacy sequential (n_jobs=1) values, i.e. the fix changes
     no historically-correct number.

Runs the metric in-process (LATENTVERSE_N_JOBS is read per call via
``latentverse.utils.get_n_jobs``), on the binary path specifically because
that is the one with a stochastic solver.
"""

import numpy as np
import pytest

from latentverse import run_expressiveness


def _make_binary(n=160, dim=10, seed=7):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    reps = rng.normal(size=(n, dim)) + labels[:, None] * 1.5
    return reps, labels.reshape(-1, 1)


def _make_regression(n=160, dim=10, seed=11):
    rng = np.random.default_rng(seed)
    reps = rng.normal(size=(n, dim))
    labels = reps[:, :3].sum(axis=1) + 0.3 * rng.normal(size=n)
    return reps, labels.reshape(-1, 1)


def _run(reps, labels, n_jobs, monkeypatch):
    monkeypatch.setenv("LATENTVERSE_N_JOBS", str(n_jobs))
    out = run_expressiveness(
        representations=reps,
        labels=labels,
        percent_to_remove_list=[0, 10, 50],
        plots=False,
        random_state=42,
    )
    # plot_data duplicates the metric values; comparing metrics covers both.
    return out["metrics"]


@pytest.mark.parametrize("maker", [_make_binary, _make_regression], ids=["binary", "regression"])
def test_expressiveness_identical_across_n_jobs_and_repeats(maker, monkeypatch):
    reps, labels = maker()
    baseline = _run(reps, labels, 1, monkeypatch)
    for n_jobs in (1, 2, 4):
        for repeat in range(2):
            got = _run(reps, labels, n_jobs, monkeypatch)
            assert got == baseline, (
                f"expressiveness diverged at n_jobs={n_jobs} repeat={repeat}: "
                f"{got} != {baseline}"
            )


def test_expressiveness_does_not_touch_global_rng(monkeypatch):
    """The metric must neither read nor reseed the process-global numpy RNG:
    a caller's unrelated np.random sequence has to be unaffected by a run."""
    reps, labels = _make_binary()
    np.random.seed(123)
    expected_next = np.random.RandomState(123).random_sample(4)
    monkeypatch.setenv("LATENTVERSE_N_JOBS", "2")
    run_expressiveness(
        representations=reps,
        labels=labels,
        percent_to_remove_list=[0, 10],
        plots=False,
        random_state=42,
    )
    np.testing.assert_array_equal(np.random.random_sample(4), expected_next)
