"""Deterministically generate the cross-parity fixture files.

The fixtures are intentionally small (under every row cap) yet exercise the
number-affecting branches the parity test must guard:

  * an ID-column inner/left merge with a few UNMATCHED representation rows
    (representation ids 110..119 have no label row),
  * a couple of NaN labels (dropped by the supervised path, kept-as-NaN by the
    clustering path),
  * a STRING/categorical label column ("group": case/control) so the label
    coercion / factorisation path actually moves the numbers,
  * an extra multiclass numeric label ("cohort") for multi-factor coverage,
  * a second representation matrix (rep2) for the multimodal path.

Run:  python tests/fixtures/generate_fixtures.py
Committed outputs: rep.csv, rep2.csv, labels.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

N = 120  # representation rows
D1 = 16  # rep1 dims
D2 = 12  # rep2 dims (multimodal)
L = 110  # labelled rows (ids 0..109); ids 110..119 are unmatched
NAN_LABEL_IDS = [5, 33, 77]  # explicit NaN labels among the matched rows


def main() -> None:
    rng = np.random.default_rng(20240718)

    # A latent factor drives both the label and part of the embedding, so the
    # supervised metrics have real (non-degenerate) signal to recover.
    z = rng.normal(size=N)

    X1 = rng.normal(size=(N, D1)).astype(np.float64)
    X1[:, 0] += 2.0 * z
    X1[:, 1] += 1.5 * z
    X1[:, 2] += 1.0 * z

    # rep2 shares the same latent factor (so "shared" subspace is meaningful)
    # plus its own modality-specific factor.
    m2 = rng.normal(size=N)
    X2 = rng.normal(size=(N, D2)).astype(np.float64)
    X2[:, 0] += 1.8 * z
    X2[:, 1] += 1.2 * m2

    group_bin = ((z + rng.normal(scale=0.5, size=N)) > 0).astype(int)
    group = np.where(group_bin == 1, "case", "control").astype(object)
    cohort = rng.integers(0, 3, size=N)
    # Continuous label (regression parity path: probing → R², expressiveness →
    # fit_linear). Drawn AFTER every pre-existing draw so adding it left the
    # previously-committed columns byte-identical.
    score = 1.7 * z + rng.normal(scale=0.6, size=N)

    rep_df = pd.DataFrame(X1, columns=[f"dim_{i}" for i in range(D1)])
    rep_df.insert(0, "sample_id", np.arange(N))
    rep_df.to_csv(os.path.join(HERE, "rep.csv"), index=False)

    rep2_df = pd.DataFrame(X2, columns=[f"dim_{i}" for i in range(D2)])
    rep2_df.insert(0, "sample_id", np.arange(N))
    rep2_df.to_csv(os.path.join(HERE, "rep2.csv"), index=False)

    lab = pd.DataFrame(
        {
            "sample_id": np.arange(L),
            "group": group[:L],
            "cohort": cohort[:L].astype(float),
            "score": score[:L],
        }
    )
    lab.loc[NAN_LABEL_IDS, "group"] = np.nan
    lab.to_csv(os.path.join(HERE, "labels.csv"), index=False)

    print(f"Wrote rep.csv ({rep_df.shape}), rep2.csv ({rep2_df.shape}), labels.csv ({lab.shape}) to {HERE}")


if __name__ == "__main__":
    main()
