"""``latentverse`` command-line interface.

A standalone CLI for researchers: point it at a representations file (and,
optionally, a labels file) and run any of the five core evaluations or the
multimodal decoupling — reproducing the numbers the LatentVerse web app
produces, because it drives the same duplicated pipeline
(``latentverse.pipeline``) and calls the same ``latentverse.evaluations.*``
metric functions.

Examples
--------
    latentverse probing --representations reps.csv --labels labels.csv \\
        --id-col sample_id --label-cols disease

    latentverse clusterability --representations reps.npy --n-clusters 5 \\
        --out result.json

    latentverse multimodal --representations img.csv --representations2 txt.csv \\
        --id-col id --rep2-id-col id --labels labels.csv --label-cols disease \\
        --test-type clusterability

Structure mirrors the companion web application's command-line entrypoint:
subparsers, ``main(argv)``, integer exit codes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from latentverse.pipeline import (
    CORE_TESTS,
    DEFAULT_NOISE_LEVELS,
    DEFAULT_PERCENT_REMOVED,
    PipelineConfig,
    run_pipeline,
)

logger = logging.getLogger("latentverse.cli")


class _NumpyJSONEncoder(json.JSONEncoder):
    """Serialise numpy scalars / arrays that the metric dicts carry."""

    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return None if np.isnan(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and np.isnan(o):
            return None
        return super().default(o)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _csv_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def _float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip() != ""]


def _add_common_args(p: argparse.ArgumentParser, *, multimodal: bool = False) -> None:
    p.add_argument(
        "--representations",
        required=True,
        metavar="FILE",
        help="Representation/embedding file (.csv, .tsv, or .npy). Required.",
    )
    p.add_argument("--labels", metavar="FILE", help="Labels file (.csv/.tsv).")
    p.add_argument(
        "--id-col",
        metavar="NAME",
        help="Identifier column shared by the representation and labels files. "
        "Use '__row_id__' when the files are already in the same row order.",
    )
    p.add_argument(
        "--label-cols",
        type=_csv_list,
        default=[],
        metavar="A,B",
        help="Comma-separated label column name(s) to evaluate.",
    )
    p.add_argument(
        "--labels-id-col",
        metavar="NAME",
        help="Identifier column in the labels file (defaults to --id-col).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument(
        "--standardize",
        action="store_true",
        help="Zero-mean / unit-variance the features before evaluation.",
    )
    p.add_argument(
        "--subsample",
        type=int,
        metavar="N",
        help="Deterministically subsample to N rows before running (seeded).",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        metavar="K",
        help="Override the KMeans cluster count k (clusterability / robustness-clustering). Ignored when < 2.",
    )
    p.add_argument(
        "--noise-levels",
        type=_float_list,
        default=list(DEFAULT_NOISE_LEVELS),
        metavar="L1,L2,...",
        help=f"Robustness noise levels (default {DEFAULT_NOISE_LEVELS}).",
    )
    p.add_argument(
        "--percent-removed",
        type=_int_list,
        default=list(DEFAULT_PERCENT_REMOVED),
        metavar="P1,P2,...",
        help=f"Expressiveness dimension-removal sweep (default {DEFAULT_PERCENT_REMOVED}).",
    )
    p.add_argument("--out", metavar="FILE", help="Write results here (default: stdout).")
    p.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format (default json).",
    )
    if multimodal:
        p.add_argument(
            "--representations2",
            required=True,
            metavar="FILE",
            help="Second representation file for multimodal decoupling. Required.",
        )
        p.add_argument(
            "--rep2-id-col",
            metavar="NAME",
            help="Identifier column in the second representation file (defaults to --id-col).",
        )
        p.add_argument(
            "--test-type",
            choices=CORE_TESTS,
            default="clusterability",
            help="Which core test to run on each decoupled subspace (default clusterability).",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentverse",
        description="Evaluate latent representations (LatentVerse). Reproduces "
        "the web app's numbers via the same duplicated pipeline.",
        epilog=(
            "Reproducibility: results are deterministic for a given --seed at any "
            "LATENTVERSE_N_JOBS (the expressiveness fold-RNG race was fixed in "
            "0.3.4 — each fold now owns its RNG). The CLI still defaults "
            "LATENTVERSE_N_JOBS=1 as a conservative baseline; raise it for speed. "
            "BLAS thread counts (OMP/OPENBLAS/MKL_NUM_THREADS) can still perturb "
            "the last floating-point bits — pin them to 1 for bit-exact runs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Print the library version and exit.")
    sub = parser.add_subparsers(dest="test", metavar="<test>")

    for test in CORE_TESTS:
        sp = sub.add_parser(test, help=f"Run the {test} evaluation.")
        _add_common_args(sp)
        if test == "robustness":
            sp.add_argument(
                "--robustness-metric",
                choices=("probing", "clustering"),
                default=None,
                help="Robustness downstream metric. Defaults to 'probing' when labels are given, else 'clustering'.",
            )

    mm = sub.add_parser("multimodal", help="Decouple two modalities and evaluate each subspace.")
    _add_common_args(mm, multimodal=True)
    mm.add_argument(
        "--robustness-metric",
        choices=("probing", "clustering"),
        default=None,
        help="Used only when --test-type robustness.",
    )
    return parser


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _flatten(prefix: str, value: Any, rows: List[Dict[str, str]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
    elif isinstance(value, (list, tuple, np.ndarray)):
        rows.append({"metric": prefix, "value": json.dumps(list(value), cls=_NumpyJSONEncoder)})
    else:
        if isinstance(value, np.generic):
            value = value.item()
        rows.append({"metric": prefix, "value": "" if value is None else str(value)})


def _to_csv(results: Dict[str, Any]) -> str:
    rows: List[Dict[str, str]] = []
    _flatten("", results, rows)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["metric", "value"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _null_non_finite(value: Any) -> Any:
    """Recursively map non-finite floats to None. json.dumps only consults the
    encoder's ``default`` for types it can't serialise, so a plain Python
    float('nan') would otherwise be emitted as the bare token ``NaN`` —
    invalid JSON for strict parsers. The output contract is: numeric fields
    are finite or explicitly null."""
    if isinstance(value, dict):
        return {k: _null_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_non_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _null_non_finite(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _emit(payload: Dict[str, Any], out: Optional[str], fmt: str) -> None:
    payload = _null_non_finite(payload)
    if fmt == "json":
        text = json.dumps(payload, indent=2, cls=_NumpyJSONEncoder, allow_nan=False)
    else:
        text = _to_csv(payload)
    if out:
        with open(out, "w") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    robustness_metric = getattr(args, "robustness_metric", None)
    if args.test == "robustness" or getattr(args, "test_type", None) == "robustness":
        if robustness_metric is None:
            robustness_metric = "probing" if args.label_cols else "clustering"
    return PipelineConfig(
        test_type=args.test if args.test != "multimodal" else args.test_type,
        representations_path=args.representations,
        labels_path=args.labels,
        rep_id_col=args.id_col,
        labels_id_col=args.labels_id_col,
        label_cols=list(args.label_cols),
        robustness_metric=robustness_metric,
        random_seed=args.seed,
        standardize=args.standardize,
        subsample_rows=args.subsample,
        num_clusters_override=args.n_clusters,
        percent_removed=list(args.percent_removed),
        noise_levels=list(args.noise_levels),
        representations2_path=getattr(args, "representations2", None),
        rep2_id_col=getattr(args, "rep2_id_col", None),
    )


def _run(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    # Some library metric functions print progress to stdout. Redirect that
    # chatter to stderr while computing so stdout carries ONLY the result
    # payload (keeping the CLI machine-parseable when piped).
    with contextlib.redirect_stdout(sys.stderr):
        if args.test == "multimodal":
            from latentverse.pipeline_multimodal import run_multimodal_pipeline

            results = run_multimodal_pipeline(cfg)
            payload = {
                "test": args.test_type,
                "multimodal": True,
                "subspaceResults": results,
            }
        else:
            results = run_pipeline(cfg)
            payload = {"test": args.test, "results": results}
    _emit(payload, args.out, args.format)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        import latentverse

        print(latentverse.__version__)
        return 0
    if not getattr(args, "test", None):
        parser.print_help(sys.stderr)
        return 2

    try:
        return _run(args)
    except ValueError as exc:
        # Project convention: ValueError = expected, user-actionable problem.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not read/write file: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        # Column lookups are pre-validated in the pipeline; this is a backstop
        # so no user input can surface a raw traceback.
        print(f"error: missing column or key: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI contract: never a raw traceback.
        if os.environ.get("LATENTVERSE_DEBUG") == "1":
            raise
        print(
            f"error: unexpected {type(exc).__name__}: {exc}\n"
            "(re-run with LATENTVERSE_DEBUG=1 for the full traceback)",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
