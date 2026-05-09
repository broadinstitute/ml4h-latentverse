"""
End-to-end CLI smoke for latentverse.

Demonstrates how a researcher would use the library from a plain
Python script (no FastAPI, no React, no GCP). Generates synthetic
data on the fly and exercises every public entrypoint.

Run:
    python examples/run_cli.py

Expected runtime on a laptop: under 30 seconds.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from latentverse import (
    run_clustering,
    run_disentanglement,
    run_expressiveness,
    run_probing,
    run_robustness,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_blobs(n: int = 600, dim: int = 16, k: int = 4, sep: float = 4.0, seed: int = 0):
    """Linearly-separable Gaussian blobs with k modes — the easy case."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(k, dim))
    labels = rng.integers(0, k, size=n)
    reps = centers[labels] + rng.normal(scale=1.0, size=(n, dim))
    return reps.astype(np.float32), labels.astype(np.float64)


def make_regression(n: int = 600, dim: int = 16, noise: float = 0.1, seed: int = 0):
    """Continuous labels, linear in a random subset of features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    w = rng.normal(size=dim)
    y = (X @ w + rng.normal(scale=noise, size=n)).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------
def fmt_metric(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={fmt_metric(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v).ravel()
        if arr.dtype.kind in "fiu":  # only summarise numeric arrays
            return f"[{arr.shape[0]}d, mean={float(arr.mean()):.4f}]"
        # non-numeric (e.g. model-complexity labels): show length + first 3 entries
        head = ", ".join(repr(x) for x in arr[:3].tolist())
        if arr.size > 3:
            head += ", ..."
        return f"[{arr.size}: {head}]"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return repr(v)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def demo_clustering() -> None:
    section("1. Clusterability")
    X, y = make_blobs(n=400, dim=12, k=4, sep=5.0, seed=1)
    out = run_clustering(X, y, random_state=42)
    print("  Metrics:")
    for name, value in out["results"].items():
        print(f"    {name:35s} {fmt_metric(value)}")
    print(f"  PCA available     : {out['pca_available']}")
    print(f"  KMeans assignments: shape={out['cluster_labels'].shape}")


def demo_disentanglement() -> None:
    section("2. Disentanglement")
    X, y = make_blobs(n=500, dim=10, k=5, seed=2)
    out = run_disentanglement(X, y, random_state=42)
    metrics = out["metrics"]
    print(f"    DCI            : {fmt_metric(metrics.get('DCI'))}")
    print(f"    MIG            : {fmt_metric(metrics.get('MIG'))}")
    print(f"    SAP            : {fmt_metric(metrics.get('SAP'))}")
    print(f"    TC             : {fmt_metric(metrics.get('TC'))}")


def demo_expressiveness() -> None:
    section("3. Expressiveness")
    X, y = make_regression(n=500, dim=12, noise=0.3, seed=3)
    out = run_expressiveness(
        X,
        y,
        percent_to_remove_list=[0, 10, 30, 60],
        plots=False,
        random_state=42,
    )
    print(f"    metrics keys: {list(out['metrics'].keys())[:6]}")


def demo_probing() -> None:
    section("4. Probing")
    X, y = make_blobs(n=500, dim=12, k=3, sep=3.5, seed=4)
    out = run_probing(X, y, random_state=42)
    if isinstance(out, tuple):
        metrics, _ = out
    else:
        metrics = out
    print(f"    metrics: {fmt_metric(metrics)}")


def demo_robustness() -> None:
    section("5. Robustness")
    X, y = make_blobs(n=400, dim=10, k=3, seed=5)
    out = run_robustness(
        X,
        y,
        noise_levels=[0.0, 0.25, 0.5, 1.0],
        metric="clustering",
        plots=False,
        random_state=42,
    )
    print(f"    metric keys: {list(out['metrics'].keys())[:6]}")


def demo_multimodal() -> None:
    section("6. Multimodal (MultiLoReFT) — instantiation + forward pass")
    try:
        import torch
    except ImportError:
        print("    Skipped (torch not installed).")
        return

    from latentverse.multiloreft import MultiLoReFT

    torch.manual_seed(0)
    input_dims = [16, 24]   # two modalities of different size
    n = 32

    model = MultiLoReFT(
        input_dims=input_dims,
        shared_rank=4,
        specific_rank=3,
        staging=False,
        encoders=None,
        device="cpu",
        pruning=False,
    )

    embeddings = [
        torch.randn(n, input_dims[0]),
        torch.randn(n, input_dims[1]),
    ]
    with torch.no_grad():
        out = model(embeddings)

    print(f"    model trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if isinstance(out, (list, tuple)):
        for i, t in enumerate(out):
            if hasattr(t, "shape"):
                print(f"    output[{i}] shape       : {tuple(t.shape)}")
    elif hasattr(out, "shape"):
        print(f"    output shape           : {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    print("latentverse CLI smoke")
    print("-" * 60)

    demos = [
        ("clusterability", demo_clustering),
        ("disentanglement", demo_disentanglement),
        ("expressiveness", demo_expressiveness),
        ("probing", demo_probing),
        ("robustness", demo_robustness),
        ("multimodal", demo_multimodal),
    ]

    failures: list[tuple[str, Exception]] = []
    for name, fn in demos:
        try:
            fn()
        except Exception as e:  # pragma: no cover — surfaces in CLI output
            failures.append((name, e))
            print(f"\n  FAILED {name!r}: {e!r}")

    print()
    print("=" * 60)
    print(f"Done in {time.time() - t0:.1f}s")
    if failures:
        print(f"{len(failures)} failure(s):")
        for name, e in failures:
            print(f"  - {name}: {e}")
        return 1
    print("All entrypoints executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
