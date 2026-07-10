"""Latentverse: evaluation suite for latent representations."""

from latentverse.evaluations import (
    run_clustering,
    run_disentanglement,
    run_expressiveness,
    run_probing,
    run_robustness,
)

__version__ = "0.3.1"

__all__ = [
    "run_clustering",
    "run_disentanglement",
    "run_expressiveness",
    "run_probing",
    "run_robustness",
]
