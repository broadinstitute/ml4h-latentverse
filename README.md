# ml4h-latentverse

A Python library for evaluating the quality of latent representations. Five
core tests, plus a separate module for decoupling multimodal embeddings.

The five tests, in brief:

| Test | What it asks |
|---|---|
| **Clusterability** | Do these representations form meaningful clusters? Optional ground-truth labels enable NMI / Cluster Learnability; without labels you still get Silhouette + Davies–Bouldin. |
| **Disentanglement** | How independent are the latent dimensions? Reports DCI, MIG, Total Correlation, SAP. |
| **Expressiveness** | How much of the label signal is recoverable from the representations, and how does that signal degrade as you remove highly-correlated dimensions? |
| **Robustness** | What happens to performance under Gaussian noise of varying magnitude? Pick `metric="clustering"` or `metric="probing"`. |
| **Probing** | Train classifiers / regressors of varying complexity on the representations and see how well they recover the labels. Auto-detects task type. |

The `multiloreft` submodule trains a small projector to push two
correlated modalities apart in latent space; it's useful when you want
"image-only" and "text-only" embeddings out of a model that learned them
jointly.

There's also a companion webapp at https://latentverse-web-783282864357.us-central1.run.app/ that
puts a UI in front of all of this; the library itself has no webapp
dependencies.

## Install

```bash
pip install git+https://github.com/broadinstitute/ml4h-latentverse.git
```

For development:

```bash
git clone https://github.com/broadinstitute/ml4h-latentverse.git
cd ml4h-latentverse
pip install -e ".[dev]"
```

Requires Python ≥ 3.9.

## Quick example

```python
import numpy as np
from ml4h_latentverse import run_clustering

# 200 samples, 32 latent dims, 4 ground-truth classes
reps = np.random.randn(200, 32)
labels = np.random.randint(0, 4, 200)

result = run_clustering(reps, labels, random_state=42)
print(result["results"])
# -> {"Silhouette Score": ..., "Normalized Mutual Information": ...,
#     "Davies-Bouldin Index": ..., "Cluster Learnability": ...}
```

Every test entrypoint takes a `random_state` keyword for reproducibility.
Where it makes sense (clusterability, robustness-clustering) you can also
pass `num_clusters` to override the auto-derived value, and
`standardize=True` to z-score the inputs before evaluation.

For a complete end-to-end demo that runs all five tests + multimodal on
synthetic data in under 30 seconds, see `examples/run_cli.py`.

## API

```python
from ml4h_latentverse import (
    run_clustering,
    run_disentanglement,
    run_expressiveness,
    run_robustness,
    run_probing,
)
from ml4h_latentverse.multiloreft import MultiLoReFTProjector
```

Each test returns a `dict` containing a `"results"` block (numeric metrics)
and, when `plots=True`, a `"plot_url"` (image path) or `"plot_data"` (raw
arrays you can render however you want).

For the full per-function signature and what each metric means,
docstrings on the entrypoints are the source of truth — open
`ml4h_latentverse/tests/clustering.py` (or any other test file) and read
the function header.

## Tests

```bash
pytest tests/
```

`tests/test_smoke.py` is a small contract-style suite that verifies every
entrypoint runs end-to-end on synthetic data and returns the keys the
public API promises. It runs in a few seconds on CPU.

## Layout

```
ml4h_latentverse/
├── __init__.py             public re-exports
├── utils.py                shared helpers (encoding, MI, etc.)
├── tests/
│   ├── clustering.py       run_clustering
│   ├── disentanglement.py  run_disentanglement
│   ├── expressiveness.py   run_expressiveness
│   ├── robustness.py       run_robustness
│   └── probing.py          run_probing, run_probing_fast
└── multiloreft/
    ├── multimodal_projector.py   MultiLoReFTProjector
    └── losses.py
examples/run_cli.py         end-to-end demo on synthetic data
tests/test_smoke.py         contract / smoke tests
```

## License

MIT. See `LICENSE`.

## Authors

Yoanna Turura · Majd Alafrange · Sana Tonekaboni. Broad Institute, ML4H.
