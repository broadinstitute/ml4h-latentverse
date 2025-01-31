# Latent Space Explorer: Evaluation library for latent representations

LSE is a library for evaluating the quality and reliability of latent representations. In includes a variety of evaluation tests to measure the following properties of latent representations:

- Clusterability
- Predictability
- Dissentanglement
- Robustness
- Expressivness

## Installation

```bash
pip install ml4h-lse
```

## Setting up the environment 


## Evaluating representations
```bash
from ml4h_lse.tests import clustering as cluster_test

clusterability_metrics = cluster_test.run_clustering(representations=representations, phenotypes=labels, num_clusters=2)
```
