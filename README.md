# Latentverse: Evaluation library for latent representations

Latentverse is a library for evaluating the quality and reliability of latent representations. In includes a variety of evaluation tests to measure the following properties of latent representations:

- Clusterability: How well the representations form distinct clusters
- Predictability: The ability to use representations for downstream prediction tasks
- Disentanglement: The extent to which latent dimensions capture independent factors of variation
- Robustness: The resilience of representations under perturbations
- Expressiveness: How well latent representations capture relevant information


## Installation
Latentverse is available on PyPI. You can install it using:

```bash
pip install ml4h-latentverse
```

Alternatively, if you are developing or modifying the package, clone the repository and install it in editable mode:
```bash
git clone https://github.com/broadinstitute/ml4h-latentverse.git
cd ml4h-latentverse
pip install -e .
```

## Setting up the environment 
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # for MacOS/Linux
venv\Scripts\activate  # for Windows
```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Evaluating representations

# 1. Clusterability Test
This test evaluates how well representations cluster given a specified number of clusters. The test computes metrics such as Normalized Mutual Information (NMI), Silhouette Score, Davies-Bouldin Index, and Cluster Learnability.

Example usage below:
```bash
from ml4h_latentverse.tests.clustering import run_clustering

representations = ...  # Load or generate your latent representations
labels = ...  # Corresponding labels for evaluation

results = run_clustering(representations=representations, labels=labels, num_clusters=2, plots=True)
print(results)
```

Expected inputs:
- representations (ndarray): Feature representations for clustering
- num_clusters (int, optional): Number of clusters (ignored if labels are provided)
- labels (ndarray, optional): True labels (used for evaluation)
- plots (bool, optional): Whether to generate clustering visualization


# 2. Disentanglement Test
This test measures how well the latent dimensions capture independent factors of variation. It computes metrics such as DCI Disentanglement, Mutual Information Gap (MIG), Total Correlation (TC), and SAP Score.

Example usage:
```bash
from ml4h_latentverse.tests.disentanglement import run_disentanglement

data, labels = ...  # Load or generate latent representations and labels
results = run_disentanglement(data, labels)
print(results)
```
Expected inputs:
- representations: A (N, D) array of latent space representation
- labels: A (N,) array of ground truth labels

# 3. Expressiveness Test
This test evaluates how much information the representations contain about labels. It assesses the impact of removing highly correlated features on prediction performance using AUC or R² scores.

Example usage:
```bash
from ml4h_latentverse.tests.expressiveness import run_expressiveness

data, labels = ...  # Load or generate data
results = run_expressiveness(data, labels, percent_to_remove_list=[0, 10, 20, 50], plots=True)
print(results)
```

Expected inputs:
- representations: (N, D) array of feature representations
- labels: (N, P) array of target labels (P phenotypes)
- folds: Number of cross-validation folds
- train_ratio: Ratio of data used for training
- percent_to_remove_list: List of percentages of highly correlated dimensions to remove
- verbose: If True, prints training details
- plots: If True, generates and saves a performance plot


# 4. Robustness Test
This test examines how well representations withstand perturbations by introducing Gaussian noise and measuring clustering or probing performance.

Example usage:
```bash
from ml4h_latentverse.tests.robustness import run_robustness

data, labels = ...  # Load or generate data
results = run_robustness(data, labels, noise_levels=[0.1, 0.5, 1.0, 1.5], metric="clustering", plots=True)
print(results)
```

Expected inputs:
- representations: (N, D) matrix (or DataFrame) of latent representations
- labels: (N,) array (or DataFrame) of target labels
- noise_levels: List of noise magnitudes to apply
- metric: "clustering" or "probing"
- plots: If True, generate a performance plot

# 5. Probing Test
This test measures representation quality by training classifiers or regressors of varying complexity.

Example usage:
```bash
from ml4h_latentverse.tests.probing import run_probing

data, labels = ...  # Load or generate data
results = run_probing(data, labels)
print(results)
```

Expected inputs:
- representations (ndarray or DataFrame): Feature representations
- labels (ndarray or DataFrame): Labels for probing
- train_ratio (float): Ratio of train to test data


## Testing the Evaluation Suite
To validate the test suite, you can use the provided test cases:

Example usage:
```bash
from ml4h_latentverse.tests.test import test_clusterability, test_disentanglement, test_expressiveness, test_robustness

test_clusterability()
test_disentanglement()
test_expressiveness()
test_robustness()
```




