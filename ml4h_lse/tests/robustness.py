import numpy as np
import matplotlib.pyplot as plt
from ml4h_lse.tests.probing import run_probing
from ml4h_lse.tests.clustering import run_clustering

def run_robustness(representations, labels, noise_levels, metric="clustering", plots=True):
    """
    Evaluates robustness of learned representations by adding noise and measuring clustering or probing performance.

    Parameters:
    - representations: (N, D) array of feature embeddings
    - labels: (N,) array of target labels
    - noise_levels: List of noise magnitudes to apply
    - metric: "clustering" (Silhouette Score) or "probing" (Predictive Performance)
    - plots: If True, generates a performance plot

    Returns:
    - noisy_scores: Dictionary mapping metric names to lists of scores for different noise levels
    """
    noisy_scores = {}  # Initialize dictionary to store results

    plt.figure(figsize=(8, 6))

    for noise_level in noise_levels:
        # Apply Gaussian noise
        noisy_representations = representations + noise_level * np.random.normal(size=representations.shape)

        if metric == "clustering":
            results = run_clustering(representations=noisy_representations, labels=labels)
            results = results['results']

        elif metric == "probing":
            results = run_probing(representations=noisy_representations, labels=labels)
            results = results["metrics"]
        # Store results in dictionary
        for key, value in results.items():
            if key not in noisy_scores:
                noisy_scores[key] = []
            if metric == "probing":
                noisy_scores[key].append(value[0])
            elif metric == "clustering":
                noisy_scores[key].append(value)

    if plots:
        plt.figure(figsize=(8, 6))
        for key, values in noisy_scores.items():
            print(values)
            plt.plot(noise_levels, values, label=key)

        plt.xlabel("Noise Level", fontsize=14)
        plt.ylabel("Performance Score", fontsize=14)
        plt.title(f"Robustness of Representations ({metric.capitalize()})", fontsize=16)
        plt.legend()
        plt.grid()
        plt.show()

    return noisy_scores
