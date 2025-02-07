import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    # Convert to NumPy arrays if they are Pandas DataFrames
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)  # Ensure labels are 1D

    # Ensure labels are 1D
    labels = np.asarray(labels).reshape(-1)

    # Ensure representations are at least 2D
    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    # Ensure labels and representations have the same number of rows
    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    # Apply mask AFTER ensuring the same shape
    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    noisy_scores = {}  # Initialize dictionary to store results

    plt.figure(figsize=(8, 6))

    for noise_level in noise_levels:
        # Apply Gaussian noise
        noisy_representations = representations + noise_level * np.random.normal(size=representations.shape)

        results = {}
        
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
