import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ml4h_lse.tests.probing import run_probing
from ml4h_lse.tests.clustering import run_clustering
import os

PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_robustness(representations_list, labels, noise_levels, metric="clustering", plots=True, representation_names=None):
    """
    Evaluates robustness of multiple learned representations by adding noise and measuring clustering or probing performance.

    Parameters:
    - representations_list: List of (N, D) arrays, each containing a set of feature embeddings.
    - labels: (N,) array of target labels.
    - noise_levels: List of noise magnitudes to apply.
    - metric: "clustering" (Silhouette Score) or "probing" (Predictive Performance).
    - plots: If True, generates a performance plot.
    - representation_names: List of names corresponding to each representation matrix.

    Returns:
    - noisy_scores: Dictionary mapping metric names to lists of scores for different noise levels.
    """
    if not isinstance(representations_list, list):
        representations_list = [representations_list]  # Convert single representation to a list

    # Assign default names if not provided
    if representation_names is None:
        representation_names = [f"Representation {i+1}" for i in range(len(representations_list))]

    # Convert labels to NumPy array if needed
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)  # Ensure labels are 1D
    labels = np.asarray(labels).reshape(-1)

    # Apply mask AFTER ensuring the same shape
    mask = ~np.isnan(labels)
    labels = labels[mask]

    noisy_scores = {name: {} for name in representation_names}  # Store scores for each representation

    for rep_idx, representations in enumerate(representations_list):
        rep_name = representation_names[rep_idx]

        # Convert to NumPy arrays if needed
        if isinstance(representations, pd.DataFrame):
            representations = representations.to_numpy()
        representations = np.asarray(representations)

        # Ensure labels and representations have the same number of rows
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

        representations = representations[mask, :]  # Apply mask to remove NaNs

        noisy_scores[rep_name] = {}  # Initialize dictionary for this representation

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
                if key not in noisy_scores[rep_name]:
                    noisy_scores[rep_name][key] = []
                noisy_scores[rep_name][key].append(value[0] if metric == "probing" else value)

    plot_url = None
    if plots:
        plt.figure(figsize=(8, 6))
        for rep_name in representation_names:
            for key, values in noisy_scores[rep_name].items():
                plt.plot(noise_levels, values, label=f"{rep_name} - {key}")

        plt.xlabel("Noise Level", fontsize=14)
        plt.ylabel("Performance Score", fontsize=14)
        plt.title(f"Representation Robustness ({metric.capitalize()})", fontsize=16)
        plt.legend()
        plt.grid()

        plot_filename = f"robustness_{metric}.png"
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        plot_url = f"/static/plots/{plot_filename}"

    return {"metrics": noisy_scores, "plot_url": plot_url}

