import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ml4h_lse.tests.probing import run_probing
from ml4h_lse.tests.clustering import run_clustering
import os

PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_robustness(representations, labels, noise_levels, metric="clustering", plots=True):
    """
    Evaluates robustness of learned representations by adding noise and measuring clustering or probing performance.
    """
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    mask = ~np.isnan(labels)
    labels = labels[mask]

    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    representations = np.asarray(representations)

    min_samples = min(labels.shape[0], representations.shape[0])
    labels = labels[:min_samples]
    representations = representations[:min_samples, :]
    representations = representations[mask, :]

    noisy_scores = {}

    for noise_level in noise_levels:
        noisy_representations = representations + noise_level * np.random.normal(size=representations.shape)

        results = {}
        if metric == "clustering":
            results = run_clustering(representations=noisy_representations, labels=labels)
            results = results.get("results", {})
        elif metric == "probing":
            results = run_probing(representations=noisy_representations, labels=labels)
            results = results.get("metrics", {})

        print(f"Noise Level: {noise_level}, Results: {results}")  # Debugging

        for key, value in results.items():
            print(f"Key: {key}, Value: {value}")  # Debugging
            if key not in noisy_scores:
                noisy_scores[key] = []

            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                if isinstance(value[0], (list, np.ndarray)):  # Fix nested lists issue
                    noisy_scores[key].append(value[0][0])
                else:
                    noisy_scores[key].append(value[0])
            else:
                noisy_scores[key].append(float(value) if isinstance(value, (int, float, np.number)) else value)

    print("Final Noisy Scores:", noisy_scores)  # Debugging

    plot_url = None
    if plots:
        plt.figure(figsize=(8, 6))
        for key, values in noisy_scores.items():
            print(f"Plotting {key}: {values}")  # Debugging

            if all(isinstance(v, (int, float, np.number)) for v in values):
                plt.plot(noise_levels, values, marker="o", label=f"{key}")
            else:
                print(f"Skipping {key}, contains non-numeric values: {values}")  # Debugging

        plt.xlabel("Noise Level", fontsize=14)
        plt.ylabel("Performance Score", fontsize=14)
        plt.title(f"Representation Robustness ({metric.capitalize()})", fontsize=16)
        plt.legend()
        plt.grid()

        plot_filename = f"robustness_{metric}.png"
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_filepath, format="png", dpi=300)
        plt.close()

        if os.path.exists(plot_filepath):
            print(f"✅ Plot saved successfully at {plot_filepath}")
            plot_url = f"/static/plots/{plot_filename}"
        else:
            print("⚠️ Plot was NOT saved correctly!")

    return {"metrics": noisy_scores, "plot_url": plot_url}
