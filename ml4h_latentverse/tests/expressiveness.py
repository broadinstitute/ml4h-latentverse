import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ml4h_latentverse.utils import fit_logistic, fit_linear

PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_expressiveness(representations, labels, folds=4, train_ratio=0.6, percent_to_remove_list=[0, 5, 10, 20], verbose=False, plots=True):
    """
    Evaluates the expressiveness of learned representations by measuring performance (AUC or R²) 
    as correlated dimensions are removed.

    Parameters:
    - representations: (N, D) array of feature representations
    - labels: (N, P) array of target labels (P labels)
    - folds: Number of cross-validation folds
    - train_ratio: Ratio of data used for training
    - percent_to_remove_list: List of percentages of highly correlated dimensions to remove
    - verbose: If True, prints training details
    - plots: If True, generates and saves a performance plot

    Returns:
    - dict: Contains performance scores and the plot URL
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()

    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    representations = np.nan_to_num(representations)

    representations = (representations - representations.mean(axis=0)) / representations.std(axis=0)

    # Compute pairwise correlation and rank most correlated pairs
    correlation_matrix = np.corrcoef(representations, rowvar=False)
    correlation_pairs = sorted(
        [(i, j, abs(correlation_matrix[i, j])) for i in range(correlation_matrix.shape[1]) for j in range(i + 1, correlation_matrix.shape[1])],
        key=lambda x: x[2], reverse=True
    )

    results = {}

    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    label_names = [f"Label {i+1}" for i in range(labels.shape[1])]

    label_scores = {label: {percent: [] for percent in percent_to_remove_list} for label in label_names}

    for label_idx, label_name in enumerate(label_names):
        y = labels[:, label_idx]

        mask = ~np.isnan(y)
        y, X = y[mask], representations[mask, :]

        is_categorical = len(np.unique(y)) <= 2

        for percent_to_remove in percent_to_remove_list:
            num_dims_to_remove = max(1, int((percent_to_remove / 100) * X.shape[1]))
    
            feature_label_corr = np.abs(np.corrcoef(X.T, y.squeeze())[:-1, -1])
            feature_importance = np.argsort(feature_label_corr)[::-1]
            dims_to_remove = set(feature_importance[:num_dims_to_remove])

            for _ in range(folds):
                indices = np.arange(len(y))
                np.random.shuffle(indices)
                train_size = int(len(y) * train_ratio)
                train_idx, test_idx = indices[:train_size], indices[train_size:]

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if dims_to_remove:
                    X_train = np.delete(X_train, list(dims_to_remove), axis=1)
                    X_test = np.delete(X_test, list(dims_to_remove), axis=1)

                metrics = fit_logistic(X_train, X_test, y_train, y_test, verbose) if is_categorical else fit_linear(X_train, X_test, y_train, y_test, verbose)
                label_scores[label_name][percent_to_remove].append(metrics['AUROC' if is_categorical else 'R²'])

    for label_name in label_scores:
        results[label_name] = {percent: np.mean(scores) for percent, scores in label_scores[label_name].items()}

    plot_url = None
    if plots:
        plt.figure(figsize=(8, 6))

        for label_name, metric_data in results.items():
            plt.plot(percent_to_remove_list, [metric_data[percent] for percent in percent_to_remove_list], marker='o', label=label_name)

        plt.xlabel("Percentage of Dimensions Removed", fontsize=14)
        plt.ylabel("Metric Score (AUC or R²)", fontsize=14)
        plt.title("Expressiveness Test Across Labels", fontsize=16)
        plt.legend()
        plt.grid(True)

        plot_filename = "expressiveness.png"
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        plot_url = f"/static/plots/{plot_filename}"

    return {"metrics": results, "plot_url": plot_url}
