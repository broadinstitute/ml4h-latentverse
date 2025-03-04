import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ml4h_lse.utils import fit_logistic, fit_linear

PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_expressiveness(representations, labels, folds=4, train_ratio=0.6, percent_to_remove_list=[0, 5, 10, 20], verbose=False, plots=True):
    """
    Evaluates the expressiveness of learned representations by measuring performance (AUC or R²) 
    as correlated dimensions are removed.

    Parameters:
    - representations: (N, D) array of feature representations
    - labels: (N, P) array of target labels (P phenotypes)
    - folds: Number of cross-validation folds
    - train_ratio: Ratio of data used for training
    - percent_to_remove_list: List of percentages of highly correlated dimensions to remove
    - verbose: If True, prints training details
    - plots: If True, generates and saves a performance plot

    Returns:
    - dict: Contains performance scores and the plot URL
    """
    # Convert to NumPy arrays if they are Pandas DataFrames
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()  # Ensure labels are 2D if multiple phenotypes

    # Ensure representations are at least 2D
    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    # Ensure no NaN values in representations
    representations = np.nan_to_num(representations)

    # Normalize representations
    representations = (representations - representations.mean(axis=0)) / representations.std(axis=0)

    # Compute pairwise correlation and rank most correlated pairs
    correlation_matrix = np.corrcoef(representations, rowvar=False)
    correlation_pairs = sorted(
        [(i, j, abs(correlation_matrix[i, j])) for i in range(correlation_matrix.shape[1]) for j in range(i + 1, correlation_matrix.shape[1])],
        key=lambda x: x[2], reverse=True
    )

    # Initialize results storage
    results = {}

    # If `labels` is 1D, convert it to 2D with one column
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    phenotype_names = [f"Phenotype {i+1}" for i in range(labels.shape[1])]  # Placeholder names

    # Store performance per phenotype
    phenotype_scores = {phenotype: {percent: [] for percent in percent_to_remove_list} for phenotype in phenotype_names}

    for phenotype_idx, phenotype_name in enumerate(phenotype_names):
        y = labels[:, phenotype_idx]  # Extract one phenotype at a time

        # Apply NaN mask
        mask = ~np.isnan(y)
        y, X = y[mask], representations[mask, :]

        is_categorical = len(np.unique(y)) <= 2  # Check if binary classification

        for percent_to_remove in percent_to_remove_list:
            num_dims_to_remove = max(1, int((percent_to_remove / 100) * X.shape[1]))
    
            feature_label_corr = np.abs(np.corrcoef(X.T, y.squeeze())[:-1, -1])
            feature_importance = np.argsort(feature_label_corr)[::-1]
            dims_to_remove = set(feature_importance[:num_dims_to_remove])

            for _ in range(folds):
                # Shuffle and split dataset into train-test
                indices = np.arange(len(y))
                np.random.shuffle(indices)
                train_size = int(len(y) * train_ratio)
                train_idx, test_idx = indices[:train_size], indices[train_size:]

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Remove selected dimensions
                if dims_to_remove:
                    X_train = np.delete(X_train, list(dims_to_remove), axis=1)
                    X_test = np.delete(X_test, list(dims_to_remove), axis=1)

                # Evaluate expressiveness using logistic or linear regression
                metrics = fit_logistic(X_train, X_test, y_train, y_test, verbose) if is_categorical else fit_linear(X_train, X_test, y_train, y_test, verbose)
                phenotype_scores[phenotype_name][percent_to_remove].append(metrics['AUROC' if is_categorical else 'R²'])

    # Compute mean performance scores per phenotype
    for phenotype_name in phenotype_scores:
        results[phenotype_name] = {percent: np.mean(scores) for percent, scores in phenotype_scores[phenotype_name].items()}

    # Generate and save the plot
    plot_url = None
    if plots:
        plt.figure(figsize=(8, 6))

        # Plot each phenotype separately
        for phenotype_name, metric_data in results.items():
            plt.plot(percent_to_remove_list, [metric_data[percent] for percent in percent_to_remove_list], marker='o', label=phenotype_name)

        plt.xlabel("Percentage of Dimensions Removed", fontsize=14)
        plt.ylabel("Metric Score (AUC or R²)", fontsize=14)
        plt.title("Expressiveness Test Across Phenotypes", fontsize=16)
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = "expressiveness.png"
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        plot_url = f"/static/plots/{plot_filename}"

    return {"metrics": results, "plot_url": plot_url}
