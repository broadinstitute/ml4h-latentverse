import numpy as np
import matplotlib.pyplot as plt
from ml4h_lse.utils import fit_logistic, fit_linear

def run_expressiveness(representations, labels, folds=4, train_ratio=0.6, percent_to_remove_list=[0, 5, 10, 20], verbose=False, plots=True):
    """
    Evaluates the expressiveness of learned representations by measuring performance (AUC or R²) 
    as correlated dimensions are removed.

    Parameters:
    - representations: (N, D) array of feature representations
    - labels: (N,) array of target labels
    - folds: Number of cross-validation folds
    - train_ratio: Ratio of data used for training
    - percent_to_remove_list: List of percentages of highly correlated dimensions to remove
    - verbose: If True, prints training details
    - plots: If True, generates a plot of performance vs. removed dimensions

    Returns:
    - results: Dictionary mapping percentage of removed dimensions to performance scores
    """
    results = {}

    # Remove NaN values from labels
    mask = ~np.isnan(labels)
    labels, representations = labels[mask], representations[mask]

    # Determine if the task is classification or regression
    is_categorical = len(np.unique(labels)) <= 2

    # Normalize representations
    representations = (representations - representations.mean(axis=0)) / representations.std(axis=0)

    # Compute pairwise correlation and rank most correlated pairs
    correlation_matrix = np.corrcoef(representations, rowvar=False)
    correlation_pairs = [
        (i, j, abs(correlation_matrix[i, j]))
        for i in range(correlation_matrix.shape[0]) for j in range(i + 1, correlation_matrix.shape[0])
    ]
    correlation_pairs = sorted(correlation_pairs, key=lambda x: x[2], reverse=True)

    # Initialize metric scores storage
    metric_scores = {percent: [] for percent in percent_to_remove_list}

    for percent_to_remove in percent_to_remove_list:
        # Identify highly correlated dimensions to remove
        num_dims_to_remove = int((percent_to_remove / 100) * representations.shape[1])
        dims_to_remove = {correlation_pairs[i][1] for i in range(num_dims_to_remove)} if percent_to_remove > 0 else set()

        for _ in range(folds):
            # Shuffle and split dataset into train-test
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            train_size = int(len(labels) * train_ratio)
            train_idx, test_idx = indices[:train_size], indices[train_size:]

            X_train, X_test = representations[train_idx], representations[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Remove selected dimensions
            if dims_to_remove:
                X_train = np.delete(X_train, list(dims_to_remove), axis=1)
                X_test = np.delete(X_test, list(dims_to_remove), axis=1)

            # Evaluate expressiveness using logistic or linear regression
            metrics = fit_logistic(X_train, X_test, y_train, y_test, verbose) if is_categorical else fit_linear(X_train, X_test, y_train, y_test, verbose)
            metric_scores[percent_to_remove].append(metrics['AUROC' if is_categorical else 'R²'])

    # Compute mean performance scores
    results = {percent: np.mean(scores) for percent, scores in metric_scores.items()}

    # Plot performance drop as correlated dimensions are removed
    if plots:
        plt.figure(figsize=(8, 6))
        plt.plot(percent_to_remove_list, [results[percent] for percent in percent_to_remove_list], marker='o')
        plt.xlabel("Percentage of Dimensions Removed", fontsize=14)
        plt.ylabel("Metric Score (AUC or R²)", fontsize=14)
        plt.title("Expressiveness Test Across Phenotypes", fontsize=16)
        plt.grid()
        plt.show()

    return results
