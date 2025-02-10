import numpy as np
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from ml4h_lse.utils import fit_logistic, fit_linear
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def run_disentanglement(representations, labels):
    """
    Evaluates disentanglement metrics for latent space representations.

    Parameters:
    - representations: (N, D) matrix of latent representations
    - labels: (N,) array of ground truth labels

    Returns:
    - Dictionary containing DCI, SAP, MIG, and Total Correlation (TC) metrics
    """
    results = {}

    # Ensure labels and representations are NumPy arrays
    labels = np.asarray(labels).reshape(-1)  # Ensure labels are 1D
    representations = np.asarray(representations)

    # Ensure labels and representations have the same number of rows
    min_samples = min(labels.shape[0], representations.shape[0])
    labels = labels[:min_samples]
    representations = representations[:min_samples, :]

    # Apply mask AFTER ensuring same shape
    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    is_categorical = len(np.unique(labels)) <= 2  # Determine if labels are categorical

    # **Fix: Standardize Embeddings** (Prevents extreme values affecting TC)
    embeddings = (representations - representations.mean(axis=0)) / (representations.std(axis=0) + 1e-8)

    # **Fix DCI Probability Normalization**
    importance_matrix = np.zeros((embeddings.shape[1],))

    for dim in range(embeddings.shape[1]):
        model = Lasso(alpha=0.01)  # Lasso regression for feature importance
        model.fit(embeddings[:, dim].reshape(-1, 1), labels)
        importance_matrix[dim] = np.abs(model.coef_[0])

    importance_matrix += 1e-8  # Prevent zero values
    total_importance = np.sum(importance_matrix, keepdims=True)
    total_importance = np.maximum(total_importance, 1e-8)  # Ensure nonzero division
    probabilities = np.clip(importance_matrix / total_importance, 1e-8, 1)  # Ensure valid probabilities

    disentanglement = 1 - entropy(probabilities)  # Fix negative values
    completeness = 1 - entropy(probabilities.T)  # Fix normalization issues

    # **Train-Test Split for Informativeness**
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_size = int(len(labels) * 0.8)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    if is_categorical:
        metrics = fit_logistic(X_train, X_test, y_train, y_test)
        informativeness = metrics['AUROC']
    else:
        metrics = fit_linear(X_train, X_test, y_train, y_test)
        informativeness = metrics['R²']

    results["DCI"] = {
        "Disentanglement": max(0, disentanglement),  # Ensure valid range
        "Completeness": max(0, completeness),
        "Informativeness": informativeness
    }

    # **Fix MIG Calculation**
    def compute_mig(embeddings, labels, is_continuous):
        """
        Compute the Mutual Information Gap (MIG).
        """
        d = embeddings.shape[1]
        I_matrix = np.zeros((d, 1))

        for i in range(d):
            X = embeddings[:, i].reshape(-1, 1)
            if is_continuous:
                I_matrix[i] = mutual_info_regression(X, labels.reshape(-1, 1)).item()
            else:
                I_matrix[i] = mutual_info_classif(X, labels)

        if is_continuous:
            hist, bin_edges = np.histogram(labels, bins=30, density=True)
            hist = hist[hist > 0]  # Avoid log(0)
            H = -np.sum(hist * np.log(hist)) * (bin_edges[1] - bin_edges[0])
        else:
            values, counts = np.unique(labels, return_counts=True)
            probabilities = counts / len(labels)
            H = entropy(probabilities)

        sorted_I = np.sort(I_matrix)[::-1]
        mig_value = (sorted_I[0] - sorted_I[1]) / H if H > 0 else 0  # Prevent division by zero
        return max(0, mig_value[0])  # Ensure MIG is non-negative

    results["MIG"] = compute_mig(embeddings, labels, is_continuous=len(np.unique(labels)) > 2)

    # **Total Correlation (TC) using Gaussian Mixture Models**
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(embeddings)
    joint_log_prob = gmm.score_samples(embeddings)

    marginal_log_prob = np.zeros_like(joint_log_prob)
    for dim in range(embeddings.shape[1]):
        gmm_dim = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm_dim.fit(embeddings[:, dim].reshape(-1, 1))
        marginal_log_prob += gmm_dim.score_samples(embeddings[:, dim].reshape(-1, 1))

    total_correlation = np.mean(joint_log_prob - marginal_log_prob)
    results["TC"] = max(0, total_correlation)  # Ensure TC is non-negative

    return results
