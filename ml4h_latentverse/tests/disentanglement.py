import numpy as np
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from ml4h_latentverse.utils import fit_logistic, fit_linear
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge


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

    is_continuous=len(np.unique(labels)) > 2

    # Standardize Embeddings** (Prevents extreme values affecting TC)
    # embeddings = (representations - representations.mean(axis=0)) / (representations.std(axis=0) + 1e-8)

    d = representations.shape[1]
    I_matrix = np.zeros((d, 1))
    for i in range(d):
        X = representations[:, i].reshape(-1, 1)
        if is_continuous:
            I_matrix[i] = mutual_info_regression(X, labels.reshape(-1, 1)).item()
        else:
            I_matrix[i] = mutual_info_classif(X, labels.reshape(-1, 1)).item()

    # DCI Probability Normalization
    # importance_matrix = np.zeros((embeddings.shape[1],))

    # for dim in range(embeddings.shape[1]):
    #     model = Lasso(alpha=0.01)  # Lasso regression for feature importance
    #     model.fit(embeddings[:, dim].reshape(-1, 1), labels)
    #     importance_matrix[dim] = np.abs(model.coef_[0])

    # importance_matrix += 1e-8  # Prevent zero values
    # total_importance = np.sum(importance_matrix, keepdims=True)
    # total_importance = np.maximum(total_importance, 1e-8)  # Ensure nonzero division
    # probabilities = np.clip(importance_matrix / total_importance, 1e-8, 1)  # Ensure valid probabilities

    disentanglement = compute_disentanglement_score(I_matrix)#1-entropy(probabilities)  # negative values
    completeness = compute_completeness_score(I_matrix) # 1-entropy(probabilities.T)  # normalization issues

    # **Train-Test Split for Informativeness**
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_size = int(len(labels) * 0.6)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = representations[train_idx], representations[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    if is_continuous:
        clf = make_pipeline(StandardScaler(), Ridge(solver="lsqr", max_iter=250000))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        informativeness = 1 - (np.mean((y_test-y_pred)**2) / np.var(y_test))
    else:
        metrics = fit_logistic(X_train, X_test, y_train, y_test)
        informativeness = metrics['AUROC']

    results["DCI"] = {
        "Disentanglement": max(0, disentanglement),  # Ensure valid range
        "Completeness": max(0, completeness),
        "Informativeness": informativeness
    }

    results["MIG"] = compute_mig(I_matrix, labels, is_continuous=len(np.unique(labels)) > 2)

    # **Total Correlation (TC) using Gaussian Mixture Models**
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(representations)
    joint_log_prob = gmm.score_samples(representations)

    marginal_log_prob = np.zeros_like(joint_log_prob)
    for dim in range(representations.shape[1]):
        gmm_dim = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm_dim.fit(representations[:, dim].reshape(-1, 1))
        marginal_log_prob += gmm_dim.score_samples(representations[:, dim].reshape(-1, 1))

    total_correlation = np.mean(joint_log_prob - marginal_log_prob)
    results["TC"] = max(0, total_correlation)  # Ensure TC is non-negative

    return results


# MIG Calculation**
def compute_mig(I_matrix, labels, is_continuous):
    """
    Compute the Mutual Information Gap (MIG).
    """
    # sorted_I = np.sort(I_matrix)[::-1]  # Sort mutual information values
    # I_matrix = sorted_I + H_z
    sorted_I = np.sort(I_matrix)[::-1]

    # Compute entropy correctly
    unique, counts = np.unique(labels, return_counts=True)
    prob_dist = counts / counts.sum()
    H = entropy(prob_dist)  # Shannon entropy
    
    if H > 0 and len(sorted_I) > 1:
        mig_value = (sorted_I[0] - sorted_I[1]) / H
    else:
        mig_value = 0  # If entropy is 0 or there's only one latent
    return min(max(mig_value, 0), 1)
    

def compute_disentanglement_score(I_matrix):
    """
    Compute the disentanglement score (D) as described.
    
    :param I_matrix: (L, 1) matrix where I_matrix[i] represents the importance of latent code i for the single factor
    :return: Disentanglement score D
    """
    L = I_matrix.shape[0]
    D_i = np.zeros(L)
    for i in range(L):
        P_i = I_matrix[i] / np.sum(I_matrix)  # Normalize importance
        H_K = -P_i * np.log(P_i + 1e-10) / np.log(L)  # Entropy normalized by log(L)
        D_i[i] = 1 - H_K
    
    rho_i = I_matrix[:, 0] / np.sum(I_matrix[:, 0])  # Weighting factor
    D = np.sum(rho_i * D_i)
    return D

def compute_completeness_score(I_matrix):
    """
    Compute the completeness score (C) as described.
    
    :param I_matrix: (L, 1) matrix where I_matrix[i] represents the importance of latent code i for the single factor
    :return: Completeness score C
    """
    L = I_matrix.shape[0]
    C_j = np.zeros(1)  # Since there's only one factor
    P_j = I_matrix[:, 0] / np.sum(I_matrix[:, 0])  # Normalize importance
    H_L = -np.sum(P_j * np.log(P_j + 1e-10)) / np.log(L)  # Entropy normalized by log(L)
    C_j[0] = 1 - H_L
    
    C = np.mean(C_j)
    return C