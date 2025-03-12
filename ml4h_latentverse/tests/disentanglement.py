import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from ml4h_latentverse.utils import fit_logistic
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


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

    labels = np.asarray(labels).reshape(-1) 
    representations = np.asarray(representations)

    min_samples = min(labels.shape[0], representations.shape[0])
    labels = labels[:min_samples]
    representations = representations[:min_samples, :]

    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    is_continuous=len(np.unique(labels)) > 2

    d = representations.shape[1]
    I_matrix = np.zeros((d, 1))
    for i in range(d):
        X = representations[:, i].reshape(-1, 1)
        if is_continuous:
            I_matrix[i] = mutual_info_regression(X, labels.reshape(-1, 1)).item()
        else:
            I_matrix[i] = mutual_info_classif(X, labels.reshape(-1, 1)).item()

    disentanglement = compute_disentanglement_score(I_matrix)
    completeness = compute_completeness_score(I_matrix)
    informativeness = compute_informativeness_score(representations, labels, is_continuous)

    results["DCI"] = {
        "Disentanglement": max(0, disentanglement),
        "Completeness": max(0, completeness),
        "Informativeness": max(0, min(informativeness, 1))
    }

    results["MIG"] = compute_mig(I_matrix, labels)

    results["TC"] = compute_total_correlation(representations)

    return results

def compute_total_correlation(representations):
    """
    Total Correlation (TC) using Gaussian Mixture Models.
    """
    n_components = min(5, len(representations))

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(representations)
    joint_log_prob = gmm.score_samples(representations)

    marginal_log_prob = np.zeros_like(joint_log_prob)
    for dim in range(representations.shape[1]):
        gmm_dim = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm_dim.fit(representations[:, dim].reshape(-1, 1))
        marginal_log_prob += gmm_dim.score_samples(representations[:, dim].reshape(-1, 1))

    total_correlation = np.mean(joint_log_prob - marginal_log_prob)
    return max(0, total_correlation)


def compute_mig(I_matrix, labels):
    """
    Compute the Mutual Information Gap (MIG).
    """
    sorted_I = np.sort(I_matrix)[::-1]

    unique, counts = np.unique(labels, return_counts=True)
    prob_dist = counts / counts.sum()
    H = entropy(prob_dist)
    
    if H > 1e-6 and len(sorted_I) > 1:  
        mig_value = (sorted_I[0] - sorted_I[1]) / (H + 1e-6)
    else:
        mig_value = 0
    return min(max(mig_value, 0), 1)
    

def compute_disentanglement_score(I_matrix):
    """
    Disentanglement score (D) for single factor.
    Measures concentration of importance across latents.
    """
    P = I_matrix.flatten() / (np.sum(I_matrix) + 1e-10)
    H = -np.sum(P * np.log(P + 1e-10)) / np.log(len(P))
    D = 1 - H
    return np.clip(D, 0, 1)

def compute_completeness_score(I_matrix):
    """
    Completeness score (C) for single factor.
    Same as disentanglement when only one factor exists.
    """
    return compute_disentanglement_score(I_matrix)

def compute_informativeness_score(representations, labels, is_continuous):
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_size = int(len(labels) * 0.6)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = representations[train_idx], representations[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    if is_continuous:
        clf = make_pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42))
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        informativeness = 1 - (np.mean((y_test-y_pred)**2) / np.var(y_test))
    else:
        metrics = fit_logistic(X_train, X_test, y_train, y_test)
        informativeness = metrics['AUROC']
        
    return informativeness