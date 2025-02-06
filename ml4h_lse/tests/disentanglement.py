import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from ml4h_lse.utils import fit_logistic, fit_linear

def estimate_intrinsic_dimension(embeddings, k=5):
    """
    Estimate the intrinsic dimensionality of embeddings using k-nearest neighbors (k-NN).
    
    Parameters:
    - embeddings: (N, D) array of embedded features
    - k: Number of nearest neighbors to consider
    
    Returns:
    - Estimated intrinsic dimension
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    distances = distances[:, 1:]  # Ignore self-distance

    log_ratios = np.log(distances[:, -1]) - np.log(distances[:, 0])
    return (k - 1) / np.mean(log_ratios)

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

    # Remove NaN entries
    mask = ~np.isnan(labels)
    labels, representations = labels[mask], representations[mask]
    
    is_categorical = len(np.unique(labels)) <= 2  # Determine if labels are categorical

    # Normalize representations
    embeddings = (representations - representations.mean(axis=0)) / representations.std(axis=0)

    # **DCI: Disentanglement, Completeness, Informativeness**
    importance_matrix = np.zeros((len(labels), embeddings.shape[1]))

    for dim in range(embeddings.shape[1]):
        model = Lasso(alpha=0.01)  # Lasso regression for feature importance
        model.fit(embeddings[:, dim].reshape(-1, 1), labels)
        importance_matrix[:, dim] = np.abs(model.coef_[0])

    importance_matrix += 1e-8  # Avoid zero values
    total_importance = importance_matrix.sum(axis=1, keepdims=True)
    total_importance[total_importance == 0] = 1e-8

    probabilities = importance_matrix / (total_importance + 1e-8)
    probabilities[probabilities == 0] = 1.0 / len(labels)  # Replace zero probs with uniform distribution

    disentanglement = 1 - entropy(probabilities, axis=1).mean()
    completeness = 1 - entropy(probabilities.T, axis=1).mean()

    # **Train-Test Split for Informativeness**
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    train_size = int(len(labels) * 0.8)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Fit logistic regression for classification tasks or linear regression for continuous variables
    if is_categorical:
        metrics = fit_logistic(X_train, X_test, y_train, y_test)
        informativeness = metrics['AUROC']
    else:
        metrics = fit_linear(X_train, X_test, y_train, y_test)
        informativeness = metrics['R²']

    results["DCI"] = {
        "Disentanglement": disentanglement,
        "Completeness": completeness,
        "Informativeness": informativeness
    }

    # **SAP: Separated Attribute Predictability**
    sap_scores = []
    for dim in range(embeddings.shape[1]):
        model = LogisticRegression() if is_categorical else Lasso(alpha=0.01)
        model.fit(embeddings[:, dim].reshape(-1, 1), labels)
        
        if is_categorical:
            sap_scores.append(roc_auc_score(labels, model.predict_proba(embeddings[:, dim].reshape(-1, 1))[:, 1]))
        else:
            sap_scores.append(model.coef_[0])

    sap_gap = np.max(sap_scores) - np.partition(sap_scores, -2)[-2]
    results["SAP"] = sap_gap

    # **MIG: Mutual Information Gap (Categorical labels only)**
    if len(np.unique(labels)) <= 10:  # Apply MIG if labels are discrete
        mutual_info = []
        for dim in range(embeddings.shape[1]):
            hist_2d, _, _ = np.histogram2d(embeddings[:, dim], labels, bins=10)
            joint_prob = hist_2d / np.sum(hist_2d)
            joint_prob += 1e-8  # Avoid division by zero
            marginals_z = np.sum(joint_prob, axis=0) + 1e-8
            marginals_v = np.sum(joint_prob, axis=1) + 1e-8

            denominator = marginals_v[:, None] * marginals_z[None, :]
            denominator[denominator == 0] = 1e-8
            
            mi = np.nansum(joint_prob * np.log(joint_prob / denominator))
            mutual_info.append(mi)

        mig = (np.max(mutual_info) - np.partition(mutual_info, -2)[-2]) / entropy(labels)
        results["MIG"] = mig

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
    results["TC"] = total_correlation

    return results
