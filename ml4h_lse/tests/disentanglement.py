import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from ml4h_lse.utils import fit_logistic, fit_linear
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def compute_sap_score(representations, labels, is_continuous):
    """
    Compute the Separated Attribute Predictability (SAP) score.
    
    :param representations: (N, d) array of inferred latents
    :param labels: (N,) array of ground-truth factors
    :param is_continuous: booleans indicating if each factor is continuous (regression) or categorical (classification)
    :return: SAP score
    """
    d = representations.shape[1]  # Number of latent dimensions   
    S = np.zeros((d, ))  # Score matrix
    for i in range(d):
        X = representations[:, i].reshape(-1, 1)
        
        if is_continuous:  # Regression task
            model = LinearRegression()
            model.fit(X, labels)
            y_pred = model.predict(X)
            covariance = np.cov(labels, y_pred, bias=True)[0, 1]
            std_y_true = np.std(labels, ddof=0)
            std_y_pred = np.std(y_pred, ddof=0)
            S[i] = (covariance / (std_y_true * std_y_pred)) ** 2
        else:  # Classification task
            le = LabelEncoder()
            labels_encoded = le.fit_transform(labels)
            unique_classes = len(np.unique(labels_encoded))
            thresholds = np.linspace(np.min(X), np.max(X), unique_classes)
            best_balanced_accuracy = 0
            for threshold in thresholds:
                y_pred = np.digitize(X.flatten(), bins=thresholds) - 1
                balanced_acc = balanced_accuracy_score(labels_encoded, y_pred)
                best_balanced_accuracy = max(best_balanced_accuracy, balanced_acc)
            S[i] = best_balanced_accuracy
    
    # Compute SAP score by averaging the best latent factor for each generative factor
    sap_score = np.max(S, axis=0)
    return sap_score


def compute_mig(representations, labels, is_continuous):
    """
    Compute the Mutual Information Gap (MIG).
    
    :param representations: (N, d) array of inferred latents
    :param labels: (N,) array of ground-truth factors
    :param is_continuous: List of booleans indicating if each factor is continuous (regression) or categorical (classification)
    :return: MIG score
    """
    d = representations.shape[1]  # Number of latent dimensions
    
    I_matrix = np.zeros((d, 1))  # Mutual information matrix
    
    for i in range(d):
        X = representations[:, i].reshape(-1, 1)
        if is_continuous:  # Regression task
            I_matrix[i] = mutual_info_regression(X, labels.reshape(-1, 1)).item()
        else:  # Classification task
            I_matrix[i] = mutual_info_classif(X, labels)
    H = np.std(labels)  # Approximate entropy with std
    sorted_I = np.sort(I_matrix)[::-1]  # Sort mutual information values
    mig_value = (sorted_I[0] - sorted_I[1]) / H
    return mig_value[0]


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

    # Ensure labels and representations are NumPy arrays
    labels = np.asarray(labels).reshape(-1)  # Ensure labels are 1D
    representations = np.asarray(representations)

    # Ensure labels and representations have the same number of rows
    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    # Apply mask AFTER ensuring same shape
    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]
    
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
    results["SAP"] = compute_sap_score(embeddings, labels, is_continuous=len(np.unique(labels))>2)

    # **MIG: Mutual Information Gap (Categorical labels only)**
    results["MIG"] = compute_mig(embeddings, labels, is_continuous=len(np.unique(labels))>2)

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
