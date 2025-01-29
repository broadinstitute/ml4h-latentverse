import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy
from ml4h_lse.tests.probing import fit_logistic, fit_linear

def estimate_intrinsic_dimension(embeddings, k=5):    
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        distances = distances[:, 1:]
    
        log_ratios = np.log(distances[:, -1]) - np.log(distances[:, 0])
        id_estimate = (k - 1) / np.mean(log_ratios)
        return id_estimate

def run_disentanglement(representations, phenotypes):
    results = {"DCI": {}, "SAP": {}, "MIG": {}, "TC": {}}

    for phenotype in phenotypes.columns:
        full_data = phenotypes[phenotypes[phenotype].notna()]
        embeddings = representations[full_data.index]
        all_labels = full_data[phenotype].values
        if all_labels.ndim > 1:
            all_labels = all_labels.ravel()
        
        is_categorical = len(np.unique(all_labels)) <= 2

        # normalize representations
        embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)

        # DCI
        importance_matrix = np.zeros((len(all_labels), embeddings.shape[1]))
        for dim in range(embeddings.shape[1]):
            model = Lasso(alpha=0.01) # lasso for linear disentanglement
            model.fit(embeddings[:, dim].reshape(-1, 1), all_labels)
            importance_matrix[:, dim] = np.abs(model.coef_[0])
        importance_matrix += 1e-8

        total_importance = importance_matrix.sum(axis=1, keepdims=True)
        total_importance[total_importance == 0] = 1e-8
        probabilities = importance_matrix / (total_importance + 1e-8)
        # replace w uniform dist if there are zero probs
        probabilities[probabilities == 0] = 1.0 / len(all_labels)

        disentanglement = 1 - entropy(probabilities, axis=1).mean()
        completeness = 1 - entropy(probabilities.T, axis=1).mean()

        indices = np.arange(len(all_labels))
        np.random.shuffle(indices)
        train_size = int(len(all_labels) * 0.8)
        train_idx, test_idx = indices[:train_size], indices[train_size:]

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = all_labels[train_idx], all_labels[test_idx]

        if is_categorical:
            metrics = fit_logistic(phenotype, X_train, X_test, y_train, y_test)
            informativeness = metrics['AUROC']
        else:
            metrics = fit_linear(phenotype, X_train, X_test, y_train, y_test)
            informativeness = metrics['R²']

        results["DCI"][phenotype] = {"Disentanglement": disentanglement, "Completeness":completeness, "Informativeness": informativeness}

        # SAP
        sap_scores = []
        for dim in range(embeddings.shape[1]):
            if is_categorical:
                model = LogisticRegression()
                model.fit(embeddings[:, dim].reshape(-1, 1), all_labels)
                sap_scores.append(roc_auc_score(all_labels, model.predict_proba(embeddings[:, dim].reshape(-1, 1))[:, 1]))
            else:
                model = Lasso(alpha=0.01)
                model.fit(embeddings[:, dim].reshape(-1, 1), all_labels)
                sap_scores.append(model.coef_[0])

        sap_gap = np.max(sap_scores) - np.partition(sap_scores, -2)[-2]
        results["SAP"][phenotype] = sap_gap

        # MIG (categorical labels only)
        if len(np.unique(all_labels)) <= 10:  # discretization threshold
            mutual_info = []
            for dim in range(embeddings.shape[1]):
                hist_2d, _, _ = np.histogram2d(embeddings[:, dim], all_labels, bins=10)
                joint_prob = hist_2d / np.sum(hist_2d)
                joint_prob += 1e-8
                marginals_z = np.sum(joint_prob, axis=0) + 1e-8
                marginals_v = np.sum(joint_prob, axis=1) + 1e-8

                denominator = marginals_v[:, None] * marginals_z[None, :]
                denominator[denominator == 0] = 1e-8
                
                mi = np.nansum(
                    joint_prob * np.log(joint_prob / (marginals_v[:, None] * marginals_z[None, :]))
                )
                mutual_info.append(mi)

            mig = (np.max(mutual_info) - np.partition(mutual_info, -2)[-2]) / entropy(all_labels)
            results["MIG"][phenotype] = mig

        # TC
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        gmm.fit(embeddings)
        joint_log_prob = gmm.score_samples(embeddings)

        marginal_log_prob = 0
        for dim in range(embeddings.shape[1]):
            gmm_dim = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
            gmm_dim.fit(embeddings[:, dim].reshape(-1, 1))
            marginal_log_prob += gmm_dim.score_samples(embeddings[:, dim].reshape(-1, 1))

        total_correlation = np.mean(joint_log_prob - marginal_log_prob)
        results["TC"][phenotype] = total_correlation

    
    print(results)
    return results