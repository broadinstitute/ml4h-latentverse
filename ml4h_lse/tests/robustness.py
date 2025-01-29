import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tests.probing import fit_logistic, fit_linear

def run_robustness_silhouette(representations, phenotypes, noise_levels, num_clusters=2):
    noisy_scores = {phenotype: [] for phenotype in phenotypes.columns}

    for phenotype in phenotypes.columns:
        print(f"Processing phenotype: {phenotype}")

        for noise_level in noise_levels:
            noisy_representations = representations + noise_level * np.random.normal(size=representations.shape)
            
            try:
                cluster_all_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(noisy_representations)
                noisy_score = silhouette_score(noisy_representations, cluster_all_labels)
                noisy_scores[phenotype].append(noisy_score)
            except Exception as e:
                print(f"Error computing silhouette score for {phenotype} at noise level {noise_level}: {e}")
                noisy_scores[phenotype].append(None)

    plt.figure(figsize=(8, 6))
    for phenotype, scores in noisy_scores.items():
        if scores:
            plt.plot(
                noise_levels,
                scores,
                marker='o',
                label=f'{phenotype}'
            )

    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.title('Silhouette Score under Perturbation', fontsize=16)
    plt.legend(title="Phenotype", loc='best')
    plt.grid()
    plt.show()

    results = {
        phenotype: {
            "Noisy Silhouette Scores": scores
        }
        for phenotype, scores in noisy_scores.items()
    }
    return results


def run_robustness_probing(representations, phenotypes, noise_levels, folds=4, train_ratio=0.6, verbose=False):
    results = {phenotype: {'auc': [], 'r2': []} for phenotype in phenotype.columns}

    for phenotype in phenotype.columns:
        try:
            full_data = phenotypes[phenotypes[phenotype].notna()]
            embeddings = representations[full_data.index]
            all_labels = full_data[phenotype].values

            is_categorical = len(np.unique(all_labels)) <= 2

            for noise_level in noise_levels:
                noisy_embeddings = embeddings + noise_level * np.random.normal(size=embeddings.shape)
                s = []

                for _ in range(folds):
                    indices = np.arange(len(all_labels))
                    np.random.shuffle(indices)
                    train_size = int(len(all_labels) * train_ratio)
                    train_idx, test_idx = indices[:train_size], indices[train_size:]

                    X_train, X_test = noisy_embeddings[train_idx], noisy_embeddings[test_idx]
                    y_train, y_test = all_labels[train_idx], all_labels[test_idx]

                    if is_categorical:
                        metrics = fit_logistic(phenotype, X_train, X_test, y_train, y_test, verbose)
                        s.append(metrics['AUROC'])  # classification
                    else:
                        metrics = fit_linear(phenotype, X_train, X_test, y_train, y_test, verbose)
                        s.append(metrics['R²'])  # regression

                if is_categorical:
                    results[phenotype]['auc'].append(np.mean(s))
                else:
                    results[phenotype]['r2'].append(np.mean(s))

        except Exception as e:
            print(f"Error in perturbation test for phenotype '{phenotype}': {e}")

    plt.figure(figsize=(8, 6))
    for phenotype, metrics in results.items():
        if metrics['auc']:
            plt.plot(noise_levels, metrics['auc'], marker='o', label=f'{phenotype} (AUC)')
        if metrics['r2']:
            plt.plot(noise_levels, metrics['r2'], marker='s', label=f'{phenotype} (R²)')

    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Metric Score (AUC or R²)', fontsize=14)
    plt.title('Probing under Perturbation', fontsize=16)
    plt.legend(title="Phenotype", loc='best')
    plt.grid()
    plt.show()

    return results