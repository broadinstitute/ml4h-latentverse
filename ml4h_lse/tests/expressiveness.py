import numpy as np
import matplotlib.pyplot as plt
from ml4h_lse.tests.probing import fit_logistic, fit_linear

def run_expressiveness(representations, phenotypes, folds=4, train_ratio=0.6, percent_to_remove_list=[0, 5, 10, 20], verbose=False, plots=True):
    results = {}

    for phenotype in phenotypes.columns:
        try:
            full_data = phenotypes[phenotypes[phenotype].notna()]
            embeddings = representations[full_data.index]
            all_labels = full_data[phenotype].values

            if all_labels.ndim > 1:
                all_labels = all_labels.ravel()

            is_categorical = len(np.unique(all_labels)) <= 2

            embeddings = (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)

            metric_scores = {percent: [] for percent in percent_to_remove_list}

            correlation_matrix = np.corrcoef(embeddings, rowvar=False)
            correlation_pairs = []
            for i in range(correlation_matrix.shape[0]):
                for j in range(i + 1, correlation_matrix.shape[0]):
                    correlation_pairs.append((i, j, abs(correlation_matrix[i, j])))

            correlation_pairs = sorted(correlation_pairs, key=lambda x: x[2], reverse=True)

            for percent_to_remove in percent_to_remove_list:
                if percent_to_remove > 0:
                    num_dims_to_remove = int((percent_to_remove / 100) * embeddings.shape[1])
                    dims_to_remove = set()
                    for i in range(num_dims_to_remove):
                        if i < len(correlation_pairs):
                            dims_to_remove.add(correlation_pairs[i][1])
                else:
                    dims_to_remove = set()

                for _ in range(folds):
                    indices = np.arange(len(all_labels))
                    np.random.shuffle(indices)
                    train_size = int(len(all_labels) * train_ratio)
                    train_idx, test_idx = indices[:train_size], indices[train_size:]

                    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                    y_train, y_test = all_labels[train_idx], all_labels[test_idx]

                    if dims_to_remove and len(dims_to_remove) > 0:
                        X_train = np.delete(X_train, list(dims_to_remove), axis=1)
                        X_test = np.delete(X_test, list(dims_to_remove), axis=1)

                    if is_categorical:
                        metrics = fit_logistic(phenotype, X_train, X_test, y_train, y_test, verbose)
                        metric_scores[percent_to_remove].append(metrics['AUROC'])
                    else:
                        metrics = fit_linear(phenotype, X_train, X_test, y_train, y_test, verbose)
                        metric_scores[percent_to_remove].append(metrics['R²'])

            mean_scores = {percent: np.mean(scores) for percent, scores in metric_scores.items()}
            results[phenotype] = mean_scores

            if plots:
                plt.figure(figsize=(8, 6))
                plt.plot(
                    percent_to_remove_list,
                    [mean_scores[percent] for percent in percent_to_remove_list],
                    marker='o',
                    label=f'{phenotype}'
                )
            plt.xlabel("Percentage of Dimensions Removed", fontsize=14)
            plt.ylabel("Metric Score (AUC or R²)", fontsize=14)
            plt.title("Expressiveness Test Across Phenotypes", fontsize=16)
            plt.legend(title="Phenotype")
            plt.grid()
            plt.show()

        except Exception as e:
            print(f"Error in expressiveness test for phenotype '{phenotype}': {e}")

    print(results)
    return results