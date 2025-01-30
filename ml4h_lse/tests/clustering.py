import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier


def run_clustering(representations, phenotypes, num_clusters, plots=False):
    results = {}

    for phenotype in phenotypes.columns:
        all_labels = phenotypes[phenotype].values

        if np.isnan(all_labels).any():
            print(f"Skipping {phenotype} due to NaN values in labels.")
            continue

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(representations)

        silhouette = silhouette_score(representations, cluster_labels)
        davies_bouldin = davies_bouldin_score(representations, cluster_labels)
        nmi = normalized_mutual_info_score(all_labels, cluster_labels)

        cluster_centers = kmeans.cluster_centers_
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(cluster_centers, np.arange(num_clusters))
        knn_labels = knn.predict(representations)
        cl_score = np.mean(knn_labels == cluster_labels)

        results[phenotype] = {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Normalized Mutual Information': nmi,
            'Cluster Learnability': cl_score
        }

        if plot:
            visualize_clusterings(representations, cluster_labels, num_clusters, phenotypes, phenotype)

    print(results)
    return results

def visualize_clusterings(representations, cluster_labels, num_clusters, phenotypes, phenotype):
    markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>', 'h', 'H', '+', 'x', 'd', '|', '_', '8', '1', '2']
    if num_clusters > len(markers):
        raise ValueError(f"Number of clusters ({num_clusters}) exceeds the number of available markers ({len(markers)}).")

    plt.figure(figsize=(8, 6))

    unique_phenotype_values = np.unique(phenotypes[phenotype].values)
    phenotype_colors = sns.color_palette('tab10', n_colors=len(unique_phenotype_values))

    for cluster_idx in range(num_clusters):
        cluster_mask = cluster_labels == cluster_idx

        for value_idx, phenotype_value in enumerate(unique_phenotype_values):
            phenotype_mask = phenotypes[phenotype].values == phenotype_value
            combined_mask = cluster_mask & phenotype_mask

            sns.scatterplot(
                x=representations[combined_mask, 0],
                y=representations[combined_mask, 1],
                color=phenotype_colors[value_idx],
                marker=markers[cluster_idx],
                label=f'Value {phenotype_value} (Cluster {cluster_idx})',
                alpha=0.7
            )

    plt.title(f'Clustering Visualization for Phenotype: {phenotype}', fontsize=16)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend(title="Legend", loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
