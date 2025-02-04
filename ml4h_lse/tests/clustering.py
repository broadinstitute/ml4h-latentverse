import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import os


def run_clustering(representations, num_clusters=None, labels=None, plots=False):
    """
    Performs KMeans clustering, evaluates clustering quality, and optionally visualizes results.

    Parameters:
        representations (ndarray): Feature representations for clustering.
        num_clusters (int, optional): Number of clusters (ignored if labels are provided).
        labels (ndarray, optional): True labels (used for evaluation).
        plots (bool, optional): Whether to generate clustering visualization.

    Returns:
        dict: Clustering evaluation metrics.
    """
    if labels is not None:
        mask = ~np.isnan(labels)
        labels, representations = labels[mask], representations[mask]
        num_clusters = len(np.unique(labels))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(representations)

    # Compute evaluation metrics
    silhouette = silhouette_score(representations, cluster_labels)
    davies_bouldin = davies_bouldin_score(representations, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels) if labels is not None else None

    # Compute cluster learnability using 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(kmeans.cluster_centers_, np.arange(num_clusters))
    cl_score = np.mean(knn.predict(representations) == cluster_labels)

    results = {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin,
        "Normalized Mutual Information": nmi,
        "Cluster Learnability": cl_score
    }

    # Optional visualization
    plot_url = None
    if plots:
        plot_url = visualize_clusterings(representations, cluster_labels, num_clusters, labels)

    return {"results": results, "plot_url": plot_url}


def visualize_clusterings(representations, cluster_labels, num_clusters, labels):
    """
    Visualizes KMeans clustering results, coloring points based on their cluster and labels.

    Parameters:
        representations (ndarray): Feature representations.
        cluster_labels (ndarray): Cluster assignments.
        num_clusters (int): Number of clusters.
        labels (ndarray): Ground truth labels for color mapping.

    """
    matplotlib.use('Agg')

    markers = ['o', 's', '^', 'P', '*', 'X', 'D', 'v', '<', '>', 'h', 'H', '+', 'x', 'd', '|', '_', '8', '1', '2']
    if num_clusters > len(markers):
        raise ValueError(f"Number of clusters ({num_clusters}) exceeds available markers ({len(markers)}).")

    plt.figure(figsize=(8, 6))
    # Compute first two principal components (PCA)
    pca = PCA(n_components=2)
    pca_rep = pca.fit_transform(representations)

    # Define color palette for groups
    unique_labels = np.unique(labels)
    colors = sns.color_palette('tab10', n_colors=len(unique_labels))

    for cluster_idx in range(num_clusters):
        cluster_mask = cluster_labels == cluster_idx

        for label_idx, label_value in enumerate(unique_labels):
            phenotype_mask = labels == label_value
            combined_mask = cluster_mask & phenotype_mask

            sns.scatterplot(
                x=pca_rep[combined_mask, 0],
                y=pca_rep[combined_mask, 1],
                color=colors[label_idx],
                marker=markers[cluster_idx],
                label=f'Class {label_value} (Cluster {cluster_idx})',
                alpha=0.7
            )

    plt.title("Clustering Visualization", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend(title="Legend", loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plot_filename = f"clustering_plot.png"
    plot_filepath = os.path.join("static/plots", plot_filename)
    plt.savefig(plot_filepath, format="png")
    plt.close()

    return f"/static/plots/{plot_filename}"

