import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import os


def run_clustering(representations, labels, num_clusters=None, plots=False):
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
    # Convert Pandas DataFrame to NumPy if necessary
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)  # Ensure labels are 1D

    # Ensure labels exist and have correct shape
    labels = np.asarray(labels) 
    has_labels = labels is not None and len(labels) > 0
    
     # Determine number of clusters
    if num_clusters is None:
        if has_labels:
            num_clusters = len(np.unique(labels))  # Use labels if no num_clusters is given
        else:
            raise ValueError("num_clusters must be provided if labels are missing.")

    if has_labels:
        mask = ~np.isnan(labels)
        labels, representations = labels[mask], representations[mask]
        labels = labels.astype(int)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(representations)

    # Compute NMI only if labels exist and match cluster labels in length
    nmi = None
    if has_labels and labels.shape[0] == cluster_labels.shape[0]:
        nmi = normalized_mutual_info_score(labels, cluster_labels)
    else:
        print("Warning: Skipping NMI because of missing or mismatched labels")

    # Compute evaluation metrics
    silhouette = silhouette_score(representations, cluster_labels)
    davies_bouldin = davies_bouldin_score(representations, cluster_labels)

    # Compute cluster learnability using 1-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(representations, cluster_labels)
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
        plot_url = visualize_clusterings(representations, cluster_labels, labels, num_clusters)

    return {"results": results, "plot_url": plot_url}


def visualize_clusterings(representations, cluster_labels, labels=None, num_clusters=None):
    """
    Visualizes KMeans clustering results, coloring points based on their cluster and labels.

    Parameters:
        representations (ndarray): Feature representations.
        cluster_labels (ndarray): Cluster assignments.
        num_clusters (int): Number of clusters.

    """
    matplotlib.use('Agg')

    plt.figure(figsize=(8, 6))
    markers = ['x', 'o', '+']
    
    # Compute first two principal components (PCA)
    pca = PCA(n_components=2)
    pca_rep = pca.fit_transform(representations)

    # Define color palette based on number of clusters
    colors = sns.color_palette('husl')#, n_colors=num_clusters)

    # Plot each cluster with its unique color
    # for cluster_idx in range(num_clusters):
    #     cluster_mask = cluster_labels == cluster_idx
    print(labels, "labels")
    print(cluster_labels, "cluster_labels")
    plt.scatter(
        x=pca_rep[:, 0],
        y=pca_rep[:, 1],
        # color=colors[cluster_labels] if labels is None else colors[labels],
        # marker=markers[cluster_labels],
        label=f'Cluster',
        alpha=0.4
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

