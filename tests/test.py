import torch
import numpy as np
import matplotlib.pyplot as plt
from ml4h_latentverse.tests.clustering import run_clustering
from ml4h_latentverse.tests.disentanglement import run_disentanglement
from ml4h_latentverse.tests.expressiveness import run_expressiveness
from ml4h_latentverse.tests.robustness import run_robustness

########### DATA GENERATION ###########
# idea: if var1 & var2 are small, cluster should be good/tight; 
# if var1 & var2 are large, cluster should be bad/overlap
def generate_cluster_data(n_samples=200, var1=1.0, var2=1.0):
    mean1, mean2 = torch.tensor([-3.0, 0.0]), torch.tensor([3.0, 0.0])
    cov1, cov2 = torch.eye(2) * var1, torch.eye(2) * var2

    dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
    dist2 = torch.distributions.MultivariateNormal(mean2, cov2)
    
    x1 = dist1.sample((n_samples,))
    x2 = dist2.sample((n_samples,))

    # labels: 0 for 1st gaussian, 1 for 2nd
    labels = torch.cat([torch.zeros(n_samples), torch.ones(n_samples)])
    data = torch.cat([x1, x2])

    return data, labels

# # good clustering
# data, labels = generate_cluster_data(n_samples=200, var1=0.5, var2=0.5)
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
# plt.title("Clusterability Test Data - low variance")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# # bad clustering
# data, labels = generate_cluster_data(n_samples=200, var1=5.0, var2=5.0)
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
# plt.title("Clusterability Test Data - high variance")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

def generate_disentanglement_data(n_samples=500, mode="fully_disentangled"):
    mean = torch.zeros(4)  # 4D Data

    if mode == "fully_disentangled":
        cov = torch.eye(4)  # No correlation
    
    elif mode == "partially_disentangled":
        cov = torch.eye(4)

    elif mode == "fully_entangled":
        cov = torch.tensor([
            [1.0, 0.8, 0.6, 0.4],
            [0.8, 1.0, 0.8, 0.6],
            [0.6, 0.8, 1.0, 0.8],
            [0.4, 0.6, 0.8, 1.0],
        ])  # Strong correlations

    # Sample data
    dist = torch.distributions.MultivariateNormal(mean, cov)
    data = dist.sample((n_samples,))

    # Label Generation
    if mode == "fully_disentangled":
        label_regress = data[:, 0]
        labels = (data[:, 0] > 0).long()  # Labels depend only on x1

    elif mode == "partially_disentangled":
        # use conditions instead of weighting
        label_regress = data[:, 0] + data[:, 1]
        labels = torch.logical_and(data[:, 0] > 0, data[:, 1] > 0)

    else:  # Fully entangled
        label_regress = data[:, 0] + data[:, 1]
        labels = torch.logical_and(data[:, 0] > 0, data[:, 1] > 0)

    return data, label_regress

def generate_expressiveness_data(n_samples=500, n_features=10, mode="high"):
    """
    Generates synthetic data for expressiveness testing.
    - 'high': Label depends on all features.
    - 'low': Label depends on only one feature.
    """
    torch.manual_seed(42)

    representations = torch.randn(n_samples, n_features)

    if mode == "high":
        # labels depend on all features (linear combination)
        weights = torch.randn(n_features)
        labels = representations @ weights + torch.randn(n_samples) * 0.1

    elif mode == "low":
        # weights = torch.zeros(n_features)
        # weights[0] = 5
        # labels = representations @ weights + torch.randn(n_samples) * 0.1
        labels = torch.randn(n_samples)


    return representations.numpy(), labels.numpy().reshape(-1, 1)

def generate_robustness_data(n_samples=500, n_features=10, mode="sensitive"):
    """
    - 'sensitive': label is 1 if dim[0] is between 0.1 and 0.2.
    - 'less_sensitive': label is 1 if dim[0] is between 0 and 1.
    """
    np.random.seed(42)

    representations = np.random.randn(n_samples, n_features)

    if mode == "sensitive":
        labels = ((representations[:, 0] > 0.1) & (representations[:, 0] < 0.2)).astype(int)
    else:
        labels = ((representations[:, 0] > 0) & (representations[:, 0] < 1)).astype(int)

    return representations, labels



# modes = ["fully_disentangled", "partially_disentangled", "fully_entangled"]
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for i, mode in enumerate(modes):
#     data, labels = generate_disentanglement_data(mode=mode)
    
#     axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
#     axes[i].set_title(f"{mode.replace('_', ' ').title()}")
#     axes[i].set_xlabel("Feature 1")
#     axes[i].set_ylabel("Feature 2")

# plt.tight_layout()
# plt.show()

############ TESTS #############
def test_clusterability():
    print("Running clusterability...\n")

    # High clusterability
    data_high, labels_high = generate_cluster_data(n_samples=500, var1=0.5, var2=0.5)

    # Low clusterability
    data_low, labels_low = generate_cluster_data(n_samples=500, var1=3.0, var2=3.0)

    for dataset, true_labels, name in [(data_high, labels_high, "High Clusterability"), 
                                       (data_low, labels_low, "Low Clusterability")]:
        results = run_clustering(dataset, true_labels, num_clusters=2)

        print(f"=== {name} ===")
        print(f"NMI: {results['results']['Normalized Mutual Information']:.4f}")
        print(f"MIG: {results['results']['Silhouette Score']:.4f}")
        print(f"DBI: {results['results']['Davies-Bouldin Index']:.4f}")
        print(f"Cluster Learnability: {results['results']['Cluster Learnability']:.4f}")
        print("-" * 50)

        assert results['results']['Normalized Mutual Information'] > 0.5 if name == "High Clusterability" else True, "NMI should be higher for high clusterability!"
        assert results['results']['Silhouette Score'] > 0.5 if name == "High Clusterability" else True, "MIG should be higher for high clusterability!"
        assert results['results']['Davies-Bouldin Index'] < 1.0 if name == "High Clusterability" else True, "DBI should be lower for high clusterability!"
        assert results['results']['Cluster Learnability'] > 0.8 if name == "High Clusterability" else True, "Learnability should be higher for high clusterability!"

    print("\nClusterability metrics are behaving as expected!")

def test_disentanglement():
    print("Running disentanglement ...\n")

    data_disentangled, label_disentangled = generate_disentanglement_data(n_samples=500, mode="fully_disentangled")
    data_partial, labels_partial = generate_disentanglement_data(n_samples=500, mode="partially_disentangled")
    data_entangled, labels_entangled = generate_disentanglement_data(n_samples=500, mode="fully_entangled")

    for dataset, true_labels, name in [(data_disentangled, label_disentangled, "Fully Disentangled"), 
                                       (data_partial, labels_partial, "Partially Disentangled"),
                                       (data_entangled, labels_entangled, "Fully Entangled")]:
        results = run_disentanglement(dataset, true_labels)
        
        mig_score = results["MIG"]
        if isinstance(mig_score, (list, np.ndarray)):  # If it's an array, extract the first element
            mig_score = float(mig_score[0])

        print(f"=== {name} ===")
        print(f"DCI Disentanglement: {results['DCI']['Disentanglement']:.4f}")
        print(f"MIG Score: {mig_score:.4f}")
        print(f"Total Correlation (TC): {results['TC']:.4f}")
        print(f"SAP Score: {results['DCI']['Informativeness']:.4f}")
        print("-" * 50)

    #     assert results["DCI"]["Disentanglement"] > 0.7 if name == "Fully Disentangled" else True, "DCI should be high for fully disentangled!"
    #     assert results["MIG"] > 0.5 if name == "Fully Disentangled" else True, "MIG should be high for fully disentangled!"
    #     assert results["TC"] < 0.5 if name == "Fully Disentangled" else True, "Total Correlation should be low for fully disentangled!"
    #     assert results["DCI"]["Informativeness"] > 0.7 if name == "Fully Disentangled" else True, "SAP should be high for fully disentangled!"

    # print("\nDisentanglement metrics are behaving as expected!")

def test_expressiveness():
    print("\n=== Running Expressiveness Tests ===")
    
    data_high, labels_high = generate_expressiveness_data(mode="high")
    data_low, labels_low = generate_expressiveness_data(mode="low") 
    
    for dataset, labels, name in [
        (data_high, labels_high, "High Expressiveness"),
        (data_low, labels_low, "Low Expressiveness"),
    ]:
        results = run_expressiveness(dataset, labels, percent_to_remove_list = [0, 10, 20, 50, 100], plots=False)
        results_for_printing = {
            phenotype: {percent: float(score) for percent, score in scores.items()}
            for phenotype, scores in results["metrics"].items()
        }

        print(f"Results for {name}: {results_for_printing}\n")
        
        
# expect: silhouette should decrease
# dbi should increase
# nmi should decrease (approach 0)
# cl should decrease
def test_robustness():
    data_sensitive, labels_sensitive = generate_robustness_data(mode="sensitive")
    data_less_sensitive, labels_less_sensitive = generate_robustness_data(mode="less_sensitive")
    
    print("\n=== Running Robustness Tests ===")
    for dataset, labels, name in [
        (data_sensitive, labels_sensitive, "Sensitive Data"),
        (data_less_sensitive, labels_less_sensitive, "Less Sensitive Data"),
    ]:
        results = run_robustness(dataset, labels, noise_levels=[0.1, 0.5, 1.0, 1.5, 2.0], metric="clustering", plots=False)
        results_for_printing = {
            metric_name: [round(float(score), 4) for score in scores] 
            for metric_name, scores in results["metrics"].items()
        }


        print(f"Results for {name}: {results_for_printing}\n")


#### RUN TESTS ####
# test_clusterability()    
# test_disentanglement()
# test_expressiveness()
test_robustness()


