import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Create directory for plots if not exists
PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

### Run Probing Function ###
def run_probing(representations, label_dict, train_ratio=0.6):
    """
    Evaluates representation quality by training probes of different complexity.

    Parameters:
        representations (ndarray or DataFrame): Feature representations.
        label_dict (dict of {str: ndarray or DataFrame}): Multiple labels for probing.
        train_ratio (float): Ratio of train to test data.

    Returns:
        dict: Performance metrics and plot URLs.
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()

    # Ensure representations are at least 2D
    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    plot_urls = {}

    for label_name, labels in label_dict.items():
        print(f"Processing label: {label_name}")
        
        if isinstance(labels, pd.DataFrame):
            labels = labels.to_numpy().reshape(-1)

        labels = np.asarray(labels).reshape(-1)

        # Ensure labels and representations have the same number of rows
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

        # Apply mask AFTER ensuring same shape
        mask = ~np.isnan(labels)
        labels = labels[mask]
        representations = representations[mask, :]

        # Determine if classification (binary) or regression task
        is_categorical = len(np.unique(labels)) <= 2
        if is_categorical:
            labels = labels.astype(int)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            representations, labels, train_size=train_ratio, random_state=42
        )

        # Define models with increasing complexity
        model_configs = {
            "Linear Regression": Ridge(),
            "1-layer MLP": MLPClassifier(hidden_layer_sizes=(32), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(32), max_iter=500),
            "5-layer MLP": MLPClassifier(hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500),
            "10-layer MLP": MLPClassifier(hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32), max_iter=500),
        }

        # Initialize performance tracking
        metrics = {"Model Complexity": [], "AUROC": [], "Accuracy": [], "R²": []}

        if is_categorical:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        # Train models and collect metrics
        for model_name, model in model_configs.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test).astype(int)

            # Compute metrics
            if is_categorical:
                if hasattr(model, "predict_proba"):
                    auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                else:
                    auroc = None
                acc = accuracy_score(y_test, preds)
                r2 = None  # Not applicable for classification
            else:
                auroc = None  # Not applicable for regression
                acc = None
                r2 = r2_score(y_test, preds)

            # Store metrics
            metrics["Model Complexity"].append(model_name)
            metrics["AUROC"].append(auroc)
            metrics["Accuracy"].append(acc)
            metrics["R²"].append(r2)

        ### Generate Plot ###
        plot_filename = f"probing_complexity_{label_name}.png"
        plot_filepath = os.path.join(PLOTS_DIR, plot_filename)

        plt.figure(figsize=(8, 6))
        if any(metrics["AUROC"]):
            sns.lineplot(x=metrics["Model Complexity"], y=metrics["AUROC"], marker="o", label=f"{label_name} - AUROC")
        if any(metrics["Accuracy"]):
            sns.lineplot(x=metrics["Model Complexity"], y=metrics["Accuracy"], marker="o", label=f"{label_name} - Accuracy")
        if any(metrics["R²"]):
            sns.lineplot(x=metrics["Model Complexity"], y=metrics["R²"], marker="o", label=f"{label_name} - R² Score")

        plt.xlabel("Model Complexity")
        plt.ylabel("Performance Metric")
        plt.title(f"Probing Performance Across Model Complexities for {label_name}")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=30)
        plt.savefig(plot_filepath)
        plt.close()

        plot_urls[label_name] = f"/{plot_filepath}"

    return {"metrics": {}, "plot_urls": plot_urls}
