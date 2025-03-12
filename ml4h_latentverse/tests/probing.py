import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier

PLOTS_DIR = "static/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_probing(representations, labels, train_ratio=0.6):
    """
    Evaluates representation quality by training probes of different complexity.

    Parameters:
        representations (ndarray or DataFrame): Feature representations.
        labels (ndarray or DataFrame): Labels for probing.
        train_ratio (float): Ratio of train to test data.

    Returns:
        dict: Performance metrics and plot URL.
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)

    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    labels = np.asarray(labels).reshape(-1)

    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    is_categorical = len(np.unique(labels)) <= 2
    
    if is_categorical:
            labels = labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, train_size=train_ratio, random_state=42
    )

    model_configs = {
        "Linear Regression": Ridge(),
        "1-layer MLP": MLPClassifier(hidden_layer_sizes=(32), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(32), max_iter=500),
        "5-layer MLP": MLPClassifier(hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500),
        "10-layer MLP": MLPClassifier(hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32), max_iter=500) if is_categorical else MLPRegressor(hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32), max_iter=500),
    }

    metrics = {"Model Complexity": [], "AUROC": [], "Accuracy": [], "R²": []}
    
    if is_categorical:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
    print("Unique y_train values:", np.unique(y_train))
    print("Unique y_test values:", np.unique(y_test))

    for model_name, model in model_configs.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test).astype(int)
    
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

        metrics["Model Complexity"].append(model_name)
        metrics["AUROC"].append(auroc)
        metrics["Accuracy"].append(acc)
        metrics["R²"].append(r2)

    plot_filename = "probing_complexity.png"
    plot_filepath = os.path.join(PLOTS_DIR, plot_filename)

    plt.figure(figsize=(8, 6))
    if any(metrics["AUROC"]):
        sns.lineplot(x=metrics["Model Complexity"], y=metrics["AUROC"], marker="o", label="AUROC")
    if any(metrics["Accuracy"]):
        sns.lineplot(x=metrics["Model Complexity"], y=metrics["Accuracy"], marker="o", label="Accuracy")
    if any(metrics["R²"]):
        sns.lineplot(x=metrics["Model Complexity"], y=metrics["R²"], marker="o", label="R² Score")

    plt.xlabel("Model Complexity")
    plt.ylabel("Performance Metric")
    plt.title("Probing Performance Across Model Complexities")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=30)
    plt.savefig(plot_filepath)
    plt.close()

    return {"metrics": metrics, "plot_url": f"/{plot_filepath}"}
