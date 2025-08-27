
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression # CHANGED: Add LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier


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

    # CHANGED: Use LogisticRegression for classification and Ridge for regression
    model_configs = {
        "Linear Model": LogisticRegression(max_iter=500) if is_categorical else Ridge(),
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
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_pred_proba)
                acc = accuracy_score(y_test, y_pred)
            else:
                # Fallback for models without predict_proba
                y_pred = model.predict(X_test)
                auroc = None
                acc = accuracy_score(y_test, y_pred)
            r2 = None
        else:
            y_pred = model.predict(X_test)
            auroc = None
            acc = None
            r2 = r2_score(y_test, y_pred)

        metrics["Model Complexity"].append(model_name)
        metrics["AUROC"].append(auroc)
        metrics["Accuracy"].append(acc)
        metrics["R²"].append(r2)

    # CHANGED: Remove all plotting code and return structured data instead
    plot_data = {
        "x_label": "Model Complexity",
        "y_label": "Performance Metric",
        "traces": []
    }
    
    if any(metrics["AUROC"]):
        plot_data["traces"].append({
            "name": "AUROC",
            "x": metrics["Model Complexity"],
            "y": metrics["AUROC"]
        })
    if any(metrics["Accuracy"]):
        plot_data["traces"].append({
            "name": "Accuracy",
            "x": metrics["Model Complexity"],
            "y": metrics["Accuracy"]
        })
    if any(metrics["R²"]):
        plot_data["traces"].append({
            "name": "R² Score",
            "x": metrics["Model Complexity"],
            "y": metrics["R²"]
        })

    return {"metrics": metrics, "plot_data": plot_data}


# Analysis of the Code

 # Incorrect Model Selection: The code uses Ridge() for all labels, even when is_categorical is true. Ridge() is a regression model and is not designed for classification tasks. It lacks a predict_proba method, which will cause the AUROC metric calculation to be skipped. The correct approach is to use a classification model like LogisticRegression.

# Incorrect Data Type Conversion: The line preds = model.predict(X_test).astype(int) incorrectly converts predictions to integers for both regression and classification. This is fine for classification but will lose precision and produce meaningless results for a regression task.