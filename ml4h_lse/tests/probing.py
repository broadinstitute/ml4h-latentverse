import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, average_precision_score, 
    f1_score, precision_score, recall_score, log_loss, 
    balanced_accuracy_score, brier_score_loss, mean_squared_error, 
    mean_absolute_error, explained_variance_score, r2_score, 
    max_error, mean_absolute_percentage_error, median_absolute_error
)

def fit_logistic(X_train, X_test, y_train, y_test, verbose=False):
    """
    Train a logistic regression model and evaluate classification metrics.

    Parameters:
    - X_train, X_test: Feature sets for training and testing
    - y_train, y_test: Corresponding labels
    - verbose: If True, prints dataset details and results

    Returns:
    - Dictionary containing classification performance metrics
    """
    if verbose:
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"\nTrain label distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"\nTest label distribution:\n{pd.Series(y_test).value_counts()}")

    clf = LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced", l1_ratio=0.5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "AUPRC": average_precision_score(y_test, y_pred_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Log Loss": log_loss(y_test, y_pred_proba),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Neg Brier Score": -brier_score_loss(y_test, y_pred_proba),
    }

    if verbose:
        print("Logistic Regression Metrics:", metrics)

    return metrics


def fit_linear(X_train, X_test, y_train, y_test, verbose=False):
    """
    Train a Ridge regression model and evaluate regression metrics.

    Parameters:
    - X_train, X_test: Feature sets for training and testing
    - y_train, y_test: Corresponding labels
    - verbose: If True, prints dataset details and results

    Returns:
    - Dictionary containing regression performance metrics
    """
    if verbose:
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"\nTrain label distribution:\n{pd.Series(y_train).value_counts()}")
        print(f"\nTest label distribution:\n{pd.Series(y_test).value_counts()}")

    clf = make_pipeline(StandardScaler(), Ridge(solver="lsqr", max_iter=250000))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "R²": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "Median Absolute Error": median_absolute_error(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred),
        "Max Error": max_error(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
    }

    if verbose:
        print("Linear Regression Metrics:", metrics)

    return metrics


def run_probing(representations, labels, folds=4, train_ratio=0.6, verbose=False):
    """
    Evaluate representations using logistic regression (classification) or Ridge regression (regression).
    Uses k-fold cross-validation to compute mean performance metrics.

    Parameters:
    - representations: (N, D) array of latent features
    - labels: (N,) array of labels (categorical or continuous)
    - folds: Number of cross-validation folds
    - train_ratio: Proportion of data used for training
    - verbose: If True, prints detailed training information

    Returns:
    - scores: Mean performance metrics across folds
    - errors: Standard deviation (uncertainty) of performance metrics
    """
    mask = ~np.isnan(labels)
    labels, representations = labels[mask], representations[mask]

    is_categorical = len(np.unique(labels)) <= 2  # Classification if <= 2 unique labels

    fold_results = []
    for _ in range(folds):
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        train_size = int(len(labels) * train_ratio)
        train_idx, test_idx = indices[:train_size], indices[train_size:]

        X_train, X_test = representations[train_idx], representations[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        metrics = fit_logistic(X_train, X_test, y_train, y_test, verbose) if is_categorical else fit_linear(X_train, X_test, y_train, y_test, verbose)
        fold_results.append(metrics)

    # Compute mean and variance of metrics across folds
    scores = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    errors = {metric: 2 * np.std([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    

    return scores, errors
