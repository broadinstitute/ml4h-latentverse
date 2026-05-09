import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import warnings


def _detect_task_type(labels, max_classes_for_classification=20):
    """
    Detect whether labels represent a classification or regression task.
    
    Heuristics:
    1. If labels are integers and have few unique values relative to sample size → classification
    2. If labels have many unique values or are continuous floats → regression
    
    Parameters:
        labels: Array of label values
        max_classes_for_classification: Maximum unique values to consider as classification
    
    Returns:
        str: 'binary', 'multiclass', or 'regression'
    """
    unique_values = np.unique(labels[~np.isnan(labels)])
    n_unique = len(unique_values)
    n_samples = len(labels)
    
    # Check if values are integer-like (classification indicator)
    is_integer_like = np.allclose(unique_values, np.round(unique_values))
    
    # Ratio of unique values to samples (high ratio suggests continuous)
    uniqueness_ratio = n_unique / n_samples
    
    if n_unique == 2:
        return 'binary'
    elif n_unique <= max_classes_for_classification and is_integer_like and uniqueness_ratio < 0.05:
        # Few unique integer values, likely categorical
        return 'multiclass'
    elif n_unique <= max_classes_for_classification and uniqueness_ratio < 0.01:
        # Very few unique values relative to samples, likely categorical
        return 'multiclass'
    else:
        # Many unique values or continuous, treat as regression
        return 'regression'


def run_probing(representations, labels, n_folds=5, random_state=42):
    """
    Evaluates representation quality by training probes of different complexity
    using k-fold cross-validation to provide mean ± standard deviation metrics.
    
    This test explores the trade-off between model complexity and performance:
    - Simpler models (linear) are more interpretable and tend to generalize better
    - Complex models (deep MLPs) can achieve higher accuracy but may overfit
    
    Supports binary classification, multi-class classification, and regression tasks.
    Task type is automatically detected based on label characteristics.

    Parameters:
        representations (ndarray or DataFrame): Feature representations (latent embeddings).
        labels (ndarray or DataFrame): Labels for probing (target variable).
        n_folds (int): Number of cross-validation folds (default: 5).

    Returns:
        dict: Performance metrics (mean ± std) and plot data with error bars.
    """
    # Convert inputs to numpy arrays
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)

    # Force conversion to numpy float64 arrays to handle PyArrow arrays
    representations = np.array(representations, dtype=np.float64)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    labels = np.array(labels, dtype=np.float64).reshape(-1)

    # Align sample counts
    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    # Remove NaN labels
    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    # Detect task type: binary, multiclass, or regression
    task_type = _detect_task_type(labels)
    is_classification = task_type in ('binary', 'multiclass')
    is_binary = task_type == 'binary'
    n_classes = len(np.unique(labels))

    if is_classification:
        # Encode labels to ensure consecutive integers starting from 0
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    
    print(f"Probing test: {len(labels)} samples, {n_folds}-fold CV, task={task_type}")
    if is_classification:
        print(f"Classes: {n_classes} unique values")
    else:
        print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")

    # Define model configurations with increasing complexity
    # Models range from simple (linear) to complex (deep MLP) to explore the trade-off
    # between interpretability/generalization and predictive power
    if is_classification:
        model_configs = {
            "Linear Model": LogisticRegression(
                max_iter=500, random_state=random_state
            ),
            "1-layer MLP": MLPClassifier(
                hidden_layer_sizes=(32,), max_iter=500, random_state=random_state
            ),
            "5-layer MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500, random_state=random_state
            ),
            "10-layer MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32, 16),  # 10 hidden layers
                max_iter=500, random_state=random_state
            ),
        }
    else:
        model_configs = {
            "Linear Model": Ridge(random_state=random_state),
            "1-layer MLP": MLPRegressor(
                hidden_layer_sizes=(32,), max_iter=500, random_state=random_state
            ),
            "5-layer MLP": MLPRegressor(
                hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=500, random_state=random_state
            ),
            "10-layer MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64, 64, 64, 64, 32, 32, 32, 32, 16),  # 10 hidden layers
                max_iter=500, random_state=random_state
            ),
        }

    # Set up cross-validation strategy and metrics based on task type
    if is_classification:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        if is_binary:
            # Binary classification: AUROC and Accuracy
            metrics = {
                "Model Complexity": [],
                "AUROC": [],
                "AUROC_std": [],
                "Accuracy": [],
                "Accuracy_std": [],
            }
        else:
            # Multi-class classification: Accuracy and F1 (macro)
            metrics = {
                "Model Complexity": [],
                "Accuracy": [],
                "Accuracy_std": [],
                "F1 (macro)": [],
                "F1 (macro)_std": [],
            }
    else:
        # Regression: R²
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        metrics = {"Model Complexity": [], "R²": [], "R²_std": []}

    # Suppress convergence warnings for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        for model_name, model in model_configs.items():
            print(f"  Evaluating {model_name}...")
            metrics["Model Complexity"].append(model_name)

            if is_classification and is_binary:
                # Binary classification metrics
                try:
                    auroc_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="roc_auc", n_jobs=-1
                    )
                    metrics["AUROC"].append(np.mean(auroc_scores))
                    metrics["AUROC_std"].append(np.std(auroc_scores))
                except Exception as e:
                    print(f"    AUROC failed: {e}")
                    metrics["AUROC"].append(None)
                    metrics["AUROC_std"].append(None)

                try:
                    acc_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="accuracy", n_jobs=-1
                    )
                    metrics["Accuracy"].append(np.mean(acc_scores))
                    metrics["Accuracy_std"].append(np.std(acc_scores))
                except Exception as e:
                    print(f"    Accuracy failed: {e}")
                    metrics["Accuracy"].append(None)
                    metrics["Accuracy_std"].append(None)

            elif is_classification and not is_binary:
                # Multi-class classification metrics
                try:
                    acc_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="accuracy", n_jobs=-1
                    )
                    metrics["Accuracy"].append(np.mean(acc_scores))
                    metrics["Accuracy_std"].append(np.std(acc_scores))
                except Exception as e:
                    print(f"    Accuracy failed: {e}")
                    metrics["Accuracy"].append(None)
                    metrics["Accuracy_std"].append(None)

                try:
                    f1_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="f1_macro", n_jobs=-1
                    )
                    metrics["F1 (macro)"].append(np.mean(f1_scores))
                    metrics["F1 (macro)_std"].append(np.std(f1_scores))
                except Exception as e:
                    print(f"    F1 failed: {e}")
                    metrics["F1 (macro)"].append(None)
                    metrics["F1 (macro)_std"].append(None)

            else:
                # Regression metrics
                try:
                    r2_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="r2", n_jobs=-1
                    )
                    metrics["R²"].append(np.mean(r2_scores))
                    metrics["R²_std"].append(np.std(r2_scores))
                except Exception as e:
                    print(f"    R² failed: {e}")
                    metrics["R²"].append(None)
                    metrics["R²_std"].append(None)

    # Build plot data with error bars
    plot_data = {
        "x_label": "Model Complexity (Number of Layers)",
        "y_label": "Performance Metric (mean ± std)",
        "n_folds": n_folds,
        "task_type": task_type,
        "traces": [],
    }

    # Add traces for each metric
    for metric_name in ["AUROC", "Accuracy", "F1 (macro)", "R²"]:
        std_key = f"{metric_name}_std"
        if metric_name in metrics and any(v is not None for v in metrics[metric_name]):
            display_name = "R² Score" if metric_name == "R²" else metric_name
            plot_data["traces"].append({
                "name": display_name,
                "x": metrics["Model Complexity"],
                "y": metrics[metric_name],
                "y_std": metrics.get(std_key, []),
            })

    return {"metrics": metrics, "plot_data": plot_data}


def run_probing_fast(representations, labels, random_state=42):
    """
    Fast probing for robustness testing - uses simple train/test split instead of CV.
    
    This is ~5x faster than full probing:
    - Single train/test split (no CV)
    - Only 2 models (Linear and 5-layer MLP)
    - No parallel jobs (avoids nested parallelism issues)
    
    Used by robustness test to avoid excessive computation when running
    probing at multiple noise levels.
    """
    # Convert inputs to numpy arrays
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)

    representations = np.array(representations, dtype=np.float64)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    labels = np.array(labels, dtype=np.float64).reshape(-1)

    # Align sample counts
    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    # Remove NaN labels
    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    # Detect task type
    task_type = _detect_task_type(labels)
    is_classification = task_type in ('binary', 'multiclass')
    is_binary = task_type == 'binary'

    if is_classification:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=0.3, random_state=random_state,
        stratify=labels if is_classification else None
    )

    # Only 2 models for speed
    if is_classification:
        model_configs = {
            "Linear Model": LogisticRegression(max_iter=300, random_state=random_state),
            "5-layer MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=300, random_state=random_state
            ),
        }
    else:
        model_configs = {
            "Linear Model": Ridge(random_state=random_state),
            "5-layer MLP": MLPRegressor(
                hidden_layer_sizes=(64, 32, 32, 16, 8), max_iter=300, random_state=random_state
            ),
        }

    # Initialize metrics
    if is_classification and is_binary:
        metrics = {"Model Complexity": [], "AUROC": [], "Accuracy": []}
    elif is_classification:
        metrics = {"Model Complexity": [], "Accuracy": []}
    else:
        metrics = {"Model Complexity": [], "R²": []}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for model_name, model in model_configs.items():
            metrics["Model Complexity"].append(model_name)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if is_classification and is_binary:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics["AUROC"].append(roc_auc_score(y_test, y_proba))
                    except:
                        metrics["AUROC"].append(None)
                    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
                elif is_classification:
                    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
                else:
                    metrics["R²"].append(r2_score(y_test, y_pred))
            except Exception as e:
                print(f"Fast probing error for {model_name}: {e}")
                if is_classification and is_binary:
                    metrics["AUROC"].append(None)
                    metrics["Accuracy"].append(None)
                elif is_classification:
                    metrics["Accuracy"].append(None)
                else:
                    metrics["R²"].append(None)

    return {"metrics": metrics, "plot_data": None}
