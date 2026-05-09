import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
import warnings

from latentverse.utils import detect_task_type, random_baseline


def run_probing(representations, labels, n_folds=3, random_state=42):
    """
    Evaluate representation quality by fitting probes of increasing complexity
    with k-fold cross-validation. Reports mean ± std over folds.

    Model lineup is intentionally short (Linear / 1-layer MLP / 5-layer MLP):
    the 10-layer model the test used to include dominated wallclock without
    adding interpretable signal, and the bigger MLPs all use sklearn's
    built-in early stopping so a 500-iter cap is the worst case, not the
    typical case.

    Default `n_folds=3` because we're estimating a mean ± std for display,
    not selecting hyperparameters; 3 folds is a reasonable accuracy/cost
    trade-off and matches the new robustness-probing fast path.

    Supports binary, multiclass, and regression labels — task type is
    detected automatically.
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)

    representations = np.array(representations, dtype=np.float64)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    labels = np.array(labels, dtype=np.float64).reshape(-1)

    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    # A probe needs at least 2 classes to be meaningful. Fail loudly here
    # so users see an actionable message instead of MLPs trivially
    # predicting the only class and reporting 1.0 accuracy on a degenerate
    # input.
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "probing requires at least 2 distinct classes; "
            f"got {len(np.unique(labels))} unique label value(s)."
        )

    task_type = detect_task_type(labels)
    is_classification = task_type in ("binary", "multiclass")
    is_binary = task_type == "binary"
    n_classes = len(np.unique(labels))

    if is_classification:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    print(f"Probing test: {len(labels)} samples, {n_folds}-fold CV, task={task_type}")
    if is_classification:
        print(f"Classes: {n_classes} unique values")
    else:
        print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")

    # MLP early-stopping settings. The original (validation_fraction=0.1,
    # n_iter_no_change=10) collapsed the 5-layer MLP to chance on small
    # datasets: with ~400 training samples per fold the val set has only
    # ~40 rows, val-loss is noisy, and patience runs out before the
    # deeper net leaves initialisation. The settings below give the
    # deeper net enough room to converge on small data while still
    # shielding against overfitting on big data.
    _MLP_KW = dict(
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        tol=1e-4,
        random_state=random_state,
    )

    if is_classification:
        model_configs = {
            "Linear Model": LogisticRegression(
                max_iter=500, random_state=random_state
            ),
            "1-layer MLP": MLPClassifier(hidden_layer_sizes=(32,), **_MLP_KW),
            "5-layer MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32, 32, 16, 8), **_MLP_KW
            ),
        }
    else:
        model_configs = {
            "Linear Model": Ridge(random_state=random_state),
            "1-layer MLP": MLPRegressor(hidden_layer_sizes=(32,), **_MLP_KW),
            "5-layer MLP": MLPRegressor(
                hidden_layer_sizes=(64, 32, 32, 16, 8), **_MLP_KW
            ),
        }

    if is_classification:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        if is_binary:
            metrics = {
                "Model Complexity": [],
                "AUROC": [],
                "AUROC_std": [],
                "Accuracy": [],
                "Accuracy_std": [],
            }
        else:
            metrics = {
                "Model Complexity": [],
                "Accuracy": [],
                "Accuracy_std": [],
                "F1 (macro)": [],
                "F1 (macro)_std": [],
            }
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        metrics = {"Model Complexity": [], "R²": [], "R²_std": []}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        for model_name, model in model_configs.items():
            print(f" Evaluating {model_name}...")
            metrics["Model Complexity"].append(model_name)

            if is_classification and is_binary:
                try:
                    auroc_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="roc_auc", n_jobs=-1
                    )
                    metrics["AUROC"].append(np.mean(auroc_scores))
                    metrics["AUROC_std"].append(np.std(auroc_scores))
                except Exception as e:
                    print(f" AUROC failed: {e}")
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
                    print(f" Accuracy failed: {e}")
                    metrics["Accuracy"].append(None)
                    metrics["Accuracy_std"].append(None)

            elif is_classification and not is_binary:
                try:
                    acc_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="accuracy", n_jobs=-1
                    )
                    metrics["Accuracy"].append(np.mean(acc_scores))
                    metrics["Accuracy_std"].append(np.std(acc_scores))
                except Exception as e:
                    print(f" Accuracy failed: {e}")
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
                    print(f" F1 failed: {e}")
                    metrics["F1 (macro)"].append(None)
                    metrics["F1 (macro)_std"].append(None)

            else:
                try:
                    r2_scores = cross_val_score(
                        clone(model), representations, labels,
                        cv=cv, scoring="r2", n_jobs=-1
                    )
                    metrics["R²"].append(np.mean(r2_scores))
                    metrics["R²_std"].append(np.std(r2_scores))
                except Exception as e:
                    print(f" R² failed: {e}")
                    metrics["R²"].append(None)
                    metrics["R²_std"].append(None)

    # Random baselines per metric — fed to the SPA so it can draw a chance
    # reference line that's correct under class imbalance.
    metric_baselines = {}
    if is_classification and is_binary:
        metric_baselines["AUROC"] = random_baseline(labels, "AUROC")
        metric_baselines["Accuracy"] = random_baseline(labels, "Accuracy")
    elif is_classification:
        metric_baselines["Accuracy"] = random_baseline(labels, "Accuracy")
        metric_baselines["F1 (macro)"] = random_baseline(labels, "F1 (macro)")
    else:
        metric_baselines["R²"] = random_baseline(labels, "R²")

    plot_data = {
        "x_label": "Model Complexity (Number of Layers)",
        "y_label": "Performance Metric (mean ± std)",
        "n_folds": n_folds,
        "task_type": task_type,
        "random_baselines": {k: v for k, v in metric_baselines.items() if v is not None},
        "traces": [],
    }

    for metric_name in ["AUROC", "Accuracy", "F1 (macro)", "R²"]:
        std_key = f"{metric_name}_std"
        if metric_name in metrics and any(v is not None for v in metrics[metric_name]):
            display_name = "R² Score" if metric_name == "R²" else metric_name
            plot_data["traces"].append({
                "name": display_name,
                "metric_type": metric_name,
                "random_baseline": metric_baselines.get(metric_name),
                "x": metrics["Model Complexity"],
                "y": metrics[metric_name],
                "y_std": metrics.get(std_key, []),
            })

    metrics["_random_baselines"] = {
        k: v for k, v in metric_baselines.items() if v is not None
    }

    return {"metrics": metrics, "plot_data": plot_data}


def run_probing_fast(representations, labels, random_state=42):
    """
    Fast probing for the robustness sweep — single train/test split, linear
    model only.

    Robustness fits a fresh probe at every noise level, so anything heavier
    than a linear model amplifies noise of its own (initialisation +
    optimisation variance) and made the test 5–10x slower without changing
    the conclusion. The deep-MLP probe stays available via `run_probing`
    for users who explicitly want to inspect probe-complexity-vs-quality.
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy().reshape(-1)

    representations = np.array(representations, dtype=np.float64)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    labels = np.array(labels, dtype=np.float64).reshape(-1)

    if labels.shape[0] != representations.shape[0]:
        min_samples = min(labels.shape[0], representations.shape[0])
        labels = labels[:min_samples]
        representations = representations[:min_samples, :]

    mask = ~np.isnan(labels)
    labels = labels[mask]
    representations = representations[mask, :]

    task_type = detect_task_type(labels)
    is_classification = task_type in ("binary", "multiclass")
    is_binary = task_type == "binary"

    if is_classification:
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=0.3, random_state=random_state,
        stratify=labels if is_classification else None
    )

    if is_classification:
        model_configs = {
            "Linear Model": LogisticRegression(max_iter=300, random_state=random_state),
        }
    else:
        model_configs = {
            "Linear Model": Ridge(random_state=random_state),
        }

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
                    except Exception:
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
