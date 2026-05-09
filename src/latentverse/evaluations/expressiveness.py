import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from latentverse.utils import (
    detect_task_type,
    fit_linear,
    fit_logistic,
    random_baseline,
)

# QUICK WIN #3: Joblib parallelization (4-8x speedup on multi-core systems)
try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not installed. Install with: pip install joblib")


# Map task type -> (metric name, primary-score key, fit-and-score callable).
# The fit callable returns the scalar score for that fold.
def _binary_score(X_train, X_test, y_train, y_test, verbose=False):
    return fit_logistic(X_train, X_test, y_train, y_test, verbose)["AUROC"]


def _regression_score(X_train, X_test, y_train, y_test, verbose=False):
    return fit_linear(X_train, X_test, y_train, y_test, verbose)["R²"]


def _multiclass_score(X_train, X_test, y_train, y_test, verbose=False):
    # Multinomial logistic + macro-F1 mirrors how the probing test reports
    # multiclass performance, so the SPA's metric-name handling stays uniform.
    clf = LogisticRegression(max_iter=500, multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return f1_score(y_test, y_pred, average="macro")


_TASK_DISPATCH = {
    "binary": ("AUROC", _binary_score),
    "multiclass": ("F1 (macro)", _multiclass_score),
    "regression": ("R²", _regression_score),
}


def run_expressiveness(
    representations,
    labels,
    folds=4,
    train_ratio=0.6,
    percent_to_remove_list=[0, 5, 10, 20],
    verbose=False,
    plots=True,
    random_state=42,
):
    """
    Evaluate the expressiveness of learned representations by measuring how
    predictive performance degrades as high-variance dimensions are removed.

    Reports two derived numbers per label:
      * Compactness     — average normalised drop across the removal sweep.
      * Intrinsic Dim.  — number of dims removable before performance falls
                          5% below baseline.

    The metric used for "performance" is picked from the labels:
      * binary labels     → AUROC
      * multiclass labels → macro-F1
      * continuous labels → R²

    Each label also gets a `Random Baseline` value derived from the actual
    label distribution (majority-class frequency for Accuracy, 0.5 for AUROC,
    one-permutation macro-F1 for F1, 0 for R²) so the frontend can draw a
    chance reference line that's correct under class imbalance.
    """
    if isinstance(representations, pd.DataFrame):
        representations = representations.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()

    representations = np.array(representations, dtype=np.float64)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    representations = np.nan_to_num(representations)
    representations_raw = representations.copy()
    rep_std = representations_raw.std(axis=0)
    rep_std[rep_std == 0] = 1.0
    representations = (representations_raw - representations_raw.mean(axis=0)) / rep_std

    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    label_names = [f"Label {i + 1}" for i in range(labels.shape[1])]
    label_scores = {
        label: {percent: [] for percent in percent_to_remove_list}
        for label in label_names
    }
    label_metric_types = {}
    label_dim_counts = {}
    label_baselines = {}

    for label_idx, label_name in enumerate(label_names):
        y = labels[:, label_idx]

        mask = ~np.isnan(y)
        y = y[mask]
        X = representations[mask, :]
        X_raw = representations_raw[mask, :]

        task_type = detect_task_type(y)
        metric_type, score_fn = _TASK_DISPATCH[task_type]

        label_metric_types[label_name] = metric_type
        label_dim_counts[label_name] = X.shape[1]
        label_baselines[label_name] = random_baseline(y, metric_type)

        feature_variance = np.var(X_raw, axis=0)
        variance_order = np.argsort(feature_variance)[::-1]

        for percent_to_remove in percent_to_remove_list:
            if percent_to_remove == 0:
                num_dims_to_remove = 0
            else:
                num_dims_to_remove = int((percent_to_remove / 100) * X.shape[1])
                num_dims_to_remove = max(1, num_dims_to_remove)
                num_dims_to_remove = min(num_dims_to_remove, X.shape[1] - 1)

            dims_to_remove = (
                set(variance_order[:num_dims_to_remove]) if num_dims_to_remove > 0 else set()
            )

            def process_fold(fold_idx):
                indices = np.arange(len(y))
                np.random.seed(random_state + fold_idx)
                np.random.shuffle(indices)
                train_size = int(len(y) * train_ratio)
                train_idx, test_idx = indices[:train_size], indices[train_size:]

                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if dims_to_remove:
                    X_train = np.delete(X_train, list(dims_to_remove), axis=1)
                    X_test = np.delete(X_test, list(dims_to_remove), axis=1)

                return score_fn(X_train, X_test, y_train, y_test, verbose)

            if HAS_JOBLIB and folds > 1:
                fold_results = Parallel(n_jobs=-1, backend="threading")(
                    delayed(process_fold)(fold_idx) for fold_idx in range(folds)
                )
                label_scores[label_name][percent_to_remove].extend(fold_results)
            else:
                for fold_idx in range(folds):
                    label_scores[label_name][percent_to_remove].append(
                        process_fold(fold_idx)
                    )

    formatted_results = {}
    raw_curves = {}

    for label_name in label_scores:
        metric_type = label_metric_types.get(label_name, "AUROC or R²")
        total_dims = label_dim_counts.get(label_name, 0)
        baseline_value = label_baselines.get(label_name)

        label_metrics = {}
        scores_by_percent = {
            percent: float(np.mean(scores))
            for percent, scores in label_scores[label_name].items()
        }

        label_metrics["Metric Type"] = metric_type
        if baseline_value is not None:
            label_metrics["Random Baseline"] = round(baseline_value, 4)

        for percent in percent_to_remove_list:
            score = scores_by_percent[percent]
            if percent == 0:
                label_metrics["Baseline (0% removed)"] = round(score, 4)
            else:
                label_metrics[f"{percent}% Removed"] = round(score, 4)

        baseline = scores_by_percent.get(0)
        if baseline is not None and baseline != 0:
            normalized_scores = [
                scores_by_percent[p] / baseline for p in percent_to_remove_list
            ]
            compactness = 1 - np.mean(normalized_scores)
            label_metrics["Compactness"] = round(max(0.0, min(1.0, compactness)), 4)
        else:
            label_metrics["Compactness"] = "N/A"

        if baseline is not None and baseline != 0 and total_dims > 0:
            threshold = baseline * 0.95
            intrinsic_dim = None
            for percent in sorted(percent_to_remove_list):
                if percent == 0:
                    continue
                if scores_by_percent[percent] < threshold:
                    intrinsic_dim = int(round((percent / 100) * total_dims))
                    break

            if intrinsic_dim is None:
                max_removed = int(round((max(percent_to_remove_list) / 100) * total_dims))
                label_metrics["Intrinsic Dimension"] = f">= {max_removed}"
            else:
                label_metrics["Intrinsic Dimension"] = intrinsic_dim
        else:
            label_metrics["Intrinsic Dimension"] = "N/A"

        formatted_results[label_name] = label_metrics
        raw_curves[label_name] = scores_by_percent

    metric_types = set(label_metric_types.values())
    if len(metric_types) == 1:
        only_metric = next(iter(metric_types))
        y_label = f"Predictive Performance ({only_metric})"
    else:
        only_metric = None
        y_label = "Predictive Performance (mixed metrics)"

    plot_data = {
        "x_label": "Percentage of Dimensions Removed (%)",
        "y_label": y_label,
        "metric_type": only_metric,
        "random_baseline": (
            list(label_baselines.values())[0]
            if len(label_baselines) == 1 and list(label_baselines.values())[0] is not None
            else None
        ),
        "traces": [],
    }

    for label_name, metric_data in raw_curves.items():
        plot_data["traces"].append(
            {
                "name": label_name,
                "x": percent_to_remove_list,
                "y": [metric_data[percent] for percent in percent_to_remove_list],
                "metric_type": label_metric_types.get(label_name),
                "random_baseline": label_baselines.get(label_name),
            }
        )

    return {"metrics": formatted_results, "plot_data": plot_data}
