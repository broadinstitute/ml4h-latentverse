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
    max_error, mean_absolute_percentage_error, mean_poisson_deviance, 
    mean_gamma_deviance, median_absolute_error
)

def fit_logistic(self, label_header, X_train, X_test, y_train, y_test, verbose=False):
        if verbose:
            print(f'{label_header} len train {len(X_train)} len test {len(X_test)}')
            print(f'\nTrain:\n{pd.Series(y_train).value_counts()} \n\nTest:\n{pd.Series(y_test).value_counts()}')
        clf = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', l1_ratio=0.5)
        clf.fit(X_train, y_train)
        
        sparsity = np.mean(clf.coef_ == 0) * 100
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
        metrics = {
            'AUROC': roc_auc_score(y_test, y_pred_proba), 
            'AUPRC': average_precision_score(y_test, y_pred_proba),
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Log Loss': log_loss(y_test, y_pred_proba),
            'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
            'Neg Brier Score': -brier_score_loss(y_test, y_pred_proba),
        }
        
        if verbose:
            print(f"Metrics for {label_header} (Logistic Regression):", metrics)
            print(f"Sparsity: {sparsity}")
        return metrics

def fit_linear(self, label_header, X_train, X_test, y_train, y_test, verbose=False):
    if verbose:
        print(f'{label_header} len train {len(X_train)} len test {len(X_test)}')
        print(f'\nTrain:\n{pd.Series(y_train).value_counts()} \n\nTest:\n{pd.Series(y_test).value_counts()}')
    
    clf = make_pipeline(StandardScaler(with_mean=True), Ridge(solver='lsqr', max_iter=250000))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
    'R²': r2_score(y_test, y_pred),
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'Median Absolute Error': median_absolute_error(y_test, y_pred),
    'Explained Variance': explained_variance_score(y_test, y_pred),
    'Max Error': max_error(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
    'Mean Poisson Deviance': mean_poisson_deviance(y_test, y_pred),
    'Mean Gamma Deviance': mean_gamma_deviance(y_test, y_pred)
    }

    if verbose:
        print(f"Metrics for {label_header} (Linear Regression):", metrics)
    return metrics

def probing_test(representations, phenotypes, folds=4, train_ratio=0.6, verbose=False):
    scores = {}
    errors = {}
    
    for label in phenotypes.columns:
        try:
            full_data = phenotypes[phenotypes[label].notna()]
            embeddings = representations[full_data.index]
            all_labels = full_data[label].values

            # check whether label is categorical or continuous
            is_categorical = len(np.unique(all_labels)) <= 2

            s = []
            for _ in range(folds):
                indices = np.arange(len(all_labels))
                np.random.shuffle(indices)
                train_size = int(len(all_labels) * train_ratio)
                train_idx, test_idx = indices[:train_size], indices[train_size:]

                X_train, X_test = embeddings[train_idx], embeddings[test_idx]
                y_train, y_test = all_labels[train_idx], all_labels[test_idx]

                if is_categorical:
                    # logistic regression (classification)
                    metrics = fit_logistic(label, X_train, X_test, y_train, y_test, verbose)
                else:
                    # ridge regression 
                    metrics = fit_linear(label, X_train, X_test, y_train, y_test, verbose)
                
                s.append(metrics)

            scores[label] = {
                metric: np.mean([fold[metric] for fold in s])
                for metric in s[0].keys()
            }
            errors[label] = {
                metric: 2 * np.std([fold[metric] for fold in s])
                for metric in s[0].keys()
            }


        except Exception as e:
            print(f"Error in probing for label '{label}': {e}")
            
    for label, metrics in scores.items():
        print(f"Results for {label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")

    return scores, errors