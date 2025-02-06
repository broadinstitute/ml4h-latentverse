import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

# Create directory for plots if not exists
PLOTS_DIR = "static/plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

### Define MLP Models ###
class SimpleLinear(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLinear, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

### Training Function ###
def train_model(X_train, y_train, X_test, y_test, model, loss_fn, epochs=50):
    """
    Trains a PyTorch model on the given dataset.

    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Testing data
    - model: PyTorch model instance
    - loss_fn: Loss function (MSE or BCEWithLogits)
    - epochs: Number of training iterations

    Returns:
    - Predictions on X_test
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        preds = model(X_test).squeeze()

    return preds

### Run Probing Function ###
def run_probing(representations, labels, train_ratio=0.6):
    """
    Evaluates representation quality by training probes of different complexity.

    Parameters:
        representations (ndarray): Feature representations.
        labels (ndarray): Labels for probing.
        train_ratio (float): Ratio of train to test data.

    Returns:
        dict: Performance metrics and plot URL.
    """
    # Ensure representations are at least 2D
    representations = np.asarray(representations)
    if representations.ndim == 1:
        representations = representations.reshape(-1, 1)

    # Filter out NaNs
    mask = ~np.isnan(labels)
    labels, representations = labels[mask], representations[mask]

    # Train/Test split
    if representations.ndim > 2:
        representations = representations.reshape(representations.shape[0], -1)    
    X_train, X_test, y_train, y_test = train_test_split(representations, labels, train_size=train_ratio, random_state=42)

    # Convert to torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Define models with increasing complexity
    model_configs = {
        "Linear Regression": Ridge(),
        "1-layer MLP": MLP(X_train.shape[1], [32]),
        "5-layer MLP": MLP(X_train.shape[1], [64, 32, 32, 16, 8]),
        "10-layer MLP": MLP(X_train.shape[1], [128] + [64]*4 + [32]*4)
    }

    # Determine if classification (binary) or regression task
    is_categorical = len(np.unique(labels)) <= 2

    # Initialize performance tracking
    metrics = {"Model Complexity": [], "AUROC": [], "Accuracy": [], "R²": []}

    # Train models and collect metrics
    for model_name, model in model_configs.items():
        if isinstance(model, Ridge):  # Traditional Ridge regression
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:  # MLP models
            loss_fn = nn.BCEWithLogitsLoss() if is_categorical else nn.MSELoss()
            preds = train_model(X_train_t, y_train_t, X_test_t, y_test_t, model, loss_fn)
        
            # Ensure preds is a NumPy array
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy().squeeze()

        # Compute metrics
        auroc = roc_auc_score(y_test, preds) if is_categorical else None
        acc = accuracy_score(y_test, preds.round()) if is_categorical else None
        r2 = r2_score(y_test, preds)

        # Store metrics
        metrics["Model Complexity"].append(model_name)
        metrics["AUROC"].append(auroc)
        metrics["Accuracy"].append(acc)
        metrics["R²"].append(r2)

    ### Generate Plot ###
    plot_filename = "probing_complexity.png"
    plot_filepath = os.path.join(PLOTS_DIR, plot_filename)
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=metrics["Model Complexity"], y=metrics["AUROC"], marker="o", label="AUROC")
    sns.lineplot(x=metrics["Model Complexity"], y=metrics["Accuracy"], marker="o", label="Accuracy")
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