#!/usr/bin/env python3
"""
Hyperparameter optimization for BiLSTM model using Optuna.

This script uses Bayesian optimization to find the best hyperparameters
for the BiLSTM network intrusion detection model.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.integration import MLflowCallback
import mlflow
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys

sys.path.append(".")
from arx_nid.models.lstm import BiLSTMClassifier


def load_data(tensor_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and prepare data for hyperparameter search."""
    print(f"Loading data from {tensor_path}...")
    X = np.load(tensor_path)
    print(f"Loaded tensor of shape: {X.shape}")

    # Create synthetic labels
    X_mean = X.mean(axis=1)
    threshold = np.percentile(X_mean.flatten(), 70)
    y = (X_mean.mean(axis=1) > threshold).astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
        X.shape[2],  # input_size
    )


def create_model(trial, input_size):
    """Create model with hyperparameters suggested by Optuna."""
    # Suggest hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = BiLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=1,
        dropout=dropout,
        bidirectional=True,
    )

    return model


def objective(trial, X_train, y_train, X_val, y_val, input_size, device):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of features
        device: Device to train on

    Returns:
        Validation F1 score
    """
    # Create model
    model = create_model(trial, input_size).to(device)

    # Suggest optimizer hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop (limited epochs for hyperparameter search)
    max_epochs = 20
    best_val_f1 = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        y_pred_list = []
        y_true_list = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                predictions = (torch.sigmoid(outputs) > 0.5).float()

                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(y_batch.cpu().numpy())

        # Calculate F1 score
        val_f1 = f1_score(y_true_list, y_pred_list, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        # Report intermediate value for pruning
        trial.report(val_f1, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_f1


def run_optimization(
    X_train,
    y_train,
    X_val,
    y_val,
    input_size: int,
    n_trials: int = 50,
    timeout: int = None,
    output_dir: str = "models/optuna",
    log_mlflow: bool = True,
):
    """
    Run Optuna hyperparameter optimization.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of features
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        output_dir: Output directory
        log_mlflow: Whether to log to MLflow

    Returns:
        Best trial and study
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="arx-nid-hyperparameter-search",
    )

    # MLflow callback
    if log_mlflow:
        mlflow.set_experiment("arx-nid-hyperparameter-search")
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="val_f1_score",
        )
        callbacks = [mlflow_callback]
    else:
        callbacks = None

    print(f"\n{'=' * 70}")
    print("Starting Hyperparameter Optimization")
    print(f"{'=' * 70}")
    print(f"Number of trials: {n_trials}")
    print(f"Timeout: {timeout}s" if timeout else "Timeout: None")

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_size, device),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
    )

    # Print results
    print(f"\n{'=' * 70}")
    print("Optimization Results")
    print(f"{'=' * 70}")
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"\nBest trial:")
    trial = study.best_trial

    print(f"  Value (F1 Score): {trial.value:.4f}")
    print(f"\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save study
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    study_path = output_path / "study.pkl"
    import joblib
    joblib.dump(study, study_path)
    print(f"\n✓ Saved study to {study_path}")

    # Save best parameters
    best_params_path = output_path / "best_params.txt"
    with open(best_params_path, "w") as f:
        f.write(f"Best F1 Score: {trial.value:.4f}\n\n")
        f.write("Best Parameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")

    print(f"✓ Saved best parameters to {best_params_path}")

    # Plot optimization history (if visualization is available)
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.write_html(str(output_path / "optimization_history.html"))

        fig2 = plot_param_importances(study)
        fig2.write_html(str(output_path / "param_importances.html"))

        print(f"✓ Saved visualization plots to {output_path}")
    except ImportError:
        print("Note: Install plotly for visualization plots")

    return trial, study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument(
        "--data",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data file",
    )
    parser.add_argument(
        "--output-dir",
        default="models/optuna",
        help="Output directory",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    args = parser.parse_args()

    # Load data
    X_train, y_train, X_val, y_val, input_size = load_data(args.data, args.test_size)

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Input size: {input_size} features")

    # Run optimization
    best_trial, study = run_optimization(
        X_train,
        y_train,
        X_val,
        y_val,
        input_size,
        n_trials=args.n_trials,
        timeout=args.timeout,
        output_dir=args.output_dir,
        log_mlflow=not args.no_mlflow,
    )

    print(f"\n✓ Hyperparameter search complete!")
    print(f"\nUse these parameters to train your final model:")
    print(f"python scripts/train_lstm.py \\")
    for key, value in best_trial.params.items():
        print(f"  --{key.replace('_', '-')} {value} \\")
    print(f"  --epochs 100")


if __name__ == "__main__":
    main()
