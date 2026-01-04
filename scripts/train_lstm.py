#!/usr/bin/env python3
"""
Train Bi-LSTM model for network intrusion detection.

This script trains a bidirectional LSTM on the preprocessed tensor data
and logs metrics to MLflow.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
import sys

sys.path.append(".")
from arx_nid.models.lstm import BiLSTMClassifier


def load_data(
    tensor_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    device: torch.device = None,
):
    """
    Load tensor data and create train/val/test split.

    Args:
        tensor_path: Path to tensor .npy file
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
        device: Device to load tensors to

    Returns:
        train_loader, val_loader, test_loader, input_size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data from {tensor_path}...")
    X = np.load(tensor_path)
    print(f"Loaded tensor of shape: {X.shape}")

    # Create synthetic labels (replace with real labels in production)
    print("\nWARNING: Using synthetic labels based on tensor statistics.")
    print("Replace with real labels for production use!\n")

    X_mean = X.mean(axis=1)
    threshold = np.percentile(X_mean.flatten(), 70)
    y = (X_mean.mean(axis=1) > threshold).astype(int)

    print(f"Label distribution: {np.bincount(y)}")
    print(f"  Class 0 (benign): {(y == 0).sum()} samples")
    print(f"  Class 1 (attack): {(y == 1).sum()} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    input_size = X.shape[2]  # Number of features

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t), input_size


def create_data_loader(X, y, batch_size: int, shuffle: bool = True):
    """
    Create PyTorch DataLoader.

    Args:
        X: Feature tensor
        y: Label tensor
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: BiLSTM model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device

    Returns:
        Average loss and accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate model.

    Args:
        model: BiLSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device

    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_model(model, X_test, y_test, device):
    """
    Evaluate model and return metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'=' * 70}")
    print("Evaluating Model")
    print(f"{'=' * 70}")

    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # Predictions
        outputs = model(X_test)
        y_proba = torch.sigmoid(outputs).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)
        y_test_np = y_test.cpu().numpy()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test_np, y_pred),
        "precision": precision_score(y_test_np, y_pred, zero_division=0),
        "recall": recall_score(y_test_np, y_pred, zero_division=0),
        "f1_score": f1_score(y_test_np, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test_np, y_proba),
    }

    # Print metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test_np, y_pred, target_names=["Benign", "Attack"]))

    return metrics


def train_lstm_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    input_size: int,
    output_dir: str,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    log_mlflow: bool = True,
):
    """
    Train BiLSTM model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        input_size: Number of input features
        output_dir: Directory to save model
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        patience: Early stopping patience
        log_mlflow: Whether to log to MLflow

    Returns:
        Trained model and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create data loaders
    train_loader = create_data_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, batch_size, shuffle=False)

    # Initialize model
    model = BiLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=1,  # Binary classification
        dropout=dropout,
        bidirectional=True,
    ).to(device)

    model.summary()

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print(f"\n{'=' * 70}")
    print("Training BiLSTM Model")
    print(f"{'=' * 70}")

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            output_path = Path(output_dir) / "best_lstm_model.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model
    model = BiLSTMClassifier.load(str(output_path), device=device)

    # Evaluate on test set
    metrics = evaluate_model(model, X_test, y_test, device)

    # Log to MLflow
    if log_mlflow:
        with mlflow.start_run(run_name="bilstm"):
            # Log parameters
            mlflow.log_params({
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epoch + 1,
                "lr": lr,
                "patience": patience,
            })

            # Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("best_val_loss", best_val_loss)

            # Log model
            mlflow.pytorch.log_model(model, "bilstm_model")

            print(f"\n✓ Logged to MLflow")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM model")
    parser.add_argument(
        "--data",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data file",
    )
    parser.add_argument(
        "--output-dir", default="models/lstm", help="Output directory for models"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation set proportion"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="LSTM hidden size"
    )
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging"
    )

    args = parser.parse_args()

    # Set MLflow experiment
    if not args.no_mlflow:
        mlflow.set_experiment("arx-nid-deep-learning")

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), input_size = load_data(
        args.data, args.test_size, args.val_size
    )

    # Train model
    model, metrics = train_lstm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        input_size=input_size,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        log_mlflow=not args.no_mlflow,
    )

    print(f"\n{'=' * 70}")
    print("Training Summary")
    print(f"{'=' * 70}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print(f"\n✓ Model saved to {args.output_dir}")
    print(f"✓ Training complete!")


if __name__ == "__main__":
    main()
