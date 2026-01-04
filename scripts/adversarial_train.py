#!/usr/bin/env python3
"""
Adversarial training for BiLSTM model using IBM ART.

This script performs adversarial training to improve model robustness
against evasion attacks like FGSM and PGD.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional
import sys

sys.path.append(".")
from arx_nid.models.lstm import BiLSTMClassifier
from arx_nid.security.art_wrapper import ARTModelWrapper
from art.defences.trainer import AdversarialTrainer


def load_data(tensor_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and prepare data for adversarial training."""
    print(f"Loading data from {tensor_path}...")
    X = np.load(tensor_path)
    print(f"Loaded tensor of shape: {X.shape}")

    # Create synthetic labels
    X_mean = X.mean(axis=1)
    threshold = np.percentile(X_mean.flatten(), 70)
    y = (X_mean.mean(axis=1) > threshold).astype(int)

    print(f"Label distribution: {np.bincount(y)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert labels to one-hot for ART
    y_train_onehot = np.zeros((len(y_train), 2))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1

    y_test_onehot = np.zeros((len(y_test), 2))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    return X_train, y_train_onehot, X_test, y_test_onehot, X.shape[2]


def adversarial_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_size: int,
    model_path: Optional[str] = None,
    output_dir: str = "models/adversarial",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    epochs: int = 30,
    batch_size: int = 32,
    attack_eps: float = 0.1,
    attack_ratio: float = 0.5,
    log_mlflow: bool = True,
):
    """
    Perform adversarial training.

    Args:
        X_train, y_train: Training data (one-hot labels)
        X_test, y_test: Test data (one-hot labels)
        input_size: Number of features
        model_path: Path to pre-trained model (optional)
        output_dir: Output directory
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        attack_eps: Epsilon for adversarial attacks
        attack_ratio: Ratio of adversarial examples in each batch
        log_mlflow: Whether to log to MLflow

    Returns:
        Trained model wrapper and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load or create model
    if model_path:
        print(f"Loading pre-trained model from {model_path}...")
        model = BiLSTMClassifier.load(model_path, device=device)
    else:
        print("Creating new BiLSTM model...")
        model = BiLSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=1,
            dropout=dropout,
            bidirectional=True,
        ).to(device)

    model.summary()

    # Create ART wrapper
    input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    art_wrapper = ARTModelWrapper(
        model=model,
        input_shape=input_shape,
        nb_classes=2,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        clip_values=(X_train.min(), X_train.max()),
    )

    print(f"\n{'=' * 70}")
    print("Starting Adversarial Training")
    print(f"{'=' * 70}")
    print(f"Attack epsilon: {attack_eps}")
    print(f"Attack ratio: {attack_ratio}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Manual adversarial training loop
    print("Starting manual adversarial training...")

    # Create attacker
    from arx_nid.security.attacks import AdversarialAttacks

    attacker = AdversarialAttacks(art_wrapper.get_classifier())

    # Training loop with adversarial examples
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(np.argmax(y_train, axis=1)).to(device)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Create batches
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Generate adversarial examples for part of the batch
            num_adv = int(len(X_batch) * attack_ratio)
            if num_adv > 0:
                X_adv = attacker.fgsm(X_batch[:num_adv], eps=attack_eps)
                X_combined = np.concatenate([X_batch[num_adv:], X_adv])
                y_combined = np.concatenate([y_batch[num_adv:], y_batch[:num_adv]])
            else:
                X_combined = X_batch
                y_combined = y_batch

            # Convert to tensors
            X_t = torch.FloatTensor(X_combined).to(device)
            y_t = torch.FloatTensor(np.argmax(y_combined, axis=1)).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_t)
            loss = criterion(outputs, y_t)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == y_t).sum().item()
            total += y_t.size(0)

        avg_loss = total_loss / (len(X_train) // batch_size + 1)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    print(f"\n✓ Adversarial training complete!")

    # Evaluate
    print(f"\n{'=' * 70}")
    print("Evaluating Model")
    print(f"{'=' * 70}")

    accuracy_clean = art_wrapper.evaluate(X_test, y_test)
    print(f"\nAccuracy on clean test set: {accuracy_clean:.4f}")

    # Test robustness against FGSM
    from arx_nid.security.attacks import AdversarialAttacks

    attacker = AdversarialAttacks(art_wrapper.get_classifier())

    print(f"\nGenerating FGSM adversarial examples (eps={attack_eps})...")
    X_test_adv = attacker.fgsm(X_test, eps=attack_eps)

    accuracy_adv = art_wrapper.evaluate(X_test_adv, y_test)
    print(f"Accuracy on adversarial test set: {accuracy_adv:.4f}")

    robustness_improvement = accuracy_adv
    print(f"\nRobustness (accuracy under attack): {robustness_improvement:.4f}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_save_path = output_path / "adversarial_trained_model.pt"
    model.save(str(model_save_path))

    metrics = {
        "accuracy_clean": accuracy_clean,
        "accuracy_adversarial": accuracy_adv,
        "robustness_score": robustness_improvement,
        "accuracy_drop": accuracy_clean - accuracy_adv,
    }

    # Log to MLflow
    if log_mlflow:
        with mlflow.start_run(run_name="adversarial_training"):
            mlflow.log_params({
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "attack_eps": attack_eps,
                "attack_ratio": attack_ratio,
            })

            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, "adversarial_model")

            print(f"\n✓ Logged to MLflow")

    return art_wrapper, metrics


def main():
    parser = argparse.ArgumentParser(description="Adversarial training for BiLSTM")
    parser.add_argument(
        "--data",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data file",
    )
    parser.add_argument(
        "--model-path",
        help="Path to pre-trained model (optional)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/adversarial",
        help="Output directory",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden size",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--attack-eps",
        type=float,
        default=0.1,
        help="Attack epsilon",
    )
    parser.add_argument(
        "--attack-ratio",
        type=float,
        default=0.5,
        help="Ratio of adversarial examples",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )

    args = parser.parse_args()

    # Set MLflow experiment
    if not args.no_mlflow:
        mlflow.set_experiment("arx-nid-adversarial-training")

    # Load data
    X_train, y_train, X_test, y_test, input_size = load_data(
        args.data, args.test_size
    )

    # Adversarial training
    wrapper, metrics = adversarial_training(
        X_train,
        y_train,
        X_test,
        y_test,
        input_size,
        model_path=args.model_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        attack_eps=args.attack_eps,
        attack_ratio=args.attack_ratio,
        log_mlflow=not args.no_mlflow,
    )

    print(f"\n{'=' * 70}")
    print("Training Summary")
    print(f"{'=' * 70}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print(f"\n✓ Model saved to {args.output_dir}")
    print(f"✓ Adversarial training complete!")


if __name__ == "__main__":
    main()
