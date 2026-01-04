#!/usr/bin/env python3
"""
Train baseline models for network intrusion detection.

This script trains Logistic Regression and Random Forest models on the
preprocessed tensor data and logs metrics to MLflow.
"""

import argparse
import numpy as np
import mlflow
import mlflow.sklearn
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
import sys

sys.path.append(".")
from arx_nid.models.baselines import BaselineModels


def load_data(tensor_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load tensor data and create train/test split.

    For this initial version, we'll create synthetic labels based on
    tensor statistics (replace with real labels when available).

    Args:
        tensor_path: Path to tensor .npy file
        test_size: Proportion of data for testing
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Loading data from {tensor_path}...")
    X = np.load(tensor_path)
    print(f"Loaded tensor of shape: {X.shape}")

    # Create synthetic labels for demonstration
    # In production, load real labels from metadata or separate file
    # For now, we'll use a simple heuristic: high byte counts = attack (1)
    # This is just for testing the pipeline!
    print("\nWARNING: Using synthetic labels based on tensor statistics.")
    print("Replace with real labels for production use!\n")

    # Calculate mean feature values across time for each sample
    X_mean = X.mean(axis=1)  # (batch, features)

    # Simple heuristic: samples with high mean values are "attacks"
    # This is just for demonstration!
    threshold = np.percentile(X_mean.flatten(), 70)
    y = (X_mean.mean(axis=1) > threshold).astype(int)

    print(f"Label distribution: {np.bincount(y)}")
    print(f"  Class 0 (benign): {(y == 0).sum()} samples")
    print(f"  Class 1 (attack): {(y == 1).sum()} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate model and return metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 70}")

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Print metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))

    return metrics


def train_baseline_model(
    model_type: str,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir: str,
    log_mlflow: bool = True,
    **model_kwargs,
):
    """
    Train and evaluate a baseline model.

    Args:
        model_type: 'logistic' or 'random_forest'
        X_train, X_test, y_train, y_test: Data splits
        output_dir: Directory to save model
        log_mlflow: Whether to log to MLflow
        **model_kwargs: Additional model parameters

    Returns:
        Trained model and metrics
    """
    print(f"\n{'=' * 70}")
    print(f"Training {model_type.upper()} model")
    print(f"{'=' * 70}")

    # Initialize model
    model = BaselineModels(model_type=model_type, **model_kwargs)

    # Train model
    print(f"\nTraining on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train)
    print("✓ Training complete")

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, model_type.upper())

    # Save model
    output_path = Path(output_dir) / f"{model_type}_model"
    model.save(str(output_path))

    # Log to MLflow
    if log_mlflow:
        with mlflow.start_run(run_name=f"baseline_{model_type}"):
            # Log parameters
            mlflow.log_params(model.get_params()["model_params"])

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model.model, f"{model_type}_model")

            # Log scaler
            mlflow.log_artifact(f"{output_path}.scaler.pkl", "scaler")

            print(f"\n✓ Logged to MLflow")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for intrusion detection"
    )
    parser.add_argument(
        "--data",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data file",
    )
    parser.add_argument(
        "--output-dir", default="models/baselines", help="Output directory for models"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "random_forest"],
        choices=["logistic", "random_forest"],
        help="Models to train",
    )

    # Random Forest specific arguments
    parser.add_argument(
        "--rf-n-estimators", type=int, default=100, help="Number of trees for RF"
    )
    parser.add_argument(
        "--rf-max-depth", type=int, default=20, help="Max depth for RF"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set MLflow experiment
    if not args.no_mlflow:
        mlflow.set_experiment("arx-nid-baselines")

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data, args.test_size)

    # Train models
    results = {}

    if "logistic" in args.models:
        model, metrics = train_baseline_model(
            "logistic",
            X_train,
            X_test,
            y_train,
            y_test,
            args.output_dir,
            log_mlflow=not args.no_mlflow,
        )
        results["logistic"] = {"model": model, "metrics": metrics}

    if "random_forest" in args.models:
        model, metrics = train_baseline_model(
            "random_forest",
            X_train,
            X_test,
            y_train,
            y_test,
            args.output_dir,
            log_mlflow=not args.no_mlflow,
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
        )
        results["random_forest"] = {"model": model, "metrics": metrics}

    # Print summary
    print(f"\n{'=' * 70}")
    print("Training Summary")
    print(f"{'=' * 70}")

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in result["metrics"].items():
            print(f"  {metric_name}: {value:.4f}")

    print(f"\n✓ All models saved to {args.output_dir}")
    print(f"✓ Training complete!")


if __name__ == "__main__":
    main()
