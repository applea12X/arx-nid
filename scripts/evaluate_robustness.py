#!/usr/bin/env python3
"""
Evaluate model robustness against adversarial attacks.

This script evaluates trained models against various adversarial attacks
(FGSM, PGD) and generates comprehensive robustness reports.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import sys

sys.path.append(".")
from arx_nid.models.lstm import BiLSTMClassifier
from arx_nid.security.art_wrapper import ARTModelWrapper
from arx_nid.security.attacks import AdversarialAttacks


def load_data(tensor_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load test data for robustness evaluation."""
    print(f"Loading data from {tensor_path}...")
    X = np.load(tensor_path)

    # Create synthetic labels
    X_mean = X.mean(axis=1)
    threshold = np.percentile(X_mean.flatten(), 70)
    y = (X_mean.mean(axis=1) > threshold).astype(int)

    # Train/test split (we only need test set)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert to one-hot
    y_test_onehot = np.zeros((len(y_test), 2))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input shape: {X_test.shape}")

    return X_test, y_test_onehot, X.shape[2]


def evaluate_model_robustness(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_size: int,
    output_dir: str = "reports/robustness",
):
    """
    Comprehensive robustness evaluation.

    Args:
        model_path: Path to trained model
        X_test: Test data
        y_test: Test labels (one-hot)
        input_size: Number of features
        output_dir: Output directory for reports

    Returns:
        Dictionary of robustness metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = BiLSTMClassifier.load(model_path, device=device)
    model.eval()

    # Create ART wrapper
    input_shape = (X_test.shape[1], X_test.shape[2])
    art_wrapper = ARTModelWrapper(
        model=model,
        input_shape=input_shape,
        nb_classes=2,
        device=device,
        clip_values=(X_test.min(), X_test.max()),
    )

    # Create attacker
    attacker = AdversarialAttacks(art_wrapper.get_classifier())

    # Evaluate clean accuracy
    print(f"\n{'=' * 70}")
    print("Evaluating Clean Accuracy")
    print(f"{'=' * 70}")

    accuracy_clean = art_wrapper.evaluate(X_test, y_test)
    print(f"Clean accuracy: {accuracy_clean:.4f}")

    # Define attack suite
    attack_configs = {
        "fgsm_eps_0.01": {"eps": 0.01},
        "fgsm_eps_0.05": {"eps": 0.05},
        "fgsm_eps_0.10": {"eps": 0.10},
        "fgsm_eps_0.20": {"eps": 0.20},
        "pgd_eps_0.01": {"eps": 0.01, "eps_step": 0.001, "max_iter": 40},
        "pgd_eps_0.05": {"eps": 0.05, "eps_step": 0.005, "max_iter": 40},
        "pgd_eps_0.10": {"eps": 0.10, "eps_step": 0.010, "max_iter": 40},
        "pgd_eps_0.20": {"eps": 0.20, "eps_step": 0.020, "max_iter": 40},
    }

    # Run attack suite
    print(f"\n{'=' * 70}")
    print("Running Attack Suite")
    print(f"{'=' * 70}")

    results = []

    for attack_name, config in attack_configs.items():
        print(f"\nAttack: {attack_name}")
        print(f"Config: {config}")

        # Generate adversarial examples
        if attack_name.startswith("fgsm"):
            X_adv = attacker.fgsm(X_test, **config)
        elif attack_name.startswith("pgd"):
            X_adv = attacker.pgd(X_test, **config)

        # Evaluate
        metrics = attacker.evaluate_attack(X_test, X_adv, y_test)
        metrics["attack_name"] = attack_name
        metrics["attack_type"] = attack_name.split("_")[0].upper()
        metrics["epsilon"] = config.get("eps", 0.0)

        # Print results
        print(f"  Accuracy drop: {metrics['accuracy_drop']:.4f}")
        print(f"  Adversarial accuracy: {metrics['accuracy_adversarial']:.4f}")
        print(f"  Attack success rate: {metrics['attack_success_rate']:.4f}")
        print(f"  L2 perturbation: {metrics['perturbation_l2']:.4f}")
        print(f"  L∞ perturbation: {metrics['perturbation_linf']:.4f}")

        results.append(metrics)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_path / "robustness_evaluation.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    # Save JSON
    json_path = output_path / "robustness_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {json_path}")

    # Generate plots
    generate_robustness_plots(df_results, output_path)

    # Print summary
    print(f"\n{'=' * 70}")
    print("Robustness Summary")
    print(f"{'=' * 70}")
    print(f"\nClean accuracy: {accuracy_clean:.4f}")
    print(f"\nWorst-case robustness:")
    worst_attack = df_results.loc[df_results['accuracy_adversarial'].idxmin()]
    print(f"  Attack: {worst_attack['attack_name']}")
    print(f"  Adversarial accuracy: {worst_attack['accuracy_adversarial']:.4f}")
    print(f"  Accuracy drop: {worst_attack['accuracy_drop']:.4f}")

    print(f"\nBest robustness:")
    best_attack = df_results.loc[df_results['accuracy_adversarial'].idxmax()]
    print(f"  Attack: {best_attack['attack_name']}")
    print(f"  Adversarial accuracy: {best_attack['accuracy_adversarial']:.4f}")
    print(f"  Accuracy drop: {best_attack['accuracy_drop']:.4f}")

    return {
        "clean_accuracy": accuracy_clean,
        "results": results,
        "summary": {
            "worst_case_accuracy": worst_attack['accuracy_adversarial'],
            "worst_case_drop": worst_attack['accuracy_drop'],
            "best_case_accuracy": best_attack['accuracy_adversarial'],
            "mean_accuracy": df_results['accuracy_adversarial'].mean(),
            "mean_drop": df_results['accuracy_drop'].mean(),
        }
    }


def generate_robustness_plots(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots for robustness evaluation."""
    print(f"\n{'=' * 70}")
    print("Generating Plots")
    print(f"{'=' * 70}")

    # Set style
    sns.set_style("whitegrid")

    # 1. Accuracy vs Epsilon
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for attack_type in df['attack_type'].unique():
        data = df[df['attack_type'] == attack_type].sort_values('epsilon')

        axes[0].plot(
            data['epsilon'],
            data['accuracy_adversarial'],
            marker='o',
            label=attack_type,
            linewidth=2,
        )

        axes[1].plot(
            data['epsilon'],
            data['accuracy_drop'],
            marker='o',
            label=attack_type,
            linewidth=2,
        )

    axes[0].set_xlabel('Epsilon (ε)', fontsize=12)
    axes[0].set_ylabel('Adversarial Accuracy', fontsize=12)
    axes[0].set_title('Robustness vs Attack Strength', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epsilon (ε)', fontsize=12)
    axes[1].set_ylabel('Accuracy Drop', fontsize=12)
    axes[1].set_title('Accuracy Degradation vs Attack Strength', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "robustness_vs_epsilon.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()

    # 2. Attack comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width/2,
        df['accuracy_clean'],
        width,
        label='Clean',
        color='green',
        alpha=0.7,
    )

    ax.bar(
        x + width/2,
        df['accuracy_adversarial'],
        width,
        label='Adversarial',
        color='red',
        alpha=0.7,
    )

    ax.set_xlabel('Attack', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Clean vs Adversarial Accuracy by Attack', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['attack_name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = output_dir / "accuracy_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()

    # 3. Perturbation analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['perturbation_l2'],
        df['accuracy_drop'],
        c=df['epsilon'],
        s=100,
        cmap='viridis',
        alpha=0.6,
    )

    for idx, row in df.iterrows():
        ax.annotate(
            row['attack_type'],
            (row['perturbation_l2'], row['accuracy_drop']),
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel('L2 Perturbation', fontsize=12)
    ax.set_ylabel('Accuracy Drop', fontsize=12)
    ax.set_title('Perturbation Magnitude vs Accuracy Drop', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Epsilon (ε)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "perturbation_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model robustness")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pt file)",
    )
    parser.add_argument(
        "--data",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data file",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/robustness",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )

    args = parser.parse_args()

    # Load data
    X_test, y_test, input_size = load_data(args.data, args.test_size)

    # Evaluate robustness
    results = evaluate_model_robustness(
        args.model,
        X_test,
        y_test,
        input_size,
        args.output_dir,
    )

    print(f"\n✓ Robustness evaluation complete!")
    print(f"  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
