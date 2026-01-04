"""
Adversarial attack generators using IBM ART.

Implements FGSM, PGD, and other evasion attacks for network intrusion detection.
"""

import numpy as np
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
)
from typing import Optional, Dict, Any


class AdversarialAttacks:
    """
    Generator for adversarial examples using various attack methods.

    Supports:
    - FGSM (Fast Gradient Sign Method)
    - PGD (Projected Gradient Descent)
    - C&W (Carlini & Wagner L2)
    """

    def __init__(self, art_classifier):
        """
        Initialize adversarial attack generator.

        Args:
            art_classifier: ART classifier (from ARTModelWrapper)
        """
        self.classifier = art_classifier

    def fgsm(
        self,
        x: np.ndarray,
        eps: float = 0.1,
        targeted: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Generate FGSM adversarial examples.

        Fast Gradient Sign Method adds perturbations in the direction
        of the gradient of the loss.

        Args:
            x: Input samples of shape (batch, seq_len, features)
            eps: Maximum perturbation
            targeted: Whether to perform targeted attack
            **kwargs: Additional attack parameters

        Returns:
            Adversarial examples
        """
        attack = FastGradientMethod(
            estimator=self.classifier,
            eps=eps,
            targeted=targeted,
            **kwargs
        )

        x_adv = attack.generate(x)
        return x_adv

    def pgd(
        self,
        x: np.ndarray,
        eps: float = 0.1,
        eps_step: float = 0.01,
        max_iter: int = 40,
        targeted: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Generate PGD adversarial examples.

        Projected Gradient Descent is an iterative version of FGSM
        that applies small perturbations over multiple steps.

        Args:
            x: Input samples of shape (batch, seq_len, features)
            eps: Maximum perturbation
            eps_step: Step size per iteration
            max_iter: Maximum number of iterations
            targeted: Whether to perform targeted attack
            **kwargs: Additional attack parameters

        Returns:
            Adversarial examples
        """
        attack = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            **kwargs
        )

        x_adv = attack.generate(x)
        return x_adv

    def carlini_wagner(
        self,
        x: np.ndarray,
        confidence: float = 0.0,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        targeted: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Generate C&W L2 adversarial examples.

        Carlini & Wagner attack optimizes for minimal L2 perturbation
        while causing misclassification.

        Args:
            x: Input samples of shape (batch, seq_len, features)
            confidence: Confidence of adversarial examples
            max_iter: Maximum number of iterations
            learning_rate: Learning rate for optimization
            targeted: Whether to perform targeted attack
            **kwargs: Additional attack parameters

        Returns:
            Adversarial examples
        """
        attack = CarliniL2Method(
            classifier=self.classifier,
            confidence=confidence,
            max_iter=max_iter,
            learning_rate=learning_rate,
            targeted=targeted,
            **kwargs
        )

        x_adv = attack.generate(x)
        return x_adv

    def evaluate_attack(
        self,
        x_clean: np.ndarray,
        x_adv: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the success of an adversarial attack.

        Args:
            x_clean: Clean input samples
            x_adv: Adversarial examples
            y_true: True labels

        Returns:
            Dictionary with attack metrics
        """
        # Get predictions
        pred_clean = np.argmax(self.classifier.predict(x_clean), axis=1)
        pred_adv = np.argmax(self.classifier.predict(x_adv), axis=1)

        # Convert labels if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        # Calculate metrics
        accuracy_clean = np.mean(pred_clean == y_true)
        accuracy_adv = np.mean(pred_adv == y_true)

        # Attack success rate (samples that changed prediction)
        success_rate = np.mean(pred_clean != pred_adv)

        # Perturbation magnitude
        perturbation_l2 = np.linalg.norm(
            (x_adv - x_clean).reshape(len(x_clean), -1),
            axis=1
        ).mean()

        perturbation_linf = np.abs(x_adv - x_clean).max(axis=(1, 2)).mean()

        metrics = {
            "accuracy_clean": accuracy_clean,
            "accuracy_adversarial": accuracy_adv,
            "accuracy_drop": accuracy_clean - accuracy_adv,
            "attack_success_rate": success_rate,
            "perturbation_l2": perturbation_l2,
            "perturbation_linf": perturbation_linf,
        }

        return metrics

    def generate_attack_suite(
        self,
        x: np.ndarray,
        attack_configs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate adversarial examples using multiple attack methods.

        Args:
            x: Input samples
            attack_configs: Dictionary mapping attack names to config dicts

        Returns:
            Dictionary mapping attack names to adversarial examples
        """
        if attack_configs is None:
            # Default attack configurations
            attack_configs = {
                "fgsm_0.05": {"eps": 0.05},
                "fgsm_0.1": {"eps": 0.1},
                "fgsm_0.2": {"eps": 0.2},
                "pgd_0.05": {"eps": 0.05, "eps_step": 0.005, "max_iter": 40},
                "pgd_0.1": {"eps": 0.1, "eps_step": 0.01, "max_iter": 40},
                "pgd_0.2": {"eps": 0.2, "eps_step": 0.02, "max_iter": 40},
            }

        adversarial_examples = {}

        for attack_name, config in attack_configs.items():
            print(f"Generating {attack_name} adversarial examples...")

            if attack_name.startswith("fgsm"):
                x_adv = self.fgsm(x, **config)
            elif attack_name.startswith("pgd"):
                x_adv = self.pgd(x, **config)
            elif attack_name.startswith("cw"):
                x_adv = self.carlini_wagner(x, **config)
            else:
                raise ValueError(f"Unknown attack: {attack_name}")

            adversarial_examples[attack_name] = x_adv

        return adversarial_examples

    @staticmethod
    def get_perturbation_stats(x_clean: np.ndarray, x_adv: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics about perturbations.

        Args:
            x_clean: Clean samples
            x_adv: Adversarial samples

        Returns:
            Dictionary of perturbation statistics
        """
        diff = x_adv - x_clean

        return {
            "mean_l0": np.mean(np.count_nonzero(diff.reshape(len(diff), -1), axis=1)),
            "mean_l1": np.mean(np.abs(diff).reshape(len(diff), -1).sum(axis=1)),
            "mean_l2": np.mean(np.linalg.norm(diff.reshape(len(diff), -1), axis=1)),
            "mean_linf": np.mean(np.abs(diff).max(axis=(1, 2))),
            "max_l2": np.max(np.linalg.norm(diff.reshape(len(diff), -1), axis=1)),
            "max_linf": np.max(np.abs(diff)),
        }
