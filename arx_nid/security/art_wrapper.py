"""
Wrapper for PyTorch models to work with IBM Adversarial Robustness Toolbox (ART).
"""

import torch
import torch.nn as nn
import numpy as np
from art.estimators.classification import PyTorchClassifier
from typing import Tuple, Optional


class ARTModelWrapper:
    """
    Wrapper to make PyTorch models compatible with IBM ART.

    Handles conversion between PyTorch tensors and NumPy arrays,
    and provides a consistent interface for adversarial attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        nb_classes: int = 2,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        clip_values: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize ART wrapper for PyTorch model.

        Args:
            model: PyTorch model
            input_shape: Shape of a single input (seq_len, features)
            nb_classes: Number of classes (2 for binary)
            loss_fn: Loss function
            optimizer: Optimizer for adversarial training
            device: Device to use
            clip_values: Min and max values for input clipping
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.input_shape = input_shape
        self.nb_classes = nb_classes

        # Default loss function for binary classification
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()

        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create ART classifier
        # Force model to use float32
        model = model.float()
        self.classifier = PyTorchClassifier(
            model=model,
            loss=loss_fn,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            device_type="cpu" if device.type == "cpu" else "gpu",
            preprocessing=(0.0, 1.0),  # Ensure proper scaling
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the wrapped model.

        Args:
            x: Input data of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, nb_classes)
        """
        # Convert to float32 for PyTorch compatibility
        x = x.astype(np.float32)
        return self.classifier.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            x: Input data of shape (batch, seq_len, features)

        Returns:
            Class probabilities of shape (batch, nb_classes)
        """
        # For binary classification, ART returns single probability
        # We need to convert to [p_benign, p_attack]
        predictions = self.classifier.predict(x)

        if self.nb_classes == 2 and predictions.shape[1] == 1:
            # Convert from (batch, 1) to (batch, 2)
            p_attack = predictions[:, 0]
            p_benign = 1 - p_attack
            return np.stack([p_benign, p_attack], axis=1)

        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model.

        Args:
            x: Training data
            y: Training labels
            **kwargs: Additional arguments for training
        """
        # Convert labels to one-hot if needed
        if len(y.shape) == 1:
            y_onehot = np.zeros((len(y), self.nb_classes))
            y_onehot[np.arange(len(y)), y.astype(int)] = 1
            y = y_onehot

        self.classifier.fit(x, y, **kwargs)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.

        Args:
            x: Test data
            y: Test labels

        Returns:
            Accuracy score
        """
        predictions = np.argmax(self.predict(x), axis=1)
        labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y.astype(int)
        return np.mean(predictions == labels)

    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model

    def get_classifier(self) -> PyTorchClassifier:
        """Get the ART classifier."""
        return self.classifier

    def save(self, path: str):
        """
        Save the wrapped model.

        Args:
            path: Output file path
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_shape': self.input_shape,
            'nb_classes': self.nb_classes,
        }, path)
        print(f"✓ Saved wrapped model to {path}")

    @classmethod
    def load(cls, path: str, model_class, device: Optional[torch.device] = None):
        """
        Load a wrapped model.

        Args:
            path: Input file path
            model_class: Class of the model to load
            device: Device to load to

        Returns:
            Loaded ARTModelWrapper instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)

        # Reconstruct model (requires knowing model architecture)
        # This is a simplified version - in practice, you'd need to save/load full config
        model = model_class.load(path, device=device)

        wrapper = cls(
            model=model,
            input_shape=checkpoint['input_shape'],
            nb_classes=checkpoint['nb_classes'],
            device=device,
        )

        print(f"✓ Loaded wrapped model from {path}")
        return wrapper
