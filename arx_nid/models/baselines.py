"""
Baseline machine learning models for network intrusion detection.

Provides simple, interpretable models for comparison against deep learning approaches.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import joblib
from pathlib import Path


class BaselineModels:
    """
    Wrapper for sklearn baseline models.

    Supports:
    - Logistic Regression
    - Random Forest

    All models work on flattened 3D tensors (batch, time, features) -> (batch, time*features)
    """

    def __init__(self, model_type: str = "logistic", **model_kwargs):
        """
        Initialize baseline model.

        Args:
            model_type: Either 'logistic' or 'random_forest'
            **model_kwargs: Additional arguments for the sklearn model
        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.scaler = StandardScaler()

        if model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **model_kwargs
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=model_kwargs.get("n_estimators", 100),
                max_depth=model_kwargs.get("max_depth", 20),
                random_state=42,
                n_jobs=-1,
                **{k: v for k, v in model_kwargs.items() if k not in ["n_estimators", "max_depth"]}
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'logistic' or 'random_forest'")

    def _flatten_tensor(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten 3D tensor to 2D for sklearn models.

        Args:
            X: Input tensor of shape (batch, time, features)

        Returns:
            Flattened array of shape (batch, time*features)
        """
        if len(X.shape) == 3:
            batch_size = X.shape[0]
            return X.reshape(batch_size, -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModels":
        """
        Train the baseline model.

        Args:
            X: Training data of shape (batch, time, features)
            y: Training labels of shape (batch,)

        Returns:
            self
        """
        X_flat = self._flatten_tensor(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_flat)

        # Train model
        self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input data.

        Args:
            X: Input data of shape (batch, time, features)

        Returns:
            Predicted labels of shape (batch,)
        """
        X_flat = self._flatten_tensor(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.

        Args:
            X: Input data of shape (batch, time, features)

        Returns:
            Class probabilities of shape (batch, n_classes)
        """
        X_flat = self._flatten_tensor(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.

        Args:
            X: Input data of shape (batch, time, features)
            y: True labels of shape (batch,)

        Returns:
            Accuracy score
        """
        X_flat = self._flatten_tensor(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.score(X_scaled, y)

    def save(self, path: str):
        """
        Save model and scaler to disk.

        Args:
            path: Output path (will save both .model and .scaler files)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path.with_suffix(".model.pkl")
        scaler_path = path.with_suffix(".scaler.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"✓ Saved model to {model_path}")
        print(f"✓ Saved scaler to {scaler_path}")

    def load(self, path: str):
        """
        Load model and scaler from disk.

        Args:
            path: Input path (must have corresponding .model and .scaler files)
        """
        path = Path(path)

        model_path = path.with_suffix(".model.pkl")
        scaler_path = path.with_suffix(".scaler.pkl")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print(f"✓ Loaded model from {model_path}")
        print(f"✓ Loaded scaler from {scaler_path}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances (only for Random Forest).

        Returns:
            Feature importance array or None for logistic regression
        """
        if self.model_type == "random_forest":
            return self.model.feature_importances_
        elif self.model_type == "logistic":
            # Return coefficients for logistic regression
            return np.abs(self.model.coef_[0])
        return None

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "model_type": self.model_type,
            "model_params": self.model.get_params(),
        }
