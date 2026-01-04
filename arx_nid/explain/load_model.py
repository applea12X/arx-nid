"""
Model loading utilities for explainability.

Provides uniform APIs for loading ONNX and PyTorch models
for use with SHAP and Integrated Gradients explanations.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path


class ONNXModel:
    """
    Wrapper for ONNX models that provides a uniform predict interface.

    This class is designed to work with both SHAP (requires 2D input)
    and standard inference (supports 3D input).
    """

    def __init__(self, path="models/model_v1_robust.onnx"):
        """
        Initialize ONNX model.

        Args:
            path: Path to ONNX model file
        """
        self.path = Path(path)
        if not self.path.exists():
            print(f"Warning: Model file {path} not found. Creating placeholder model.")
            self._create_placeholder_model()

        self.sess = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )

        # Get input/output info
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape

    def _create_placeholder_model(self):
        """Create a placeholder ONNX model for testing."""
        import torch
        import torch.nn as nn

        # Create placeholder PyTorch model
        class PlaceholderModel(nn.Module):
            def __init__(self, input_size=34, hidden_size=64):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size * 5, hidden_size)  # 5 time steps
                self.fc2 = nn.Linear(hidden_size, 32)
                self.fc3 = nn.Linear(32, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # x shape: (batch, time, features) -> (batch, time*features)
                x = self.flatten(x)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        # Create and export model
        model = PlaceholderModel()
        dummy_input = torch.randn(1, 5, 34)

        # Ensure models directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(self.path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Created placeholder ONNX model at {self.path}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the ONNX model.

        Args:
            x: Input tensor of shape (batch, time, features) or (batch, time*features)

        Returns:
            Predictions of shape (batch,)
        """
        # Handle both 2D (flattened for SHAP) and 3D inputs
        if x.ndim == 2:
            # Determine time steps and features from flattened input
            batch_size = x.shape[0]
            total_features = x.shape[1]
            
            # Common configurations: 5*34=170, 20*34=680
            if total_features == 170:  # 5 time steps, 34 features
                x = x.reshape(batch_size, 5, 34)
            elif total_features == 680:  # 20 time steps, 34 features  
                x = x.reshape(batch_size, 20, 34)
            elif total_features % 34 == 0:  # Any multiple of 34 features
                time_steps = total_features // 34
                x = x.reshape(batch_size, time_steps, 34)
            else:
                raise ValueError(f"Cannot reshape input with {total_features} features (not a multiple of 34)")
        elif x.ndim == 3:
            # Already in correct shape
            pass
        else:
            raise ValueError(f"Unexpected input dimensions: {x.ndim}D")

        # Run inference
        result = self.sess.run(
            [self.output_name], {self.input_name: x.astype(np.float32)}
        )[0]

        return result.ravel()  # Return 1D array

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for compatibility).

        Args:
            x: Input tensor

        Returns:
            Array of shape (batch, 2) with [prob_benign, prob_attack]
        """
        probs = self.predict(x)
        # Convert single probability to binary classification format
        prob_attack = probs.reshape(-1, 1)
        prob_benign = 1 - prob_attack
        return np.hstack([prob_benign, prob_attack])


def create_pytorch_model_wrapper():
    """
    Create a PyTorch model wrapper for Integrated Gradients.

    This creates a placeholder BiLSTM model if no real model exists.
    """
    import torch
    import torch.nn as nn

    class BiLSTM(nn.Module):
        """
        Bidirectional LSTM model for sequence classification.
        """

        def __init__(self, input_size=34, hidden_size=128, num_layers=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0,
            )

            # Output layer (bidirectional doubles the hidden size)
            self.fc = nn.Linear(hidden_size * 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)

            # Use last time step output
            output = lstm_out[:, -1, :]  # (batch, hidden_size * 2)

            # Final classification
            output = self.fc(output)
            output = self.sigmoid(output)

            return output.squeeze(-1)  # Remove last dimension

    # Create model
    model = BiLSTM(input_size=34, hidden_size=128, num_layers=2)

    # Try to load existing weights, otherwise use random initialization
    model_path = Path("models/model_v1_robust.pt")
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"Loaded existing PyTorch model from {model_path}")
        except Exception as e:
            print(f"Could not load model weights: {e}")
            print("Using randomly initialized model")
    else:
        print("PyTorch model file not found, using randomly initialized model")
        # Save the placeholder model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved placeholder PyTorch model to {model_path}")

    model.eval()
    return model
