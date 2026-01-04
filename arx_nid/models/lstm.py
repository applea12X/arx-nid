"""
Bi-directional LSTM model for network intrusion detection.

Implements a sequence model that captures temporal dependencies in network flow data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for network intrusion detection.

    Architecture:
    - Bi-LSTM layers for temporal feature extraction
    - Dropout for regularization
    - Fully connected layers for classification
    - Supports both binary and multi-class classification
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize Bi-LSTM classifier.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            num_classes: Number of output classes (1 for binary, >1 for multiclass)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BiLSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Calculate the size after LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, num_classes)

        # Batch normalization
        self.bn = nn.BatchNorm1d(lstm_output_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, num_classes) for multiclass
            or (batch,) for binary classification
        """
        # LSTM forward pass
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the final hidden state from both directions
        if self.bidirectional:
            # Concatenate final hidden states from forward and backward
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]

        # h_n shape: (batch, lstm_output_size)

        # Apply dropout
        out = self.dropout(h_n)

        # First fully connected layer
        out = self.fc1(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Second fully connected layer
        out = self.fc2(out)

        # For binary classification, squeeze to (batch,)
        if self.num_classes == 1:
            out = out.squeeze(1)

        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Probabilities of shape (batch, num_classes) or (batch,) for binary
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            if self.num_classes == 1:
                # Binary classification - apply sigmoid
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification - apply softmax
                probs = F.softmax(logits, dim=1)

        return probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            threshold: Decision threshold for binary classification

        Returns:
            Predicted labels of shape (batch,)
        """
        probs = self.predict_proba(x)

        if self.num_classes == 1:
            # Binary classification
            predictions = (probs > threshold).long()
        else:
            # Multi-class classification
            predictions = torch.argmax(probs, dim=1)

        return predictions

    def save(self, path: str):
        """
        Save model weights to disk.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_classes': self.num_classes,
                'dropout': self.dropout_p,
                'bidirectional': self.bidirectional,
            }
        }, path)

        print(f"✓ Saved model to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "BiLSTMClassifier":
        """
        Load model weights from disk.

        Args:
            path: Input file path
            device: Device to load model to (defaults to CPU)

        Returns:
            Loaded model instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device)

        # Create model with saved configuration
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        print(f"✓ Loaded model from {path}")
        return model

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """
        Print model summary.
        """
        print("=" * 70)
        print(f"BiLSTM Classifier Summary")
        print("=" * 70)
        print(f"Input size:        {self.input_size}")
        print(f"Hidden size:       {self.hidden_size}")
        print(f"Num layers:        {self.num_layers}")
        print(f"Num classes:       {self.num_classes}")
        print(f"Dropout:           {self.dropout_p}")
        print(f"Bidirectional:     {self.bidirectional}")
        print(f"Total parameters:  {self.get_num_parameters():,}")
        print("=" * 70)
