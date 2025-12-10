"""CNN-BiLSTM model for voice stress detection."""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTM(nn.Module):
    """CNN-BiLSTM model for voice stress and emotion detection.
    
    The model consists of:
    1. A CNN frontend for local feature extraction
    2. A BiLSTM backend for temporal modeling
    3. A classification head
    """
    
    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        """Initialize the CNN-BiLSTM model.
        
        Args:
            input_dim: Input feature dimension (e.g., number of MFCCs)
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # CNN layers
        self.conv = nn.Sequential(
            # Input shape: (batch, 1, input_dim, time_steps)
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(dropout),
        )
        
        # Calculate the output dimensions after CNN
        # This depends on the input dimensions and the CNN architecture
        # We'll compute this dynamically in the forward pass
        self.cnn_output_dim = None
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=128,  # This will be set in the first forward pass
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)
            return_features: If True, also return the features before the final layer
            
        Returns:
            Output logits of shape (batch_size, num_classes)
            If return_features is True, also returns the features before the final layer
        """
        # CNN forward pass
        # Input shape: (batch, 1, n_mels, time_steps)
        batch_size = x.size(0)
        
        # Apply CNN
        x = self.conv(x)  # (batch, channels, height, width)
        
        # Prepare for LSTM
        # We need to reshape the CNN output to (batch, time_steps, features)
        # The exact reshaping depends on the CNN architecture
        if self.cnn_output_dim is None:
            # This is just a placeholder - you'll need to calculate this based on your CNN
            self.cnn_output_dim = x.size(1) * x.size(2)  # channels * height
            
            # Re-initialize LSTM with the correct input size
            self.lstm = self.lstm.to(x.device)
            
        # Reshape for LSTM: (batch, time_steps, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, height)
        x = x.reshape(batch_size, -1, self.cnn_output_dim)  # (batch, time_steps, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_dim * 2)
        
        # Use the last time step's output for classification
        features = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        # Classification head
        logits = self.fc(features)  # (batch, num_classes)
        
        if return_features:
            return logits, features
        return logits


def create_model(
    config: Optional[dict] = None,
    **kwargs,
) -> CNNBiLSTM:
    """Create a CNN-BiLSTM model with the given configuration.
    
    Args:
        config: Model configuration dictionary
        **kwargs: Additional arguments to override config
        
    Returns:
        CNNBiLSTM model
    """
    # Default configuration
    default_config = {
        'input_dim': 40,  # Number of MFCCs
        'num_classes': 7,  # Number of emotion/stress classes
        'hidden_dim': 128,
        'num_lstm_layers': 2,
        'dropout': 0.3,
    }
    
    # Update default config with provided config and kwargs
    if config is not None:
        default_config.update(config)
    default_config.update(kwargs)
    
    # Create and return the model
    return CNNBiLSTM(**default_config)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a test input
    batch_size = 4
    n_mels = 40
    time_steps = 100
    x = torch.randn(batch_size, 1, n_mels, time_steps).to(device)
    
    # Create model
    model = create_model().to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
