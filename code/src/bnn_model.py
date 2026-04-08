"""
BNN Model Implementation

Implements small BNN models for experiments
"""

import torch
import torch.nn as nn
from .binary_layer import BinaryLinear


class SimpleBNN(nn.Module):
    """
    Simple 3-layer Binary Neural Network for MNIST.
    
    Architecture:
        Input (784) -> BinaryLinear (784 -> 256)
                    -> BinaryLinear (256 -> 256)  
                    -> BinaryLinear (256 -> 10)
                    -> Output (10)
    
    All hidden layers use binary activations {-1, +1}.
    Output layer is not binarized (standard for classification).
    """
    
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Binary layers
        self.layer1 = BinaryLinear(input_dim, hidden_dim)
        self.layer2 = BinaryLinear(hidden_dim, hidden_dim)
        
        # Output layer (non-binary for classification)
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 784) for MNIST
        
        Returns:
            Logits of shape (batch_size, 10)
        """
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Binary layers
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Output layer (not binarized)
        x = self.output(x)
        
        return x
    
    def forward_with_activations(self, x):
        """
        Forward pass that also returns intermediate activation patterns.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (logits, activation_patterns)
            where activation_patterns is a list of binary activation tensors
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        activations = []
        
        x = self.layer1(x)
        activations.append(x.clone())
        
        x = self.layer2(x)
        activations.append(x.clone())
        
        x = self.output(x)
        
        return x, activations


class DeepBNN(nn.Module):
    """
    Deeper Binary Neural Network for larger inputs.
    
    Architecture:
        Input -> [BinaryLinear -> BinaryLinear]* -> Output
    """
    
    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10, num_layers=4):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256] * (num_layers - 1)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Build binary layers
        self.binary_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.binary_layers.append(BinaryLinear(dims[i], dims[i + 1]))
        
        # Output layer
        self.output = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x):
        """Forward pass through all binary layers."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.binary_layers:
            x = layer(x)
        
        x = self.output(x)
        return x
    
    def forward_with_activations(self, x):
        """Forward pass with activation tracking."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        activations = []
        
        for layer in self.binary_layers:
            x = layer(x)
            activations.append(x.clone())
        
        x = self.output(x)
        
        return x, activations
