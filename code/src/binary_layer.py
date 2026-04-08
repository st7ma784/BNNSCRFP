"""
Binary Layer Implementation

Implements BinaryLinear layer with binary weights {-1, +1}
and sign activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLinear(nn.Module):
    """
    Fully connected layer with binary weights {-1, +1}.
    
    Forward pass:
        z = sign(W @ x + b)
    where W ∈ {-1, +1}^{out_features × in_features}
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias term (default: True)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights randomly, will be binarized at forward
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
                or (in_features,) for single input
        
        Returns:
            Binary output tensor of shape (..., out_features) with values in {-1, +1}
        """
        # Binarize weights to {-1, +1}
        w_binary = torch.sign(self.weight)
        
        # Linear transformation
        z = F.linear(x, w_binary, self.bias)
        
        # Binary activation: sign function
        # Handle x=0 case: sign(0) = 0, but we want -1 or +1
        # Use torch.sign which returns 0 for x=0, then map 0->1
        out = torch.sign(z)
        out = torch.where(out == 0, torch.ones_like(out), out)
        
        return out
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
