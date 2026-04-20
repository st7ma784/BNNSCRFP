"""
Quantization utilities for different precision levels.

Supports quantizing models to different bit widths for circuit collapse analysis.
"""

import torch
import torch.nn as nn
import copy
from typing import Tuple


def quantize_tensor(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Quantize a tensor to specified bit width.
    
    Args:
        x: Input tensor (any shape)
        bits: Number of bits for quantization (1-32)
              1 = binary, 8 = int8, 16 = float16, 32 = float32
    
    Returns:
        Quantized tensor
    """
    if bits == 32:
        return x
    
    if bits == 16:
        # Use float16
        return x.half().float()
    
    if bits == 8:
        # Symmetric int8 quantization
        # Map to [-127, 127]
        scale = 127.0 / x.abs().max() if x.abs().max() > 0 else 1.0
        x_int8 = torch.round(x * scale).clamp(-127, 127)
        x_dequant = x_int8 / scale
        return x_dequant
    
    if bits == 4:
        # Map to [-7, 7]
        scale = 7.0 / x.abs().max() if x.abs().max() > 0 else 1.0
        x_int4 = torch.round(x * scale).clamp(-7, 7)
        x_dequant = x_int4 / scale
        return x_dequant
    
    if bits == 2:
        # Map to [-1, 1]
        scale = 1.0 / x.abs().max() if x.abs().max() > 0 else 1.0
        x_int2 = torch.round(x * scale).clamp(-1, 1)
        x_dequant = x_int2 / scale
        return x_dequant
    
    if bits == 1:
        # Binary quantization (sign function)
        return torch.sign(x)
    
    raise ValueError(f"Unsupported bit width: {bits}")


class QuantizedModel(nn.Module):
    """
    Wrapper around a BNN model that quantizes weights to specified precision.
    """
    
    def __init__(self, model: nn.Module, bits: int = 32):
        """
        Args:
            model: Original BNN model
            bits: Bit width for quantization
        """
        super().__init__()
        self.original_model = copy.deepcopy(model)
        self.bits = bits
        self.quantized_model = copy.deepcopy(model)
        
        # Quantize all weights
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Quantize all weights in the model."""
        for name, param in self.quantized_model.named_parameters():
            if 'weight' in name:
                param.data = quantize_tensor(param.data, self.bits)
    
    def forward(self, x):
        """Forward pass through quantized model."""
        return self.quantized_model(x)
    
    def forward_with_activations(self, x):
        """Forward with activation tracking (for BNN models)."""
        if hasattr(self.quantized_model, 'forward_with_activations'):
            return self.quantized_model.forward_with_activations(x)
        else:
            raise NotImplementedError("Model must have forward_with_activations method")


def get_quantized_model(model: nn.Module, bits: int = 32) -> nn.Module:
    """
    Create a quantized version of a model.
    
    Args:
        model: Original model
        bits: Bit width for quantization
    
    Returns:
        Quantized model
    """
    if bits == 32:
        return copy.deepcopy(model)
    else:
        return QuantizedModel(model, bits)


def quantize_model_inplace(model: nn.Module, bits: int) -> None:
    """
    Quantize a model's weights in-place.
    
    Args:
        model: Model to quantize
        bits: Bit width for quantization
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = quantize_tensor(param.data, bits)
