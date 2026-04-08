"""
BNN as Discrete Path Decompositions - Core Module
"""

from .binary_layer import BinaryLinear
from .bnn_model import SimpleBNN
from .routing import get_activation_pattern, jaccard_similarity, compute_similarity_matrix
from .visualization import plot_activation_heatmap, plot_similarity_matrix, plot_expert_histogram

__all__ = [
    'BinaryLinear',
    'SimpleBNN',
    'get_activation_pattern',
    'jaccard_similarity',
    'compute_similarity_matrix',
    'plot_activation_heatmap',
    'plot_similarity_matrix',
    'plot_expert_histogram',
]
