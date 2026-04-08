"""
Routing Pattern Analysis

Tools to extract and analyze routing patterns from BNNs.
A routing pattern is the sequence of binary activations through the network.
"""

import torch
import numpy as np


def get_activation_pattern(model, x):
    """
    Extract activation patterns from input x.
    
    Args:
        model: BNN model with forward_with_activations method
        x: Input tensor, shape (batch_size, ...)
    
    Returns:
        List of activation tensors, one per binary layer
    """
    with torch.no_grad():
        _, activations = model.forward_with_activations(x)
    
    return activations


def jaccard_similarity(a, b):
    """
    Compute Jaccard similarity between two binary vectors.
    
    J(a, b) = |a ∩ b| / |a ∪ b|
    
    For binary vectors with values {-1, +1}, interpret as:
    - Active neuron: value = +1
    - Inactive neuron: value = -1
    
    Args:
        a: Binary tensor of shape (d,)
        b: Binary tensor of shape (d,)
    
    Returns:
        Scalar Jaccard similarity ∈ [0, 1]
    """
    # Convert to binary {0, 1} for set operations
    a_bin = (a == 1).float()
    b_bin = (b == 1).float()
    
    intersection = (a_bin * b_bin).sum()
    union = (torch.maximum(a_bin, b_bin)).sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    return (intersection / union).item()


def compute_similarity_matrix(activations):
    """
    Compute pairwise Jaccard similarity matrix for a batch of activations.
    
    Args:
        activations: Tensor of shape (batch_size, d) with binary values {-1, +1}
    
    Returns:
        Similarity matrix of shape (batch_size, batch_size)
    """
    batch_size = activations.shape[0]
    sim_matrix = torch.zeros(batch_size, batch_size)
    
    for i in range(batch_size):
        for j in range(batch_size):
            sim_matrix[i, j] = jaccard_similarity(activations[i], activations[j])
    
    return sim_matrix.numpy()


def get_routing_sparsity(activations):
    """
    Compute sparsity of activation patterns.
    
    Sparsity = fraction of neurons that are inactive (-1)
    
    Args:
        activations: List of activation tensors or single tensor
    
    Returns:
        Sparsity value ∈ [0, 1]
    """
    if isinstance(activations, list):
        # Average sparsity across all layers
        sparsities = []
        for act in activations:
            inactive = (act == -1).float().mean()
            sparsities.append(inactive.item())
        return np.mean(sparsities)
    else:
        # Single tensor
        inactive = (activations == -1).float().mean()
        return inactive.item()


def get_routing_diversity(activations):
    """
    Compute diversity of routing patterns in a batch.
    
    Diversity measures how many distinct routing patterns are realized.
    Computed as average pairwise dissimilarity.
    
    Args:
        activations: Tensor of shape (batch_size, d)
    
    Returns:
        Diversity score ∈ [0, 1]
    """
    sim_matrix = compute_similarity_matrix(activations)
    
    # Average dissimilarity (1 - similarity)
    dissimilarities = 1.0 - sim_matrix
    np.fill_diagonal(dissimilarities, 0)  # Exclude diagonal
    
    if dissimilarities.size > 0:
        return dissimilarities.mean()
    else:
        return 0.0


def cluster_by_activation(activations, labels=None):
    """
    Cluster samples by their activation patterns.
    
    Args:
        activations: Tensor of shape (batch_size, d)
        labels: Optional class labels for grouping
    
    Returns:
        Dictionary mapping patterns (as tuples) to sample indices
    """
    batch_size = activations.shape[0]
    pattern_to_indices = {}
    
    for i in range(batch_size):
        pattern = tuple(activations[i].cpu().numpy().astype(int))
        if pattern not in pattern_to_indices:
            pattern_to_indices[pattern] = []
        pattern_to_indices[pattern].append(i)
    
    return pattern_to_indices
