"""
Circuit collapse metrics computation.

Analyzes how routing patterns change across precision levels.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import entropy


def unique_pattern_count(activations: torch.Tensor) -> int:
    """
    Count the number of unique activation patterns in a batch.
    
    Args:
        activations: Tensor of shape (batch_size, dim) with values {-1, +1}
    
    Returns:
        Number of unique patterns
    """
    batch_size = activations.shape[0]
    patterns = set()
    
    for i in range(batch_size):
        # Convert to tuple for hashing
        pattern = tuple(activations[i].cpu().numpy().astype(np.int8))
        patterns.add(pattern)
    
    return len(patterns)


def routing_entropy(activations: torch.Tensor) -> float:
    """
    Compute Shannon entropy of routing distribution.
    
    Measures disorder in how samples are distributed across different paths.
    
    Args:
        activations: Tensor of shape (batch_size, dim)
    
    Returns:
        Shannon entropy
    """
    batch_size = activations.shape[0]
    
    # Count how many samples follow each unique pattern
    pattern_counts = {}
    for i in range(batch_size):
        pattern = tuple(activations[i].cpu().numpy().astype(np.int8))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Compute probability distribution
    probs = np.array(list(pattern_counts.values())) / batch_size
    
    # Shannon entropy
    return entropy(probs, base=2)


def path_convergence_histogram(activations: torch.Tensor) -> np.ndarray:
    """
    Compute histogram of how many samples follow each routing path.
    
    Returning the counts allows visualization of distribution concentration.
    
    Args:
        activations: Tensor of shape (batch_size, dim)
    
    Returns:
        Array of pattern frequencies
    """
    batch_size = activations.shape[0]
    
    pattern_counts = {}
    for i in range(batch_size):
        pattern = tuple(activations[i].cpu().numpy().astype(np.int8))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    return np.array(sorted(pattern_counts.values(), reverse=True))


def gini_coefficient(pattern_freqs: np.ndarray) -> float:
    """
    Compute Gini coefficient of pattern frequency distribution.
    
    Gini = 0: all patterns equally used (uniform)
    Gini = 1: all samples on one pattern (total collapse)
    
    Args:
        pattern_freqs: Array of pattern frequencies
    
    Returns:
        Gini coefficient in [0, 1]
    """
    n = len(pattern_freqs)
    if n == 0:
        return 0.0
    
    # Sort in ascending order
    sorted_freqs = np.sort(pattern_freqs)
    
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1) / n
    numerator = 2 * np.sum(np.arange(1, n + 1) * sorted_freqs)
    denominator = n * np.sum(sorted_freqs)
    
    gini = (numerator / denominator) - (n + 1) / n
    
    return max(0.0, gini)  # Ensure non-negative


def class_wise_similarity(activations: torch.Tensor, labels: torch.Tensor) -> Dict[int, float]:
    """
    Compute average Jaccard similarity within each class.
    
    Args:
        activations: Tensor of shape (batch_size, dim)
        labels: Class labels of shape (batch_size,)
    
    Returns:
        Dictionary mapping class -> avg within-class similarity
    """
    num_classes = int(labels.max().item()) + 1
    class_sims = {}
    
    for c in range(num_classes):
        class_mask = labels == c
        class_activations = activations[class_mask]
        
        if class_activations.shape[0] < 2:
            class_sims[c] = 1.0  # Only one sample
            continue
        
        # Compute pairwise similarities
        sims = []
        for i in range(class_activations.shape[0]):
            for j in range(i + 1, class_activations.shape[0]):
                a_bin = (class_activations[i] == 1).float()
                b_bin = (class_activations[j] == 1).float()
                intersection = (a_bin * b_bin).sum()
                union = torch.maximum(a_bin, b_bin).sum()
                if union > 0:
                    sims.append((intersection / union).item())
        
        class_sims[c] = np.mean(sims) if sims else 1.0
    
    return class_sims


def sparsity_metric(activations: torch.Tensor) -> float:
    """
    Compute sparsity as fraction of inactive neurons.
    
    Args:
        activations: Tensor with values {-1, +1}
    
    Returns:
        Fraction of inactive neurons [0, 1]
    """
    inactive = (activations == -1).float().mean()
    return inactive.item()


def gate_efficiency(activations: torch.Tensor) -> float:
    """
    Compute how efficiently the gating is used.
    
    Ratio of unique patterns to maximum possible patterns (2^d).
    
    Args:
        activations: Tensor of shape (batch_size, d)
    
    Returns:
        Efficiency in [0, 1]
    """
    batch_size, dim = activations.shape
    unique_patterns = unique_pattern_count(activations)
    max_possible = min(2 ** dim, batch_size)  # Can't have more than batch_size unique
    
    if max_possible == 0:
        return 0.0
    
    return unique_patterns / max_possible


def path_determinism(model, x, num_runs=5):
    """
    Check if the same input produces the same routing path across multiple runs.
    
    Useful for detecting stochasticity introduced by quantization noise.
    
    Args:
        model: BNN model with forward_with_activations
        x: Input tensor
        num_runs: Number of forward passes to check
    
    Returns:
        Fraction of samples with deterministic paths
    """
    paths_all_runs = []
    
    for _ in range(num_runs):
        with torch.no_grad():
            _, activations = model.forward_with_activations(x)
            # Concatenate all layer activations into single path vector
            path = torch.cat([a.view(a.shape[0], -1) for a in activations], dim=1)
            paths_all_runs.append(path)
    
    # Check if all runs produce identical paths
    batch_size = x.shape[0]
    deterministic_count = 0
    
    for i in range(batch_size):
        paths_for_sample = [paths_all_runs[r][i] for r in range(num_runs)]
        
        # Check if all paths are identical
        all_same = all(torch.allclose(paths_for_sample[0], p) for p in paths_for_sample[1:])
        if all_same:
            deterministic_count += 1
    
    return deterministic_count / batch_size


def compute_all_metrics(model, test_loader, layer_idx=None, num_samples=None):
    """
    Compute all routing metrics for a model on test data.
    
    Args:
        model: BNN model
        test_loader: DataLoader with (images, labels)
        layer_idx: If specified, only compute metrics for this layer
        num_samples: If specified, only use first N samples
    
    Returns:
        Dictionary of all metrics
    """
    all_activations = []
    all_labels = []
    
    sample_count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            _, activations = model.forward_with_activations(images)
            
            all_activations.extend(activations)
            all_labels.extend(labels.numpy())
            
            sample_count += images.shape[0]
            if num_samples and sample_count >= num_samples:
                break
    
    # Stack activations per layer
    layers_activations = []
    num_layers = len(all_activations) // len(all_labels)
    
    for layer_idx_computed in range(num_layers):
        layer_acts = [all_activations[i * num_layers + layer_idx_computed] 
                     for i in range(len(all_labels))]
        layers_activations.append(torch.cat(layer_acts, dim=0))
    
    all_labels = np.array(all_labels)
    
    metrics = {}
    
    for l, acts in enumerate(layers_activations):
        prefix = f"layer_{l+1}"
        
        metrics[f"{prefix}_unique_patterns"] = unique_pattern_count(acts)
        metrics[f"{prefix}_entropy"] = routing_entropy(acts)
        metrics[f"{prefix}_sparsity"] = sparsity_metric(acts)
        metrics[f"{prefix}_gate_efficiency"] = gate_efficiency(acts)
        
        pattern_freqs = path_convergence_histogram(acts)
        metrics[f"{prefix}_gini"] = gini_coefficient(pattern_freqs)
        
        class_sims = class_wise_similarity(acts, torch.from_numpy(all_labels))
        for c, sim in class_sims.items():
            metrics[f"{prefix}_class_{c}_similarity"] = sim
    
    return metrics
