"""
Visualization Tools

Visualization functions for routing patterns and activations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from .routing import compute_similarity_matrix


def plot_activation_heatmap(activations, layer_idx, title=None, save_path=None):
    """
    Plot activation heatmap for a specific layer.
    
    Shows binary activation patterns across a batch of samples.
    Rows = samples, Columns = neurons. Dark = -1, Light = +1
    
    Args:
        activations: List of activation tensors or single tensor
        layer_idx: Which layer to visualize (if list provided)
        title: Optional title for the plot
        save_path: Path to save figure, or None to just show
    """
    if isinstance(activations, list):
        act = activations[layer_idx].cpu().numpy()
        layer_name = f"Layer {layer_idx + 1}"
    else:
        act = activations.cpu().numpy()
        layer_name = "Activation"
    
    # Convert {-1, +1} to {0, 1} for visualization
    act_binary = (act == 1).astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(act_binary, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Samples')
    
    if title is None:
        title = f'Activation Heatmap - {layer_name}'
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Activation (+1)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_similarity_matrix(activations, labels=None, title=None, save_path=None):
    """
    Plot similarity matrix as heatmap.
    
    Shows Jaccard similarity between routing patterns.
    
    Args:
        activations: Batch of activation tensors, shape (batch_size, d)
        labels: Optional class labels for sample ordering
        title: Optional title
        save_path: Path to save figure
    """
    sim_matrix = compute_similarity_matrix(activations)
    
    # Optionally sort by label
    if labels is not None:
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        sort_idx = np.argsort(labels_np)
        sim_matrix = sim_matrix[sort_idx][:, sort_idx]
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    im = ax.imshow(sim_matrix, cmap='hot', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')
    
    if title is None:
        title = "Routing Pattern Similarity (Jaccard)"
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_tsne_routing(activations, labels=None, title=None, save_path=None):
    """
    Plot t-SNE embedding of routing patterns.
    
    Args:
        activations: Batch of activation tensors
        labels: Optional class labels for coloring
        title: Optional title
        save_path: Path to save figure
    """
    # t-SNE embedding
    activations_np = activations.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(activations_np)-1))
    embedding = tsne.fit_transform(activations_np)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels_np, 
                            cmap='tab10', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=50, alpha=0.7, edgecolors='k')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    if title is None:
        title = "t-SNE of Routing Patterns"
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_expert_histogram(activations, labels=None, num_experts=10, title=None, save_path=None):
    """
    Plot expert activation frequency.
    
    Groups neurons into "experts" and shows activation frequency.
    
    Args:
        activations: Batch of activation tensors, shape (batch_size, d)
        labels: Optional class labels
        num_experts: Number of expert groups to create
        title: Optional title
        save_path: Path to save figure
    """
    act_np = activations.cpu().numpy()
    batch_size, num_neurons = act_np.shape
    
    # Group neurons into experts
    neurons_per_expert = max(1, num_neurons // num_experts)
    actual_experts = (num_neurons + neurons_per_expert - 1) // neurons_per_expert
    
    # Count activations per expert
    expert_counts = np.zeros((actual_experts, 10 if labels is not None else 1))
    
    for expert in range(actual_experts):
        start_idx = expert * neurons_per_expert
        end_idx = min(start_idx + neurons_per_expert, num_neurons)
        
        expert_activations = act_np[:, start_idx:end_idx]
        active_count = (expert_activations == 1).sum(axis=1)
        
        if labels is not None:
            labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            for class_id in range(10):
                class_mask = labels_np == class_id
                expert_counts[expert, class_id] = active_count[class_mask].mean()
        else:
            expert_counts[expert, 0] = active_count.mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if labels is not None:
        x = np.arange(actual_experts)
        width = 0.08
        for class_id in range(10):
            ax.bar(x + class_id * width, expert_counts[:, class_id], width, 
                  label=f'Class {class_id}', alpha=0.8)
        ax.set_xlabel('Expert Group')
        ax.set_ylabel('Avg Active Neurons per Sample')
        ax.legend(ncol=5, fontsize=8)
    else:
        ax.bar(range(actual_experts), expert_counts[:, 0], alpha=0.8)
        ax.set_xlabel('Expert Group')
        ax.set_ylabel('Avg Active Neurons')
    
    if title is None:
        title = "Expert Activation Frequency"
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_all_visualizations(model, dataloader, output_dir=None):
    """
    Generate all visualizations for a model on a dataset.
    
    Args:
        model: BNN model
        dataloader: DataLoader with (images, labels) batches
        output_dir: Directory to save plots, or None to just display
    """
    # Collect all activations and labels
    all_activations = []
    all_labels = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, activations = model.forward_with_activations(images)
            all_activations.append(activations)
            all_labels.append(labels)
    
    # Concatenate across batches
    activations_by_layer = []
    for layer_idx in range(len(all_activations[0])):
        layer_acts = torch.cat([acts[layer_idx] for acts in all_activations], dim=0)
        activations_by_layer.append(layer_acts)
    
    all_labels = torch.cat(all_labels, dim=0)
    
    # Plot for each layer
    for layer_idx, layer_activations in enumerate(activations_by_layer):
        save_path = f"{output_dir}/activation_heatmap_layer{layer_idx}.png" if output_dir else None
        plot_activation_heatmap(layer_activations, layer_idx=0, 
                               title=f"Layer {layer_idx + 1} Activations",
                               save_path=save_path)
        
        save_path = f"{output_dir}/similarity_matrix_layer{layer_idx}.png" if output_dir else None
        plot_similarity_matrix(layer_activations, labels=all_labels,
                              title=f"Similarity Matrix - Layer {layer_idx + 1}",
                              save_path=save_path)
        
        save_path = f"{output_dir}/tsne_routing_layer{layer_idx}.png" if output_dir else None
        plot_tsne_routing(layer_activations, labels=all_labels,
                         title=f"t-SNE Routing - Layer {layer_idx + 1}",
                         save_path=save_path)
        
        save_path = f"{output_dir}/expert_histogram_layer{layer_idx}.png" if output_dir else None
        plot_expert_histogram(layer_activations, labels=all_labels,
                             title=f"Expert Activations - Layer {layer_idx + 1}",
                             save_path=save_path)
