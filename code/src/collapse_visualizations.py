"""
Visualization suite for circuit collapse and SCRFP analysis.

Implements all 10 visualizations from the visualization plan.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from scipy import stats
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


def setup_style():
    """Configure matplotlib style for professional visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 13


# Color scheme for precision levels
PRECISION_COLORS = {
    32: '#1f77b4',  # Blue
    16: '#ff7f0e',  # Orange
    8: '#2ca02c',   # Green
    4: '#d62728',   # Red
    2: '#9467bd',   # Purple
    1: '#8c564b'    # Brown
}

PRECISION_NAMES = {
    32: 'float32',
    16: 'float16',
    8: 'int8',
    4: '4-bit',
    2: '2-bit',
    1: 'binary'
}


def vis_1_routing_diversity_collapse(metrics_dict: Dict, save_path: str = None):
    """
    1.1 Routing Diversity Collapse
    
    Multi-panel line plot showing unique activation patterns and entropy across precisions.
    """
    setup_style()
    
    precisions = sorted(metrics_dict.keys())
    num_layers = max(int(k.split('_')[1]) for k in metrics_dict[precisions[0]].keys()
                     if 'layer_' in k and 'unique' in k)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Unique patterns
    for layer in range(1, num_layers + 1):
        unique_counts = [metrics_dict[p][f'layer_{layer}_unique_patterns'] for p in precisions]
        max_count = max(unique_counts)
        normalized = [u / max_count for u in unique_counts]
        
        ax1.plot(range(len(precisions)), normalized, 'o-', 
                label=f'Layer {layer}', linewidth=2.5, markersize=8)
    
    ax1.set_xticks(range(len(precisions)))
    ax1.set_xticklabels([PRECISION_NAMES[p] for p in precisions], rotation=45)
    ax1.set_ylabel('Normalized Unique Patterns', fontsize=11)
    ax1.set_xlabel('Precision Level', fontsize=11)
    ax1.set_title('1.1: Routing Diversity Collapse - Pattern Count', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy
    for layer in range(1, num_layers + 1):
        entropies = [metrics_dict[p][f'layer_{layer}_entropy'] for p in precisions]
        ax2.plot(range(len(precisions)), entropies, 's-',
                label=f'Layer {layer}', linewidth=2.5, markersize=8)
    
    ax2.set_xticks(range(len(precisions)))
    ax2.set_xticklabels([PRECISION_NAMES[p] for p in precisions], rotation=45)
    ax2.set_ylabel('Shannon Entropy (bits)', fontsize=11)
    ax2.set_xlabel('Precision Level', fontsize=11)
    ax2.set_title('1.1: Routing Diversity Collapse - Entropy', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_2_path_convergence_heatmaps(activations_dict: Dict, save_path: str = None):
    """
    1.2 Path Convergence Heatmap
    
    3D heatmap array showing how distinct paths merge across precision levels.
    """
    setup_style()
    
    precisions = sorted(activations_dict.keys())
    layer_idx = 0  # Analyze first layer
    
    fig, axes = plt.subplots(1, len(precisions), figsize=(18, 4))
    if len(precisions) == 1:
        axes = [axes]
    
    for idx, (p, acts_list) in enumerate(sorted(activations_dict.items())):
        acts = acts_list[layer_idx][:32]  # Use first 32 samples
        
        # Convert to binary {0, 1}
        acts_binary = (acts.cpu().numpy() == 1).astype(float)
        
        im = axes[idx].imshow(acts_binary, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f'{PRECISION_NAMES[p]}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Neurons')
        if idx == 0:
            axes[idx].set_ylabel('Samples')
    
    fig.suptitle('1.2: Path Convergence Heatmaps (Layer 1)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_3_jaccard_cascades(similarity_dict: Dict, labels_dict: Dict, save_path: str = None):
    """
    1.3 Jaccard Similarity Distribution Cascades
    
    Violin plots showing similarity distributions across precision levels.
    """
    setup_style()
    
    precisions = sorted(similarity_dict.keys())
    all_data = []
    all_precision_labels = []
    all_class_labels = []
    
    for p in precisions:
        sim_matrix = similarity_dict[p]
        labels = labels_dict[p]
        
        # Extract off-diagonal elements and label them
        n = sim_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                all_data.append(sim_matrix[i, j])
                all_precision_labels.append(PRECISION_NAMES[p])
                
                # Determine if within-class or between-class
                if labels[i] == labels[j]:
                    all_class_labels.append('Within-class')
                else:
                    all_class_labels.append('Between-class')
    
    df_data = {
        'Similarity': all_data,
        'Precision': all_precision_labels,
        'Type': all_class_labels
    }
    
    import pandas as pd
    df = pd.DataFrame(df_data)
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    sns.violinplot(data=df, x='Precision', y='Similarity', hue='Type', ax=ax, palette=['skyblue', 'salmon'])
    
    ax.set_xlabel('Precision Level', fontsize=11)
    ax.set_ylabel('Jaccard Similarity', fontsize=11)
    ax.set_title('1.3: Jaccard Similarity Cascades', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_4_phase_diagram(metrics_dict: Dict, save_path: str = None):
    """
    1.4 Routing Collapse Phase Diagram
    
    2D scatter plot identifying routing zones (stable, degraded, collapsed).
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    precisions = sorted(metrics_dict.keys())
    num_layers = max(int(k.split('_')[1]) for k in metrics_dict[precisions[0]].keys()
                     if 'layer_' in k and 'gini' in k)
    
    for layer in range(1, num_layers + 1):
        x_vals = np.log2(precisions)  # Log scale
        y_vals = [1 - (metrics_dict[p][f'layer_{layer}_entropy'] / np.log2(min(256, 2**p)))
                 for p in precisions]
        
        ax.scatter(x_vals, y_vals, s=200, label=f'Layer {layer}', 
                  alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add region shading
    ax.axhspan(0.5, 1.0, alpha=0.1, color='green', label='Stable routing')
    ax.axhspan(0.2, 0.5, alpha=0.1, color='yellow', label='Degraded routing')
    ax.axhspan(0.0, 0.2, alpha=0.1, color='red', label='Collapsed routing')
    
    ax.set_xlabel('Precision (log2 bits)', fontsize=11)
    ax.set_ylabel('Routing Disorder (1 - normalized entropy)', fontsize=11)
    ax.set_title('1.4: Routing Collapse Phase Diagram', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.set_xticks(np.log2(precisions))
    ax.set_xticklabels([PRECISION_NAMES[p] for p in precisions])
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_5_sparsity_trade_off(metrics_dict: Dict, save_path: str = None):
    """
    2.1 Sparsity vs Precision Trade-off
    
    Two-panel plot: activation sparsity and gating efficiency.
    """
    setup_style()
    
    precisions = sorted(metrics_dict.keys())
    num_layers = max(int(k.split('_')[1]) for k in metrics_dict[precisions[0]].keys()
                     if 'layer_' in k and 'sparsity' in k)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Sparsity
    for layer in range(1, num_layers + 1):
        sparsities = [metrics_dict[p][f'layer_{layer}_sparsity'] for p in precisions]
        ax1.plot(range(len(precisions)), sparsities, 'o-',
                label=f'Layer {layer}', linewidth=2.5, markersize=8)
    
    ax1.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='Target SCRFP (20%)')
    
    ax1.set_xticks(range(len(precisions)))
    ax1.set_xticklabels([PRECISION_NAMES[p] for p in precisions], rotation=45)
    ax1.set_ylabel('Neuron Sparsity (inactive fraction)', fontsize=11)
    ax1.set_xlabel('Precision Level', fontsize=11)
    ax1.set_title('2.1a: Activation Sparsity', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right: Gating efficiency
    for layer in range(1, num_layers + 1):
        efficiencies = [metrics_dict[p][f'layer_{layer}_gate_efficiency'] for p in precisions]
        ax2.plot(range(len(precisions)), efficiencies, 's-',
                label=f'Layer {layer}', linewidth=2.5, markersize=8)
    
    ax2.set_xticks(range(len(precisions)))
    ax2.set_xticklabels([PRECISION_NAMES[p] for p in precisions], rotation=45)
    ax2.set_ylabel('Gate Efficiency', fontsize=11)
    ax2.set_xlabel('Precision Level', fontsize=11)
    ax2.set_title('2.1b: Gating Efficiency', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_6_gate_specialization(specialization_dict: Dict, save_path: str = None):
    """
    2.2 Gate Specialization Degradation
    
    Confusion matrices showing neuron-class specialization across precisions.
    """
    setup_style()
    
    precisions = sorted(specialization_dict.keys())
    
    fig, axes = plt.subplots(1, len(precisions), figsize=(15, 4))
    if len(precisions) == 1:
        axes = [axes]
    
    for idx, (p, spec_matrix) in enumerate(sorted(specialization_dict.items())):
        im = axes[idx].imshow(spec_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        axes[idx].set_title(f'{PRECISION_NAMES[p]}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Class')
        if idx == 0:
            axes[idx].set_ylabel('Neuron Subset')
        plt.colorbar(im, ax=axes[idx], label='Specialization')
    
    fig.suptitle('2.2: Gate Specialization Degradation', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_7_path_stability(stability_dict: Dict, save_path: str = None):
    """
    2.3 Fixed Path Stability (SCRFP Compliance)
    
    Heatmap showing path determinism across precisions.
    """
    setup_style()
    
    precisions = sorted(stability_dict.keys())
    determinism = [stability_dict[p]['determinism'] for p in precisions]
    stochasticity = [1 - d for d in determinism]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(precisions))
    width = 0.6
    
    bars1 = ax.bar(x, determinism, width, label='Deterministic Paths', color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x, stochasticity, width, bottom=determinism, label='Stochastic Paths', color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Fraction of Samples', fontsize=11)
    ax.set_xlabel('Precision Level', fontsize=11)
    ax.set_title('2.3: Fixed Path Stability (SCRFP Compliance)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([PRECISION_NAMES[p] for p in precisions])
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_y() + height/2.,
                       f'{height:.2f}', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_8_efficiency_waterfall(waterfall_data: Dict, save_path: str = None):
    """
    2.4 Router Efficiency: Bits per Expertly-Routed Sample
    
    Waterfall diagram showing information loss across quantization stages.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    categories = ['Full Precision', 'Quantization\nLoss', 'Path\nDuplication', 
                  'Routing\nCollapse', 'Final\nCapacity']
    values = waterfall_data.get('values', [100, -20, -15, -30, 35])
    
    cumulative = np.array([0] + list(np.cumsum(values)[:-1]))
    colors = ['#2ca02c', '#d62728', '#d62728', '#d62728', '#ff7f0e']
    
    for i, (cat, val) in enumerate(zip(categories, values)):
        if val < 0:
            ax.bar(i, val, bottom=cumulative[i], color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)
        else:
            ax.bar(i, val, bottom=0 if i == 0 else cumulative[i], color=colors[i], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        label = f'{abs(val)}'
        ax.text(i, cumulative[i] + val/2, label, ha='center', va='center', 
               fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Information / Capacity (bits)', fontsize=11)
    ax.set_title('2.4: Router Efficiency Waterfall', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def vis_9_capability_pyramid(metrics_dict: Dict, save_path: str = None):
    """
    3.1 The "Routing Capability Pyramid"
    
    Stacked visualization showing effective routing capacity across precisions.
    """
    setup_style()
    
    precisions = sorted(metrics_dict.keys())
    num_layers = max(int(k.split('_')[1]) for k in metrics_dict[precisions[0]].keys()
                     if 'layer_' in k and 'gate_efficiency' in k)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(precisions))
    bottom = np.zeros(len(precisions))
    
    colors_gradient = plt.cm.RdYlGn(np.linspace(0, 1, num_layers))
    
    for layer in range(1, num_layers + 1):
        efficiencies = [metrics_dict[p][f'layer_{layer}_gate_efficiency'] for p in precisions]
        ax.bar(x_pos, efficiencies, bottom=bottom, label=f'Layer {layer}',
              color=colors_gradient[layer-1], alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += np.array(efficiencies)
    
    ax.set_ylabel('Normalized Routing Capacity', fontsize=11)
    ax.set_xlabel('Precision Level', fontsize=11)
    ax.set_title('3.1: Routing Capability Pyramid', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([PRECISION_NAMES[p] for p in precisions])
    ax.set_ylim([0, num_layers * 1.1])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_animation_frames(activations_dict: Dict, metrics_dict: Dict,
                           similarity_dict: Dict, output_dir: str):
    """
    3.2 Interactive Precision Slider Animation
    
    Generate individual frames for animated sequence (gif/video).
    """
    setup_style()
    
    precisions = sorted(activations_dict.keys())
    
    for p in precisions:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Frame 1: Layer 1 heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        acts = activations_dict[p][0][:32]
        acts_binary = (acts.cpu().numpy() == 1).astype(float)
        im1 = ax1.imshow(acts_binary, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('Layer 1 Activations', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Neurons')
        ax1.set_ylabel('Samples')
        plt.colorbar(im1, ax=ax1)
        
        # Frame 2: Layer 2 heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        acts = activations_dict[p][1][:32]
        acts_binary = (acts.cpu().numpy() == 1).astype(float)
        im2 = ax2.imshow(acts_binary, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Layer 2 Activations', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Neurons')
        ax2.set_ylabel('Samples')
        plt.colorbar(im2, ax=ax2)
        
        # Frame 3: Similarity matrix
        ax3 = fig.add_subplot(gs[1, 0])
        sim = similarity_dict[p][:32, :32]
        im3 = ax3.imshow(sim, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Jaccard Similarity Matrix', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Sample')
        plt.colorbar(im3, ax=ax3)
        
        # Frame 4: Capacity gauge
        ax4 = fig.add_subplot(gs[1, 1])
        capacity = metrics_dict[p]['layer_1_gate_efficiency']
        ax4.barh([0], [capacity], color='#2ca02c', alpha=0.8, height=0.3)
        ax4.barh([0], [1 - capacity], left=[capacity], color='#d62728', alpha=0.8, height=0.3)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([-0.5, 0.5])
        ax4.set_xlabel('Routing Capacity', fontsize=11)
        ax4.set_title(f'Gate Efficiency: {capacity:.2%}', fontsize=11, fontweight='bold')
        ax4.set_yticks([])
        
        fig.suptitle(f'Precision: {PRECISION_NAMES[p]} ({p}-bit)', fontsize=14, fontweight='bold')
        
        frame_path = Path(output_dir) / f'frame_{p:02d}_{PRECISION_NAMES[p]}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()


def generate_all_visualizations(data_dict: Dict, output_dir: str = 'visualizations'):
    """
    Generate all 10 visualizations.
    
    Args:
        data_dict: Dictionary containing all collected data
        output_dir: Directory to save visualizations
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    # Extract data
    metrics_dict = data_dict.get('metrics', {})
    activations_dict = data_dict.get('activations', {})
    similarity_dict = data_dict.get('similarities', {})
    labels_dict = data_dict.get('labels', {})
    specialization_dict = data_dict.get('specialization', {})
    stability_dict = data_dict.get('stability', {})
    waterfall_data = data_dict.get('waterfall', {})
    
    # Generate each visualization
    print("  1.1: Routing Diversity Collapse...")
    vis_1_routing_diversity_collapse(metrics_dict, 
                                    str(output_path / '1_1_diversity_collapse.png'))
    
    print("  1.2: Path Convergence Heatmaps...")
    vis_2_path_convergence_heatmaps(activations_dict,
                                   str(output_path / '1_2_path_convergence.png'))
    
    print("  1.3: Jaccard Similarity Cascades...")
    vis_3_jaccard_cascades(similarity_dict, labels_dict,
                          str(output_path / '1_3_similarity_cascades.png'))
    
    print("  1.4: Phase Diagram...")
    vis_4_phase_diagram(metrics_dict,
                       str(output_path / '1_4_phase_diagram.png'))
    
    print("  2.1: Sparsity Trade-off...")
    vis_5_sparsity_trade_off(metrics_dict,
                            str(output_path / '2_1_sparsity_tradeoff.png'))
    
    print("  2.2: Gate Specialization...")
    vis_6_gate_specialization(specialization_dict,
                             str(output_path / '2_2_gate_specialization.png'))
    
    print("  2.3: Path Stability...")
    vis_7_path_stability(stability_dict,
                        str(output_path / '2_3_path_stability.png'))
    
    print("  2.4: Efficiency Waterfall...")
    vis_8_efficiency_waterfall(waterfall_data,
                              str(output_path / '2_4_efficiency_waterfall.png'))
    
    print("  3.1: Capability Pyramid...")
    vis_9_capability_pyramid(metrics_dict,
                            str(output_path / '3_1_capability_pyramid.png'))
    
    print("  3.2: Animation Frames...")
    create_animation_frames(activations_dict, metrics_dict, similarity_dict, output_path)
    
    print(f"All visualizations saved to {output_path}")
