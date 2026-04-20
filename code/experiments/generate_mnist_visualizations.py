"""
Generate visualizations from MNIST precision collapse metrics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load metrics
metrics_path = Path('results/mnist_precision_metrics.json')
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Extract data
precisions = sorted([int(k) for k in metrics.keys()])
patterns_l1 = [metrics[str(p)]['unique_patterns_l1'] for p in precisions]
entropy_l1 = [metrics[str(p)]['entropy_l1'] for p in precisions]
sparsity_l1 = [metrics[str(p)]['sparsity_l1'] for p in precisions]

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# ============================================================================
# VISUALIZATION 1.1: Routing Diversity Collapse
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Patterns
colors = ['#2E86AB' if p >= 8 else '#A23B72' if p >= 2 else '#F18F01' for p in precisions]
ax1.bar(range(len(precisions)), patterns_l1, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(precisions)))
ax1.set_xticklabels([f'{p}-bit' for p in precisions])
ax1.set_ylabel('Unique Routing Patterns', fontsize=12, fontweight='bold')
ax1.set_title('Routing Diversity Collapse Across Precisions', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Entropy
ax2.plot(range(len(precisions)), entropy_l1, marker='o', linewidth=2.5, 
         markersize=10, color='#E63946', markerfacecolor='#F1FAEE',
         markeredgewidth=2, markeredgecolor='#E63946')
ax2.fill_between(range(len(precisions)), entropy_l1, alpha=0.2, color='#E63946')
ax2.set_xticks(range(len(precisions)))
ax2.set_xticklabels([f'{p}-bit' for p in precisions])
ax2.set_ylabel('Information Entropy (bits)', fontsize=12, fontweight='bold')
ax2.set_title('Entropy Collapse (Information Loss)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/1_1_diversity_collapse_mnist.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/1_1_diversity_collapse_mnist.png")
plt.close()

# ============================================================================
# VISUALIZATION 1.4: Phase Diagram / Stability Zones
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

# Create regions
x = np.arange(len(precisions))
stable_mask = np.array([p >= 8 for p in precisions])
degraded_mask = np.array([(4 <= p < 8) for p in precisions])
collapsed_mask = np.array([p < 4 for p in precisions])

# Plot zones
for i in range(len(precisions)-1):
    if precisions[i] >= 8:
        ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='green', label='Stable' if i == 0 else '')
    elif precisions[i] >= 4:
        ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='yellow', label='Degraded' if i == 1 else '')
    else:
        ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='red', label='Collapsed' if i == 2 else '')

# Overlay metrics
ax2 = ax.twinx()
ax.plot(x, patterns_l1, marker='s', linewidth=3, markersize=10, color='#003D82', 
        markerfacecolor='#4A90E2', label='Unique Patterns')
ax2.plot(x, sparsity_l1, marker='^', linewidth=3, markersize=10, color='#E5004B',
        markerfacecolor='#FF69B4', label='Sparsity')

ax.set_xticks(x)
ax.set_xticklabels([f'{p}-bit' for p in precisions])
ax.set_ylabel('Unique Routing Patterns', fontsize=12, fontweight='bold', color='#003D82')
ax2.set_ylabel('Neuron Sparsity', fontsize=12, fontweight='bold', color='#E5004B')
ax.set_title('Circuit Collapse Phase Diagram (Layer 1)', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelcolor='#003D82')
ax2.tick_params(axis='y', labelcolor='#E5004B')
ax.grid(True, alpha=0.3, axis='y')

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('visualizations/1_4_phase_diagram_mnist.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/1_4_phase_diagram_mnist.png")
plt.close()

# ============================================================================
# VISUALIZATION 2.1: Sparsity Trade-off
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(range(len(precisions)), [s*100 for s in sparsity_l1], 
              color=['#2E86AB', '#2E86AB', '#355C7D', '#A23B72', '#F18F01', '#FF0040'],
              edgecolor='black', linewidth=2, alpha=0.85)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, sparsity_l1)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(precisions)))
ax.set_xticklabels([f'{p}-bit' for p in precisions], fontsize=11)
ax.set_ylabel('Neuron Sparsity (%)', fontsize=12, fontweight='bold')
ax.set_title('Sparsity Trade-off Across Precisions (MNIST Data)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/2_1_sparsity_tradeoff_mnist.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/2_1_sparsity_tradeoff_mnist.png")
plt.close()

# ============================================================================
# VISUALIZATION 3.1: Capability Pyramid
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Create stacked visualization
x_pos = np.arange(len(precisions))
heights = np.array(patterns_l1)
colors_gradient = ['#2E86AB', '#356BA1', '#385597', '#3B4A8D', '#3D3F83', '#400000']

bars = ax.bar(x_pos, heights, color=colors_gradient, edgecolor='black', linewidth=2, alpha=0.9)

# Add labels
for i, (bar, val, entropy, sparsity) in enumerate(zip(bars, patterns_l1, entropy_l1, sparsity_l1)):
    height = bar.get_height()
    label_y = height / 2
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{val}\npatterns\n{entropy:.2f} bits\n{sparsity*100:.0f}%',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

ax.set_xticks(x_pos)
ax.set_xticklabels([f'{p}-bit' for p in precisions], fontsize=11)
ax.set_ylabel('Routing Pattern Capacity', fontsize=12, fontweight='bold')
ax.set_title('Circuit Capability Pyramid Across Precisions', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/3_1_capability_pyramid_mnist.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/3_1_capability_pyramid_mnist.png")
plt.close()

# ============================================================================
# ADDITIONAL: Precision vs Metrics Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create matrix: rows = metrics, columns = precisions
metrics_matrix = np.array([
    [patterns_l1[i]/1000 for i in range(len(precisions))],  # Normalized patterns
    entropy_l1,  # Entropy
    sparsity_l1  # Sparsity
])

im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(precisions)))
ax.set_yticks(np.arange(3))
ax.set_xticklabels([f'{p}-bit' for p in precisions])
ax.set_yticklabels(['Patterns\n(normalized)', 'Entropy\n(bits)', 'Sparsity'])

# Add text annotations
for i in range(3):
    for j in range(len(precisions)):
        val = metrics_matrix[i, j]
        text = ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                      color="white" if val < 0.5 else "black", fontweight='bold')

ax.set_title('MNIST Circuit Collapse Metrics Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Value')

plt.tight_layout()
plt.savefig('visualizations/mnist_metrics_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/mnist_metrics_heatmap.png")
plt.close()

print("\n" + "="*70)
print("ALL MNIST VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
