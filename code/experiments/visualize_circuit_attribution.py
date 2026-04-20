"""
Circuit Attribution Visualizations
Shows which inputs activate which circuits and how attribution changes with precision.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE

# Load MNIST metrics and model
metrics_path = Path('results/mnist_precision_metrics.json')
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

model_path = Path('results/bnn_mnist_model.pth')

# ============================================================================
# PART 1: Load Model and Data
# ============================================================================

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        bw = torch.sign(self.weight)
        ba = torch.sign(x)
        output = nn.functional.linear(ba, bw, self.bias)
        return output

class SimpleBNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(SimpleBNN, self).__init__()
        self.fc1 = BinaryLinear(input_dim, hidden_dim, bias=True)
        self.fc2 = BinaryLinear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, num_classes, bias=True)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sign(x)
        x = self.fc2(x)
        x = torch.sign(x)
        x = self.fc3(x)
        return x
    
    def forward_with_activations(self, x):
        x1 = self.fc1(x)
        a1 = torch.sign(x1)
        x2 = self.fc2(a1)
        a2 = torch.sign(x2)
        output = self.fc3(a2)
        return output, [a1, a2]

# Load model
device = torch.device('cpu')
model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
model.load_state_dict(torch.load(model_path, weights_only=False))
model = model.to(device)
model.eval()

# Load test data
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_subset = torch.utils.data.Subset(test_dataset, range(1000))
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=100, shuffle=False)

# ============================================================================
# VISUALIZATION 1: Input-to-Circuit Attribution (Saliency Maps)
# ============================================================================

print("Computing input-to-circuit attributions...")

# Collect inputs and their circuit activations
all_inputs = []
all_activations_l1 = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images_flat = images.view(images.size(0), -1).to(device)
        _, activations = model.forward_with_activations(images_flat)
        
        all_inputs.append(images.numpy())
        all_activations_l1.append(activations[0].numpy())
        all_labels.append(labels.numpy())

all_inputs = np.concatenate(all_inputs)  # (1000, 1, 28, 28)
all_activations_l1 = np.concatenate(all_activations_l1)  # (1000, 256)
all_labels = np.concatenate(all_labels)  # (1000,)

# Compute mean activation pattern per digit class
mean_activations_by_class = {}
mean_inputs_by_class = {}
for digit in range(10):
    mask = all_labels == digit
    mean_activations_by_class[digit] = all_activations_l1[mask].mean(axis=0)
    mean_inputs_by_class[digit] = all_inputs[mask].mean(axis=0)

# Visualize: For each digit, show input and top activated neurons
fig, axes = plt.subplots(10, 3, figsize=(12, 20))

for digit in range(10):
    # Input image
    ax = axes[digit, 0]
    ax.imshow(mean_inputs_by_class[digit][0], cmap='gray')
    ax.set_title(f'Digit {digit}', fontweight='bold')
    ax.axis('off')
    
    # Top 5 activated neurons in layer 1
    activations = mean_activations_by_class[digit]
    top_neurons = np.argsort(-np.abs(activations))[:5]
    
    # Neuron activation bar chart
    ax = axes[digit, 1]
    ax.barh(range(5), activations[top_neurons], color='#E63946')
    ax.set_yticks(range(5))
    ax.set_yticklabels([f'N{n}' for n in top_neurons])
    ax.set_xlabel('Activation')
    ax.set_title(f'Top Neurons', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Circuit encoding (activation pattern as binary)
    ax = axes[digit, 2]
    circuit_pattern = (activations > 0).astype(int)
    unique_circuits = len(set(tuple(circuit_pattern)))
    
    # Show histogram of neuron activation states
    active_count = (activations > 0).sum()
    inactive_count = (activations <= 0).sum()
    ax.bar(['Active', 'Inactive'], [active_count, inactive_count], 
           color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Neuron Count')
    ax.set_title(f'Circuit Composition\n(Active: {active_count})', fontsize=10)
    ax.set_ylim(0, 256)

plt.tight_layout()
plt.savefig('visualizations/4_1_input_circuit_attribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/4_1_input_circuit_attribution.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Circuit Collapse Saliency - How precision degrades attribution
# ============================================================================

print("Computing gradient-based saliency maps...")

# Compute saliency for 32-bit and 1-bit cases
def compute_saliency(model, input_image, target_layer=0):
    """Compute gradient of hidden layer activation w.r.t. input"""
    input_image = input_image.clone().detach().requires_grad_(True)
    
    # Forward to first layer
    x = model.fc1(input_image)
    
    # Sum of activations as loss
    loss = torch.sign(x).sum()
    
    if input_image.grad is not None:
        input_image.grad.zero_()
    
    loss.backward()
    
    saliency = input_image.grad.abs().detach()
    return saliency

# Select a few representative test samples
sample_indices = [0, 100, 200, 300, 400]  # Different digit examples
fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 12))

for row, idx in enumerate(sample_indices):
    sample_image = all_inputs[idx:idx+1]
    sample_tensor = torch.from_numpy(sample_image).float().view(1, 784).to(device)
    sample_tensor.requires_grad = True
    
    with torch.enable_grad():
        saliency = compute_saliency(model, sample_tensor)
    
    # Original input
    ax = axes[row, 0]
    ax.imshow(sample_image[0, 0], cmap='gray')
    ax.set_title(f'Input {idx}', fontweight='bold', fontsize=9)
    ax.axis('off')
    
    # Saliency map
    ax = axes[row, 1]
    saliency_img = saliency[0].view(28, 28).cpu().numpy()
    ax.imshow(saliency_img, cmap='hot')
    ax.set_title('Attribution', fontsize=9)
    ax.axis('off')
    
    # Circuit activation pattern (which neurons active)
    with torch.no_grad():
        x1 = model.fc1(sample_tensor)
        a1_pattern = (torch.sign(x1) > 0).float().cpu().numpy()
    
    ax = axes[row, 2]
    active_neurons = a1_pattern[0]
    ax.imshow(active_neurons.reshape(16, 16), cmap='Greys', interpolation='nearest')
    ax.set_title(f'L1 Neurons\n({active_neurons.sum():.0f}/256)', fontsize=9)
    ax.axis('off')
    
    # Attribution × Circuit: Highlight pixels that feed active neurons
    ax = axes[row, 3]
    # Weight each pixel's saliency by how much it contributes to active neurons
    w = model.fc1.weight.data.abs()  # (256, 784)
    active_weight = w[active_neurons > 0].sum(axis=0)  # (784,)
    contribution = saliency_img.flatten() * active_weight.cpu().numpy()
    contribution = contribution.reshape(28, 28)
    
    ax.imshow(contribution, cmap='viridis')
    ax.set_title('Attribution\n→ Active Neurons', fontsize=9)
    ax.axis('off')

plt.suptitle('Input Attribution to Circuits (Layer 1)', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/4_2_saliency_attribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/4_2_saliency_attribution.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Circuit Manifold - t-SNE of routing patterns
# ============================================================================

print("Computing circuit manifold (t-SNE)...")

# Use activation patterns as features
circuit_space = all_activations_l1  # (1000, 256)

# t-SNE to 2D
print("  Running t-SNE (this takes ~30 seconds)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
circuit_2d = tsne.fit_transform(circuit_space)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: By digit class
ax = axes[0]
colors_by_digit = {
    0: '#E63946', 1: '#F1FAEE', 2: '#A8DADC', 3: '#457B9D', 4: '#1D3557',
    5: '#F4A261', 6: '#E76F51', 7: '#2A9D8F', 8: '#264653', 9: '#D62828'
}
for digit in range(10):
    mask = all_labels == digit
    ax.scatter(circuit_2d[mask, 0], circuit_2d[mask, 1], 
              label=f'{digit}', alpha=0.7, s=100, 
              color=colors_by_digit.get(digit, 'gray'), edgecolors='black', linewidth=0.5)
ax.set_xlabel('Circuit t-SNE 1', fontweight='bold')
ax.set_ylabel('Circuit t-SNE 2', fontweight='bold')
ax.set_title('Circuit Space - Colored by Digit Class', fontweight='bold', fontsize=12)
ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05))
ax.grid(True, alpha=0.2)

# Plot 2: By sparsity (number of active neurons)
ax = axes[1]
sparsity_per_sample = (all_activations_l1 > 0).sum(axis=1)
scatter = ax.scatter(circuit_2d[:, 0], circuit_2d[:, 1], 
                    c=sparsity_per_sample, cmap='RdYlGn_r', 
                    s=100, edgecolors='black', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Circuit t-SNE 1', fontweight='bold')
ax.set_ylabel('Circuit t-SNE 2', fontweight='bold')
ax.set_title('Circuit Space - Colored by Sparsity', fontweight='bold', fontsize=12)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Active Neurons (L1)', fontweight='bold')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('visualizations/4_3_circuit_manifold_tsne.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/4_3_circuit_manifold_tsne.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Precision Collapse Impact on Attribution Quality
# ============================================================================

print("Analyzing precision impact on attribution...")

# For each precision level, compute how much attribution is preserved
precisions = [32, 16, 8, 4, 2, 1]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for idx, bits in enumerate(precisions):
    metrics_data = metrics[str(bits)]
    
    ax = axes[idx]
    
    # Show key metrics
    patterns = metrics_data['unique_patterns_l1']
    entropy = metrics_data['entropy_l1']
    sparsity = metrics_data['sparsity_l1']
    
    # Create a circuit representation bar
    active_neurons = int(256 * (1 - sparsity))
    inactive_neurons = int(256 * sparsity)
    
    # Left bar: circuit composition
    ax.barh([0], [active_neurons], left=0, height=0.6, 
           color='#2E86AB', edgecolor='black', linewidth=2, label='Active')
    ax.barh([0], [inactive_neurons], left=active_neurons, height=0.6,
           color='#A23B72', edgecolor='black', linewidth=2, label='Inactive')
    
    # Add text annotations
    ax.text(active_neurons/2, 0, f'{active_neurons}\nactive', 
           ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    ax.text(active_neurons + inactive_neurons/2, 0, f'{inactive_neurons}\ninactive',
           ha='center', va='center', fontweight='bold', fontsize=9, color='white')
    
    ax.set_xlim(0, 256)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 64, 128, 192, 256])
    ax.set_xlabel('Neurons')
    ax.set_title(f'{bits}-bit: {patterns} patterns\nEntropy: {entropy:.2f} bits | Sparsity: {sparsity*100:.0f}%',
                fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Precision Impact on Circuit Composition', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/4_4_precision_circuits.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/4_4_precision_circuits.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Input Feature → Circuit Flow Diagram
# ============================================================================

print("Creating information flow diagrams...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for plot_idx, digit in enumerate([0, 3, 6, 9]):  # Corners of digit space
    ax = axes[plot_idx // 2, plot_idx % 2]
    
    # Get mean input and circuit for this digit
    mask = all_labels == digit
    mean_input = mean_inputs_by_class[digit][0]
    mean_circuit = mean_activations_by_class[digit]
    
    # Draw flow diagram
    ax.text(0.5, 1.0, f'Input: Digit {digit}', ha='center', va='top', fontsize=12, fontweight='bold',
           transform=ax.transAxes)
    
    # Input space (small MNIST image)
    ax_input = fig.add_axes([ax.get_position().x0 + 0.01, ax.get_position().y0 + 0.55, 0.08, 0.35])
    ax_input.imshow(mean_input, cmap='gray')
    ax_input.set_xticks([])
    ax_input.set_yticks([])
    ax_input.set_title('Input\n784 dims', fontsize=9)
    
    # Hidden layer (circuit visualization)
    ax_circuit = fig.add_axes([ax.get_position().x0 + 0.45, ax.get_position().y0 + 0.55, 0.35, 0.35])
    
    # Reshape circuit pattern to grid
    circuit_pattern = (mean_circuit > 0).astype(int)
    circuit_grid = circuit_pattern.reshape(16, 16)
    
    ax_circuit.imshow(circuit_grid, cmap='Greys', interpolation='nearest')
    ax_circuit.set_title(f'L1 Circuit\n256 neurons\n{(circuit_pattern==1).sum()} active ({(circuit_pattern==1).sum()/256*100:.0f}%)',
                        fontsize=9, fontweight='bold')
    ax_circuit.set_xticks([])
    ax_circuit.set_yticks([])
    
    # Add arrow
    ax.annotate('', xy=(0.45, 0.75), xytext=(0.12, 0.75),
               xycoords='axes fraction', textcoords='axes fraction',
               arrowprops=dict(arrowstyle='->', lw=3, color='#2E86AB'))
    
    # Statistics
    ax.text(0.05, 0.45, f'Statistics:', fontsize=10, fontweight='bold', transform=ax.transAxes)
    active_pct = (mean_circuit > 0).sum() / len(mean_circuit) * 100
    ax.text(0.05, 0.35, f'Active Neurons: {(mean_circuit > 0).sum()}/256 ({active_pct:.0f}%)',
           fontsize=9, transform=ax.transAxes)
    
    # Top contributing features
    top_neurons = np.argsort(-mean_circuit)[:3]
    ax.text(0.05, 0.25, f'Top Neurons: {", ".join(map(str, top_neurons))}',
           fontsize=9, transform=ax.transAxes)
    
    # Pattern entropy (from our metrics - approximate)
    unique_representations = np.sum(mean_circuit > 0)  # Simplified
    ax.text(0.05, 0.15, f'Pattern Info: {np.log2(max(unique_representations, 2)):.2f} bits',
           fontsize=9, transform=ax.transAxes)
    
    ax.axis('off')

plt.savefig('visualizations/4_5_input_circuit_flow.png', dpi=150, bbox_inches='tight')
print("✓ Saved: visualizations/4_5_input_circuit_flow.png")
plt.close()

print("\n" + "="*70)
print("ALL CIRCUIT ATTRIBUTION VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print("""
Visualizations created:
  4.1 - Input-to-Circuit Attribution (Top neurons per digit)
  4.2 - Saliency Maps (Gradient-based input importance)
  4.3 - Circuit Manifold (t-SNE of routing patterns)
  4.4 - Precision Impact on Circuits (Circuit composition across bits)
  4.5 - Input→Circuit Flow Diagrams (Information flow visualization)
""")
