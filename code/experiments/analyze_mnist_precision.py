"""
CIRCUIT COLLAPSE ANALYSIS - MNIST VERSION
Full pipeline with embedded code, no external module dependencies (except torch/numpy).
Analyzes how routing patterns collapse under quantization using real MNIST data.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# ============================================================================
# PART 1: BINARY NEURAL NETWORK IMPLEMENTATION
# ============================================================================

class BinaryLinear(nn.Module):
    """Binary linear layer with sign activation"""
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
        # Binary weights and activations
        bw = torch.sign(self.weight)
        ba = torch.sign(x)
        
        # Linear operation with binary values (for testing, use binarized forward)
        output = nn.functional.linear(ba, bw, self.bias)
        return output


class SimpleBNN(nn.Module):
    """Simple Binary Neural Network"""
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(SimpleBNN, self).__init__()
        
        self.fc1 = BinaryLinear(input_dim, hidden_dim, bias=True)
        self.fc2 = BinaryLinear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, num_classes, bias=True)  # Final layer not binarized
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = torch.sign(x)  # Binary activation
        
        # Layer 2
        x = self.fc2(x)
        x = torch.sign(x)  # Binary activation
        
        # Output layer
        x = self.fc3(x)
        return x
    
    def forward_with_activations(self, x):
        """Forward pass that returns intermediate activations"""
        # Layer 1
        x1 = self.fc1(x)
        a1 = torch.sign(x1)  # Binary activation
        
        # Layer 2
        x2 = self.fc2(a1)
        a2 = torch.sign(x2)  # Binary activation
        
        # Output layer
        output = self.fc3(a2)
        
        return output, [a1, a2]


# ============================================================================
# PART 2: QUANTIZATION FUNCTIONS
# ============================================================================

def quantize_tensor(x, bits):
    """Quantize a tensor to specified bit precision"""
    if bits == 32:
        return x.float()
    
    # Compute quantization scale
    x_min = x.min()
    x_max = x.max()
    
    if x_max == x_min:
        return x
    
    # Symmetric quantization
    scale = (2 ** (bits - 1) - 1) / max(abs(x_min), abs(x_max))
    
    # Quantize
    x_q = torch.round(x * scale) / scale
    
    # Clamp to valid range
    max_val = 2 ** (bits - 1) - 1
    x_q = torch.clamp(x_q, -max_val / scale, max_val / scale)
    
    return x_q


class QuantizedModel(nn.Module):
    """Wraps a model with quantization"""
    def __init__(self, model, bits):
        super(QuantizedModel, self).__init__()
        self.model = model
        self.bits = bits
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Quantize all model weights"""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.dim() > 1:  # Only quantize weights, not biases
                    param.data = quantize_tensor(param.data, self.bits)
    
    def forward(self, x):
        return self.model(x)
    
    def forward_with_activations(self, x):
        return self.model.forward_with_activations(x)


# ============================================================================
# PART 3: METRIC COMPUTATION FUNCTIONS
# ============================================================================

def get_routing_pattern(activations):
    """Convert activations to routing patterns (binary patterns)"""
    # Convert to numpy and binarize
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    
    # Binarize activation patterns (positive = 1, negative = 0)
    patterns = (activations > 0).astype(int)
    
    # Convert each sample's pattern to a tuple (hashable)
    pattern_hashes = [tuple(p) for p in patterns]
    return pattern_hashes


def compute_metrics(model, data_loader, device):
    """Compute routing metrics for given model and data"""
    model.eval()
    
    all_patterns_l1 = []
    all_patterns_l2 = []
    all_sparsity_l1 = []
    all_sparsity_l2 = []
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.view(images.size(0), -1).to(device)
            
            # Get activations from both layers
            _, (activations_l1, activations_l2) = model.forward_with_activations(images)
            
            # Compute patterns
            patterns_l1 = get_routing_pattern(activations_l1)
            patterns_l2 = get_routing_pattern(activations_l2)
            
            all_patterns_l1.extend(patterns_l1)
            all_patterns_l2.extend(patterns_l2)
            
            # Compute sparsity (fraction of inactive neurons)
            sparsity_l1 = (activations_l1 <= 0).float().mean().item()
            sparsity_l2 = (activations_l2 <= 0).float().mean().item()
            
            all_sparsity_l1.extend([sparsity_l1] * len(patterns_l1))
            all_sparsity_l2.extend([sparsity_l2] * len(patterns_l2))
    
    # Count unique patterns
    unique_patterns_l1 = len(set(all_patterns_l1))
    unique_patterns_l2 = len(set(all_patterns_l2))
    
    # Compute entropy
    def entropy(patterns):
        if not patterns:
            return 0
        pattern_counts = {}
        for p in patterns:
            pattern_counts[p] = pattern_counts.get(p, 0) + 1
        
        total = len(patterns)
        probs = np.array(list(pattern_counts.values())) / total
        
        # Shannon entropy in bits
        ent = -np.sum(probs * np.log2(probs + 1e-10))
        return ent
    
    entropy_l1 = entropy(all_patterns_l1)
    entropy_l2 = entropy(all_patterns_l2)
    
    avg_sparsity_l1 = np.mean(all_sparsity_l1)
    avg_sparsity_l2 = np.mean(all_sparsity_l2)
    
    return {
        'unique_patterns_l1': unique_patterns_l1,
        'unique_patterns_l2': unique_patterns_l2,
        'entropy_l1': entropy_l1,
        'entropy_l2': entropy_l2,
        'sparsity_l1': avg_sparsity_l1,
        'sparsity_l2': avg_sparsity_l2,
        'patterns_l1': all_patterns_l1,
        'patterns_l2': all_patterns_l2,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("CIRCUIT COLLAPSE ANALYSIS - MNIST VERSION")
    print("=" * 80)
    print()
    
    device = torch.device('cpu')  # Use CPU for compatibility
    print(f"Device: {device}\n")
    
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 0.001
    PRECISION_LEVELS = [32, 16, 8, 4, 2, 1]
    
    # Create output directories
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # PHASE 1: TRAIN MODEL ON MNIST
    # ========================================================================
    print("=" * 80)
    print("PHASE 1: TRAINING MODEL ON MNIST")
    print("=" * 80)
    
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard normalization
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Use smaller subsets for faster training
    train_subset = torch.utils.data.Subset(train_dataset, range(6000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")
    print()
    
    # Initialize model
    print("Initializing BNN model...")
    model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"Training for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        train_loss /= batch_count
        
        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={test_acc:.2f}%")
    
    print("Model saved!\n")
    
    # Save model
    model_path = results_dir / 'bnn_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    
    # ========================================================================
    # PHASE 2: COLLECT DATA ACROSS PRECISIONS
    # ========================================================================
    print("=" * 80)
    print("PHASE 2: COLLECTING DATA ACROSS PRECISIONS")
    print("=" * 80)
    print()
    
    precision_metrics = {}
    precision_data = {}
    
    for bits in PRECISION_LEVELS:
        print(f"Analyzing {bits:2d}-bit precision...")
        
        # Quantize model
        model_q = QuantizedModel(model, bits)
        model_q = model_q.to(device)
        
        # Compute metrics
        metrics = compute_metrics(model_q, test_loader, device)
        
        precision_metrics[bits] = {
            'unique_patterns_l1': metrics['unique_patterns_l1'],
            'unique_patterns_l2': metrics['unique_patterns_l2'],
            'entropy_l1': float(metrics['entropy_l1']),
            'entropy_l2': float(metrics['entropy_l2']),
            'sparsity_l1': float(metrics['sparsity_l1']),
            'sparsity_l2': float(metrics['sparsity_l2']),
        }
        
        precision_data[bits] = {
            'patterns_l1': metrics['patterns_l1'],
            'patterns_l2': metrics['patterns_l2'],
        }
        
        print(f"  L1 patterns: {metrics['unique_patterns_l1']}, entropy: {metrics['entropy_l1']:6.2f}, sparsity: {metrics['sparsity_l1']:6.2%}")
        print(f"  L2 patterns: {metrics['unique_patterns_l2']}, entropy: {metrics['entropy_l2']:6.2f}, sparsity: {metrics['sparsity_l2']:6.2%}")
    
    print()
    
    # ========================================================================
    # PHASE 3: SAVE RESULTS
    # ========================================================================
    print("=" * 80)
    print("PHASE 3: SAVING RESULTS")
    print("=" * 80)
    print()
    
    metrics_path = results_dir / 'mnist_precision_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(precision_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    data_path = results_dir / 'mnist_precision_data.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(precision_data, f)
    print(f"Data saved to {data_path}")
    print()
    
    # ========================================================================
    # PHASE 4: SUMMARY
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: CIRCUIT COLLAPSE ANALYSIS")
    print("=" * 80)
    print()
    
    print("Routing Diversity (Unique Patterns - Layer 1):")
    for bits in PRECISION_LEVELS:
        patterns = precision_metrics[bits]['unique_patterns_l1']
        print(f"  {bits:2d}-bit: {patterns:4d} patterns")
    print()
    
    print("Routing Entropy (bits - Layer 1):")
    for bits in PRECISION_LEVELS:
        entropy = precision_metrics[bits]['entropy_l1']
        print(f"  {bits:2d}-bit: {entropy:6.2f}")
    print()
    
    print("Neuron Sparsity (Layer 1):")
    for bits in PRECISION_LEVELS:
        sparsity = precision_metrics[bits]['sparsity_l1']
        print(f"  {bits:2d}-bit: {sparsity:6.2%}")
    print()
    
    # Compute collapse metrics
    patterns_32 = precision_metrics[32]['unique_patterns_l1']
    patterns_1 = precision_metrics[1]['unique_patterns_l1']
    collapse_ratio = patterns_1 / patterns_32 if patterns_32 > 0 else 1.0
    
    entropy_32 = precision_metrics[32]['entropy_l1']
    entropy_1 = precision_metrics[1]['entropy_l1']
    entropy_loss = entropy_32 - entropy_1
    
    sparsity_32 = precision_metrics[32]['sparsity_l1']
    sparsity_1 = precision_metrics[1]['sparsity_l1']
    sparsity_change = sparsity_1 - sparsity_32
    
    print("Key Findings:")
    print(f"  - Routing diversity collapsed {collapse_ratio:.2f}x from float32 to binary")
    print(f"  - Entropy decreased by {entropy_loss:.2f} bits")
    print(f"  - Sparsity changed by {sparsity_change:+.2%}")
    print()
    
    print("=" * 80)
    print("Analysis complete! Check results/ and visualizations/ directories.")
    print("=" * 80)


if __name__ == '__main__':
    main()
