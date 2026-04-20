"""
Standalone precision collapse analysis (minimal dependencies version).

This runs the full analysis pipeline without requiring all src imports.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path

# ============================================================================
# MINIMAL BNN IMPLEMENTATION (to avoid __init__.py imports)
# ============================================================================

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        # Binarize weights
        w_binary = torch.sign(self.weight)
        # Forward pass with binary weights
        z = nn.functional.linear(x, w_binary, self.bias)
        # Binary activation
        a = torch.sign(z)
        return a


class SimpleBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BinaryLinear(784, 256)
        self.layer2 = BinaryLinear(256, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        return x
    
    def forward_with_activations(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        a1 = x.clone()
        x = self.layer2(x)
        a2 = x.clone()
        x = self.output(x)
        return x, [a1, a2]


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_tensor(x, bits):
    if bits == 32:
        return x
    if bits == 16:
        return x.half().float()
    if bits == 8:
        scale = 127.0 / (x.abs().max() + 1e-8)
        return torch.round(x * scale).clamp(-127, 127) / scale
    if bits == 1:
        return torch.sign(x)
    return x


def apply_quantization_to_model(model, bits):
    """Quantize model weights in-place."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = quantize_tensor(param.data, bits)


# ============================================================================
# METRICS
# ============================================================================

def unique_patterns(acts):
    patterns = set()
    for i in range(acts.shape[0]):
        p = tuple(acts[i].cpu().numpy().astype(np.int8))
        patterns.add(p)
    return len(patterns)


def routing_entropy(acts):
    patterns = {}
    for i in range(acts.shape[0]):
        p = tuple(acts[i].cpu().numpy().astype(np.int8))
        patterns[p] = patterns.get(p, 0) + 1
    probs = np.array(list(patterns.values())) / acts.shape[0]
    # Manual entropy: -sum(p * log2(p))
    entropy_val = 0
    for prob in probs:
        if prob > 0:
            entropy_val -= prob * np.log2(prob)
    return entropy_val


def sparsity(acts):
    return ((acts == -1).float().mean()).item()


def jaccard_sim(a, b):
    a_bin = (a == 1).float()
    b_bin = (b == 1).float()
    intersection = (a_bin * b_bin).sum()
    union = torch.maximum(a_bin, b_bin).sum()
    return (intersection / union).item() if union > 0 else 1.0


def similarity_matrix(acts):
    mat = np.zeros((acts.shape[0], acts.shape[0]))
    for i in range(acts.shape[0]):
        for j in range(acts.shape[0]):
            mat[i, j] = jaccard_sim(acts[i], acts[j])
    return mat


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CIRCUIT COLLAPSE ANALYSIS - STANDALONE VERSION")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create output directories
    Path("results").mkdir(exist_ok=True)
    Path("visualizations").mkdir(exist_ok=True)
    
    # ========== PHASE 1: TRAIN MODEL ==========
    print("PHASE 1: TRAINING MODEL")
    print("-" * 80)
    
    model = SimpleBNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic data
    print("Generating synthetic data...")
    X_train = torch.randn(6000, 784)
    y_train = torch.randint(0, 10, (6000,))
    X_test = torch.randn(1000, 784)
    y_test = torch.randint(0, 10, (1000,))
    
    # Training
    print("Training (5 epochs)...")
    for epoch in range(1, 6):
        model.train()
        loss_sum = 0
        for i in range(0, 6000, 64):
            x = X_train[i:i+64].to(device)
            y = y_train[i:i+64].to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test.to(device))
            acc = (outputs.argmax(1) == y_test.to(device)).float().mean().item()
        
        print(f"  Epoch {epoch}: Loss={loss_sum/95:.4f}, Acc={acc:.2%}")
    
    # Save model
    torch.save(model.state_dict(), "results/bnn_model.pth")
    print("Model saved!\n")
    
    # ========== PHASE 2: COLLECT DATA ==========
    print("PHASE 2: COLLECTING DATA ACROSS PRECISIONS")
    print("-" * 80)
    
    precisions = [32, 16, 8, 4, 2, 1]
    metrics_dict = {}
    activations_dict = {}
    similarities_dict = {}
    
    for bits in precisions:
        print(f"Analyzing {bits}-bit precision...")
        
        # Create quantized copy
        model_q = SimpleBNN().to(device)
        model_q.load_state_dict(torch.load("results/bnn_model.pth"))
        apply_quantization_to_model(model_q, bits)
        model_q.eval()
        
        # Collect activations
        all_acts_l1 = []
        all_acts_l2 = []
        
        with torch.no_grad():
            for i in range(0, 1000, 64):
                x = X_test[i:i+64].to(device)
                _, (a1, a2) = model_q.forward_with_activations(x)
                all_acts_l1.append(a1)
                all_acts_l2.append(a2)
        
        acts_l1 = torch.cat(all_acts_l1)
        acts_l2 = torch.cat(all_acts_l2)
        
        activations_dict[bits] = [acts_l1, acts_l2]
        similarities_dict[bits] = similarity_matrix(acts_l1)
        
        # Compute metrics
        metrics_dict[bits] = {
            'layer_1_unique_patterns': unique_patterns(acts_l1),
            'layer_1_entropy': routing_entropy(acts_l1),
            'layer_1_sparsity': sparsity(acts_l1),
            'layer_2_unique_patterns': unique_patterns(acts_l2),
            'layer_2_entropy': routing_entropy(acts_l2),
            'layer_2_sparsity': sparsity(acts_l2)
        }
        
        print(f"  L1 patterns: {metrics_dict[bits]['layer_1_unique_patterns']:4d}, "
              f"entropy: {metrics_dict[bits]['layer_1_entropy']:5.2f}, "
              f"sparsity: {metrics_dict[bits]['layer_1_sparsity']:.2%}")
    
    print()
    
    # ========== PHASE 3: SAVE RESULTS ==========
    print("PHASE 3: SAVING RESULTS")
    print("-" * 80)
    
    # Save metrics as JSON
    metrics_json = {}
    for p, m in metrics_dict.items():
        metrics_json[str(p)] = {k: float(v) for k, v in m.items()}
    
    with open("results/precision_metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save detailed data
    with open("results/precision_data.pkl", 'wb') as f:
        pickle.dump({
            'metrics': metrics_dict,
            'similarities': similarities_dict,
            'activations': {k: [a.cpu() for a in v] for k, v in activations_dict.items()}
        }, f)
    
    print("Metrics saved to results/precision_metrics.json")
    print("Data saved to results/precision_data.pkl\n")
    
    # ========== PHASE 4: SUMMARY ==========
    print("="*80)
    print("SUMMARY: CIRCUIT COLLAPSE ANALYSIS")
    print("="*80 + "\n")
    
    print("Routing Diversity (Unique Patterns - Layer 1):")
    for p in precisions:
        count = metrics_dict[p]['layer_1_unique_patterns']
        print(f"  {p:2d}-bit: {count:5d} patterns")
    
    print("\nRouting Entropy (bits - Layer 1):")
    for p in precisions:
        ent = metrics_dict[p]['layer_1_entropy']
        print(f"  {p:2d}-bit: {ent:6.2f}")
    
    print("\nNeuron Sparsity (Layer 1):")
    for p in precisions:
        sp = metrics_dict[p]['layer_1_sparsity']
        print(f"  {p:2d}-bit: {sp:6.2%}")
    
    # Key insights
    print("\nKey Findings:")
    collapse_factor = (metrics_dict[32]['layer_1_unique_patterns'] / 
                      max(1, metrics_dict[1]['layer_1_unique_patterns']))
    print(f"  - Routing diversity collapsed {collapse_factor:.1f}x from float32 to binary")
    
    entropy_increase = (metrics_dict[32]['layer_1_entropy'] - 
                       metrics_dict[1]['layer_1_entropy'])
    print(f"  - Entropy decreased by {entropy_increase:.2f} bits (more disorder)")
    
    sparsity_change = (metrics_dict[32]['layer_1_sparsity'] - 
                      metrics_dict[1]['layer_1_sparsity'])
    print(f"  - Sparsity changed by {sparsity_change:+.2%}")
    
    print("\n" + "="*80)
    print("Analysis complete! Check results/ and visualizations/ directories.")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
