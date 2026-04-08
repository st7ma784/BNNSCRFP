# Walkthrough: BNN as Discrete Path Decompositions

This document provides a step-by-step guide to understanding and running the experiments.

## Overview

This repository implements a unified view of Binary Neural Networks (BNNs) as discrete path-sum models that connect to SCRFP and Mixture of Experts. The key insight is:

**BNNs are implicit Mixture of Experts systems where:**
- Each neuron acts as a binary gate (decision boundary)
- Combinations of neuron activations define discrete "paths" through the network
- The number of possible paths grows exponentially: $|R| \leq \prod_{l=1}^L 2^{d_l}$
- This provides exponential routing capacity despite 1-bit weights

## Quick Start

### 1. Install Dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Train the MNIST Model

```bash
python experiments/train_mnist.py
```

This will:
- Download MNIST dataset
- Train a 3-layer BNN: 784 → 256 → 256 → 10
- Save model to `results/bnn_mnist_model.pth`
- Save metrics to `results/metrics.json`
- Display training progress and final accuracy

**Expected output:**
```
Epoch 50: Loss=0.0123, Accuracy=96.45%, Sparsity=0.4231, Diversity=0.6789
Final Results:
  Test Accuracy: 96.45%
  Routing Sparsity: 0.4231
  Routing Diversity: 0.6789
```

### 3. Analyze Routing Patterns

```bash
python experiments/analyze_routing.py
```

This will:
- Load the trained model
- Analyze routing patterns on the test set
- Generate 4 types of visualizations per layer:
  1. **Activation Heatmaps**: Show which neurons are active for each sample
  2. **Similarity Matrices**: Jaccard similarity between routing patterns
  3. **t-SNE Embeddings**: Low-dimensional projection of routing vectors
  4. **Expert Histograms**: Activation frequency across neuron groups

All visualizations are saved to `visualizations/`

## Understanding the Concept

### What is a Routing Pattern?

For an input $x$, a routing pattern is the sequence of binary activations through the network:

$$A(x) = [a_1(x), a_2(x), a_3(x)]$$

Where each $a_l(x) \in \{-1, +1\}^{d_l}$ is the binary output of layer $l$.

**Example:**
- Sample 1: Layer 1 activations = [+1, -1, +1, +1, ...]
- Sample 2: Layer 1 activations = [+1, +1, -1, +1, ...]
- These follow different routing patterns in the network

### Jaccard Similarity

We measure similarity between two routing patterns using Jaccard index:

$$\text{sim}(x_i, x_j) = \frac{|A(x_i) \cap A(x_j)|}{|A(x_i) \cup A(x_j)|}$$

**Interpretation:**
- `sim = 1.0`: Both samples follow identical paths (all neurons activate the same)
- `sim = 0.5`: Half the neurons activate similarly
- `sim = 0.0`: No overlap in active neurons

### Why This Matters

The similarity matrix reveals:
- **Class Clustering**: Samples from the same class tend to have similar routing patterns
- **Expert Specialization**: Certain neuron combinations specialize in different classes
- **Implicit Gating**: The network learns to route different classes through different "experts"

## Interpreting Results

### Activation Heatmaps

**What to look for:**
- Strong horizontal bands of the same color → some neurons always active/inactive
- Vertical stripes → some samples have similar activation patterns
- Class-based structure → samples from same class cluster vertically

### Similarity Matrices

**What to look for:**
- Block diagonal structure → within-class similarity is high
- Off-diagonal entries are low → between-class dissimilarity
- This shows emergent class-specific routing

### t-SNE Embeddings

**What to look for:**
- Distinct clusters by color → classes form separate regions in routing space
- Good separation → strong routing-based class discrimination
- Overlapping clusters → ambiguous samples use similar paths

### Expert Histograms

**What to look for:**
- Different bars for different colors → experts specialize by class
- High bars for some experts per class → frequently used features
- Varying heights → uneven expert utilization

## Code Structure

### Core Modules

**`src/binary_layer.py`**
- `BinaryLinear`: Layer with binary weights {-1, +1}
- Forward pass: z = sign(W @ x + b)

**`src/bnn_model.py`**
- `SimpleBNN`: 3-layer BNN for MNIST
- `DeepBNN`: Configurable deeper networks
- Both support `forward_with_activations()` for routing analysis

**`src/routing.py`**
- `get_activation_pattern()`: Extract routing patterns
- `jaccard_similarity()`: Compute similarity between patterns
- `compute_similarity_matrix()`: Pairwise similarities
- `get_routing_sparsity()`: Measure activation sparsity
- `get_routing_diversity()`: Measure pattern variety

**`src/visualization.py`**
- `plot_activation_heatmap()`: Layer activations
- `plot_similarity_matrix()`: Jaccard similarities
- `plot_tsne_routing()`: Low-dimensional embedding
- `plot_expert_histogram()`: Expert activation frequency
- `plot_all_visualizations()`: Generate all plots for a model

### Experiment Scripts

**`experiments/train_mnist.py`**
- Train BNN on MNIST
- Tracks accuracy, sparsity, and diversity
- Saves model and metrics

**`experiments/analyze_routing.py`**
- Load trained model
- Generate all visualizations
- Save to `visualizations/`

## Extending the Code

### Train on Different Dataset

1. Modify `train_mnist.py`:
```python
# Change dataset
train_dataset = datasets.CIFAR10(...)  # Instead of MNIST
input_dim = 3072  # 32x32x3
num_classes = 10
```

2. Adjust model architecture:
```python
model = SimpleBNN(input_dim=3072, hidden_dim=512, num_classes=10)
```

### Add More Visualization Methods

1. Create new function in `src/visualization.py`:
```python
def custom_visualization(activations, ...):
    # Your visualization code
    pass
```

2. Call it in `plot_all_visualizations()`:
```python
custom_visualization(layer_activations, ...)
```

### Analyze Specific Samples

```python
# Load model and data
model.eval()
with torch.no_grad():
    outputs, activations = model.forward_with_activations(images)

# Extract routing for first sample
routing_pattern = activations[0][0]  # Layer 0, Sample 0
print(routing_pattern)  # {-1, +1}^d tensor
```

## Common Issues

### Out of Memory

If you get CUDA out-of-memory errors:
```python
# Reduce batch size
batch_size = 32  # Instead of 64

# Or use CPU
device = torch.device('cpu')
```

### Slow Training

- The BNN training can be slow due to sign() operations
- Reduce epochs: `epochs = 20`
- Use smaller hidden dimension: `hidden_dim = 128`

### Visualizations Don't Show

Ensure `visualizations/` directory exists:
```bash
mkdir -p code/visualizations
```

## Next Steps

1. **Paper Writing**: Use LaTeX files in `paper/` to write formal writeup
2. **Extend Experiments**: Try different architectures, datasets, training techniques
3. **Theoretical Analysis**: Prove bounds on routing capacity
4. **Practical Applications**: Use routing patterns for model compression, interpretability

## References

- **Binary Neural Networks**: Courbariaux et al., "Binarized Neural Networks" (2016)
- **Mixture of Experts**: Shazeer et al., "Outrageously Large Neural Networks" (2017)
- **SCRFP**: [Your framework paper]

## Implementation Details

### Binary Weight Binarization

Weights are binarized to $\{-1, +1\}$ using the sign function:

```python
w_binary = torch.sign(self.weight)
```

This is a straight-through estimator - during backprop, gradients flow through as if using the real-valued weights.

### Forward Pass

```python
z = F.linear(x, w_binary, bias)
x_out = torch.sign(z)
```

The sign function maps to $\{-1, +1\}$. We use $+1$ for zero (no explicit threshold).

### Routing Pattern Extraction

```python
def forward_with_activations(self, x):
    activations = []
    x = self.layer1(x)
    activations.append(x.clone())  # Binary activations
    ...
    return output, activations
```

This records the discrete binary outputs at each layer, forming the routing pattern.

---

For more details, see the main paper in `paper/main.tex`
