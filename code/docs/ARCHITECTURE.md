# Architecture and Design Details

## System Overview

```
BNN as Discrete Path Decomposition
├── Theory (paper/)
│   ├── Path-sum representation
│   ├── Routing capacity theorem
│   └── Connection to SCRFP/MoE
│
├── Implementation (src/)
│   ├── Binary layers (1-bit weights)
│   ├── BNN models (SimpleBNN, DeepBNN)
│   ├── Routing analysis tools
│   └── Visualization suite
│
└── Experiments (experiments/)
    ├── MNIST training
    └── Routing pattern analysis
```

## Core Components

### 1. Binary Linear Layer (`src/binary_layer.py`)

**Purpose:** Implements fully connected layer with binary weights.

**Key Features:**
- Weights initialized as `N(0, 1)` then binarized via `sign()`
- Forward pass: `output = sign(W_binary @ x + b)`
- Straight-through estimator for gradients

**Math:**
```
z_l = W_l ⊙ a_{l-1} + b_l    (where W_l ∈ {-1, +1})
a_l = sign(z_l) ∈ {-1, +1}
```

**Gradient Flow:**
- Backward: Treat `sign()` as identity (straight-through)
- Updates: `dL/dW` flows back as if no quantization

### 2. BNN Models (`src/bnn_model.py`)

#### SimpleBNN
```
Input (784)
    ↓
BinaryLinear(784 → 256) + sign()
    ↓
BinaryLinear(256 → 256) + sign()
    ↓
BinaryLinear(256 → 10)
    ↓
Output (10)
```

**Architecture Choice:**
- 2 hidden binary layers capture complex routing
- 256 neurons per layer = $2^{256}$ possible activation states per layer
- Non-binary output layer for classification (no quantization loss)

**Why This Works:**
- Binary layers learn discrete decision boundaries
- Exponential routing capacity despite 1-bit weights
- Mimics MoE with implicit expert gating

#### DeepBNN
- Configurable number of layers
- Flexible hidden dimensions
- Support for deeper exploration (L > 3)

### 3. Routing Pattern Analysis (`src/routing.py`)

**Core Concept:** Extract and analyze discrete routing patterns

**Functions:**

**`get_activation_pattern(model, x)`**
- Returns list of binary activation tensors
- One tensor per hidden layer
- Shape: (batch_size, hidden_dim)

**`jaccard_similarity(a, b)`**
```
sim(a, b) = |a ∩ b| / |a ∪ b|
```
- Interprets {-1, +1} as {inactive, active}
- Range: [0, 1]
- 1.0 = identical paths
- 0.0 = completely different paths

**`compute_similarity_matrix(activations)`**
- Pairwise Jaccard similarity for batch
- Output: (batch_size × batch_size) matrix
- Enables analysis of within vs. between-class routing

**`get_routing_sparsity(activations)`**
```
Sparsity = fraction of neurons set to -1 (inactive)
         = (activations == -1).mean()
```
- Measures how selective each neuron is
- High sparsity = sparse activation (few active neurons)
- Reflects MoE sparsity principle

**`get_routing_diversity(activations)`**
- Average dissimilarity between routing patterns
- Measures variety of paths used
- Related to effective number of experts

### 4. Visualization Suite (`src/visualization.py`)

#### Method 1: Activation Heatmap
```
Rows:    Samples (test set)
Columns: Neurons in layer
Colors:  -1 (inactive) → +1 (active)
```

**Interpretation:**
- Vertical patterns = sample-specific neurons
- Horizontal patterns = class-universal neurons
- Block structure = emergent classes

#### Method 2: Similarity Matrix
```
Heatmap of Jaccard similarities
- Diagonal: 1.0 (self-similarity)
- Within-class block: high values
- Cross-class: low values
```

**Expected Structure:**
```
[High  High  High  | Low   Low  ]
[High  High  High  | Low   Low  ]
[High  High  High  | Low   Low  ]
[-----|---------|---------|-----]
[Low   Low   Low  | High  High ]
[Low   Low   Low  | High  High ]
```

Shows routing is organized by class.

#### Method 3: t-SNE Embedding
- Project routing vectors to 2D
- Color by class label
- Reveals cluster structure in routing space

**Good Result:** Clear circular/separated clusters per class

**Bad Result:** Random scattered points (no structure learned)

#### Method 4: Expert Histogram
- Divide neurons into `num_experts` groups
- Count average activations per group per class
- Bar chart: neurons vs. classes

**Expected Pattern:**
- Some experts light up for class A
- Different experts for class B
- Sparse utilization (not all experts always used)

## Data Flow: Training

```
Input Batch (64, 784)
    ↓
BinaryLinear layer 1: sign(W₁x + b₁)
    ↓ (batch, 256) binary activations
BinaryLinear layer 2: sign(W₂a₁ + b₂)
    ↓ (batch, 256) binary activations
Linear output layer: W₃a₂ + b₃
    ↓ (batch, 10) logits
CrossEntropyLoss
    ↓ (scalar) loss
Backward Pass (straight-through):
    ↓
Update W₁, W₂, W₃ using gradients
```

## Data Flow: Routing Analysis

```
Trained Model + Test Batch
    ↓
forward_with_activations() returns:
    - Logits (for accuracy)
    - Activation list [a₁, a₂]
    ↓
Extract patterns: A(x) = [a₁, a₂]
    ↓
Compute similarities: sim(A(xᵢ), A(xⱼ))
    ↓
Build similarity matrix (batch × batch)
    ↓
Visualize: heatmap, t-SNE, histogram
```

## Key Design Decisions

### Why Binary Weights?

1. **Exponential Routing Capacity**: $2^d$ configurations per layer
2. **Discrete Interpretability**: Clear on/off decisions
3. **Compression**: 1 bit per weight vs. 32 bits
4. **Connection to SCRFP**: Natural gating mechanism

### Why {-1, +1} Instead of {0, 1}?

- Zero-mean (helps with gradient flow)
- Natural from `sign()` function
- Symmetric interpretation: -1 = gate closed, +1 = gate open

### Why Sign Activation?

- Creates discrete routing patterns
- Differentiable (with straight-through estimator)
- Hard selection (not soft like sigmoid/tanh)
- Equivalent to threshold at 0

### Why Jaccard Similarity?

- Treats activations as sets (natural for discrete patterns)
- Ignores magnitude (all that matters is on/off)
- Range [0,1] for easy interpretation
- Interpretable: overlap fraction

## Implementation Considerations

### Numerical Stability

**Issue:** `sign(0)` is undefined, gradients can vanish.

**Solution:**
```python
out = torch.sign(z)
out = torch.where(out == 0, torch.ones_like(out), out)
```
Map 0 → +1 explicitly.

### Straight-Through Estimator

**Issue:** `sign()` has zero gradient almost everywhere.

**Solution:** During backprop, treat as identity:
```
∇z loss = gradient from output
∇x loss = ∇z loss @ W.T  (ignoring sign's derivative)
```

PyTorch handles this automatically for user-defined backward.

### Memory Efficiency

- Binary activations stored as float (could be uint8 for true 1-bit)
- Similarity matrix is dense (batch_size²)
- For 1000 samples: 1000×1000 = 1M floats ≈ 4MB

### Scalability

Can extend to:
- **Larger networks**: Just increase hidden_dim or num_layers
- **Convolutional layers**: Binary convolution (same routing idea)
- **Recurrent layers**: Binary RNN units
- **Vision Transformers**: Binary attention heads

## Testing and Validation

### Unit Tests
```python
# Test BinaryLinear
x = torch.randn(10, 784)
layer = BinaryLinear(784, 256)
y = layer(x)
assert y.shape == (10, 256)
assert (y == -1).any() or (y == 1).any()
```

### Integration Tests
```python
# Test SimpleBNN
model = SimpleBNN()
x = torch.randn(8, 784)
out, acts = model.forward_with_activations(x)
assert out.shape == (8, 10)
assert len(acts) == 2  # Two hidden layers
```

### Experiment Validation
```python
# Check training progression
assert test_accuracies[-1] > 90  # Should reach decent accuracy
assert routing_stats[-1]['sparsity'] < 0.6  # Not too sparse
assert routing_stats[-1]['diversity'] > 0.3  # Some diversity
```

## Performance Metrics

### Accuracy
- **Expected on MNIST**: 94-97%
- **With quantization**: Usually only 1-3% loss vs. full precision

### Sparsity
- **Expected**: 40-50% of neurons inactive per layer
- **Interpretation**: Selective routing, MoE-like behavior

### Diversity
- **Expected**: 0.5-0.8
- **Interpretation**: Network uses variety of routing patterns

### Routing Quality
- **Within-class similarity**: Should be > 0.6
- **Cross-class similarity**: Should be < 0.4
- **Good indicator**: Clear structure in similarity matrix

## Mathematical Summary

### Path-Sum Representation
$$f(x) = \sum_{p \mathcal{P}} w_p \cdot g_p(x)$$

where $p \in \mathcal{P}$ are discrete paths indexed by routing patterns.

### Routing Capacity Theorem
For L-layer BNN with width $d_l$:
$$|R| \leq \prod_{l=1}^L 2^{d_l}$$

This bounds the number of realizable routing patterns.

### Mixture of Experts Connection
Each routing pattern selects an "expert" (subset of neuron combinations).
The routing function is the binary activation pattern.

### SCRFP Connection
[Discrete path decomposition directly formalizes routing in SCRFP framework]

---

## File Structure Detailed

```
code/
├── src/
│   ├── __init__.py              # Public API
│   ├── binary_layer.py          # BinaryLinear (184 lines)
│   ├── bnn_model.py             # SimpleBNN, DeepBNN (150 lines)
│   ├── routing.py               # Analysis tools (180 lines)
│   └── visualization.py         # Plots (350 lines)
│
├── experiments/
│   ├── train_mnist.py           # Training loop (200 lines)
│   └── analyze_routing.py        # Visualization runner (80 lines)
│
├── docs/
│   ├── WALKTHROUGH.md           # (This document)
│   └── ARCHITECTURE.md          # Design details
│
└── results/                      # Generated after training
    ├── bnn_mnist_model.pth      # Trained weights
    └── metrics.json             # Training metrics
```

Total implementation: ~1500 lines of core code
