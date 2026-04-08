# BNN as Discrete Path Decompositions: Code Repository

This repository implements the theoretical framework connecting Binary Neural Networks to SCRFP and Mixture of Experts through discrete path decomposition.

## Overview

Binary Neural Networks (BNNs) are formalized as discrete path-sum models that:
- Route inputs through exponentially many possible neuron combinations
- Act as implicit Mixture of Experts with binary gating
- Connect to SCRFP routing principles
- Exhibit interpretable activation patterns

## Structure

```
code/
├── src/                      # Core BNN implementation
│   ├── binary_layer.py      # BinaryLinear layer
│   ├── bnn_model.py         # Full BNN models
│   ├── routing.py           # Routing pattern analysis
│   └── visualization.py      # Visualization utilities
├── experiments/             # Training and analysis scripts
│   ├── train_mnist.py       # MNIST training experiment
│   └── analyze_routing.py    # Routing pattern analysis
├── visualizations/          # Generated plots and heatmaps
├── docs/                     # Documentation
│   ├── WALKTHROUGH.md       # Step-by-step guide
│   └── ARCHITECTURE.md      # Design details
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the MNIST experiment
python experiments/train_mnist.py

# Analyze routing patterns
python experiments/analyze_routing.py

# See docs/WALKTHROUGH.md for detailed instructions
```

## Key Components

### 1. Binary Layer (`src/binary_layer.py`)
- `BinaryLinear`: Fully connected layer with binary weights
- Forward pass: sign activation on weighted sum

### 2. BNN Model (`src/bnn_model.py`)
- `SimpleBNN`: 3-layer BNN for MNIST
- Training loop with standard supervised learning
- Routing pattern tracking

### 3. Routing Analysis (`src/routing.py`)
- Extract activation patterns from a batch
- Compute Jaccard similarity between routing patterns
- Build similarity matrices and graphs

### 4. Visualizations (`src/visualization.py`)
- Layer activation heatmaps
- Similarity matrices
- t-SNE embeddings
- Expert activation histograms

## Experiments

### MNIST Classification
Train a small BNN on MNIST and analyze its routing behavior.

**Expected results:**
- Test accuracy: 95%+
- Clear clustering in routing space
- Sparse activations (30–50% of neurons)
- Emergent expert-like specialization

### Routing Pattern Analysis
Visualize how the network uses different paths for different classes.

## Expected Outputs

After running experiments, find:
- `visualizations/activation_heatmaps_*.png` - Layer-wise activation patterns
- `visualizations/similarity_matrix.png` - Jaccard similarity between samples
- `visualizations/tsne_routing.png` - t-SNE embedding of routing vectors
- `visualizations/expert_histogram.png` - Expert activation frequency
- `results/metrics.json` - Accuracy, sparsity, and routing statistics

## References

- BNNs: Courbariaux et al., "Binarized Neural Networks" (2016)
- MoE: Shazeer et al., "Outrageously Large Neural Networks" (2017)
- SCRFP: [Your framework paper]

## Author Notes

This codebase bridges quantized neural networks and modular computation, 
exposing the implicit routing structure that emerges from binarization.

