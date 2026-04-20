# Visualization Plan: Circuit Collapse in Low Precision and SCRFP Intersection

## Executive Summary

This document outlines a comprehensive suite of visualizations to illustrate how BNN routing circuits collapse as precision decreases, and how SCRFP properties (sparse, fixed-path routing) degrade during quantization.

---

## Part 1: Circuit Collapse Visualizations

### 1.1 Routing Diversity Collapse (Core Metric)

**Purpose:** Show how the number of *distinct routing patterns* decreases as precision drops.

**Visualization Type:** Multi-panel line plot with confidence bands

**Precision Levels:** float32 → float16 → int8 → binary

**Metrics:**
- **Y-axis (left):** Number of unique activation patterns per layer (normalized to max observed)
- **Y-axis (right):** Shannon entropy of routing distribution
- **X-axis:** Model layers (L1, L2, L3)

**What it reveals:**
- Clear "collapse point" where precision drop causes dramatic loss of routing diversity
- Which layers maintain routing patterns longest (typically early layers)
- Trade-off between precision and routing capacity

**Implementation approach:**
```python
# For each precision level:
# 1. Forward entire test set
# 2. Record activation patterns
# 3. Count unique combinations: len(set(tuple(a) for a in activations))
# 4. Compute entropy: -sum(p*log(p)) for histogram
```

---

### 1.2 Path Convergence Heatmap

**Purpose:** Visualize how distinct paths merge as precision drops.

**Visualization Type:** 3D heatmap array (one heatmap per precision level, arranged left-right)

**Dimensions:**
- X-axis: Neuron index (layer 1 or 2)
- Y-axis: Sample index (32 representative samples from test set)
- Z-axis (color): Count of samples following this exact pattern
- Multiple panels for: float32, float16, int8, binary

**What it reveals:**
- Patterns become "fatter" (more samples per path)
- Critical neurons become bottlenecks
- Emergence of "default" paths that most inputs follow

**Comparison metric:**  
Compare Gini coefficient of path frequency distribution: 
- High Gini (float32) = paths unevenly distributed
- Low Gini (binary) = many samples forced to same path

---

### 1.3 Jaccard Similarity Distribution Cascades

**Purpose:** Show how within-sample similarity changes across precisions.

**Visualization Type:** Violin plots or ridge plots (one per precision level)

**Data:**
- Compute pairwise Jaccard similarity for all test samples
- Separate by class (10 colors for MNIST digits)
- Pool into distribution

**Y-axis:** Similarity value (0 to 1)  
**X-axis:** Precision level (float32 → binary)  
**Color:** Class label

**What it reveals:**
- Within-class similarity increases (paths converge to class-specific "modes")
- Between-class similarity also increases (but more slowly)
- Critical precision threshold where routing becomes undifferentiated

---

### 1.4 Routing Collapse Phase Diagram

**Purpose:** Operational characterization of circuit failure.

**Visualization Type:** 2D scatter plot with regions

**Axes:**
- X-axis: Precision bits (32 → 16 → 8 → 1, log scale)
- Y-axis: Routing disorder metric (`1 - normalized_entropy`)

**Points:** Average per layer, colored by layer depth

**Overlay:** Three regions
- **Green zone** (high precision): "Stable routing" - distinct paths maintained
- **Yellow zone** (medium): "Degraded routing" - paths start merging
- **Red zone** (low precision): "Collapsed routing" - most samples on 1-2 paths

**What it reveals:**
- Precise threshold values for routing stability
- Which layers collapse first
- Extrapolate: would layer N recover with slightly higher precision?

---

## Part 2: SCRFP Intersection Visualizations

### 2.1 Sparsity vs Precision Trade-off

**Purpose:** How SCRFP's core property (sparse activation) is affected by quantization.

**Visualization Type:** Two-panel plot

**Left panel - Activation sparsity:**
- X-axis: Precision level
- Y-axis: % of neurons active (across entire test set)
- Line per layer
- Theoretical SCRFP target (e.g., 20% active) as horizontal line

**Right panel - Gating efficiency:**
- Compute: (# unique gates / total possible gates)
- Shows whether binary gating remains "efficient" or becomes random

**What it reveals:**
- Whether low precision increases or decreases sparsity
- Does the network compensate for lost precision by activating more neurons?
- How far is the network from ideal SCRFP (sparse gates, clear paths)?

---

### 2.2 Gate Specialization Degradation

**Purpose:** Visualize how SCRFP's implicit "expert" specialization breaks down.

**Visualization Type:** Confusion matrix progression

**For each precision level:**
- Compute: Which (class, layer, neuron subset) combinations co-occur?
- Build a "specialization score" matrix
- Rows: neuron subsets (top 10 most active)
- Columns: digit classes (MNIST: 0-9)

**Visualization:**
- Show matrices side-by-side: float32 → float16 → int8 → binary
- Use color intensity = specialization (dark = one class only, light = all classes)

**What it reveals:**
- How neurons lose class-specific gating behavior
- Which neurons remain specialized even at low precision
- Emergence of "jack-of-all-trades" neurons that serve all classes

---

### 2.3 Fixed Path Stability (SCRFP Compliance)

**Purpose:** Check whether SCRFP's assumption (paths are "fixed" for a sample) holds under quantization.

**Visualization Type:** Multi-layer pathway diagram

**Approach:**
- For each sample, identify its "routing path" (chain of active neurons across layers)
- Repeat forward pass 5 times for each precision level
- Check: do we get the same path each time? (determinism)

**Visualization:**
- Heatmap: X-axis = precision level, Y-axis = % of samples with deterministic path
- Add scatter: size = # resamples needed for path to change

**What it reveals:**
- Quantization noise → non-deterministic routing
- At what precision do we lose "fixed path" guarantee?
- How stochastic does routing become in binary networks?

---

### 2.4 Router Efficiency: Bits per Expertly-Routed Sample

**Purpose:** Information-theoretic view of SCRFP under precision constraints.

**Visualization Type:** Waterfall/sankey diagram

**Layers:**
1. Start: Total information in full-precision model
2. Arrow down: Information lost per binary layer (Shannon capacity)
3. Arrow down: Information lost to quantization error
4. Arrow down: Information lost to duplicate paths
5. End: Effective routing capacity in binary network

**Overlays:**
- Compare float32 vs binary
- Show where most information leaks

**What it reveals:**
- Quantitative view of "where" capacity is lost
- Identifies whether bottleneck is precision or architecture
- Whether alternative routing strategies (e.g., stochastic paths) could help

---

## Part 3: Integrated Collapse-SCRFP Visualizations

### 3.1 The "Routing Capability Pyramid"

**Purpose:** Comprehensive single visualization showing circuit collapse through SCRFP lens.

**Visualization Type:** Stacked 3D bar chart or pyramid

**Dimensions:**
- **Base (bottom):** float32 network - shows maximum routing capacity
- **Tiers (upward):** float16, int8, binary
- **Height of each tier:** Effective routing capacity relative to float32
- **Color segregation:** Separate colors for "sparse paths" vs "collapsed paths"

**Annotations:**
- SCRFP region: "efficiently routed samples"
- Collapse region: "forced into default paths"

**What it reveals:**
- At a glance: how much capability is lost at each precision step
- Where SCRFP assumptions break down
- Extrapolation: what precision would we need for 90% original capacity?

---

### 3.2 Interactive Precision Slider Animation

**Purpose:** Temporal visualization of collapse in action.

**Format:** Animated sequence (gif or video)

**Frames (one per precision level):**
- Layer 1 activation heatmap
- Layer 2 activation heatmap
- Similarity matrix
- Routing capacity gauge (needle pointing left = collapsed)

**Animation speed:** 1 second per frame, pause at binary

**What it reveals:**
- Intuitive grasp of "smooth collapse" vs "sudden failure"
- Critical transition points
- Visual confirmation of theory

---

## Part 4: Implementation Roadmap

### Phase 1: Data Collection (1-2 hours)
```
For each precision in [32, 16, 8, 4, 2, 1] bits:
  - Quantize model weights
  - Forward pass entire test set (10,000 samples)
  - Record: activations, predictions, routing patterns
  - Save to .pkl for analysis
```

### Phase 2: Metric Computation (2-3 hours)
```
For each precision-layer pair:
  - Unique path count
  - Shannon entropy
  - Jaccard similarity matrix
  - Class-specific statistics
  - Sparsity metrics
```

### Phase 3: Visualization Generation (3-4 hours)
```
- 1.1: Diversity collapse plot
- 1.2: Path convergence heatmaps
- 1.3: Similarity cascades
- 1.4: Phase diagram
- 2.1: Sparsity trade-off
- 2.2: Gate specialization matrices
- 2.3: Path stability heatmap
- 2.4: Waterfall diagram
- 3.1: Pyramid summary
- 3.2: Animated sequence
```

### Phase 4: Interpretation & Publication (1-2 hours)
```
- Caption each visualization with key insights
- Cross-reference with theory
- Create summary "viewer's guide"
```

---

## Key Research Questions Addressed

1. **When does the network "give up" on routing?**  
   → Visualizations 1.1, 1.4 answer this

2. **Does low precision force all samples onto a few paths?**  
   → Visualizations 1.2, 1.3 answer this

3. **Does SCRFP remain valid for low-precision networks?**  
   → Visualizations 2.1, 2.3, 3.1 answer this

4. **Which layers are most sensitive to quantization?**  
   → Visualizations 1.1, 2.2 answer this

5. **Is there a "sweet spot" precision for SCRFP compliance?**  
   → Visualizations 1.4, 2.3, 2.4 answer this

---

## Notes

- **Color schemes:** Use diverging colormaps for "active/inactive" (RdYlGn). Use sequential for metrics (viridis).
- **Consistency:** All visualizations should use the same precision color coding (e.g., blue=float32, orange=int8, red=binary).
- **Reproducibility:** Save all raw data (activations, patterns) in `results/precision_analysis/` for reproducibility.
- **Interactivity:** Consider Plotly for hover details on complex plots (e.g., viewing exact path for a hovered sample).

