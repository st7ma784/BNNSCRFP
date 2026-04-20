"""
Main analysis script: Circuit collapse under precision reduction and SCRFP implications.

Orchestrates:
1. Data collection (quantize at different precisions, collect activations)
2. Metric computation (routing diversity, sparsity, specialization, etc.)
3. Visualization generation (all 10 visualizations)
4. Results saving and interpretation
"""

import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bnn_model import SimpleBNN
from src.quantization import QuantizedModel, get_quantized_model
from src.routing import (
    get_activation_pattern,
    compute_similarity_matrix,
    get_routing_sparsity
)
from src.collapse_metrics import (
    unique_pattern_count,
    routing_entropy,
    path_convergence_histogram,
    gini_coefficient,
    class_wise_similarity,
    sparsity_metric,
    gate_efficiency,
    path_determinism,
    compute_all_metrics
)
from src.collapse_visualizations import generate_all_visualizations


class PrecisionCollapseAnalyzer:
    """Main analyzer for circuit collapse across precision levels."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Path to trained BNN model
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Precision levels to analyze
        self.precisions = [32, 16, 8, 4, 2, 1]
        
        # Data storage
        self.metrics_dict = {}
        self.activations_dict = {}
        self.similarities_dict = {}
        self.labels_dict = {}
        self.specialization_dict = {}
        self.stability_dict = {}
        self.waterfall_data = {}
    
    def load_test_data(self, batch_size=64, num_samples=None):
        """Load MNIST test set."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if num_samples:
            test_dataset = Subset(test_dataset, range(min(num_samples, len(test_dataset))))
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return test_loader
    
    def collect_activations(self, test_loader):
        """
        Phase 1: Collect activations and metrics for each precision level.
        """
        print("\n" + "="*80)
        print("PHASE 1: DATA COLLECTION")
        print("="*80)
        
        for precision in tqdm(self.precisions, desc="Precisions"):
            print(f"\n  Collecting data for {precision}-bit precision...")
            
            # Create quantized model
            quantized_model = get_quantized_model(self.model, bits=precision)
            quantized_model.to(self.device)
            quantized_model.eval()
            
            # Collect activations
            all_activations = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="    Batches", leave=False):
                    images = images.view(images.size(0), -1).to(self.device)
                    labels = labels.numpy()
                    
                    _, activations = quantized_model.forward_with_activations(images)
                    
                    all_activations.append([a.cpu() for a in activations])
                    all_labels.extend(labels)
            
            # Stack by layer
            num_layers = len(all_activations[0])
            layers_activations = []
            for layer_idx in range(num_layers):
                layer_acts = torch.cat([batch[layer_idx] for batch in all_activations], dim=0)
                layers_activations.append(layer_acts)
            
            self.activations_dict[precision] = layers_activations
            self.labels_dict[precision] = np.array(all_labels)
            
            print(f"    Collected {len(all_labels)} samples across {num_layers} layers")
    
    def compute_metrics(self):
        """
        Phase 2: Compute all routing metrics.
        """
        print("\n" + "="*80)
        print("PHASE 2: METRIC COMPUTATION")
        print("="*80)
        
        for precision in tqdm(self.precisions, desc="Computing metrics"):
            print(f"\n  Computing metrics for {precision}-bit precision...")
            
            metrics = {}
            acts_list = self.activations_dict[precision]
            labels = self.labels_dict[precision]
            
            # Per-layer metrics
            for layer_idx, acts in enumerate(acts_list):
                prefix = f"layer_{layer_idx + 1}"
                
                metrics[f"{prefix}_unique_patterns"] = unique_pattern_count(acts)
                metrics[f"{prefix}_entropy"] = routing_entropy(acts)
                metrics[f"{prefix}_sparsity"] = sparsity_metric(acts)
                metrics[f"{prefix}_gate_efficiency"] = gate_efficiency(acts)
                
                pattern_freqs = path_convergence_histogram(acts)
                metrics[f"{prefix}_gini"] = gini_coefficient(pattern_freqs)
                
                # Class-wise similarity
                class_sims = class_wise_similarity(acts, torch.from_numpy(labels))
                for c, sim in class_sims.items():
                    metrics[f"{prefix}_class_{c}_similarity"] = sim
            
            self.metrics_dict[precision] = metrics
    
    def compute_similarities(self):
        """
        Phase 2b: Compute Jaccard similarity matrices.
        """
        print("\n  Computing similarity matrices...")
        
        for precision in tqdm(self.precisions, desc="Similarities"):
            acts = self.activations_dict[precision][0]  # Use layer 1
            sim_matrix = compute_similarity_matrix(acts)
            self.similarities_dict[precision] = sim_matrix
    
    def compute_specialization(self):
        """
        Phase 2c: Compute neuron-class specialization matrices.
        """
        print("\n  Computing specialization matrices...")
        
        for precision in tqdm(self.precisions, desc="Specialization"):
            acts = self.activations_dict[precision][0]  # Layer 1
            labels = self.labels_dict[precision]
            
            # Get top 10 most active neuron combinations
            num_classes = 10
            num_neuron_subsets = 10
            
            # Create specialization matrix (neuron subsets x classes)
            specialization = np.zeros((num_neuron_subsets, num_classes))
            
            # For each neuron subset, compute which classes use it
            batch_size = acts.shape[0]
            acts_binary = (acts == 1).float().numpy()
            
            # Get top neurons by activation frequency
            neuron_freqs = acts_binary.sum(axis=0)
            top_neurons = np.argsort(neuron_freqs)[-num_neuron_subsets:]
            
            for subset_idx, neuron in enumerate(top_neurons):
                neuron_active = acts_binary[:, neuron] > 0.5
                for c in range(num_classes):
                    class_mask = labels == c
                    if np.sum(class_mask) > 0:
                        specialization[subset_idx, c] = np.mean(neuron_active[class_mask])
            
            self.specialization_dict[precision] = specialization
    
    def compute_stability(self, test_loader):
        """
        Phase 2d: Compute path determinism (stability under repeated forwards).
        """
        print("\n  Computing path stability...")
        
        for precision in tqdm(self.precisions, desc="Stability"):
            quantized_model = get_quantized_model(self.model, bits=precision)
            quantized_model.to(self.device)
            quantized_model.eval()
            
            # Get first batch
            for images, _ in test_loader:
                images = images.view(images.size(0), -1).to(self.device)
                det = path_determinism(quantized_model, images, num_runs=5)
                break
            
            self.stability_dict[precision] = {'determinism': det}
    
    def compute_waterfall(self):
        """
        Phase 2e: Compute information loss waterfall.
        """
        print("\n  Computing efficiency waterfall...")
        
        # Use float32 as baseline (100%)
        baseline_capacity = self.metrics_dict[32]['layer_1_gate_efficiency']
        
        # Estimate losses
        loss_quantization = (self.metrics_dict[32]['layer_1_gate_efficiency'] - 
                            self.metrics_dict[16]['layer_1_gate_efficiency']) * 100
        loss_duplication = (self.metrics_dict[16]['layer_1_gate_efficiency'] - 
                           self.metrics_dict[8]['layer_1_gate_efficiency']) * 100
        loss_collapse = (self.metrics_dict[8]['layer_1_gate_efficiency'] - 
                        self.metrics_dict[1]['layer_1_gate_efficiency']) * 100
        final_capacity = self.metrics_dict[1]['layer_1_gate_efficiency'] * 100
        
        self.waterfall_data = {
            'values': [baseline_capacity * 100, -loss_quantization, -loss_duplication,
                      -loss_collapse, final_capacity]
        }
    
    def generate_visualizations(self, output_dir='visualizations'):
        """
        Phase 3: Generate all 10 visualizations.
        """
        print("\n" + "="*80)
        print("PHASE 3: VISUALIZATION GENERATION")
        print("="*80)
        
        data_dict = {
            'metrics': self.metrics_dict,
            'activations': self.activations_dict,
            'similarities': self.similarities_dict,
            'labels': self.labels_dict,
            'specialization': self.specialization_dict,
            'stability': self.stability_dict,
            'waterfall': self.waterfall_data
        }
        
        generate_all_visualizations(data_dict, output_dir)
    
    def save_results(self, output_dir='results'):
        """Save all data and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("PHASE 4: SAVING RESULTS")
        print("="*80)
        
        # Save metrics
        metrics_path = output_path / 'precision_collapse_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy values to JSON-serializable types
            metrics_json = {}
            for p, m in self.metrics_dict.items():
                metrics_json[str(p)] = {k: float(v) if isinstance(v, (np.number, float)) else v
                                       for k, v in m.items()}
            json.dump(metrics_json, f, indent=2)
        
        # Save detailed data
        data_path = output_path / 'precision_collapse_data.pkl'
        with open(data_path, 'wb') as f:
            pickle.dump({
                'metrics': self.metrics_dict,
                'similarities': {str(k): v for k, v in self.similarities_dict.items()},
                'stability': self.stability_dict,
                'specialization': {str(k): v for k, v in self.specialization_dict.items()}
            }, f)
        
        print(f"  Metrics saved to {metrics_path}")
        print(f"  Data saved to {data_path}")
    
    def print_summary(self):
        """Print summary of findings."""
        print("\n" + "="*80)
        print("SUMMARY: CIRCUIT COLLAPSE ANALYSIS")
        print("="*80)
        
        print("\nRouting Diversity (Unique Patterns per Layer 1):")
        for p in self.precisions:
            unique = self.metrics_dict[p]['layer_1_unique_patterns']
            print(f"  {p:2d}-bit: {unique:5d} patterns")
        
        print("\nRouting Entropy (bits):")
        for p in self.precisions:
            ent = self.metrics_dict[p]['layer_1_entropy']
            print(f"  {p:2d}-bit: {ent:6.2f} bits")
        
        print("\nGate Efficiency:")
        for p in self.precisions:
            eff = self.metrics_dict[p]['layer_1_gate_efficiency']
            print(f"  {p:2d}-bit: {eff:6.2%}")
        
        print("\nNeuron Sparsity:")
        for p in self.precisions:
            sparsity = self.metrics_dict[p]['layer_1_sparsity']
            print(f"  {p:2d}-bit: {sparsity:6.2%} inactive")
        
        print("\nPath Determinism (SCRFP Stability):")
        for p in self.precisions:
            det = self.stability_dict[p]['determinism']
            print(f"  {p:2d}-bit: {det:6.2%} samples deterministic")
        
        print("\nKey Insights:")
        div_collapse = (self.metrics_dict[32]['layer_1_unique_patterns'] / 
                       self.metrics_dict[1]['layer_1_unique_patterns'])
        print(f"  - Routing diversity collapsed by {div_collapse:.1f}x from full to binary precision")
        
        eff_loss = ((self.metrics_dict[32]['layer_1_gate_efficiency'] - 
                    self.metrics_dict[1]['layer_1_gate_efficiency']) * 100)
        print(f"  - Gate efficiency lost {eff_loss:.1%} points across precision reduction")
        
        most_stable_precision = max(self.stability_dict.keys(),
                                   key=lambda p: self.stability_dict[p]['determinism'])
        print(f"  - Most stable precision for SCRFP: {most_stable_precision}-bit")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("CIRCUIT COLLAPSE UNDER PRECISION REDUCTION ANALYSIS")
    print("="*80)
    
    # Check if trained model exists
    model_path = Path(__file__).parent.parent / 'results' / 'bnn_mnist_model.pth'
    
    if not model_path.exists():
        print(f"\nError: Trained model not found at {model_path}")
        print("Please run: python experiments/train_mnist.py")
        return
    
    # Initialize analyzer
    analyzer = PrecisionCollapseAnalyzer(str(model_path))
    
    # Load test data (use full test set for comprehensive analysis)
    test_loader = analyzer.load_test_data(batch_size=64)
    
    # Run analysis
    analyzer.collect_activations(test_loader)
    analyzer.compute_metrics()
    analyzer.compute_similarities()
    analyzer.compute_specialization()
    analyzer.compute_stability(test_loader)
    analyzer.compute_waterfall()
    
    # Generate visualizations
    output_dir = Path(__file__).parent.parent / 'visualizations'
    analyzer.generate_visualizations(str(output_dir))
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    analyzer.save_results(str(results_dir))
    
    # Print summary
    analyzer.print_summary()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Results saved to: {results_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
