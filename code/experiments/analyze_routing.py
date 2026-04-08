"""
Routing Pattern Analysis

Load trained model and analyze routing patterns with visualizations.
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bnn_model import SimpleBNN
from src.visualization import plot_all_visualizations


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_dir = Path(__file__).parent.parent / 'results'
    model_path = model_dir / 'bnn_mnist_model.pth'
    viz_dir = Path(__file__).parent.parent / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please run train_mnist.py first")
        return
    
    # Load model
    print("Loading model...")
    model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load test data
    print("Loading test data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Generate visualizations
    print("Generating visualizations...")
    print("(This may take a few minutes)")
    
    plot_all_visualizations(model, test_loader, output_dir=str(viz_dir))
    
    print(f"\nVisualizations saved to {viz_dir}")
    print("Generated files:")
    print("  - activation_heatmap_layer*.png")
    print("  - similarity_matrix_layer*.png")
    print("  - tsne_routing_layer*.png")
    print("  - expert_histogram_layer*.png")


if __name__ == '__main__':
    main()
