"""
MNIST Training Experiment

Train a Binary Neural Network on MNIST and analyze routing patterns.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bnn_model import SimpleBNN
from src.routing import (
    get_activation_pattern, 
    get_routing_sparsity, 
    get_routing_diversity
)


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 64
    epochs = 50
    learning_rate = 0.01
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Model initialization
    print("Initializing BNN model...")
    model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    train_losses = []
    test_accuracies = []
    routing_stats = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        sparsity_values = []
        diversity_values = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                
                # Forward pass with activations
                outputs, activations = model.forward_with_activations(images)
                
                # Accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Routing statistics
                for act in activations:
                    sparsity_values.append(get_routing_sparsity(act))
                    diversity_values.append(get_routing_diversity(act))
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        avg_sparsity = sum(sparsity_values) / len(sparsity_values) if sparsity_values else 0
        avg_diversity = sum(diversity_values) / len(diversity_values) if diversity_values else 0
        
        routing_stats.append({
            'epoch': epoch + 1,
            'sparsity': avg_sparsity,
            'diversity': avg_diversity
        })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={test_accuracy:.2f}%, "
                  f"Sparsity={avg_sparsity:.4f}, Diversity={avg_diversity:.4f}")
    
    # Save model
    model_path = output_dir / 'bnn_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics = {
        'final_accuracy': float(test_accuracies[-1]),
        'final_sparsity': routing_stats[-1]['sparsity'],
        'final_diversity': routing_stats[-1]['diversity'],
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'routing_stats': routing_stats
    }
    
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    print("\nFinal Results:")
    print(f"  Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"  Routing Sparsity: {routing_stats[-1]['sparsity']:.4f}")
    print(f"  Routing Diversity: {routing_stats[-1]['diversity']:.4f}")


if __name__ == '__main__':
    main()
