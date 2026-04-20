"""
Quick synthetic model for precision collapse demonstration.

Trains a model quickly for testing the full precision collapse pipeline.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bnn_model import SimpleBNN


def main():
    print("Using device: cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data for quick testing
    print("Generating synthetic data...")
    num_train = 6000
    num_test = 1000
    input_dim = 784
    num_classes = 10
    
    X_train = torch.randn(num_train, input_dim)
    y_train = torch.randint(0, num_classes, (num_train,))
    
    X_test = torch.randn(num_test, input_dim)
    y_test = torch.randint(0, num_classes, (num_test,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Model initialization
    print("Initializing BNN model...")
    model = SimpleBNN(input_dim=784, hidden_dim=256, num_classes=10)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop (10 epochs for speed)
    epochs = 10
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for images, labels in train_loader:
            images = images.to(device)
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
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Accuracy={test_accuracy:.2f}%")
    
    # Save model
    model_path = output_dir / 'bnn_mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
