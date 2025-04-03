import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import argparse
import time
import os

class DifferentiableKernel(nn.Module):
    """
    Differentiable approximation of kernel method using Random Fourier Features.
    """
    def __init__(self, input_dim, output_dim, gamma=0.1, n_components=1024):
        super(DifferentiableKernel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.gamma = gamma
        
        # Random Fourier Features for RBF kernel
        self.register_buffer('random_weights', 
                            torch.randn(input_dim, n_components) * np.sqrt(2 * gamma))
        self.register_buffer('random_offset', 
                            torch.rand(n_components) * 2 * np.pi)
        
        # Learnable projection to output dimension
        self.projection = nn.Linear(n_components, output_dim)
        
        # Optional support vector parameterization for SVR-like behavior
        self.sv_weights = nn.Parameter(torch.zeros(output_dim, n_components))
        self.sv_bias = nn.Parameter(torch.zeros(output_dim))
        
    def kernel_features(self, x):
        # Project input to random space
        projection = torch.mm(x, self.random_weights) + self.random_offset
        # Apply cosine as per Random Fourier Features
        kernel_feat = torch.cos(projection) * np.sqrt(2.0 / self.n_components)
        return kernel_feat
        
    def forward(self, x):
        # Apply kernel transformation
        features = self.kernel_features(x)
        # Project to output space
        output = self.projection(features)
        
        # Add SVR-like component
        sv_output = F.linear(features, self.sv_weights, self.sv_bias)
        
        # Combine both outputs
        return output + sv_output

class JointKStoNet(nn.Module):
    """
    Joint Kernel-Expanded Stochastic Neural Network.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=0.1, 
                n_components=1024, dropout_rate=0.1, use_batchnorm=True):
        super(JointKStoNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        
        # Kernel layer
        self.kernel_layer = DifferentiableKernel(
            input_dim, hidden_dims[0], gamma, n_components)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.mlp_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x, return_features=False):
        # Pass through kernel layer
        h = self.kernel_layer(x)
        
        # Store features for regularization if needed
        features = [h]
        
        # Apply activation
        h = torch.tanh(h)
        
        # Pass through MLP layers
        for i, layer in enumerate(self.mlp_layers):
            h = layer(h)
            if self.use_batchnorm:
                h = self.bn_layers[i](h)
            h = torch.tanh(h)
            h = self.dropout_layers[i](h)
            features.append(h)
        
        # Output layer
        output = self.output_layer(h)
        
        if return_features:
            return output, features
        return output

    def l1_regularization(self):
        """Calculate L1 regularization term for sparsity"""
        l1_reg = 0
        for param in self.parameters():
            l1_reg += torch.sum(torch.abs(param))
        return l1_reg
    
    def mc_dropout_predict(self, x, num_samples=10):
        """Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # Enable dropout during inference
        
        predictions = []
        for _ in range(num_samples):
            predictions.append(self(x).unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        
        # Mean prediction and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_pred, uncertainty

def train_epoch(model, train_loader, optimizer, criterion, l1_lambda=0.0001, device='cuda'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        # Standard loss
        loss = criterion(outputs, targets)
        
        # Add L1 regularization for sparsity (like LASSO)
        if l1_lambda > 0:
            l1_loss = model.l1_regularization()
            loss += l1_lambda * l1_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # For classification tasks
        if outputs.size(1) > 1:  # If not regression
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, l1_lambda=0.0001, device='cuda'):
    """Validate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # Standard loss
            loss = criterion(outputs, targets)
            
            # Add L1 regularization for sparsity (like LASSO)
            if l1_lambda > 0:
                l1_loss = model.l1_regularization()
                loss += l1_lambda * l1_loss
            
            running_loss += loss.item() * inputs.size(0)
            
            # For classification tasks
            if outputs.size(1) > 1:  # If not regression
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def test_uncertainty(model, test_loader, device='cuda'):
    """Test with uncertainty estimation"""
    model.eval()
    all_uncertainties = []
    all_correct = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions with uncertainty
            mean_preds, uncertainties = model.mc_dropout_predict(inputs)
            
            _, predicted = mean_preds.max(1)
            correct = predicted.eq(targets)
            
            # Store uncertainty and correctness
            all_uncertainties.extend(uncertainties.mean(1).cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
    
    # Analyze relationship between uncertainty and correctness
    all_uncertainties = np.array(all_uncertainties)
    all_correct = np.array(all_correct)
    
    # Sort by uncertainty
    sort_indices = np.argsort(all_uncertainties)
    sorted_uncertainties = all_uncertainties[sort_indices]
    sorted_correct = all_correct[sort_indices]
    
    # Compute cumulative accuracy at different uncertainty thresholds
    cumulative_accuracies = []
    thresholds = np.linspace(0, np.max(all_uncertainties), 100)
    
    for threshold in thresholds:
        mask = all_uncertainties <= threshold
        if np.sum(mask) > 0:
            acc = np.mean(all_correct[mask])
            cumulative_accuracies.append(acc)
        else:
            cumulative_accuracies.append(1.0)  # No samples below threshold
    
    return {
        'uncertainties': all_uncertainties,
        'correct': all_correct,
        'thresholds': thresholds,
        'cumulative_accuracies': cumulative_accuracies
    }

def plot_uncertainty_vs_accuracy(uncertainty_results, save_path=None):
    """Plot the relationship between uncertainty and accuracy"""
    plt.figure(figsize=(10, 6))
    
    # Plot uncertainty histogram for correct and incorrect predictions
    plt.subplot(1, 2, 1)
    correct_uncertainties = uncertainty_results['uncertainties'][uncertainty_results['correct']]
    incorrect_uncertainties = uncertainty_results['uncertainties'][~uncertainty_results['correct']]
    
    plt.hist(correct_uncertainties, alpha=0.5, bins=50, label='Correct Predictions')
    plt.hist(incorrect_uncertainties, alpha=0.5, bins=50, label='Incorrect Predictions')
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Uncertainty Distribution')
    
    # Plot cumulative accuracy vs uncertainty threshold
    plt.subplot(1, 2, 2)
    plt.plot(uncertainty_results['thresholds'], uncertainty_results['cumulative_accuracies'])
    plt.xlabel('Uncertainty Threshold')
    plt.ylabel('Accuracy for Samples Below Threshold')
    plt.title('Accuracy vs Uncertainty Threshold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_mnist_data(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='Joint K-StoNet for MNIST')
    
    # Model parameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[100, 50],
                        help='Hidden dimensions for MLP layers')
    parser.add_argument('--n-components', type=int, default=1024,
                        help='Number of random Fourier features')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='RBF kernel parameter gamma')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--no-batchnorm', action='store_false', dest='use_batchnorm',
                        help='Disable batch normalization')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--l1-lambda', type=float, default=0.0001,
                        help='L1 regularization coefficient')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay (L2 regularization)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load MNIST data
    train_loader, val_loader, test_loader = load_mnist_data(args.batch_size)
    
    # Create model
    model = JointKStoNet(
        input_dim=28*28,  # MNIST image size
        hidden_dims=args.hidden_dims,
        output_dim=10,  # 10 classes for MNIST
        gamma=args.gamma,
        n_components=args.n_components,
        dropout_rate=args.dropout_rate,
        use_batchnorm=args.use_batchnorm
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, 
            l1_lambda=args.l1_lambda, device=device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, 
            l1_lambda=args.l1_lambda, device=device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pt")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    print(f"Training complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    # Load best model for testing
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt"))
    
    # Test with uncertainty estimation
    print("Testing with uncertainty estimation...")
    uncertainty_results = test_uncertainty(model, test_loader, device=device)
    
    # Plot and save uncertainty vs accuracy
    plot_uncertainty_vs_accuracy(
        uncertainty_results, 
        save_path=f"{args.save_dir}/uncertainty_vs_accuracy.png"
    )
    
    # Final evaluation on test set
    test_loss, test_acc = validate(
        model, test_loader, criterion, 
        l1_lambda=args.l1_lambda, device=device
    )
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Plot and save training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/training_curves.png")
    plt.show()
    
    # Save model hyperparameters and results
    results = {
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    torch.save(results, f"{args.save_dir}/results.pt")
    
    print(f"Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()