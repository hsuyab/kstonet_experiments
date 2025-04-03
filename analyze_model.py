import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from main import JointKStoNet

def load_model_and_data(model_path, batch_size=128):
    """Load the trained model and MNIST test data"""
    # Load MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model config from saved results
    results = torch.load(f"{os.path.dirname(model_path)}/results.pt")
    args = results['args']

    # Create model with same config
    model = JointKStoNet(
        input_dim=28*28,
        hidden_dims=args['hidden_dims'],
        output_dim=10,
        gamma=args['gamma'],
        n_components=args['n_components'],
        dropout_rate=args['dropout_rate'],
        use_batchnorm=args['use_batchnorm']
    )

    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, test_loader, test_dataset

def extract_features(model, test_loader, device='cuda'):
    """Extract features from different layers of the model"""
    features_by_layer = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Flatten the input
            data = data.view(data.size(0), -1)

            # Get features from all layers
            outputs, all_features = model(data, return_features=True)

            for i, feat in enumerate(all_features):
                if len(features_by_layer) <= i:
                    features_by_layer.append([])
                features_by_layer[i].append(feat.cpu().numpy())

            labels.append(target.cpu().numpy())

            # Only process a subset for visualization
            if batch_idx >= 5:  # ~500-600 samples
                break

    # Concatenate batches
    features_by_layer = [np.concatenate(features) for features in features_by_layer]
    labels = np.concatenate(labels)

    return features_by_layer, labels

def visualize_features(features_by_layer, labels, save_dir):
    """Visualize features from different layers using t-SNE"""
    os.makedirs(save_dir, exist_ok=True)

    for i, features in enumerate(features_by_layer):
        # Use PCA to reduce dimensions before t-SNE for computational efficiency
        pca = PCA(n_components=min(50, features.shape[1]))
        features_pca = pca.fit_transform(features)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_pca)

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels,
                    alpha=0.6, cmap='tab10', s=5)
        plt.colorbar(scatter, label='Digit Class')
        plt.title(f'Layer {i} Features (t-SNE)')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/layer_{i}_tsne.png")
        plt.close()

def analyze_weight_sparsity(model, save_dir):
    """Analyze the sparsity of model weights"""
    os.makedirs(save_dir, exist_ok=True)

    layer_names = []
    sparsity_values = []
    weight_norms = []

    # Kernel layer SVR weights
    kernel_weights = model.kernel_layer.sv_weights.detach().cpu().numpy()
    sv_sparsity = np.mean(np.abs(kernel_weights) < 1e-3)
    layer_names.append('Kernel SVR')
    sparsity_values.append(sv_sparsity)
    weight_norms.append(np.linalg.norm(kernel_weights))

    # Kernel layer projection weights
    projection_weights = model.kernel_layer.projection.weight.detach().cpu().numpy()
    proj_sparsity = np.mean(np.abs(projection_weights) < 1e-3)
    layer_names.append('Kernel Projection')
    sparsity_values.append(proj_sparsity)
    weight_norms.append(np.linalg.norm(projection_weights))

    # MLP layers
    for i, layer in enumerate(model.mlp_layers):
        weights = layer.weight.detach().cpu().numpy()
        sparsity = np.mean(np.abs(weights) < 1e-3)
        layer_names.append(f'MLP Layer {i+1}')
        sparsity_values.append(sparsity)
        weight_norms.append(np.linalg.norm(weights))

    # Output layer
    output_weights = model.output_layer.weight.detach().cpu().numpy()
    out_sparsity = np.mean(np.abs(output_weights) < 1e-3)
    layer_names.append('Output Layer')
    sparsity_values.append(out_sparsity)
    weight_norms.append(np.linalg.norm(output_weights))

    # Plot sparsity
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(layer_names, sparsity_values)
    plt.ylabel('Sparsity (% of near-zero weights)')
    plt.title('Weight Sparsity by Layer')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(layer_names, weight_norms)
    plt.ylabel('Frobenius Norm')
    plt.title('Weight Norms by Layer')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/weight_sparsity.png")
    plt.close()

    # Plot weight distributions
    plt.figure(figsize=(15, 10))

    # Kernel layer
    plt.subplot(3, 2, 1)
    plt.hist(kernel_weights.flatten(), bins=50)
    plt.title('Kernel SVR Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.subplot(3, 2, 2)
    plt.hist(projection_weights.flatten(), bins=50)
    plt.title('Kernel Projection Weights')
    plt.xlabel('Weight Value')

    # Sample MLP layers
    if len(model.mlp_layers) > 0:
        plt.subplot(3, 2, 3)
        first_mlp = model.mlp_layers[0].weight.detach().cpu().numpy()
        plt.hist(first_mlp.flatten(), bins=50)
        plt.title('First MLP Layer Weights')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')

    if len(model.mlp_layers) > 1:
        plt.subplot(3, 2, 4)
        last_mlp = model.mlp_layers[-1].weight.detach().cpu().numpy()
        plt.hist(last_mlp.flatten(), bins=50)
        plt.title('Last MLP Layer Weights')
        plt.xlabel('Weight Value')

    # Output layer
    plt.subplot(3, 2, 5)
    plt.hist(output_weights.flatten(), bins=50)
    plt.title('Output Layer Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/weight_distributions.png")
    plt.close()

    return {
        'layer_names': layer_names,
        'sparsity_values': sparsity_values,
        'weight_norms': weight_norms
    }

def analyze_uncertainty(model, test_dataset, save_dir, num_samples=30):
    """Analyze model uncertainty on specific examples"""
    os.makedirs(save_dir, exist_ok=True)

    # Select a few samples from each class
    samples_per_class = 2
    samples = []
    labels = []

    for class_idx in range(10):
        class_indices = np.where(np.array(test_dataset.targets) == class_idx)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for idx in selected_indices:
            samples.append(test_dataset[idx][0])
            labels.append(class_idx)

    samples = torch.stack(samples)
    labels = torch.tensor(labels)

    # Get predictions with uncertainty
    model.eval()
    all_predictions = []

    for _ in range(num_samples):
        with torch.no_grad():
            flattened_samples = samples.view(samples.size(0), -1)
            pred = model(flattened_samples)
            all_predictions.append(pred.unsqueeze(0))

    all_predictions = torch.cat(all_predictions, dim=0)

    # Calculate mean and variance
    mean_preds = torch.mean(all_predictions, dim=0)
    var_preds = torch.var(all_predictions, dim=0)

    # Convert to probabilities
    probs = torch.softmax(mean_preds, dim=1)

    # Get predicted classes
    _, predicted = torch.max(probs, 1)

    # Get uncertainty (entropy of the predicted distribution)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

    # Create visualization
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(len(samples)):
        # Original image
        ax = axes[i]
        img = samples[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')

        # Add prediction and uncertainty information
        true_class = labels[i].item()
        pred_class = predicted[i].item()
        uncertainty = entropy[i].item()

        title = f"True: {true_class}, Pred: {pred_class}\nUncertainty: {uncertainty:.3f}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/uncertainty_examples.png")
    plt.close()

    # Plot certainty vs correctness
    correct = (predicted == labels)

    plt.figure(figsize=(8, 6))
    # Plot certainty vs correctness
    correct = (predicted == labels)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(entropy[correct], [1] * sum(correct), color='green', label='Correct', alpha=0.7)
    plt.scatter(entropy[~correct], [0] * sum(~correct), color='red', label='Incorrect', alpha=0.7)
    plt.xlabel('Prediction Entropy (Uncertainty)')
    plt.ylabel('Prediction Correctness')
    plt.legend()
    plt.title('Relationship Between Uncertainty and Correctness')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_dir}/uncertainty_vs_correctness.png")
    plt.close()
    
    # Visualize per-class uncertainty
    class_uncertainties = []
    for class_idx in range(10):
        class_mask = labels == class_idx
        if sum(class_mask) > 0:
            class_uncertainties.append((class_idx, entropy[class_mask].mean().item()))
    
    class_indices, class_mean_uncertainties = zip(*class_uncertainties)
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_indices, class_mean_uncertainties)
    plt.xlabel('Digit Class')
    plt.ylabel('Mean Prediction Uncertainty')
    plt.title('Uncertainty by Class')
    plt.xticks(class_indices)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{save_dir}/uncertainty_by_class.png")
    plt.close()
    
    return {
        'entropy': entropy.numpy(),
        'correct': correct.numpy(),
        'class_uncertainties': class_uncertainties
    }

def analyze_kernel_activations(model, test_loader, save_dir, num_batches=5):
    """Analyze the activations of the kernel layer"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_activations = []
    all_labels = []
    
    # Collect activations from the kernel layer
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
                
            # Flatten input
            flattened_data = data.view(data.size(0), -1)
            
            # Get kernel features directly
            kernel_features = model.kernel_layer.kernel_features(flattened_data)
            all_activations.append(kernel_features.cpu().numpy())
            all_labels.append(target.numpy())
    
    # Concatenate activations and labels
    all_activations = np.concatenate(all_activations, axis=0)
    all_labels = np.concatenate(all_labels)
    
    # Analyze activation statistics
    activation_means = np.mean(all_activations, axis=0)
    activation_stds = np.std(all_activations, axis=0)
    
    # Plot activation statistics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(activation_means, bins=50)
    plt.xlabel('Mean Activation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Kernel Feature Means')
    
    plt.subplot(1, 2, 2)
    plt.hist(activation_stds, bins=50)
    plt.xlabel('Activation Standard Deviation')
    plt.title('Distribution of Kernel Feature Standard Deviations')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/kernel_activation_stats.png")
    plt.close()
    
    # Identify most active features
    mean_activations_by_class = []
    for class_idx in range(10):
        class_mask = all_labels == class_idx
        if sum(class_mask) > 0:
            mean_act = np.mean(all_activations[class_mask], axis=0)
            mean_activations_by_class.append(mean_act)
    
    mean_activations_by_class = np.array(mean_activations_by_class)
    
    # Find features with highest variance across classes
    feature_variance = np.var(mean_activations_by_class, axis=0)
    top_features = np.argsort(-feature_variance)[:20]  # Top 20 discriminative features
    
    # Plot class-specific activations for top features
    plt.figure(figsize=(15, 12))
    
    for i, feature_idx in enumerate(top_features[:min(20, len(top_features))]):
        plt.subplot(4, 5, i+1)
        
        feature_values = [mean_act[feature_idx] for mean_act in mean_activations_by_class]
        plt.bar(range(10), feature_values)
        plt.title(f'Feature {feature_idx}')
        plt.xlabel('Digit Class')
        plt.ylabel('Mean Activation')
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/top_discriminative_features.png")
    plt.close()
    
    # Visualize feature correlations for top features
    if len(top_features) > 1:
        top_feature_activations = all_activations[:, top_features[:10]]
        correlation_matrix = np.corrcoef(top_feature_activations.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                    xticklabels=[f'F{i}' for i in top_features[:10]],
                    yticklabels=[f'F{i}' for i in top_features[:10]])
        plt.title('Correlation Between Top Discriminative Features')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_correlations.png")
        plt.close()
    
    return {
        'activation_means': activation_means,
        'activation_stds': activation_stds,
        'top_features': top_features,
        'mean_activations_by_class': mean_activations_by_class
    }

def compare_to_standard_models(model, test_loader, save_dir):
    """Compare the performance with standard models (e.g., linear, SVM)"""
    # This would require implementing standard models for comparison
    # For brevity, just creating a placeholder for the complete implementation
    pass

def main():
    parser = argparse.ArgumentParser(description='Analyze trained Joint K-StoNet model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--save-dir', type=str, default='./analysis', help='Directory to save analysis results')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load model and data
    model, test_loader, test_dataset = load_model_and_data(args.model_path)
    model = model.to(device)
    
    print("Extracting features for visualization...")
    features_by_layer, labels = extract_features(model, test_loader, device)
    
    print("Visualizing features...")
    visualize_features(features_by_layer, labels, f"{args.save_dir}/feature_visualization")
    
    print("Analyzing weight sparsity...")
    sparsity_results = analyze_weight_sparsity(model, f"{args.save_dir}/weight_analysis")
    
    print("Analyzing uncertainty...")
    uncertainty_results = analyze_uncertainty(model, test_dataset, f"{args.save_dir}/uncertainty_analysis")
    
    print("Analyzing kernel activations...")
    kernel_results = analyze_kernel_activations(model, test_loader, f"{args.save_dir}/kernel_analysis")
    
    print(f"Analysis complete. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()