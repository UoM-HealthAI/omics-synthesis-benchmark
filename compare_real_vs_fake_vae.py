#!/usr/bin/env python3
"""
Compare real Tabula Muris data (VAE-compressed) vs scDiffusion generated data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
sys.path.append('benchmark/scDiffusion')
from benchmark.scDiffusion.guided_diffusion.cell_datasets_GB import load_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_real_data_vae(data_dir, vae_path, batch_size=64, hidden_dim=128):
    """Load real data using VAE compression"""
    print(f"Loading real data from: {data_dir}")
    print(f"Using VAE model: {vae_path}")
    
    # Use the load_data function from cell_datasets_GB.py
    data_loader = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        vae_path=vae_path,
        train_vae=False,
        hidden_dim=hidden_dim
    )
    
    # Collect all data from the loader
    all_data = []
    all_labels = []
    
    print("Collecting data from loader...")
    for i, (batch_data, batch_info) in enumerate(data_loader):
        if isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cpu().numpy()
        
        all_data.append(batch_data)
        if 'y' in batch_info:
            all_labels.append(batch_info['y'])
        
        if i >= 100:  # Limit to ~6400 samples (64*100)
            break
    
    real_data = np.vstack(all_data)
    real_labels = np.hstack(all_labels) if all_labels else None
    
    print(f"Collected {real_data.shape[0]} real samples with {real_data.shape[1]} features")
    return real_data, real_labels

def load_generated_data(filepath):
    """Load generated data from npz file"""
    data = np.load(filepath)
    return data['cell_gen']

def subsample_data(X, labels=None, n_samples=3000, random_state=42):
    """Subsample data to manageable size for visualization"""
    if X.shape[0] <= n_samples:
        return X, labels, np.arange(X.shape[0])
    
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    
    X_sub = X[indices]
    labels_sub = labels[indices] if labels is not None else None
    
    return X_sub, labels_sub, indices

def compute_pca(real_X, fake_X, n_components=2):
    """Compute PCA on combined data"""
    # Combine data for fitting PCA
    combined_X = np.vstack([real_X, fake_X])
    
    # Standardize
    scaler = StandardScaler()
    combined_X_scaled = scaler.fit_transform(combined_X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined_X_scaled)
    
    # Split back
    real_pca = combined_pca[:real_X.shape[0]]
    fake_pca = combined_pca[real_X.shape[0]:]
    
    return real_pca, fake_pca, pca

def compute_tsne(real_X, fake_X, n_components=2, perplexity=30):
    """Compute t-SNE on combined data"""
    # Combine data for fitting t-SNE
    combined_X = np.vstack([real_X, fake_X])
    
    # Standardize
    scaler = StandardScaler()
    combined_X_scaled = scaler.fit_transform(combined_X)
    
    # Fit t-SNE
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    combined_tsne = tsne.fit_transform(combined_X_scaled)
    
    # Split back
    real_tsne = combined_tsne[:real_X.shape[0]]
    fake_tsne = combined_tsne[real_X.shape[0]:]
    
    return real_tsne, fake_tsne

def plot_comparison(real_pca, fake_pca, real_tsne, fake_tsne, pca_obj, real_labels=None, save_path=None):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA comparison
    axes[0, 0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, s=10, 
                      color='red', label='Real Data (VAE-compressed)')
    axes[0, 0].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.6, s=10, 
                      color='blue', label='Generated Data')
    axes[0, 0].set_title(f'PCA Comparison (Explained Variance: {pca_obj.explained_variance_ratio_.sum():.3f})')
    axes[0, 0].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE comparison
    axes[0, 1].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.6, s=10, 
                      color='red', label='Real Data (VAE-compressed)')
    axes[0, 1].scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.6, s=10, 
                      color='blue', label='Generated Data')
    axes[0, 1].set_title('t-SNE Comparison')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - Real data only (colored by cell type if available)
    if real_labels is not None:
        unique_labels = np.unique(real_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = real_labels == label
            axes[1, 0].scatter(real_pca[mask, 0], real_pca[mask, 1], 
                             alpha=0.6, s=10, color=colors[i], label=f'Type {label}')
        axes[1, 0].set_title('PCA - Real Data (Colored by Cell Type)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        axes[1, 0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, s=10, color='red')
        axes[1, 0].set_title('PCA - Real Data Only')
    
    axes[1, 0].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[1, 0].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # PCA - Generated data only
    axes[1, 1].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.6, s=10, color='blue')
    axes[1, 1].set_title('PCA - Generated Data Only')
    axes[1, 1].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[1, 1].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Real (VAE-compressed) vs Generated Single-Cell Data Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def generate_statistics_comparison(real_X, fake_X, real_labels=None):
    """Generate comparison statistics"""
    print("=== Real (VAE-compressed) vs Generated Data Comparison ===")
    print(f"Real data shape: {real_X.shape}")
    print(f"Generated data shape: {fake_X.shape}")
    
    if real_labels is not None:
        print(f"Real data cell types: {len(np.unique(real_labels))}")
        print(f"Cell type distribution: {np.bincount(real_labels)}")
    print()
    
    print("Expression Statistics:")
    print("Real Data (VAE-compressed):")
    print(f"  Mean: {np.mean(real_X):.6f}")
    print(f"  Std:  {np.std(real_X):.6f}")
    print(f"  Min:  {np.min(real_X):.6f}")
    print(f"  Max:  {np.max(real_X):.6f}")
    
    print("Generated Data:")
    print(f"  Mean: {np.mean(fake_X):.6f}")
    print(f"  Std:  {np.std(fake_X):.6f}")
    print(f"  Min:  {np.min(fake_X):.6f}")
    print(f"  Max:  {np.max(fake_X):.6f}")
    print()
    
    print("Sparsity Analysis:")
    real_sparsity = np.sum(real_X == 0) / real_X.size
    fake_sparsity = np.sum(fake_X == 0) / fake_X.size
    print(f"Real data sparsity: {real_sparsity:.3f} ({real_sparsity*100:.1f}%)")
    print(f"Generated data sparsity: {fake_sparsity:.3f} ({fake_sparsity*100:.1f}%)")

def main():
    # Data paths
    data_dir = '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data/tabula_muris/all.h5ad'
    vae_path = '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/checkpoint/AE/my_VAE/model_seed=0_step=0.pt'
    generated_data_path = 'benchmark/output/sample/scDiffusion/unconditional_sampling.npz'
    
    # Load real data using VAE compression
    print("Loading real data with VAE compression...")
    real_X, real_labels = load_real_data_vae(data_dir, vae_path, batch_size=64, hidden_dim=128)
    
    # Load generated data
    print("Loading generated data...")
    fake_X = load_generated_data(generated_data_path)
    
    # Subsample real data for computational efficiency
    print("Subsampling real data...")
    real_X_sub, real_labels_sub, real_indices = subsample_data(real_X, real_labels, n_samples=3000)
    
    # Ensure both datasets have same number of samples for fair comparison
    min_samples = min(real_X_sub.shape[0], fake_X.shape[0])
    real_X_sub = real_X_sub[:min_samples]
    fake_X = fake_X[:min_samples]
    if real_labels_sub is not None:
        real_labels_sub = real_labels_sub[:min_samples]
    
    # Generate statistics
    generate_statistics_comparison(real_X_sub, fake_X, real_labels_sub)
    
    print("\nComputing PCA...")
    real_pca, fake_pca, pca_obj = compute_pca(real_X_sub, fake_X)
    
    print("Computing t-SNE...")
    real_tsne, fake_tsne = compute_tsne(real_X_sub, fake_X)
    
    print("Creating comparison plots...")
    plot_comparison(real_pca, fake_pca, real_tsne, fake_tsne, pca_obj, 
                   real_labels_sub, 'plots/dimensionality_reduction_real_vs_fake.png')

if __name__ == "__main__":
    main()