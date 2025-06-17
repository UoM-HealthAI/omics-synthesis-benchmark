#!/usr/bin/env python3
"""
Compare real Tabula Muris data vs scDiffusion generated data
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_real_data(filepath):
    """Load real Tabula Muris data"""
    adata = sc.read_h5ad(filepath)
    # Convert to dense if sparse
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    return X, adata.obs

def load_generated_data(filepath):
    """Load generated data from npz file"""
    data = np.load(filepath)
    return data['cell_gen']

def subsample_data(X, n_samples=3000, random_state=42):
    """Subsample data to manageable size for visualization"""
    if X.shape[0] <= n_samples:
        return X, np.arange(X.shape[0])
    
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    return X[indices], indices

def match_dimensions(real_X, fake_X):
    """Match dimensions between real and fake data"""
    # Take first N genes from real data to match fake data dimensions
    min_genes = min(real_X.shape[1], fake_X.shape[1])
    real_X_matched = real_X[:, :min_genes]
    fake_X_matched = fake_X[:, :min_genes]
    return real_X_matched, fake_X_matched

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

def plot_comparison(real_pca, fake_pca, real_tsne, fake_tsne, pca_obj, save_path):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA comparison
    axes[0, 0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.6, s=10, 
                      color='red', label='Real Data')
    axes[0, 0].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.6, s=10, 
                      color='blue', label='Generated Data')
    axes[0, 0].set_title(f'PCA Comparison (Explained Variance: {pca_obj.explained_variance_ratio_.sum():.3f})')
    axes[0, 0].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE comparison
    axes[0, 1].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.6, s=10, 
                      color='red', label='Real Data')
    axes[0, 1].scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.6, s=10, 
                      color='blue', label='Generated Data')
    axes[0, 1].set_title('t-SNE Comparison')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - Real data only
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
    
    plt.suptitle('Real vs Generated Single-Cell Data Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def generate_statistics_comparison(real_X, fake_X):
    """Generate comparison statistics"""
    print("=== Real vs Generated Data Comparison ===")
    print(f"Real data shape: {real_X.shape}")
    print(f"Generated data shape: {fake_X.shape}")
    print()
    
    print("Expression Statistics:")
    print("Real Data:")
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
    # Load data
    print("Loading real Tabula Muris data...")
    real_X, real_obs = load_real_data('/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data/tabula_muris/all.h5ad')
    
    print("Loading generated data...")
    fake_X = load_generated_data('benchmark/output/sample/scDiffusion/unconditional_sampling.npz')
    
    # Subsample real data for computational efficiency
    print("Subsampling real data...")
    real_X_sub, real_indices = subsample_data(real_X, n_samples=3000)
    
    # Match dimensions
    print("Matching dimensions...")
    real_X_matched, fake_X_matched = match_dimensions(real_X_sub, fake_X)
    
    # Generate statistics
    generate_statistics_comparison(real_X_matched, fake_X_matched)
    
    print("\nComputing PCA...")
    real_pca, fake_pca, pca_obj = compute_pca(real_X_matched, fake_X_matched)
    
    print("Computing t-SNE...")
    real_tsne, fake_tsne = compute_tsne(real_X_matched, fake_X_matched)
    
    print("Creating comparison plots...")
    plot_comparison(real_pca, fake_pca, real_tsne, fake_tsne, pca_obj, 
                   'plots/dimensionality_reduction_real_vs_fake.png')

if __name__ == "__main__":
    main()