#!/usr/bin/env python3
"""
Compare real Tabula Muris data vs scGAN generated data
"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def load_real_data(data_path):
    """Load real single-cell data from h5ad file"""
    print(f"Loading real data from: {data_path}")
    
    adata = sc.read_h5ad(data_path)
    print(f"Real data shape: {adata.shape}")
    
    # Extract expression matrix
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Extract cell type labels if available
    labels = None
    if 'cluster' in adata.obs.columns:
        labels = adata.obs['cluster']
        # Convert to numeric if categorical
        if hasattr(labels, 'cat'):
            labels = labels.cat.codes.values
        else:
            # Convert string/object labels to numeric
            try:
                labels = labels.astype(int).values
            except (ValueError, TypeError):
                unique_labels = np.unique(labels)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                labels = np.array([label_map[label] for label in labels])
    elif 'celltype' in adata.obs.columns:
        labels = adata.obs['celltype']
        if hasattr(labels, 'cat'):
            labels = labels.cat.codes.values
        else:
            # Convert string/object labels to numeric
            try:
                labels = labels.astype(int).values
            except (ValueError, TypeError):
                unique_labels = np.unique(labels)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                labels = np.array([label_map[label] for label in labels])
    
    print(f"Extracted expression matrix: {X.shape}")
    if labels is not None:
        print(f"Cell type labels: {len(np.unique(labels))} unique types")
        print(f"Label distribution: {np.bincount(labels)}")
    
    return X, labels, adata.var.index.values  # Return gene names too

def load_generated_data(filepath):
    """Load generated data from h5ad file"""
    print(f"Loading generated data from: {filepath}")
    
    if filepath.endswith('.h5ad'):
        adata = sc.read_h5ad(filepath)
        if hasattr(adata.X, 'toarray'):
            X = adata.X.toarray()
        else:
            X = adata.X
        print(f"Generated data shape: {X.shape}")
        return X, adata.var.index.values if hasattr(adata, 'var') else None
    elif filepath.endswith('.npz'):
        data = np.load(filepath)
        return data['cell_gen'], None
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

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
    
    # Fit t-SNE with adjusted perplexity if needed
    n_samples = combined_X_scaled.shape[0]
    perplexity = min(perplexity, (n_samples - 1) // 3)
    
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, 
                init='pca', learning_rate=200.0)
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
                      color='red', label='Real Data')
    axes[0, 0].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.6, s=10, 
                      color='blue', label='scGAN Generated')
    axes[0, 0].set_title(f'PCA Comparison (Explained Variance: {pca_obj.explained_variance_ratio_.sum():.3f})')
    axes[0, 0].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[0, 0].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE comparison
    axes[0, 1].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.6, s=10, 
                      color='red', label='Real Data')
    axes[0, 1].scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.6, s=10, 
                      color='blue', label='scGAN Generated')
    axes[0, 1].set_title('t-SNE Comparison')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - Real data only (colored by cell type if available)
    if real_labels is not None:
        unique_labels = np.unique(real_labels)
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(unique_labels)))
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
    axes[1, 1].set_title('PCA - scGAN Generated Data Only')
    axes[1, 1].set_xlabel(f'PC1 ({pca_obj.explained_variance_ratio_[0]:.3f})')
    axes[1, 1].set_ylabel(f'PC2 ({pca_obj.explained_variance_ratio_[1]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Real vs scGAN Generated Single-Cell Data Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def compute_quality_metrics(real_X, fake_X, real_labels=None):
    """Compute quality metrics for generated data"""
    print("Computing quality metrics...")
    
    # 1. Mean pairwise distance comparison
    real_sample = real_X[:min(1000, real_X.shape[0])]  # Sample for efficiency
    fake_sample = fake_X[:min(1000, fake_X.shape[0])]
    
    real_distances = cdist(real_sample, real_sample, metric='euclidean')
    fake_distances = cdist(fake_sample, fake_sample, metric='euclidean')
    
    real_mean_dist = np.mean(real_distances[np.triu_indices_from(real_distances, k=1)])
    fake_mean_dist = np.mean(fake_distances[np.triu_indices_from(fake_distances, k=1)])
    
    print(f"Mean pairwise distance - Real: {real_mean_dist:.4f}, Generated: {fake_mean_dist:.4f}")
    
    # 2. Feature correlation comparison
    real_corr = np.corrcoef(real_X.T)
    fake_corr = np.corrcoef(fake_X.T)
    
    # Compare correlation structures
    corr_diff = np.abs(real_corr - fake_corr)
    mean_corr_diff = np.mean(corr_diff[np.triu_indices_from(corr_diff, k=1)])
    
    print(f"Mean correlation difference: {mean_corr_diff:.4f}")
    
    # 3. Silhouette score if labels available
    if real_labels is not None:
        # Subsample for silhouette computation
        n_samples = min(1000, real_X.shape[0])
        indices = np.random.choice(real_X.shape[0], n_samples, replace=False)
        
        real_silhouette = silhouette_score(real_X[indices], real_labels[indices])
        print(f"Real data silhouette score: {real_silhouette:.4f}")
    
    return {
        'real_mean_distance': real_mean_dist,
        'fake_mean_distance': fake_mean_dist,
        'mean_correlation_difference': mean_corr_diff
    }

def generate_statistics_comparison(real_X, fake_X, real_labels=None):
    """Generate comprehensive comparison statistics"""
    print("=== Real vs scGAN Generated Data Comparison ===")
    print(f"Real data shape: {real_X.shape}")
    print(f"Generated data shape: {fake_X.shape}")
    
    if real_labels is not None:
        print(f"Real data cell types: {len(np.unique(real_labels))}")
        print(f"Cell type distribution: {np.bincount(real_labels)}")
    print()
    
    print("Expression Statistics:")
    print("Real Data:")
    print(f"  Mean: {np.mean(real_X):.6f}")
    print(f"  Std:  {np.std(real_X):.6f}")
    print(f"  Min:  {np.min(real_X):.6f}")
    print(f"  Max:  {np.max(real_X):.6f}")
    print(f"  Median: {np.median(real_X):.6f}")
    
    print("scGAN Generated Data:")
    print(f"  Mean: {np.mean(fake_X):.6f}")
    print(f"  Std:  {np.std(fake_X):.6f}")
    print(f"  Min:  {np.min(fake_X):.6f}")
    print(f"  Max:  {np.max(fake_X):.6f}")
    print(f"  Median: {np.median(fake_X):.6f}")
    print()
    
    print("Sparsity Analysis:")
    real_sparsity = np.sum(real_X == 0) / real_X.size
    fake_sparsity = np.sum(fake_X == 0) / fake_X.size
    print(f"Real data sparsity: {real_sparsity:.3f} ({real_sparsity*100:.1f}%)")
    print(f"Generated data sparsity: {fake_sparsity:.3f} ({fake_sparsity*100:.1f}%)")
    print()
    
    # Per-cell statistics
    real_cell_totals = np.sum(real_X, axis=1)
    fake_cell_totals = np.sum(fake_X, axis=1)
    
    print("Per-cell Total Expression:")
    print(f"Real data - Mean: {np.mean(real_cell_totals):.2f}, Std: {np.std(real_cell_totals):.2f}")
    print(f"Generated data - Mean: {np.mean(fake_cell_totals):.2f}, Std: {np.std(fake_cell_totals):.2f}")
    print()
    
    # Per-gene statistics
    real_gene_totals = np.sum(real_X, axis=0)
    fake_gene_totals = np.sum(fake_X, axis=0)
    
    print("Per-gene Total Expression:")
    print(f"Real data - Mean: {np.mean(real_gene_totals):.2f}, Std: {np.std(real_gene_totals):.2f}")
    print(f"Generated data - Mean: {np.mean(fake_gene_totals):.2f}, Std: {np.std(fake_gene_totals):.2f}")
    print()

def plot_distribution_comparison(real_X, fake_X, save_path=None):
    """Plot distribution comparisons"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall expression distribution
    axes[0, 0].hist(real_X.flatten(), bins=50, alpha=0.7, label='Real', density=True, color='red')
    axes[0, 0].hist(fake_X.flatten(), bins=50, alpha=0.7, label='scGAN Generated', density=True, color='blue')
    axes[0, 0].set_xlabel('Expression Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overall Expression Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Per-cell total expression
    real_cell_totals = np.sum(real_X, axis=1)
    fake_cell_totals = np.sum(fake_X, axis=1)
    
    axes[0, 1].hist(real_cell_totals, bins=50, alpha=0.7, label='Real', density=True, color='red')
    axes[0, 1].hist(fake_cell_totals, bins=50, alpha=0.7, label='scGAN Generated', density=True, color='blue')
    axes[0, 1].set_xlabel('Total Expression per Cell')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Per-cell Total Expression Distribution')
    axes[0, 1].legend()
    
    # Per-gene total expression
    real_gene_totals = np.sum(real_X, axis=0)
    fake_gene_totals = np.sum(fake_X, axis=0)
    
    axes[1, 0].hist(real_gene_totals, bins=50, alpha=0.7, label='Real', density=True, color='red')
    axes[1, 0].hist(fake_gene_totals, bins=50, alpha=0.7, label='scGAN Generated', density=True, color='blue')
    axes[1, 0].set_xlabel('Total Expression per Gene')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Per-gene Total Expression Distribution')
    axes[1, 0].legend()
    
    # Non-zero expression distribution
    real_nonzero = real_X[real_X > 0]
    fake_nonzero = fake_X[fake_X > 0]
    
    axes[1, 1].hist(real_nonzero, bins=50, alpha=0.7, label='Real', density=True, color='red')
    axes[1, 1].hist(fake_nonzero, bins=50, alpha=0.7, label='scGAN Generated', density=True, color='blue')
    axes[1, 1].set_xlabel('Non-zero Expression Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Non-zero Expression Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        dist_save_path = save_path.replace('.png', '_distributions.png')
        plt.savefig(dist_save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plots saved to: {dist_save_path}")
    
    plt.show()

def main():
    # Data paths - update these as needed
    real_data_path = '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scGAN/data/tabula_muris/all_converted.h5ad'
    generated_data_path = '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/output/sample/scGAN/tabula_muris/fake.h5ad'
    
    # Check if files exist
    if not os.path.exists(real_data_path):
        print(f"Real data file not found: {real_data_path}")
        print("Please update the real_data_path variable")
        return
    
    if not os.path.exists(generated_data_path):
        print(f"Generated data file not found: {generated_data_path}")
        print("Please update the generated_data_path variable")
        return
    
    # Load real data
    print("Loading real data...")
    real_X, real_labels, real_gene_names = load_real_data(real_data_path)
    
    # Load generated data
    print("Loading generated data...")
    fake_X, fake_gene_names = load_generated_data(generated_data_path)
    
    # Ensure compatible dimensions
    min_features = min(real_X.shape[1], fake_X.shape[1])
    real_X = real_X[:, :min_features]
    fake_X = fake_X[:, :min_features]
    
    print(f"Using {min_features} features for comparison")
    
    # Subsample for computational efficiency
    print("Subsampling data for visualization...")
    real_X_sub, real_labels_sub, real_indices = subsample_data(real_X, real_labels, n_samples=3000)
    
    # Ensure both datasets have same number of samples for fair comparison
    min_samples = min(real_X_sub.shape[0], fake_X.shape[0])
    real_X_sub = real_X_sub[:min_samples]
    fake_X = fake_X[:min_samples]
    if real_labels_sub is not None:
        real_labels_sub = real_labels_sub[:min_samples]
    
    print(f"Using {min_samples} samples for comparison")
    
    # Generate statistics
    generate_statistics_comparison(real_X_sub, fake_X, real_labels_sub)
    
    # Compute quality metrics
    quality_metrics = compute_quality_metrics(real_X_sub, fake_X, real_labels_sub)
    
    print("\nComputing PCA...")
    real_pca, fake_pca, pca_obj = compute_pca(real_X_sub, fake_X)
    
    print("Computing t-SNE...")
    real_tsne, fake_tsne = compute_tsne(real_X_sub, fake_X)
    
    print("Creating comparison plots...")
    
    # Create output directory
    output_dir = 'plots/scGAN'
    os.makedirs(output_dir, exist_ok=True)
    
    # Main comparison plot
    plot_comparison(real_pca, fake_pca, real_tsne, fake_tsne, pca_obj, 
                   real_labels_sub, f'{output_dir}/scgan_dimensionality_reduction_real_vs_fake.png')
    
    # Distribution comparison plots
    print("Creating distribution comparison plots...")
    plot_distribution_comparison(real_X_sub, fake_X, 
                               f'{output_dir}/scgan_expression_distributions.png')
    
    print(f"\nAnalysis complete! Plots saved in: {output_dir}/")

if __name__ == "__main__":
    main()