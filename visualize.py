#!/usr/bin/env python3
"""
Visualization script for scDiffusion unconditional sampling results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import os

def load_data(filepath):
    """Load the npz file and return the cell data"""
    data = np.load(filepath)
    return data['cell_gen']

def plot_distribution_heatmap(data, save_path=None):
    """Plot heatmap of cell-gene expression distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall distribution
    axes[0, 0].hist(data.flatten(), bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Overall Expression Distribution')
    axes[0, 0].set_xlabel('Expression Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # Heatmap of first 50 cells and genes
    sns.heatmap(data[:50, :50], ax=axes[0, 1], cmap='viridis', cbar=True)
    axes[0, 1].set_title('Expression Heatmap (First 50 cells x 50 genes)')
    axes[0, 1].set_xlabel('Genes')
    axes[0, 1].set_ylabel('Cells')
    
    # Cell-wise expression statistics
    cell_means = np.mean(data, axis=1)
    axes[1, 0].hist(cell_means, bins=30, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Mean Expression per Cell')
    axes[1, 0].set_xlabel('Mean Expression')
    axes[1, 0].set_ylabel('Number of Cells')
    
    # Gene-wise expression statistics
    gene_means = np.mean(data, axis=0)
    axes[1, 1].hist(gene_means, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 1].set_title('Mean Expression per Gene')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Number of Genes')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_dimensionality_reduction(data, save_path=None):
    """Plot PCA and t-SNE visualizations"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    axes[0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=20)
    axes[0].set_title(f'PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    
    # t-SNE (using subset for speed)
    subset_size = min(1000, data.shape[0])
    subset_idx = np.random.choice(data.shape[0], subset_size, replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(data[subset_idx])
    
    axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=20)
    axes[1].set_title(f't-SNE (n={subset_size} cells)')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_quality_metrics(data, save_path=None):
    """Plot various quality metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sparsity analysis
    sparsity = np.sum(data == 0, axis=1) / data.shape[1]
    axes[0, 0].hist(sparsity, bins=30, alpha=0.7, color='orange')
    axes[0, 0].set_title('Sparsity Distribution (Fraction of Zero Values)')
    axes[0, 0].set_xlabel('Sparsity')
    axes[0, 0].set_ylabel('Number of Cells')
    
    # Expression variance
    gene_var = np.var(data, axis=0)
    axes[0, 1].hist(gene_var, bins=30, alpha=0.7, color='purple')
    axes[0, 1].set_title('Gene Expression Variance')
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_ylabel('Number of Genes')
    
    # Cell-cell correlation heatmap (subset)
    subset_cells = data[:100]  # Use first 100 cells
    corr_matrix = np.corrcoef(subset_cells)
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_title('Cell-Cell Correlation (First 100 cells)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Top expressed genes
    gene_means = np.mean(data, axis=0)
    top_genes_idx = np.argsort(gene_means)[-20:]
    axes[1, 1].barh(range(20), gene_means[top_genes_idx])
    axes[1, 1].set_title('Top 20 Expressed Genes')
    axes[1, 1].set_xlabel('Mean Expression')
    axes[1, 1].set_ylabel('Gene Index')
    axes[1, 1].set_yticks(range(20))
    axes[1, 1].set_yticklabels([f'Gene_{i}' for i in top_genes_idx])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_stats(data):
    """Generate and print summary statistics"""
    print("=== scDiffusion Sampling Results Summary ===")
    print(f"Dataset shape: {data.shape[0]} cells x {data.shape[1]} genes")
    print(f"Data type: {data.dtype}")
    print(f"Memory usage: {data.nbytes / 1024**2:.2f} MB")
    print()
    
    print("Expression Statistics:")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std:  {np.std(data):.6f}")
    print(f"  Min:  {np.min(data):.6f}")
    print(f"  Max:  {np.max(data):.6f}")
    print()
    
    print("Sparsity Analysis:")
    sparsity = np.sum(data == 0) / data.size
    print(f"  Overall sparsity: {sparsity:.3f} ({sparsity*100:.1f}%)")
    
    cell_sparsity = np.mean(np.sum(data == 0, axis=1) / data.shape[1])
    print(f"  Average cell sparsity: {cell_sparsity:.3f}")
    
    gene_sparsity = np.mean(np.sum(data == 0, axis=0) / data.shape[0])
    print(f"  Average gene sparsity: {gene_sparsity:.3f}")
    print()
    
    print("Expression Range per Cell:")
    cell_ranges = np.max(data, axis=1) - np.min(data, axis=1)
    print(f"  Mean range: {np.mean(cell_ranges):.6f}")
    print(f"  Std range:  {np.std(cell_ranges):.6f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize scDiffusion sampling results')
    parser.add_argument('--input', '-i', 
                       default='benchmark/output/sample/scDiffusion/unconditional_sampling.npz',
                       help='Path to input npz file')
    parser.add_argument('--output-dir', '-o', 
                       default='./plots',
                       help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Only show summary statistics')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.input}")
    data = load_data(args.input)
    
    # Generate summary statistics
    generate_summary_stats(data)
    
    if not args.no_plots:
        print("\nGenerating visualizations...")
        
        # Plot distributions and heatmaps
        plot_distribution_heatmap(data, 
                                 os.path.join(args.output_dir, 'distribution_heatmap.png'))
        
        # Plot dimensionality reduction
        plot_dimensionality_reduction(data, 
                                    os.path.join(args.output_dir, 'dimensionality_reduction.png'))
        
        # Plot quality metrics
        plot_quality_metrics(data, 
                           os.path.join(args.output_dir, 'quality_metrics.png'))
        
        print(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()