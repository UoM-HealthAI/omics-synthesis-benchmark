#!/usr/bin/env python3
"""
Create data compatible with old scGAN environment (scanpy 1.2.2, anndata 0.6.x)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

def create_compatible_data(output_path, n_cells=1000, n_genes=1200, n_clusters=3, seed=42):
    """
    Create synthetic single-cell RNA-seq data using old versions of scanpy/anndata.
    
    AnnData Structure Explanation:
    - X: Expression matrix (n_cells × n_genes) - main data, expression values per cell per gene
    - obs: Cell metadata (observations) - one row per cell, describes cell properties
    - var: Gene metadata (variables) - one row per gene, describes gene properties
    
    Key Metrics:
    - total_counts: Sum of all gene expression values (like total RNA content)
    - n_genes/n_cells: Number of detected features (sparsity measure)
    - cluster: Cell type assignment based on expression similarity
    
    Parameters:
    -----------
    output_path : str
        Path where to save the synthetic h5ad file
    n_cells : int
        Number of synthetic cells to generate
    n_genes : int  
        Number of genes (must be >1000 for scGAN zheng17 recipe)
    n_clusters : int
        Number of cell types/clusters to simulate
    seed : int
        Random seed for reproducible data generation
    """
    np.random.seed(seed)
    
    print(f"Creating compatible data with old scanpy {sc.__version__}, anndata {ad.__version__}")
    print(f"Data shape: {n_cells} cells x {n_genes} genes")
    
    # Create synthetic expression data
    cells_per_cluster = n_cells // n_clusters
    remainder = n_cells % n_clusters
    
    # Generate expression data
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    clusters = []
    
    start_idx = 0
    for cluster_id in range(n_clusters):
        # Handle remainder cells
        n_cells_cluster = cells_per_cluster + (1 if cluster_id < remainder else 0)
        end_idx = start_idx + n_cells_cluster
        
        # Create simple expression patterns
        # Base expression with some noise
        base_expr = np.random.exponential(0.5, (n_cells_cluster, n_genes))
        
        # Add cluster-specific patterns
        if cluster_id == 0:
            # Cluster 0: higher expression in first third of genes
            base_expr[:, :n_genes//3] *= 2
        elif cluster_id == 1:
            # Cluster 1: higher expression in middle third of genes
            base_expr[:, n_genes//3:2*n_genes//3] *= 2
        else:
            # Cluster 2: higher expression in last third of genes
            base_expr[:, 2*n_genes//3:] *= 2
        
        X[start_idx:end_idx, :] = base_expr
        clusters.extend([str(cluster_id)] * n_cells_cluster)
        start_idx = end_idx
    
    # Create cell and gene names
    cell_names = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create obs (observations/cell metadata) - contains information about each cell
    # In single-cell data: rows = cells, so obs describes properties of each cell
    obs = pd.DataFrame(index=cell_names)
    obs['n_genes'] = np.count_nonzero(X, axis=1)  # Number of genes expressed per cell (non-zero values)
    obs['total_counts'] = np.sum(X, axis=1)       # Total RNA molecules detected per cell (UMI/read count)
    obs['cluster'] = pd.Categorical(clusters)     # Cell type/cluster assignment (0, 1, 2)
    
    # Create var (variables/gene metadata) - contains information about each gene
    # In single-cell data: columns = genes, so var describes properties of each gene
    var = pd.DataFrame(index=gene_names)
    var['n_cells'] = np.count_nonzero(X, axis=0)  # Number of cells expressing this gene (detection rate)
    var['total_counts'] = np.sum(X, axis=0)       # Total expression of this gene across all cells
    
    # Create AnnData object with old version
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    # Make sure gene names are unique
    adata.var_names_make_unique()
    
    print(f"Created AnnData object: {adata.shape}")
    print(f"Sparsity: {100 * (1 - np.count_nonzero(X) / X.size):.1f}%")
    print(f"Value range: {X.min():.3f} to {X.max():.3f}")
    print(f"Clusters: {obs['cluster'].nunique()}")
    
    # Save with old version
    try:
        adata.write(output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Test reading it back
        test_adata = ad.read_h5ad(output_path)
        print(f"✅ Verification read successful: {test_adata.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    
    success = create_compatible_data(
        "data/test_small.h5ad",
        n_cells=1000,
        n_genes=1200,  # More than 1000 to avoid zheng17 recipe issue
        n_clusters=3
    )
    
    if success:
        print("\n✅ Compatible data created successfully!")
        print("💡 Now you can run: python main.py --param parameters_test.json --process")
    else:
        print("\n❌ Failed to create compatible data!")