#!/usr/bin/env python3
"""
Create Tabula Muris data compatible with old scGAN environment

This script creates a two-step process:
1. Extract data from modern h5ad format (run in scdiffusion environment)
2. Create compatible h5ad using old environment (run in scgan environment)
"""

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path

def extract_dataset_data(dataset_name="tabula_muris", input_path=None):
    """
    Step 1: Extract data from modern h5ad format
    Run this in scdiffusion environment
    
    Args:
        dataset_name: Name of the dataset (tabula_muris, pbmc68k, human_pf_lung)
        input_path: Custom input path (optional)
    """
    import scanpy as sc
    
    if input_path is None:
        base_path = "/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data"
        
        # Dataset-specific file mapping
        dataset_files = {
            "tabula_muris": f"{base_path}/tabula_muris/all.h5ad",
            "human_pf_lung": f"{base_path}/Human_PF_Lung/human_lung.h5ad", 
            "pbmc68k": f"{base_path}/pbmc68k/filtered_matrices_mex",  # This is a directory with MTX format
            "sapiens": f"{base_path}/sapiens/sapiens_ood_data.h5ad",
            "wot": f"{base_path}/WOT/filted_data.h5ad"
        }
        
        if dataset_name not in dataset_files:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_files.keys())}")
            
        input_path = dataset_files[dataset_name]
    
    print("Step 1: Extracting data with modern scanpy...")
    print(f"Dataset: {dataset_name}")
    print(f"Loading from: {input_path}")
    
    # Load data based on format
    if dataset_name == "pbmc68k":
        # PBMC68k is in MTX format, need to handle differently
        adata = sc.read_10x_mtx(input_path, var_names='gene_symbols', cache=True)
        adata.var_names_make_unique()
        
        # Load cell type annotations if available
        barcode_path = f"{input_path}/../68k_pbmc_barcodes_annotation.tsv"
        if os.path.exists(barcode_path):
            annotations = pd.read_csv(barcode_path, sep='\t', index_col=0)
            # Match barcodes between adata and annotations
            common_barcodes = adata.obs.index.intersection(annotations.index)
            adata = adata[common_barcodes, :].copy()
            adata.obs['celltype'] = annotations.loc[common_barcodes, 'celltype'] if 'celltype' in annotations.columns else 'unknown'
    else:
        # Standard h5ad format
        adata = sc.read(input_path)
    
    print(f"Original shape: {adata.shape}")
    print(f"Available metadata columns: {list(adata.obs.columns)}")
    
    # Subsample for performance
    np.random.seed(42)
    max_cells = 5000000000
    max_genes = 2000000000
    
    if adata.n_obs > max_cells:
        cell_indices = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[cell_indices, :]
        print(f"Subsampled to {max_cells} cells")
    
    if adata.n_vars > max_genes:
        if hasattr(adata.X, 'toarray'):
            gene_means = np.array(adata.X.mean(axis=0)).flatten()
        else:
            gene_means = np.mean(adata.X, axis=0)
        top_gene_indices = np.argsort(gene_means)[-max_genes:]
        adata = adata[:, top_gene_indices]
        print(f"Selected top {max_genes} genes")
    
    # Convert to dense and extract components
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray().astype(np.float32)
    else:
        X_dense = adata.X.astype(np.float32)
    
    # Extract metadata
    cell_names = list(adata.obs.index)
    gene_names = list(adata.var.index)
    
    # Get cluster information and convert to integer IDs
    if 'celltype' in adata.obs.columns:
        # Get unique tissue types and create mapping to integers
        tissue_types = adata.obs['celltype'].astype(str)
        unique_tissues = sorted(tissue_types.unique())  # Sort for consistent mapping
        tissue_to_id = {tissue: idx for idx, tissue in enumerate(unique_tissues)}
        
        # Convert tissue names to integer IDs
        clusters = [str(tissue_to_id[tissue]) for tissue in tissue_types]
        
        print(f"Tissue to ID mapping:")
        for tissue, tid in tissue_to_id.items():
            print(f"  {tissue} -> {tid}")
            
    else:
        # Create simple clusters based on expression
        count_totals = np.sum(X_dense, axis=1)
        count_percentiles = np.percentile(count_totals, [33, 66])
        cluster_labels = np.digitize(count_totals, count_percentiles)
        clusters = [str(c) for c in cluster_labels]
    
    # Save extracted data as pickle for step 2
    extracted_data = {
        'X': X_dense,
        'cell_names': cell_names,
        'gene_names': gene_names,
        'clusters': clusters,
        'shape': X_dense.shape,
        'tissue_mapping': tissue_to_id if 'celltype' in adata.obs.columns else None
    }
    
    output_dir = Path(f"data/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "extracted_data.pkl", "wb") as f:
        pickle.dump(extracted_data, f)
    
    print(f"✅ Data extracted and saved to: {output_dir}/extracted_data.pkl")
    print(f"Final shape: {X_dense.shape}")
    print(f"Clusters: {len(set(clusters))} unique types")
    print(f"Cluster distribution: {pd.Series(clusters).value_counts().to_dict()}")
    
    return True

def create_compatible_h5ad(dataset_name="tabula_muris"):
    """
    Step 2: Create compatible h5ad using old environment
    Run this in scgan environment
    
    Args:
        dataset_name: Name of the dataset to process
    """
    import scanpy as sc
    import anndata as ad
    
    print("Step 2: Creating compatible h5ad with old scanpy...")
    print(f"Dataset: {dataset_name}")
    print(f"Using scanpy {sc.__version__}, anndata {ad.__version__}")
    
    # Load extracted data
    pickle_path = f"data/{dataset_name}/extracted_data.pkl"
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    X = data['X']
    cell_names = data['cell_names']
    gene_names = data['gene_names']
    clusters = data['clusters']
    tissue_mapping = data.get('tissue_mapping', None)
    
    print(f"Loaded data shape: {X.shape}")
    if tissue_mapping:
        print(f"Tissue mapping: {tissue_mapping}")
        print(f"Cluster IDs: {sorted(set(clusters))}")
    
    # Create obs (observations/cell metadata) - contains information about each cell
    # In single-cell data: rows = cells, so obs describes properties of each cell
    obs = pd.DataFrame(index=cell_names)
    obs['n_genes'] = np.count_nonzero(X, axis=1)  # Number of genes expressed per cell (non-zero values)
    obs['total_counts'] = np.sum(X, axis=1)       # Total RNA molecules detected per cell (UMI/read count)
    obs['cluster'] = pd.Categorical(clusters)     # Cell type/cluster assignment from original data
    
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
    output_path = f"data/{dataset_name}/all_converted.h5ad"
    try:
        adata.write(output_path)
        print(f"✅ Saved to: {output_path}")
        
        # Test reading it back
        test_adata = ad.read_h5ad(output_path)
        print(f"✅ Verification read successful: {test_adata.shape}")
        
        # Save tissue mapping for reference
        if tissue_mapping:
            mapping_path = f"data/{dataset_name}/tissue_to_cluster_mapping.txt"
            with open(mapping_path, "w") as f:
                f.write("Tissue Type -> Cluster ID Mapping\n")
                f.write("="*40 + "\n")
                for tissue, cluster_id in tissue_mapping.items():
                    f.write(f"{tissue} -> {cluster_id}\n")
            print(f"📝 Saved tissue mapping to: {mapping_path}")
        
        # Clean up temporary file
        os.remove(f"data/{dataset_name}/extracted_data.pkl")
        print("🧹 Cleaned up temporary files")
        
        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def main():
    import sys
    
    # Parse command line arguments
    dataset_name = "tabula_muris"  # default
    
    # Check for dataset argument
    if "--dataset" in sys.argv:
        dataset_idx = sys.argv.index("--dataset")
        if dataset_idx + 1 < len(sys.argv):
            dataset_name = sys.argv[dataset_idx + 1]
    
    if len(sys.argv) > 1 and sys.argv[1] == "step2":
        # Step 2: Create compatible h5ad (run in scgan environment)
        success = create_compatible_h5ad(dataset_name)
    else:
        # Step 1: Extract data (run in scdiffusion environment)
        success = extract_dataset_data(dataset_name)
        if success:
            print("\n" + "="*60)
            print("✅ Step 1 completed successfully!")
            print(f"💡 Now run: conda activate scgan && python create_tabula_muris_compatible.py step2 --dataset {dataset_name}")
            print("="*60)
    
    return success

if __name__ == "__main__":
    main()