#!/usr/bin/env python
"""
Prepare Tabula Muris data for ACTIVA by adding train/test split
"""
import scanpy as sc
import numpy as np

# Read the data
print("Loading Tabula Muris data...")
adata = sc.read_h5ad('/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data/tabula_muris/all.h5ad')
print(f"Data shape: {adata.shape}")

# Create train/test split (80/20)
n_cells = adata.shape[0]
train_size = int(0.8 * n_cells)

# Random split
np.random.seed(42)
indices = np.random.permutation(n_cells)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Add split column
adata.obs['split'] = 'test'
adata.obs.loc[adata.obs.index[train_indices], 'split'] = 'train'

# Create cluster column from celltype (convert to numeric)
celltype_to_int = {ct: i for i, ct in enumerate(adata.obs['celltype'].cat.categories)}
adata.obs['cluster'] = adata.obs['celltype'].map(celltype_to_int)

print(f"Train cells: {sum(adata.obs['split'] == 'train')}")
print(f"Test cells: {sum(adata.obs['split'] == 'test')}")
print(f"Cell type mapping: {celltype_to_int}")

# Save the modified data
output_path = '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/ACTIVA/data/tabula_muris_split.h5ad'
adata.write(output_path)
print(f"Saved to: {output_path}")