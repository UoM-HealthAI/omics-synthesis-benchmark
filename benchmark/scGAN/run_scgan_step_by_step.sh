#!/bin/bash

# Load modules
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Clean output directory
# rm -rf /ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scGAN/output/benchmark_scGAN_test_tabula_muris

# echo "Step 1: Processing data..."
# python main.py --param parameters_test.json --process

if [ $? -eq 0 ]; then
    echo "Data processing completed successfully!"
    echo "Step 2: Starting training..."
    python main.py --param parameters_test.json --train
else
    echo "Data processing failed!"
    exit 1
fi