#!/bin/bash
#SBATCH --job-name=scgan_gpu_training
#SBATCH --output=scgan_gpu_%j.out
#SBATCH --error=scgan_gpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G

# Load necessary modules for TensorFlow 2.6.2 compatibility
module load CUDA/11.8.0
module load cuDNN/8.7.0.84-CUDA-11.8.0

# Navigate to scGAN directory
cd /ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scGAN

# Activate conda environment
source ~/miniconda/bin/activate scgan

# Set environment variables for GPU usage
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run scGAN training with GPU
python main.py --param parameters_tabula_muris.json --train