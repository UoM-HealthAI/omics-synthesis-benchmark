#!/bin/bash
#SBATCH --job-name=omics_gpu_jinyuaner
#SBATCH --output=gpu_job_%j.out
#SBATCH --error=gpu_job_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G

# Load necessary modules
module load CUDA/12.0.0
module load Python/3.9.16-GCCcore-12.2.0

# Navigate to your working directory
cd /ceph/hpc/home/eujinyuanw/omics_analysis

# Activate conda environment if needed
source ~/miniconda/bin/activate scdiffusion

# Run your GPU task
# python your_script.py
CUDA_VISIBLE_DEVICES=0 python benchmark/scDiffusion/classifier_train.py --data_dir '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/data/data/tabula_muris/all.h5ad' --model_path "/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/checkpoint/classifier_2" \
         --iterations 40000 --vae_path '/ceph/hpc/home/eujinyuanw/omics_analysis/benchmark/scDiffusion/checkpoint/AE/my_VAE/model_seed=0_step=0.pt'
