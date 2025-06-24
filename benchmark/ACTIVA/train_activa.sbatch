#!/bin/bash
#SBATCH --job-name=activa_train
#SBATCH --output=outputs/activa_%j.out
#SBATCH --error=outputs/activa_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Load necessary modules (adjust based on your cluster)
# module load python/3.8
# module load cuda/11.1

# Activate conda environment
source activate activa

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Fix protobuf issue
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Default parameters
DATASET=${1:-"tabula_muris"}
LR=${2:-0.0002}
EPOCHS=${3:-500}
BATCH_SIZE=${4:-128}
WORKERS=${5:-16}
OUTPUT_DIR=${6:-"./outputs/${DATASET}_${SLURM_JOB_ID}/"}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Print job info
echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Running on node: ${SLURMD_NODENAME}"
echo "Dataset: ${DATASET}"
echo "Learning rate: ${LR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Workers: ${WORKERS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Run training
python ACTIVA.py \
    --example_data ${DATASET} \
    --lr ${LR} \
    --lr_e ${LR} \
    --lr_g ${LR} \
    --nEpochs ${EPOCHS} \
    --batchSize ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --outf ${OUTPUT_DIR} \
    --tensorboard \
    --print_frequency 10 \
    --cf_print_frequency 5

echo "Training completed!"