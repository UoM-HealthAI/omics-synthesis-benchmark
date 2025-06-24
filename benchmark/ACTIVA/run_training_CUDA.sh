# nohup bash ./run_training_CUDA.sh &
# Load necessary modules (adjust based on your cluster)
# module load python/3.8
# module load cuda/11.1

# Activate conda environment
source activate activa

# Set CUDA devices
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Fix protobuf issue
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Default parameters
DATASET=${1:-"tabula_muris"}
LR=${2:-0.0002}
EPOCHS=${3:-500}
BATCH_SIZE=${4:-1024}
WORKERS=${5:-200}
OUTPUT_DIR=${6:-"./outputs/test_CUDA/"}

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Print job info
echo "=========================================="
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