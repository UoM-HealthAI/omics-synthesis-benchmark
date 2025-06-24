#!/bin/bash
# ACTIVA Experiment Steps - JY
# This file documents all steps for running ACTIVA experiments

echo "=================================================="
echo "ACTIVA Experiment Steps"
echo "Date: $(date)"
echo "=================================================="

# Step 1: Activate conda environment
echo "Step 1: Activating conda environment"
source activate activa

# Step 2: Fix PyTorch for A100 GPU compatibility
echo "Step 2: Fixing PyTorch for A100 GPU compatibility"
./fix_a100_pytorch.sh

# Step 3: Prepare Tabula Muris data (add train/test split)
echo "Step 3: Preparing Tabula Muris data with train/test split"
python prepare_tabula_muris.py

# Step 4: Check GPU availability
echo "Step 4: Checking GPU availability"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Step 5: Test ACTIVA with GPU mode (1 epoch, local test)
echo "Step 5: Testing ACTIVA on GPU with 1 epoch (local test)"
# Fix protobuf issue and test GPU training
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python ACTIVA.py --example_data tabula_muris --nEpochs 100 --batchSize 256 --workers 32 --print_frequency 1 --lr 0.0002 --lr_e 0.0002 --lr_g 0.0002

# Step 6: Test ACTIVA with CPU mode (1 epoch) - if GPU fails
echo "Step 6: Testing ACTIVA on CPU with 1 epoch (backup test)"
# Note: CPU patch applied to enable CPU training
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python ACTIVA.py --example_data tabula_muris --cpu --nEpochs 1 --batchSize 16 --workers 4 --print_frequency 1

# Step 7: Submit full training job with GPU
echo "Step 7: Submitting GPU training job (after local GPU test success)"
# sbatch train_activa.sbatch tabula_muris

# Optional: Monitor job
echo "To monitor job:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/activa_JOBID.out"

# Local GPU test instructions
echo ""
echo "=== Local GPU Test Instructions ==="
echo "1. Run: ./experiment_steps_jy.sh"
echo "2. Check Step 4 output for GPU detection"
echo "3. If Step 5 (GPU test) succeeds, uncomment Step 7 to submit full job"
echo "4. If Step 5 fails, uncomment Step 6 to test CPU fallback"

echo "=================================================="