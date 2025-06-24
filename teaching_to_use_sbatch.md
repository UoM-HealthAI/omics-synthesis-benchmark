# SLURM and sbatch Tutorial

## Understanding Your GPU Allocation Example

Your command:
```bash
srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty nvidia-smi
```

This allocates 1 A100 GPU (40GB VRAM) for 5 minutes in interactive mode and runs `nvidia-smi` to check GPU status.

## Key SLURM Commands

### `srun` vs `sbatch`
- **`srun`**: Interactive jobs (immediate execution, terminal stays connected)
- **`sbatch`**: Batch jobs (submit and disconnect, runs in background)

## Basic sbatch Script Structure

Create a `.sbatch` file:

```bash
#!/bin/bash
#SBATCH --job-name=my_job           # Job name
#SBATCH --partition=gpu             # Queue/partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --time=01:00:00            # Time limit (HH:MM:SS)
#SBATCH --mem=16G                  # Memory request
#SBATCH --cpus-per-task=4          # CPU cores
#SBATCH --output=logs/%j.out       # Output file (%j = job ID)
#SBATCH --error=logs/%j.err        # Error file

# Your commands here
python your_script.py
```

## GPU Resource Options

### Basic GPU allocation:
```bash
#SBATCH --gres=gpu:1               # Any 1 GPU
#SBATCH --gres=gpu:a100:1          # Specific GPU type
#SBATCH --gres=gpu:2               # 2 GPUs
```

### Memory and CPU for GPU jobs:
```bash
#SBATCH --mem=32G                  # System RAM
#SBATCH --cpus-per-task=8          # CPU cores (4-8 per GPU typical)
```

## Common sbatch Commands

```bash
# Submit job
sbatch my_script.sbatch

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# View job details
scontrol show job <job_id>

# Check available partitions
sinfo

# Check GPU availability
sinfo -p gpu --format="%.15N %.6D %.11T %.4c %.8z %.6m %.8G"
```

## Example Scripts

### 1. Simple Python GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=gpu_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/training_%j.out

# Load modules if needed
module load python/3.9

# Activate conda environment
source activate your_env

# Run your script
python train_model.py
```

### 2. Multi-GPU Job
```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

export CUDA_VISIBLE_DEVICES=0,1
python distributed_training.py
```

### 3. Array Job (Multiple Similar Tasks)
```bash
#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-10               # Run 10 instances
#SBATCH --time=01:00:00
#SBATCH --output=logs/array_%A_%a.out

# Use $SLURM_ARRAY_TASK_ID for different inputs
python process_data.py --task-id $SLURM_ARRAY_TASK_ID
```

## Best Practices

### 1. Resource Estimation
```bash
# Start with shorter time limits, extend if needed
#SBATCH --time=00:30:00    # 30 minutes initially

# Monitor resource usage:
sstat -j <job_id>
```

### 2. File Organization
```bash
# Create logs directory
mkdir -p logs

# Use meaningful job names
#SBATCH --job-name=scgan_training_epoch50
```

### 3. Environment Setup
```bash
# In your script:
source ~/.bashrc
conda activate myenv
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 4. GPU Memory Management
```bash
# Check GPU memory in your script:
nvidia-smi
# Or in Python:
import torch
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

## Understanding Your System

Based on your `nvidia-smi` output:
- **GPU**: NVIDIA A100-SXM4-40GB (high-end datacenter GPU)
- **VRAM**: 40,960 MiB (40GB) - excellent for large models
- **CUDA**: Version 12.7
- **Driver**: 565.57.01

## Interactive vs Batch Usage

### Interactive (srun) - for testing:
```bash
# Get interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash

# Test your code interactively
python test_script.py

# Exit when done
exit
```

### Batch (sbatch) - for production:
```bash
# Submit and disconnect
sbatch my_training.sbatch

# Check progress
tail -f logs/training_12345.out
```

## Troubleshooting

### Common Issues:

1. **Job pending**: Check resource availability
   ```bash
   squeue -u $USER
   sinfo -p gpu
   ```

2. **Out of memory**: Reduce batch size or request more memory
   ```bash
   #SBATCH --mem=64G  # Increase system RAM
   ```

3. **Time limit exceeded**: Extend time or optimize code
   ```bash
   #SBATCH --time=12:00:00  # 12 hours
   ```

4. **GPU not detected**: Verify CUDA setup
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   nvidia-smi
   ```

### Monitoring Jobs:
```bash
# Watch job queue
watch squeue -u $USER

# Monitor GPU usage during job
srun --jobid=<job_id> --pty nvidia-smi -l 1
```

## Example Workflow

1. **Test interactively**:
   ```bash
   srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
   python test_model.py
   ```

2. **Create batch script**:
   ```bash
   vim train_model.sbatch
   ```

3. **Submit job**:
   ```bash
   sbatch train_model.sbatch
   ```

4. **Monitor**:
   ```bash
   squeue -u $USER
   tail -f logs/training_*.out
   ```

This should get you started with SLURM and GPU scheduling on your HPC system!