#!/bin/bash
#SBATCH --job-name=dev_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=dev_gpu_%j.out
#SBATCH --error=dev_gpu_%j.err

echo "=== GPU Development Session ==="
echo "Node: $SLURM_JOB_NODELIST"
echo "SSH command: ssh $SLURM_JOB_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "================================"
echo ""
echo "To connect: ssh $SLURM_JOB_NODELIST"
echo "To check GPU: nvidia-smi"
echo ""

# Keep session alive for 4 hours
sleep 4h