#!/bin/bash
#SBATCH --job-name=max_resources_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=max_resources_test_%j.out
#SBATCH --error=max_resources_test_%j.err

# Print resource allocation information
echo "=== Resource Allocation Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Time limit: $SLURM_JOB_TIME_LIMIT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "======================================="

# Check actual hardware resources
echo ""
echo "=== Hardware Information ==="
echo "CPU info:"
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket"
echo ""
echo "Memory info:"
free -h
echo ""
echo "GPU info:"
nvidia-smi
echo "=========================="

# Keep job running for testing (can be interrupted)
echo ""
echo "Job started at: $(date)"
echo "Job will run for 3 hours. You can connect and test your applications."
echo "To connect to this node, use: ssh $SLURM_JOB_NODELIST"

# Simple resource monitoring loop
for i in {1..1080}; do  # 1080 * 10 seconds = 3 hours
    sleep 10
    if [ $((i % 360)) -eq 0 ]; then  # Print every hour
        echo "$(date): Job still running ($(($i/360)) hour(s) elapsed)"
    fi
done

echo "Job completed at: $(date)"