#!/bin/bash
#SBATCH --job-name=svgd_multinode
#SBATCH --output=svgd_multinode_%j.out
#SBATCH --error=svgd_multinode_%j.err
#
# Multi-Node Configuration
#SBATCH --nodes=4                    # Number of nodes (machines)
#SBATCH --ntasks-per-node=1          # One task (process) per node
#SBATCH --cpus-per-task=8            # CPUs per task (devices per node)
#
# Time and Resources
#SBATCH --time=01:00:00              # 1 hour time limit
#SBATCH --mem-per-cpu=4G             # Memory per CPU
#
# Partition (adjust for your cluster)
#SBATCH --partition=compute
#
# Optional: Email notifications
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=your.email@institution.edu

echo "========================================================================"
echo "SLURM Multi-Node SVGD Job"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "========================================================================"
echo ""

# Load required modules (adjust for your cluster)
# module load python/3.11
# module load cuda/12.0  # If using GPUs

# Activate conda/pixi environment
# source activate your_env
# OR for pixi:
# eval "$(pixi shell-hook)"

# Set up JAX coordinator
# The first node in the allocation becomes the coordinator
COORDINATOR_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_COORDINATOR_ADDRESS=$COORDINATOR_NODE
export JAX_COORDINATOR_PORT=12345

echo "JAX Coordinator: $SLURM_COORDINATOR_ADDRESS:$JAX_COORDINATOR_PORT"
echo ""

# Set XLA flags for CPU devices (will be overridden per process)
export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# Set JAX configuration
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

# Print environment
echo "Environment:"
echo "  XLA_FLAGS: $XLA_FLAGS"
echo "  JAX_PLATFORMS: $JAX_PLATFORMS"
echo "  JAX_ENABLE_X64: $JAX_ENABLE_X64"
echo ""

# Run the distributed computation
# srun launches one process per task (4 processes total, one per node)
echo "Starting multi-node computation..."
echo ""

srun python examples/slurm_multinode_example.py

echo ""
echo "========================================================================"
echo "Job completed at $(date)"
echo "========================================================================"
