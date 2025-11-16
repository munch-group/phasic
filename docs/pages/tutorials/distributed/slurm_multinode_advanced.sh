#!/bin/bash
#SBATCH --job-name=svgd_multinode_advanced
#SBATCH --output=logs/svgd_%j.out
#SBATCH --error=logs/svgd_%j.err
#
# ============================================================================
# Multi-Node Configuration for Large-Scale SVGD
# ============================================================================
# This configuration is optimized for SVGD with 100+ particles
#
#SBATCH --nodes=8                    # 8 nodes (machines)
#SBATCH --ntasks-per-node=1          # 1 process per node
#SBATCH --cpus-per-task=16           # 16 CPUs per process
#
# Total compute: 8 nodes × 16 CPUs = 128 devices
# With 512 particles: 4 particles per device
#
# ============================================================================
# GPU Configuration (Optional - uncomment if using GPUs)
# ============================================================================
##SBATCH --gres=gpu:4                # 4 GPUs per node
##SBATCH --gpus-per-task=4           # 4 GPUs per task
## Total GPUs: 8 nodes × 4 GPUs = 32 GPUs
#
# ============================================================================
# Time and Memory
# ============================================================================
#SBATCH --time=04:00:00              # 4 hour time limit
#SBATCH --mem-per-cpu=8G             # 8GB per CPU
#
# ============================================================================
# Queue/Partition Configuration
# ============================================================================
#SBATCH --partition=compute          # Adjust for your cluster
#SBATCH --qos=normal                 # Quality of service
#
# ============================================================================
# Job Management
# ============================================================================
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@institution.edu
#SBATCH --requeue                    # Requeue if preempted
#SBATCH --exclusive                  # Exclusive node access

# ============================================================================
# Environment Setup
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "========================================================================"
echo "Multi-Node SVGD with JAX Distributed Computing"
echo "========================================================================"
echo "Job started at: $(date)"
echo ""
echo "SLURM Configuration:"
echo "  Job ID:           $SLURM_JOB_ID"
echo "  Job Name:         $SLURM_JOB_NAME"
echo "  Nodes:            $SLURM_JOB_NUM_NODES"
echo "  Tasks per node:   $SLURM_NTASKS_PER_NODE"
echo "  CPUs per task:    $SLURM_CPUS_PER_TASK"
echo "  Total tasks:      $SLURM_NTASKS"
echo "  Total CPUs:       $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "  Node list:        $SLURM_JOB_NODELIST"
echo "  Working dir:      $(pwd)"
echo "========================================================================"
echo ""

# Create log directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Module Loading (Adjust for your cluster)
# ============================================================================

# Example for typical HPC cluster:
# module purge
# module load python/3.11
# module load gcc/11.2.0
# module load openmpi/4.1.4
# module load cuda/12.0  # If using GPUs

# Print loaded modules
echo "Loaded modules:"
module list 2>&1
echo ""

# ============================================================================
# Python Environment Activation
# ============================================================================

# Option 1: Conda environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate phasic

# Option 2: Pixi environment (recommended)
eval "$(pixi shell-hook)"

# Verify Python and JAX installation
echo "Python environment:"
which python
python --version
python -c "import jax; print(f'JAX version: {jax.__version__}')"
python -c "import phasic; print('phasic imported successfully')"
echo ""

# ============================================================================
# JAX Distributed Configuration
# ============================================================================

# Get the first node as coordinator
COORDINATOR_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export SLURM_COORDINATOR_ADDRESS=$COORDINATOR_NODE

# Use a fixed port (must be available on all nodes)
# You may need to adjust this if the port is in use
export JAX_COORDINATOR_PORT=12345

echo "JAX Distributed Configuration:"
echo "  Coordinator:      $SLURM_COORDINATOR_ADDRESS:$JAX_COORDINATOR_PORT"
echo "  Process count:    $SLURM_NTASKS"
echo "  Devices per proc: $SLURM_CPUS_PER_TASK"
echo "  Total devices:    $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo ""

# ============================================================================
# JAX Environment Variables
# ============================================================================

# Force CPU device count (will be set per process via XLA_FLAGS)
export XLA_FLAGS="--xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

# Platform selection
export JAX_PLATFORMS=cpu  # or "gpu" if using GPUs

# Enable 64-bit precision
export JAX_ENABLE_X64=1

# Disable preallocation (for CPU, helps with memory management)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Enable GPU if requested (uncomment if using GPUs)
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export JAX_PLATFORMS=gpu

# Performance tuning
export XLA_FLAGS="$XLA_FLAGS --xla_cpu_multi_thread_eigen=true"
export XLA_FLAGS="$XLA_FLAGS --xla_force_host_platform_device_count=$SLURM_CPUS_PER_TASK"

echo "JAX Configuration:"
echo "  XLA_FLAGS:                    $XLA_FLAGS"
echo "  JAX_PLATFORMS:                $JAX_PLATFORMS"
echo "  JAX_ENABLE_X64:               $JAX_ENABLE_X64"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo ""

# ============================================================================
# Network Configuration for Multi-Node Communication
# ============================================================================

# Set timeouts for network operations (adjust as needed)
export NCCL_SOCKET_IFNAME=ib0  # Adjust for your cluster's network interface
export NCCL_DEBUG=INFO          # Set to INFO or WARN for debugging

# For InfiniBand clusters (uncomment if applicable):
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=5

# For Ethernet clusters (uncomment if applicable):
# export NCCL_SOCKET_IFNAME=eth0

echo "Network Configuration:"
echo "  NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-not set}"
echo "  NCCL_DEBUG:         ${NCCL_DEBUG:-not set}"
echo ""

# ============================================================================
# Computation Parameters
# ============================================================================

# Number of particles (should be divisible by total device count)
TOTAL_DEVICES=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
PARTICLES_PER_DEVICE=4
NUM_PARTICLES=$((TOTAL_DEVICES * PARTICLES_PER_DEVICE))

echo "Computation Parameters:"
echo "  Total devices:      $TOTAL_DEVICES"
echo "  Particles per dev:  $PARTICLES_PER_DEVICE"
echo "  Total particles:    $NUM_PARTICLES"
echo ""

# ============================================================================
# Run Multi-Node Computation
# ============================================================================

echo "========================================================================"
echo "Starting multi-node SVGD computation..."
echo "========================================================================"
echo ""

# srun launches one process per task (distributed across nodes)
# Each process will initialize JAX distributed and participate in pmap
srun --kill-on-bad-exit=1 \
     --cpu-bind=cores \
     python examples/slurm_multinode_example.py

EXITCODE=$?

echo ""
echo "========================================================================"
echo "Multi-Node SVGD Completed"
echo "========================================================================"
echo "Exit code: $EXITCODE"
echo "Completed at: $(date)"
echo ""

# ============================================================================
# Post-Processing and Cleanup
# ============================================================================

if [ $EXITCODE -eq 0 ]; then
    echo "SUCCESS: Multi-node SVGD completed successfully"

    # Optional: Collect results from all nodes
    # python examples/collect_results.py --job-id $SLURM_JOB_ID

else
    echo "FAILURE: Multi-node SVGD failed with exit code $EXITCODE"
    echo "Check error log: logs/svgd_${SLURM_JOB_ID}.err"
fi

echo ""
echo "========================================================================"

exit $EXITCODE
