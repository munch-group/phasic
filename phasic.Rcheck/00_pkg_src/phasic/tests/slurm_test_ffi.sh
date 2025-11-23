#!/bin/bash
#SBATCH --job-name=phasic_ffi_test
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_ffi_test_%j.log
#SBATCH --error=slurm_ffi_test_%j.err

# SLURM batch script for testing phasic FFI functions across multiple nodes
#
# This script runs the multi-node FFI tests to verify that:
# 1. JAX distributed initialization works correctly
# 2. FFI functions serialize/deserialize across node boundaries
# 3. Thread-local caching works independently on each node
# 4. Results are numerically consistent across all nodes
#
# Usage:
#   sbatch tests/slurm_test_ffi.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm_ffi_test_<JOBID>.log

echo "=========================================="
echo "Phasic Multi-Node FFI Tests"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Nodelist: $SLURM_NODELIST"
echo "=========================================="

# Load modules if needed (adjust for your cluster)
# module load python/3.13
# module load gcc/11.2.0

# Activate conda/virtual environment if needed
# source /path/to/venv/bin/activate

# Set JAX/XLA flags for distributed execution
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=${SLURM_CPUS_PER_TASK}"

# Optional: Set phasic CPU count (otherwise auto-detected)
# export PTDALG_CPUS=${SLURM_CPUS_PER_TASK}

# Print environment info from first task
if [ "$SLURM_PROCID" == "0" ]; then
    echo ""
    echo "Environment:"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version)"
    echo "  Working directory: $(pwd)"
    echo "  PTDALG_CPUS: ${PTDALG_CPUS:-auto}"
    echo ""
fi

# Run tests with srun (distributes across nodes)
echo "Starting multi-node FFI tests..."
echo ""

srun python -m pytest tests/test_slurm_multinode_ffi.py -v -s

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests PASSED"
else
    echo "❌ Tests FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
