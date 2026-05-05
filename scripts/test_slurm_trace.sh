#!/bin/bash
#SBATCH --job-name=phasic_trace_slurm_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=phasic_trace_slurm_test_%j.out
#SBATCH --error=phasic_trace_slurm_test_%j.err
#
# Manual driver for the SLURM-multi-node trace cache test in
# tests/pytest/inference/test_trace_jax_compat.py.
#
# Submit from a machine that has phasic installed and a writable home dir
# accessible from every compute node:
#
#     sbatch scripts/test_slurm_trace.sh
#
# Or to run interactively on the SLURM allocation:
#
#     srun --nodes=2 --ntasks-per-node=1 --cpus-per-task=4 --pty bash
#     bash scripts/test_slurm_trace.sh
#
# The test inside test_trace_jax_compat.py is gated on $SLURM_JOB_ID, so
# it skips cleanly outside SLURM and only runs when launched this way.

set -euo pipefail

# Use a per-job trace path under $HOME so all ranks see the same file
# (assumes a shared filesystem; adjust if your cluster does not have one).
export PHASIC_TEST_TRACE_PATH="${HOME}/phasic_test_slurm_trace_${SLURM_JOB_ID:-local}.json"

# Cap each rank's CPU count to match --cpus-per-task. Lets phasic's
# auto-multi-CPU configuration play nicely with SLURM's cpu-binding.
export PTDALG_CPUS="${SLURM_CPUS_PER_TASK:-4}"

# Run via srun so SLURM environment variables (SLURM_PROCID, SLURM_NTASKS,
# SLURM_NODEID) are populated per rank. The test reads those to coordinate
# rank-0 record + all-rank load.
srun --kill-on-bad-exit=1 \
    pixi run -- pytest \
        tests/pytest/inference/test_trace_jax_compat.py::test_slurm_multi_node_trace_cache_round_trip \
        -v -s
