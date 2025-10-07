#!/usr/bin/env python3
"""
Multi-Node SVGD with JAX Distributed Computing on SLURM

This script demonstrates how to distribute SVGD particle computation across
multiple nodes in a SLURM cluster, with each node having multiple CPUs.

Architecture:
- Multiple nodes (machines) in SLURM cluster
- Multiple CPUs per node
- JAX distributed initialization for inter-node communication
- pmap for intra-node parallelization
- Global particle distribution across all devices

Example SLURM Setup:
- 4 nodes × 8 CPUs/node = 32 total devices
- 128 particles = 4 particles per device
- Efficient for large-scale SVGD inference

Author: Claude Code
Date: 2025-10-07
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
import json
import time
from typing import Optional

# Enable 64-bit types in JAX
config.update('jax_enable_x64', True)

# ============================================================================
# Step 1: JAX Distributed Initialization for Multi-Node
# ============================================================================

def initialize_jax_distributed():
    """
    Initialize JAX for distributed computing across multiple nodes.

    This function should be called at the start of your script when running
    on a SLURM cluster with multiple nodes.

    Environment Variables Required:
    - SLURM_JOB_ID: Job identifier
    - SLURM_PROCID: Process rank (0 to N-1)
    - SLURM_NTASKS: Total number of processes
    - SLURM_JOB_NODELIST: List of nodes
    - SLURM_CPUS_PER_TASK: CPUs per task

    Returns:
    -------
    dict
        Information about the distributed setup
    """
    # Check if running under SLURM
    if 'SLURM_JOB_ID' not in os.environ:
        print("WARNING: Not running under SLURM. Using single-node setup.")
        return {
            'num_nodes': 1,
            'num_processes': 1,
            'process_id': 0,
            'local_device_count': len(jax.local_devices()),
            'global_device_count': len(jax.devices()),
        }

    # Get SLURM environment variables
    job_id = os.environ['SLURM_JOB_ID']
    process_id = int(os.environ.get('SLURM_PROCID', 0))
    num_processes = int(os.environ.get('SLURM_NTASKS', 1))
    cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    print(f"SLURM Job ID: {job_id}")
    print(f"Process ID: {process_id}/{num_processes}")
    print(f"CPUs per task: {cpus_per_task}")

    # Set XLA flags for local CPU devices
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpus_per_task}'

    # Initialize JAX distributed
    # For SLURM, we need to construct the coordinator address
    # Typically the first node in the allocation
    coordinator_address = os.environ.get('SLURM_COORDINATOR_ADDRESS', None)

    if coordinator_address is None:
        # Try to get first node from nodelist
        import subprocess
        nodelist = os.environ.get('SLURM_JOB_NODELIST', '')
        if nodelist:
            # Use scontrol to expand nodelist
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', nodelist],
                capture_output=True, text=True
            )
            nodes = result.stdout.strip().split('\n')
            coordinator_address = nodes[0] if nodes else 'localhost'
        else:
            coordinator_address = 'localhost'

    # Port for coordinator (should be fixed across all processes)
    coordinator_port = int(os.environ.get('JAX_COORDINATOR_PORT', 12345))

    print(f"Coordinator: {coordinator_address}:{coordinator_port}")

    # Initialize JAX distributed
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_address}:{coordinator_port}",
        num_processes=num_processes,
        process_id=process_id,
    )

    # Get device information
    local_devices = jax.local_devices()
    global_devices = jax.devices()

    info = {
        'job_id': job_id,
        'process_id': process_id,
        'num_processes': num_processes,
        'cpus_per_task': cpus_per_task,
        'coordinator': f"{coordinator_address}:{coordinator_port}",
        'local_device_count': len(local_devices),
        'global_device_count': len(global_devices),
        'local_devices': local_devices,
        'global_devices': global_devices,
    }

    print(f"Local devices: {len(local_devices)}")
    print(f"Global devices: {len(global_devices)}")
    print(f"Devices: {local_devices}")

    return info


# ============================================================================
# Step 2: Multi-Node SVGD with Global pmap
# ============================================================================

def distributed_svgd_example(
    erlang_json: str,
    num_particles: int,
    observations: np.ndarray,
    target_moments: np.ndarray,
    dist_info: dict
):
    """
    Run SVGD with particles distributed across multiple nodes.

    Parameters:
    -----------
    erlang_json : str
        Graph structure JSON
    num_particles : int
        Total number of particles (should be divisible by global_device_count)
    observations : np.ndarray
        Observation data for likelihood
    target_moments : np.ndarray
        Target moments for regularization
    dist_info : dict
        Distribution information from initialize_jax_distributed()
    """
    from ptdalgorithms.ffi_wrappers import compute_pmf_and_moments_ffi

    global_device_count = dist_info['global_device_count']
    process_id = dist_info['process_id']

    print(f"\n[Process {process_id}] Setting up distributed SVGD")
    print(f"  Total particles: {num_particles}")
    print(f"  Global devices: {global_device_count}")
    print(f"  Particles per device: {num_particles // global_device_count}")

    # Ensure particles divide evenly across all global devices
    if num_particles % global_device_count != 0:
        raise ValueError(
            f"num_particles ({num_particles}) must be divisible by "
            f"global_device_count ({global_device_count})"
        )

    particles_per_device = num_particles // global_device_count

    # Initialize particles (each process initializes its local portion)
    # For reproducibility, use process_id in the random seed
    local_device_count = dist_info['local_device_count']
    particles_per_local_device = particles_per_device  # Since we're using pmap

    # Create particle batch for this process
    # Shape: (local_device_count, particles_per_device, n_params)
    np.random.seed(42 + process_id)  # Different seed per process
    local_particles = np.random.uniform(
        0.5, 2.0,
        (local_device_count, particles_per_device, 1)
    )
    local_particles_jax = jnp.array(local_particles)

    print(f"[Process {process_id}] Local particles shape: {local_particles_jax.shape}")

    # Convert observations and target moments to JAX arrays
    observations_jax = jnp.array(observations)
    target_moments_jax = jnp.array(target_moments)

    # Define SVGD objective for a single particle
    def svgd_objective_single(theta):
        """Compute objective for a single particle."""
        pdf, moments = compute_pmf_and_moments_ffi(
            erlang_json, theta, observations_jax,
            nr_moments=len(target_moments_jax), discrete=False
        )
        log_likelihood = jnp.sum(jnp.log(pdf + 1e-10))
        moment_penalty = jnp.sum((moments - target_moments_jax)**2)
        return log_likelihood - 0.1 * moment_penalty

    # Use pmap to distribute across local devices on this node
    # This will automatically participate in the global pmap across all nodes
    print(f"[Process {process_id}] Setting up pmap...")

    # Create pmapped computation
    # vmap over particles within each device, pmap across devices
    vmap_objective = jax.vmap(svgd_objective_single)
    pmap_objective = jax.pmap(vmap_objective, axis_name='devices')

    # Compute objectives
    print(f"[Process {process_id}] Computing objectives...")
    start_time = time.time()
    local_objectives = pmap_objective(local_particles_jax)
    elapsed = (time.time() - start_time) * 1000

    print(f"[Process {process_id}] Computation time: {elapsed:.2f}ms")
    print(f"[Process {process_id}] Local objectives shape: {local_objectives.shape}")
    print(f"[Process {process_id}] Local objectives: {local_objectives.flatten()[:5]}...")

    # Find best particle locally
    local_objectives_flat = local_objectives.flatten()
    local_best_idx = jnp.argmax(local_objectives_flat)
    local_best_objective = local_objectives_flat[local_best_idx]

    print(f"[Process {process_id}] Local best objective: {local_best_objective:.4f}")

    # For finding global best, we would need to use jax.lax.pmax or MPI
    # This is left as an exercise - typically you'd use:
    # global_best = jax.lax.pmax(local_best_objective, axis_name='devices')

    return {
        'local_objectives': local_objectives,
        'local_particles': local_particles_jax,
        'local_best_objective': float(local_best_objective),
        'local_best_idx': int(local_best_idx),
        'elapsed_ms': elapsed,
    }


# ============================================================================
# Step 3: Main Execution
# ============================================================================

def main():
    """Main execution for multi-node SVGD."""

    print("=" * 80)
    print("Multi-Node SVGD with JAX Distributed Computing")
    print("=" * 80)
    print()

    # Initialize JAX distributed
    dist_info = initialize_jax_distributed()
    process_id = dist_info.get('process_id', 0)

    print()
    print(f"[Process {process_id}] JAX Distributed Initialized")
    print(f"  Processes: {dist_info.get('num_processes', 1)}")
    print(f"  Local devices: {dist_info['local_device_count']}")
    print(f"  Global devices: {dist_info['global_device_count']}")
    print()

    # Create Erlang distribution structure
    # (Only process 0 needs to do this, but it's lightweight so all can do it)
    num_stages = 3
    states = [[0]] + [[i+1] for i in range(1, num_stages + 1)] + [[1]]
    edges = []
    start_edges = [[1, 1.0]]
    param_edges = []
    for i in range(1, num_stages + 1):
        param_edges.append([i, i+1, 1.0])
    start_param_edges = []

    erlang_structure = {
        'states': states,
        'edges': edges,
        'start_edges': start_edges,
        'param_edges': param_edges,
        'start_param_edges': start_param_edges,
        'param_length': 1,
        'state_length': 1,
        'n_vertices': len(states)
    }
    erlang_json = json.dumps(erlang_structure)

    # Observations and target moments
    observations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target_moments = np.array([3.0, 12.0, 60.0])

    # Number of particles (should be divisible by global device count)
    global_device_count = dist_info['global_device_count']
    num_particles = global_device_count * 4  # 4 particles per device

    print(f"[Process {process_id}] Starting SVGD with {num_particles} particles")
    print()

    # Run distributed SVGD
    try:
        results = distributed_svgd_example(
            erlang_json=erlang_json,
            num_particles=num_particles,
            observations=observations,
            target_moments=target_moments,
            dist_info=dist_info
        )

        print()
        print(f"[Process {process_id}] Results:")
        print(f"  Computation time: {results['elapsed_ms']:.2f}ms")
        print(f"  Local best objective: {results['local_best_objective']:.4f}")
        print()

        # Only process 0 prints summary
        if process_id == 0:
            print("=" * 80)
            print("Multi-Node SVGD Complete!")
            print("=" * 80)
            print()
            print("Summary:")
            print(f"  Total devices: {global_device_count}")
            print(f"  Total particles: {num_particles}")
            print(f"  Particles per device: {num_particles // global_device_count}")
            print(f"  Computation time: {results['elapsed_ms']:.2f}ms")
            print()

    except Exception as e:
        print(f"[Process {process_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
