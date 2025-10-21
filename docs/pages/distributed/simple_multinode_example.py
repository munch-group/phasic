#!/usr/bin/env python3
"""
Simple Multi-Node SVGD Example (Using Distributed Utils)

This example demonstrates how clean multi-node code can be when using
the distributed_utils module. Compare this to slurm_multinode_example.py
which has 150+ lines of boilerplate - this version has just 80 lines!

All SLURM detection, coordinator setup, and JAX initialization is handled
automatically by initialize_distributed().

Usage:
    # Single-node (local testing)
    python simple_multinode_example.py

    # Multi-node (SLURM)
    sbatch <(python generate_slurm_script.py --profile small --script simple_multinode_example.py)

Author: phasic Team
Date: 2025-10-07
"""

import json
import numpy as np
import jax
import jax.numpy as jnp

# Import the magic module that handles all boilerplate!
from phasic.distributed_utils import initialize_distributed
from phasic.ffi_wrappers import compute_pmf_and_moments_ffi


def create_erlang_structure(num_stages: int = 3) -> str:
    """Create parameterized Erlang distribution structure."""
    states = [[0]] + [[i+1] for i in range(1, num_stages + 1)] + [[1]]
    edges = []
    start_edges = [[1, 1.0]]
    param_edges = [[i, i+1, 1.0] for i in range(1, num_stages + 1)]
    start_param_edges = []

    structure = {
        'states': states,
        'edges': edges,
        'start_edges': start_edges,
        'param_edges': param_edges,
        'start_param_edges': start_param_edges,
        'param_length': 1,
        'state_length': 1,
        'n_vertices': len(states)
    }
    return json.dumps(structure)


def run_distributed_svgd(dist_info, erlang_json: str, num_particles: int):
    """Run SVGD with particles distributed across all devices."""
    observations = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target_moments = jnp.array([3.0, 12.0, 60.0])

    # Initialize particles for this process
    np.random.seed(42 + dist_info.process_id)
    particles_per_device = num_particles // dist_info.global_device_count
    local_particles = np.random.uniform(
        0.5, 2.0,
        (dist_info.local_device_count, particles_per_device, 1)
    )
    local_particles_jax = jnp.array(local_particles)

    # Define objective
    def svgd_objective(theta):
        pdf, moments = compute_pmf_and_moments_ffi(
            erlang_json, theta, observations, nr_moments=3, discrete=False
        )
        log_likelihood = jnp.sum(jnp.log(pdf + 1e-10))
        moment_penalty = jnp.sum((moments - target_moments)**2)
        return log_likelihood - 0.1 * moment_penalty

    # Compute objectives with pmap
    vmap_objective = jax.vmap(svgd_objective)
    pmap_objective = jax.pmap(vmap_objective)

    print(f"[Process {dist_info.process_id}] Computing objectives...", flush=True)
    objectives = pmap_objective(local_particles_jax)

    # Find best
    best_obj = float(jnp.max(objectives))
    best_idx = int(jnp.argmax(objectives.flatten()))

    print(f"[Process {dist_info.process_id}] Best objective: {best_obj:.4f}", flush=True)

    return objectives, best_obj


def main():
    """Main execution."""
    print("=" * 80)
    print("Simple Multi-Node SVGD Example")
    print("=" * 80)
    print()

    # ========================================================================
    # THIS IS ALL YOU NEED! One line handles everything:
    # - SLURM detection
    # - Coordinator setup
    # - JAX distributed initialization
    # - Device configuration
    # ========================================================================
    dist_info = initialize_distributed()

    print()
    print(f"[Process {dist_info.process_id}] Ready for computation!")
    print()

    # Create graph structure
    erlang_json = create_erlang_structure(num_stages=3)

    # Compute particles: 4 per device
    num_particles = dist_info.global_device_count * 4

    print(f"[Process {dist_info.process_id}] Configuration:")
    print(f"  Total particles: {num_particles}")
    print(f"  Particles per device: {num_particles // dist_info.global_device_count}")
    print()

    # Run SVGD
    objectives, best_obj = run_distributed_svgd(dist_info, erlang_json, num_particles)

    # Only coordinator prints summary
    if dist_info.is_coordinator:
        print()
        print("=" * 80)
        print("Computation Complete!")
        print("=" * 80)
        print(f"Total devices: {dist_info.global_device_count}")
        print(f"Total particles: {num_particles}")
        print(f"Best objective: {best_obj:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
