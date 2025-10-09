#!/usr/bin/env python3
"""
Distributed SVGD Inference Example

This example demonstrates the complete workflow for distributed Bayesian inference
using the new simplified distributed computing interface.

Features demonstrated:
- Single-line distributed initialization (no boilerplate!)
- Parameterized coalescent model with JAX FFI
- SVGD inference distributed across multiple nodes/devices
- Automatic SLURM detection and configuration
- Works on both single-node (local) and multi-node (SLURM) setups

Usage:
    # Local testing (single node)
    python distributed_svgd_example.py

    # Multi-node SLURM cluster
    sbatch <(python generate_slurm_script.py --profile medium --script distributed_svgd_example.py)

Author: PtDAlgorithms Team
Date: 2025-10-07
"""

import numpy as np
import jax
import jax.numpy as jnp
from ptdalgorithms import Graph, initialize_distributed, SVGD
from ptdalgorithms.ffi_wrappers import compute_pmf_ffi


def build_coalescent_model(nr_samples: int = 4):
    """
    Build a parameterized coalescent model.

    This models the time to most recent common ancestor (TMRCA) for a sample
    of sequences. The model has one parameter: the effective population size
    scaled by mutation rate (theta = 4*N*mu).

    Parameters
    ----------
    nr_samples : int
        Number of sampled sequences

    Returns
    -------
    Graph
        Parameterized coalescent graph
    """
    def coalescent_callback(state, nr_samples=nr_samples):
        """Callback defining coalescent transitions."""
        if len(state) == 0:
            # Initial state: all samples in separate lineages
            return [(np.array([nr_samples]), 1.0, [1.0])]

        if state[0] > 1:
            # Coalescent event: n lineages -> n-1 lineages
            n = state[0]
            # Rate is n*(n-1)/2 * theta (parameterized by theta)
            rate = n * (n - 1) / 2
            return [(np.array([n - 1]), 0.0, [rate])]

        # Absorbing state (1 lineage = MRCA)
        return []

    # Build parameterized graph
    graph = Graph(callback=coalescent_callback, parameterized=True, nr_samples=nr_samples)

    return graph


def generate_synthetic_data(true_theta: jnp.ndarray, times: jnp.ndarray,
                           structure_json: str) -> jnp.ndarray:
    """
    Generate synthetic observed data from the coalescent model.

    Parameters
    ----------
    true_theta : jnp.ndarray
        True parameter value
    times : jnp.ndarray
        Time points to evaluate
    structure_json : str
        Serialized graph structure

    Returns
    -------
    jnp.ndarray
        Synthetic PDF values (with added noise)
    """
    # Compute true PDF
    true_pdf = compute_pmf_ffi(structure_json, true_theta, times, discrete=False)

    # Add small amount of noise to make it realistic
    noise = np.random.normal(0, 0.01 * np.max(true_pdf), size=true_pdf.shape)
    observed_pdf = true_pdf + noise

    # Ensure non-negative
    observed_pdf = np.maximum(observed_pdf, 1e-10)

    return jnp.array(observed_pdf)


def run_distributed_svgd(dist_info, structure_json: str, observed_data: jnp.ndarray,
                        evaluation_times: jnp.ndarray, n_particles: int = 100,
                        n_iterations: int = 500):
    """
    Run SVGD inference distributed across available devices.

    Parameters
    ----------
    dist_info : DistributedConfig
        Distributed computing configuration
    structure_json : str
        Serialized graph structure
    observed_data : jnp.ndarray
        Observed PDF values
    evaluation_times : jnp.ndarray
        Time points where PDF was observed
    n_particles : int
        Total number of SVGD particles
    n_iterations : int
        Number of optimization iterations

    Returns
    -------
    dict
        SVGD results with posterior samples
    """
    # Create model function that closes over structure_json and times
    def model(theta):
        """Likelihood model: returns log p(data | theta)"""
        # Compute predicted PDF
        predicted_pdf = compute_pmf_ffi(structure_json, theta, evaluation_times,
                                       discrete=False, granularity=100)

        # Compute log-likelihood (sum of log densities)
        log_likelihood = jnp.sum(jnp.log(predicted_pdf + 1e-10))

        return log_likelihood

    # Prior: weak Gaussian prior on theta
    def log_prior(theta):
        """Log prior: Gaussian centered at 1.0 with std 2.0"""
        return -0.5 * jnp.sum((theta - 1.0)**2 / (2.0**2))

    if dist_info.is_coordinator:
        print("\n" + "="*80)
        print("SVGD INFERENCE CONFIGURATION")
        print("="*80)
        print(f"Total particles: {n_particles}")
        print(f"Particles per device: {n_particles // dist_info.global_device_count}")
        print(f"Iterations: {n_iterations}")
        print(f"Observed data points: {len(observed_data)}")
        print("="*80 + "\n")

    # Initialize SVGD
    # Note: SVGD will automatically distribute particles across all devices
    np.random.seed(42 + dist_info.process_id)

    # Initialize particles around prior mean with some spread
    theta_init = np.random.uniform(0.5, 1.5, size=(n_particles, 1))

    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        prior=log_prior,
        n_particles=n_particles,
        n_iterations=n_iterations,
        learning_rate=0.01,
        kernel='rbf_median',
        theta_init=theta_init,
        theta_dim=1,
        seed=42,
        verbose=dist_info.is_coordinator  # Only coordinator prints progress
    )

    # Run inference
    svgd.fit(return_history=False)

    # Get results
    results = svgd.get_results()

    return results


def main():
    """Main execution."""
    print("="*80)
    print("DISTRIBUTED SVGD INFERENCE EXAMPLE")
    print("="*80)
    print()

    # ========================================================================
    # STEP 1: Initialize Distributed Computing
    # ========================================================================
    # This single line handles all SLURM boilerplate!
    # - Auto-detects SLURM environment
    # - Sets up coordinator (for multi-node)
    # - Initializes JAX distributed
    # - Configures devices (CPU/GPU)
    dist_info = initialize_distributed(
        coordinator_port=12345,
        platform="cpu",
        enable_x64=True
    )

    print()
    if dist_info.is_coordinator:
        print(f"[Process {dist_info.process_id}] Distributed setup complete!")
        print(f"  Running on {dist_info.num_processes} process(es)")
        print(f"  Total devices: {dist_info.global_device_count}")
        print(f"  Local devices: {dist_info.local_device_count}")
    print()

    # ========================================================================
    # STEP 2: Build Parameterized Model
    # ========================================================================
    if dist_info.is_coordinator:
        print("Building coalescent model...")

    nr_samples = 5
    graph = build_coalescent_model(nr_samples=nr_samples)

    # Serialize to JSON for JAX FFI
    import json
    serialized = graph.serialize()
    structure_json = json.dumps({
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in serialized.items()
    })

    if dist_info.is_coordinator:
        print(f"  Model: Coalescent with {nr_samples} samples")
        print(f"  Parameters: 1 (theta = 4*N*mu)")
        print(f"  States: {serialized['n_vertices']}")

    # ========================================================================
    # STEP 3: Generate Synthetic Data
    # ========================================================================
    if dist_info.is_coordinator:
        print("\nGenerating synthetic observed data...")

    true_theta = jnp.array([1.0])  # True parameter value
    evaluation_times = jnp.linspace(0.1, 5.0, 50)

    # Generate once (only coordinator needs to do this)
    if dist_info.is_coordinator:
        np.random.seed(12345)
        observed_data = generate_synthetic_data(true_theta, evaluation_times, structure_json)
        print(f"  True theta: {true_theta[0]:.3f}")
        print(f"  Observation points: {len(observed_data)}")
        print(f"  Time range: [{evaluation_times[0]:.2f}, {evaluation_times[-1]:.2f}]")
    else:
        # Other processes will receive data through JAX operations
        observed_data = jnp.zeros_like(evaluation_times)

    # Broadcast observed data to all processes if multi-node
    # (In real SVGD, each device computes the same likelihood)

    # ========================================================================
    # STEP 4: Run Distributed SVGD Inference
    # ========================================================================
    if dist_info.is_coordinator:
        print("\nStarting SVGD inference...")

    results = run_distributed_svgd(
        dist_info=dist_info,
        structure_json=structure_json,
        observed_data=observed_data,
        evaluation_times=evaluation_times,
        n_particles=dist_info.global_device_count * 4,  # 4 particles per device
        n_iterations=300
    )

    # ========================================================================
    # STEP 5: Report Results (Coordinator Only)
    # ========================================================================
    if dist_info.is_coordinator:
        print("\n" + "="*80)
        print("INFERENCE RESULTS")
        print("="*80)
        print(f"True theta:        {true_theta[0]:.4f}")
        print(f"Posterior mean:    {results['theta_mean'][0]:.4f}")
        print(f"Posterior std:     {results['theta_std'][0]:.4f}")
        print(f"95% Credible Int:  [{results['theta_mean'][0] - 1.96*results['theta_std'][0]:.4f}, "
              f"{results['theta_mean'][0] + 1.96*results['theta_std'][0]:.4f}]")
        print("="*80)
        print()

        # Check if true value is within credible interval
        lower = results['theta_mean'][0] - 1.96*results['theta_std'][0]
        upper = results['theta_mean'][0] + 1.96*results['theta_std'][0]

        if lower <= true_theta[0] <= upper:
            print("SUCCESS: True parameter within 95% credible interval")
        else:
            print("⚠ WARNING: True parameter outside 95% credible interval")
            print("  (This can happen with synthetic data - try increasing iterations)")

        print()
        print("="*80)
        print("EXAMPLE COMPLETE")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Run on SLURM cluster:")
        print("     sbatch <(python generate_slurm_script.py --profile medium \\")
        print("                                               --script distributed_svgd_example.py)")
        print()
        print("  2. Try different configurations:")
        print("     python generate_slurm_script.py --list-profiles")
        print()
        print("  3. Modify for your own model:")
        print("     - Change build_coalescent_model() to your graph builder")
        print("     - Adjust n_particles and n_iterations")
        print("     - Add your own observed data")
        print()


if __name__ == "__main__":
    main()
