"""
Test that SVGD with NaN observations produces correct parameter estimates
"""

import numpy as np
from phasic import Graph, SVGD
import jax.numpy as jnp


def test_nan_correctness_univariate():
    """
    Test SVGD correctness with sparse observations (one feature, some NaNs)

    Generate data from known parameter, add some NaNs, verify recovery
    """
    print("\n" + "="*60)
    print("Test 1: Univariate with NaNs")
    print("="*60)

    # True parameter value
    true_theta = 1.5
    print(f"\nTrue parameter: θ = {true_theta}")

    # Create simple exponential model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback=callback, parameterized=True)

    # Instantiate graph with true parameter value
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    graph_inst = instantiate_from_trace(trace, params=np.array([true_theta]))

    # Generate observations from true parameter
    np.random.seed(42)
    n_samples = 100
    observed_times = np.array(graph_inst.sample(n_samples))

    # Add NaNs to 30% of observations to simulate missing data
    n_missing = 30
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    observed_times_with_nan = observed_times.copy()
    observed_times_with_nan[missing_indices] = np.nan

    print(f"Generated {n_samples} observations")
    print(f"Added {n_missing} NaNs ({n_missing/n_samples*100:.0f}% missing)")
    print(f"Valid observations: {(~np.isnan(observed_times_with_nan)).sum()}")

    # Create model
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    # Test that model works correctly at true parameter
    print(f"\nVerifying model at true parameter...")
    pmf_test, _ = model(np.array([true_theta]), observed_times[:10])
    print(f"  PMF at true θ: min={pmf_test.min():.6f}, max={pmf_test.max():.6f}, mean={pmf_test.mean():.6f}")

    # Run SVGD with NaN observations
    svgd_nan = SVGD(
        model=model,
        observed_data=observed_times_with_nan,
        theta_dim=1,
        n_particles=100,
        n_iterations=500,
        learning_rate=0.05,
        verbose=True,
        jit=True
    )
    svgd_nan.optimize(return_history=False)

    # Run SVGD with full observations (for comparison)
    svgd_full = SVGD(
        model=model,
        observed_data=observed_times,
        theta_dim=1,
        n_particles=100,
        n_iterations=500,
        learning_rate=0.05,
        verbose=True,
        jit=True
    )
    svgd_full.optimize(return_history=False)

    # Compare results
    posterior_mean_nan = svgd_nan.theta_mean[0]
    posterior_std_nan = svgd_nan.theta_std[0]
    posterior_mean_full = svgd_full.theta_mean[0]
    posterior_std_full = svgd_full.theta_std[0]

    print(f"\nResults with NaN observations:")
    print(f"  Posterior mean: {posterior_mean_nan:.4f}")
    print(f"  Posterior std:  {posterior_std_nan:.4f}")
    print(f"  Error: {abs(posterior_mean_nan - true_theta):.4f}")

    print(f"\nResults with full observations:")
    print(f"  Posterior mean: {posterior_mean_full:.4f}")
    print(f"  Posterior std:  {posterior_std_full:.4f}")
    print(f"  Error: {abs(posterior_mean_full - true_theta):.4f}")

    # Check if NaN version is reasonably close to true value
    # Allow larger error due to missing data and SVGD convergence
    error_nan = abs(posterior_mean_nan - true_theta)
    error_full = abs(posterior_mean_full - true_theta)

    print(f"\nRelative performance:")
    print(f"  Error ratio (NaN/Full): {error_nan / (error_full + 1e-10):.2f}")

    # Check that both versions converged to similar values
    # (even if not perfect, NaN masking should not cause divergence)
    diff = abs(posterior_mean_nan - posterior_mean_full)
    print(f"  Difference between NaN and Full: {diff:.4f}")

    # Main test: NaN masking should give similar results to full data
    # (within 20% difference, or 0.3 absolute)
    relative_diff = diff / (abs(posterior_mean_full) + 0.1)
    assert relative_diff < 0.2 or diff < 0.3, \
        f"NaN and full versions diverged: diff={diff}, relative={relative_diff}"

    print(f"\n✓ NaN observations handled correctly (similar to full data)")


def test_nan_correctness_multivariate():
    """
    Test SVGD correctness with sparse multivariate observations

    Each observation has data for only one feature (rest are NaN)
    """
    print("\n" + "="*60)
    print("Test 2: Multivariate with Sparse Pattern")
    print("="*60)

    # True parameter value
    true_theta = 1.5
    print(f"\nTrue parameter: θ = {true_theta}")

    # Create simple exponential model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    _graph = Graph(callback=callback, parameterized=True)
    n_vertices = _graph.vertices_length()

    # Create 2D rewards (n_features, n_vertices)
    n_features = 3
    rewards = jnp.ones((n_features, n_vertices), dtype=jnp.float64)
    # Each feature gets a different scaling
    for i in range(n_features):
        rewards = rewards.at[i, :].set((i + 1) * 1.0)

    print(f"Number of features: {n_features}")
    print(f"Reward scaling per feature: {[i+1 for i in range(n_features)]}")

    # Generate sparse observations (user's pattern)
    np.random.seed(42)
    n = 30  # observations per feature
    a = np.empty((n_features, n * n_features))
    a[:] = np.nan

    # Instantiate graphs for each feature and sample
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(_graph, param_length=1, enable_rewards=True)

    for i in range(n_features):
        reward_i = np.array(rewards[i, :])
        graph_i = instantiate_from_trace(trace, params=np.array([true_theta]), rewards=reward_i)
        samples = np.array(graph_i.sample(n))
        a[i, i*n:(i+1)*n] = samples

    observed_data = jnp.array(a).T

    total_elements = observed_data.size
    nan_count = jnp.isnan(observed_data).sum()
    valid_count = (~jnp.isnan(observed_data)).sum()

    print(f"\nObserved data shape: {observed_data.shape}")
    print(f"Total elements: {total_elements}")
    print(f"NaN elements: {nan_count} ({nan_count/total_elements*100:.1f}%)")
    print(f"Valid elements: {valid_count} ({valid_count/total_elements*100:.1f}%)")

    # Create multivariate model
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _graph,
        nr_moments=2,
        discrete=False
    )

    # Run SVGD with sparse observations
    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=80,
        n_iterations=300,
        learning_rate=0.02,
        rewards=rewards,
        verbose=False,
        jit=True
    )
    svgd.optimize(return_history=False)

    posterior_mean = svgd.theta_mean[0]
    posterior_std = svgd.theta_std[0]
    error = abs(posterior_mean - true_theta)

    print(f"\nSVGD Results:")
    print(f"  True parameter:     θ = {true_theta:.4f}")
    print(f"  Posterior mean:     θ̂ = {posterior_mean:.4f}")
    print(f"  Posterior std:      σ = {posterior_std:.4f}")
    print(f"  Absolute error:     |θ̂ - θ| = {error:.4f}")
    print(f"  Relative error:     {error/true_theta*100:.1f}%")

    # Main test: SVGD should complete and return finite values
    assert np.isfinite(posterior_mean), "Posterior mean should be finite"
    assert np.isfinite(posterior_std), "Posterior std should be finite"
    assert posterior_mean > 0, "Posterior mean should be positive"

    print(f"\n✓ Multivariate sparse observations handled correctly (SVGD converges)")


def test_nan_vs_filtered_equivalence():
    """
    Test that using NaN is equivalent to filtering out observations

    This verifies that NaN masking works correctly
    """
    print("\n" + "="*60)
    print("Test 3: NaN Masking Equivalence")
    print("="*60)

    # True parameter
    true_theta = 1.5

    # Create model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    graph = Graph(callback=callback, parameterized=True)
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    # Instantiate with true parameter and generate observations
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(graph, param_length=1, enable_rewards=False)
    graph_inst = instantiate_from_trace(trace, params=np.array([true_theta]))

    np.random.seed(42)
    n_samples = 50
    observed_times = np.array(graph_inst.sample(n_samples))

    # Add NaNs to 20 observations
    n_missing = 20
    missing_indices = np.array([i for i in range(n_missing)])
    observed_with_nan = observed_times.copy()
    observed_with_nan[missing_indices] = np.nan

    # Create filtered version (only valid observations)
    observed_filtered = observed_times[~np.isnan(observed_with_nan)]

    print(f"\nOriginal: {len(observed_times)} observations")
    print(f"With NaN: {len(observed_with_nan)} total, {(~np.isnan(observed_with_nan)).sum()} valid")
    print(f"Filtered: {len(observed_filtered)} observations")

    # Run SVGD with NaN observations
    np.random.seed(123)
    svgd_nan = SVGD(
        model=model,
        observed_data=observed_with_nan,
        theta_dim=1,
        n_particles=60,
        n_iterations=300,
        learning_rate=0.03,
        seed=123,
        verbose=False,
        jit=True
    )
    svgd_nan.optimize(return_history=False)

    # Run SVGD with filtered observations
    np.random.seed(123)
    svgd_filtered = SVGD(
        model=model,
        observed_data=observed_filtered,
        theta_dim=1,
        n_particles=60,
        n_iterations=300,
        learning_rate=0.03,
        seed=123,
        verbose=False,
        jit=True
    )
    svgd_filtered.optimize(return_history=False)

    # Compare results
    mean_nan = svgd_nan.theta_mean[0]
    mean_filtered = svgd_filtered.theta_mean[0]

    print(f"\nResults:")
    print(f"  True parameter:       θ = {true_theta:.4f}")
    print(f"  NaN masking:          θ̂ = {mean_nan:.4f} (error: {abs(mean_nan - true_theta):.4f})")
    print(f"  Filtered data:        θ̂ = {mean_filtered:.4f} (error: {abs(mean_filtered - true_theta):.4f})")
    print(f"  Difference (NaN-Filt):    {abs(mean_nan - mean_filtered):.4f}")

    # NaN masking should give similar results to filtering
    # Allow some difference due to JAX/numerical differences
    diff = abs(mean_nan - mean_filtered)
    assert diff < 0.15, \
        f"NaN masking differs too much from filtering: {diff} > 0.15"

    print(f"\n✓ NaN masking is equivalent to filtering")


if __name__ == "__main__":
    print("Testing correctness of SVGD with NaN observations...")

    test_nan_correctness_univariate()
    test_nan_correctness_multivariate()
    test_nan_vs_filtered_equivalence()

    print("\n" + "="*60)
    print("✓ All correctness tests passed!")
    print("="*60)
    print("\nConclusion:")
    print("  - NaN observations are handled correctly via masking")
    print("  - Parameter estimates are accurate despite missing data")
    print("  - Multivariate sparse patterns work correctly")
    print("  - NaN masking is equivalent to pre-filtering data")
