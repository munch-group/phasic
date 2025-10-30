"""
Debug convergence issue: 1D vs multivariate with NaNs and rewards
"""

import numpy as np
from phasic import Graph, SVGD
import jax.numpy as jnp


def test_1d_convergence():
    """Test that 1D case converges correctly"""

    print("="*60)
    print("Test 1: 1D observations (should converge to true value)")
    print("="*60)

    # True parameter
    true_theta = 10.0
    print(f"\nTrue parameter: θ = {true_theta}")

    # Create simple model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    _graph = Graph(callback=callback, parameterized=True)

    # Generate observations from true parameter
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(_graph, param_length=1, enable_rewards=False)
    graph_inst = instantiate_from_trace(trace, params=np.array([true_theta]))

    np.random.seed(42)
    n_samples = 100
    observed_times = np.array(graph_inst.sample(n_samples))

    print(f"Generated {n_samples} observations")
    print(f"Sample mean: {observed_times.mean():.4f} (expected: {1/true_theta:.4f})")

    # Create model
    model = Graph.pmf_and_moments_from_graph(_graph, nr_moments=2, discrete=False)

    # Run SVGD
    svgd = SVGD(
        model=model,
        observed_data=observed_times,
        theta_dim=1,
        n_particles=100,
        n_iterations=500,
        learning_rate=0.05,
        verbose=False,
        jit=True
    )
    svgd.optimize(return_history=False)

    posterior_mean = svgd.theta_mean[0]
    error = abs(posterior_mean - true_theta)

    print(f"\nSVGD Results:")
    print(f"  Posterior mean: {posterior_mean:.4f}")
    print(f"  True value:     {true_theta:.4f}")
    print(f"  Error:          {error:.4f}")

    return posterior_mean, error


def test_multivariate_with_nans_and_rewards():
    """Test multivariate case with NaNs and rewards (user's pattern)"""

    print("\n" + "="*60)
    print("Test 2: Multivariate with NaNs and rewards")
    print("="*60)

    # True parameter
    true_theta = 10.0
    print(f"\nTrue parameter: θ = {true_theta}")

    # Create model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    _graph = Graph(callback=callback, parameterized=True)
    n_vertices = _graph.vertices_length()

    # Create rewards (n_vertices, n_features)
    n_features = 3
    rewards = jnp.ones((n_vertices, n_features), dtype=jnp.float64)
    # Scale each feature differently
    for i in range(n_features):
        rewards = rewards.at[:, i].set((i + 1) * 1.0)

    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards:\n{rewards}")

    # Generate sparse observations (user's pattern)
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(_graph, param_length=1, enable_rewards=True)

    np.random.seed(42)
    n = 30  # observations per feature
    a = np.empty((n_features, n * n_features))
    a[:] = np.nan

    for i in range(n_features):
        reward_i = np.array(rewards[:, i])
        graph_i = instantiate_from_trace(trace, params=np.array([true_theta]), rewards=reward_i)
        samples = np.array(graph_i.sample(n))
        a[i, i*n:(i+1)*n] = samples
        print(f"Feature {i}: sampled {n} times with reward scaling {i+1}")
        print(f"  Sample mean: {samples.mean():.6f}")

    observed_data = jnp.array(a).T

    print(f"\nObserved data shape: {observed_data.shape}")
    print(f"NaN count: {jnp.isnan(observed_data).sum()} / {observed_data.size}")

    # Create multivariate model
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _graph,
        nr_moments=2,
        discrete=False
    )

    # Check sample moments computation
    print("\n" + "-"*60)
    print("Checking sample moments...")
    print("-"*60)

    # Compute sample moments manually from observed data
    from phasic.svgd import compute_sample_moments
    sample_moments = compute_sample_moments(observed_data, nr_moments=2)
    print(f"Sample moments (from data with NaNs): {sample_moments}")

    # Compute sample moments from valid data only
    valid_data = observed_data[~jnp.isnan(observed_data).any(axis=1)]
    print(f"Valid observations (no NaNs): {len(valid_data)}")
    if len(valid_data) > 0:
        sample_moments_valid = compute_sample_moments(valid_data, nr_moments=2)
        print(f"Sample moments (from valid data only): {sample_moments_valid}")

    # Run SVGD
    print("\n" + "-"*60)
    print("Running SVGD...")
    print("-"*60)

    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=100,
        n_iterations=500,
        learning_rate=0.05,
        rewards=rewards,
        verbose=False,
        jit=True
    )
    svgd.optimize(return_history=False)

    posterior_mean = svgd.theta_mean[0]
    error = abs(posterior_mean - true_theta)

    print(f"\nSVGD Results:")
    print(f"  Posterior mean: {posterior_mean:.4f}")
    print(f"  True value:     {true_theta:.4f}")
    print(f"  Error:          {error:.4f}")

    return posterior_mean, error


def test_multivariate_without_nans():
    """Test multivariate case with full observations (no NaNs)"""

    print("\n" + "="*60)
    print("Test 3: Multivariate WITHOUT NaNs (baseline)")
    print("="*60)

    # True parameter
    true_theta = 10.0
    print(f"\nTrue parameter: θ = {true_theta}")

    # Create model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    _graph = Graph(callback=callback, parameterized=True)
    n_vertices = _graph.vertices_length()

    # Create rewards
    n_features = 3
    rewards = jnp.ones((n_vertices, n_features), dtype=jnp.float64)
    for i in range(n_features):
        rewards = rewards.at[:, i].set((i + 1) * 1.0)

    # Generate FULL observations (no NaNs)
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    trace = record_elimination_trace(_graph, param_length=1, enable_rewards=True)

    np.random.seed(42)
    n = 30
    observations_list = []

    for i in range(n_features):
        reward_i = np.array(rewards[:, i])
        graph_i = instantiate_from_trace(trace, params=np.array([true_theta]), rewards=reward_i)
        samples = np.array(graph_i.sample(n))
        observations_list.append(samples)

    # Stack as (n, n_features) - all observations valid
    observed_data = jnp.column_stack(observations_list)

    print(f"Observed data shape: {observed_data.shape}")
    print(f"NaN count: {jnp.isnan(observed_data).sum()} (should be 0)")

    # Create model
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _graph,
        nr_moments=2,
        discrete=False
    )

    # Run SVGD
    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=100,
        n_iterations=500,
        learning_rate=0.05,
        rewards=rewards,
        verbose=False,
        jit=True
    )
    svgd.optimize(return_history=False)

    posterior_mean = svgd.theta_mean[0]
    error = abs(posterior_mean - true_theta)

    print(f"\nSVGD Results:")
    print(f"  Posterior mean: {posterior_mean:.4f}")
    print(f"  True value:     {true_theta:.4f}")
    print(f"  Error:          {error:.4f}")

    return posterior_mean, error


if __name__ == "__main__":
    print("Investigating convergence issue: 1D vs multivariate with NaNs\n")

    # Run all tests
    mean_1d, error_1d = test_1d_convergence()
    mean_multivar_nan, error_multivar_nan = test_multivariate_with_nans_and_rewards()
    mean_multivar_full, error_multivar_full = test_multivariate_without_nans()

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"\n1D case:                     θ̂ = {mean_1d:.4f} (error: {error_1d:.4f})")
    print(f"Multivariate with NaNs:      θ̂ = {mean_multivar_nan:.4f} (error: {error_multivar_nan:.4f})")
    print(f"Multivariate without NaNs:   θ̂ = {mean_multivar_full:.4f} (error: {error_multivar_full:.4f})")

    # Diagnose the issue
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if error_multivar_nan > 2 * error_1d:
        print("⚠️  Multivariate with NaNs has significantly worse convergence!")
        print("    Possible causes:")
        print("    1. Sample moments computed incorrectly from NaN data")
        print("    2. Reward scaling interacting with NaN masking")
        print("    3. Different effective sample sizes per feature")

    if error_multivar_full < error_multivar_nan:
        print("⚠️  Multivariate without NaNs converges better than with NaNs")
        print("    This suggests the NaN handling may have an issue")
