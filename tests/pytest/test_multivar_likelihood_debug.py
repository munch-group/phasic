"""
Debug multivariate likelihood computation with NaN observations
"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp


def test_likelihood_at_true_parameter():
    """Test that likelihood is maximized at true parameter"""

    print("="*60)
    print("Testing likelihood at true parameter")
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

    # Generate data using the CORRECT approach (from notebook)
    _graph.update_parameterized_weights(np.array([true_theta]))

    # Create rewards
    n_features = 3
    rewards = jnp.ones((n_features, n_vertices), dtype=jnp.float64)
    for i in range(n_features):
        rewards = rewards.at[i, :].set((i + 1) * 1.0)

    print(f"\nRewards shape: {rewards.shape}")
    print(f"Rewards:\n{rewards}")

    # Generate sparse observations (user's pattern)
    np.random.seed(42)
    n = 100  # observations per feature
    a = np.empty((n_features, n * n_features))
    a[:] = np.nan

    for i in range(n_features):
        reward_i = np.array(rewards[i, :])
        samples = np.array(_graph.sample(n, rewards=reward_i))
        a[i, i*n:(i+1)*n] = samples
        print(f"\nFeature {i} (reward scaling {i+1}):")
        print(f"  Sample mean: {samples.mean():.6f}")
        print(f"  Expected mean (for Exp(θ*r)): {1/(true_theta * (i+1)):.6f}")

    observed_data = jnp.array(a).T

    print(f"\n\nObserved data shape: {observed_data.shape}")
    print(f"NaN count: {jnp.isnan(observed_data).sum()} / {observed_data.size}")

    # Create multivariate model
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _graph,
        nr_moments=2,
        discrete=False
    )

    # Test likelihood at different parameter values
    print("\n" + "="*60)
    print("Evaluating likelihood at different θ values")
    print("="*60)

    theta_values = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    for theta_test in theta_values:
        # Compute PMF and moments
        pmf_vals, moments = model(jnp.array([theta_test]), observed_data, rewards=rewards)

        # Compute log-likelihood manually
        mask = ~jnp.isnan(pmf_vals)
        valid_count = mask.sum()
        log_lik = jnp.sum(jnp.where(mask, jnp.log(pmf_vals + 1e-10), 0.0))

        print(f"\nθ = {theta_test:5.1f}:")
        print(f"  Valid observations: {valid_count}")
        print(f"  Log-likelihood: {log_lik:.4f}")
        print(f"  Mean PMF (valid): {jnp.where(mask, pmf_vals, jnp.nan).mean():.6f}")

    # Check sample moments
    print("\n" + "="*60)
    print("Sample moments")
    print("="*60)

    from phasic.svgd import compute_sample_moments
    sample_moments = compute_sample_moments(observed_data, nr_moments=2)
    print(f"Sample moments from data: {sample_moments}")

    # Compute moments per feature
    for i in range(n_features):
        feature_data = observed_data[:, i]
        valid_data = feature_data[~jnp.isnan(feature_data)]
        if len(valid_data) > 0:
            print(f"\nFeature {i}:")
            print(f"  Valid observations: {len(valid_data)}")
            print(f"  Sample mean: {valid_data.mean():.6f}")
            print(f"  Expected (Exp(θ*r)): {1/(true_theta * (i+1)):.6f}")


def test_1d_vs_multivar_likelihood():
    """Compare 1D likelihood (no NaNs) vs multivariate likelihood"""

    print("\n\n" + "="*60)
    print("Comparing 1D vs multivariate likelihood")
    print("="*60)

    true_theta = 10.0

    # Create model
    def callback(state):
        if len(state) == 0:
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            return []
        return []

    _graph = Graph(callback=callback, parameterized=True)
    _graph.update_parameterized_weights(np.array([true_theta]))

    # Generate 1D data (no rewards)
    np.random.seed(42)
    n_samples = 100
    data_1d = jnp.array(_graph.sample(n_samples))

    print(f"\n1D data:")
    print(f"  Sample mean: {data_1d.mean():.6f}")
    print(f"  Expected: {1/true_theta:.6f}")

    # Generate multivariate data with same samples, no NaNs
    data_multivar = jnp.column_stack([data_1d, data_1d, data_1d])
    rewards_multivar = jnp.ones((3, _graph.vertices_length()), dtype=jnp.float64)

    print(f"\nMultivariate data (3 copies of 1D):")
    print(f"  Shape: {data_multivar.shape}")

    # Create models
    model_1d = Graph.pmf_and_moments_from_graph(_graph, nr_moments=2, discrete=False)
    model_multivar = Graph.pmf_and_moments_from_graph_multivariate(_graph, nr_moments=2, discrete=False)

    # Evaluate at true parameter
    theta_test = jnp.array([true_theta])

    pmf_1d, _ = model_1d(theta_test, data_1d)
    pmf_multivar, _ = model_multivar(theta_test, data_multivar, rewards=rewards_multivar)

    log_lik_1d = jnp.sum(jnp.log(pmf_1d + 1e-10))
    log_lik_multivar = jnp.sum(jnp.log(pmf_multivar + 1e-10))

    print(f"\n1D log-likelihood: {log_lik_1d:.4f}")
    print(f"Multivariate log-likelihood: {log_lik_multivar:.4f}")
    print(f"Expected ratio (3x data): ~{log_lik_1d * 3:.4f}")

    if abs(log_lik_multivar - log_lik_1d * 3) > 0.1:
        print("\n⚠️  WARNING: Multivariate likelihood not 3x the 1D likelihood!")
        print("   This suggests an issue with multivariate evaluation")


if __name__ == "__main__":
    print("Debugging multivariate likelihood with NaN observations...\n")

    test_likelihood_at_true_parameter()
    test_1d_vs_multivar_likelihood()

    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)
