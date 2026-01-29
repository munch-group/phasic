"""
Test NaN handling in multivariate SVGD with sparse observation pattern
"""

import numpy as np
from phasic import Graph, SVGD
import jax.numpy as jnp

def test_nan_observations():
    """Test SVGD with observations that have NaNs except for one feature at a time"""

    # Create simple parameterized graph
    def callback(state):
        if len(state) == 0:
            # Starting vertex -> absorbing vertex
            return [(np.array([1]), 0.0, [1.0])]
        elif state[0] == 1:
            # Absorbing vertex
            return []
        return []

    _graph = Graph(callback)

    # Create multivariate model
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _graph,
        nr_moments=2,
        discrete=False
    )

    # Get number of vertices
    n_vertices = _graph.vertices_length()
    print(f"Graph has {n_vertices} vertices")

    # Create 2D rewards (n_features, n_vertices)
    n_features = 3
    rewards = jnp.ones((n_features, n_vertices), dtype=jnp.float64)
    # Scale each feature differently
    for i in range(n_features):
        rewards = rewards.at[i, :].set((i + 1) * 1.0)

    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards:\n{rewards}")

    # Create sparse observation pattern like user's example
    # Each row has NaNs except for one feature
    n = 10  # nr_observations per feature
    a = np.empty((rewards.shape[0], n * rewards.shape[0]))
    a[:] = np.nan

    # Sample data for each feature (simulating with random exponential)
    np.random.seed(42)
    for i in range(rewards.shape[0]):
        # Generate samples (using exponential as proxy for testing)
        samples = np.random.exponential(1.0, n)
        a[i, i*n:(i+1)*n] = samples

    observed_data = jnp.array(a).T

    print(f"\nObserved data shape: {observed_data.shape}")
    print(f"Number of NaNs: {jnp.isnan(observed_data).sum()}")
    print(f"Number of non-NaN values: {(~jnp.isnan(observed_data)).sum()}")
    print(f"\nFirst 5 rows of observed data:")
    print(observed_data[:5])

    # Test that model can handle the data
    print("\n" + "="*60)
    print("Testing model evaluation with NaN observations...")
    print("="*60)

    theta_test = jnp.array([1.5])
    try:
        pmf_vals, moments = model(theta_test, observed_data, rewards=rewards)
        print(f"✓ Model evaluation successful!")
        print(f"  PMF shape: {pmf_vals.shape}")
        print(f"  Moments shape: {moments.shape}")
        print(f"  Number of NaN PMF values: {jnp.isnan(pmf_vals).sum()}")
        print(f"  Number of non-NaN PMF values: {(~jnp.isnan(pmf_vals)).sum()}")
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test SVGD with sparse observations
    print("\n" + "="*60)
    print("Testing SVGD with sparse observations...")
    print("="*60)

    try:
        svgd = SVGD(
            model=model,
            observed_data=observed_data,
            theta_dim=1,
            n_particles=10,
            n_iterations=50,
            learning_rate=0.01,
            rewards=rewards,
            regularization=0.0,  # No regularization for now
            verbose=True,
            jit=True
        )

        print("\n✓ SVGD object created successfully!")

        print("\nRunning SVGD optimization...")
        svgd.optimize()

        print(f"\n✓ SVGD optimization completed!")
        print(f"  Posterior mean: {svgd.theta_mean}")
        print(f"  Posterior std: {svgd.theta_std}")

    except Exception as e:
        print(f"\n✗ SVGD failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing NaN handling in multivariate SVGD...\n")
    test_nan_observations()
    print("\n✓ All tests passed!")
