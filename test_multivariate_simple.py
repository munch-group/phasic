"""Simple test for multivariate phase-type distributions (no pytest required)"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

def create_simple_exponential_graph():
    """Create a simple exponential distribution graph for testing"""
    graph = Graph(state_length=1, parameterized=True)
    v_start = graph.starting_vertex()
    v_transient = graph.find_or_create_vertex([1])
    v_absorb = graph.find_or_create_vertex([0])

    v_start.add_edge(v_transient, 1.0)
    # rate = theta[0]
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

    return graph


def test_1d_backward_compatibility():
    """Test that 1D rewards work exactly as before"""
    print("Testing 1D backward compatibility...")
    graph = create_simple_exponential_graph()

    # Create both model types
    model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

    theta = jnp.array([2.0])
    times = jnp.array([0.5, 1.0, 1.5])
    rewards_1d = jnp.array([1.0, 2.0, 0.5, 1.5])

    # Both should give same results
    pmf_1d, moments_1d = model_1d(theta, times, rewards=rewards_1d)
    pmf_mv, moments_mv = model_mv(theta, times, rewards=rewards_1d)

    assert pmf_1d.shape == pmf_mv.shape == (3,)
    assert moments_1d.shape == moments_mv.shape == (2,)

    np.testing.assert_allclose(pmf_1d, pmf_mv, rtol=1e-10)
    np.testing.assert_allclose(moments_1d, moments_mv, rtol=1e-10)

    print("✓ 1D backward compatibility test passed")


def test_no_rewards_backward_compatibility():
    """Test that None rewards work correctly"""
    print("Testing no rewards backward compatibility...")
    graph = create_simple_exponential_graph()

    model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

    theta = jnp.array([2.0])
    times = jnp.array([0.5, 1.0, 1.5])

    pmf_1d, moments_1d = model_1d(theta, times, rewards=None)
    pmf_mv, moments_mv = model_mv(theta, times, rewards=None)

    np.testing.assert_allclose(pmf_1d, pmf_mv, rtol=1e-10)
    np.testing.assert_allclose(moments_1d, moments_mv, rtol=1e-10)

    print("✓ No rewards backward compatibility test passed")


def test_2d_rewards_shape():
    """Test that 2D rewards produce correct output shapes"""
    print("Testing 2D rewards shape...")
    graph = create_simple_exponential_graph()
    model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

    theta = jnp.array([2.0])
    n_times = 5
    n_features = 3
    n_vertices = 4

    times = jnp.linspace(0.5, 2.5, n_times)
    rewards_2d = jnp.ones((n_vertices, n_features))

    pmf, moments = model(theta, times, rewards=rewards_2d)

    # Check shapes
    assert pmf.shape == (n_times, n_features), f"Expected {(n_times, n_features)}, got {pmf.shape}"
    assert moments.shape == (n_features, 2), f"Expected {(n_features, 2)}, got {moments.shape}"

    print(f"✓ 2D rewards shape test passed: PMF {pmf.shape}, moments {moments.shape}")


def test_2d_rewards_independence():
    """Test that each feature dimension is computed independently"""
    print("Testing 2D rewards independence...")
    graph = create_simple_exponential_graph()
    model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

    theta = jnp.array([2.0])
    times = jnp.array([0.5, 1.0, 1.5])
    n_vertices = 4
    n_features = 3

    # Create 2D rewards with different values per feature
    rewards_2d = jnp.array([
        [1.0, 2.0, 0.5],
        [2.0, 1.0, 1.5],
        [0.5, 1.5, 2.0],
        [1.5, 0.5, 1.0]
    ])

    # Compute 2D result
    pmf_2d, moments_2d = model_mv(theta, times, rewards=rewards_2d)

    # Compute each feature separately and verify they match
    for j in range(n_features):
        reward_j = rewards_2d[:, j]
        pmf_j, moments_j = model_1d(theta, times, rewards=reward_j)

        np.testing.assert_allclose(pmf_2d[:, j], pmf_j, rtol=1e-10,
                                  err_msg=f"Feature {j} PMF mismatch")
        np.testing.assert_allclose(moments_2d[j, :], moments_j, rtol=1e-10,
                                  err_msg=f"Feature {j} moments mismatch")

    print("✓ 2D rewards independence test passed")


def test_svgd_accepts_rewards():
    """Test that SVGD accepts rewards parameter"""
    print("Testing SVGD accepts rewards...")
    graph = create_simple_exponential_graph()

    observed_data = jnp.array([0.5, 1.0, 1.5, 2.0])
    rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

    from phasic import SVGD
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=5,
        regularization=0.0,
        verbose=False,
        rewards=rewards
    )

    assert svgd.rewards is not None
    np.testing.assert_array_equal(svgd.rewards, rewards)

    print("✓ SVGD accepts rewards test passed")


def test_svgd_2d_inference():
    """Test SVGD inference with 2D rewards and 2D observations"""
    print("Testing SVGD 2D inference...")
    graph = create_simple_exponential_graph()

    # Generate synthetic 2D data
    np.random.seed(42)
    true_rate = 2.0
    n_obs = 15
    n_features = 2

    # Simulate 2D observations
    observed_data = jnp.array([
        np.random.exponential(scale=1/true_rate, size=n_features)
        for _ in range(n_obs)
    ])

    n_vertices = 4
    rewards_2d = jnp.array([
        [1.0, 0.5],
        [2.0, 1.0],
        [0.5, 2.0],
        [1.5, 1.5]
    ])

    from phasic import SVGD
    model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=20,
        n_iterations=10,
        learning_rate=0.01,
        regularization=0.0,
        verbose=False,
        rewards=rewards_2d
    )

    svgd.optimize()

    # Should have converged
    assert svgd.theta_mean is not None
    # Note: n_particles may be adjusted for pmap (e.g., 20 -> 24 for 8 devices)
    assert svgd.particles.shape[1] == 1, f"Expected theta_dim=1, got {svgd.particles.shape[1]}"
    assert svgd.particles.shape[0] >= 20, f"Expected at least 20 particles, got {svgd.particles.shape[0]}"

    print(f"✓ SVGD 2D inference test passed (theta_mean: {svgd.theta_mean}, particles: {svgd.particles.shape})")


def test_graph_svgd_with_rewards():
    """Test that Graph.svgd() accepts rewards parameter"""
    print("Testing Graph.svgd() with rewards...")
    graph = create_simple_exponential_graph()

    # Generate synthetic data
    np.random.seed(42)
    observed_data = np.random.exponential(scale=0.5, size=20)
    rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

    svgd = graph.svgd(
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=5,
        regularization=0.0,
        nr_moments=0,
        verbose=False,
        rewards=rewards
    )

    assert svgd.rewards is not None

    print("✓ Graph.svgd() with rewards test passed")


if __name__ == '__main__':
    print("=" * 70)
    print("MULTIVARIATE PHASE-TYPE DISTRIBUTION TESTS")
    print("=" * 70)
    print()

    try:
        test_1d_backward_compatibility()
        test_no_rewards_backward_compatibility()
        test_2d_rewards_shape()
        test_2d_rewards_independence()
        test_svgd_accepts_rewards()
        test_svgd_2d_inference()
        test_graph_svgd_with_rewards()

        print()
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED ✗: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
