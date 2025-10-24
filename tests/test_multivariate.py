"""Test multivariate phase-type distributions with 2D observations and 2D rewards"""

import numpy as np
import jax.numpy as jnp
import pytest
from phasic import Graph

def create_simple_exponential_graph(parameterized=True):
    """Create a simple exponential distribution graph for testing"""
    graph = Graph(state_length=1, parameterized=parameterized)
    v_start = graph.starting_vertex()
    v_transient = graph.find_or_create_vertex([1])
    v_absorb = graph.find_or_create_vertex([0])

    v_start.add_edge(v_transient, 1.0)
    if parameterized:
        # rate = theta[0]
        v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])
    else:
        v_transient.add_edge(v_absorb, 2.0)  # Fixed rate

    return graph


class TestMultivariateModel:
    """Test pmf_and_moments_from_graph_multivariate() wrapper"""

    def test_1d_backward_compatibility(self):
        """Test that 1D rewards work exactly as before"""
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

    def test_no_rewards_backward_compatibility(self):
        """Test that None rewards work correctly"""
        graph = create_simple_exponential_graph()

        model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
        model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

        theta = jnp.array([2.0])
        times = jnp.array([0.5, 1.0, 1.5])

        pmf_1d, moments_1d = model_1d(theta, times, rewards=None)
        pmf_mv, moments_mv = model_mv(theta, times, rewards=None)

        np.testing.assert_allclose(pmf_1d, pmf_mv, rtol=1e-10)
        np.testing.assert_allclose(moments_1d, moments_mv, rtol=1e-10)

    def test_2d_rewards_shape(self):
        """Test that 2D rewards produce correct output shapes"""
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

    def test_2d_rewards_with_2d_times(self):
        """Test that 2D rewards work with 2D times"""
        graph = create_simple_exponential_graph()
        model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

        theta = jnp.array([2.0])
        n_times = 5
        n_features = 3
        n_vertices = 4

        # Different times for each feature
        times_2d = jnp.array([
            [0.5, 1.0, 1.5],
            [1.0, 2.0, 3.0],
            [1.5, 3.0, 4.5],
            [2.0, 4.0, 6.0],
            [2.5, 5.0, 7.5]
        ])
        rewards_2d = jnp.ones((n_vertices, n_features))

        pmf, moments = model(theta, times_2d, rewards=rewards_2d)

        assert pmf.shape == (n_times, n_features)
        assert moments.shape == (n_features, 2)

    def test_2d_rewards_independence(self):
        """Test that each feature dimension is computed independently"""
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

    def test_invalid_rewards_shape(self):
        """Test that 3D rewards raise an error"""
        graph = create_simple_exponential_graph()
        model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

        theta = jnp.array([2.0])
        times = jnp.array([0.5, 1.0])
        rewards_3d = jnp.ones((4, 2, 2))  # Invalid 3D shape

        with pytest.raises(ValueError, match="Rewards must be 1D.*or 2D"):
            model(theta, times, rewards=rewards_3d)


class TestSVGDMultivariate:
    """Test SVGD integration with multivariate models"""

    def test_svgd_accepts_rewards(self):
        """Test that SVGD accepts rewards parameter"""
        graph = create_simple_exponential_graph()

        observed_data = jnp.array([0.5, 1.0, 1.5, 2.0])
        rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

        # This should not raise an error
        from phasic import SVGD
        model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

        svgd = SVGD(
            model=model,
            observed_data=observed_data,
            theta_dim=1,
            n_particles=10,
            n_iterations=5,
            regularization=0.0,  # No regularization for fast test
            verbose=False,
            rewards=rewards
        )

        assert svgd.rewards is not None
        np.testing.assert_array_equal(svgd.rewards, rewards)

    def test_svgd_1d_inference(self):
        """Test SVGD inference with 1D rewards"""
        graph = create_simple_exponential_graph()

        # Generate synthetic data
        np.random.seed(42)
        true_rate = 2.0
        observed_data = np.random.exponential(scale=1/true_rate, size=20)

        rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

        from phasic import SVGD
        model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

        svgd = SVGD(
            model=model,
            observed_data=observed_data,
            theta_dim=1,
            n_particles=20,
            n_iterations=10,
            learning_rate=0.01,
            regularization=0.0,
            verbose=False,
            rewards=rewards
        )

        svgd.optimize()

        # Should have converged to something
        assert svgd.theta_mean is not None
        assert svgd.particles is not None
        assert svgd.particles.shape == (20, 1)

    def test_svgd_2d_inference(self):
        """Test SVGD inference with 2D rewards and 2D observations"""
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
        assert svgd.particles.shape == (20, 1)

    def test_svgd_2d_moments_regularization(self):
        """Test that 2D moments work with regularization"""
        graph = create_simple_exponential_graph()

        # Generate synthetic 2D data
        np.random.seed(42)
        true_rate = 2.0
        n_obs = 15
        n_features = 2

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

        # Use regularization
        svgd = SVGD(
            model=model,
            observed_data=observed_data,
            theta_dim=1,
            n_particles=20,
            n_iterations=10,
            learning_rate=0.01,
            regularization=1.0,  # Enable regularization
            nr_moments=2,
            verbose=False,
            rewards=rewards_2d
        )

        svgd.optimize()

        # Should still converge
        assert svgd.theta_mean is not None
        assert svgd.particles.shape == (20, 1)


class TestGraphSVGDAPI:
    """Test Graph.svgd() convenience method with rewards"""

    def test_graph_svgd_with_rewards(self):
        """Test that Graph.svgd() accepts rewards parameter"""
        graph = create_simple_exponential_graph()

        # Generate synthetic data
        np.random.seed(42)
        observed_data = np.random.exponential(scale=0.5, size=20)
        rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

        # This should not raise an error
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

    def test_graph_svgd_2d_rewards(self):
        """Test Graph.svgd() with 2D rewards and observations"""
        graph = create_simple_exponential_graph()

        # Generate synthetic 2D data
        np.random.seed(42)
        n_obs = 15
        n_features = 2
        observed_data = jnp.array([
            np.random.exponential(scale=0.5, size=n_features)
            for _ in range(n_obs)
        ])

        rewards_2d = jnp.array([
            [1.0, 0.5],
            [2.0, 1.0],
            [0.5, 2.0],
            [1.5, 1.5]
        ])

        svgd = graph.svgd(
            observed_data=observed_data,
            theta_dim=1,
            n_particles=10,
            n_iterations=5,
            regularization=0.0,
            nr_moments=0,
            verbose=False,
            rewards=rewards_2d
        )

        assert svgd.rewards is not None
        assert svgd.rewards.shape == (4, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
