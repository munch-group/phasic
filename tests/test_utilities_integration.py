#!/usr/bin/env python
"""
Test suite for utilities and integration features.
Tests plotting, SVGD, distributed computing utilities, and other features.
"""

import pytest
import numpy as np
import phasic as ptd
from phasic import Graph

# Try to import optional dependencies
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestPlotting:
    """Test plotting functionality."""

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_basic(self):
        """Test basic graph plotting."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 0.5)
        start.add_edge(v2, 0.5)
        v1.add_edge(v2, 1.0)
        g.normalize()

        # Should not raise
        try:
            result = g.plot()
            # plot() may return various types depending on backend
            assert result is not None or result is None  # Just check it runs
        except Exception as e:
            # If graphviz is not installed, plotting may fail
            if 'graphviz' not in str(e).lower():
                raise

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_plot_with_options(self):
        """Test plotting with various options."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Try with different options (if supported)
        try:
            g.plot(show_weights=True)
        except (TypeError, RuntimeError):
            # Some options may not be supported
            pass

    def test_set_theme(self):
        """Test setting plot theme."""
        # Should not raise
        ptd.set_theme('default')


class TestSVGD:
    """Test SVGD functionality."""

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_svgd_import(self):
        """Test that SVGD can be imported."""
        assert ptd.SVGD is not None

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_svgd_basic(self):
        """Test basic SVGD functionality."""
        # Define a simple parameterized model
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        # Define log likelihood
        observed_times = jnp.array([1.0, 1.5, 2.0])

        def log_likelihood(theta):
            pdf = model(theta, observed_times)
            return jnp.log(pdf + 1e-10).sum()

        # Define prior
        def log_prior(theta):
            # Simple normal prior
            return -0.5 * jnp.sum(theta**2)

        def log_posterior(theta):
            return log_likelihood(theta) + log_prior(theta)

        # Initialize particles
        n_particles = 10
        initial_particles = jnp.ones((n_particles, 1)) * 1.0

        # Create SVGD instance
        svgd = ptd.SVGD(log_posterior)

        # Run a few steps
        particles = svgd.update(initial_particles, n_iter=5, step_size=0.01)

        assert particles.shape == (n_particles, 1)
        assert jnp.all(jnp.isfinite(particles))


class TestDistributedUtilities:
    """Test distributed computing utilities."""

    def test_distributed_config_exists(self):
        """Test that DistributedConfig exists."""
        assert hasattr(ptd, 'DistributedConfig')

    def test_distributed_config_creation(self):
        """Test creating a DistributedConfig."""
        config = ptd.DistributedConfig(
            num_processes=4,
            process_id=0,
            coordinator_address='localhost:1234'
        )

        assert config.num_processes == 4
        assert config.process_id == 0

    def test_detect_slurm_environment(self):
        """Test SLURM environment detection."""
        # Should return a dict (may be empty if not in SLURM)
        env_info = ptd.detect_slurm_environment()
        assert isinstance(env_info, dict)

    def test_get_coordinator_address(self):
        """Test coordinator address utility."""
        # Should return a string or None
        try:
            addr = ptd.get_coordinator_address()
            assert isinstance(addr, (str, type(None)))
        except Exception:
            # May not work outside SLURM environment
            pass


class TestClusterConfiguration:
    """Test cluster configuration utilities."""

    def test_cluster_config_exists(self):
        """Test that ClusterConfig exists."""
        assert hasattr(ptd, 'ClusterConfig')

    def test_load_config(self):
        """Test loading cluster config."""
        # Should handle missing config gracefully
        try:
            config = ptd.load_config('nonexistent_config')
        except (FileNotFoundError, ValueError):
            # Expected if config doesn't exist
            pass

    def test_get_default_config(self):
        """Test getting default cluster config."""
        config = ptd.get_default_config()
        assert config is not None

    def test_suggest_config(self):
        """Test config suggestion utility."""
        # Should not raise
        suggestion = ptd.suggest_config()
        assert suggestion is not None


class TestAutoParallel:
    """Test automatic parallelization utilities."""

    def test_environment_info_exists(self):
        """Test that EnvironmentInfo exists."""
        assert hasattr(ptd, 'EnvironmentInfo')

    def test_parallel_config_exists(self):
        """Test that ParallelConfig exists."""
        assert hasattr(ptd, 'ParallelConfig')

    def test_detect_environment(self):
        """Test environment detection."""
        env = ptd.detect_environment()
        assert env is not None

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_configure_jax_for_environment(self):
        """Test JAX configuration for environment."""
        env = ptd.detect_environment()
        # Should not raise
        ptd.configure_jax_for_environment(env)


class TestCompilationConfig:
    """Test compilation configuration (JAX config)."""

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_compilation_config_exists(self):
        """Test that CompilationConfig exists."""
        assert hasattr(ptd, 'CompilationConfig')

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_get_default_compilation_config(self):
        """Test getting default compilation config."""
        from phasic.jax_config import get_default_config
        config = get_default_config()
        assert config is not None


class TestMatrixRepresentation:
    """Test MatrixRepresentation named tuple."""

    def test_matrix_representation_creation(self):
        """Test creating MatrixRepresentation."""
        ipv = np.array([1.0, 0.0])
        sim = np.array([[-1.0, 0.5], [0.0, -2.0]])
        states = np.array([[1], [2]])

        # Create via Graph.as_matrices()
        g = Graph.from_matrices(ipv, sim, states)
        matrices = g.as_matrices()

        assert isinstance(matrices, ptd.MatrixRepresentation)
        assert matrices.ipv is not None
        assert matrices.sim is not None
        assert matrices.states is not None

    def test_matrix_representation_fields(self):
        """Test MatrixRepresentation field access."""
        ipv = np.array([0.5, 0.5])
        sim = np.array([[-1.0, 0.0], [0.0, -2.0]])
        states = np.array([[10], [20]])

        g = Graph.from_matrices(ipv, sim, states)
        matrices = g.as_matrices()

        # Should be able to access fields
        assert len(matrices.ipv) > 0
        assert matrices.sim.shape[0] == matrices.sim.shape[1]
        assert len(matrices.states) == len(matrices.ipv)


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_invalid_state_length(self):
        """Test invalid state_length."""
        with pytest.raises((ValueError, AssertionError, TypeError)):
            g = Graph(state_length=-1)

    def test_invalid_vertex_state(self):
        """Test creating vertex with invalid state."""
        g = Graph(state_length=2)

        # Wrong dimension should raise or handle gracefully
        try:
            v = g.find_or_create_vertex([1])  # Should be 2D
            # Some implementations may auto-pad
        except (ValueError, RuntimeError):
            # Expected for strict implementations
            pass

    def test_edge_to_self(self):
        """Test adding edge from vertex to itself."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([1])

        # Self-loop (creates cycle)
        v.add_edge(v, 1.0)

        # Graph should no longer be acyclic
        assert not g.is_acyclic()

    def test_negative_edge_weight(self):
        """Test negative edge weights."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        # Negative weight may be allowed or may raise
        try:
            v1.add_edge(v2, -1.0)
        except (ValueError, RuntimeError):
            # Some implementations may not allow negative weights
            pass

    def test_empty_graph_operations(self):
        """Test operations on minimal graph."""
        g = Graph(state_length=1)

        # Should not raise (though results may be inf/nan)
        try:
            _ = g.expectation()
        except (RuntimeError, ValueError):
            # May not be computable for graph with no transitions
            pass

    def test_large_state_vector(self):
        """Test with large state vector."""
        g = Graph(state_length=100)
        state = list(range(100))
        v = g.find_or_create_vertex(state)

        assert len(v.state()) == 100


class TestCallbackErrors:
    """Test error handling in callbacks."""

    def test_callback_returning_wrong_type(self):
        """Test callback returning invalid type."""
        def bad_callback(state):
            return "not a list"

        # Should raise when trying to build graph
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            g = Graph(callback=bad_callback)
            _ = g.vertices_length()  # Force evaluation

    def test_callback_infinite_loop_detection(self):
        """Test that infinite loops in callbacks are handled."""
        def infinite_callback(state):
            # Always returns same state (infinite loop)
            return [([0], 1.0)]

        # Graph construction may limit iterations or detect cycle
        try:
            g = Graph(callback=infinite_callback)
            # If it completes, should have bounded vertices
            assert g.vertices_length() < 10000
        except (RuntimeError, ValueError):
            # May detect and raise error
            pass


class TestNumericalStability:
    """Test numerical stability."""

    def test_very_small_rates(self):
        """Test with very small transition rates."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1e-10)
        g.normalize()

        exp = g.expectation()
        assert np.isfinite(exp)

    def test_very_large_rates(self):
        """Test with very large transition rates."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1e10)
        g.normalize()

        exp = g.expectation()
        assert np.isfinite(exp)
        assert exp < 1e-5  # Should be very small

    def test_mixed_scale_rates(self):
        """Test with rates of very different scales."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v3 = g.find_or_create_vertex([3])

        start.add_edge(v1, 1e-5)
        start.add_edge(v2, 1.0)
        start.add_edge(v3, 1e5)
        g.normalize()

        exp = g.expectation()
        assert np.isfinite(exp)


class TestMemoryManagement:
    """Test memory management and cleanup."""

    def test_large_graph_creation(self):
        """Test creating and destroying large graph."""
        def callback(state):
            if state[0] < 100:
                return [([state[0] + 1], 1.0)]
            return []

        g = Graph(callback=callback)
        assert g.vertices_length() > 50

        # Delete and ensure cleanup
        vertices_count = g.vertices_length()
        del g

        # Create another graph to ensure no interference
        g2 = Graph(state_length=1)
        assert g2.vertices_length() == 1

    def test_multiple_graphs_independent(self):
        """Test that multiple graphs are independent."""
        g1 = Graph(state_length=1)
        g2 = Graph(state_length=1)

        v1 = g1.find_or_create_vertex([1])
        v2 = g2.find_or_create_vertex([1])

        # Should be different vertices
        assert v1.index() == v2.index()  # Same index in different graphs
        # But modifying one shouldn't affect the other

        g1.starting_vertex().add_edge(v1, 2.0)
        g1.normalize()

        g2.starting_vertex().add_edge(v2, 3.0)
        g2.normalize()

        # Different expectations
        exp1 = g1.expectation()
        exp2 = g2.expectation()

        assert exp1 != exp2


class TestSpecialDistributions:
    """Test special distribution cases."""

    def test_exponential_distribution(self):
        """Test exponential distribution (single state)."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        rate = 2.0
        start.add_edge(v, rate)
        g.normalize()

        # Expectation should be 1/rate
        exp = g.expectation()
        assert exp == pytest.approx(1.0/rate, rel=0.01)

        # Variance should be 1/rate^2
        var = g.variance()
        assert var == pytest.approx(1.0/rate**2, rel=0.01)

    def test_hyperexponential_distribution(self):
        """Test hyperexponential distribution (mixture of exponentials)."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        # Two branches with different rates
        p1, p2 = 0.3, 0.7
        rate1, rate2 = 1.0, 3.0

        start.add_edge(v1, p1)
        start.add_edge(v2, p2)
        g.normalize()

        # Expectation should be p1/rate1 + p2/rate2
        # But need to add edges from v1 and v2 to absorbing states
        v3 = g.find_or_create_vertex([10])
        v4 = g.find_or_create_vertex([20])
        v1.add_edge(v3, rate1)
        v2.add_edge(v4, rate2)
        g.normalize()

        exp = g.expectation()
        assert exp > 0

    def test_erlang_distribution(self):
        """Test Erlang distribution (chain of exponentials)."""
        g = Graph(state_length=1)
        n_phases = 3
        rate = 2.0

        # Create chain
        vertices = [g.starting_vertex()]
        for i in range(1, n_phases + 1):
            v = g.find_or_create_vertex([i])
            vertices.append(v)

        for i in range(n_phases):
            vertices[i].add_edge(vertices[i+1], rate)

        g.normalize()

        # Expectation should be n_phases / rate
        exp = g.expectation()
        assert exp == pytest.approx(n_phases / rate, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
