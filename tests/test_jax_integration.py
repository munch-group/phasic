#!/usr/bin/env python
"""
Test suite for JAX integration in ptdalgorithms.
Tests pmf_from_graph, gradients, JIT compilation, and vectorization.
"""

import pytest

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

import numpy as np
import ptdalgorithms as ptd
from ptdalgorithms import Graph

# Skip all tests if JAX is not available
pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not available")


class TestPMFFromGraph:
    """Test pmf_from_graph functionality."""

    def test_pmf_from_graph_continuous(self):
        """Test pmf_from_graph for continuous distributions."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)
        g.normalize()

        # Create model without parameters
        model = Graph.pmf_from_graph(g, discrete=False)

        # Test computation
        times = jnp.array([0.5, 1.0, 2.0])
        pdf = model(times)

        assert pdf.shape == times.shape
        assert jnp.all(pdf >= 0)

    def test_pmf_from_graph_discrete(self):
        """Test pmf_from_graph for discrete distributions."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Create discrete model
        model = Graph.pmf_from_graph(g, discrete=True)

        # Test computation
        jumps = jnp.array([1, 2, 3, 5])
        pmf = model(jumps)

        assert pmf.shape == jumps.shape
        assert jnp.all(pmf >= 0)

    def test_pmf_from_graph_batch(self):
        """Test pmf_from_graph with batch of times."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 1.0)
        v1.add_edge(v2, 0.5)
        g.normalize()

        model = Graph.pmf_from_graph(g, discrete=False)

        # Large batch
        times = jnp.linspace(0.1, 5.0, 100)
        pdf = model(times)

        assert pdf.shape == times.shape
        assert jnp.all(pdf >= 0)


class TestPMFFromGraphParameterized:
    """Test pmf_from_graph_parameterized functionality."""

    def test_parameterized_basic(self):
        """Test basic parameterized graph."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v1 = g.find_or_create_vertex([1])
            v2 = g.find_or_create_vertex([2])
            g.starting_vertex().add_edge_parameterized(v1, 0.0, [1.0, 0.0])
            v1.add_edge_parameterized(v2, 0.0, [0.0, 1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        # Test with parameters
        theta = jnp.array([1.0, 2.0])
        times = jnp.array([0.5, 1.0, 2.0])

        pdf = model(theta, times)
        assert pdf.shape == times.shape
        assert jnp.all(pdf >= 0)

    def test_parameterized_discrete(self):
        """Test parameterized graph in discrete mode."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=True)

        theta = jnp.array([1.5])
        jumps = jnp.array([1, 2, 3])

        pmf = model(theta, jumps)
        assert pmf.shape == jumps.shape

    def test_parameterized_with_callback(self):
        """Test parameterized graph with callback function."""
        def callback(state, theta):
            if state[0] < 3:
                # Birth with rate theta[0], death with rate theta[1]
                return [
                    ([state[0] + 1], 0.0, [1.0, 0.0]),  # Birth
                    ([state[0] - 1] if state[0] > 0 else [0], 0.0, [0.0, 1.0])  # Death
                ]
            return []

        def build_graph(theta):
            return Graph(callback=lambda s: callback(s, theta), parameterized=True)

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        theta = jnp.array([2.0, 1.0])
        times = jnp.array([0.5, 1.0])

        pdf = model(theta, times)
        assert pdf.shape == times.shape


class TestJAXGradients:
    """Test automatic differentiation with JAX."""

    def test_gradient_parameterized(self):
        """Test gradient computation for parameterized graph."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        def loss_fn(theta):
            times = jnp.array([1.0])
            pdf = model(theta, times)
            return pdf.sum()

        theta = jnp.array([1.0])
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(theta)

        assert gradient.shape == theta.shape
        assert jnp.isfinite(gradient).all()

    def test_gradient_multi_parameter(self):
        """Test gradients with multiple parameters."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v1 = g.find_or_create_vertex([1])
            v2 = g.find_or_create_vertex([2])
            g.starting_vertex().add_edge_parameterized(v1, 0.0, [1.0, 0.0])
            v1.add_edge_parameterized(v2, 0.0, [0.0, 1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        def loss_fn(theta):
            times = jnp.array([1.0, 2.0])
            pdf = model(theta, times)
            return pdf.sum()

        theta = jnp.array([1.5, 2.5])
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(theta)

        assert gradient.shape == (2,)
        assert jnp.isfinite(gradient).all()

    def test_value_and_grad(self):
        """Test value_and_grad for simultaneous computation."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        def loss_fn(theta):
            times = jnp.array([1.0])
            return model(theta, times).sum()

        theta = jnp.array([2.0])
        value_and_grad_fn = jax.value_and_grad(loss_fn)
        value, gradient = value_and_grad_fn(theta)

        assert jnp.isfinite(value)
        assert jnp.isfinite(gradient).all()


class TestJAXJIT:
    """Test JIT compilation."""

    def test_jit_pmf_from_graph(self):
        """Test JIT compilation of pmf_from_graph."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        model = Graph.pmf_from_graph(g, discrete=False)
        jit_model = jax.jit(model)

        times = jnp.array([0.5, 1.0, 2.0])

        # First call (compilation)
        pdf1 = jit_model(times)

        # Second call (cached)
        pdf2 = jit_model(times)

        assert jnp.allclose(pdf1, pdf2)

    def test_jit_parameterized(self):
        """Test JIT compilation of parameterized model."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)
        jit_model = jax.jit(model)

        theta = jnp.array([1.5])
        times = jnp.array([1.0, 2.0])

        pdf1 = jit_model(theta, times)
        pdf2 = jit_model(theta, times)

        assert jnp.allclose(pdf1, pdf2)

    def test_jit_with_grad(self):
        """Test JIT compilation of gradient function."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        def loss_fn(theta):
            times = jnp.array([1.0])
            return model(theta, times).sum()

        grad_fn = jax.grad(loss_fn)
        jit_grad_fn = jax.jit(grad_fn)

        theta = jnp.array([1.0])

        grad1 = jit_grad_fn(theta)
        grad2 = jit_grad_fn(theta)

        assert jnp.allclose(grad1, grad2)


class TestJAXVmap:
    """Test vectorization with vmap."""

    def test_vmap_over_parameters(self):
        """Test vmap over parameter batch."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        # Batch of parameters
        theta_batch = jnp.array([[1.0], [2.0], [3.0]])
        times = jnp.array([1.0, 2.0])

        # Vmap over first axis (parameters)
        vmap_model = jax.vmap(lambda t: model(t, times))
        pdf_batch = vmap_model(theta_batch)

        assert pdf_batch.shape == (3, 2)  # (n_params, n_times)

    def test_vmap_over_times(self):
        """Test vmap over time batch."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        model = Graph.pmf_from_graph(g, discrete=False)

        # Batch of times (each as single value)
        times_batch = jnp.array([0.5, 1.0, 1.5, 2.0])

        # Vmap over individual times
        vmap_model = jax.vmap(lambda t: model(jnp.array([t])))
        pdf_batch = vmap_model(times_batch)

        assert pdf_batch.shape == (4, 1)

    def test_vmap_nested(self):
        """Test nested vmap."""
        def build_graph(theta):
            g = Graph(state_length=1)
            v = g.find_or_create_vertex([1])
            g.starting_vertex().add_edge_parameterized(v, 0.0, [1.0])
            return g

        model = Graph.pmf_from_graph_parameterized(build_graph, discrete=False)

        theta_batch = jnp.array([[1.0], [2.0]])
        times_batch = jnp.array([[0.5, 1.0], [1.5, 2.0]])

        # Nested vmap: outer over parameters, inner over times
        vmap_model = jax.vmap(lambda t, ts: model(t, ts))
        pdf_batch = vmap_model(theta_batch, times_batch)

        assert pdf_batch.shape == (2, 2)


class TestMomentsFromGraph:
    """Test moments_from_graph functionality."""

    def test_moments_from_graph_basic(self):
        """Test moments computation from graph."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Create moments function
        moments_fn = Graph.moments_from_graph(g, nr_moments=2)

        # Compute moments
        moments = moments_fn()

        assert len(moments) == 2
        assert jnp.all(jnp.isfinite(moments))
        assert moments[0] > 0  # First moment should be positive

    def test_moments_from_graph_higher_order(self):
        """Test higher-order moments."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)
        g.normalize()

        moments_fn = Graph.moments_from_graph(g, nr_moments=4)
        moments = moments_fn()

        assert len(moments) == 4
        assert jnp.all(jnp.isfinite(moments))


class TestPMFAndMomentsFromGraph:
    """Test combined PMF and moments computation."""

    def test_pmf_and_moments_basic(self):
        """Test combined PMF and moments computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

        times = jnp.array([0.5, 1.0])
        pdf, moments = model(times)

        assert pdf.shape == times.shape
        assert len(moments) == 2
        assert jnp.all(jnp.isfinite(pdf))
        assert jnp.all(jnp.isfinite(moments))

    def test_pmf_and_moments_discrete(self):
        """Test combined computation in discrete mode."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        model = Graph.pmf_and_moments_from_graph(g, nr_moments=3, discrete=True)

        jumps = jnp.array([1, 2, 3])
        pmf, moments = model(jumps)

        assert pmf.shape == jumps.shape
        assert len(moments) == 3


class TestBatchOperations:
    """Test batch computation methods."""

    def test_pdf_batch(self):
        """Test pdf_batch method."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        times = np.linspace(0.1, 5.0, 100)
        pdf_values = g.pdf_batch(times)

        assert pdf_values.shape == times.shape
        assert np.all(pdf_values >= 0)

    def test_dph_pmf_batch(self):
        """Test dph_pmf_batch method."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        jumps = np.array([1, 2, 3, 5, 10])
        pmf_values = g.dph_pmf_batch(jumps)

        assert pmf_values.shape == jumps.shape
        assert np.all(pmf_values >= 0)

    def test_moments_batch(self):
        """Test moments_batch method."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        powers = np.array([1, 2, 3])
        moments = g.moments_batch(powers)

        assert moments.shape == powers.shape
        assert np.all(np.isfinite(moments))


class TestMultivariateSampling:
    """Test multivariate sampling methods."""

    def test_sample_multivariate(self):
        """Test multivariate sampling."""
        g = Graph(state_length=2)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1, 1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Sample with rewards
        rewards = np.array([[1, 0], [0, 1]], dtype=float)
        sample = g.sample_multivariate(rewards)

        assert len(sample) == 2
        assert np.all(np.isfinite(sample))

    def test_sample_multivariate_discrete(self):
        """Test discrete multivariate sampling."""
        g = Graph(state_length=2)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1, 1])
        start.add_edge(v, 1.0)
        g.normalize()

        rewards = np.array([[1, 0], [0, 1]], dtype=float)
        sample = g.sample_multivariate_discrete(rewards)

        assert len(sample) == 2
        assert np.all(np.isfinite(sample))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
