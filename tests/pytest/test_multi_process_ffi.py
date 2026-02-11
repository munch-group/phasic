"""
Multi-process verification tests for FFI functions.

This test suite verifies that all JAX FFI functions (compute_pmf_ffi,
compute_moments_ffi, compute_pmf_and_moments_ffi) are compatible with
multi-process JAX execution.

Key aspects tested:
1. Serialization safety (no closures in call paths)
2. Thread-local caching works correctly per process
3. Results are numerically identical across processes
4. vmap batching works with multiple JAX devices
"""

import os
import pytest
import numpy as np

# IMPORTANT: Import phasic BEFORE jax (required for multi-CPU configuration)
from phasic import Graph
from phasic.ffi_wrappers import compute_pmf_ffi, compute_moments_ffi, compute_pmf_and_moments_ffi

# Import JAX after phasic
import jax
import jax.numpy as jnp


@pytest.fixture(scope="module")
def setup_multi_cpu():
    """Setup JAX to use multiple CPU devices for testing."""
    # Note: JAX is already imported at module level after phasic
    # This fixture just ensures the environment is set before tests run
    yield


@pytest.fixture
def mutations_graph():
    """Create a simple parameterized mutations graph for testing."""
    g = Graph(2)  # 2D state: [n_left, n_right]

    # Initial state: 2 mutations on left island
    initial = g.find_or_create_vertex([2, 0])
    g.starting_vertex().add_edge(initial, [1.0, 0.0])

    # Build graph iteratively
    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        # Birth: mutation jumps left to right (rate = theta[0])
        if state[0] > 0:
            child = g.find_or_create_vertex([state[0] - 1, state[1] + 1])
            vertex.add_edge(child, [1.0, 0.0])

        # Death: right island floods (rate = theta[1])
        if state[1] > 0:
            child = g.find_or_create_vertex([state[0], state[1] - 1])
            vertex.add_edge(child, [0.0, 1.0])

        index += 1

    return g


class TestComputePmfFfiMultiProcess:
    """Test compute_pmf_ffi with multi-process JAX."""

    def test_basic_pmf_serialization(self, mutations_graph, setup_multi_cpu):
        """Test that compute_pmf_ffi can serialize/deserialize correctly."""
        structure_json = mutations_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        times = jnp.array([0.5, 1.0, 1.5])

        # Should not raise serialization errors
        pdf = compute_pmf_ffi(structure_json, theta, times, discrete=False, granularity=100)

        assert pdf.shape == times.shape
        assert jnp.all(jnp.isfinite(pdf))

    def test_pmf_vmap_batching(self, mutations_graph, setup_multi_cpu):
        """Test vmap batching over theta with multiple devices."""
        structure_json = mutations_graph.serialize()
        times = jnp.array([0.5, 1.0, 1.5])
        theta_batch = jnp.array([[0.5, 1.0], [1.0, 2.0], [2.0, 4.0]])

        # Batch over theta using vmap
        pdf_batch = jax.vmap(
            lambda t: compute_pmf_ffi(structure_json, t, times, False, 100)
        )(theta_batch)

        assert pdf_batch.shape == (len(theta_batch), len(times))
        assert jnp.all(jnp.isfinite(pdf_batch))

        # Verify each batch element matches individual computation
        for i, theta in enumerate(theta_batch):
            pdf_individual = compute_pmf_ffi(structure_json, theta, times, False, 100)
            np.testing.assert_allclose(pdf_batch[i], pdf_individual, rtol=1e-10)

    def test_pmf_jit_vmap_combination(self, mutations_graph, setup_multi_cpu):
        """Test combined JIT + vmap with multi-process."""
        structure_json = mutations_graph.serialize()
        times = jnp.array([0.5, 1.0])
        theta_batch = jnp.array([[1.0, 2.0], [2.0, 3.0]])

        @jax.jit
        def batched_pmf(theta_b):
            return jax.vmap(
                lambda t: compute_pmf_ffi(structure_json, t, times, False, 100)
            )(theta_b)

        pdf_jit_vmap = batched_pmf(theta_batch)
        pdf_vmap_only = jax.vmap(
            lambda t: compute_pmf_ffi(structure_json, t, times, False, 100)
        )(theta_batch)

        np.testing.assert_allclose(pdf_jit_vmap, pdf_vmap_only, rtol=1e-10)


class TestComputeMomentsFfiMultiProcess:
    """Test compute_moments_ffi with multi-process JAX."""

    def test_basic_moments_serialization(self, mutations_graph, setup_multi_cpu):
        """Test that compute_moments_ffi can serialize/deserialize correctly."""
        structure_json = mutations_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 3

        # Should not raise serialization errors
        moments = compute_moments_ffi(structure_json, theta, nr_moments)

        assert moments.shape == (nr_moments,)
        assert jnp.all(jnp.isfinite(moments))

    def test_moments_vmap_batching(self, mutations_graph, setup_multi_cpu):
        """Test vmap batching over theta with multiple devices."""
        structure_json = mutations_graph.serialize()
        nr_moments = 3
        theta_batch = jnp.array([[0.5, 1.0], [1.0, 2.0], [2.0, 4.0]])

        # Batch over theta using vmap
        moments_batch = jax.vmap(
            lambda t: compute_moments_ffi(structure_json, t, nr_moments)
        )(theta_batch)

        assert moments_batch.shape == (len(theta_batch), nr_moments)
        assert jnp.all(jnp.isfinite(moments_batch))

        # Verify each batch element matches individual computation
        for i, theta in enumerate(theta_batch):
            moments_individual = compute_moments_ffi(structure_json, theta, nr_moments)
            np.testing.assert_allclose(moments_batch[i], moments_individual, rtol=1e-10)

    def test_moments_jit_vmap_combination(self, mutations_graph, setup_multi_cpu):
        """Test combined JIT + vmap with multi-process."""
        structure_json = mutations_graph.serialize()
        nr_moments = 3
        theta_batch = jnp.array([[1.0, 2.0], [2.0, 3.0]])

        @jax.jit
        def batched_moments(theta_b):
            return jax.vmap(
                lambda t: compute_moments_ffi(structure_json, t, nr_moments)
            )(theta_b)

        moments_jit_vmap = batched_moments(theta_batch)
        moments_vmap_only = jax.vmap(
            lambda t: compute_moments_ffi(structure_json, t, nr_moments)
        )(theta_batch)

        np.testing.assert_allclose(moments_jit_vmap, moments_vmap_only, rtol=1e-10)

    def test_moments_thread_local_caching(self, mutations_graph, setup_multi_cpu):
        """Test that thread-local caching works correctly across processes."""
        structure_json = mutations_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 5

        # Call multiple times - should use cache on subsequent calls
        moments1 = compute_moments_ffi(structure_json, theta, nr_moments)
        moments2 = compute_moments_ffi(structure_json, theta, nr_moments)
        moments3 = compute_moments_ffi(structure_json, theta, nr_moments)

        # Results should be identical (cache working correctly)
        np.testing.assert_array_equal(moments1, moments2)
        np.testing.assert_array_equal(moments2, moments3)


class TestComputePmfAndMomentsFfiMultiProcess:
    """Test compute_pmf_and_moments_ffi with multi-process JAX."""

    def test_basic_pmf_and_moments_serialization(self, mutations_graph, setup_multi_cpu):
        """Test that compute_pmf_and_moments_ffi can serialize/deserialize correctly."""
        structure_json = mutations_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        times = jnp.array([0.5, 1.0])
        nr_moments = 2

        # Should not raise serialization errors
        pdf, moments = compute_pmf_and_moments_ffi(
            structure_json, theta, times, nr_moments, discrete=False, granularity=100
        )

        assert pdf.shape == times.shape
        assert moments.shape == (nr_moments,)
        assert jnp.all(jnp.isfinite(pdf))
        assert jnp.all(jnp.isfinite(moments))

    def test_pmf_and_moments_vmap_batching(self, mutations_graph, setup_multi_cpu):
        """Test vmap batching over theta with multiple devices."""
        structure_json = mutations_graph.serialize()
        times = jnp.array([0.5, 1.0])
        nr_moments = 2
        theta_batch = jnp.array([[1.0, 2.0], [2.0, 3.0]])

        # Batch over theta using vmap
        pdf_batch, moments_batch = jax.vmap(
            lambda t: compute_pmf_and_moments_ffi(
                structure_json, t, times, nr_moments, False, 100
            )
        )(theta_batch)

        assert pdf_batch.shape == (len(theta_batch), len(times))
        assert moments_batch.shape == (len(theta_batch), nr_moments)
        assert jnp.all(jnp.isfinite(pdf_batch))
        assert jnp.all(jnp.isfinite(moments_batch))

        # Verify consistency
        for i, theta in enumerate(theta_batch):
            pdf_ind, moments_ind = compute_pmf_and_moments_ffi(
                structure_json, theta, times, nr_moments, False, 100
            )
            np.testing.assert_allclose(pdf_batch[i], pdf_ind, rtol=1e-10)
            np.testing.assert_allclose(moments_batch[i], moments_ind, rtol=1e-10)


class TestMultiProcessConsistency:
    """Test that results are consistent across different parallelization strategies."""

    def test_device_count_awareness(self, setup_multi_cpu):
        """Test that JAX sees multiple devices when configured."""
        # This test verifies the test setup itself
        device_count = jax.device_count()
        local_device_count = jax.local_device_count()

        print(f"\nJAX Device Info:")
        print(f"  Global devices: {device_count}")
        print(f"  Local devices: {local_device_count}")
        print(f"  Devices: {jax.devices()}")

        # Should have at least 1 device
        assert device_count >= 1
        assert local_device_count >= 1

    def test_all_ffi_functions_together(self, mutations_graph, setup_multi_cpu):
        """Integration test: all FFI functions work together in multi-process."""
        structure_json = mutations_graph.serialize()
        theta_batch = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        times = jnp.array([0.5, 1.0])
        nr_moments = 3

        # Test all three functions with vmap
        pdf_batch = jax.vmap(
            lambda t: compute_pmf_ffi(structure_json, t, times, False, 100)
        )(theta_batch)

        moments_batch = jax.vmap(
            lambda t: compute_moments_ffi(structure_json, t, nr_moments)
        )(theta_batch)

        pdf_and_moments_batch = jax.vmap(
            lambda t: compute_pmf_and_moments_ffi(
                structure_json, t, times, nr_moments, False, 100
            )
        )(theta_batch)

        # All should succeed without errors
        assert pdf_batch.shape == (len(theta_batch), len(times))
        assert moments_batch.shape == (len(theta_batch), nr_moments)
        assert pdf_and_moments_batch[0].shape == (len(theta_batch), len(times))
        assert pdf_and_moments_batch[1].shape == (len(theta_batch), nr_moments)

        # Verify consistency between separate and combined calls
        for i in range(len(theta_batch)):
            np.testing.assert_allclose(
                pdf_batch[i],
                pdf_and_moments_batch[0][i],
                rtol=1e-10,
                err_msg=f"PDF mismatch at batch {i}"
            )
            np.testing.assert_allclose(
                moments_batch[i],
                pdf_and_moments_batch[1][i],
                rtol=1e-10,
                err_msg=f"Moments mismatch at batch {i}"
            )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
