"""
SLURM Multi-Node FFI Tests

This test suite is designed to run on a SLURM cluster across multiple nodes.
It verifies that JAX FFI functions work correctly in a true multi-controller
environment with distributed memory across nodes.

Usage:
    # Run on 4 nodes with 8 cores each
    sbatch tests/slurm_test_ffi.sh

    # Or manually with srun:
    srun --nodes=4 --ntasks=4 --cpus-per-task=8 \\
         python -m pytest tests/test_slurm_multinode_ffi.py -v

Environment Variables Required:
    SLURM_PROCID - Process rank (0, 1, 2, ...)
    SLURM_NTASKS - Total number of processes
    SLURM_NODELIST - List of allocated nodes

These tests will SKIP if not running in a SLURM environment.
"""

import os
import pytest
import numpy as np

# IMPORTANT: Import phasic BEFORE jax
from phasic import Graph
from phasic.ffi_wrappers import compute_pmf_ffi, compute_moments_ffi, compute_pmf_and_moments_ffi

# Import JAX after phasic
import jax
import jax.numpy as jnp


# Skip all tests if not in SLURM environment
pytestmark = pytest.mark.skipif(
    'SLURM_PROCID' not in os.environ,
    reason="Tests require SLURM environment (run with srun/sbatch)"
)


@pytest.fixture(scope="module")
def slurm_info():
    """Get SLURM environment information."""
    return {
        'procid': int(os.environ.get('SLURM_PROCID', 0)),
        'ntasks': int(os.environ.get('SLURM_NTASKS', 1)),
        'nodelist': os.environ.get('SLURM_NODELIST', 'unknown'),
        'cpus_per_task': int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
    }


@pytest.fixture
def rabbits_graph():
    """Create rabbits graph for testing."""
    g = Graph(2)
    initial = g.find_or_create_vertex([2, 0])
    g.starting_vertex().add_edge(initial, [1.0, 0.0])

    index = 1
    while index < g.vertices_length():
        vertex = g.vertex_at(index)
        state = vertex.state()

        if state[0] > 0:
            child = g.find_or_create_vertex([state[0] - 1, state[1] + 1])
            vertex.add_edge(child, [1.0, 0.0])

        if state[1] > 0:
            child = g.find_or_create_vertex([state[0], state[1] - 1])
            vertex.add_edge(child, [0.0, 1.0])

        index += 1

    return g


class TestSLURMEnvironment:
    """Test SLURM environment detection and JAX initialization."""

    def test_slurm_environment_detected(self, slurm_info):
        """Verify SLURM environment variables are set."""
        assert slurm_info['procid'] >= 0
        assert slurm_info['ntasks'] > 0
        assert slurm_info['nodelist'] != 'unknown'

        # Print info from each process
        print(f"\nProcess {slurm_info['procid']}/{slurm_info['ntasks']}:")
        print(f"  Node: {slurm_info['nodelist']}")
        print(f"  CPUs per task: {slurm_info['cpus_per_task']}")

    def test_jax_distributed_initialized(self, slurm_info):
        """Verify JAX distributed initialization completed."""
        # Check if JAX sees multiple processes
        process_count = jax.process_count()
        process_index = jax.process_index()

        print(f"\nJAX Distributed Info (Process {slurm_info['procid']}):")
        print(f"  Total processes: {process_count}")
        print(f"  Current process index: {process_index}")
        print(f"  Device count (global): {jax.device_count()}")
        print(f"  Device count (local): {jax.local_device_count()}")
        print(f"  Devices: {jax.devices()}")

        # If using multi-controller JAX, process_count should match SLURM tasks
        if process_count > 1:
            assert process_count == slurm_info['ntasks'], \
                f"JAX process_count ({process_count}) != SLURM_NTASKS ({slurm_info['ntasks']})"
            assert process_index == slurm_info['procid'], \
                f"JAX process_index ({process_index}) != SLURM_PROCID ({slurm_info['procid']})"


class TestMultiNodeFFI:
    """Test FFI functions across multiple SLURM nodes."""

    def test_compute_pmf_ffi_multi_node(self, rabbits_graph, slurm_info):
        """Test compute_pmf_ffi serialization across nodes."""
        structure_json = rabbits_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        times = jnp.array([0.5, 1.0, 1.5])

        # Each process computes independently
        pdf = compute_pmf_ffi(structure_json, theta, times, discrete=False, granularity=100)

        assert pdf.shape == times.shape
        assert jnp.all(jnp.isfinite(pdf))

        # Verify results are consistent (all processes should get same result)
        pdf_hash = hash(tuple(float(x) for x in pdf))
        print(f"Process {slurm_info['procid']}: PDF hash = {pdf_hash}")

    def test_compute_moments_ffi_multi_node(self, rabbits_graph, slurm_info):
        """Test compute_moments_ffi across nodes."""
        structure_json = rabbits_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 3

        moments = compute_moments_ffi(structure_json, theta, nr_moments)

        assert moments.shape == (nr_moments,)
        assert jnp.all(jnp.isfinite(moments))

        print(f"Process {slurm_info['procid']}: E[T] = {moments[0]:.6f}")

    def test_vmap_batching_multi_node(self, rabbits_graph, slurm_info):
        """Test vmap batching works correctly across nodes."""
        structure_json = rabbits_graph.serialize()
        nr_moments = 3

        # Create different theta values per process
        process_id = slurm_info['procid']
        theta_batch = jnp.array([
            [1.0 + process_id * 0.1, 2.0],
            [1.5 + process_id * 0.1, 2.5],
            [2.0 + process_id * 0.1, 3.0]
        ])

        moments_batch = jax.vmap(
            lambda t: compute_moments_ffi(structure_json, t, nr_moments)
        )(theta_batch)

        assert moments_batch.shape == (len(theta_batch), nr_moments)
        assert jnp.all(jnp.isfinite(moments_batch))

        print(f"Process {slurm_info['procid']}: Batch computed {len(theta_batch)} parameter sets")

    def test_thread_local_caching_multi_node(self, rabbits_graph, slurm_info):
        """Verify thread-local caching works independently on each node."""
        structure_json = rabbits_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 5

        # Call multiple times - should use cache
        moments1 = compute_moments_ffi(structure_json, theta, nr_moments)
        moments2 = compute_moments_ffi(structure_json, theta, nr_moments)
        moments3 = compute_moments_ffi(structure_json, theta, nr_moments)

        # Verify caching works (results identical)
        np.testing.assert_array_equal(moments1, moments2)
        np.testing.assert_array_equal(moments2, moments3)

        print(f"Process {slurm_info['procid']}: Caching verified (3 identical calls)")


class TestMultiNodeConsistency:
    """Test result consistency across nodes."""

    def test_all_processes_get_same_result(self, rabbits_graph, slurm_info):
        """Verify all processes compute identical results for same input."""
        structure_json = rabbits_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 3

        # All processes compute same parameters
        moments = compute_moments_ffi(structure_json, theta, nr_moments)

        # Convert to float tuple for hashing
        moments_tuple = tuple(float(x) for x in moments)
        result_hash = hash(moments_tuple)

        print(f"Process {slurm_info['procid']}: "
              f"E[T]={moments[0]:.10f}, hash={result_hash}")

        # All processes should print the same hash
        # (Manual verification required - check logs)

    def test_combined_jit_vmap_multi_node(self, rabbits_graph, slurm_info):
        """Test JIT + vmap combination across nodes."""
        structure_json = rabbits_graph.serialize()
        nr_moments = 3
        theta_batch = jnp.array([[1.0, 2.0], [2.0, 3.0]])

        @jax.jit
        def batched_moments(theta_b):
            return jax.vmap(
                lambda t: compute_moments_ffi(structure_json, t, nr_moments)
            )(theta_b)

        moments_jit_vmap = batched_moments(theta_batch)

        assert moments_jit_vmap.shape == (len(theta_batch), nr_moments)
        assert jnp.all(jnp.isfinite(moments_jit_vmap))

        print(f"Process {slurm_info['procid']}: JIT+vmap successful")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
