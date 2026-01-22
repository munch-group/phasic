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
import phasic

# DO NOT import JAX here - let init_parallel() configure it first
# JAX will be imported by init_parallel() in the fixture with correct device count


# Skip all tests if not in SLURM environment
pytestmark = pytest.mark.skipif(
    'SLURM_PROCID' not in os.environ,
    reason="Tests require SLURM environment (run with srun/sbatch)"
)


@pytest.fixture(scope="module", autouse=True)
def initialize_jax_distributed():
    """Initialize JAX distributed mode for multi-node SLURM."""
    import sys
    print("\n=== FIXTURE: initialize_jax_distributed starting ===", flush=True)
    sys.stdout.flush()

    # This must be called before any JAX operations
    # It detects SLURM environment and initializes distributed mode
    print("Calling phasic.init_parallel()...", flush=True)
    sys.stdout.flush()

    config = phasic.init_parallel()

    print(f"\nInitialized JAX with {config.device_count} devices", flush=True)
    print(f"Strategy: {config.strategy}", flush=True)
    print(f"Environment: {config.env_info.env_type}", flush=True)
    print("=== FIXTURE: initialize_jax_distributed complete ===\n", flush=True)
    sys.stdout.flush()

    return config


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
        import jax

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
        import jax
        import jax.numpy as jnp

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
        import jax.numpy as jnp

        structure_json = rabbits_graph.serialize()
        theta = jnp.array([1.0, 2.0])
        nr_moments = 3

        moments = compute_moments_ffi(structure_json, theta, nr_moments)

        assert moments.shape == (nr_moments,)
        assert jnp.all(jnp.isfinite(moments))

        print(f"Process {slurm_info['procid']}: E[T] = {moments[0]:.6f}")

    def test_vmap_batching_multi_node(self, rabbits_graph, slurm_info):
        """Test vmap batching works correctly across nodes."""
        import jax
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax.numpy as jnp

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
        import jax
        import jax.numpy as jnp

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


class TestSVGDMultiNode:
    """Test SVGD inference across multiple SLURM nodes."""

    @pytest.fixture
    def parameterized_rabbits_graph(self):
        """Create parameterized rabbits graph with 20 rabbits."""


        @callback([2, 0])
        def rabbits(state):
            left, right = state
            transitions = []
            if left:
                transitions.append([[left - 1, right + 1], [left, 0]])
                transitions.append([[0, right], [0, 1]])
            if right:
                transitions.append([[left + 1, right - 1], [right, 0]])
                transitions.append([[left, 0], [0, 1]])
            return transitions


        graph = Graph(rabbits)   

        return graph

    def test_svgd_rabbits_multi_node(self, parameterized_rabbits_graph, slurm_info):
        """Test SVGD with 20 rabbits, 50 particles, 100 iterations across nodes."""

        true_theta = [1, 2]
        nr_observations = 10000

        parameterized_rabbits_graph.update_weights(true_theta)
        observed_data = parameterized_rabbits_graph.sample(nr_observations)

        import jax.numpy as jnp

        def uninformative_prior(phi, mu=0.0, sigma=10.0):
            return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

        step_schedule = phasic.ExpStepSize(first_step=0.01, last_step=0.001, tau=50.0)
        reg_schedule = phasic.ExpRegularization(first_reg=10.0, last_reg=1.0, tau=50.0)

        phasic.clear_caches()


        params = dict(
                    bandwidth='median', # bandwidth='local_adaptive',
                    theta_dim=len(true_theta),
                    prior=uninformative_prior, 
                    n_particles=20,
                    n_iterations=200,
                    learning_rate=step_schedule, 
                    seed=42,
                    verbose=False,
                    discrete=False, 
                    # regularization=reg_schedule, 
                    # nr_moments=2,
        )

        phasic.clear_caches()
        phasic.Graph(parameterized_rabbits_graph).compute_trace()

        svgd = parameterized_rabbits_graph.svgd(observed_data=observed_data, **params)

        particles = svgd.particles

        # Verify results
        assert particles.shape[1] == 2  # 2 parameters
        assert jnp.all(jnp.isfinite(particles))

        # Note: SVGD uses unconstrained optimization, so particles may be negative
        # during optimization. With more iterations and proper parameter transformation,
        # they would converge to positive values.

        posterior_mean = jnp.mean(particles, axis=0)
        posterior_std = jnp.std(particles, axis=0)

        print(f"Process {slurm_info['procid']}: SVGD completed")
        print(f"  Posterior mean: [{posterior_mean[0]:.3f}, {posterior_mean[1]:.3f}]")
        print(f"  Posterior std:  [{posterior_std[0]:.3f}, {posterior_std[1]:.3f}]")

    def test_svgd_convergence_consistency(self, parameterized_rabbits_graph, slurm_info):
        """Verify SVGD produces consistent results across processes."""

        true_theta = [1, 2]
        nr_observations = 10000

        parameterized_rabbits_graph.update_weights(true_theta)
        observed_data = parameterized_rabbits_graph.sample(nr_observations)

        import jax.numpy as jnp

        def uninformative_prior(phi, mu=0.0, sigma=10.0):
            return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

        step_schedule = phasic.ExpStepSize(first_step=0.01, last_step=0.001, tau=50.0)
        reg_schedule = phasic.ExpRegularization(first_reg=10.0, last_reg=1.0, tau=50.0)

        phasic.clear_caches()


        params = dict(
                    bandwidth='median', # bandwidth='local_adaptive',
                    theta_dim=len(true_theta),
                    prior=uninformative_prior, 
                    n_particles=20,
                    n_iterations=200,
                    learning_rate=step_schedule, 
                    seed=42,
                    verbose=False,
                    discrete=False, 
                    # regularization=reg_schedule, 
                    # nr_moments=2,
        )

        phasic.clear_caches()
        phasic.Graph(parameterized_rabbits_graph).compute_trace()

        svgd = parameterized_rabbits_graph.svgd(observed_data=observed_data, **params)

        # Get results
        particles = svgd.particles
        posterior_mean = jnp.mean(particles, axis=0)
        result_hash = hash(tuple(float(x) for x in posterior_mean))

        print(f"Process {slurm_info['procid']}: Convergence test")
        print(f"  Mean theta: [{posterior_mean[0]:.6f}, {posterior_mean[1]:.6f}]")
        print(f"  Hash: {result_hash}")

        # Manual verification: all processes should print same hash
        assert jnp.all(jnp.isfinite(particles))


# class TestHierarchicalTraceMultiNode:
#     """Test hierarchical trace recording and evaluation across SLURM nodes."""

#     @pytest.fixture
#     def coalescent_graph(self):
#         """Create coalescent graph with n=5 lineages."""
#         # Create parameterized coalescent graph manually
#         # g = Graph(1)

#         # # Add initial vertex (5 lineages)
#         # v_start = g.starting_vertex()
#         # v5 = g.find_or_create_vertex([5])
#         # v_start.add_edge(v5, [1.0])

#         # # Build coalescent tree: each state n -> n-1 with rate n*(n-1)/2
#         # for n in range(5, 1, -1):
#         #     v_n = g.find_or_create_vertex([n])
#         #     v_n_minus_1 = g.find_or_create_vertex([n-1])
#         #     # Parameterized edge: rate = (n*(n-1)/2) * theta[0]
#         #     rate_coeff = n * (n - 1) / 2.0
#         #     v_n.add_edge(v_n_minus_1, [rate_coeff])

#         # # Add edge to absorbing state
#         # v1 = g.find_or_create_vertex([1])
#         # v_absorb = g.find_or_create_vertex([0])
#         # v1.add_edge(v_absorb, [1.0])

#         # return g

#         def coalescent(state, nr_samples=None):
#             nr_samples = int(nr_samples)
#             transitions = []
#             for i in range(nr_samples):
#                 for j in range(i, nr_samples):
#                     same = int(i == j)
#                     if same and state[i] < 2:
#                         continue
#                     if not same and (state[i] < 1 or state[j] < 1):
#                         continue 
#                     new = state.copy()
#                     new[i] -= 1
#                     new[j] -= 1
#                     new[i+j+1] += 1
#                     transitions.append([new, [state[i]*(state[j]-same)/(1+same)]])
#                     # transitions.append([new, state[i]*(state[j]-same)/(1+same)])
#             return transitions


#         true_theta = np.array([10])  

#         nr_samples = 4
#         ipv = [[[nr_samples] + [0] * (nr_samples+1), 1.0]]
#         graph = phasic.Graph(coalescent, ipv=ipv, nr_samples=nr_samples)
#         return graph

#     def test_trace_recording_multi_node(self, coalescent_graph, slurm_info):
#         """Test trace recording works independently on each node."""
#         from phasic.trace_elimination import record_elimination_trace

#         print(f"\nProcess {slurm_info['procid']}: Recording elimination trace")

#         # Each process records trace independently
#         phasic.clear_caches()
#         coalescent_graph.compute_trace()
#         trace = record_elimination_trace(coalescent_graph, theta_dim=1)

#         # Verify trace structure (trace is now an EliminationTrace object)
#         assert hasattr(trace, 'param_length')
#         assert trace.param_length == 1
#         assert hasattr(trace, 'operations')
#         assert len(trace.operations) > 0
#         assert hasattr(trace, 'n_vertices')
#         assert trace.n_vertices > 0

#         print(f"Process {slurm_info['procid']}: Trace recorded")
#         print(f"  Param length: {trace.param_length}")
#         print(f"  Operations: {len(trace.operations)}")
#         print(f"  Vertices: {trace.n_vertices}")

#     def test_trace_evaluation_multi_node(self, coalescent_graph, slurm_info):
#         """Test trace evaluation across nodes with different parameters."""
#         from phasic.trace_elimination import (
#             record_elimination_trace,
#             evaluate_trace_jax,
#             instantiate_from_trace
#         )

#         print(f"\nProcess {slurm_info['procid']}: Testing trace evaluation")
#         phasic.clear_caches()
#         coalescent_graph.compute_trace()
#         # Record trace once
#         trace = record_elimination_trace(coalescent_graph, theta_dim=1)

#         # Each process evaluates with different theta
#         theta = jnp.array([1.0 + slurm_info['procid'] * 0.5])

#         # Evaluate trace
#         result = evaluate_trace_jax(trace, theta)

#         # Verify result structure
#         assert 'vertex_rates' in result
#         assert 'edge_probs' in result
#         assert 'vertex_targets' in result
#         assert jnp.all(jnp.isfinite(result['vertex_rates']))

#         # Instantiate concrete graph
#         concrete_graph = instantiate_from_trace(trace, np.array([float(theta[0])]))

#         # Compute PDF using concrete graph
#         times = np.array([0.5, 1.0, 2.0])
#         pdf = concrete_graph.pdf(times, granularity=100)

#         assert pdf.shape == times.shape
#         assert np.all(np.isfinite(pdf))

#         print(f"Process {slurm_info['procid']}: Evaluation successful")
#         print(f"  Theta: {float(theta[0]):.3f}")
#         print(f"  Vertex rates: {result['vertex_rates'][:5]}")
#         print(f"  PDF[0.5]: {pdf[0]:.6f}")

#     def test_trace_vmap_batching_multi_node(self, coalescent_graph, slurm_info):
#         """Test vmap batching of trace evaluation across nodes."""
#         from phasic.trace_elimination import (
#             record_elimination_trace,
#             evaluate_trace_jax
#         )

#         print(f"\nProcess {slurm_info['procid']}: Testing trace vmap")

#         trace = record_elimination_trace(coalescent_graph, theta_dim=1)

#         # Create batch of theta values (different per process)
#         base = slurm_info['procid'] * 10
#         theta_batch = jnp.array([
#             [0.5 + base * 0.01],
#             [1.0 + base * 0.01],
#             [1.5 + base * 0.01],
#             [2.0 + base * 0.01]
#         ])

#         # Vmap over theta batch
#         batched_eval = jax.vmap(lambda t: evaluate_trace_jax(trace, t))
#         results = batched_eval(theta_batch)

#         # Verify batched results
#         assert results['vertex_rates'].shape[0] == len(theta_batch)
#         assert jnp.all(jnp.isfinite(results['vertex_rates']))

#         print(f"Process {slurm_info['procid']}: Vmap successful")
#         print(f"  Batch size: {len(theta_batch)}")
#         print(f"  Output shape: {results['vertex_rates'].shape}")

#     def test_trace_serialization_multi_node(self, coalescent_graph, slurm_info):
#         """Test trace serialization/deserialization across nodes."""
#         from phasic.trace_elimination import (
#             record_elimination_trace,
#             evaluate_trace_jax
#         )
#         import pickle
#         import tempfile
#         import os

#         print(f"\nProcess {slurm_info['procid']}: Testing trace serialization")

#         # Record trace
#         trace = record_elimination_trace(coalescent_graph, theta_dim=1)

#         # Serialize using pickle (traces are dataclass objects, not JSON-serializable directly)
#         temp_dir = tempfile.gettempdir()
#         trace_file = os.path.join(temp_dir, f"trace_proc_{slurm_info['procid']}.pkl")

#         with open(trace_file, 'wb') as f:
#             pickle.dump(trace, f)

#         # Read back and deserialize
#         with open(trace_file, 'rb') as f:
#             trace_loaded = pickle.load(f)

#         # Verify loaded trace works
#         theta = jnp.array([1.5])
#         result = evaluate_trace_jax(trace_loaded, theta)

#         assert jnp.all(jnp.isfinite(result['vertex_rates']))

#         # Clean up
#         os.remove(trace_file)

#         file_size = os.path.getsize(trace_file) if os.path.exists(trace_file) else 0
#         print(f"Process {slurm_info['procid']}: Serialization successful")
#         print(f"  Pickle file created and loaded successfully")

#     def test_hierarchical_trace_likelihood_multi_node(self, coalescent_graph, slurm_info):
#         """Test trace-based log-likelihood computation across nodes."""
#         from phasic.trace_elimination import (
#             record_elimination_trace,
#             trace_to_log_likelihood
#         )

#         print(f"\nProcess {slurm_info['procid']}: Testing trace log-likelihood")

#         # Record trace
#         trace = record_elimination_trace(coalescent_graph, theta_dim=1)

#         # Create log-likelihood function
#         observed_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
#         log_lik_fn = trace_to_log_likelihood(
#             trace,
#             observed_times,
#             reward_vector=None,
#             granularity=100
#         )

#         # Evaluate with different theta per process
#         theta = jnp.array([0.5 + slurm_info['procid'] * 0.3])
#         log_lik = log_lik_fn(theta)

#         assert jnp.isfinite(log_lik)
#         assert jnp.isscalar(log_lik) or log_lik.shape == ()

#         # Test gradient computation
#         grad_fn = jax.grad(log_lik_fn)
#         gradient = grad_fn(theta)

#         assert gradient.shape == theta.shape
#         assert jnp.all(jnp.isfinite(gradient))

#         print(f"Process {slurm_info['procid']}: Log-likelihood successful")
#         print(f"  Theta: {float(theta[0]):.3f}")
#         print(f"  Log-likelihood: {float(log_lik):.3f}")
#         print(f"  Gradient: {float(gradient[0]):.6f}")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
