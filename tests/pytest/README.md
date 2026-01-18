# Test Suite Directory

This directory contains comprehensive tests for the phasic library covering graph construction, trace-based computation, JAX integration, multivariate distributions, and SVGD inference.

## Test Files

### Core API & Construction
- [test_api_comprehensive.py](test_api_comprehensive.py) - Comprehensive API tests for Graph, Vertex, Edge classes (standalone)
- [test_comprehensive_api.py](test_comprehensive_api.py) - Core API functionality tests (pytest version)
- [test_graph_construction.py](test_graph_construction.py) - Graph construction methods
- [test_graph_serialization.py](test_graph_serialization.py) - Graph serialization/deserialization (from_serialized)
- [test_from_matrices.py](test_from_matrices.py) - Graph.from_matrices() method
- [test_as_matrices_fix.py](test_as_matrices_fix.py) - graph.as_matrices() segfault regression test
- [test_state_indexing.py](test_state_indexing.py) - Flexible state indexing system

### Edges & Weights
- [test_parameterized_edges.py](test_parameterized_edges.py) - Parameterized edges feature
- [test_unified_edge_correctness.py](test_unified_edge_correctness.py) - Unified edge interface correctness
- [test_callback_weights.py](test_callback_weights.py) - Callback-based update_weights() functionality
- [test_log_space_weights.py](test_log_space_weights.py) - Log-space weight computation
- [test_param_length_flexibility.py](test_param_length_flexibility.py) - Flexible param_length coefficient handling

### Trace-Based Computation
- [test_hierarchical_graph.py](test_hierarchical_graph.py) - Hierarchical trace-based computation
- [test_trace_both_versions.py](test_trace_both_versions.py) - Simple vs full trace versions (with/without rewards)
- [test_trace_simple_comparison.py](test_trace_simple_comparison.py) - Verify simple and full traces produce identical results
- [test_trace_rewards.py](test_trace_rewards.py) - Reward transformation in trace elimination
- [test_trace_rewards_simple.py](test_trace_rewards_simple.py) - Simple reward transformation tests
- [test_default_rewards.py](test_default_rewards.py) - Default rewards behavior
- [test_manual_vs_trace_graph.py](test_manual_vs_trace_graph.py) - Manual vs trace-instantiated graph comparison
- [test_trace_select_operation.py](test_trace_select_operation.py) - SELECT operation in trace system
- [test_trace_stitching.py](test_trace_stitching.py) - Trace stitching algorithm

### Caching
- [test_hierarchical_cache.py](test_hierarchical_cache.py) - Hierarchical caching (Phase 3a)
- [test_universal_caching.py](test_universal_caching.py) - Universal trace caching
- [test_trace_repository.py](test_trace_repository.py) - IPFS-based trace repository
- [test_repo.py](test_repo.py) - Trace repository functionality

### Multivariate Distributions
- [test_multivariate.py](test_multivariate.py) - 2D observations & 2D rewards
- [test_multivariate_correctness.py](test_multivariate_correctness.py) - Comprehensive multivariate correctness tests
- [test_multivariate_ffi.py](test_multivariate_ffi.py) - FFI handler for multivariate distributions
- [test_multivariate_length1.py](test_multivariate_length1.py) - Length-1 vectors in multivariate PMF
- [test_notebook_multivar_reproduction.py](test_notebook_multivar_reproduction.py) - Reproduce exact notebook multivariate SVGD
- [test_convergence_1d_vs_multivar.py](test_convergence_1d_vs_multivar.py) - Debug 1D vs multivariate convergence with NaNs

### NaN Handling
- [test_nan_handling.py](test_nan_handling.py) - NaN observations & per-feature moment computation
- [test_nan_observations.py](test_nan_observations.py) - Multivariate SVGD with sparse observation patterns
- [test_nan_observations_correctness.py](test_nan_observations_correctness.py) - SVGD correctness with NaN observations

### SVGD Inference
- [test_svgd_correctness.py](test_svgd_correctness.py) - SVGD inference correctness
- [test_svgd_jax.py](test_svgd_jax.py) - SVGD configuration options showcase (jit, parallel, devices)
- [test_simple_example.py](test_simple_example.py) - Simple SVGD example

### JAX Integration
- [test_jax_integration.py](test_jax_integration.py) - JAX integration test suite

### FFI & Performance
- [test_ffi_sojourn_basic.py](test_ffi_sojourn_basic.py) - compute_sojourn_times_ffi functionality
- [test_multi_process_ffi.py](test_multi_process_ffi.py) - Multi-process FFI verification
- [test_slurm_multinode_ffi.py](test_slurm_multinode_ffi.py) - SLURM multi-node FFI tests
- [test_joint_index_performance.py](test_joint_index_performance.py) - accumulated_visiting_time vs expected_sojourn_time performance
- [test_optimized_joint_index.py](test_optimized_joint_index.py) - Optimized joint_index using expected_sojourn_time()

### Rewards & Sampling
- [test_rewards_support.py](test_rewards_support.py) - Reward vector support in pmf_and_moments_from_graph()
- [test_sample_with_rewards.py](test_sample_with_rewards.py) - Sampling with rewards (notebook pattern)

### SCC (Strongly Connected Components)
- [test_scc_api.py](test_scc_api.py) - SCC API unit tests (Phase 1)

### Debug & Development
- [test_graph_edges_debug.py](test_graph_edges_debug.py) - Debug edges before trace recording
- [test_graph_instantiation_debug.py](test_graph_instantiation_debug.py) - Debug graph instantiation from trace
- [test_graph_structure_debug.py](test_graph_structure_debug.py) - Debug graph structure after trace instantiation
- [test_multivar_likelihood_debug.py](test_multivar_likelihood_debug.py) - Debug multivariate likelihood with NaN observations
- [test_graphbuilder_1d_correctness.py](test_graphbuilder_1d_correctness.py) - Phase 1: GraphBuilder 1D correctness

### Examples & Models
- [test_exp_geom.py](test_exp_geom.py) - Exponential/geometric distribution tests
- [test_dummy.py](test_dummy.py) - Basic assertion tests
- [multivar_test.py](multivar_test.py) - Multivariate test (dev file, commented out execution)
- [multivar_test_fixed.py](multivar_test_fixed.py) - Fixed multivariate test with correct array shapes
- [user_test.py](user_test.py) - User test file (dev/example)

### Utilities
- [test_utilities_integration.py](test_utilities_integration.py) - Utilities and integration features
- [run_all_tests.py](run_all_tests.py) - Master test runner

## Running Tests

```bash
# Run all tests
pixi run test

# Run specific test
pytest tests/pytest/test_api_comprehensive.py

# Run with verbose output
pytest -v tests/pytest/

# Run tests matching pattern
pytest -k "multivariate" tests/pytest/
```

## Coverage Areas

- **Graph Construction**: Manual building, callbacks, from_matrices
- **Trace System**: Recording, evaluation, caching, stitching, rewards
- **Multivariate**: 2D observations/rewards, NaN handling, FFI
- **SVGD**: Inference correctness, JAX integration, configuration
- **Performance**: FFI, multi-process, optimized algorithms
- **Edge Cases**: NaN handling, sparse observations, log-space weights
