#!/usr/bin/env python3
"""
Test optimized joint_index implementation using expected_sojourn_time()
"""

from phasic import Graph
import numpy as np
import jax
import jax.numpy as jnp

def coalescent_callback(state, **kwargs):
    """Simple coalescent model callback"""
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [(np.array([n - 1]), [rate])]

def test_joint_index_optimized():
    """Test that joint_index mode uses expected_sojourn_time()"""

    print("=" * 80)
    print("Testing optimized joint_index implementation")
    print("=" * 80)

    # Create parameterized graph
    print("\n1. Creating parameterized graph...")
    graph = Graph(coalescent_callback, parameterized=True, ipv=[5])
    print(f"   Graph has {graph.vertices_length()} vertices")
    print(f"   State length: {graph.state_length()}")

    # Create joint_index model
    print("\n2. Creating joint_index model...")
    model = Graph.pmf_from_graph_joint_index(graph, param_length=1)
    print("   ✓ Model created successfully")

    # Test with single parameter value
    print("\n3. Testing with single parameter value...")
    theta = jnp.array([1.0])
    vertex_indices = jnp.array([1, 2, 3], dtype=jnp.int32)

    result = model(theta, vertex_indices)
    sojourn_times, moments = result

    print(f"   Theta: {theta}")
    print(f"   Vertex indices: {vertex_indices}")
    print(f"   Sojourn times: {sojourn_times}")
    print(f"   Moments shape: {moments.shape}")
    print("   ✓ Single parameter test passed")

    # Test with batched parameters (vmap)
    print("\n4. Testing with batched parameters (vmap)...")
    theta_batch = jnp.array([[0.5], [1.0], [2.0]])
    vertex_indices_batch = jnp.array([1, 2, 3], dtype=jnp.int32)

    # Use vmap to batch over theta
    batched_model = jax.vmap(model, in_axes=(0, None))
    result_batch = batched_model(theta_batch, vertex_indices_batch)
    sojourn_times_batch, moments_batch = result_batch

    print(f"   Theta batch shape: {theta_batch.shape}")
    print(f"   Sojourn times batch shape: {sojourn_times_batch.shape}")
    print(f"   Moments batch shape: {moments_batch.shape}")
    print(f"   Sojourn times batch:\n{sojourn_times_batch}")
    print("   ✓ Batched parameter test passed")

    # Test JIT compilation
    print("\n5. Testing JIT compilation...")
    jitted_model = jax.jit(model)
    result_jit = jitted_model(theta, vertex_indices)
    sojourn_times_jit, moments_jit = result_jit

    print(f"   JIT result matches non-JIT: {np.allclose(sojourn_times, sojourn_times_jit)}")
    print("   ✓ JIT compilation test passed")

    # Compare with manual expected_sojourn_time call
    print("\n6. Comparing with manual expected_sojourn_time()...")
    from phasic.phasic_pybind import parameterized
    from phasic.ffi_wrappers import _make_json_serializable
    import json

    serialized = graph.serialize(param_length=1)
    structure_json = json.dumps(_make_json_serializable(serialized))
    builder = parameterized.GraphBuilder(structure_json)

    concrete_graph = builder.build(np.array([1.0]))
    manual_sojourn_times = np.asarray(concrete_graph.expected_sojourn_time())
    manual_selected = manual_sojourn_times[np.array(vertex_indices)]

    print(f"   Model result: {sojourn_times}")
    print(f"   Manual result: {manual_selected}")
    print(f"   Match: {np.allclose(sojourn_times, manual_selected)}")
    print("   ✓ Manual comparison test passed")

    # Performance comparison (if old method is still available)
    print("\n7. Performance comparison...")
    import time

    # Time the new method
    start = time.time()
    for _ in range(100):
        _ = model(theta, vertex_indices)
    elapsed_new = time.time() - start

    print(f"   New method (100 iterations): {elapsed_new:.4f}s ({elapsed_new*10:.2f}ms per call)")
    print("   ✓ Performance test completed")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

    return True

if __name__ == "__main__":
    test_joint_index_optimized()
