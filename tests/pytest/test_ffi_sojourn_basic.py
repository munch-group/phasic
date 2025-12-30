#!/usr/bin/env python
"""
Basic test for compute_sojourn_times_ffi functionality.
Tests FFI registration, basic computation, and vmap batching.
"""

import numpy as np
from phasic import Graph
from phasic.ffi_wrappers import compute_sojourn_times_ffi
import jax
import jax.numpy as jnp


def test_basic_sojourn_ffi():
    """Test basic FFI sojourn time computation."""
    print("=" * 70)
    print("TEST 1: Basic sojourn time FFI computation")
    print("=" * 70)

    # Build simple coalescent graph
    def coalescent_callback(state, nr_samples=None, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [([n - 1], [rate])]  # Parameterized: (state, coeffs_list)

    graph = Graph(coalescent_callback, ipv=[4], parameterized=True)  # Small graph: 4 vertices

    print(f"✓ Created graph with {graph.vertices_length()} vertices")

    # Serialize graph
    structure_json = graph.serialize()

    # Parameters: just theta=1.0 (rate multiplier)
    theta = jnp.array([1.0])

    # Compute for specific vertices
    indices = jnp.array([0, 1, 2], dtype=jnp.int32)

    # Call FFI function
    sojourn_ffi = compute_sojourn_times_ffi(structure_json, theta, indices)

    print(f"✓ FFI sojourn times: {sojourn_ffi}")
    print(f"  Shape: {sojourn_ffi.shape}")
    print(f"  Dtype: {sojourn_ffi.dtype}")

    # Compare with direct C++ call via GraphBuilder
    import json
    theta_np = np.array([1.0])
    structure_json_str = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                      for k, v in structure_json.items()})
    import phasic
    builder = phasic.parameterized.GraphBuilder(structure_json_str)
    concrete_graph = builder.build(theta_np)
    sojourn_direct = concrete_graph.expected_sojourn_time(indices.tolist())

    print(f"✓ Direct C++ sojourn times: {sojourn_direct}")

    # Check match
    diff = np.abs(np.array(sojourn_ffi) - np.array(sojourn_direct))
    max_diff = np.max(diff)
    print(f"✓ Max difference: {max_diff:.2e}")

    assert max_diff < 1e-10, f"FFI mismatch: max diff = {max_diff}"
    print("✓ FFI matches direct computation\n")


def test_vmap_batching():
    """Test vmap batching with FFI."""
    print("=" * 70)
    print("TEST 2: vmap batching")
    print("=" * 70)

    # Build simple graph
    def coalescent_callback(state, nr_samples=None, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [([n - 1], [rate])]

    graph = Graph(coalescent_callback, ipv=[4], parameterized=True)

    structure_json = graph.serialize()

    # Batch of theta values
    theta_batch = jnp.array([
        [1.0],
        [2.0],
        [0.5]
    ])  # (3, 1)

    # Singleton indices (broadcast to all batches)
    indices = jnp.array([0, 1], dtype=jnp.int32)

    # vmap over theta batch
    batched_fn = jax.vmap(
        lambda t: compute_sojourn_times_ffi(structure_json, t, indices)
    )

    sojourn_batch = batched_fn(theta_batch)

    print(f"✓ Batched sojourn times shape: {sojourn_batch.shape}")
    print(f"  Expected: (3, 2) - 3 batches × 2 indices")
    print(f"✓ Values:\n{sojourn_batch}")

    assert sojourn_batch.shape == (3, 2), f"Wrong shape: {sojourn_batch.shape}"

    # Verify each batch individually
    for i in range(3):
        theta_i = theta_batch[i]
        sojourn_i = compute_sojourn_times_ffi(structure_json, theta_i, indices)
        diff = np.abs(sojourn_batch[i] - sojourn_i)
        max_diff = np.max(diff)
        assert max_diff < 1e-10, f"Batch {i} mismatch: {max_diff}"

    print("✓ All batches match individual computations\n")


def test_jit_compilation():
    """Test JIT compilation."""
    print("=" * 70)
    print("TEST 3: JIT compilation")
    print("=" * 70)

    # Build simple graph
    def coalescent_callback(state, nr_samples=None, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        return [([n - 1], [rate])]

    graph = Graph(coalescent_callback, ipv=[4], parameterized=True)

    # Serialize to dict, then convert to JSON string for JIT (must be hashable)
    import json
    structure_dict = graph.serialize()
    structure_json = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                  for k, v in structure_dict.items()})

    # JIT compile
    jit_fn = jax.jit(
        compute_sojourn_times_ffi,
        static_argnums=(0,)  # structure_json is static
    )

    theta = jnp.array([1.0])
    indices = jnp.array([0, 1, 2], dtype=jnp.int32)

    # First call (compile)
    sojourn_jit = jit_fn(structure_json, theta, indices)
    print(f"✓ JIT compiled result: {sojourn_jit}")

    # Second call (cached)
    sojourn_jit2 = jit_fn(structure_json, theta, indices)

    # Compare with non-JIT
    sojourn_nojit = compute_sojourn_times_ffi(structure_json, theta, indices)

    diff = np.abs(sojourn_jit - sojourn_nojit)
    max_diff = np.max(diff)
    print(f"✓ JIT vs non-JIT diff: {max_diff:.2e}")

    assert max_diff < 1e-10, f"JIT mismatch: {max_diff}"
    print("✓ JIT compilation works correctly\n")


if __name__ == "__main__":
    test_basic_sojourn_ffi()
    test_vmap_batching()
    test_jit_compilation()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
