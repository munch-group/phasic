"""
Test FFI handler for multivariate phase-type distributions.

Comprehensive correctness tests including:
- FFI vs pybind11 comparison (both should give identical results)
- Length-1 vectors (must match existing compute_pmf_ffi)
- Sparse observations (zeros produce zero PDF)
- Multiple features (independent computation)
- vmap batching (parallel parameter sweeps)
"""

import numpy as np
import pytest

# IMPORTANT: phasic must be imported BEFORE JAX
import phasic
from phasic import Graph

# Now import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None
    print("JAX not available - skipping JAX-dependent tests")

# Check FFI availability
HAS_FFI = False
try:
    # Try to import FFI wrappers
    from phasic.ffi_wrappers import compute_pmf_ffi, compute_pmf_multivariate_ffi
    # FFI should be enabled by default if available
    HAS_FFI = True
    print("FFI is available")
except Exception as e:
    print(f"FFI not available: {e}")
    HAS_FFI = False


def create_test_graph():
    """Create a simple parameterized graph for testing."""
    def callback(state, **kwargs):
        n = state[0]
        if n <= 0:
            return []
        rate = n
        return [(np.array([n - 1]), [rate])]

    graph = Graph(callback, ipv=[3])
    return graph


def test_ffi_vs_pybind_length1():
    """Test FFI multivariate vs pybind11 with length-1 vectors."""
    print("\n" + "=" * 70)
    print("TEST: FFI vs pybind11 comparison (length-1 vectors)")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()

    # Test parameters
    theta = jnp.array([2.0])
    times_1d = jnp.array([0.5, 1.0, 1.5, 2.0])
    times_2d = times_1d.reshape(-1, 1)  # Shape: (4, 1)

    # Rewards (all ones for no transformation)
    n_vertices = graph.vertices_length()
    rewards_2d = jnp.ones((n_vertices, 1))

    # 1. Compute using pybind11 (via GraphBuilder)
    import json
    structure_json = json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                  for k, v in structure_dict.items()})
    builder = phasic.parameterized.GraphBuilder(structure_json)
    pdf_pybind = builder.compute_pmf_multivariate(
        np.asarray(theta), np.asarray(times_2d), np.asarray(rewards_2d),
        discrete=False, granularity=100, compute_joint=False
    )

    # 2. Compute using FFI
    pdf_ffi = compute_pmf_multivariate_ffi(
        structure_dict, theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    # 3. Compare
    pdf_pybind_1d = pdf_pybind[:, 0]
    pdf_ffi_1d = np.asarray(pdf_ffi[:, 0])

    print(f"\npybind11 PDF values: {pdf_pybind_1d}")
    print(f"FFI PDF values:      {pdf_ffi_1d}")
    print(f"Difference:          {pdf_pybind_1d - pdf_ffi_1d}")

    max_diff = np.max(np.abs(pdf_pybind_1d - pdf_ffi_1d))
    print(f"\nMax absolute difference: {max_diff:.2e}")

    assert max_diff < 1e-12, f"FFI and pybind11 results differ: {max_diff}"
    print("✓ PASS: FFI matches pybind11")


def test_ffi_length1_vs_existing_ffi():
    """Test multivariate FFI with length-1 vs existing compute_pmf_ffi."""
    print("\n" + "=" * 70)
    print("TEST: Multivariate FFI (length-1) vs existing compute_pmf_ffi")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()

    theta = jnp.array([2.0])
    times_1d = jnp.array([0.5, 1.0, 1.5, 2.0])
    times_2d = times_1d.reshape(-1, 1)

    n_vertices = graph.vertices_length()
    rewards_2d = jnp.ones((n_vertices, 1))

    # 1. Existing compute_pmf_ffi (1D)
    pdf_existing = compute_pmf_ffi(
        structure_dict, theta, times_1d,
        discrete=False, granularity=100
    )

    # 2. New multivariate FFI (2D with length-1)
    pdf_multivariate = compute_pmf_multivariate_ffi(
        structure_dict, theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    # 3. Compare
    pdf_multivariate_1d = np.asarray(pdf_multivariate[:, 0])
    pdf_existing_np = np.asarray(pdf_existing)

    print(f"\nExisting FFI PDF:    {pdf_existing_np}")
    print(f"Multivariate FFI:    {pdf_multivariate_1d}")
    print(f"Difference:          {pdf_existing_np - pdf_multivariate_1d}")

    max_diff = np.max(np.abs(pdf_existing_np - pdf_multivariate_1d))
    print(f"\nMax absolute difference: {max_diff:.2e}")

    assert max_diff < 1e-12, f"Multivariate FFI differs from existing FFI: {max_diff}"
    print("✓ PASS: Multivariate FFI matches existing compute_pmf_ffi")


def test_sparse_observations():
    """Test that zero observations produce zero PDF."""
    print("\n" + "=" * 70)
    print("TEST: Sparse observations (zeros produce zero PDF)")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()

    theta = jnp.array([2.0])

    # Mix of zero and non-zero observations
    times_2d = jnp.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [1.5, 0.5],
        [0.0, 0.0]
    ])

    n_vertices = graph.vertices_length()
    rewards_2d = jnp.ones((n_vertices, 2))

    pdf = compute_pmf_multivariate_ffi(
        structure_dict, theta, times_2d, rewards_2d,
        discrete=False, granularity=100, compute_joint=False
    )

    pdf_np = np.asarray(pdf)

    print(f"\nTimes:\n{times_2d}")
    print(f"\nPDF values:\n{pdf_np}")

    # Check zeros
    print("\nChecking zero entries:")
    assert pdf_np[0, 1] == 0.0, "times[0,1]=0 should give PDF=0"
    print(f"  PDF[0,1] = {pdf_np[0, 1]} ✓ (times[0,1] = 0.0)")

    assert pdf_np[1, 0] == 0.0, "times[1,0]=0 should give PDF=0"
    print(f"  PDF[1,0] = {pdf_np[1, 0]} ✓ (times[1,0] = 0.0)")

    assert pdf_np[3, 0] == 0.0, "times[3,0]=0 should give PDF=0"
    print(f"  PDF[3,0] = {pdf_np[3, 0]} ✓ (times[3,0] = 0.0)")

    assert pdf_np[3, 1] == 0.0, "times[3,1]=0 should give PDF=0"
    print(f"  PDF[3,1] = {pdf_np[3, 1]} ✓ (times[3,1] = 0.0)")

    # Check non-zeros
    print("\nChecking non-zero entries:")
    assert pdf_np[0, 0] > 0.0, "times[0,0]=1.0 should give PDF>0"
    print(f"  PDF[0,0] = {pdf_np[0, 0]:.6f} ✓ (times[0,0] = 1.0)")

    assert pdf_np[1, 1] > 0.0, "times[1,1]=2.0 should give PDF>0"
    print(f"  PDF[1,1] = {pdf_np[1, 1]:.6f} ✓ (times[1,1] = 2.0)")

    assert pdf_np[2, 0] > 0.0, "times[2,0]=1.5 should give PDF>0"
    print(f"  PDF[2,0] = {pdf_np[2, 0]:.6f} ✓ (times[2,0] = 1.5)")

    assert pdf_np[2, 1] > 0.0, "times[2,1]=0.5 should give PDF>0"
    print(f"  PDF[2,1] = {pdf_np[2, 1]:.6f} ✓ (times[2,1] = 0.5)")

    print("\n✓ PASS: Sparse observations work correctly")


def test_independent_features():
    """Test that features are computed independently with different rewards."""
    print("\n" + "=" * 70)
    print("TEST: Independent feature computation with different rewards")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()

    theta = jnp.array([2.0])
    times = jnp.array([[1.0], [2.0]])

    n_vertices = graph.vertices_length()

    # Two different reward vectors
    rewards_1 = jnp.ones((n_vertices, 1))
    rewards_2 = jnp.ones((n_vertices, 1)) * 2.0

    # Compute separately
    pdf_1 = compute_pmf_multivariate_ffi(
        structure_dict, theta, times, rewards_1,
        discrete=False, granularity=100, compute_joint=False
    )

    pdf_2 = compute_pmf_multivariate_ffi(
        structure_dict, theta, times, rewards_2,
        discrete=False, granularity=100, compute_joint=False
    )

    # Compute together (2 features)
    times_2feat = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    rewards_2feat = jnp.concatenate([rewards_1, rewards_2], axis=1)

    pdf_combined = compute_pmf_multivariate_ffi(
        structure_dict, theta, times_2feat, rewards_2feat,
        discrete=False, granularity=100, compute_joint=False
    )

    # Check that combined result matches separate computations
    pdf_1_np = np.asarray(pdf_1[:, 0])
    pdf_2_np = np.asarray(pdf_2[:, 0])
    pdf_combined_np = np.asarray(pdf_combined)

    print(f"\nFeature 0 (rewards=1.0):")
    print(f"  Separate:  {pdf_1_np}")
    print(f"  Combined:  {pdf_combined_np[:, 0]}")
    print(f"  Diff:      {pdf_1_np - pdf_combined_np[:, 0]}")

    print(f"\nFeature 1 (rewards=2.0):")
    print(f"  Separate:  {pdf_2_np}")
    print(f"  Combined:  {pdf_combined_np[:, 1]}")
    print(f"  Diff:      {pdf_2_np - pdf_combined_np[:, 1]}")

    max_diff_0 = np.max(np.abs(pdf_1_np - pdf_combined_np[:, 0]))
    max_diff_1 = np.max(np.abs(pdf_2_np - pdf_combined_np[:, 1]))

    print(f"\nMax difference feature 0: {max_diff_0:.2e}")
    print(f"Max difference feature 1: {max_diff_1:.2e}")

    assert max_diff_0 < 1e-12, f"Feature 0 differs: {max_diff_0}"
    assert max_diff_1 < 1e-12, f"Feature 1 differs: {max_diff_1}"

    # Also check they're different (rewards are different!)
    assert not np.allclose(pdf_1_np, pdf_2_np), "Different rewards should give different PDFs"
    print(f"\n✓ Different rewards produce different PDFs (as expected)")
    print("✓ PASS: Features computed independently")


def test_vmap_batching():
    """Test vmap batching over multiple theta values."""
    print("\n" + "=" * 70)
    print("TEST: vmap batching over parameter values")
    print("=" * 70)

    graph = create_test_graph()
    structure_dict = graph.serialize()

    # Multiple theta values to test
    theta_values = jnp.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0]
    ])  # Shape: (4, 1)

    times = jnp.array([[1.0, 2.0]])  # Single observation, 2 features
    n_vertices = graph.vertices_length()
    rewards = jnp.ones((n_vertices, 2))

    # Create batched function
    batched_fn = vmap(
        lambda theta: compute_pmf_multivariate_ffi(
            structure_dict, theta, times, rewards,
            discrete=False, granularity=100, compute_joint=False
        )
    )

    # Compute all at once
    pdf_batch = batched_fn(theta_values)

    print(f"\nBatch shape: {pdf_batch.shape}")
    print(f"Expected shape: (4, 1, 2)")  # (n_theta, n_times, n_features)

    assert pdf_batch.shape == (4, 1, 2), f"Unexpected shape: {pdf_batch.shape}"

    # Verify by computing individually
    print("\nComparing batched vs individual computation:")
    for i, theta in enumerate(theta_values):
        pdf_individual = compute_pmf_multivariate_ffi(
            structure_dict, theta, times, rewards,
            discrete=False, granularity=100, compute_joint=False
        )

        pdf_batch_i = pdf_batch[i]
        diff = np.max(np.abs(np.asarray(pdf_individual) - np.asarray(pdf_batch_i)))

        print(f"  theta={theta[0]:.1f}: batch={pdf_batch_i[0]}, "
              f"individual={np.asarray(pdf_individual)[0]}, diff={diff:.2e}")

        assert diff < 1e-12, f"Batch result differs from individual for theta={theta}"

    print("\n✓ PASS: vmap batching works correctly")


def test_correctness_analytical():
    """Test correctness against analytical solution for exponential distribution."""
    print("\n" + "=" * 70)
    print("TEST: Correctness against analytical exponential PDF")
    print("=" * 70)

    # Create exponential(rate) graph with single state
    def exp_callback(state, **kwargs):
        # Single transient state with rate parameter
        if state[0] == 1:
            return [(np.array([0]), [1.0])]  # Coefficient = 1.0
        return []

    graph = Graph(exp_callback, ipv=[1])
    structure_dict = graph.serialize()

    # Test with different rates (avoid very high rates due to discretization error)
    rates = jnp.array([0.5, 1.0, 2.0, 3.0])
    times = jnp.array([[0.5], [1.0], [2.0]])

    n_vertices = graph.vertices_length()
    rewards = jnp.ones((n_vertices, 1))

    print(f"\nGraph has {n_vertices} vertices")

    for rate in rates:
        theta = jnp.array([rate])

        # Compute using FFI (use auto granularity for best accuracy)
        pdf_ffi = compute_pmf_multivariate_ffi(
            structure_dict, theta, times, rewards,
            discrete=False, granularity=0, compute_joint=False
        )

        # Analytical exponential PDF: f(t) = rate * exp(-rate * t)
        times_np = np.asarray(times[:, 0])
        pdf_analytical = rate * np.exp(-rate * times_np)
        pdf_ffi_np = np.asarray(pdf_ffi[:, 0])

        diff = np.abs(pdf_ffi_np - pdf_analytical)
        rel_error = diff / pdf_analytical

        print(f"\nRate = {rate}:")
        print(f"  Times:       {times_np}")
        print(f"  FFI PDF:     {pdf_ffi_np}")
        print(f"  Analytical:  {pdf_analytical}")
        print(f"  Abs error:   {diff}")
        print(f"  Rel error:   {rel_error}")

        # Allow small relative error due to discretization
        max_rel_error = np.max(rel_error)
        assert max_rel_error < 0.01, f"Relative error too large: {max_rel_error}"
        print(f"  ✓ Max relative error: {max_rel_error:.2e} < 1%")

    print("\n✓ PASS: Results match analytical exponential distribution")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MULTIVARIATE FFI CORRECTNESS TESTS")
    print("=" * 70)

    test_ffi_vs_pybind_length1()
    test_ffi_length1_vs_existing_ffi()
    test_sparse_observations()
    test_independent_features()
    test_vmap_batching()
    test_correctness_analytical()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
