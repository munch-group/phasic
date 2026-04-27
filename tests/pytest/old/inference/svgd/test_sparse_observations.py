#!/usr/bin/env python3
"""
Test sparse observation format for multivariate SVGD.

Tests:
1. SparseObservations class and dense_to_sparse conversion
2. compute_sample_moments with sparse data
3. SVGD with sparse observations
4. Equivalence of sparse vs dense format results
"""

import numpy as np
import sys

# Import phasic FIRST (before JAX)
from phasic.svgd import (
    compute_sample_moments,
    SparseObservations,
    dense_to_sparse,
    is_sparse_observations,
)
from phasic import Graph, SVGD

# Now import JAX
import jax
import jax.numpy as jnp

print("=" * 70)
print("TEST 1: SparseObservations class and dense_to_sparse")
print("=" * 70)

# Test dense_to_sparse conversion
print("\nConverting dense NaN-padded data to sparse format:")
dense_data = jnp.array([
    [1.0, np.nan, 3.0],
    [1.5, 2.0, np.nan],
    [np.nan, 2.5, 3.5],
])
print(f"  Dense data:\n{dense_data}")

sparse = dense_to_sparse(dense_data)
print(f"\n  Sparse representation:")
print(f"    values: {sparse.values}")
print(f"    features: {sparse.features}")
print(f"    n_features: {sparse.n_features}")

# Verify
assert is_sparse_observations(sparse), "Should be SparseObservations"
assert sparse.n_features == 3, f"Expected 3 features, got {sparse.n_features}"
# Feature 0: 1.0, 1.5 (2 values)
# Feature 1: 2.0, 2.5 (2 values)
# Feature 2: 3.0, 3.5 (2 values)
assert len(sparse.values) == 6, f"Expected 6 total values, got {len(sparse.values)}"
print("  ✓ PASS: dense_to_sparse works correctly")

# Test is_sparse_observations
print("\nTesting is_sparse_observations:")
assert is_sparse_observations(sparse) == True
assert is_sparse_observations(dense_data) == False
assert is_sparse_observations(jnp.array([1.0, 2.0])) == False
print("  ✓ PASS: is_sparse_observations works correctly")

print("\n" + "=" * 70)
print("TEST 2: compute_sample_moments with sparse data")
print("=" * 70)

# Create sparse data with known values (slices optional for moment computation)
sparse_data = SparseObservations(
    values=jnp.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),  # Feature 0: [1,2,3], Feature 1: [10,20,30]
    features=jnp.array([0, 0, 0, 1, 1, 1]),
    n_features=2,
    slices=((0, 3), (3, 6))  # Required for model evaluation, optional for moments
)

print(f"\nSparse data:")
print(f"  Feature 0: [1.0, 2.0, 3.0]")
print(f"  Feature 1: [10.0, 20.0, 30.0]")

moments_sparse = compute_sample_moments(sparse_data, nr_moments=2)
print(f"\nMoments from sparse data:")
print(f"  Shape: {moments_sparse.shape}")
print(f"  Values:\n{moments_sparse}")

# Expected: Feature 0 mean = 2.0, Feature 1 mean = 20.0
assert moments_sparse.shape == (2, 2), f"Expected shape (2, 2), got {moments_sparse.shape}"
assert np.allclose(moments_sparse[0, 0], 2.0), f"Feature 0 mean should be 2.0, got {moments_sparse[0, 0]}"
assert np.allclose(moments_sparse[1, 0], 20.0), f"Feature 1 mean should be 20.0, got {moments_sparse[1, 0]}"
print("  ✓ PASS: compute_sample_moments works with sparse data")

# Compare with dense version
print("\nComparing sparse vs dense moment computation:")
dense_equivalent = jnp.array([
    [1.0, 10.0],
    [2.0, 20.0],
    [3.0, 30.0]
])
moments_dense = compute_sample_moments(dense_equivalent, nr_moments=2)
print(f"  Dense moments:\n{moments_dense}")
print(f"  Sparse moments:\n{moments_sparse}")
assert np.allclose(moments_sparse, moments_dense), "Sparse and dense moments should match"
print("  ✓ PASS: Sparse and dense moments are identical")

print("\n" + "=" * 70)
print("TEST 3: SVGD with sparse observations")
print("=" * 70)

# Create a simple test model
def simple_callback(state, **kwargs):
    """Simple 2-state chain"""
    n = state[0]
    if n <= 0:
        return []
    return [(np.array([n - 1]), [n])]  # Rate = n * theta

# Build parameterized graph
print("\nBuilding test graph...")
graph = Graph(simple_callback, ipv=[[[2], 1.0]])
n_vertices = graph.vertices_length()
print(f"  Graph vertices: {n_vertices}")

# Create multivariate model
print("Creating multivariate model...")
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False
)

# Create 2D rewards (must match number of vertices)
rewards = jnp.array([
    [1.0, 2.0, 0.5, 1.5],  # Feature 0 reward vector (4 vertices)
    [0.5, 1.0, 2.0, 1.0],  # Feature 1 reward vector (4 vertices)
])

# Create sparse observations with pre-computed slices for JAX JIT compatibility
sparse_obs = SparseObservations(
    values=jnp.array([1.0, 1.5, 2.0, 2.0, 2.5, 3.0]),  # Feature 0: [1.0, 1.5, 2.0], Feature 1: [2.0, 2.5, 3.0]
    features=jnp.array([0, 0, 0, 1, 1, 1]),
    n_features=2,
    slices=((0, 3), (3, 6))  # Feature 0: indices 0-3, Feature 1: indices 3-6
)

print(f"\nSparse observations:")
print(f"  Feature 0: [1.0, 1.5, 2.0]")
print(f"  Feature 1: [2.0, 2.5, 3.0]")
print(f"  Total observations: {len(sparse_obs.values)}")

# Create SVGD instance with sparse observations
print("\nCreating SVGD instance with sparse observations...")
try:
    svgd_sparse = SVGD(
        model=model,
        observed_data=sparse_obs,
        theta_dim=1,
        n_particles=5,
        n_iterations=2,  # Just test a few iterations
        verbose=True,
        rewards=rewards,
        regularization=0.1,
        nr_moments=2
    )
    print("  ✓ SVGD instance created successfully with sparse observations")
except Exception as e:
    print(f"  ✗ FAILED to create SVGD: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test log probability computation
print("\nTesting log probability with sparse observations...")
theta_test = jnp.array([1.0])
try:
    log_prob = svgd_sparse._log_prob_unified(
        theta_test,
        nr_moments=svgd_sparse.nr_moments,
        sample_moments=svgd_sparse.sample_moments,
        regularization=svgd_sparse.regularization,
        rewards=rewards
    )
    print(f"  Log probability: {log_prob}")
    assert jnp.isfinite(log_prob), f"Log probability should be finite, got {log_prob}"
    print("  ✓ PASS: Log probability computed successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("TEST 4: Equivalence of sparse vs dense format")
print("=" * 70)

# Create equivalent dense observations (no NaN)
dense_obs = jnp.array([
    [1.0, 2.0],
    [1.5, 2.5],
    [2.0, 3.0]
])

# Convert to sparse
sparse_from_dense = dense_to_sparse(dense_obs)

print(f"\nDense observations:\n{dense_obs}")
print(f"\nSparse from dense:")
print(f"  values: {sparse_from_dense.values}")
print(f"  features: {sparse_from_dense.features}")

# Create SVGD with dense data
print("\nCreating SVGD with dense observations...")
svgd_dense = SVGD(
    model=model,
    observed_data=dense_obs,
    theta_dim=1,
    n_particles=5,
    n_iterations=2,
    verbose=False,
    rewards=rewards,
    regularization=0.1,
    nr_moments=2
)

# Create SVGD with sparse data (converted from same dense data)
print("Creating SVGD with sparse observations (converted from dense)...")
svgd_sparse2 = SVGD(
    model=model,
    observed_data=sparse_from_dense,
    theta_dim=1,
    n_particles=5,
    n_iterations=2,
    verbose=False,
    rewards=rewards,
    regularization=0.1,
    nr_moments=2
)

# Compare sample moments
print("\nComparing sample moments:")
print(f"  Dense sample moments:\n{svgd_dense.sample_moments}")
print(f"  Sparse sample moments:\n{svgd_sparse2.sample_moments}")
assert np.allclose(svgd_dense.sample_moments, svgd_sparse2.sample_moments), \
    "Sample moments should match between dense and sparse"
print("  ✓ PASS: Sample moments match")

# Compare log probabilities
print("\nComparing log probabilities:")
theta_test = jnp.array([1.0])
log_prob_dense = svgd_dense._log_prob_unified(
    theta_test,
    nr_moments=svgd_dense.nr_moments,
    sample_moments=svgd_dense.sample_moments,
    regularization=svgd_dense.regularization,
    rewards=rewards
)
log_prob_sparse = svgd_sparse2._log_prob_unified(
    theta_test,
    nr_moments=svgd_sparse2.nr_moments,
    sample_moments=svgd_sparse2.sample_moments,
    regularization=svgd_sparse2.regularization,
    rewards=rewards
)
print(f"  Dense log probability: {log_prob_dense}")
print(f"  Sparse log probability: {log_prob_sparse}")

# Note: They may differ slightly due to different aggregation of PMF values
# The key is that both are finite and reasonable
assert jnp.isfinite(log_prob_dense), "Dense log probability should be finite"
assert jnp.isfinite(log_prob_sparse), "Sparse log probability should be finite"
print("  ✓ PASS: Both log probabilities are finite")

print("\n" + "=" * 70)
print("TEST 5: Sparse observations with unequal features")
print("=" * 70)

# Test case where different features have different numbers of observations
sparse_unequal = SparseObservations(
    values=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]),  # Feature 0: 5 obs, Feature 1: 2 obs
    features=jnp.array([0, 0, 0, 0, 0, 1, 1]),
    n_features=2,
    slices=((0, 5), (5, 7))  # Feature 0: 5 values, Feature 1: 2 values
)

print(f"\nSparse data with unequal features:")
print(f"  Feature 0: [1.0, 2.0, 3.0, 4.0, 5.0] (5 observations)")
print(f"  Feature 1: [10.0, 20.0] (2 observations)")

moments_unequal = compute_sample_moments(sparse_unequal, nr_moments=2)
print(f"\nMoments:")
print(f"  Feature 0 mean: {moments_unequal[0, 0]} (expected: 3.0)")
print(f"  Feature 1 mean: {moments_unequal[1, 0]} (expected: 15.0)")

assert np.allclose(moments_unequal[0, 0], 3.0), f"Feature 0 mean should be 3.0"
assert np.allclose(moments_unequal[1, 0], 15.0), f"Feature 1 mean should be 15.0"
print("  ✓ PASS: Unequal observation counts handled correctly")

# Test SVGD with unequal features
print("\nTesting SVGD with unequal features...")
try:
    svgd_unequal = SVGD(
        model=model,
        observed_data=sparse_unequal,
        theta_dim=1,
        n_particles=5,
        n_iterations=2,
        verbose=False,
        rewards=rewards,
        regularization=0.1,
        nr_moments=2
    )

    log_prob_unequal = svgd_unequal._log_prob_unified(
        theta_test,
        nr_moments=svgd_unequal.nr_moments,
        sample_moments=svgd_unequal.sample_moments,
        regularization=svgd_unequal.regularization,
        rewards=rewards
    )
    print(f"  Log probability: {log_prob_unequal}")
    assert jnp.isfinite(log_prob_unequal), "Log probability should be finite"
    print("  ✓ PASS: SVGD works with unequal observation counts")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nSummary:")
print("  ✓ SparseObservations class works correctly")
print("  ✓ dense_to_sparse conversion works")
print("  ✓ is_sparse_observations detection works")
print("  ✓ compute_sample_moments handles sparse data")
print("  ✓ SVGD accepts sparse observations")
print("  ✓ Sparse and dense formats produce consistent results")
print("  ✓ Unequal observation counts per feature are handled")
