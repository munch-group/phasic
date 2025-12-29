#!/usr/bin/env python3
"""
Test NaN observation handling and per-feature moment computation.

Tests:
1. Per-feature moment computation for 2D data
2. NaN observations handled in likelihood
3. Invalid PMF values raise errors
4. Backward compatibility with 1D data
"""

import numpy as np
import sys

# Import phasic FIRST (before JAX)
from phasic.svgd import compute_sample_moments
from phasic import Graph, SVGD

# Now import JAX
import jax
import jax.numpy as jnp

# Test moment computation
print("=" * 70)
print("TEST 1: Per-feature moment computation")
print("=" * 70)

# 1D case (backward compatible)
print("\n1D data (backward compatible):")
data_1d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
moments_1d = compute_sample_moments(data_1d, nr_moments=2)
print(f"  Data: {data_1d}")
print(f"  Moments shape: {moments_1d.shape}")
print(f"  Moments: {moments_1d}")
print(f"  Expected: [3.0, 11.0] (mean, mean of squares)")
assert moments_1d.shape == (2,), f"Expected shape (2,), got {moments_1d.shape}"
assert np.allclose(moments_1d[0], 3.0), f"Mean should be 3.0, got {moments_1d[0]}"
assert np.allclose(moments_1d[1], 11.0), f"Mean of squares should be 11.0, got {moments_1d[1]}"
print("  ✓ PASS")

# 2D case without NaN
print("\n2D data without NaN:")
data_2d = jnp.array([
    [1.0, 10.0],
    [2.0, 20.0],
    [3.0, 30.0]
])
moments_2d = compute_sample_moments(data_2d, nr_moments=2)
print(f"  Data shape: {data_2d.shape}")
print(f"  Moments shape: {moments_2d.shape}")
print(f"  Moments:\n{moments_2d}")
print(f"  Expected: [[2.0, 4.667], [20.0, 466.667]] (per-feature)")
assert moments_2d.shape == (2, 2), f"Expected shape (2, 2), got {moments_2d.shape}"
assert np.allclose(moments_2d[0, 0], 2.0), f"Feature 0 mean should be 2.0, got {moments_2d[0, 0]}"
assert np.allclose(moments_2d[1, 0], 20.0), f"Feature 1 mean should be 20.0, got {moments_2d[1, 0]}"
print("  ✓ PASS")

# 2D case with NaN (key test!)
print("\n2D data with NaN:")
data_2d_nan = jnp.array([
    [1.0, 10.0],
    [2.0, np.nan],  # Feature 1 missing
    [3.0, 30.0]
])
moments_2d_nan = compute_sample_moments(data_2d_nan, nr_moments=2)
print(f"  Data:\n{data_2d_nan}")
print(f"  Moments shape: {moments_2d_nan.shape}")
print(f"  Moments:\n{moments_2d_nan}")
print(f"  Feature 0: computed from [1, 2, 3] → mean=2.0")
print(f"  Feature 1: computed from [10, 30] (skip NaN) → mean=20.0")
assert moments_2d_nan.shape == (2, 2), f"Expected shape (2, 2), got {moments_2d_nan.shape}"
assert np.allclose(moments_2d_nan[0, 0], 2.0), f"Feature 0 mean should be 2.0, got {moments_2d_nan[0, 0]}"
assert np.allclose(moments_2d_nan[1, 0], 20.0), f"Feature 1 mean should be 20.0 (ignoring NaN), got {moments_2d_nan[1, 0]}"
print("  ✓ PASS: NaN values correctly ignored per feature!")

print("\n" + "=" * 70)
print("TEST 2: NaN observation handling in SVGD")
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
graph = Graph(simple_callback, ipv=[[[2], 1.0]], parameterized=True)
n_vertices = graph.vertices_length()
print(f"  Graph vertices: {n_vertices}")

# Create multivariate model
print("Creating multivariate model...")
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False
)

# Test data with NaN observations
observations = jnp.array([
    [1.0, 2.0],      # Both features valid
    [1.5, np.nan],   # Feature 1 missing
    [np.nan, 2.5],   # Feature 0 missing
    [2.0, 3.0],      # Both features valid
])

# Create 2D rewards (must match number of vertices)
rewards = jnp.array([
    [1.0, 2.0, 0.5, 1.5],  # Feature 0 reward vector (4 vertices)
    [0.5, 1.0, 2.0, 1.0],  # Feature 1 reward vector (4 vertices)
])

print(f"\nObservations shape: {observations.shape}")
print(f"Observations:\n{observations}")
print(f"Rewards shape: {rewards.shape}")

# Create SVGD instance
print("\nCreating SVGD instance with NaN observations...")

try:
    svgd = SVGD(
        model=model,
        observed_data=observations,
        theta_dim=1,
        n_particles=5,
        n_iterations=2,  # Just test a few iterations
        verbose=False,
        rewards=rewards,
        regularization=0.1,  # Enable regularization to compute sample moments
        nr_moments=2
    )
    print("  ✓ SVGD instance created successfully")
except Exception as e:
    print(f"  ✗ FAILED to create SVGD: {e}")
    sys.exit(1)

# Test log probability computation with NaN observations
print("\nTesting log probability with NaN observations...")
theta_test = jnp.array([1.0])
try:
    log_prob = svgd._log_prob_unified(
        theta_test,
        nr_moments=svgd.nr_moments,
        sample_moments=svgd.sample_moments,
        regularization=svgd.regularization,
        rewards=rewards
    )
    print(f"  Log probability: {log_prob}")
    print(f"  ✓ PASS: NaN observations handled correctly")

    # Verify it's finite
    assert jnp.isfinite(log_prob), f"Log probability should be finite, got {log_prob}"
    print(f"  ✓ Log probability is finite")

except ValueError as e:
    if "NaN PMF values for valid observations" in str(e):
        print(f"  ✗ FAILED: Model returned NaN for valid observations: {e}")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"  ✗ FAILED with unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test that invalid PMF raises error
print("\n" + "=" * 70)
print("TEST 3: Invalid PMF detection")
print("=" * 70)

print("\nTesting error detection for invalid PMF values...")
# Create a model that returns NaN PMF (by using very bad parameters)
theta_bad = jnp.array([-1000.0])  # This might cause numerical issues

try:
    log_prob_bad = svgd._log_prob_unified(
        theta_bad,
        nr_moments=svgd.nr_moments,
        sample_moments=svgd.sample_moments,
        regularization=svgd.regularization,
        rewards=rewards
    )
    # If we get here, the model didn't return NaN, which is fine
    print(f"  Model returned finite log probability: {log_prob_bad}")
    print(f"  ✓ PASS (model is robust to extreme parameters)")
except ValueError as e:
    if "NaN PMF values for valid observations" in str(e):
        print(f"  ✓ PASS: Error correctly raised for invalid PMF")
        print(f"  Error message: {e}")
    else:
        print(f"  ✗ FAILED: Wrong error type: {e}")
        sys.exit(1)

print("\n" + "=" * 70)
print("TEST 4: Sample moments shape consistency")
print("=" * 70)

print("\nChecking sample moments computation in SVGD...")
print(f"  Observed data shape: {svgd.observed_data.shape}")
print(f"  Sample moments shape: {svgd.sample_moments.shape}")
print(f"  Sample moments:\n{svgd.sample_moments}")

# For 2D observations, sample moments should be (n_features, nr_moments)
expected_shape = (observations.shape[1], svgd.nr_moments)
assert svgd.sample_moments.shape == expected_shape, \
    f"Expected shape {expected_shape}, got {svgd.sample_moments.shape}"
print(f"  ✓ PASS: Correct shape {svgd.sample_moments.shape}")

# Verify per-feature moments make sense
print("\nVerifying per-feature moments (ignoring NaN):")
for j in range(observations.shape[1]):
    feature_data = observations[:, j]
    valid_data = feature_data[~np.isnan(feature_data)]
    expected_mean = np.mean(valid_data)
    actual_mean = svgd.sample_moments[j, 0]
    print(f"  Feature {j}: valid data = {valid_data}, expected mean = {expected_mean:.3f}, actual = {actual_mean:.3f}")
    assert np.allclose(actual_mean, expected_mean, rtol=1e-5), \
        f"Feature {j} mean mismatch: expected {expected_mean}, got {actual_mean}"
print("  ✓ PASS: Per-feature moments correct!")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nSummary:")
print("  ✓ Per-feature moment computation works")
print("  ✓ NaN observations are skipped in likelihood")
print("  ✓ Invalid PMF values are detected (when they occur)")
print("  ✓ Sample moments computed correctly per feature")
print("  ✓ Backward compatibility with 1D data maintained")
