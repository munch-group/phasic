#!/usr/bin/env python3
"""
Example: Sparse Observation Format for Multivariate SVGD

This example demonstrates the new SparseObservations format for multivariate
phase-type models where different features have different numbers of observations.

The sparse format avoids NaN propagation through JAX gradients that can occur
with dense NaN-padded arrays during SVGD optimization.
"""

import numpy as np

# Import phasic FIRST (before JAX) - this configures JAX for multi-CPU
from phasic import (
    Graph,
    SVGD,
    SparseObservations,
    dense_to_sparse,
    is_sparse_observations
)

# Now import JAX
import jax.numpy as jnp

# =============================================================================
# 1. Create a simple parameterized phase-type model
# =============================================================================

def simple_callback(state, **kwargs):
    """
    Simple 2-state chain: state 2 -> state 1 -> absorbing.
    Rate of transition is n * theta where n is the state.
    """
    n = state[0]
    if n <= 0:
        return []  # Absorbing state
    # Return: (next_state, [coefficient for theta])
    # Edge weight = coefficient * theta
    return [(np.array([n - 1]), [float(n)])]

# Build parameterized graph starting from state n=2
print("Building parameterized graph...")
graph = Graph(simple_callback, ipv=[[[2], 1.0]])
n_vertices = graph.vertices_length()
print(f"  Graph has {n_vertices} vertices")

# Create multivariate model (returns PMF and moments)
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph,
    nr_moments=2,
    discrete=False
)

# =============================================================================
# 2. Define reward vectors for multivariate distribution
# =============================================================================

# Each row is the reward vector for one feature (marginal distribution)
# Shape: (n_features, n_vertices)
rewards = jnp.array([
    [1.0, 0.5, 0.2, 0.1],  # Feature 0: weighted sum of sojourn times
    [0.0, 1.0, 1.0, 0.0],  # Feature 1: time in middle states only
])
n_features = rewards.shape[0]
print(f"\nReward matrix shape: {rewards.shape} (n_features={n_features}, n_vertices={n_vertices})")

# =============================================================================
# 3. Generate synthetic observations with different counts per feature
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Creating sparse observations directly")
print("="*70)

# Scenario: Feature 0 has 20 observations, Feature 1 has only 5 observations
# This is common in practice when some measurements are more expensive/rare

np.random.seed(42)
true_theta = 2.0

# Feature 0: 20 observations
obs_feature_0 = np.random.exponential(scale=1/true_theta, size=20)

# Feature 1: 5 observations
obs_feature_1 = np.random.exponential(scale=1/true_theta, size=5)

print(f"\nTrue theta: {true_theta}")
print(f"Feature 0: {len(obs_feature_0)} observations, mean={np.mean(obs_feature_0):.3f}")
print(f"Feature 1: {len(obs_feature_1)} observations, mean={np.mean(obs_feature_1):.3f}")

# Create SparseObservations directly
# Values are concatenated, grouped by feature
# Slices indicate the (start, end) index for each feature
all_values = jnp.concatenate([
    jnp.array(obs_feature_0),
    jnp.array(obs_feature_1)
])
all_features = jnp.concatenate([
    jnp.zeros(len(obs_feature_0), dtype=jnp.int32),  # Feature 0
    jnp.ones(len(obs_feature_1), dtype=jnp.int32),   # Feature 1
])
slices = (
    (0, len(obs_feature_0)),                                    # Feature 0 slice
    (len(obs_feature_0), len(obs_feature_0) + len(obs_feature_1))  # Feature 1 slice
)

sparse_obs = SparseObservations(
    values=all_values,
    features=all_features,
    n_features=n_features,
    slices=slices
)

print(f"\nSparseObservations created:")
print(f"  Total observations: {len(sparse_obs.values)}")
print(f"  Feature indices: {sparse_obs.features[:5]}... (first 5)")
print(f"  Slices: {sparse_obs.slices}")

# Verify we can extract feature values
print(f"\nExtracting values per feature:")
for j in range(n_features):
    vals = sparse_obs.get_feature_values(j)
    print(f"  Feature {j}: {len(vals)} values, mean={float(jnp.mean(vals)):.3f}")

# =============================================================================
# 4. Run SVGD with sparse observations
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: SVGD inference with sparse observations")
print("="*70)

svgd = SVGD(
    model=model,
    observed_data=sparse_obs,
    theta_dim=1,
    n_particles=50,
    n_iterations=100,
    learning_rate=0.01,
    rewards=rewards,
    regularization=0.0,  # Pure likelihood-based inference
    verbose=True
)

print("\nRunning SVGD optimization...")
svgd.optimize()

print(f"\nResults:")
print(f"  True theta: {true_theta}")
print(f"  Estimated theta (mean): {float(svgd.theta_mean[0]):.4f}")
print(f"  Estimated theta (std): {float(svgd.theta_std[0]):.4f}")
print(f"  Particles shape: {svgd.particles.shape}")

# =============================================================================
# 5. Alternative: Convert from dense NaN-padded array
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Converting dense NaN-padded array to sparse")
print("="*70)

# Sometimes you have data in dense format with NaN for missing values
# dense_to_sparse() handles the conversion automatically

# Create dense data with NaN padding (all rows same length)
max_obs = max(len(obs_feature_0), len(obs_feature_1))
dense_data = np.full((max_obs, n_features), np.nan)
dense_data[:len(obs_feature_0), 0] = obs_feature_0
dense_data[:len(obs_feature_1), 1] = obs_feature_1

print(f"\nDense data shape: {dense_data.shape}")
print(f"Dense data (first 5 rows):\n{dense_data[:5]}")
print(f"NaN count per feature: {np.sum(np.isnan(dense_data), axis=0)}")

# Convert to sparse format
sparse_from_dense = dense_to_sparse(jnp.array(dense_data))

print(f"\nConverted to sparse:")
print(f"  Total observations: {len(sparse_from_dense.values)}")
print(f"  Slices: {sparse_from_dense.slices}")
print(f"  Is sparse: {is_sparse_observations(sparse_from_dense)}")

# Verify the conversion preserved the data
print(f"\nVerifying conversion:")
for j in range(n_features):
    orig_vals = sparse_obs.get_feature_values(j)
    conv_vals = sparse_from_dense.get_feature_values(j)
    match = jnp.allclose(jnp.sort(orig_vals), jnp.sort(conv_vals))
    print(f"  Feature {j}: values match = {match}")

# =============================================================================
# 6. Comparison: Sparse vs Dense performance
# =============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Comparing sparse and dense formats")
print("="*70)

# For comparison, run SVGD with dense format (same data, with NaN padding)
# Note: Dense format with NaN can cause gradient issues in some cases

print("\nRunning SVGD with dense format (NaN-padded)...")
svgd_dense = SVGD(
    model=model,
    observed_data=jnp.array(dense_data),
    theta_dim=1,
    n_particles=50,
    n_iterations=100,
    learning_rate=0.01,
    rewards=rewards,
    regularization=0.0,
    verbose=False
)
svgd_dense.optimize()

print("\nRunning SVGD with sparse format...")
svgd_sparse = SVGD(
    model=model,
    observed_data=sparse_from_dense,
    theta_dim=1,
    n_particles=50,
    n_iterations=100,
    learning_rate=0.01,
    rewards=rewards,
    regularization=0.0,
    verbose=False
)
svgd_sparse.optimize()

print(f"\nComparison:")
print(f"  True theta: {true_theta}")
print(f"  Dense format estimate: {float(svgd_dense.theta_mean[0]):.4f} ± {float(svgd_dense.theta_std[0]):.4f}")
print(f"  Sparse format estimate: {float(svgd_sparse.theta_mean[0]):.4f} ± {float(svgd_sparse.theta_std[0]):.4f}")

# =============================================================================
# 7. Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
The SparseObservations format is useful when:

1. Different features have very different numbers of observations
2. You want to avoid NaN propagation through JAX gradients
3. Memory efficiency matters for very sparse observation patterns

Key API:
- SparseObservations(values, features, n_features, slices): Create directly
- dense_to_sparse(data): Convert from 2D NaN-padded array
- is_sparse_observations(data): Check format type
- sparse.get_feature_values(j): Extract values for feature j (JAX JIT compatible)

The slices parameter is required for JAX JIT compatibility - it pre-computes
the (start, end) indices for each feature to avoid dynamic boolean indexing.
""")

print("Done!")
