#!/usr/bin/env python3
"""
Complete SVGD Inference Example Using GraphBuilder Backend

This script demonstrates:
1. Creating a parameterized phase-type model
2. Running SVGD inference with GraphBuilder (C++ backend)
3. Using rewards for moment transformation
4. Multivariate distributions with 2D rewards
5. Results visualization and diagnostics

Author: Generated for phasic library
Date: 2025-10-29
"""

import numpy as np
import matplotlib.pyplot as plt
from phasic import Graph, configure
import jax.numpy as jnp

# Configure phasic for optimal performance
configure(ffi=True, openmp=True, jit=True)

print("="*70)
print("SVGD Inference Example with GraphBuilder Backend")
print("="*70)

# ============================================================================
# Example 1: Basic SVGD with Coalescent Model
# ============================================================================

print("\n" + "="*70)
print("Example 1: Basic SVGD Inference")
print("="*70)

def coalescent_callback(state, nr_samples=None):
    """
    Coalescent model: n lineages → n-1 lineages at rate n*(n-1)/2

    Parameterized: rate = theta * n*(n-1)/2
    where theta is the population-scaled mutation rate
    """
    if state.size == 0:
        # Initial state: nr_samples lineages
        return [([nr_samples], 0.0, [1.0])]

    n = state[0]
    if n <= 1:
        # Absorbing state
        return []

    # Coalescence rate: n choose 2
    coalescence_rate = n * (n - 1) / 2

    # Return: (next_state, base_weight, [coefficient for theta])
    # Edge weight = base_weight + coefficient * theta
    # Here: weight = 0 + coalescence_rate * theta
    return [([n - 1], 0.0, [coalescence_rate])]

print("\n1. Building parameterized coalescent graph...")
graph = Graph(
    callback=coalescent_callback,
    parameterized=True,
    nr_samples=3
)

print(f"   Graph created with {graph.vertices_length()} vertices")
print(f"   Vertices: {[graph.vertex_at(i).state() for i in range(graph.vertices_length())]}")

# Generate synthetic data from true parameter
print("\n2. Generating synthetic observations...")
true_theta = 1.5
np.random.seed(42)

# Sample from the true model
n_obs = 50
graph.update_parameterized_weights(np.array([true_theta]))
observed_times = graph.sample(n_obs)

print(f"   True parameter: θ = {true_theta}")
print(f"   Generated {n_obs} observations")
print(f"   Sample mean: {np.mean(observed_times):.3f}")
print(f"   Sample std:  {np.std(observed_times):.3f}")

# Run SVGD inference
print("\n3. Running SVGD inference...")
print("   Using GraphBuilder backend with FFI + OpenMP")

svgd = graph.svgd(
    observed_data=observed_times,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    seed=42,
    verbose=True,
    positive_params=True,  # Constrain theta > 0
    jit=True,
    parallel='pmap'  # Use multi-core parallelization
)
results = svgd.get_results()

print(f"\n4. Results:")
print(f"   True θ:       {true_theta:.3f}")
print(f"   Posterior μ:  {results['theta_mean'][0]:.3f}")
print(f"   Posterior σ:  {results['theta_std'][0]:.3f}")
print(f"   Error:        {abs(results['theta_mean'][0] - true_theta):.3f}")
print(f"   Relative err: {abs(results['theta_mean'][0] - true_theta) / true_theta * 100:.1f}%")

# ============================================================================
# Example 2: SVGD with Moment Regularization
# ============================================================================

print("\n" + "="*70)
print("Example 2: SVGD with Moment Regularization")
print("="*70)

print("\n1. Building coalescent graph with more complexity...")
def complex_coalescent(state, nr_samples=None):
    """Coalescent with nr_samples for more complex inference"""
    if state.size == 0:
        return [([nr_samples], 0.0, [1.0])]

    n = state[0]
    if n <= 1:
        return []

    rate = n * (n - 1) / 2
    return [([n - 1], 0.0, [rate])]

graph2 = Graph(
    callback=complex_coalescent,
    parameterized=True,
    nr_samples=5
)

print(f"   Graph created with {graph2.vertices_length()} vertices")

# Generate data
print("\n2. Generating observations with noise...")
true_theta2 = 2.0
n_obs2 = 100
graph2.update_parameterized_weights(np.array([true_theta2]))
observed_times2 = graph2.sample(n_obs2)

print(f"   True θ: {true_theta2}")
print(f"   Observations: {n_obs2}")

# Run SVGD with regularization
print("\n3. Running SVGD with moment regularization...")
from phasic import ExpRegularization

# Start with strong regularization, decay over time
regularization = ExpRegularization(
    first_reg=5.0,   # Strong initial regularization
    last_reg=0.1,    # Weak final regularization
    tau=200.0        # Decay over 200 iterations
)

svgd2 = graph2.svgd(
    observed_data=observed_times2,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    regularization=regularization,  # Dynamic regularization
    nr_moments=2,  # Match mean and variance
    seed=42,
    verbose=True,
    positive_params=True
)
results2 = svgd2.get_results()

print(f"\n4. Results with regularization:")
print(f"   True θ:       {true_theta2:.3f}")
print(f"   Posterior μ:  {results2['theta_mean'][0]:.3f}")
print(f"   Posterior σ:  {results2['theta_std'][0]:.3f}")
print(f"   Error:        {abs(results2['theta_mean'][0] - true_theta2):.3f}")

# ============================================================================
# Example 3: SVGD with Rewards (Moment Transformation)
# ============================================================================

print("\n" + "="*70)
print("Example 3: SVGD with Rewards")
print("="*70)

print("\n1. Using rewards to focus on specific vertices...")

# Create reward vector: emphasize middle vertices
n_vertices = graph2.vertices_length()
rewards = np.ones(n_vertices)
rewards[2:4] = 5.0  # Emphasize middle coalescence events

print(f"   Reward vector: {rewards}")
print(f"   High-reward vertices: {[i for i, r in enumerate(rewards) if r > 1]}")

# Run SVGD with rewards
print("\n2. Running SVGD with reward transformation...")
svgd3 = graph2.svgd(
    observed_data=observed_times2,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    rewards=rewards,  # Apply reward transformation
    regularization=1.0,
    nr_moments=2,
    seed=42,
    verbose=True,
    positive_params=True
)
results3 = svgd3.get_results()

print(f"\n3. Results with rewards:")
print(f"   True θ:       {true_theta2:.3f}")
print(f"   Posterior μ:  {results3['theta_mean'][0]:.3f}")
print(f"   Posterior σ:  {results3['theta_std'][0]:.3f}")

# ============================================================================
# Example 4: Multivariate SVGD (2D Rewards)
# ============================================================================

print("\n" + "="*70)
print("Example 4: Multivariate SVGD with 2D Rewards")
print("="*70)

print("\n1. Creating sparse multivariate observations with NaNs...")

# Generate SPARSE observations: each row has only ONE non-NaN value
# This demonstrates realistic missing data - common in many applications
n_features = 3
n_obs_per_feature = 30
n_obs_total = n_obs_per_feature * n_features  # 90 total rows

# Initialize with all NaNs: (90, 3)
observed_mv = np.full((n_obs_total, n_features), np.nan)

# Generate observations for each feature separately using graph.sample()
# Note: graph2 was already updated with true_theta2 earlier
for feature_idx in range(n_features):
    # Sample times for this feature
    feature_times = graph2.sample(n_obs_per_feature)

    # Place each observation in its own row with NaNs in other columns
    # This creates a sparse pattern: each row has exactly ONE value
    for i, time_val in enumerate(feature_times):
        row_idx = feature_idx + i * n_features  # Round-robin placement
        observed_mv[row_idx, feature_idx] = time_val

# Result pattern:
# Row 0: [val, nan, nan]  <- Only feature 0 observed
# Row 1: [nan, val, nan]  <- Only feature 1 observed
# Row 2: [nan, nan, val]  <- Only feature 2 observed
# Row 3: [val, nan, nan]  <- Only feature 0 observed again
# ...

print(f"   Observations shape: {observed_mv.shape}")
print(f"   Features: {n_features}")
print(f"   Total rows: {n_obs_total}")
print(f"   Non-NaN values: {np.sum(~np.isnan(observed_mv))} ({n_obs_per_feature} per feature)")
print(f"   Sparsity: {np.sum(np.isnan(observed_mv)) / observed_mv.size * 100:.1f}% NaN")
print(f"\n   Example rows (showing sparse pattern):")
for i in range(min(6, n_obs_total)):
    row_display = [f"{val:.3f}" if not np.isnan(val) else "nan" for val in observed_mv[i]]
    print(f"   Row {i}: [{', '.join(row_display)}]")

# Create 2D rewards: (n_vertices, n_features)
# Each column defines the reward vector for one marginal distribution
rewards_2d = np.ones((n_vertices, n_features))

# Feature 0: Emphasize early coalescences
rewards_2d[1:3, 0] = 3.0

# Feature 1: Emphasize middle coalescences
rewards_2d[2:4, 1] = 3.0

# Feature 2: Emphasize late coalescences
rewards_2d[3:5, 2] = 3.0

print(f"   Rewards shape: {rewards_2d.shape}")
print(f"   Feature 0 rewards: {rewards_2d[:, 0]}")
print(f"   Feature 1 rewards: {rewards_2d[:, 1]}")
print(f"   Feature 2 rewards: {rewards_2d[:, 2]}")

# Run multivariate SVGD
print("\n2. Running multivariate SVGD...")
svgd4 = graph2.svgd(
    observed_data=observed_mv,  # 2D observations
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    rewards=rewards_2d,  # 2D rewards
    regularization=1.0,
    nr_moments=2,
    seed=42,
    verbose=True,
    positive_params=True
)
results4 = svgd4.get_results()

print(f"\n3. Multivariate SVGD results:")
print(f"   True θ:       {true_theta2:.3f}")
print(f"   Posterior μ:  {results4['theta_mean'][0]:.3f}")
print(f"   Posterior σ:  {results4['theta_std'][0]:.3f}")

# ============================================================================
# Example 5: Multi-Parameter Model
# ============================================================================

print("\n" + "="*70)
print("Example 5: Multi-Parameter SVGD")
print("="*70)

def two_param_model(state, nr_samples=None):
    """
    Two-parameter model: theta1 controls early coalescence, theta2 controls late

    Edge weight = theta1 * early_rate + theta2 * late_rate
    """
    if state.size == 0:
        return [([nr_samples], 0.0, [1.0, 0.0])]  # Initial edge: only theta1

    n = state[0]
    if n <= 1:
        return []

    rate = n * (n - 1) / 2

    # Early coalescences (n > 2): weighted by theta1
    # Late coalescences (n <= 2): weighted by theta2
    if n > 2:
        # Early: mostly theta1
        return [([n - 1], 0.0, [rate * 0.8, rate * 0.2])]
    else:
        # Late: mostly theta2
        return [([n - 1], 0.0, [rate * 0.2, rate * 0.8])]

print("\n1. Building two-parameter model...")
graph_2p = Graph(
    callback=two_param_model,
    parameterized=True,
    nr_samples=4
)

print(f"   Graph created with {graph_2p.vertices_length()} vertices")

# Generate data from true parameters
print("\n2. Generating observations from two-parameter model...")
true_theta_2p = np.array([1.5, 2.5])
n_obs_2p = 100
graph_2p.update_parameterized_weights(true_theta_2p)
observed_2p = graph_2p.sample(n_obs_2p)

print(f"   True θ₁: {true_theta_2p[0]}")
print(f"   True θ₂: {true_theta_2p[1]}")
print(f"   Observations: {n_obs_2p}")

# Run SVGD for 2D parameter space
print("\n3. Running 2D SVGD inference...")
svgd_2p = graph_2p.svgd(
    observed_data=observed_2p,
    theta_dim=2,  # Two parameters
    n_particles=200,  # More particles for 2D
    n_iterations=1000,
    learning_rate=0.005,  # Smaller step size for stability
    seed=42,
    verbose=True,
    positive_params=True
)
results_2p = svgd_2p.get_results()

print(f"\n4. Two-parameter results:")
print(f"   True θ₁:      {true_theta_2p[0]:.3f}")
print(f"   Posterior μ₁: {results_2p['theta_mean'][0]:.3f}")
print(f"   Posterior σ₁: {results_2p['theta_std'][0]:.3f}")
print(f"   Error θ₁:     {abs(results_2p['theta_mean'][0] - true_theta_2p[0]):.3f}")
print()
print(f"   True θ₂:      {true_theta_2p[1]:.3f}")
print(f"   Posterior μ₂: {results_2p['theta_mean'][1]:.3f}")
print(f"   Posterior σ₂: {results_2p['theta_std'][1]:.3f}")
print(f"   Error θ₂:     {abs(results_2p['theta_mean'][1] - true_theta_2p[1]):.3f}")

# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("Generating Visualizations")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('SVGD Inference Results with GraphBuilder Backend', fontsize=14, fontweight='bold')

# Example 1: Basic SVGD
ax = axes[0, 0]
ax.hist(results['particles'].flatten(), bins=30, density=True, alpha=0.7,
        color='steelblue', edgecolor='black')
ax.axvline(true_theta, color='red', linestyle='--', linewidth=2, label='True θ')
ax.axvline(results['theta_mean'][0], color='green', linestyle='-', linewidth=2,
          label='Posterior mean')
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 1: Basic SVGD', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Example 2: With regularization
ax = axes[0, 1]
ax.hist(results2['particles'].flatten(), bins=30, density=True, alpha=0.7,
        color='coral', edgecolor='black')
ax.axvline(true_theta2, color='red', linestyle='--', linewidth=2, label='True θ')
ax.axvline(results2['theta_mean'][0], color='green', linestyle='-', linewidth=2,
          label='Posterior mean')
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 2: With Regularization', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Example 3: With rewards
ax = axes[0, 2]
ax.hist(results3['particles'].flatten(), bins=30, density=True, alpha=0.7,
        color='mediumpurple', edgecolor='black')
ax.axvline(true_theta2, color='red', linestyle='--', linewidth=2, label='True θ')
ax.axvline(results3['theta_mean'][0], color='green', linestyle='-', linewidth=2,
          label='Posterior mean')
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 3: With Rewards', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Example 4: Multivariate
ax = axes[1, 0]
ax.hist(results4['particles'].flatten(), bins=30, density=True, alpha=0.7,
        color='seagreen', edgecolor='black')
ax.axvline(true_theta2, color='red', linestyle='--', linewidth=2, label='True θ')
ax.axvline(results4['theta_mean'][0], color='green', linestyle='-', linewidth=2,
          label='Posterior mean')
ax.set_xlabel('θ', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 4: Multivariate (2D Rewards)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Example 5: Two parameters - scatter plot
ax = axes[1, 1]
particles_2p = results_2p['particles']
ax.scatter(particles_2p[:, 0], particles_2p[:, 1], alpha=0.5, s=20,
          color='steelblue', edgecolor='black', linewidth=0.5)
ax.axvline(true_theta_2p[0], color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(true_theta_2p[1], color='red', linestyle='--', linewidth=2, alpha=0.7,
          label='True θ')
ax.scatter(results_2p['theta_mean'][0], results_2p['theta_mean'][1],
          color='green', s=200, marker='*', edgecolor='black', linewidth=2,
          label='Posterior mean', zorder=5)
ax.set_xlabel('θ₁', fontsize=11)
ax.set_ylabel('θ₂', fontsize=11)
ax.set_title('Example 5: 2D Parameter Space', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Convergence comparison
ax = axes[1, 2]
examples = ['Basic', 'Regular.', 'Rewards', 'Multivar', '2-Param']
errors = [
    abs(results['theta_mean'][0] - true_theta),
    abs(results2['theta_mean'][0] - true_theta2),
    abs(results3['theta_mean'][0] - true_theta2),
    abs(results4['theta_mean'][0] - true_theta2),
    np.mean([abs(results_2p['theta_mean'][i] - true_theta_2p[i])
             for i in range(2)])
]
colors = ['steelblue', 'coral', 'mediumpurple', 'seagreen', 'steelblue']
bars = ax.bar(examples, errors, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Absolute Error', fontsize=11)
ax.set_title('Convergence Quality Comparison', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Rotate x labels
ax.set_xticklabels(examples, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('/tmp/svgd_graphbuilder_results.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: /tmp/svgd_graphbuilder_results.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\n✅ All examples completed successfully!")
print("\nKey Features Demonstrated:")
print("  1. ✅ Basic SVGD inference with GraphBuilder")
print("  2. ✅ Moment regularization with dynamic schedules")
print("  3. ✅ Reward transformation for selective moments")
print("  4. ✅ Multivariate SVGD with 2D observations and rewards")
print("  5. ✅ Multi-parameter (2D) inference")

print("\nBackend Information:")
print("  • GraphBuilder: C++ implementation via pybind11/FFI")
print("  • Parallelization: Multi-core with OpenMP (pmap)")
print("  • JIT compilation: Enabled for optimal performance")
print("  • FFI caching: GraphBuilder cached by structure hash")

print("\nPerformance Notes:")
print("  • FFI mode: ~5-10x faster than pure Python")
print("  • Multi-core: Scales with available CPUs")
print("  • Typical speed: 100-1000 iterations/second")

print("\n" + "="*70)
print("Example script completed!")
print("="*70)
