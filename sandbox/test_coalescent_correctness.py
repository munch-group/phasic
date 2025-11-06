#!/usr/bin/env python3
"""
Coalescent Model Correctness Test - 10k Observations
=====================================================

Tests SVGD inference on coalescent model with 10,000 observations.
Target: < 2% relative error
"""

from phasic import Graph, SVGD, configure
import numpy as np
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("="*80)
print("Coalescent Model Correctness Test - 10,000 Observations")
print("="*80)

# ============================================================================
# Coalescent Model (from simple_example.ipynb)
# ============================================================================

def coalescent(state, nr_samples=None):
    """Coalescent model with nr_samples lineages"""
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]
        return ipv
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i+j+1] += 1
                transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
        return transitions

# ============================================================================
# Build Graph
# ============================================================================

print("\nBuilding coalescent graph...")
true_theta = 10.0
nr_samples = 4

graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

print(f"Graph vertices: {graph.vertices_length()}")
print(f"True parameter: θ = {true_theta}")

# ============================================================================
# Generate Synthetic Data
# ============================================================================

print("\nGenerating 10,000 observations from true model...")
np.random.seed(42)
n_obs = 10000

# Update graph with true parameter
graph.update_parameterized_weights(np.array([true_theta]))

# Sample from the model
observed_data = graph.sample(n_obs)

print(f"Observations: {n_obs}")
print(f"Sample mean: {np.mean(observed_data):.4f}")
print(f"Sample std:  {np.std(observed_data):.4f}")

# ============================================================================
# Test 1: SVGD Without Regularization
# ============================================================================

print("\n" + "="*80)
print("Test 1: SVGD Without Regularization")
print("="*80)

# Create model without moments (no regularization)
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=0, discrete=False)

print("\nRunning SVGD...")
print(f"  Particles: 100")
print(f"  Iterations: 500")
print(f"  Learning rate: 0.01 (auto-scaled)")

svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    regularization=0.0,  # NO regularization
    nr_moments=0,
    verbose=True,
    positive_params=True
)

svgd.optimize()

# Results
print(f"\n{'='*80}")
print("RESULTS")
print("="*80)

posterior_mean = svgd.theta_mean[0]
posterior_std = svgd.theta_std[0]
error = abs(posterior_mean - true_theta)
rel_error = error / true_theta

print(f"\nTrue parameter:     θ = {true_theta:.4f}")
print(f"Posterior mean:     θ̂ = {posterior_mean:.4f}")
print(f"Posterior std:      σ = {posterior_std:.4f}")
print(f"Absolute error:     |θ̂ - θ| = {error:.4f}")
print(f"Relative error:     {rel_error*100:.2f}%")

# Check convergence
if rel_error < 0.02:
    print(f"\n✅ PASSED: Relative error {rel_error*100:.2f}% < 2%")
    exit_code = 0
else:
    print(f"\n❌ FAILED: Relative error {rel_error*100:.2f}% >= 2%")
    exit_code = 1

# ============================================================================
# Test 2: SVGD With Regularization
# ============================================================================

print("\n" + "="*80)
print("Test 2: SVGD With Regularization (nr_moments=2)")
print("="*80)

# Create model with moments
model_reg = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

print("\nRunning SVGD with regularization...")
print(f"  Regularization: λ = 1.0")

svgd_reg = SVGD(
    model=model_reg,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=100,
    n_iterations=500,
    learning_rate=0.01,
    regularization=1.0,  # With regularization
    nr_moments=2,
    verbose=True,
    positive_params=True
)

svgd_reg.optimize()

# Results
print(f"\n{'='*80}")
print("RESULTS (WITH REGULARIZATION)")
print("="*80)

posterior_mean_reg = svgd_reg.theta_mean[0]
posterior_std_reg = svgd_reg.theta_std[0]
error_reg = abs(posterior_mean_reg - true_theta)
rel_error_reg = error_reg / true_theta

print(f"\nTrue parameter:     θ = {true_theta:.4f}")
print(f"Posterior mean:     θ̂ = {posterior_mean_reg:.4f}")
print(f"Posterior std:      σ = {posterior_std_reg:.4f}")
print(f"Absolute error:     |θ̂ - θ| = {error_reg:.4f}")
print(f"Relative error:     {rel_error_reg*100:.2f}%")

# Check convergence
if rel_error_reg < 0.02:
    print(f"\n✅ PASSED: Relative error {rel_error_reg*100:.2f}% < 2%")
else:
    print(f"\n❌ FAILED: Relative error {rel_error_reg*100:.2f}% >= 2%")
    exit_code = 1

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nCoalescent model (nr_samples={nr_samples}):")
print(f"  Graph vertices: {graph.vertices_length()}")
print(f"  Observations: {n_obs}")

print(f"\nWithout regularization:")
print(f"  Error: {rel_error*100:.2f}% {'✅' if rel_error < 0.02 else '❌'}")

print(f"\nWith regularization (λ=1.0, nr_moments=2):")
print(f"  Error: {rel_error_reg*100:.2f}% {'✅' if rel_error_reg < 0.02 else '❌'}")

if exit_code == 0:
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - Coalescent model correctness verified!")
    print("="*80)
else:
    print("\n" + "="*80)
    print("❌ SOME TESTS FAILED - See results above")
    print("="*80)

import sys
sys.exit(exit_code)
