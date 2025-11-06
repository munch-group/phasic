#!/usr/bin/env python3
"""
Verify Error Scaling with θ=2.0
===============================

Quick test to verify if SVGD error scales correctly with 10k observations
when θ is in the working range [1.5, 2.0].

Expected: From θ=1.5, 50 obs, 15.5% error → θ=2.0, 10k obs should give ~7.8% error
(scaling: 15.5% / √(10000/50) = 15.5% / √200 ≈ 1.1%)
"""

import numpy as np
from phasic import Graph, configure
import jax.numpy as jnp

configure(ffi=True, openmp=True, jit=True)

print("="*80)
print("Verification Test: θ=2.0 with 10,000 Observations")
print("="*80)

# Coalescent callback
def coalescent_callback(state, nr_samples=None):
    if state.size == 0:
        return [([nr_samples], 0.0, [1.0])]
    n = state[0]
    if n <= 1:
        return []
    coalescence_rate = n * (n - 1) / 2
    return [([n - 1], 0.0, [coalescence_rate])]

# Build graph
print("\nBuilding coalescent graph...")
graph = Graph(
    callback=coalescent_callback,
    parameterized=True,
    nr_samples=3
)

# Generate data with θ=2.0 (in working range)
true_theta = 2.0
np.random.seed(42)
n_obs = 10000

graph.update_parameterized_weights(np.array([true_theta]))
observed_times = graph.sample(n_obs)

print(f"\nTrue parameter: θ = {true_theta}")
print(f"Observations: {n_obs}")
print(f"Sample mean: {np.mean(observed_times):.4f}")

# Run SVGD
print("\nRunning SVGD...")
svgd = graph.svgd(
    observed_data=observed_times,
    theta_dim=1,
    n_particles=100,
    n_iterations=2000,
    learning_rate=0.01,
    seed=42,
    verbose=True,
    positive_params=True,
    jit=True,
    parallel='pmap'
)
results = svgd.get_results()

# Results
print(f"\n" + "="*80)
print("RESULTS")
print("="*80)

posterior_mean = results['theta_mean'][0]
posterior_std = results['theta_std'][0]
error = abs(posterior_mean - true_theta)
rel_error = error / true_theta

print(f"\nTrue parameter:     θ = {true_theta:.4f}")
print(f"Posterior mean:     θ̂ = {posterior_mean:.4f}")
print(f"Posterior std:      σ = {posterior_std:.4f}")
print(f"Absolute error:     |θ̂ - θ| = {error:.4f}")
print(f"Relative error:     {rel_error*100:.2f}%")

if rel_error < 0.02:
    print(f"\n✅ PASSED: Relative error {rel_error*100:.2f}% < 2%")
else:
    print(f"\n⚠️  MARGINAL: Relative error {rel_error*100:.2f}% >= 2%")
    print("   (But θ=2.0 is in working range, so this helps diagnose the θ=10 issue)")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nIf error is ~1-2%: Scaling works, θ=10 issue is prior initialization")
print("If error is ~40%:  General SVGD problem, not specific to θ=10")
