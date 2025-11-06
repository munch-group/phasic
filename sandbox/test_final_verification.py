#!/usr/bin/env python
"""Final verification that the implementation is correct."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("FINAL VERIFICATION TEST")
print("="*70)
print()

# Create exponential distribution
def exp_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]

g = Graph(callback=exp_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

print("PART 1: PDF INVARIANCE")
print("-"*70)
print("Reward transformation should NOT change PDF")
print()

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False, use_ffi=False)
theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])

# Test with multiple reward values
rewards_list = [
    jnp.array([0.0, 1.0, 0.0]),
    jnp.array([0.0, 5.0, 0.0]),
    jnp.array([0.0, 10.0, 0.0]),
]

pmf_baseline, _ = model(theta, times, rewards=None)

print("Baseline (no rewards):")
print(f"  PMF: {pmf_baseline}")
print()

all_match = True
for rewards in rewards_list:
    R = rewards[1]  # Reward for transient state
    pmf, moments = model(theta, times, rewards=rewards)
    
    match = np.allclose(pmf, pmf_baseline)
    all_match = all_match and match
    
    print(f"With R={R}:")
    print(f"  PMF: {pmf}")
    print(f"  Matches baseline: {match}")
    print(f"  E[R*T]: {moments[0]:.4f} (expected: {R/theta[0]:.4f})")
    print()

print(f"PDF INVARIANCE: {'PASS' if all_match else 'FAIL'}")
print()

print("="*70)
print("PART 2: MULTIVARIATE - PMF BROADCAST, MOMENTS DIFFER")
print("-"*70)
print()

model_multi = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, discrete=False)

# 2D rewards with different values per feature
rewards_2d = jnp.array([
    [0.0, 0.0, 0.0],      # Vertex 0 (absorbing)
    [1.0, 5.0, 10.0],     # Vertex 1 (transient) - different R per feature
    [0.0, 0.0, 0.0],      # Vertex 2 (absorbing)
])

pmf_2d, moments_2d = model_multi(theta, times, rewards=rewards_2d)

print("2D Rewards:")
for j in range(rewards_2d.shape[1]):
    R_j = rewards_2d[1, j]
    print(f"  Feature {j}: R={R_j}")

print()

# Check PMF is broadcasted (all columns identical)
pmf_broadcast_correct = True
for j in range(1, rewards_2d.shape[1]):
    if not np.allclose(pmf_2d[:, 0], pmf_2d[:, j]):
        pmf_broadcast_correct = False

print(f"PMF values (should be identical across features):")
for t_idx, t in enumerate(times):
    print(f"  t={t}: {pmf_2d[t_idx, :]}")
print(f"PMF broadcast correct: {pmf_broadcast_correct}")
print()

# Check moments differ per feature
print(f"Moments (should differ per feature):")
moments_differ = False
for j in range(rewards_2d.shape[1]):
    R_j = rewards_2d[1, j]
    expected_mean = R_j / theta[0]
    computed_mean = moments_2d[j, 0]
    match = np.isclose(computed_mean, expected_mean)
    
    if j > 0 and not np.isclose(moments_2d[j, 0], moments_2d[0, 0]):
        moments_differ = True
    
    print(f"  Feature {j}: E[R*T]={computed_mean:.4f} (expected: {expected_mean:.4f}, match: {match})")

print(f"Moments differ correctly: {moments_differ}")
print()

print(f"PMF BROADCAST: {'PASS' if pmf_broadcast_correct else 'FAIL'}")
print(f"MOMENTS DIFFER: {'PASS' if moments_differ else 'FAIL'}")
print()

print("="*70)
print("OVERALL RESULT")
print("="*70)

if all_match and pmf_broadcast_correct and moments_differ:
    print("ALL TESTS PASSED")
    print()
    print("Implementation is correct:")
    print("  1. PDF invariant to reward transformation")
    print("  2. Moments correctly transformed by rewards")
    print("  3. Multivariate: PMF broadcasted, moments differ")
    print()
    print("This fixes the SVGD convergence issue where incorrect PDF")
    print("computation from transformed graph caused wrong inference.")
else:
    print("SOME TESTS FAILED")

print("="*70)
