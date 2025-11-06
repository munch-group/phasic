#!/usr/bin/env python
"""Test PDF correctness with reward transformation."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("PDF CORRECTNESS TEST")
print("="*70)
print()

# Create exponential distribution with rate = theta[0]
def exp_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]

g = Graph(callback=exp_callback, parameterized=True, nr_samples=1)

# For exponential(rate) with reward R:
# The reward-transformed distribution is still exponential with the same rate
# PDF should NOT change!
# Only moments change: E[R*T] = R * E[T] = R/rate

theta = jnp.array([2.0])
rate = theta[0]
times = jnp.array([0.5, 1.0, 1.5])

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

print(f"Exponential distribution: rate = {rate}")
print(f"Expected PDF(t) = rate * exp(-rate * t)")
print()

# Test 1: No rewards
print("TEST 1: No rewards (baseline)")
print("-"*70)
pmf_no_rewards, moments_no_rewards = model(theta, times, rewards=None)

for i, t in enumerate(times):
    expected_pdf = rate * np.exp(-rate * t)
    computed_pdf = pmf_no_rewards[i]
    error = abs(computed_pdf - expected_pdf)
    print(f"t={t}: computed={computed_pdf:.6f}, expected={expected_pdf:.6f}, error={error:.6e}")

print()

# Test 2: With reward R=5
print("TEST 2: With reward R=5")
print("-"*70)
print("Reward transformation should NOT change the PDF!")
print("PDF should remain: rate * exp(-rate * t)")
print()

rewards = jnp.array([0.0, 5.0, 0.0])  # R=5 for transient state
pmf_with_rewards, moments_with_rewards = model(theta, times, rewards=rewards)

print("Comparing PDFs:")
for i, t in enumerate(times):
    expected_pdf = rate * np.exp(-rate * t)
    computed_pdf = pmf_with_rewards[i]
    error = abs(computed_pdf - expected_pdf)
    ratio = computed_pdf / expected_pdf if expected_pdf > 0 else 0
    print(f"t={t}: computed={computed_pdf:.6f}, expected={expected_pdf:.6f}, ratio={ratio:.6f}, error={error:.6e}")

print()

# Check if PDFs match between no-rewards and with-rewards
print("PDF comparison (no rewards vs with rewards):")
for i, t in enumerate(times):
    match = np.allclose(pmf_no_rewards[i], pmf_with_rewards[i])
    ratio = pmf_with_rewards[i] / pmf_no_rewards[i] if pmf_no_rewards[i] > 0 else 0
    print(f"t={t}: no_rewards={pmf_no_rewards[i]:.6f}, with_rewards={pmf_with_rewards[i]:.6f}, ratio={ratio:.6f}, match={match}")

print()

# Test moments
print("Moments comparison:")
print(f"No rewards:   E[T]={moments_no_rewards[0]:.6f} (expected: {1/rate:.6f})")
print(f"With R=5:     E[R*T]={moments_with_rewards[0]:.6f} (expected: {5/rate:.6f})")
print()

print("="*70)
print("ISSUE SUMMARY")
print("="*70)
print("If PDF changes with reward transformation, there's a bug!")
print("Reward transformation should only affect moments, not the PDF.")
print("="*70)
