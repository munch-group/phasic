#!/usr/bin/env python
"""Debug current state to understand what's wrong."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("DEBUG: Current State")
print("="*70)
print()

# Simple exponential model
def exp_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]

g = Graph(callback=exp_callback, parameterized=True, nr_samples=1)

# For exponential(rate=2.0):
# - Without rewards: E[T] = 1/rate = 0.5
# - With reward R on transient state: ??? (this is what we need to figure out)

theta = jnp.array([2.0])
rate = 2.0
times = jnp.array([0.5, 1.0, 1.5])

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False, use_ffi=False)

print("TEST 1: No rewards (baseline)")
print("-"*70)
pmf_baseline, moments_baseline = model(theta, times, rewards=None)
print(f"PMF: {pmf_baseline}")
print(f"Moments: {moments_baseline}")
print(f"Expected E[T] = 1/rate = {1/rate}")
print(f"Actual E[T] = {moments_baseline[0]}")
print()

print("TEST 2: With rewards R=1 (should this equal baseline?)")
print("-"*70)
rewards = jnp.array([0.0, 1.0, 0.0])
pmf_r1, moments_r1 = model(theta, times, rewards=rewards)
print(f"PMF: {pmf_r1}")
print(f"Moments: {moments_r1}")
print(f"PMF matches baseline: {np.allclose(pmf_baseline, pmf_r1)}")
print(f"Moments match baseline: {np.allclose(moments_baseline, moments_r1)}")
print()

print("TEST 3: With rewards R=5")
print("-"*70)
rewards = jnp.array([0.0, 5.0, 0.0])
pmf_r5, moments_r5 = model(theta, times, rewards=rewards)
print(f"PMF: {pmf_r5}")
print(f"Moments: {moments_r5}")
print()

# What SHOULD happen?
print("="*70)
print("QUESTION: What is the CORRECT behavior?")
print("="*70)
print()
print("Option A: Reward affects ONLY moments (not PDF)")
print("  - PDF stays the same with/without rewards")
print("  - Only moments change: E[R*T] = R * E[T]")
print("  - Use case: Moment-based regularization in SVGD")
print()
print("Option B: Reward transforms the DISTRIBUTION (affects PDF)")
print("  - PDF changes with rewards (different distribution)")
print("  - Moments also change")
print("  - Use case: Data sampled from reward-transformed graph")
print()
print("Which one is YOUR use case?")
print("="*70)
