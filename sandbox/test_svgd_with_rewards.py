#!/usr/bin/env python
"""Test SVGD behavior with rewards."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("SVGD WITH REWARDS TEST")
print("="*70)
print()

# The question: What should happen when we use rewards in SVGD?
#
# Scenario: We have a coalescent model with rate θ
# We observe data and want to infer θ
# We use rewards to compute multivariate distributions
#
# Key insight: The PDF/PMF used for likelihood should represent
# the probability of observing the data under the model.
#
# If we apply reward transformation, we're changing the distribution.
# But for likelihood computation, we want P(data | theta), not some
# transformed version.

def coalescent_callback(state, nr_samples=None):
    if state.size == 0:
        return [([3], 0.0, [1.0])]  # 3 lineages initially
    n = state[0]
    if n <= 1:
        return []
    rate = n * (n - 1) / 2
    return [([n - 1], 0.0, [rate])]

g = Graph(callback=coalescent_callback, parameterized=True, nr_samples=3)
print(f"Coalescent model: {g.vertices_length()} vertices")
print()

# Generate synthetic data from known parameter
true_theta = 1.5
np.random.seed(42)
g.update_parameterized_weights(np.array([true_theta]))
observed_data = g.sample(20)

print(f"True parameter: θ = {true_theta}")
print(f"Observed data (first 5): {observed_data[:5]}")
print(f"Data mean: {np.mean(observed_data):.3f}")
print()

# Now: what should the model function return?
#
# For SVGD likelihood computation, we need P(data | theta)
# This is the PDF/PMF of the original (non-transformed) graph
#
# Rewards are used for moment-based regularization, not for likelihood

model_no_ffi = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False, use_ffi=False)

theta_test = jnp.array([1.5])
times_test = jnp.array([0.5, 1.0, 1.5])

print("TEST: What does the model return with/without rewards?")
print("-"*70)

# Without rewards
pmf_no_rewards, moments_no_rewards = model_no_ffi(theta_test, times_test, rewards=None)
print("No rewards:")
print(f"  PMF: {pmf_no_rewards}")
print(f"  Moments: {moments_no_rewards}")
print()

# With rewards
n_vertices = g.vertices_length()
rewards_test = jnp.ones(n_vertices)
pmf_with_rewards, moments_with_rewards = model_no_ffi(theta_test, times_test, rewards=rewards_test)
print("With rewards (all ones):")
print(f"  PMF: {pmf_with_rewards}")
print(f"  Moments: {moments_with_rewards}")
print()

# The question: Should PMF be the same or different?
pmf_match = np.allclose(pmf_no_rewards, pmf_with_rewards)
moments_match = np.allclose(moments_no_rewards, moments_with_rewards)

print(f"PMF match: {pmf_match}")
print(f"Moments match: {moments_match}")
print()

print("="*70)
print("INTERPRETATION")
print("="*70)
print("For SVGD inference:")
print("  - PMF/PDF should be the SAME (likelihood of data under model)")
print("  - Moments should DIFFER (reward-transformed for regularization)")
print()
print("If user reports 'converges to 20 instead of 10', this suggests")
print("the PDF is being incorrectly computed from a transformed graph,")
print("which changes the likelihood and causes wrong inference.")
print("="*70)
