#!/usr/bin/env python
"""Test correctness of reward transformation."""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("CORRECTNESS TEST: Reward Transformation")
print("="*70)
print()

# Create exponential distribution with rate = theta[0]
def exp_callback(state, nr_samples=None):
    if state.size == 0:
        return [([1], 0.0, [1.0])]  # Parameterized starting edge
    if state[0] == 0:
        return []
    return [([0], 0.0, [1.0])]  # Weight = theta[0]

g = Graph(callback=exp_callback, parameterized=True, nr_samples=1)
n_vertices = g.vertices_length()

print(f"Graph: {n_vertices} vertices")
print(f"Model: Exponential distribution with rate = theta[0]")
print()

# For exponential(rate):
# - E[T] = 1/rate
# - E[T^2] = 2/rate^2
# - E[R*T] = R * E[T] = R/rate
# - E[R*T^2] = R * E[T^2] = 2R/rate^2

theta = jnp.array([2.0])  # rate = 2.0
rate = theta[0]

print(f"Parameters: rate = {rate}")
print(f"Expected moments (no rewards): E[T] = {1/rate}, E[T^2] = {2/rate**2}")
print()

# Test 1: No rewards
print("TEST 1: No rewards")
print("-"*70)
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)
times = jnp.array([0.5, 1.0, 1.5])
pmf, moments = model(theta, times, rewards=None)

print(f"Computed moments: E[T] = {moments[0]:.6f}, E[T^2] = {moments[1]:.6f}")
print(f"Expected moments: E[T] = {1/rate:.6f}, E[T^2] = {2/rate**2:.6f}")
print(f"Match: {np.allclose(moments[0], 1/rate) and np.allclose(moments[1], 2/rate**2)}")
print()

# Test 2: 1D rewards
print("TEST 2: 1D rewards")
print("-"*70)

# Set rewards for the 3 vertices
# Vertex 0: state=[0] (absorbing)
# Vertex 1: state=[1] (transient)
# Vertex 2: state=[0] (absorbing, seems to be duplicate?)

# Let's check which vertex is which
print("Vertex states:")
for i in range(n_vertices):
    v = g.vertex_at(i)
    print(f"  Vertex {i}: state={v.state()}")
print()

# For exponential distribution, the reward should be applied to time spent in transient state
# Vertex 1 (state=[1]) is the transient state
# If we set reward R for vertex 1:
# E[R*T] = R * E[T] = R / rate

rewards_1d = jnp.array([0.0, 5.0, 0.0])  # R=5 for transient state
R = 5.0

pmf_1d, moments_1d = model(theta, times, rewards=rewards_1d)
print(f"Rewards: {rewards_1d}")
print(f"Computed moments: E[R*T] = {moments_1d[0]:.6f}, E[R*T^2] = {moments_1d[1]:.6f}")
print(f"Expected moments: E[R*T] = {R/rate:.6f}, E[R*T^2] = {R*2/rate**2:.6f}")
print(f"Match: {np.allclose(moments_1d[0], R/rate)}")
print()

# Test 3: Multiple reward values
print("TEST 3: Multiple 1D reward values")
print("-"*70)

test_rewards = [1.0, 2.0, 5.0, 10.0]
for R in test_rewards:
    rewards = jnp.array([0.0, R, 0.0])
    _, moments = model(theta, times, rewards=rewards)
    expected_mean = R / rate
    computed_mean = moments[0]
    error = abs(computed_mean - expected_mean)
    print(f"R={R:5.1f}: computed={computed_mean:8.4f}, expected={expected_mean:8.4f}, error={error:.6f}")
print()

# Test 4: 2D rewards (multivariate)
print("TEST 4: 2D rewards (multivariate)")
print("-"*70)

model_multi = Graph.pmf_and_moments_from_graph_multivariate(g, nr_moments=2, discrete=False)

# Test with different reward values for each feature
rewards_2d = jnp.array([
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 5.0, 10.0],  # Vertex 1 (transient) - different R per feature
    [0.0, 0.0, 0.0],   # Vertex 2
])

n_features = rewards_2d.shape[1]
pmf_2d, moments_2d = model_multi(theta, times, rewards=rewards_2d)

print(f"Rewards shape: {rewards_2d.shape}")
print(f"Rewards for transient state (vertex 1): {rewards_2d[1, :]}")
print()

print("Results per feature:")
all_correct = True
for j in range(n_features):
    R_j = rewards_2d[1, j]  # Reward for transient state
    expected_mean = R_j / rate
    computed_mean = moments_2d[j, 0]
    error = abs(computed_mean - expected_mean)
    match = np.allclose(computed_mean, expected_mean)
    all_correct = all_correct and match

    print(f"Feature {j}: R={R_j:5.1f}")
    print(f"  Computed: E[R*T] = {computed_mean:8.4f}")
    print(f"  Expected: E[R*T] = {expected_mean:8.4f}")
    print(f"  Error: {error:.6f}")
    print(f"  Match: {match}")
    print()

print("="*70)
print("SUMMARY")
print("="*70)
if all_correct:
    print("✓ SUCCESS: All reward transformations are correct!")
else:
    print("✗ FAILURE: Some reward transformations are incorrect")
    print()
    print("This suggests an issue with how rewards are applied in the")
    print("reward_transform or moment computation.")
print("="*70)
