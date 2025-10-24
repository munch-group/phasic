"""Verify inference correctness against known distributions"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("Verifying inference correctness...\n")

# Test 1: Single exponential (should have E[T] = 1/rate, E[T^2] = 2/rate^2)
print("="*60)
print("Test 1: Exponential(rate=2)")
print("="*60)

graph1 = Graph(state_length=1, parameterized=True)
v_start = graph1.starting_vertex()
v_transient = graph1.find_or_create_vertex([1])
v_absorb = graph1.find_or_create_vertex([0])

# Starting vertex connects to initial transient state with probability 1
v_start.add_edge(v_transient, 1.0)
# Transient state transitions to absorbing state with rate = theta[0]
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])  # rate = theta[0]

model1 = Graph.pmf_and_moments_from_graph(graph1, nr_moments=2, discrete=False)

theta = jnp.array([2.0])  # rate = 2
times = jnp.array([0.5, 1.0])

pmf, moments = model1(theta, times)

print(f"Computed moments: {moments}")
print(f"Expected E[T] = 1/2 = 0.5")
print(f"Expected E[T²] = 2/4 = 0.5")
print(f"Computed PMF at t=0.5: {pmf[0]:.6f}")
print(f"Expected PDF(0.5) = 2*exp(-1) = {2*np.exp(-1):.6f}")

if np.abs(moments[0] - 0.5) < 0.01 and np.abs(moments[1] - 0.5) < 0.01:
    print("✓ Moments correct")
else:
    print(f"✗ Moments incorrect! Got {moments}, expected [0.5, 0.5]")

if np.abs(pmf[0] - 2*np.exp(-1)) < 0.01:
    print("✓ PDF correct")
else:
    print(f"✗ PDF incorrect! Got {pmf[0]:.6f}, expected {2*np.exp(-1):.6f}")

# Test 2: Erlang(2, rate=2) - two exponentials in series
print("\n" + "="*60)
print("Test 2: Erlang(2, rate=2)")
print("="*60)

graph2 = Graph(state_length=1, parameterized=True)
v_start2 = graph2.starting_vertex()
v1 = graph2.find_or_create_vertex([1])
v2 = graph2.find_or_create_vertex([2])
v_absorb2 = graph2.find_or_create_vertex([0])

# Starting vertex connects to first transient state
v_start2.add_edge(v1, 1.0)
# First transient state → second transient state with rate = theta[0]
v1.add_edge_parameterized(v2, 0.0, [1.0])  # rate = theta[0]
# Second transient state → absorbing state with rate = theta[0]
v2.add_edge_parameterized(v_absorb2, 0.0, [1.0])  # rate = theta[0]

model2 = Graph.pmf_and_moments_from_graph(graph2, nr_moments=2, discrete=False)

pmf2, moments2 = model2(theta, times)

print(f"Computed moments: {moments2}")
print(f"Expected E[T] = k/λ = 2/2 = 1.0")
print(f"Expected E[T²] = k(k+1)/λ² = 2*3/4 = 1.5")

if np.abs(moments2[0] - 1.0) < 0.01 and np.abs(moments2[1] - 1.5) < 0.01:
    print("✓ Moments correct")
else:
    print(f"✗ Moments incorrect! Got {moments2}, expected [1.0, 1.5]")

# Test 3: Verify rewards=ones gives same as no rewards
print("\n" + "="*60)
print("Test 3: Rewards=ones should match no rewards")
print("="*60)

pmf_none, moments_none = model2(theta, times, rewards=None)
# Graph2 has 4 vertices, so rewards must be length 4
pmf_ones, moments_ones = model2(theta, times, rewards=jnp.ones(4))

print(f"Moments (no rewards): {moments_none}")
print(f"Moments (rewards=ones): {moments_ones}")

if np.allclose(moments_none, moments_ones):
    print("✓ rewards=ones matches no rewards")
else:
    print(f"✗ rewards=ones differs from no rewards!")
    print(f"  Difference: {moments_ones - moments_none}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Check if all moments match expected theoretical values above.")
