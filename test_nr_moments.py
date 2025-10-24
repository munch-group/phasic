"""Test that SVGD supports any number of moments"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("Testing SVGD with different numbers of moments...\n")

# Create a simple parameterized exponential distribution
graph = Graph(state_length=1, parameterized=True)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

# Test with nr_moments=2 (default)
print("="*60)
print("Test 1: nr_moments=2 (default)")
print("="*60)

model2 = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0])

pmf2, moments2 = model2(theta, times)
print(f"PMF shape: {pmf2.shape}")
print(f"Moments shape: {moments2.shape}")
print(f"Moments: {moments2}")
print(f"Expected: [0.5, 0.5] (for Exp(rate=2))")
assert moments2.shape == (2,), f"Expected moments shape (2,), got {moments2.shape}"
print("✓ Test 1 passed\n")

# Test with nr_moments=3
print("="*60)
print("Test 2: nr_moments=3")
print("="*60)

model3 = Graph.pmf_and_moments_from_graph(graph, nr_moments=3, discrete=False)
pmf3, moments3 = model3(theta, times)
print(f"PMF shape: {pmf3.shape}")
print(f"Moments shape: {moments3.shape}")
print(f"Moments: {moments3}")
print(f"Expected: [0.5, 0.5, 0.75] (E[T], E[T^2], E[T^3] for Exp(rate=2))")
# For Exp(lambda): E[T^k] = k! / lambda^k
# E[T^3] = 3! / 2^3 = 6/8 = 0.75
expected3 = np.array([1/2, 2/4, 6/8])
assert moments3.shape == (3,), f"Expected moments shape (3,), got {moments3.shape}"
assert np.allclose(moments3, expected3, atol=0.01), f"Moments mismatch: {moments3} vs {expected3}"
print("✓ Test 2 passed\n")

# Test with nr_moments=4
print("="*60)
print("Test 3: nr_moments=4")
print("="*60)

model4 = Graph.pmf_and_moments_from_graph(graph, nr_moments=4, discrete=False)
pmf4, moments4 = model4(theta, times)
print(f"PMF shape: {pmf4.shape}")
print(f"Moments shape: {moments4.shape}")
print(f"Moments: {moments4}")
# E[T^4] = 4! / 2^4 = 24/16 = 1.5
expected4 = np.array([1/2, 2/4, 6/8, 24/16])
print(f"Expected: {expected4}")
assert moments4.shape == (4,), f"Expected moments shape (4,), got {moments4.shape}"
assert np.allclose(moments4, expected4, atol=0.01), f"Moments mismatch: {moments4} vs {expected4}"
print("✓ Test 3 passed\n")

# Test with nr_moments=1
print("="*60)
print("Test 4: nr_moments=1 (just mean)")
print("="*60)

model1 = Graph.pmf_and_moments_from_graph(graph, nr_moments=1, discrete=False)
pmf1, moments1 = model1(theta, times)
print(f"PMF shape: {pmf1.shape}")
print(f"Moments shape: {moments1.shape}")
print(f"Moments: {moments1}")
print(f"Expected: [0.5] (just mean)")
assert moments1.shape == (1,), f"Expected moments shape (1,), got {moments1.shape}"
assert np.allclose(moments1, [0.5], atol=0.01), f"Moments mismatch: {moments1} vs [0.5]"
print("✓ Test 4 passed\n")

print("="*60)
print("SUMMARY: All tests passed!")
print("="*60)
print("SVGD now supports any number of moments (nr_moments parameter)")
