"""Test that discretize() returns the correct Graph type"""

import phasic
import numpy as np

print("=" * 70)
print("TEST: discretize() Returns Correct Type")
print("=" * 70)

# Test 1: discretize() returns phasic.Graph with all methods
print("\nTest 1: discretize() returns phasic.Graph")
graph = phasic.Graph(2)
graph.find_or_create_vertex([1, 0])
graph.find_or_create_vertex([2, 0])

carrot_graph, rewards = graph.discretize(0.5)

print(f"  Type: {type(carrot_graph)}")
print(f"  Module: {type(carrot_graph).__module__}")
print(f"  Is phasic.Graph? {isinstance(carrot_graph, phasic.Graph)}")
assert isinstance(carrot_graph, phasic.Graph), "Should be phasic.Graph, not phasic_pybind.Graph"

print(f"  Has plot() method? {hasattr(carrot_graph, 'plot')}")
assert hasattr(carrot_graph, 'plot'), "Should have plot() method"

print(f"  Has discretize() method? {hasattr(carrot_graph, 'discretize')}")
assert hasattr(carrot_graph, 'discretize'), "Should have discretize() method"

print(f"  Has copy() method? {hasattr(carrot_graph, 'copy')}")
assert hasattr(carrot_graph, 'copy'), "Should have copy() method"

print(f"  Vertices: {carrot_graph.vertices_length()}")
print("  ✅ Test 1 PASSED")

# Test 2: Can chain discretize operations
print("\nTest 2: Can chain discretize operations")
graph2 = phasic.Graph(2)
graph2.find_or_create_vertex([1, 0])
graph2.find_or_create_vertex([2, 0])

discrete1, rewards1 = graph2.discretize(0.3)
print(f"  First discretize: {discrete1.vertices_length()} vertices")

discrete2, rewards2 = discrete1.discretize(0.4)
print(f"  Second discretize: {discrete2.vertices_length()} vertices")

assert isinstance(discrete2, phasic.Graph), "Chained discretize should return phasic.Graph"
print("  ✅ Test 2 PASSED")

# Test 3: Rewards array has correct shape
print("\nTest 3: Rewards array has correct shape")
graph3 = phasic.Graph(2)
graph3.find_or_create_vertex([1, 0])
graph3.find_or_create_vertex([2, 0])
graph3.find_or_create_vertex([3, 0])

discrete_graph, rewards_array = graph3.discretize(0.5)
print(f"  Graph vertices: {discrete_graph.vertices_length()}")
print(f"  Rewards shape: {rewards_array.shape}")
print(f"  Rewards length: {len(rewards_array)}")

assert len(rewards_array) == discrete_graph.vertices_length(), "Rewards should have one entry per vertex"
print("  ✅ Test 3 PASSED")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✅")
print("discretize() returns correct phasic.Graph type with all methods!")
print("=" * 70)
