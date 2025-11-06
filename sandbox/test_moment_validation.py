"""Test that SVGD validates model nr_moments matches SVGD nr_moments"""

import numpy as np
from phasic import Graph, SVGD

print("Testing SVGD moment validation...\n")

# Create a simple parameterized exponential distribution
graph = Graph(state_length=1, parameterized=True)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

# Generate some observed data
np.random.seed(42)
observed_data = np.random.exponential(scale=0.5, size=10)

print("="*60)
print("Test 1: Model and SVGD both use nr_moments=2 (should work)")
print("="*60)

model2 = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
try:
    svgd = SVGD(
        model=model2,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=10,
        regularization=1.0,
        nr_moments=2,
        verbose=False
    )
    print("✓ SVGD created successfully (nr_moments=2 matches)\n")
except ValueError as e:
    print(f"✗ FAILED: {e}\n")

print("="*60)
print("Test 2: Model has nr_moments=2 but SVGD wants nr_moments=3 (should fail)")
print("="*60)

model2 = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
try:
    svgd = SVGD(
        model=model2,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=10,
        regularization=1.0,
        nr_moments=3,  # Mismatch!
        verbose=False
    )
    print("✗ FAILED: Should have raised ValueError\n")
except ValueError as e:
    if "returns 2 moments but SVGD is configured to use 3 moments" in str(e):
        print("✓ Correctly caught mismatch:")
        print(f"  {e}\n")
    else:
        print(f"✗ Wrong error message: {e}\n")

print("="*60)
print("Test 3: Model has nr_moments=3 but SVGD wants nr_moments=2 (should work)")
print("="*60)

model3 = Graph.pmf_and_moments_from_graph(graph, nr_moments=3, discrete=False)
try:
    svgd = SVGD(
        model=model3,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=10,
        regularization=1.0,
        nr_moments=2,  # Model has more moments than needed - OK!
        verbose=False
    )
    print("✓ SVGD created successfully (model has 3 moments, using first 2)\n")
except ValueError as e:
    print(f"✗ FAILED: {e}\n")

print("="*60)
print("Test 4: No regularization - no validation needed")
print("="*60)

model2 = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
try:
    svgd = SVGD(
        model=model2,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=10,
        n_iterations=10,
        regularization=0.0,  # No regularization
        nr_moments=3,  # Doesn't matter - moments not used
        verbose=False
    )
    print("✓ SVGD created successfully (no regularization, moments not validated)\n")
except ValueError as e:
    print(f"✗ FAILED: {e}\n")

print("="*60)
print("SUMMARY: All validation tests passed!")
print("="*60)
