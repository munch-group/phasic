#!/usr/bin/env python
"""
Test for Starting Vertex Edge Scaling Fix

This test verifies that starting vertex edges are NOT scaled when
update_weights() is called on parameterized graphs.

The issue: After unified edge interface refactoring, starting vertex edges
were being incorrectly scaled along with other edges, breaking the initial
probability vector (IPV) semantics.

The fix: Skip starting vertex in ptd_graph_update_weights() loop.
"""

import numpy as np
from phasic import Graph

print("=" * 70)
print("TEST: Starting Vertex Edges Should Not Be Scaled")
print("=" * 70)

# =============================================================================
# Test 1: Manual Construction - Constant Edge
# =============================================================================

print("\n1. Manual Construction - Constant Starting Edge")
print("-" * 70)

g1 = Graph(state_length=1)
v0 = g1.starting_vertex()
v1 = g1.find_or_create_vertex([1])
v2 = g1.find_or_create_vertex([0])

# Starting edge with scalar weight (constant)
v0.add_edge(v1, 5.0)

# Parameterized edge
v1.add_edge(v2, [3.0])

print(f"Graph properties:")
print(f"  - param_length: {g1.param_length()}")
print(f"  - is_parameterized: {g1.is_parameterized()}")

print(f"\nBefore update_weights:")
w_start_before = v0.edges()[0].weight()
w_other_before = v1.edges()[0].weight()
print(f"  - Starting edge weight: {w_start_before:.1f}")
print(f"  - Other edge weight: {w_other_before:.1f}")

g1.update_weights([2.0])

print(f"\nAfter update_weights([2.0]):")
w_start_after = v0.edges()[0].weight()
w_other_after = v1.edges()[0].weight()
print(f"  - Starting edge weight: {w_start_after:.1f}")
print(f"  - Other edge weight: {w_other_after:.1f}")

# Verify
if abs(w_start_after - 5.0) < 0.01:
    print(f"  ✓ PASS: Starting edge unchanged (5.0)")
else:
    print(f"  ✗ FAIL: Starting edge changed from 5.0 to {w_start_after:.1f}")

if abs(w_other_after - 6.0) < 0.01:
    print(f"  ✓ PASS: Other edge scaled correctly (3.0 → 6.0)")
else:
    print(f"  ✗ FAIL: Other edge is {w_other_after:.1f}, expected 6.0")

# =============================================================================
# Test 2: Manual Construction - Parameterized Starting Edge
# =============================================================================

print("\n2. Manual Construction - Parameterized Starting Edge")
print("-" * 70)

g2 = Graph(state_length=1)
v0 = g2.starting_vertex()
v1 = g2.find_or_create_vertex([1])
v2 = g2.find_or_create_vertex([0])

# Starting edge with array coefficient (this becomes parameterized)
v0.add_edge(v1, [7.0])

# Other parameterized edge
v1.add_edge(v2, [3.0])

print(f"Graph properties:")
print(f"  - param_length: {g2.param_length()}")
print(f"  - is_parameterized: {g2.is_parameterized()}")

print(f"\nBefore update_weights:")
w_start_before = v0.edges()[0].weight()
w_other_before = v1.edges()[0].weight()
print(f"  - Starting edge weight: {w_start_before:.1f}")
print(f"  - Other edge weight: {w_other_before:.1f}")

g2.update_weights([2.0])

print(f"\nAfter update_weights([2.0]):")
w_start_after = v0.edges()[0].weight()
w_other_after = v1.edges()[0].weight()
print(f"  - Starting edge weight: {w_start_after:.1f}")
print(f"  - Other edge weight: {w_other_after:.1f}")

# Verify - starting edge should remain at initial weight (7.0)
# even though it has coefficients, because it's from starting vertex
if abs(w_start_after - 7.0) < 0.01:
    print(f"  ✓ PASS: Starting edge unchanged (7.0)")
else:
    print(f"  ✗ FAIL: Starting edge changed from 7.0 to {w_start_after:.1f}")

if abs(w_other_after - 6.0) < 0.01:
    print(f"  ✓ PASS: Other edge scaled correctly (3.0 → 6.0)")
else:
    print(f"  ✗ FAIL: Other edge is {w_other_after:.1f}, expected 6.0")

# =============================================================================
# Test 3: Multi-Parameter Graph
# =============================================================================

print("\n3. Multi-Parameter Graph")
print("-" * 70)

g3 = Graph(state_length=1)
v0 = g3.starting_vertex()
v1 = g3.find_or_create_vertex([1])
v2 = g3.find_or_create_vertex([0])

# Starting edge with 2-parameter coefficients
v0.add_edge(v1, [1.0, 2.0])

# Other edge with 2-parameter coefficients
v1.add_edge(v2, [3.0, 4.0])

print(f"Graph properties:")
print(f"  - param_length: {g3.param_length()}")

print(f"\nInitial weights (theta=[1.0, 1.0]):")
w_start = v0.edges()[0].weight()
w_other = v1.edges()[0].weight()
print(f"  - Starting edge: 1*1 + 2*1 = {w_start:.1f}")
print(f"  - Other edge: 3*1 + 4*1 = {w_other:.1f}")

g3.update_weights([2.0, 0.5])

print(f"\nAfter update_weights([2.0, 0.5]):")
w_start_after = v0.edges()[0].weight()
w_other_after = v1.edges()[0].weight()
print(f"  - Starting edge weight: {w_start_after:.1f}")
print(f"  - Other edge weight: {w_other_after:.1f}")
print(f"  - Expected other: 3*2 + 4*0.5 = 8.0")

# Starting edge should remain unchanged
if abs(w_start_after - w_start) < 0.01:
    print(f"  ✓ PASS: Starting edge unchanged ({w_start:.1f})")
else:
    print(f"  ✗ FAIL: Starting edge changed from {w_start:.1f} to {w_start_after:.1f}")

# Other edge should be 3*2 + 4*0.5 = 8.0
if abs(w_other_after - 8.0) < 0.01:
    print(f"  ✓ PASS: Other edge scaled correctly (7.0 → 8.0)")
else:
    print(f"  ✗ FAIL: Other edge is {w_other_after:.1f}, expected 8.0")

# =============================================================================
# Test 4: Verify PDF Computation Works
# =============================================================================

print("\n4. PDF Computation with Fixed Starting Vertex")
print("-" * 70)

g4 = Graph(state_length=1)
v0 = g4.starting_vertex()
v1 = g4.find_or_create_vertex([1])
v2 = g4.find_or_create_vertex([0])

# Erlang(2, λ) distribution
v0.add_edge(v1, 1.0)  # Starting edge: probability 1.0 to enter v1
v1.add_edge(v2, [3.0])  # Rate = 3*theta

# With theta=1.0, this is Erlang(2, 3)
g4.update_weights([1.0])
t = 1.0
pdf_theta1 = g4.pdf(t, granularity=200)
pdf_analytical = 9.0 * t * np.exp(-3.0 * t)

print(f"Theta = 1.0:")
print(f"  - PDF computed: {pdf_theta1:.6f}")
print(f"  - PDF analytical (Erlang(2,3)): {pdf_analytical:.6f}")
print(f"  - Error: {abs(pdf_theta1 - pdf_analytical)/pdf_analytical:.2%}")

if abs(pdf_theta1 - pdf_analytical) / pdf_analytical < 0.02:
    print(f"  ✓ PASS: PDF computation correct")
else:
    print(f"  ✗ FAIL: PDF error too large")

# With theta=2.0, rate becomes 6.0, so Erlang(2, 6)
g4.update_weights([2.0])
pdf_theta2 = g4.pdf(t, granularity=200)
pdf_analytical_theta2 = 36.0 * t * np.exp(-6.0 * t)

print(f"\nTheta = 2.0:")
print(f"  - PDF computed: {pdf_theta2:.6f}")
print(f"  - PDF analytical (Erlang(2,6)): {pdf_analytical_theta2:.6f}")
print(f"  - Error: {abs(pdf_theta2 - pdf_analytical_theta2)/pdf_analytical_theta2:.2%}")

if abs(pdf_theta2 - pdf_analytical_theta2) / pdf_analytical_theta2 < 0.02:
    print(f"  ✓ PASS: PDF computation correct with scaled parameter")
else:
    print(f"  ✗ FAIL: PDF error too large with scaled parameter")

# =============================================================================
# Test 5: Ensure Multiple update_weights() Calls Work
# =============================================================================

print("\n5. Multiple update_weights() Calls")
print("-" * 70)

g5 = Graph(state_length=1)
v0 = g5.starting_vertex()
v1 = g5.find_or_create_vertex([1])
v2 = g5.find_or_create_vertex([0])

v0.add_edge(v1, [4.0])
v1.add_edge(v2, [3.0])

w_start_initial = v0.edges()[0].weight()
print(f"Initial starting edge weight: {w_start_initial:.1f}")

# Multiple updates
for theta_val in [1.0, 2.0, 0.5, 3.0, 1.5]:
    g5.update_weights([theta_val])
    w_start = v0.edges()[0].weight()
    w_other = v1.edges()[0].weight()
    print(f"  theta={theta_val:.1f}: start={w_start:.1f}, other={w_other:.1f}")

    if abs(w_start - w_start_initial) > 0.01:
        print(f"  ✗ FAIL: Starting edge changed!")
        break
else:
    print(f"  ✓ PASS: Starting edge remained constant through all updates")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("""
All tests verify that starting vertex edges remain unchanged when
update_weights(theta) is called, while non-starting edges are correctly
scaled by the parameter values.

This preserves the initial probability vector (IPV) semantics required
for phase-type distributions.
""")
print("=" * 70)
