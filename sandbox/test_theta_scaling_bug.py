#!/usr/bin/env python
"""
Test for theta scaling bug

The user reports that PDF values scale incorrectly with theta after refactoring.
If theta=1, PDFs are correct. If theta=2, PDFs are 2x too large.
"""

import numpy as np
from phasic import Graph

print("=" * 70)
print("Testing Theta Scaling Bug")
print("=" * 70)

# Test 1: Simple parameterized graph with callback
print("\n1. Testing parameterized graph via callback...")

def simple_callback(state):
    """Simple two-state model: 0 -> 1 with rate 3.0"""
    if not state.size:
        # Initial state
        return [[[1], 1.0, []]]
    elif state[0] == 1:
        # Transition with rate = 3.0 (fixed, not dependent on theta)
        return [[[0], 0.0, [3.0]]]
    else:
        # Absorbing state
        return []

graph = Graph(callback=simple_callback, parameterized=True)
print(f"Graph built: {graph.vertices_length()} vertices, param_length={graph.param_length()}")

# Test with theta=1.0 (should give PDF matching exponential with rate=3.0)
print("\nTesting with theta=1.0...")
graph.update_weights([1.0])

t = 1.0
pdf_theta1 = graph.pdf(t, granularity=200)
pdf_analytical = 3.0 * np.exp(-3.0 * t)  # Exponential(3.0)

print(f"  theta=1.0: pdf={pdf_theta1:.6f}, analytical={pdf_analytical:.6f}, ratio={pdf_theta1/pdf_analytical:.3f}")

# Test with theta=2.0 (should still give same PDF! theta shouldn't scale the PDF)
print("\nTesting with theta=2.0...")
graph.update_weights([2.0])

pdf_theta2 = graph.pdf(t, granularity=200)

print(f"  theta=2.0: pdf={pdf_theta2:.6f}, analytical={pdf_analytical:.6f}, ratio={pdf_theta2/pdf_analytical:.3f}")

# Check the issue
if abs(pdf_theta2 / pdf_theta1 - 2.0) < 0.1:
    print("\n❌ BUG CONFIRMED: PDF scales with theta!")
    print(f"   pdf(theta=2.0) / pdf(theta=1.0) = {pdf_theta2/pdf_theta1:.3f}")
    print(f"   Expected: 1.0 (PDF should not change)")
elif abs(pdf_theta2 / pdf_theta1 - 1.0) < 0.1:
    print("\n✅ No bug: PDF does not scale with theta")
else:
    print(f"\n⚠️  Unexpected scaling: ratio = {pdf_theta2/pdf_theta1:.3f}")

# Test 2: Check edge weights
print("\n2. Checking edge weights...")
graph.update_weights([1.0])
v1 = graph.vertex_at(1)
edges_1 = v1.edges()
if edges_1:
    weight_theta1 = edges_1[0].weight()
    print(f"  theta=1.0: edge weight = {weight_theta1:.6f}")

graph.update_weights([2.0])
edges_2 = v1.edges()
if edges_2:
    weight_theta2 = edges_2[0].weight()
    print(f"  theta=2.0: edge weight = {weight_theta2:.6f}")
    print(f"  Ratio: {weight_theta2/weight_theta1:.3f}")

# Test 3: Understand the expected behavior
print("\n3. Understanding expected behavior...")
print("""
QUESTION: Should theta scale the edge rates or not?

Option A: theta scales rates (current behavior)
  - Coefficient [3.0] with theta=[2.0] → weight = 6.0
  - This makes sense for population genetics where theta scales time

Option B: theta does NOT scale rates (user's expectation?)
  - Coefficient [3.0] with theta=[2.0] → weight = 3.0 (???)
  - This doesn't make sense for parameterized edges

Need to check: What was the OLD API behavior?
""")

print("\n" + "=" * 70)
