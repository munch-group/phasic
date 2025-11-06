#!/usr/bin/env python3
"""
Minimal 3-state exponential test to debug PDF scaling bug
=========================================================

Model:
- Start (state [0]) → State 1 (state [1]) with weight=1.0 (instantaneous)
- State 1 (state [1]) → Absorbing (state [2]) at rate θ

Expected PDF: f(t) = θ × exp(-θt)
Expected ∫ PDF dt = 1.0
"""

import numpy as np
from phasic import Graph, configure
from scipy.integrate import trapezoid

configure(ffi=True, openmp=True, jit=True)

print("=" * 80)
print("Minimal 3-State Exponential Model - Debug Test")
print("=" * 80)

def minimal_exponential(state, nr_samples=None):
    """
    Exponential model with parameterized starting edge:
    - Start → State 1: weight = θ (unnormalized, will be normalized to 1.0)
    - State 1 → Absorbing: rate θ
    """
    if state.size == 0:
        return [([1], 0.0, [1.0])]  # Start → State 1 with weight=θ (will be normalized)
    if state[0] == 1:
        return [([0], 0.0, [1.0])]  # State 1 → Absorbing at rate θ
    return []

# Test with θ = 5.0
theta = 5.0
print(f"\nBuilding graph with θ = {theta}...")
graph = Graph(callback=minimal_exponential, parameterized=True, nr_samples=None)

print(f"\nGraph structure:")
print(f"  Vertices: {graph.vertices_length()}")
print(f"  States: {graph.states()}")

print(f"\nUpdating parameterized weights with θ = {theta}...")
graph.update_parameterized_weights(np.array([theta]))

print(f"\nComputing moment (should be 1/θ = {1.0/theta:.6f})...")
moment = graph.moments(1)[0]
print(f"  Moment: {moment:.6f}")

print(f"\nComputing PDF at various granularities...")
times = np.linspace(0.001, 10.0, 1000)

# Test different granularities
for gran in [0, 10, 100, 1000, 10000]:
    if gran == 0:
        gran_label = "auto"
    else:
        gran_label = str(gran)

    pdf = graph.pdf(times, granularity=gran)
    integral = trapezoid(pdf, times)
    print(f"  granularity={gran_label:>5}: ∫ PDF dt = {integral:.6f} (expected: 1.0)")

print(f"\n" + "=" * 80)
print("Expected: All integrals should be ≈ 1.0")
print("If bug present: Integrals will be ≈ θ = 5.0")
print("=" * 80)
