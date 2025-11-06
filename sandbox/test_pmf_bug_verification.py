#!/usr/bin/env python3
"""
Quick verification of PMF scaling bug
======================================

Tests a simple exponential distribution to confirm that graph.pdf()
integrates to θ instead of 1.0.

Exponential(θ):
- PDF: f(t) = θ × exp(-θt)
- Mean: 1/θ
- ∫₀^∞ f(t) dt = 1.0
"""

import numpy as np
from phasic import Graph, configure
from scipy.integrate import trapezoid

configure(ffi=True, openmp=True, jit=True)

print("=" * 80)
print("PMF Scaling Bug Verification - Simple Exponential")
print("=" * 80)

def simple_exponential(state, nr_samples=None):
    """
    Simple exponential distribution: Exp(θ)
    - Start → State 1 (instantaneous)
    - State 1 → Absorbing at rate θ
    """
    if state.size == 0:
        return [([1], 0.0, [1.0])]  # Start → State 1
    if state[0] == 1:
        return [([0], 0.0, [1.0])]  # State 1 → Absorbing at rate 1×θ
    return []

# Test for multiple θ values
theta_values = [1.0, 2.0, 5.0, 10.0]

print(f"\n{'θ':<8} {'Moment':<12} {'Expected':<12} {'∫ PDF dt':<12} {'Expected':<12}")
print(f"{'':─<8} {'':─<12} {'':─<12} {'':─<12} {'':─<12}")

for theta in theta_values:
    # Build graph
    graph = Graph(callback=simple_exponential, parameterized=True, nr_samples=None)
    graph.update_parameterized_weights(np.array([theta]))

    # Check moment (should be 1/θ)
    moment = graph.moments(1)[0]
    expected_moment = 1.0 / theta

    # Check PDF integral (should be 1.0)
    times = np.linspace(0.001, 10.0, 1000)
    pdf = graph.pdf(times)
    integral = trapezoid(pdf, times)

    print(f"{theta:<8.1f} {moment:<12.6f} {expected_moment:<12.6f} {integral:<12.6f} {'1.0':<12}")

print(f"\n{'=' * 80}")
print("DIAGNOSIS")
print("=" * 80)

print("\nIf bug is present:")
print("  - Moments should be correct (1/θ)")
print("  - ∫ PDF dt should equal θ (NOT 1.0)")
print("\nIf bug is fixed:")
print("  - Both moments and integrals should be correct")

print("\n" + "=" * 80)
