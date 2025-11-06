#!/usr/bin/env python
"""
Test the corrected callback pattern
"""

import numpy as np
from phasic import Graph

print("=" * 70)
print("Testing Corrected Callback Pattern")
print("=" * 70)

# OLD (BROKEN) callback - passes rate as both weight and coefficient
def broken_callback(state):
    """Broken: passes rate as both weight and edge_state"""
    if not state.size:
        return [[[1], 1.0, []]]
    elif state[0] == 1:
        rate = 3.0
        return [[[0], rate, [rate]]]  # ← WRONG: rate in both places
    else:
        return []

# NEW (CORRECT) callback - passes rate only in coefficient
def fixed_callback(state):
    """Fixed: passes rate only in edge_state"""
    if not state.size:
        return [[[1], 1.0, []]]
    elif state[0] == 1:
        rate = 3.0
        return [[[0], 0.0, [rate]]]  # ← CORRECT: weight=0.0, rate in coefficient
    else:
        return []

# Even BETTER - use new unified API
def better_callback(state):
    """Better: use unified add_edge() API"""
    if not state.size:
        return [[[1], 1.0]]
    elif state[0] == 1:
        rate = 3.0
        return [[[0], [rate]]]  # ← BEST: just return coefficient array
    else:
        return []

print("\n1. Testing BROKEN callback (OLD pattern)...")
g_broken = Graph(callback=broken_callback, parameterized=True)
g_broken.update_weights([1.0])
pdf_broken_theta1 = g_broken.pdf(1.0, granularity=200)

g_broken.update_weights([2.0])
pdf_broken_theta2 = g_broken.pdf(1.0, granularity=200)

print(f"   theta=1.0: PDF={pdf_broken_theta1:.6f}")
print(f"   theta=2.0: PDF={pdf_broken_theta2:.6f}")
print(f"   Scaling ratio: {pdf_broken_theta2/pdf_broken_theta1:.3f}")

# Analytical: exponential with rate=3.0
pdf_analytical = 3.0 * np.exp(-3.0 * 1.0)
print(f"   Analytical (rate=3.0): PDF={pdf_analytical:.6f}")
print(f"   Error at theta=1.0: {abs(pdf_broken_theta1-pdf_analytical)/pdf_analytical:.2%}")

if pdf_broken_theta2 < pdf_broken_theta1:
    print("   ❌ BUG: PDF decreases when theta increases (wrong scaling)")

print("\n2. Testing FIXED callback (corrected pattern)...")
g_fixed = Graph(callback=fixed_callback, parameterized=True)
g_fixed.update_weights([1.0])
pdf_fixed_theta1 = g_fixed.pdf(1.0, granularity=200)

g_fixed.update_weights([2.0])
pdf_fixed_theta2 = g_fixed.pdf(1.0, granularity=200)

print(f"   theta=1.0: PDF={pdf_fixed_theta1:.6f}")
print(f"   theta=2.0: PDF={pdf_fixed_theta2:.6f}")
print(f"   Error at theta=1.0: {abs(pdf_fixed_theta1-pdf_analytical)/pdf_analytical:.2%}")

# With theta=2, rate becomes 6.0, so PDF should change accordingly
pdf_analytical_theta2 = 6.0 * np.exp(-6.0 * 1.0)
print(f"   Analytical at theta=2.0 (rate=6.0): PDF={pdf_analytical_theta2:.6f}")
print(f"   Error at theta=2.0: {abs(pdf_fixed_theta2-pdf_analytical_theta2)/pdf_analytical_theta2:.2%}")

if abs(pdf_fixed_theta1 - pdf_analytical) / pdf_analytical < 0.02:
    print("   ✅ Correct: theta=1.0 gives base model")
else:
    print("   ❌ Error too large at theta=1.0")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
The issue is NOT with the unified edge interface refactoring.

The issue is that the tutorial callback was designed for the OLD API
(before Oct 31, 2025) which had a base_weight field:

   OLD: weight = base_weight + dot(coefficients, theta)

After base_weight was removed on Oct 31:

   NEW: weight = dot(coefficients, theta)

The tutorial callback needs to be updated to pass weight=0.0:

   OLD (broken):
     transitions.append([new, rate, [rate]])

   NEW (fixed):
     transitions.append([new, 0.0, [rate]])

   BEST (unified API):
     transitions.append([new, [rate]])
""")
