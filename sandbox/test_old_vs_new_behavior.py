#!/usr/bin/env python
"""
Understanding the OLD vs NEW API behavior
"""

import numpy as np

print("=" * 70)
print("OLD API Behavior (before unified edge refactoring)")
print("=" * 70)

print("\nTutorial callback:")
print("  transitions.append([new, rate, [rate]])")
print("  where rate = state[i]*(state[j]-same)/(1+same)")
print("\nExample: rate = 6.0")

print("\n1. add_edge_parameterized(child, weight=6.0, edge_state=[6.0])")
print("   Result: edge.weight = 6.0 (uses weight parameter, ignores coefficients)")

print("\n2. User computes PDF without calling update_weights()")
print("   PDF uses edge.weight = 6.0")
print("   PDF is CORRECT because weight=6.0 is the actual rate")

print("\n3. User calls update_weights([1.0])")
print("   edge.weight = dot([6.0], [1.0]) = 6.0")
print("   PDF still CORRECT")

print("\n4. User calls update_weights([2.0])")
print("   edge.weight = dot([6.0], [2.0]) = 12.0")
print("   PDF uses rate=12.0 instead of rate=6.0")
print("   PDF is WRONG (2x too large)")

print("\n" + "=" * 70)
print("NEW API Behavior (after unified edge refactoring)")
print("=" * 70)

print("\n1. add_edge_parameterized(child, weight=6.0, edge_state=[6.0])")
print("   Result: edge.weight = dot([6.0], [1.0]) = 6.0 (default theta=[1.0])")

print("\n2. User computes PDF without calling update_weights()")
print("   PDF uses edge.weight = 6.0")
print("   PDF is CORRECT")

print("\n3. User calls update_weights([1.0])")
print("   edge.weight = dot([6.0], [1.0]) = 6.0")
print("   PDF still CORRECT")

print("\n4. User calls update_weights([2.0])")
print("   edge.weight = dot([6.0], [2.0]) = 12.0")
print("   PDF uses rate=12.0 instead of rate=6.0")
print("   PDF is WRONG (2x too large)")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("""
BOTH the old and new APIs have the SAME problem!

If the tutorial callback returns [new, rate, [rate]], then:
- coefficients = [rate]
- update_weights([theta]) computes weight = rate * theta
- This scales the PDF by theta

The issue is that the tutorial callback is INCORRECTLY DESIGNED.

For coalescent models, theta typically represents:
- theta = 4*Ne*mu (population-scaled mutation rate)
- Coalescent time scale: t' = t / (2*Ne)
- Coalescent rate: λ' = k(k-1)/2

The callback should return:
- coefficients = [k(k-1)/2]
- theta = [1/(2*Ne)]

So that weight = [k(k-1)/2] * [1/(2*Ne)] = k(k-1)/(4*Ne)

NOT:
- coefficients = [k(k-1)/(4*Ne)]
- theta = [2*Ne] (which would make weight depend on Ne²!)
""")

print("\n" + "=" * 70)
print("QUESTION FOR USER")
print("=" * 70)
print("""
Can you confirm:
1. What does theta represent in your model?
2. What were the PDFs like BEFORE my refactoring?
   - Did update_weights([2.0]) give correct or incorrect PDFs?

If update_weights([2.0]) gave CORRECT PDFs before, then there's a bug in my refactoring.
If update_weights([2.0]) gave INCORRECT PDFs before, then the callback needs to be fixed.
""")
