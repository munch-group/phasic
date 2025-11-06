#!/usr/bin/env python
"""
Test to understand the base_weight vs coefficients issue

The Oct 31 commit removed base_weight, changing:
  OLD: weight = base_weight + dot(coefficients, theta)
  NEW: weight = dot(coefficients, theta)

This breaks tutorials/callbacks that relied on base_weight.
"""

import numpy as np
from phasic import Graph

print("=" * 70)
print("Understanding the Base Weight Issue")
print("=" * 70)

print("\nTutorial callback pattern (BROKEN after Oct 31):")
print("  vertex.add_edge_parameterized(child, weight=6.0, edge_state=[6.0])")
print("\nOLD API (with base_weight):")
print("  weight = base_weight + dot(coefficients, theta)")
print("  weight = 6.0 + 6.0*theta")
print("  With theta=[1.0]: weight = 6.0 + 6.0*1.0 = 12.0")
print("  With theta=[0.0]: weight = 6.0 + 6.0*0.0 = 6.0  ← This is the actual rate!")

print("\nNEW API (no base_weight, after Oct 31):")
print("  weight = dot(coefficients, theta)")
print("  weight = 6.0*theta")
print("  With theta=[1.0]: weight = 6.0*1.0 = 6.0  ← Correct!")
print("  With theta=[0.0]: weight = 6.0*0.0 = 0.0  ← Wrong!")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("""
The tutorial callback was designed for the OLD API where:
1. base_weight held the actual rate
2. coefficients were ADDED to it (for sensitivity analysis or perturbations)
3. theta=[0.0] gave the base model

After the Oct 31 commit that removed base_weight, the semantics changed:
1. There is NO base_weight
2. Coefficients ARE the rate
3. theta=[1.0] should give the base model

The tutorial callback needs to be updated to:
  vertex.add_edge_parameterized(child, weight=0.0, edge_state=[6.0])

Or better yet, use the NEW unified API:
  vertex.add_edge(child, [6.0])
""")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("""
For the UNIFIED EDGE interface (my refactoring):
- The weight parameter in add_edge_parameterized() should be IGNORED
- Only the coefficients should be used
- This maintains compatibility with the Oct 31 commit

The tutorial examples need to be updated to:
1. Use vertex.add_edge(child, [rate]) for parameterized edges
2. Or understand that weight parameter is ignored
""")
