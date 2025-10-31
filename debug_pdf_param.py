#!/usr/bin/env python3
"""
Debug PDF evaluation with different parameter values
"""
import numpy as np
from phasic import Graph, configure

configure(ffi=True, openmp=True, jit=True)

# Build simple exponential graph
g = Graph(state_length=1)
start = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0 = g.find_or_create_vertex([0])

# Starting edge: non-parameterized, weight=1.0
start.add_edge(v1, 1.0)

# Transition: parameterized, base_weight=0.0, coefficients=[1.0]
# So weight = 0.0 + 1.0*θ = θ
v1.add_edge_parameterized(v0, 0.0, [1.0])

print("=" * 80)
print("Exponential Distribution PDF Evaluation Test")
print("=" * 80)
print()

# Check edge properties
param_edges = v1.parameterized_edges()
if len(param_edges) > 0:
    edge = param_edges[0]
    print(f"Parameterized edge created:")
    print(f"  base_weight() = {edge.base_weight()}")
    print(f"  weight() before update = {edge.weight()}")
    print()
else:
    print("ERROR: No parameterized edges found!")
    import sys
    sys.exit(1)

# Test PDF at t=0.5 with different parameter values
test_params = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
test_time = 0.5

print(f"PDF evaluation at t={test_time}:")
print(f"{'θ':>6} | {'PDF(t)':>12} | {'Expected':>12} | {'Error %':>10}")
print("-" * 50)

for theta in test_params:
    g.update_parameterized_weights(np.array([theta]))

    # Check weight after update
    param_edges_updated = v1.parameterized_edges()
    if len(param_edges_updated) > 0:
        weight_after = param_edges_updated[0].weight()
    else:
        weight_after = -1.0

    pdf = g.pdf(test_time, granularity=100)
    expected = theta * np.exp(-theta * test_time)  # Exponential PDF
    error_pct = abs(pdf - expected) / expected * 100

    print(f"{theta:6.2f} | {pdf:12.6f} | {expected:12.6f} | {error_pct:9.2f}%")

    if error_pct > 1.0:
        print(f"  WARNING: Large error! edge.weight() after update = {weight_after}")

print()
print("=" * 80)
