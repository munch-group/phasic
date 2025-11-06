#!/usr/bin/env python3
"""
Debug instantiate_from_trace to find the PDF=0 bug
"""
import sys
import importlib

import numpy as np
from phasic import Graph, configure

# Force reload to pick up changes
import phasic.trace_elimination
importlib.reload(phasic.trace_elimination)
from phasic.trace_elimination import record_elimination_trace, evaluate_trace, instantiate_from_trace

configure(ffi=True, openmp=True, jit=True)

# Build simple exponential graph
g = Graph(state_length=1)
start = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0 = g.find_or_create_vertex([0])

# Starting edge: non-parameterized
start.add_edge(v1, 1.0)

# Transition: parameterized, base_weight=0.0, coefficients=[1.0]
# So weight = 0.0 + 1.0*θ = θ
v1.add_edge_parameterized(v0, 0.0, [1.0])

print("=" * 80)
print("instantiate_from_trace() Debug")
print("=" * 80)
print()

# Record trace
trace = record_elimination_trace(g, param_length=1)
print("Trace recorded:")
print(f"  n_vertices: {trace.n_vertices}")
print(f"  starting_vertex_idx: {trace.starting_vertex_idx}")
print()

# Test with θ = 2.0
theta = np.array([2.0])
print(f"Testing with θ = {theta[0]}")
print()

# Step 1: Evaluate trace
result = evaluate_trace(trace, theta)
print("Trace evaluation result:")
for i in range(trace.n_vertices):
    state = trace.states[i]
    rate = result['vertex_rates'][i]
    probs = result['edge_probs'][i]
    targets = result['vertex_targets'][i]
    print(f"  Vertex {i} (state={state}): rate={rate:.6f}")
    if len(targets) > 0:
        for j, (prob, to_idx) in enumerate(zip(probs, targets)):
            to_state = trace.states[to_idx]
            print(f"    Edge {j} → vertex {to_idx} (state={to_state}): prob={prob:.6f}")
print()

# Check states more carefully
print("Trace states (checking for duplicates):")
for i in range(trace.n_vertices):
    state = trace.states[i]
    is_starting = (i == trace.starting_vertex_idx)
    print(f"  Vertex {i}: state={state} {'[STARTING]' if is_starting else ''}")
print()

# Step 2: Check weight recovery formula
print("Weight recovery check:")
for i in range(trace.n_vertices):
    rate = result['vertex_rates'][i]
    probs = result['edge_probs'][i]
    targets = result['vertex_targets'][i]

    if len(targets) == 0:
        continue

    print(f"  Vertex {i}: rate={rate:.6f}")

    for j, (prob, to_idx) in enumerate(zip(probs, targets)):
        # Two interpretations:
        # 1. rate = 1/sum(weights), so weight = prob / rate
        weight_interp1 = prob / rate if rate > 0 else 0.0

        # 2. inv_rate = sum(weights) = 1/rate, so weight = prob * inv_rate
        inv_rate = 1.0 / rate if rate > 0 else 0.0
        weight_interp2 = prob * inv_rate

        # Current buggy code: weight = prob / inv_rate (treating rate as inv_rate)
        weight_buggy = prob / rate if rate > 0 else 0.0

        print(f"    Edge {j}: prob={prob:.6f}")
        print(f"      Correct: weight = prob / rate = {weight_interp1:.6f}")
        print(f"      Also correct: weight = prob * (1/rate) = {weight_interp2:.6f}")
        print(f"      Buggy code does: weight = prob / rate = {weight_buggy:.6f}")
print()

# Step 3: Instantiate and check
print("Instantiating graph from trace...")
g_from_trace = instantiate_from_trace(trace, theta)

print(f"Graph vertices: {g_from_trace.vertices_length()}")
print()

# Check edge weights in instantiated graph
print("Checking instantiated graph structure:")
print(f"  Total vertices: {g_from_trace.vertices_length()}")
start_vertex = g_from_trace.starting_vertex()
print(f"  Starting vertex state: {start_vertex.state()}")
print(f"  Starting vertex edges: {len(start_vertex.edges())}")
for j, edge in enumerate(start_vertex.edges()):
    weight = edge.weight()
    to_state = edge.to().state()
    print(f"    Edge {j} → state={to_state}: weight={weight:.6f}")
print()

# Step 4: Compute PDF
test_time = 0.5
pdf = g_from_trace.pdf(test_time, granularity=100)
expected = theta[0] * np.exp(-theta[0] * test_time)

print(f"PDF at t={test_time}:")
print(f"  From trace: {pdf:.6f}")
print(f"  Expected:   {expected:.6f}")
print(f"  Error:      {abs(pdf - expected) / expected * 100:.2f}%")
print()
print("=" * 80)
