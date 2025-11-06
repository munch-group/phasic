#!/usr/bin/env python3
"""
Test if trace-based PDF evaluation matches direct graph PDF evaluation
"""
import numpy as np
from phasic import Graph, configure
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

configure(ffi=True, openmp=True, jit=True)

# Build simple exponential graph
g = Graph(state_length=1)
start = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0 = g.find_or_create_vertex([0])

# Starting edge: non-parameterized
start.add_edge(v1, 1.0)

# Transition: parameterized, base_weight=0.0, coefficients=[1.0]
v1.add_edge_parameterized(v0, 0.0, [1.0])

print("=" * 80)
print("Trace vs Direct PDF Evaluation Test")
print("=" * 80)
print()

# Record trace
trace = record_elimination_trace(g, param_length=1)
print(f"Trace recorded")
print()

# Test with different parameter values
test_params = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
test_time = 0.5

print(f"PDF evaluation at t={test_time}:")
print(f"{'θ':>6} | {'Direct':>12} | {'Trace':>12} | {'Error %':>10}")
print("-" * 54)

for theta in test_params:
    # Direct evaluation
    g.update_parameterized_weights(np.array([theta]))
    pdf_direct = g.pdf(test_time, granularity=100)

    # Trace-based evaluation
    g_from_trace = instantiate_from_trace(trace, np.array([theta]))
    pdf_trace = g_from_trace.pdf(test_time, granularity=100)

    # Compare
    if pdf_direct > 0:
        error_pct = abs(pdf_trace - pdf_direct) / pdf_direct * 100
    else:
        error_pct = 0.0

    print(f"{theta:6.2f} | {pdf_direct:12.6f} | {pdf_trace:12.6f} | {error_pct:9.2f}%")

print()
print("=" * 80)
