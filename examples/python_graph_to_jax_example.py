#!/usr/bin/env python
"""
Example: Building Phase-Type Models with Python API for JAX

This demonstrates how to build phase-type models using the Python API
and convert them to JAX-compatible functions without writing C++ code.
"""

import numpy as np
import jax
import jax.numpy as jnp
from ptdalgorithms import Graph
import time

print("=" * 80)
print("PYTHON GRAPH TO JAX: Build Models Without C++")
print("=" * 80)

# ==============================================================================
# 1. SIMPLE FIXED GRAPH
# ==============================================================================
print("\n1. SIMPLE FIXED GRAPH")
print("-" * 40)

# Build an exponential distribution using Python API
print("Building exponential distribution graph...")
g = Graph(1)  # 1-dimensional state
start = g.starting_vertex()
# Use states that don't conflict with starting vertex state [0]
v_transient = g.find_or_create_vertex([1])  # Transient state
v_absorbing = g.find_or_create_vertex([2])  # Absorbing state

# Add transitions
start.add_edge(v_transient, 1.0)  # Start to transient state
v_transient.add_edge(v_absorbing, 1.0)  # Transient to absorption with rate 1.0

# Normalize the graph
g.normalize()
print(f"✅ Graph built with {g.vertices_length()} vertices")

# Test serialization
print("\nSerializing graph...")
serialized = g.serialize()
print(f"  States shape: {serialized['states'].shape}")
print(f"  Edges shape: {serialized['edges'].shape}")
print(f"  Start edges shape: {serialized['start_edges'].shape}")

# Convert to JAX function
print("\nConverting to JAX function...")
model = Graph.from_python_graph(g)
print(f"✅ JAX-compatible function created")

# Test the model
times = jnp.linspace(0.1, 5.0, 50)
pdf = model(times)
print(f"\nPDF evaluation:")
print(f"  Times shape: {times.shape}")
print(f"  PDF shape: {pdf.shape}")
print(f"  PDF at t=1.0: {pdf[9]:.6f}")
print(f"  Total probability: {jnp.sum(pdf):.4f}")

# Test JAX features
print("\nTesting JAX features:")

# JIT compilation
jit_model = jax.jit(model)
pdf_jit = jit_model(times)
print(f"✅ JIT compilation works: {jnp.allclose(pdf, pdf_jit)}")

# Note about gradients
print("⚠️  Note: Gradient support requires custom_jvp (not implemented yet)")
print("    For gradient-based optimization, use load_cpp_model() instead")

# ==============================================================================
# 2. PARAMETERIZED MODEL
# ==============================================================================
print("\n2. PARAMETERIZED MODEL")
print("-" * 40)

def build_exponential(rate):
    """Build exponential distribution with given rate"""
    g = Graph(1)
    start = g.starting_vertex()
    v_transient = g.find_or_create_vertex([1])
    v_absorbing = g.find_or_create_vertex([2])

    start.add_edge(v_transient, 1.0)
    v_transient.add_edge(v_absorbing, float(rate))  # Use the rate parameter

    g.normalize()
    return g

print("Creating parameterized model...")
param_model = Graph.from_python_graph_parameterized(build_exponential)
print("✅ Parameterized model created")

# Test with different rates
rates = jnp.array([0.5, 1.0, 1.5, 2.0])
print("\nTesting with different rates:")
for rate in rates:
    theta = jnp.array([rate])
    pdf = param_model(theta, times)
    mean_time = jnp.sum(times * pdf) / jnp.sum(pdf)
    print(f"  Rate={rate:.1f}: Mean time={mean_time:.4f} (expected={1/rate:.4f})")

# Test JAX features with parameterized model
print("\nTesting JAX with parameterized model:")

# JIT compilation
jit_param_model = jax.jit(param_model)
theta = jnp.array([1.5])
pdf_param = jit_param_model(theta, times)
print(f"✅ JIT compilation works")

# Note about gradients
print("⚠️  Note: Gradient support requires custom_jvp (not implemented yet)")

# Note about vmap
print("⚠️  vmap support requires vmap_method parameter (not implemented yet)")

# ==============================================================================
# 3. SUMMARY AND LIMITATIONS
# ==============================================================================
print("\n3. SUMMARY AND LIMITATIONS")
print("-" * 40)
print("""
Current Implementation Status:
✅ Python API graph building
✅ Graph serialization to arrays
✅ C++ reconstruction from arrays
✅ JIT compilation support
⚠️  Gradient support (requires custom_jvp)
⚠️  vmap support (requires vmap_method)

For full JAX support including gradients:
  Use Graph.load_cpp_model() with C++ model files
""")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Python Graph to JAX provides a way to:
✅ Build models with Python API (no C++ required)
✅ Convert to JAX-compatible functions
✅ Use JIT compilation for performance
✅ Support both fixed and parameterized models

Current limitations (to be addressed):
⚠️  Gradient support requires custom_jvp implementation
⚠️  vmap requires vmap_method parameter

Usage patterns:
1. Fixed graph: Graph.from_python_graph(graph)
2. Parameterized: Graph.from_python_graph_parameterized(builder_fn)

For full JAX support including gradients, use:
  Graph.load_cpp_model() with C++ model files
""")