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
model = Graph.pmf_from_graph(g)
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
print("⚠️  Note: Gradients are not available for non-parameterized graphs")
print("    Use parameterized edges (add_edge_parameterized) for gradient support")

# ==============================================================================
# 2. PARAMETERIZED MODEL WITH GRADIENT SUPPORT
# ==============================================================================
print("\n2. PARAMETERIZED MODEL WITH GRADIENT SUPPORT")
print("-" * 40)

# Build model with parameterized edges
print("Building model with parameterized edge...")
g_param = Graph(1)
start = g_param.starting_vertex()
v_transient = g_param.find_or_create_vertex([1])
v_absorbing = g_param.find_or_create_vertex([2])

start.add_edge(v_transient, 1.0)
# Use parameterized edge: weight = edge_state · theta
# edge_state=[1.0] means weight = 1.0 * theta[0]
v_transient.add_edge_parameterized(v_absorbing, weight=0.0, edge_state=[1.0])

g_param.normalize()
print(f"✅ Graph with parameterized edge built")

# Convert to JAX function
param_model = Graph.pmf_from_graph(g_param)
print("✅ Parameterized model created (supports gradients!)")

# Test with different rates
rates = [0.5, 1.0, 1.5, 2.0]
print("\nTesting with different rates:")
for rate in rates:
    theta = jnp.array([rate])
    pdf = param_model(theta, times)
    peak_time = times[jnp.argmax(pdf)]
    print(f"  Rate={rate:.1f}: Peak at t={peak_time:.4f}")

# Test JAX features with parameterized model
print("\nTesting JAX features with parameterized model:")

# JIT compilation
theta_test = jnp.array([1.5])
jit_param_model = jax.jit(param_model)
pdf_param = jit_param_model(theta_test, times)
print(f"✅ JIT compilation works")

# GRADIENT SUPPORT!
print("\n✅ Testing GRADIENTS (this is the key feature!):")
def loss_fn(theta):
    """Example: sum of PDF values"""
    return jnp.sum(param_model(theta, times))

grad_fn = jax.grad(loss_fn)
gradient = grad_fn(theta_test)
print(f"✅ Gradient computation works!")
print(f"   ∂loss/∂θ = {gradient[0]:.6f}")

# VMAP SUPPORT!
print("\n✅ Testing VMAP (vectorization):")
theta_batch = jnp.array([[0.5], [1.0], [1.5]])
vmap_model = jax.vmap(lambda t: param_model(t, times[:10]))
pdf_batch = vmap_model(theta_batch)
print(f"✅ vmap works: output shape = {pdf_batch.shape}")
print(f"   (3 parameter sets × 10 time points)")

# ==============================================================================
# 3. SUMMARY
# ==============================================================================
print("\n3. SUMMARY")
print("-" * 40)
print("""
Current Implementation Status:
✅ Python API graph building (no C++ required!)
✅ Graph serialization to arrays
✅ Automatic C++ code generation from Python graphs
✅ JIT compilation support
✅ Full gradient support with parameterized edges
✅ vmap support for vectorization
✅ Works in both continuous and discrete modes

Key Features:
• Regular edges: Fixed weights
• Parameterized edges: Weights as functions of parameters (gradient support!)
• Automatic detection of parameterized vs non-parameterized graphs
• Full JAX ecosystem integration

Usage patterns:
1. Non-parameterized: Graph.pmf_from_graph(graph)
   - Model signature: model(times)
   - JIT compilation supported
   - No gradient support

2. Parameterized: Graph.pmf_from_graph(graph_with_param_edges)
   - Model signature: model(theta, times)
   - JIT, gradients, and vmap all supported
   - Enable gradient-based inference (SVGD, MLE, etc.)
""")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print("""
Python Graph to JAX with Parameterized Edges:
✅ Build models with Python API (no C++ required)
✅ Automatic C++ code generation for performance
✅ Full JAX support: JIT, gradients, vmap
✅ Gradient-based inference ready (SVGD, optimization)
✅ Works in continuous (PDF) and discrete (PMF) modes

Two approaches available:
1. Python graphs (this file): Best for iterative development
2. C++ models (see jit_pdf.py): Best for complex/reusable models

Both support parameterized edges for gradient-based inference!
""")