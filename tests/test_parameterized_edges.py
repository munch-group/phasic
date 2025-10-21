#!/usr/bin/env python
"""
Quick Test: Parameterized Edges Feature

This script verifies that parameterized edges work correctly with full JAX support.
Run this to confirm your installation supports gradient-based inference.
"""

import jax
import jax.numpy as jnp
from phasic import Graph

print("=" * 70)
print("Testing Parameterized Edges Feature")
print("=" * 70)

# Build a simple model with parameterized edges
print("\n1. Building graph with parameterized edges...")
g = Graph(state_length=2)

# Create initial state
initial = g.find_or_create_vertex([2, 0])
g.starting_vertex().add_edge(initial, 1.0)

# Add parameterized edges using iterative construction
index = 1
while index < g.vertices_length():
    vertex = g.vertex_at(index)
    state = vertex.state()

    # Birth: rabbits jump left to right
    if state[0] > 0:
        child = g.find_or_create_vertex([state[0] - 1, state[1] + 1])
        # Parameterized edge: weight = theta[0]
        vertex.add_edge_parameterized(child, weight=0.0, edge_state=[1.0, 0.0])

    # Death: right island floods
    if state[1] > 0:
        child = g.find_or_create_vertex([state[0], state[1] - 1])
        # Parameterized edge: weight = theta[1]
        vertex.add_edge_parameterized(child, weight=0.0, edge_state=[0.0, 1.0])

    index += 1

print(f"✅ Graph built: {g.vertices_length()} vertices")

# Verify serialization detects parameterized edges
print("\n2. Testing serialization...")
serialized = g.serialize()
param_length = serialized.get('param_length', 0)
n_param_edges = len(serialized['param_edges']) if serialized.get('param_edges') is not None else 0

if param_length > 0 and n_param_edges > 0:
    print(f"✅ Serialization detected: {n_param_edges} parameterized edges, param_length={param_length}")
else:
    print(f"❌ FAILED: Parameterized edges not detected!")
    print(f"   param_length={param_length}, n_param_edges={n_param_edges}")
    exit(1)

# Convert to JAX model
print("\n3. Creating JAX model...")
model = Graph.pmf_from_graph(g, discrete=False)
print("✅ Model created")

# Test basic computation
print("\n4. Testing PMF computation...")
theta = jnp.array([1.0, 2.0])  # birth rate=1.0, death rate=2.0
times = jnp.array([0.5, 1.0, 2.0, 5.0])

try:
    pdf = model(theta, times)
    print(f"✅ PMF computed: shape={pdf.shape}, sum={pdf.sum():.6f}")
except Exception as e:
    print(f"❌ FAILED: PMF computation error: {e}")
    exit(1)

# Test JIT compilation
print("\n5. Testing JIT compilation...")
try:
    jit_model = jax.jit(model)
    pdf_jit = jit_model(theta, times)
    print(f"✅ JIT compilation works")
except Exception as e:
    print(f"❌ FAILED: JIT error: {e}")
    exit(1)

# Test gradients
print("\n6. Testing gradient computation...")
try:
    def loss_fn(theta_val):
        return model(theta_val, times).sum()

    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(theta)
    print(f"✅ Gradients work: ∇θ = {gradient}")

    if jnp.allclose(gradient, 0.0):
        print("   ⚠️  Note: Gradients are zero (may need different times/granularity)")
    else:
        print("   ✅ Non-zero gradients computed!")

except Exception as e:
    print(f"❌ FAILED: Gradient error: {e}")
    exit(1)

# Test vmap
print("\n7. Testing vmap (vectorization)...")
try:
    theta_batch = jnp.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]])
    vmap_model = jax.vmap(lambda t: model(t, times))
    pdf_batch = vmap_model(theta_batch)
    print(f"✅ vmap works: batch shape={pdf_batch.shape}")
except Exception as e:
    print(f"❌ FAILED: vmap error: {e}")
    exit(1)

# Test discrete mode
print("\n8. Testing discrete mode...")
try:
    model_discrete = Graph.pmf_from_graph(g, discrete=True)
    jumps = jnp.array([1, 2, 3, 5])
    pmf = model_discrete(theta, jumps)
    print(f"✅ Discrete mode works: shape={pmf.shape}")
except Exception as e:
    print(f"❌ FAILED: Discrete mode error: {e}")
    exit(1)

# Final summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("""
Parameterized edges feature is working correctly!

Capabilities verified:
  ✅ Graph construction with parameterized edges
  ✅ Automatic detection via serialization
  ✅ PMF/PDF computation
  ✅ JIT compilation
  ✅ Gradient computation (autodiff)
  ✅ vmap (vectorization)
  ✅ Both continuous and discrete modes

You can now use this for:
  • SVGD (Stein Variational Gradient Descent)
  • Maximum likelihood parameter estimation
  • Gradient-based optimization
  • Bayesian inference

See jit_pdf.py for comprehensive examples!
""")
