#!/usr/bin/env python
"""
Simple Example: How to Load and Use C++ Phase-Type Models

This is a minimal example showing the basic usage of Graph.load_cpp_model()

Two approaches are available:
1. JAX-compatible (default): Full JAX support, rebuilds graph each call
2. FFI approach (use_ffi=True): Build once, reuse many times

See approach_comparison.py for detailed comparison.
"""

import numpy as np
import jax
import jax.numpy as jnp
from ptdalgorithms import Graph

print("=" * 70)
print("SIMPLE EXAMPLE: Using Graph.load_cpp_model()")
print("=" * 70)

# ============================================================================
# STEP 1: Load a C++ model
# ============================================================================
print("\n1. Loading the C++ model:")
print("-" * 40)

# Load an exponential distribution model
model = Graph.load_cpp_model("examples/user_models/simple_exponential.cpp")
print("✅ Model loaded successfully")

# ============================================================================
# STEP 2: Evaluate the model
# ============================================================================
print("\n2. Computing PDF values:")
print("-" * 40)

# Parameters for the model (check the C++ file for parameter meanings)
theta = jnp.array([1.0])  # Rate parameter = 1.0

# Time points where we want to evaluate the PDF
times = jnp.array([0.5, 1.0, 1.5, 2.0, 3.0])

# Compute the PDF
pdf = model(theta, times)

print(f"Parameters: rate = {theta[0]}")
print(f"Times: {times}")
print(f"PDF values: {pdf}")

# For exponential distribution with rate λ, PDF = λ * exp(-λ*t)
expected = theta[0] * jnp.exp(-theta[0] * times)
print(f"Expected (theoretical): {expected}")
print(f"Match: {jnp.allclose(pdf, expected, rtol=0.01)}")

# ============================================================================
# STEP 3: Use with JAX features
# ============================================================================
print("\n3. JAX Features:")
print("-" * 40)

# JIT compilation for speed
jit_model = jax.jit(model)
pdf_jit = jit_model(theta, times)
print(f"✅ JIT compilation works")

# Compute gradients
def loss(params):
    """Simple loss function: sum of PDF values"""
    return jnp.sum(model(params, times))

grad_fn = jax.grad(loss)
gradient = grad_fn(theta)
print(f"✅ Gradient computation works: grad = {gradient}")

# Batch processing with vmap
theta_batch = jnp.array([[0.5], [1.0], [2.0]])
batch_model = jax.vmap(lambda p: model(p, times))
pdf_batch = batch_model(theta_batch)
print(f"✅ Batch processing works: shape = {pdf_batch.shape}")

# ============================================================================
# STEP 4: Load a more complex model
# ============================================================================
print("\n4. Complex Model Example:")
print("-" * 40)

# Load the rabbit flooding model
rabbit_model = Graph.load_cpp_model("examples/user_models/rabbit_flooding.cpp")

# Parameters: [number_of_rabbits, flood_rate_left, flood_rate_right]
rabbit_params = jnp.array([3.0, 0.5, 0.5])

# Compute PDF at different times
rabbit_times = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
rabbit_pdf = rabbit_model(rabbit_params, rabbit_times)

print(f"Rabbit model parameters:")
print(f"  - Starting rabbits: {int(rabbit_params[0])}")
print(f"  - Flood rate (left): {rabbit_params[1]}")
print(f"  - Flood rate (right): {rabbit_params[2]}")
print(f"PDF values: {rabbit_pdf}")
print(f"Total absorption probability: {jnp.sum(rabbit_pdf):.4f}")

# ============================================================================
# STEP 5: FFI Approach (Alternative)
# ============================================================================
print("\n5. FFI Approach (Build Once, Use Many Times):")
print("-" * 40)

# Load with FFI approach
builder = Graph.load_cpp_model("examples/user_models/simple_exponential.cpp", use_ffi=True)
graph = builder(np.array([1.0]))  # Build graph once

# Use the same graph multiple times
print("Using the same graph without rebuilding:")
for t in [0.5, 1.0, 1.5]:
    pdf = graph.pdf(t, 100)
    print(f"  PDF at t={t}: {pdf:.6f}")

print("✅ FFI approach: More efficient for repeated evaluations")

# ============================================================================
# QUICK REFERENCE
# ============================================================================
print("\n" + "=" * 70)
print("QUICK REFERENCE")
print("=" * 70)
print("""
# APPROACH 1: JAX-COMPATIBLE (default)
# Best for: gradients, JAX integration, research
model = Graph.load_cpp_model("path/to/model.cpp")
pdf = model(parameters, times)  # Returns JAX array
jit_model = jax.jit(model)                           # JIT compilation
gradient = jax.grad(lambda p: model(p, times))(p)    # Gradients
batch_pdf = jax.vmap(lambda p: model(p, times))(batch_params)  # Batching

# APPROACH 2: FFI (Foreign Function Interface)
# Best for: performance, fixed parameters, production
builder = Graph.load_cpp_model("path/to/model.cpp", use_ffi=True)
graph = builder(parameters)     # Build graph once
pdf1 = graph.pdf(t1, 100)       # Use many times
pdf2 = graph.pdf(t2, 100)       # No rebuild!

# Available example models:
- examples/user_models/simple_exponential.cpp  # Exponential distribution
- examples/user_models/erlang_distribution.cpp  # Erlang distribution
- examples/user_models/birth_death_process.cpp  # Birth-death process
- examples/user_models/mm1_queue.cpp           # M/M/1 queue
- examples/user_models/rabbit_flooding.cpp     # Rabbit flooding simulation

# For detailed comparison, see:
- examples/approach_comparison.py      # Side-by-side comparison
- examples/jax_compatible_example.py   # JAX approach details
- examples/ffi_approach_example.py     # FFI approach details
""")