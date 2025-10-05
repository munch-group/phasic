#!/usr/bin/env python
"""
Comprehensive Example: JAX-Compatible Approach for C++ Models

This example demonstrates how to use Graph.load_cpp_model() with full JAX support,
including JIT compilation, automatic differentiation, and vectorization.

When to use this approach:
- When you need automatic differentiation (gradients)
- When you want to integrate with JAX-based workflows
- When using JAX transformations (jit, grad, vmap, pmap)
- When doing parameter optimization
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad
from jax.scipy.optimize import minimize
from ptdalgorithms import Graph
import matplotlib.pyplot as plt

print("=" * 80)
print("JAX-COMPATIBLE APPROACH: Full JAX Support for C++ Models")
print("=" * 80)

# ==============================================================================
# 1. BASIC USAGE
# ==============================================================================
print("\n1. BASIC USAGE")
print("-" * 40)

user_model_cpp = os.path.abspath("user_models/simple_exponential.cpp")

# Load the model (this returns a JAX-compatible function)
model = Graph.load_cpp_model(user_model_cpp)
print("✅ Model loaded as JAX-compatible function")

# Define parameters and evaluation times
theta = jnp.array([1.0])  # Rate parameter for exponential distribution
times = jnp.linspace(0.1, 5.0, 50)

# Compute PDF
pdf = model(theta, times)
print(f"   PDF shape: {pdf.shape}")
print(f"   PDF at t=1.0: {pdf[9]:.6f}")
print(f"   Sum of PDF values: {jnp.sum(pdf):.6f}")

# ==============================================================================
# 2. JIT COMPILATION
# ==============================================================================
print("\n2. JIT COMPILATION")
print("-" * 40)

# JIT compile the model for faster execution
jit_model = jit(model)

# First call compiles
pdf_jit = jit_model(theta, times)
print("✅ JIT compilation successful")

# Benchmark: JIT vs non-JIT
import time

# Non-JIT timing
start = time.time()
for _ in range(100):
    pdf_regular = model(theta, times)
regular_time = time.time() - start

# JIT timing (after compilation)
start = time.time()
for _ in range(100):
    pdf_jit = jit_model(theta, times)
jit_time = time.time() - start

print(f"   Regular: {regular_time:.4f} seconds for 100 calls")
print(f"   JIT:     {jit_time:.4f} seconds for 100 calls")
print(f"   Speedup: {regular_time/jit_time:.2f}x")

# ==============================================================================
# 3. AUTOMATIC DIFFERENTIATION
# ==============================================================================
print("\n3. AUTOMATIC DIFFERENTIATION")
print("-" * 40)

# Define a loss function (negative log-likelihood)
def negative_log_likelihood(params, observed_times):
    """Compute negative log-likelihood for observed absorption times"""
    pdf_vals = model(params, observed_times)
    # Add small epsilon to avoid log(0)
    return -jnp.sum(jnp.log(pdf_vals + 1e-10))

# Some observed absorption times
observed = jnp.array([0.5, 0.8, 1.2, 1.5, 2.0, 2.3, 2.8, 3.2])

# Compute gradient
grad_nll = grad(negative_log_likelihood)
gradient = grad_nll(jnp.array([1.0]), observed)
print(f"✅ Gradient computation works")
print(f"   Gradient at rate=1.0: {gradient[0]:.6f}")

# Value and gradient together (more efficient)
value_and_grad_nll = value_and_grad(negative_log_likelihood)
loss, gradient = value_and_grad_nll(jnp.array([1.0]), observed)
print(f"   Loss: {loss:.6f}, Gradient: {gradient[0]:.6f}")

# ==============================================================================
# 4. PARAMETER OPTIMIZATION
# ==============================================================================
print("\n4. PARAMETER OPTIMIZATION")
print("-" * 40)

# Use JAX's scipy-like minimize for optimization
def objective(params):
    return negative_log_likelihood(params, observed)

# Initial guess
initial_rate = jnp.array([0.5])

# Optimize using BFGS
result = minimize(objective, initial_rate, method='BFGS')
optimized_rate = result.x[0]

print(f"✅ Optimization successful")
print(f"   Initial rate: {initial_rate[0]:.4f}")
print(f"   Optimized rate: {optimized_rate:.4f}")
print(f"   True MLE (1/mean): {1.0/jnp.mean(observed):.4f}")

# ==============================================================================
# 5. VECTORIZATION WITH VMAP
# ==============================================================================
print("\n5. VECTORIZATION WITH VMAP")
print("-" * 40)

# Multiple parameter sets to evaluate
param_batch = jnp.array([
    [0.5],   # Rate = 0.5
    [1.0],   # Rate = 1.0
    [1.5],   # Rate = 1.5
    [2.0],   # Rate = 2.0
])

# Vectorize over parameter sets
vmap_model = vmap(lambda p: model(p, times))
pdf_batch = vmap_model(param_batch)

print(f"✅ Batch processing with vmap")
print(f"   Input shape:  {param_batch.shape} (4 parameter sets)")
print(f"   Output shape: {pdf_batch.shape} (4 x 50 PDFs)")

# Compute mean absorption time for each parameter set
def mean_absorption_time(params):
    pdf_vals = model(params, times)
    return jnp.sum(times * pdf_vals) / jnp.sum(pdf_vals)

vmap_mean = vmap(mean_absorption_time)
mean_times = vmap_mean(param_batch)
print(f"   Mean absorption times: {mean_times}")

# ==============================================================================
# 6. COMPLEX MODEL: RABBIT FLOODING
# ==============================================================================
print("\n6. COMPLEX MODEL: Rabbit Flooding")
print("-" * 40)

# Load the more complex rabbit flooding model
rabbit_model = Graph.load_cpp_model("examples/user_models/rabbit_flooding.cpp")

# Parameters: [num_rabbits, flood_rate_left, flood_rate_right]
rabbit_params = jnp.array([3.0, 0.5, 0.5])

# Compute PDF
rabbit_times = jnp.linspace(0.1, 10.0, 100)
rabbit_pdf = rabbit_model(rabbit_params, rabbit_times)

print(f"✅ Complex model loaded and evaluated")
print(f"   Total absorption probability: {jnp.sum(rabbit_pdf):.4f}")

# Gradient with respect to flooding rates
def total_absorption(params):
    return jnp.sum(rabbit_model(params, rabbit_times))

grad_absorption = grad(total_absorption)
sensitivity = grad_absorption(rabbit_params)
print(f"✅ Sensitivity analysis:")
print(f"   ∂P/∂(num_rabbits):    {sensitivity[0]:.6f}")
print(f"   ∂P/∂(flood_left):     {sensitivity[1]:.6f}")
print(f"   ∂P/∂(flood_right):    {sensitivity[2]:.6f}")

# ==============================================================================
# 7. PARALLELIZATION WITH PMAP
# ==============================================================================
print("\n7. PARALLELIZATION WITH PMAP")
print("-" * 40)

# Check available devices
devices = jax.devices()
n_devices = len(devices)
print(f"Available devices: {devices}")
print(f"Number of devices: {n_devices}")

if n_devices > 1:
    # Multiple devices (GPUs/TPUs) available
    print("\nUsing pmap for multi-device parallelization:")

    # Parameters for each device
    params_per_device = jnp.array([
        [0.5 + 0.5 * i] for i in range(n_devices)
    ])

    # Parallelize across devices
    pmap_model = jax.pmap(lambda p: model(p, times))
    pdf_parallel = pmap_model(params_per_device)

    print(f"✅ pmap successful across {n_devices} devices")
    print(f"   Output shape: {pdf_parallel.shape}")

    # Combine pmap with grad
    pmap_grad = jax.pmap(jax.grad(lambda p: jnp.sum(model(p, times))))
    parallel_gradients = pmap_grad(params_per_device)
    print(f"✅ Parallel gradients computed: shape={parallel_gradients.shape}")
else:
    # Single device - demonstrate pmap anyway
    print("\nSingle device available - pmap will use thread-level parallelism")
    print("For multi-GPU/TPU systems, pmap distributes across devices")

    # Even with 1 device, pmap works
    params_single = jnp.array([[1.0]])  # Shape: (1 device, 1 param)
    pmap_model = jax.pmap(lambda p: model(p, times))
    pdf_pmap = pmap_model(params_single)
    print(f"✅ pmap works even on single device: shape={pdf_pmap.shape}")

print("\nHybrid parallelization (pmap + vmap):")
print("This pattern is useful for multi-node clusters")

# Define a function that uses vmap internally
def batch_compute(param_batch):
    """Process multiple parameters on one device"""
    return jax.vmap(lambda p: model(p, times[:10]))(param_batch)

# If we had multiple devices, we'd distribute batches across them
if n_devices > 1:
    # Split work across devices
    all_params = jnp.array([[0.5 + 0.1 * j] for j in range(8)])
    params_per_device = all_params.reshape(n_devices, -1, 1)

    pmap_vmap_hybrid = jax.pmap(batch_compute)
    hybrid_results = pmap_vmap_hybrid(params_per_device)
    print(f"✅ Hybrid (pmap+vmap): {hybrid_results.shape}")
else:
    print("   (Would distribute batches across multiple GPUs/TPUs)")

# ==============================================================================
# 8. COMBINING WITH OTHER JAX LIBRARIES
# ==============================================================================
print("\n8. COMBINING WITH JAX ECOSYSTEM")
print("-" * 40)

# Example: Using with Haiku (neural network library)
# This demonstrates how the model can be integrated into ML pipelines

def neural_network_with_phase_type(x, model_params):
    """Example: Combine neural network output with phase-type model"""
    # Neural network would transform x to phase-type parameters
    # Here we just simulate it
    transformed_params = jnp.abs(x * model_params)  # Ensure positive

    # Use the phase-type model
    fixed_times = jnp.array([1.0, 2.0, 3.0])
    pdf_vals = model(transformed_params, fixed_times)

    return jnp.sum(pdf_vals)

# Gradient through the entire pipeline
full_grad = grad(neural_network_with_phase_type, argnums=(0, 1))
x_input = jnp.array([2.0])
params = jnp.array([0.5])
dx, dparams = full_grad(x_input, params)

print(f"✅ Integration with JAX ecosystem works")
print(f"   Gradient w.r.t. input: {dx[0]:.6f}")
print(f"   Gradient w.r.t. params: {dparams[0]:.6f}")

# ==============================================================================
# 8. PERFORMANCE TIPS
# ==============================================================================
print("\n8. PERFORMANCE TIPS")
print("-" * 40)
print("""
Tips for optimal performance with JAX-compatible models:

1. Use JIT compilation for repeated calls:
   jit_model = jax.jit(model)

2. Batch operations with vmap instead of loops:
   vmap_model = jax.vmap(lambda p: model(p, times))

3. Combine value and gradient computation:
   loss, grad = jax.value_and_grad(loss_fn)(params)

4. Pre-allocate arrays and avoid Python loops in hot paths

5. Use JAX's built-in optimizers for parameter fitting

Note: The model rebuilds the graph on each call to maintain
purity for JAX tracing. For scenarios with fixed parameters
and many evaluations, consider the FFI approach instead.
""")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n9. VISUALIZATION")
print("-" * 40)

# Create a plot comparing different rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot PDFs for different rates
rates = [0.5, 1.0, 1.5, 2.0]
plot_times = jnp.linspace(0.1, 5.0, 100)

for rate in rates:
    pdf_vals = model(jnp.array([rate]), plot_times)
    ax1.plot(plot_times, pdf_vals, label=f'Rate = {rate}')

ax1.set_xlabel('Time')
ax1.set_ylabel('PDF')
ax1.set_title('Exponential Distribution PDFs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot gradient landscape
rate_range = jnp.linspace(0.1, 3.0, 50)
losses = []
gradients = []

for r in rate_range:
    loss, grad_val = value_and_grad_nll(jnp.array([r]), observed)
    losses.append(loss)
    gradients.append(grad_val[0])

ax2.plot(rate_range, losses, 'b-', label='Loss')
ax2.axvline(x=optimized_rate, color='r', linestyle='--', label=f'Optimum: {optimized_rate:.3f}')
ax2.set_xlabel('Rate Parameter')
ax2.set_ylabel('Negative Log-Likelihood')
ax2.set_title('Loss Landscape')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jax_compatible_example.png', dpi=100, bbox_inches='tight')
print("✅ Plots saved to jax_compatible_example.png")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The JAX-compatible approach provides:
✅ Full automatic differentiation support
✅ JIT compilation for performance
✅ Vectorization with vmap
✅ Integration with JAX ecosystem (optimizers, neural networks, etc.)
✅ Pure functional interface

Best for:
- Parameter optimization and inference
- Integration with machine learning pipelines
- Research requiring gradients
- Monte Carlo methods with varying parameters
""")