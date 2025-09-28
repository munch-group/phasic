#!/usr/bin/env python
"""
Side-by-Side Comparison: JAX-Compatible vs FFI Approaches

This example directly compares both approaches for loading C++ models,
helping you choose the right one for your use case.
"""

import numpy as np
import jax
import jax.numpy as jnp
from ptdalgorithms import Graph
import time
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPARISON: JAX-Compatible vs FFI Approaches")
print("=" * 80)

# Model file to use for comparison
model_file = "examples/user_models/simple_exponential.cpp"

# ==============================================================================
# APPROACH 1: JAX-COMPATIBLE
# ==============================================================================
print("\n" + "="*60)
print("APPROACH 1: JAX-COMPATIBLE (default)")
print("="*60)

# Load model with JAX support
jax_model = Graph.load_cpp_model(model_file, jax_compatible=True)
print("✅ Loaded JAX-compatible model")

# Test parameters and times
theta_jax = jnp.array([1.0])
times_jax = jnp.array([0.5, 1.0, 1.5, 2.0])

# --- Feature: Basic Evaluation ---
print("\n📊 Basic Evaluation:")
pdf_jax = jax_model(theta_jax, times_jax)
print(f"   PDF values: {pdf_jax}")

# --- Feature: JIT Compilation ---
print("\n⚡ JIT Compilation:")
jit_model = jax.jit(jax_model)
pdf_jit = jit_model(theta_jax, times_jax)
print(f"   ✅ JIT works: {jnp.allclose(pdf_jax, pdf_jit)}")

# --- Feature: Automatic Differentiation ---
print("\n🔢 Automatic Differentiation:")
grad_fn = jax.grad(lambda p: jnp.sum(jax_model(p, times_jax)))
gradient = grad_fn(theta_jax)
print(f"   ✅ Gradient: {gradient}")

# --- Feature: Vectorization ---
print("\n📦 Vectorization (vmap):")
batch_params = jnp.array([[0.5], [1.0], [1.5], [2.0]])
vmap_model = jax.vmap(lambda p: jax_model(p, times_jax))
batch_results = vmap_model(batch_params)
print(f"   ✅ Batch shape: {batch_results.shape}")

# --- Performance Test ---
print("\n⏱️  Performance (1000 evaluations):")
start = time.time()
for _ in range(1000):
    _ = jax_model(theta_jax, times_jax)
jax_time = time.time() - start
print(f"   Time: {jax_time:.4f} seconds")
print(f"   Note: Rebuilds graph each call")

# ==============================================================================
# APPROACH 2: FFI (Foreign Function Interface)
# ==============================================================================
print("\n" + "="*60)
print("APPROACH 2: FFI (use_ffi=True)")
print("="*60)

# Load model with FFI
ffi_builder = Graph.load_cpp_model(model_file, use_ffi=True)
print("✅ Loaded FFI builder function")

# Build graph once
theta_np = np.array([1.0])
graph = ffi_builder(theta_np)
print(f"✅ Built graph: {type(graph)}")

# --- Feature: Basic Evaluation ---
print("\n📊 Basic Evaluation:")
pdf_ffi = [graph.pdf(t, 100) for t in [0.5, 1.0, 1.5, 2.0]]
print(f"   PDF values: {pdf_ffi}")

# --- Feature: JIT Compilation ---
print("\n⚡ JIT Compilation:")
print("   ❌ Not directly JIT-compatible (graph is C++ object)")

# --- Feature: Automatic Differentiation ---
print("\n🔢 Automatic Differentiation:")
print("   ❌ Not directly differentiable (graph is C++ object)")

# --- Feature: Vectorization ---
print("\n📦 Vectorization:")
print("   ⚠️  Manual: Build multiple graphs")
graphs = [ffi_builder(np.array([p])) for p in [0.5, 1.0, 1.5, 2.0]]
print(f"   Built {len(graphs)} graphs for different parameters")

# --- Feature: Graph Reuse ---
print("\n♻️  Graph Reuse (unique to FFI):")
print("   ✅ Can reuse same graph without rebuilding:")
for t in [0.3, 0.7, 1.2, 1.8]:
    pdf = graph.pdf(t, 100)
    print(f"      t={t}: {pdf:.6f}")

# --- Performance Test ---
print("\n⏱️  Performance (1000 evaluations):")
start = time.time()
for _ in range(1000):
    _ = [graph.pdf(t, 100) for t in [0.5, 1.0, 1.5, 2.0]]
ffi_time = time.time() - start
print(f"   Time: {ffi_time:.4f} seconds")
print(f"   Note: Graph built once, reused 1000 times")
print(f"   Speedup: {jax_time/ffi_time:.1f}x faster than JAX approach")

# ==============================================================================
# FEATURE COMPARISON TABLE
# ==============================================================================
print("\n" + "="*80)
print("FEATURE COMPARISON TABLE")
print("="*80)

comparison_table = """
┌─────────────────────────┬──────────────────┬──────────────────┐
│ Feature                 │ JAX-Compatible   │ FFI Approach     │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Load function          │ load_cpp_model() │ load_cpp_model(  │
│                        │                  │   use_ffi=True)  │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Returns                │ Function         │ Builder function │
│                        │ (theta,times)→pdf│ (theta)→Graph    │
├─────────────────────────┼──────────────────┼──────────────────┤
│ JAX JIT                │ ✅ Full support  │ ❌ Not available │
├─────────────────────────┼──────────────────┼──────────────────┤
│ JAX grad               │ ✅ Full support  │ ❌ Not available │
├─────────────────────────┼──────────────────┼──────────────────┤
│ JAX vmap               │ ✅ Full support  │ ❌ Manual only   │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Graph reuse            │ ❌ Rebuilds      │ ✅ Build once    │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Performance (fixed θ)   │ Slower          │ ✅ Much faster   │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Performance (varying θ) │ ✅ With JIT     │ Moderate         │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Memory usage           │ Higher          │ ✅ Lower         │
├─────────────────────────┼──────────────────┼──────────────────┤
│ Integration with JAX   │ ✅ Seamless     │ ⚠️ Limited       │
└─────────────────────────┴──────────────────┴──────────────────┘
"""
print(comparison_table)

# ==============================================================================
# USE CASE RECOMMENDATIONS
# ==============================================================================
print("USE CASE RECOMMENDATIONS")
print("="*80)

use_cases = """
WHEN TO USE JAX-COMPATIBLE (default):
--------------------------------------
✅ Parameter optimization (needs gradients)
✅ Integration with JAX/ML libraries
✅ Research requiring automatic differentiation
✅ Bayesian inference with JAX-based MCMC
✅ Neural network integration
✅ When parameters change frequently

Example:
    model = Graph.load_cpp_model("model.cpp")
    optimizer = optax.adam(learning_rate=0.01)

    def loss(params):
        return -jnp.sum(model(params, observed_times))

    params = optimize(loss, initial_params, optimizer)


WHEN TO USE FFI APPROACH:
-------------------------
✅ Monte Carlo with fixed parameters
✅ Real-time systems (low latency required)
✅ Production systems with known parameters
✅ Likelihood evaluation in classical MCMC
✅ Large-scale simulations
✅ When evaluating same model many times

Example:
    builder = Graph.load_cpp_model("model.cpp", use_ffi=True)
    graph = builder(fixed_parameters)

    for _ in range(1000000):
        t = sample_time()
        pdf = graph.pdf(t, 100)
        # Process result
"""
print(use_cases)

# ==============================================================================
# PRACTICAL EXAMPLE: PARAMETER ESTIMATION
# ==============================================================================
print("\nPRACTICAL EXAMPLE: Parameter Estimation")
print("="*80)

# Generate some synthetic data
np.random.seed(42)
true_rate = 1.5
observed_times = np.random.exponential(1/true_rate, 20)
print(f"Generated 20 samples from exponential(rate={true_rate})")

# --- JAX Approach: Gradient-based optimization ---
print("\nJAX Approach (Gradient-based):")
print("-" * 40)

def neg_log_likelihood(rate_param, obs_times):
    """Negative log-likelihood for exponential distribution"""
    pdf_vals = jax_model(rate_param, obs_times)
    return -jnp.sum(jnp.log(pdf_vals + 1e-10))

# Optimize using gradient descent
learning_rate = 0.1
rate_est = jnp.array([0.5])  # Initial guess

for i in range(20):
    loss, grad = jax.value_and_grad(neg_log_likelihood)(rate_est, observed_times)
    rate_est = rate_est - learning_rate * grad
    if i % 5 == 0:
        print(f"  Iter {i:2d}: rate={rate_est[0]:.4f}, loss={loss:.4f}")

print(f"✅ Final estimate: {rate_est[0]:.4f} (true: {true_rate})")

# --- FFI Approach: Grid search ---
print("\nFFI Approach (Grid search):")
print("-" * 40)

# Grid of possible rates
rate_grid = np.linspace(0.5, 3.0, 50)
best_likelihood = -np.inf
best_rate = None

for rate in rate_grid:
    # Build graph for this rate
    g = ffi_builder(np.array([rate]))

    # Compute likelihood
    log_lik = 0
    for t in observed_times:
        pdf = g.pdf(t, 100)
        log_lik += np.log(pdf + 1e-10)

    if log_lik > best_likelihood:
        best_likelihood = log_lik
        best_rate = rate

print(f"✅ Best rate: {best_rate:.4f} (true: {true_rate})")
print(f"   Log-likelihood: {best_likelihood:.4f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\nGenerating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Performance comparison
ax1 = axes[0, 0]
methods = ['JAX\n(1000 calls)', 'FFI\n(1000 calls)']
times = [jax_time, ffi_time]
colors = ['blue', 'green']
bars = ax1.bar(methods, times, color=colors)
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Performance Comparison')
for bar, t in zip(bars, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{t:.3f}s', ha='center')

# Plot 2: Feature support
ax2 = axes[0, 1]
features = ['JIT', 'Grad', 'Vmap', 'Reuse']
jax_support = [1, 1, 1, 0]
ffi_support = [0, 0, 0, 1]
x = np.arange(len(features))
width = 0.35
ax2.bar(x - width/2, jax_support, width, label='JAX', color='blue')
ax2.bar(x + width/2, ffi_support, width, label='FFI', color='green')
ax2.set_xticks(x)
ax2.set_xticklabels(features)
ax2.set_ylabel('Support (1=Yes, 0=No)')
ax2.set_title('Feature Support')
ax2.legend()
ax2.set_ylim([0, 1.2])

# Plot 3: Optimization paths
ax3 = axes[1, 0]
# This would show the optimization path if we tracked it
ax3.text(0.5, 0.5, 'JAX: Gradient-based\nconvergence in 20 steps\n\n' +
         'FFI: Grid search\nover 50 points',
         ha='center', va='center', fontsize=12)
ax3.set_title('Optimization Approaches')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.axis('off')

# Plot 4: Use case matrix
ax4 = axes[1, 1]
use_case_data = np.array([
    [1, 0],  # ML integration
    [1, 0],  # Gradients needed
    [0, 1],  # Fixed parameters
    [0, 1],  # Real-time
    [1, 0],  # Research
    [0, 1],  # Production
])
im = ax4.imshow(use_case_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['JAX', 'FFI'])
ax4.set_yticks(range(6))
ax4.set_yticklabels(['ML Integration', 'Need Gradients', 'Fixed Params',
                     'Real-time', 'Research', 'Production'])
ax4.set_title('Best Choice by Use Case')
for i in range(6):
    for j in range(2):
        text = ax4.text(j, i, '✓' if use_case_data[i, j] else '',
                       ha='center', va='center', color='white', fontsize=16)

plt.tight_layout()
plt.savefig('approach_comparison.png', dpi=100, bbox_inches='tight')
print("✅ Comparison plots saved to approach_comparison.png")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("QUICK DECISION GUIDE")
print("="*80)
print("""
Choose JAX-COMPATIBLE when you need:
  • Automatic differentiation (gradients)
  • JAX ecosystem integration
  • Research and experimentation

Choose FFI when you need:
  • Maximum performance with fixed parameters
  • Real-time/production systems
  • Large-scale simulations

Both approaches use the SAME C++ model files!
You can switch between them by changing one parameter.
""")