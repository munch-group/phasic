#!/usr/bin/env python
"""
Test that SVGD notebook cells will work with the FFI fix
"""
import numpy as np
import jax
import jax.numpy as jnp

# Enable 64-bit precision (needed for notebook cells)
jax.config.update("jax_enable_x64", True)

from phasic import Graph
from phasic.ffi_wrappers import compute_pmf_ffi

# Build a simple parameterized model (like notebook cell 187)
print("Building parameterized rabbit model...")
graph = Graph(state_length=2)
initial_state = [2, 0]
graph.starting_vertex().add_edge(graph.find_or_create_vertex(initial_state), 1)

index = 1
while index < graph.vertices_length():
    vertex = graph.vertex_at(index)
    state = vertex.state()

    if state[0] > 0:
        child_state = [state[0] - 1, state[1] + 1]
        vertex.add_edge_parameterized(
            graph.find_or_create_vertex(child_state), 0, [state[0], 0, 0]
        )
        child_state = [0, state[1]]
        vertex.add_edge_parameterized(
            graph.find_or_create_vertex(child_state), 0, [0, 1]
        )

    if state[1] > 0:
        child_state = [state[0] + 1, state[1] - 1]
        vertex.add_edge_parameterized(
            graph.find_or_create_vertex(child_state), 0, [state[1], 0]
        )
        child_state = [state[0], 0]
        vertex.add_edge_parameterized(
            graph.find_or_create_vertex(child_state), 0, [0, 1]
        )

    index += 1

print(f"Model has {graph.vertices_length()} vertices")

# Generate synthetic data (like notebook cell 188)
print("\nGenerating synthetic data...")
true_params = np.array([0.5, 0.1])
np.random.seed(42)
observed_times = np.random.exponential(1.0 / true_params[0], size=10)
print(f"Observed times: {observed_times[:5]}...")

# Define log-likelihood (like notebook cell 189)
print("\nDefining log-likelihood function...")

def log_likelihood_fn(params):
    """Log-likelihood: log p(data|params)"""
    # THIS IS THE KEY FIX: graph.serialize() now returns dict
    # and compute_pmf_ffi() accepts both dict and string!
    structure_json = graph.serialize()  # Returns dict

    pdf_values = compute_pmf_ffi(
        structure_json,  # Dict input now works!
        params,
        observed_times,
        discrete=False,
        granularity=50
    )

    log_lik = jnp.sum(jnp.log(pdf_values + 1e-10))
    return log_lik

# Test log-likelihood evaluation
print("\nTesting log-likelihood evaluation...")
try:
    test_params = jnp.array([0.4, 0.08])
    log_lik = log_likelihood_fn(test_params)
    print(f"✅ SUCCESS: Log-likelihood = {log_lik:.4f}")

    # Test JIT compilation
    print("\nTesting JIT compilation...")
    jit_log_lik = jax.jit(log_likelihood_fn)
    log_lik_jit = jit_log_lik(test_params)
    print(f"✅ SUCCESS: JIT log-likelihood = {log_lik_jit:.4f}")

    # Test gradient computation
    print("\nTesting gradient computation...")
    grad_fn = jax.grad(log_likelihood_fn)
    gradient = grad_fn(test_params)
    print(f"✅ SUCCESS: Gradient = {gradient}")

    print("\n=== All SVGD notebook cell tests passed! ===")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
