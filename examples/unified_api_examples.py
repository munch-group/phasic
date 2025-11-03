#!/usr/bin/env python
"""
Unified Edge API - Comprehensive Examples

This script demonstrates the new unified edge interface for both constant
and parameterized graphs, using manual construction and callback functions.

Author: Claude Code
Date: November 3, 2025
Version: 0.22.0 (post-refactoring)
"""

import numpy as np
from phasic import Graph

print("=" * 70)
print("UNIFIED EDGE API - COMPREHENSIVE EXAMPLES")
print("=" * 70)

# =============================================================================
# PART 1: NON-PARAMETERIZED (CONSTANT) GRAPHS
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: NON-PARAMETERIZED GRAPHS (Constant Edges)")
print("=" * 70)

# -----------------------------------------------------------------------------
# Example 1A: Manual Construction - Constant Graph
# -----------------------------------------------------------------------------

print("\n1A. Manual Construction - Erlang(2, λ=3) Distribution")
print("-" * 70)

# Create graph with 1-dimensional state
g_const_manual = Graph(state_length=1)

# Get vertices
v0 = g_const_manual.starting_vertex()
v1 = g_const_manual.find_or_create_vertex([1])
v2 = g_const_manual.find_or_create_vertex([0])  # Absorbing state

# Add edges with SCALAR weights (constant edges)
v0.add_edge(v1, 3.0)  # Rate = 3.0
v1.add_edge(v2, 3.0)  # Rate = 3.0

print(f"Graph properties:")
print(f"  - param_length: {g_const_manual.param_length()}")
print(f"  - is_parameterized: {g_const_manual.is_parameterized()}")
print(f"  - vertices: {g_const_manual.vertices_length()}")

# Compute PDF at t=1.0
t = 1.0
pdf = g_const_manual.pdf(t, granularity=200)
pdf_analytical = 9.0 * t * np.exp(-3.0 * t)  # Erlang(2, 3) formula

print(f"\nPDF at t={t}:")
print(f"  - Computed: {pdf:.6f}")
print(f"  - Analytical: {pdf_analytical:.6f}")
print(f"  - Error: {abs(pdf - pdf_analytical)/pdf_analytical:.2%}")

# -----------------------------------------------------------------------------
# Example 1B: Callback Construction - Constant Graph
# -----------------------------------------------------------------------------

print("\n1B. Callback Construction - Simple Coalescent Model")
print("-" * 70)

def coalescent_constant(state, nr_samples=4):
    """
    Non-parameterized coalescent model for n samples.

    Returns constant rates (not dependent on theta).
    """
    if not state.size:
        # Initial state: n samples, no coalescences
        initial_state = np.array([nr_samples] + [0] * nr_samples)
        return [(initial_state, 1.0)]  # TUPLE: (state, weight)
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                # Check if coalescence is possible
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue

                # New state after coalescence
                new_state = state.copy()
                new_state[i] -= 1
                new_state[j] -= 1
                new_state[i + j + 1] += 1

                # Coalescence rate (constant, not scaled by theta)
                rate = state[i] * (state[j] - same) / (1 + same)

                # Return TUPLE: (new_state, rate)
                transitions.append((new_state, rate))

        return transitions

# Build graph using callback
g_const_callback = Graph(callback=coalescent_constant, parameterized=False, nr_samples=4)

print(f"Graph properties:")
print(f"  - param_length: {g_const_callback.param_length()}")
print(f"  - is_parameterized: {g_const_callback.is_parameterized()}")
print(f"  - vertices: {g_const_callback.vertices_length()}")

# Compute expectation (time to most recent common ancestor)
expectation = g_const_callback.expectation()
print(f"\nExpectation (TMRCA for 4 samples): {expectation:.6f}")
print(f"  (Analytical: 1.5 coalescent time units)")

# =============================================================================
# PART 2: PARAMETERIZED GRAPHS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: PARAMETERIZED GRAPHS")
print("=" * 70)

# -----------------------------------------------------------------------------
# Example 2A: Manual Construction - Parameterized Graph
# -----------------------------------------------------------------------------

print("\n2A. Manual Construction - Linear Combination of Rates")
print("-" * 70)

# Create graph
g_param_manual = Graph(state_length=1)

v0 = g_param_manual.starting_vertex()
v1 = g_param_manual.find_or_create_vertex([1])
v2 = g_param_manual.find_or_create_vertex([0])

# Add edges with ARRAY coefficients (parameterized edges)
# Edge weight will be: coefficients[0]*theta[0] + coefficients[1]*theta[1]

v0.add_edge(v1, [2.0, 1.0])  # weight = 2*theta[0] + 1*theta[1]
v1.add_edge(v2, [1.0, 3.0])  # weight = 1*theta[0] + 3*theta[1]

print(f"Graph properties:")
print(f"  - param_length: {g_param_manual.param_length()}")
print(f"  - is_parameterized: {g_param_manual.is_parameterized()}")
print(f"  - vertices: {g_param_manual.vertices_length()}")

# Test with different theta values
print("\nEdge weights with different theta values:")

theta1 = [1.0, 1.0]
g_param_manual.update_weights(theta1)
w0 = v0.edges()[0].weight()
w1 = v1.edges()[0].weight()
print(f"  theta={theta1}: edge0={w0:.1f}, edge1={w1:.1f}")

theta2 = [2.0, 0.5]
g_param_manual.update_weights(theta2)
w0 = v0.edges()[0].weight()
w1 = v1.edges()[0].weight()
print(f"  theta={theta2}: edge0={w0:.1f}, edge1={w1:.1f}")

# Compute PDF
pdf = g_param_manual.pdf(1.0, granularity=200)
print(f"\nPDF at t=1.0 with theta={theta2}: {pdf:.6f}")

# -----------------------------------------------------------------------------
# Example 2B: Callback Construction - Parameterized Graph
# -----------------------------------------------------------------------------

print("\n2B. Callback Construction - Population-Scaled Coalescent")
print("-" * 70)

def coalescent_parameterized(state, nr_samples=4):
    """
    Parameterized coalescent model where theta scales the time.

    Coalescent theory:
    - Coalescent rate in coalescent units: k(k-1)/2
    - Scaling to real time: divide by 2*Ne
    - theta = 1/(2*Ne) converts coalescent time to generations

    Edge weight = [k(k-1)/2] * theta = k(k-1)/(4*Ne)

    Returns (new_state, weight, [coefficient]) where:
    - coefficient = k(k-1)/2 (coalescent rate in coalescent units)
    - weight = ignored (will be computed from coefficients)
    - final weight = coefficient * theta (rate in generations)
    """
    if not state.size:
        # Initial state: starting vertex edges use constant weight
        initial_state = np.array([nr_samples] + [0] * nr_samples)
        return [(initial_state, 1.0, [])]  # TUPLE: (state, weight, empty coefficients)
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue

                # New state after coalescence
                new_state = state.copy()
                new_state[i] -= 1
                new_state[j] -= 1
                new_state[i + j + 1] += 1

                # Coalescence rate in coalescent time units
                # This is the COEFFICIENT, not the final rate
                rate_coalescent = state[i] * (state[j] - same) / (1 + same)

                # Return TUPLE: (new_state, weight, [coefficients])
                # weight parameter is ignored, final weight = coefficient * theta
                transitions.append((new_state, 0.0, [rate_coalescent]))

        return transitions

# Build parameterized graph
g_param_callback = Graph(callback=coalescent_parameterized, parameterized=True, nr_samples=4)

print(f"Graph properties:")
print(f"  - param_length: {g_param_callback.param_length()}")
print(f"  - is_parameterized: {g_param_callback.is_parameterized()}")
print(f"  - vertices: {g_param_callback.vertices_length()}")

# Test with different theta values
print("\nExpectation (TMRCA) with different theta values:")

# theta = 1/(2*Ne) where Ne is effective population size
Ne_values = [1000, 10000, 100000]

for Ne in Ne_values:
    theta = [1.0 / (2.0 * Ne)]  # Single parameter
    g_param_callback.update_weights(theta)
    tmrca = g_param_callback.expectation()

    # Analytical expectation in coalescent units: 1.5
    # In generations: 1.5 * 2*Ne
    tmrca_analytical = 1.5 * 2.0 * Ne

    print(f"  Ne={Ne:6d}, theta={theta[0]:.2e}: TMRCA={tmrca:.1f} gen (analytical: {tmrca_analytical:.1f})")

# =============================================================================
# PART 3: MIXED USAGE PATTERNS
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: ADVANCED USAGE PATTERNS")
print("=" * 70)

# -----------------------------------------------------------------------------
# Example 3A: Switching Between Constant and Parameterized
# -----------------------------------------------------------------------------

print("\n3A. Creating Multiple Graphs with Different Parameterization")
print("-" * 70)

def flexible_callback(state, rate_type='constant'):
    """Callback that can create either constant or parameterized graphs"""
    if not state.size:
        if rate_type == 'constant':
            return [(np.array([1]), 1.0)]
        else:
            return [(np.array([1]), 1.0, [])]
    elif state[0] == 1:
        if rate_type == 'constant':
            # Return TUPLE: (state, weight) for constant
            return [(np.array([0]), 5.0)]
        else:
            # Return TUPLE: (state, weight, [coefficients]) for parameterized
            return [(np.array([0]), 0.0, [5.0])]
    else:
        return []

g_flex_const = Graph(callback=flexible_callback, parameterized=False, rate_type='constant')
g_flex_param = Graph(callback=flexible_callback, parameterized=True, rate_type='parameterized')

print(f"Constant graph:")
print(f"  - param_length: {g_flex_const.param_length()}")
print(f"  - PDF at t=1.0: {g_flex_const.pdf(1.0, granularity=200):.6f}")

print(f"\nParameterized graph:")
print(f"  - param_length: {g_flex_param.param_length()}")
g_flex_param.update_weights([1.0])
print(f"  - PDF at t=1.0 (theta=[1.0]): {g_flex_param.pdf(1.0, granularity=200):.6f}")

# -----------------------------------------------------------------------------
# Example 3B: Edge Validation (Type Safety)
# -----------------------------------------------------------------------------

print("\n3B. Edge Type Validation")
print("-" * 70)

print("Attempting to mix constant and parameterized edges...")

g_mixed = Graph(state_length=1)
v0 = g_mixed.starting_vertex()
v1 = g_mixed.find_or_create_vertex([1])
v2 = g_mixed.find_or_create_vertex([0])

# Add first edge as constant
v0.add_edge(v1, 3.0)
print(f"✓ Added constant edge (param_length={g_mixed.param_length()})")

# Try to add parameterized edge - should fail
try:
    v1.add_edge(v2, [1.0, 2.0])
    print("✗ ERROR: Should have rejected mixed types!")
except RuntimeError as e:
    print(f"✓ Validation works: {str(e)[:60]}...")

# =============================================================================
# PART 4: JAX INTEGRATION (Parameterized Graphs Only)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: JAX INTEGRATION (Parameterized Graphs)")
print("=" * 70)

try:
    import jax
    import jax.numpy as jnp

    print("\n4. JAX-Compatible Model from Parameterized Graph")
    print("-" * 70)

    # Use the manually constructed parameterized graph
    model = Graph.pmf_from_graph(g_param_manual, discrete=False)

    theta = jnp.array([2.0, 0.5])
    times = jnp.array([0.5, 1.0, 1.5])

    # Basic computation
    pdf = model(theta, times)
    print(f"PDF computed: shape={pdf.shape}")

    # JIT compilation
    jit_model = jax.jit(model)
    pdf_jit = jit_model(theta, times)
    print(f"✓ JIT compilation works")

    # Gradients
    def loss_fn(theta_val):
        return model(theta_val, times).sum()

    grad_fn = jax.grad(loss_fn)
    gradient = grad_fn(theta)
    print(f"✓ Gradients: ∇θ = [{gradient[0]:.3f}, {gradient[1]:.3f}]")

    # Vectorization
    theta_batch = jnp.array([[1.0, 1.0], [2.0, 0.5], [1.5, 1.5]])
    vmap_model = jax.vmap(lambda t: model(t, times))
    pdf_batch = vmap_model(theta_batch)
    print(f"✓ vmap works: batch shape={pdf_batch.shape}")

except ImportError:
    print("\n⚠️  JAX not available - skipping JAX integration examples")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY - UNIFIED EDGE API")
print("=" * 70)

print("""
KEY CONCEPTS:

1. CONSTANT (NON-PARAMETERIZED) GRAPHS:
   - Manual: vertex.add_edge(child, 3.0)  # SCALAR weight
   - Callback: return [(new_state, rate)]  # TUPLE (state, weight)
   - param_length = 1, is_parameterized = False
   - Edges have fixed weights

2. PARAMETERIZED GRAPHS:
   - Manual: vertex.add_edge(child, [2.0, 1.0])  # ARRAY coefficients
   - Callback: return [(new_state, 0.0, [coeff1, coeff2])]  # TUPLE (state, weight, coeffs)
   - param_length = len(coefficients), is_parameterized = True
   - Edge weight = dot(coefficients, theta)
   - Call update_weights(theta) to change parameters

3. TYPE SAFETY:
   - First add_edge() call locks the graph mode
   - Cannot mix scalar and array edges in same graph
   - Validation prevents programming errors

4. CALLBACK PATTERNS:
   ⚠️  IMPORTANT: Callbacks must return TUPLES, not lists!
   - Constant: [(np.array(state), weight)]
   - Parameterized: [(np.array(state), weight, [coefficients])]
   - Starting edges can use empty coefficients: [(state, 1.0, [])]
   - For parameterized: coefficients should be BASIS, theta is SCALE

5. JAX INTEGRATION:
   - Only for parameterized graphs (param_length > 1)
   - Full support: jit, grad, vmap
   - Use Graph.pmf_from_graph() to create JAX model

DEPRECATED API (still works with warnings):
   - vertex.add_edge_parameterized(child, weight, edge_state)
   - graph.update_parameterized_weights(theta)

NEW RECOMMENDED API:
   - vertex.add_edge(child, [coefficients])  # array for parameterized
   - vertex.add_edge(child, weight)          # scalar for constant
   - graph.update_weights(theta)
""")

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
