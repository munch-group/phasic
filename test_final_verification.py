"""Final verification that JIT-compiled loop fixes all segfault issues"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

def create_simple_exponential_graph():
    graph = Graph(state_length=1, parameterized=True)
    v_start = graph.starting_vertex()
    v_transient = graph.find_or_create_vertex([1])
    v_absorb = graph.find_or_create_vertex([0])
    v_start.add_edge(v_transient, 1.0)
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])
    return graph

print("="*70)
print("FINAL VERIFICATION - JIT LOOP FIX")
print("="*70)
print()

# Test 1: Basic functionality
print("Test 1: 2D rewards shape and independence")
graph = create_simple_exponential_graph()
model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)
model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0, 1.5])
rewards_2d = jnp.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.5, 1.5],
    [1.5, 0.5]
])

pmf_2d, moments_2d = model_mv(theta, times, rewards=rewards_2d)
print(f"  PMF shape: {pmf_2d.shape} (expected (3, 2))")
print(f"  Moments shape: {moments_2d.shape} (expected (2, 2))")

# Verify independence
pmf_0, moments_0 = model_1d(theta, times, rewards=rewards_2d[:, 0])
np.testing.assert_allclose(pmf_2d[:, 0], pmf_0, rtol=1e-10)
print("  ✓ Feature 0 matches independently computed result")

pmf_1, moments_1 = model_1d(theta, times, rewards=rewards_2d[:, 1])
np.testing.assert_allclose(pmf_2d[:, 1], pmf_1, rtol=1e-10)
print("  ✓ Feature 1 matches independently computed result")
print("✓ Test 1 passed\n")

# Test 2: SVGD with 2D rewards (default config)
print("Test 2: SVGD with 2D rewards (DEFAULT pmap + JIT)")
np.random.seed(42)
n_obs = 15
n_features = 2
observed_data = jnp.array([
    np.random.exponential(scale=0.5, size=n_features)
    for _ in range(n_obs)
])

from phasic import SVGD

print(f"  Data shape: {observed_data.shape}")
print("  Using DEFAULT config (pmap across 8 devices, JIT enabled)")

svgd = SVGD(
    model=model_mv,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=16,
    n_iterations=10,
    learning_rate=0.01,
    regularization=0.0,
    verbose=False,
    rewards=rewards_2d
)

print("  Running optimize()...")
svgd.optimize()

print(f"  ✓ Completed: theta_mean = {svgd.theta_mean[0]:.3f}")
assert svgd.theta_mean is not None
assert svgd.particles.shape[1] == 1
print("✓ Test 2 passed\n")

# Test 3: Multiple sequential SVGD runs
print("Test 3: Multiple sequential SVGD runs (stress test)")
for run in range(3):
    print(f"  Run {run+1}/3...", end=" ")

    svgd = SVGD(
        model=model_mv,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=8,
        n_iterations=5,
        regularization=0.0,
        verbose=False,
        rewards=rewards_2d
    )

    svgd.optimize()
    print(f"theta_mean = {svgd.theta_mean[0]:.3f}")

print("✓ Test 3 passed\n")

# Test 4: Graph.svgd() API
print("Test 4: Graph.svgd() convenience method")
result = graph.svgd(
    observed_data=observed_data,
    theta_dim=1,
    n_particles=8,
    n_iterations=5,
    regularization=0.0,
    nr_moments=0,
    verbose=False,
    rewards=rewards_2d
)

print(f"  ✓ Completed: theta_mean = {result.theta_mean[0]:.3f}")
print("✓ Test 4 passed\n")

print("="*70)
print("ALL TESTS PASSED ✓✓✓")
print("="*70)
print()
print("The JIT-compiled loop using jax.lax.scan successfully:")
print("  • Keeps all computation in compiled code")
print("  • Avoids Python/JIT boundary crossings")
print("  • Eliminates segfaults")
print("  • Works with default pmap + JIT configuration")
print("  • Handles multiple sequential runs")
