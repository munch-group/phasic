"""Basic test for multivariate phase-type distributions"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

def create_simple_exponential_graph():
    """Create a simple exponential distribution graph for testing"""
    graph = Graph(state_length=1, parameterized=True)
    v_start = graph.starting_vertex()
    v_transient = graph.find_or_create_vertex([1])
    v_absorb = graph.find_or_create_vertex([0])

    v_start.add_edge(v_transient, 1.0)
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

    return graph


print("="*70)
print("BASIC MULTIVARIATE TESTS")
print("="*70)
print()

# Test 1: Basic 2D rewards shape
print("Test 1: 2D rewards produce correct shapes")
graph = create_simple_exponential_graph()
model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.linspace(0.5, 2.5, 5)
rewards_2d = jnp.ones((4, 3))  # 4 vertices, 3 features

pmf, moments = model(theta, times, rewards=rewards_2d)

print(f"  PMF shape: {pmf.shape} (expected (5, 3))")
print(f"  Moments shape: {moments.shape} (expected (3, 2))")

assert pmf.shape == (5, 3)
assert moments.shape == (3, 2)
print("✓ Test 1 passed\n")

# Test 2: Independence of features
print("Test 2: Each feature computed independently")
model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

rewards_2d = jnp.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.5, 1.5],
    [1.5, 0.5]
])

pmf_2d, moments_2d = model(theta, times, rewards=rewards_2d)

# Check feature 0
pmf_0, moments_0 = model_1d(theta, times, rewards=rewards_2d[:, 0])
np.testing.assert_allclose(pmf_2d[:, 0], pmf_0, rtol=1e-10)
np.testing.assert_allclose(moments_2d[0, :], moments_0, rtol=1e-10)
print("  Feature 0 matches ✓")

# Check feature 1
pmf_1, moments_1 = model_1d(theta, times, rewards=rewards_2d[:, 1])
np.testing.assert_allclose(pmf_2d[:, 1], pmf_1, rtol=1e-10)
np.testing.assert_allclose(moments_2d[1, :], moments_1, rtol=1e-10)
print("  Feature 1 matches ✓")

print("✓ Test 2 passed\n")

# Test 3: SVGD accepts rewards
print("Test 3: SVGD accepts rewards parameter")
from phasic import SVGD

observed_data = jnp.array([0.5, 1.0, 1.5, 2.0])
rewards = jnp.array([1.0, 2.0, 0.5, 1.5])

svgd = SVGD(
    model=model_1d,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=10,
    n_iterations=2,  # Just 2 iterations
    regularization=0.0,
    verbose=False,
    rewards=rewards
)

assert svgd.rewards is not None
print(f"  SVGD stored rewards: shape {svgd.rewards.shape}")
print("✓ Test 3 passed\n")

print("="*70)
print("ALL BASIC TESTS PASSED ✓")
print("="*70)
