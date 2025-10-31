"""Debug segmentation fault in multivariate tests"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("Creating graph...")
graph = Graph(state_length=1, parameterized=True)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

print("✓ Graph created")

# Test 1: 1D model works
print("\nTest 1: 1D model")
model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0])
rewards_1d = jnp.array([1.0, 2.0, 0.5, 1.5])

print("  Calling 1D model...")
pmf, moments = model_1d(theta, times, rewards=rewards_1d)
print(f"  ✓ 1D model works: PMF shape {pmf.shape}, moments shape {moments.shape}")

# Test 2: Multivariate model with 1D rewards
print("\nTest 2: Multivariate model with 1D rewards")
model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

print("  Calling multivariate model with 1D rewards...")
pmf, moments = model_mv(theta, times, rewards=rewards_1d)
print(f"  ✓ Multivariate with 1D rewards works: PMF shape {pmf.shape}, moments shape {moments.shape}")

# Test 3: Multivariate model with None rewards
print("\nTest 3: Multivariate model with None rewards")
print("  Calling multivariate model with None rewards...")
pmf, moments = model_mv(theta, times, rewards=None)
print(f"  ✓ Multivariate with None rewards works: PMF shape {pmf.shape}, moments shape {moments.shape}")

# Test 4: Multivariate model with 2D rewards (simple)
print("\nTest 4: Multivariate model with 2D rewards (2 features)")
rewards_2d = jnp.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.5, 1.5],
    [1.5, 0.5]
])

print(f"  Rewards shape: {rewards_2d.shape}")
print(f"  Times shape: {times.shape}")

print("  Calling multivariate model with 2D rewards...")
try:
    pmf, moments = model_mv(theta, times, rewards=rewards_2d)
    print(f"  ✓ Multivariate with 2D rewards works: PMF shape {pmf.shape}, moments shape {moments.shape}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("If you see this message, no segfault occurred in basic tests")
print("="*70)
