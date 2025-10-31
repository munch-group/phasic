"""Test if JIT-compiled loop fixes segfault with DEFAULT configuration"""

import numpy as np
from phasic import Graph
import jax.numpy as jnp

print("="*70)
print("TESTING JIT-COMPILED LOOP WITH DEFAULT CONFIG")
print("="*70)
print()

def create_simple_exponential_graph():
    graph = Graph(state_length=1, parameterized=True)
    v_start = graph.starting_vertex()
    v_transient = graph.find_or_create_vertex([1])
    v_absorb = graph.find_or_create_vertex([0])
    v_start.add_edge(v_transient, 1.0)
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])
    return graph

# Test with DEFAULT configuration (no parallel='none', no jit=False)
print("Test: SVGD with 2D rewards using DEFAULT config (pmap + JIT)")
print("  Previously this caused segfaults")
print()

graph = create_simple_exponential_graph()

# Generate 2D data
np.random.seed(42)
n_obs = 15
n_features = 2
observed_data = jnp.array([
    np.random.exponential(scale=0.5, size=n_features)
    for _ in range(n_obs)
])

rewards_2d = jnp.array([
    [1.0, 0.5],
    [2.0, 1.0],
    [0.5, 2.0],
    [1.5, 1.5]
])

print(f"  Observed data: {observed_data.shape}")
print(f"  Rewards: {rewards_2d.shape}")
print()

from phasic import SVGD
model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)

print("  Creating SVGD with DEFAULT configuration:")
print("    - parallel: auto (will use pmap with 8 devices)")
print("    - jit: True (default)")
print("    - n_particles: 20")
print("    - n_iterations: 10")
print()

try:
    svgd = SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        n_particles=20,
        n_iterations=10,
        learning_rate=0.01,
        regularization=0.0,
        verbose=True,
        rewards=rewards_2d
        # NOTE: NOT using parallel='none' or jit=False
        # This is the DEFAULT configuration that caused segfaults before
    )

    print("\n  ✓ SVGD created successfully")
    print("  Running optimize()...")

    svgd.optimize()

    print()
    print("="*70)
    print("✓✓✓ SUCCESS! JIT-compiled loop FIXED the segfault! ✓✓✓")
    print("="*70)
    print()
    print(f"  Final theta_mean: {svgd.theta_mean}")
    print(f"  Particles shape: {svgd.particles.shape}")
    print()
    print("The JIT-compiled loop keeps everything in compiled code,")
    print("avoiding Python/JIT boundary issues that caused segfaults.")

except Exception as e:
    print()
    print("="*70)
    print("✗ FAILED - Still getting errors")
    print("="*70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
