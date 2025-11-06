"""Debug SVGD segmentation fault"""

import numpy as np
from phasic import Graph, SVGD
import jax.numpy as jnp

print("Creating graph...")
graph = Graph(state_length=1, parameterized=True)
v_start = graph.starting_vertex()
v_transient = graph.find_or_create_vertex([1])
v_absorb = graph.find_or_create_vertex([0])

v_start.add_edge(v_transient, 1.0)
v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])

print("✓ Graph created")

# Test 1: SVGD with 1D data, no rewards
print("\nTest 1: SVGD with 1D data, no rewards")
observed_data = jnp.array([0.5, 1.0, 1.5, 2.0])
model_1d = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)

print("  Creating SVGD object...")
svgd = SVGD(
    model=model_1d,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=5,
    n_iterations=2,
    regularization=0.0,
    verbose=False,
    rewards=None
)
print("  ✓ SVGD created")

print("  Running optimize...")
svgd.optimize()
print(f"  ✓ SVGD optimize completed: theta_mean = {svgd.theta_mean}")

# Test 2: SVGD with 1D data, 1D rewards
print("\nTest 2: SVGD with 1D data, 1D rewards")
rewards_1d = jnp.array([1.0, 2.0, 0.5, 1.5])

print("  Creating SVGD object with 1D rewards...")
svgd = SVGD(
    model=model_1d,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=5,
    n_iterations=2,
    regularization=0.0,
    verbose=False,
    rewards=rewards_1d
)
print("  ✓ SVGD created")

print("  Running optimize...")
try:
    svgd.optimize()
    print(f"  ✓ SVGD optimize completed: theta_mean = {svgd.theta_mean}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: SVGD with 2D data, 2D rewards
print("\nTest 3: SVGD with 2D data, 2D rewards")
n_obs = 10
n_features = 2

observed_data_2d = jnp.array([
    [0.5, 1.0],
    [1.0, 1.5],
    [1.5, 2.0],
    [2.0, 2.5],
    [0.8, 1.2],
    [1.2, 1.8],
    [1.8, 2.2],
    [2.2, 2.8],
    [0.7, 1.1],
    [1.3, 1.9]
])

rewards_2d = jnp.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.5, 1.5],
    [1.5, 0.5]
])

print(f"  Observed data shape: {observed_data_2d.shape}")
print(f"  Rewards shape: {rewards_2d.shape}")

print("  Creating multivariate model...")
model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)
print("  ✓ Model created")

print("  Creating SVGD object with 2D rewards...")
try:
    svgd = SVGD(
        model=model_mv,
        observed_data=observed_data_2d,
        theta_dim=1,
        n_particles=5,
        n_iterations=2,
        regularization=0.0,
        verbose=False,
        rewards=rewards_2d
    )
    print("  ✓ SVGD created")

    print("  Running optimize...")
    svgd.optimize()
    print(f"  ✓ SVGD optimize completed: theta_mean = {svgd.theta_mean}")

except Exception as e:
    print(f"  ✗ FAILED during creation or optimize: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("If you see this message, no segfault occurred in SVGD tests")
print("="*70)
