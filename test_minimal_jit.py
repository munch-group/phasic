"""Minimal test to isolate segfault"""

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
print("✓ Graph created\n")

print("Creating multivariate model...")
model_mv = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2, discrete=False)
print("✓ Model created\n")

print("Testing model with 2D rewards...")
theta = jnp.array([2.0])
times = jnp.array([0.5, 1.0])
rewards_2d = jnp.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.5, 1.5],
    [1.5, 0.5]
])

print("Calling model...")
pmf, moments = model_mv(theta, times, rewards=rewards_2d)
print(f"✓ Model returned: PMF {pmf.shape}, moments {moments.shape}\n")

print("Creating SVGD...")
np.random.seed(42)
observed_data = jnp.array([[0.5, 1.0], [1.0, 1.5], [1.5, 2.0]])

from phasic import SVGD

svgd = SVGD(
    model=model_mv,
    observed_data=observed_data,
    theta_dim=1,
    n_particles=8,
    n_iterations=5,
    regularization=0.0,
    verbose=True,
    rewards=rewards_2d
)
print("✓ SVGD created\n")

print("Running optimize...")
svgd.optimize()
print(f"\n✓✓✓ SUCCESS! theta_mean = {svgd.theta_mean}")
