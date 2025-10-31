#!/usr/bin/env python3
"""Quick SVGD test with separate graph instances"""

import numpy as np
from phasic import Graph, SVGD, configure, ExponentialDecayStepSize

configure(ffi=True, openmp=True, jit=True)

def build_exponential_graph():
    """Build simple exponential distribution"""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v0 = g.find_or_create_vertex([0])
    start.add_edge(v1, 1.0)
    v1.add_edge_parameterized(v0, 0.0, [1.0])
    return g

# Generate data with one graph instance
print("Generating data...")
graph_for_data = build_exponential_graph()
graph_for_data.update_parameterized_weights(np.array([2.0]))
data = np.array(graph_for_data.sample(1000))
print(f"Data: mean={np.mean(data):.3f}, std={np.std(data):.3f}")

# Run SVGD with fresh graph instance (no update_parameterized_weights called)
print("\nRunning SVGD with fresh graph...")
graph_for_svgd = build_exponential_graph()

model = Graph.pmf_and_moments_from_graph(graph_for_svgd, nr_moments=2, discrete=False, param_length=1)

step_schedule = ExponentialDecayStepSize(first_step=0.01, last_step=0.001, tau=500.0)

svgd = SVGD(
    model=model,
    observed_data=data,
    theta_dim=1,
    n_particles=20,
    n_iterations=500,
    learning_rate=step_schedule,
    positive_params=True,
    regularization=0.0,
    nr_moments=2,
    seed=42,
    verbose=False
)

svgd.optimize()
results = svgd.get_results()

true_theta = 2.0
theta_mean = results['theta_mean'][0]
error_pct = abs(theta_mean - true_theta) / true_theta * 100

print(f"\nResults:")
print(f"  True θ:      {true_theta:.3f}")
print(f"  SVGD θ_mean: {theta_mean:.3f}")
print(f"  Error:       {error_pct:.1f}%")

if error_pct <= 10:
    print("  ✓ Much better! (< 10% error)")
else:
    print(f"  ✗ Still high error")
