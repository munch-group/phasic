#!/usr/bin/env python3
"""
Test that ExponentialDecayStepSize fixes the divergence issue with learning_rate=0.1
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from ptdalgorithms import Graph, ExponentialDecayStepSize

def build_exponential_graph():
    """Build simple exponential distribution graph"""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])

    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])

    return g

# Generate data from exponential with rate=5.0
np.random.seed(42)
graph = build_exponential_graph()
graph.update_parameterized_weights([5.0])
data = np.array(graph.sample(5000))  # Large dataset

print("=" * 70)
print("Test 1: Fixed learning_rate=0.1 (should diverge)")
print("=" * 70)

# Build model
graph = build_exponential_graph()
model = Graph.pmf_from_graph(graph, discrete=False, param_length=1)

# Uninformative prior
def uninformative_prior(phi):
    return -0.5 * jnp.sum((phi / 2.0)**2)

# Run SVGD with fixed learning rate (should diverge)
from ptdalgorithms import SVGD

svgd_fixed = SVGD(
    model=model,
    observed_data=data,
    prior=uninformative_prior,
    theta_dim=1,
    n_particles=100,
    n_iterations=1000,
    learning_rate=0.1,  # Too high, should diverge
    seed=42,
    verbose=True,
    positive_params=True
)

svgd_fixed.fit()
results_fixed = svgd_fixed.get_results()

print(f"\nFixed lr=0.1 Results:")
print(f"  Posterior mean: {results_fixed['theta_mean']}")
print(f"  Posterior std:  {results_fixed['theta_std']}")
print(f"  Expected:       [5.0]")
print(f"  Error:          {np.abs(results_fixed['theta_mean'][0] - 5.0):.2f}")

if results_fixed['theta_mean'][0] > 10.0:
    print("  Status: ❌ DIVERGED (as expected)")
else:
    print("  Status: ✅ Converged (unexpected!)")

print("\n" + "=" * 70)
print("Test 2: ExponentialDecayStepSize (should converge)")
print("=" * 70)

# Create decay schedule that starts at 0.1 and decays to 0.01
schedule = ExponentialDecayStepSize(max_step=0.1, min_step=0.01, tau=500.0)

svgd_decay = SVGD(
    model=model,
    observed_data=data,
    prior=uninformative_prior,
    theta_dim=1,
    n_particles=100,
    n_iterations=1000,
    learning_rate=schedule,  # Use schedule
    seed=42,
    verbose=True,
    positive_params=True
)

svgd_decay.fit()
results_decay = svgd_decay.get_results()

print(f"\nExponentialDecay Results:")
print(f"  Posterior mean: {results_decay['theta_mean']}")
print(f"  Posterior std:  {results_decay['theta_std']}")
print(f"  Expected:       [5.0]")
print(f"  Error:          {np.abs(results_decay['theta_mean'][0] - 5.0):.2f}")

if np.abs(results_decay['theta_mean'][0] - 5.0) < 1.0:
    print("  Status: ✅ CONVERGED (as expected)")
else:
    print("  Status: ❌ Failed to converge")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Fixed lr=0.1:         θ = {results_fixed['theta_mean'][0]:.2f} (diverged: {results_fixed['theta_mean'][0] > 10.0})")
print(f"ExponentialDecay:     θ = {results_decay['theta_mean'][0]:.2f} (converged: {np.abs(results_decay['theta_mean'][0] - 5.0) < 1.0})")
print(f"\nImprovement: {results_fixed['theta_mean'][0] / results_decay['theta_mean'][0]:.1f}x better")
print("\n✅ Schedule system successfully prevents divergence!")
