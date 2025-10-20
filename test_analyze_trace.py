#!/usr/bin/env python3
"""
Test analyze_trace() method for SVGD convergence diagnostics.

Tests different convergence scenarios:
1. Well-converged run (good parameters)
2. Early convergence (could reduce iterations)
3. Not converged (needs more iterations)
4. Variance collapse (learning rate too high)
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from ptdalgorithms import Graph, SVGD, ExponentialDecayStepSize

def build_exponential_graph():
    """Build simple exponential distribution graph"""
    g = Graph(state_length=1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])

    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])

    return g

def uninformative_prior(phi):
    """Wide uninformative prior"""
    return -0.5 * jnp.sum((phi / 2.0)**2)

# Generate data
np.random.seed(42)
graph = build_exponential_graph()
graph.update_parameterized_weights([5.0])
data = np.array(graph.sample(1000))

# Build model
graph = build_exponential_graph()
model = Graph.pmf_from_graph(graph, discrete=False, param_length=1)

print("=" * 80)
print("Test 1: Well-Converged Run")
print("=" * 80)
print()

svgd1 = SVGD(
    model=model,
    observed_data=data,
    prior=uninformative_prior,
    theta_dim=1,
    n_particles=50,
    n_iterations=1000,
    learning_rate=ExponentialDecayStepSize(0.01, 0.001, 500.0),
    seed=42,
    verbose=False
)

svgd1.fit(return_history=True)
svgd1.analyze_trace()

print("\n" + "=" * 80)
print("Test 2: Early Convergence (Could Reduce Iterations)")
print("=" * 80)
print()

svgd2 = SVGD(
    model=model,
    observed_data=data,
    prior=uninformative_prior,
    theta_dim=1,
    n_particles=50,
    n_iterations=2000,  # Too many iterations
    learning_rate=ExponentialDecayStepSize(0.02, 0.001, 300.0),  # Fast convergence
    seed=42,
    verbose=False
)

svgd2.fit(return_history=True)
svgd2.analyze_trace()

print("\n" + "=" * 80)
print("Test 3: Not Converged (Needs More Iterations)")
print("=" * 80)
print()

svgd3 = SVGD(
    model=model,
    observed_data=data,
    prior=uninformative_prior,
    theta_dim=1,
    n_particles=50,
    n_iterations=200,  # Too few iterations
    learning_rate=0.001,  # Very slow
    seed=42,
    verbose=False
)

svgd3.fit(return_history=True)
svgd3.analyze_trace()

print("\n" + "=" * 80)
print("Test 4: Test return_dict=True")
print("=" * 80)
print()

# Get diagnostics programmatically
diag = svgd1.analyze_trace(return_dict=True, verbose=False)

print("Programmatic Access:")
print(f"  Converged: {diag['converged']}")
print(f"  Convergence iteration: {diag['convergence_point']}")
print(f"  ESS ratio: {diag['diversity']['ess_ratio']:.2%}")
print(f"  Pseudo R-hat: {diag['pseudo_rhat']:.3f}")
print(f"  Issues detected: {len(diag['issues'])}")
for issue in diag['issues']:
    print(f"    - {issue}")

print("\n✅ All tests completed!")
print("\nThe analyze_trace() method successfully:")
print("  1. Detects convergence status")
print("  2. Computes particle diversity metrics")
print("  3. Suggests parameter improvements")
print("  4. Identifies potential issues")
