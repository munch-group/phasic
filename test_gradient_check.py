from phasic import Graph
import jax
import jax.numpy as jnp
import numpy as np

g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

# Starting edge: constant
v0.add_edge(v1, 1.0)
# Parameterized edge
v1.add_edge(v2, [1.0])

print("Graph structure:")
print(f"  v0 (start) -> v1: {v0.edges()[0].weight()}")
print(f"  v1 -> v2: {v1.edges()[0].weight()}")

# Test manual update_weights
g.update_weights([2.0])
print(f"\nAfter update_weights([2.0]):")
print(f"  v0 -> v1: {v0.edges()[0].weight()}")
print(f"  v1 -> v2: {v1.edges()[0].weight()}")

# Test JAX model
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

theta = jnp.array([2.0])
times = jnp.array([1.0])

print(f"\nTesting JAX model with theta={theta}:")
pmf, moments = model(theta, times)
print(f"  PDF: {pmf}")
print(f"  Moments: {moments}")

# Test gradient
def loss(t):
    pmf, _ = model(t, times)
    return jnp.sum(pmf)

grad = jax.grad(loss)(theta)
print(f"  Gradient dPDF/dtheta: {grad}")

if abs(grad[0]) < 1e-10:
    print("  ❌ GRADIENT IS ZERO!")
else:
    print("  ✓ Gradient is non-zero")

# Test at different theta values
print(f"\nPDF values at different theta:")
for t_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
    pmf, _ = model(jnp.array([t_val]), times)
    print(f"  theta={t_val:.1f}: PDF={pmf[0]:.6f}")
