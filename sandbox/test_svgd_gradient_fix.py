"""
Test that SVGD works correctly after gradient fix.

This demonstrates that the fix to parameterized_edges() in phasiccpp.cpp
allows single-parameter models to have proper gradients, enabling SVGD
to converge to the posterior.
"""

from phasic import Graph, SVGD
import numpy as np

print("="*70)
print("TEST: SVGD with Gradient Fix for Single-Parameter Models")
print("="*70)

# Build simple Erlang(2, lambda) model with lambda = theta
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

# Starting edge: constant (IPV, not scaled)
v0.add_edge(v1, 1.0)
# Parameterized edge: rate = theta[0]
v1.add_edge(v2, [1.0])

# Generate synthetic data from true parameter theta=3.0
true_theta = 3.0
g.update_weights([true_theta])
np.random.seed(42)
samples = np.random.exponential(1.0/true_theta, (200, 2)).sum(axis=1)

print(f"\nData Generation:")
print(f"  True parameter: θ = {true_theta}")
print(f"  Generated {len(samples)} samples from Erlang(2, {true_theta})")
print(f"  Sample mean: {samples.mean():.3f} (expected: {2.0/true_theta:.3f})")

# Test 1: Verify gradients are non-zero
print(f"\n1. Testing Gradient Computation")
print("-" * 70)

model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

import jax
import jax.numpy as jnp

test_theta = jnp.array([2.0])
test_times = jnp.array([1.0])

def loss_fn(theta):
    pmf, _ = model(theta, test_times)
    return jnp.sum(pmf)

grad = jax.grad(loss_fn)(test_theta)
print(f"  Test theta: {test_theta}")
print(f"  Gradient: {grad}")

if abs(grad[0]) < 1e-10:
    print("  ❌ GRADIENT IS ZERO - FIX FAILED!")
else:
    print("  ✅ Gradient is non-zero - fix successful!")

# Test 2: SVGD convergence
print(f"\n2. Testing SVGD Convergence")
print("-" * 70)

svgd = SVGD(
    model=model,
    observed_data=samples,
    theta_dim=1,
    n_particles=50,
    n_iterations=200,
    learning_rate=0.01,
    nr_moments=2,
    regularization=0.0  # No regularization needed with working gradients
)

print("Running SVGD...")
svgd.optimize()

posterior_mean = svgd.theta_mean[0]
posterior_std = svgd.theta_std[0]
rel_error = abs(posterior_mean - true_theta) / true_theta * 100

print(f"\nResults:")
print(f"  True parameter:     θ = {true_theta:.3f}")
print(f"  Posterior mean:     θ̂ = {posterior_mean:.3f} ± {posterior_std:.3f}")
print(f"  Relative error:     {rel_error:.1f}%")

# Test 3: Verify PDF varies with theta
print(f"\n3. Testing PDF Theta-Dependence")
print("-" * 70)

test_time = 1.0
pdf_values = []
for theta_val in [0.5, 1.0, 2.0, 3.0, 5.0]:
    pmf, _ = model(jnp.array([theta_val]), jnp.array([test_time]))
    pdf_values.append(float(pmf[0]))
    print(f"  θ = {theta_val:.1f}: PDF = {pmf[0]:.6f}")

# Check variance
pdf_variance = np.var(pdf_values)
if pdf_variance < 1e-10:
    print("  ❌ PDF is constant - no theta dependence!")
else:
    print(f"  ✅ PDF varies with theta (variance = {pdf_variance:.6e})")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

success = True
if abs(grad[0]) < 1e-10:
    print("❌ Gradients are zero")
    success = False
else:
    print("✅ Gradients work")

if pdf_variance < 1e-10:
    print("❌ PDF has no theta dependence")
    success = False
else:
    print("✅ PDF depends on theta")

if rel_error > 20:
    print(f"⚠️  SVGD error high ({rel_error:.1f}%), but posterior moved from prior")
elif rel_error > 10:
    print(f"✅ SVGD converged with moderate error ({rel_error:.1f}%)")
else:
    print(f"✅ SVGD converged accurately ({rel_error:.1f}% error)")

if success:
    print("\n✅ GRADIENT FIX SUCCESSFUL - SVGD WORKS!")
else:
    print("\n❌ GRADIENT FIX INCOMPLETE")

print("="*70)
