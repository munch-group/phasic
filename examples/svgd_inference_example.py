#!/usr/bin/env python
"""
Example: Bayesian Parameter Inference with SVGD

This example demonstrates how to use the SVGD class for Bayesian inference
to infer parameters of a coalescent model from observed data.

SVGD (Stein Variational Gradient Descent) is a powerful Bayesian inference
method that approximates the posterior distribution p(theta | data) using
a set of particles.

This example showcases:
- Creating a parameterized coalescent model
- Running SVGD inference using the object-oriented API
- Accessing results as attributes
- Generating diagnostic plots
"""

import numpy as np
import jax.numpy as jnp
from ptdalgorithms import Graph, SVGD

print("=" * 70)
print("Bayesian Parameter Inference with SVGD")
print("=" * 70)

# ============================================================================
# Step 1: Define a parameterized coalescent model
# ============================================================================

def coalescent(state, nr_samples=2):
    """
    Simple coalescent model with parameterized coalescence rate.

    The coalescence rate is controlled by theta, where the actual rate
    for n lineages is: theta * n * (n-1) / 2

    This is a simple Kingman coalescent model.
    """
    transitions = []

    if len(state) == 0:
        # Initial state: all lineages in a single deme
        initial = np.array([nr_samples])
        return [(initial, 1.0, [1.0])]  # edge_state = [1.0] for parameterization

    if state[0] > 1:
        # Coalescence event: two lineages merge
        n = state[0]
        new_state = np.array([state[0] - 1])

        # Rate = n*(n-1)/2 (unparameterized rate)
        # With parameterization: actual_rate = edge_state @ theta
        # We set edge_state = [n*(n-1)/2] so theta scales the coalescence rate
        rate = n * (n - 1) / 2
        transitions.append((new_state, 0.0, [rate]))

    return transitions

print("\n1. Building parameterized coalescent graph (nr_samples=4)...")
graph = Graph(callback=coalescent, parameterized=True, nr_samples=4)
print(f"   Graph has {graph.vertices_length()} vertices")

# ============================================================================
# Step 2: Convert to JAX-compatible model
# ============================================================================

print("\n2. Converting to JAX-compatible model...")
model = Graph.pmf_from_graph(graph)
print("   Model signature: model(theta, times) -> probabilities")

# ============================================================================
# Step 3: Generate synthetic observed data
# ============================================================================

print("\n3. Generating synthetic data with known parameter...")

# True coalescence rate parameter
true_theta = jnp.array([0.8])
print(f"   True theta = {true_theta[0]}")

# Observation times
observed_times = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

# Generate "observed" probabilities from the true model
observed_data = model(true_theta, observed_times)
print(f"   Generated {len(observed_times)} observations")

# ============================================================================
# Step 4: Create SVGD object and run inference
# ============================================================================

print("\n4. Creating SVGD object and running inference...")

# Create SVGD inference object
svgd = SVGD(
    model=model,
    observed_data=observed_data,
    theta_dim=1,              # One parameter to infer
    n_particles=20,           # Use 20 particles for posterior approximation
    n_iterations=100,         # 100 optimization steps
    learning_rate=0.01,       # Step size
    seed=42,                  # For reproducibility
    verbose=True              # Show progress
)

# Run inference and save history for diagnostic plots
svgd.fit(return_history=True)

# ============================================================================
# Step 5: Analyze results
# ============================================================================

print("\n5. Results:")
print("-" * 70)

# Access results as attributes
print(f"   True theta:            {true_theta[0]:.4f}")
print(f"   Posterior mean:        {svgd.theta_mean[0]:.4f}")
print(f"   Posterior std:         {svgd.theta_std[0]:.4f}")
print(f"   95% credible interval: [{svgd.theta_mean[0] - 1.96*svgd.theta_std[0]:.4f}, "
      f"{svgd.theta_mean[0] + 1.96*svgd.theta_std[0]:.4f}]")

# The particles represent samples from the posterior distribution
print(f"\n   Posterior samples (first 5 particles):")
for i in range(min(5, svgd.n_particles)):
    print(f"     Particle {i+1}: theta = {svgd.particles[i, 0]:.4f}")

# Print comprehensive summary
print()
svgd.summary()

# ============================================================================
# Step 6: Generate diagnostic plots using built-in methods
# ============================================================================

try:
    import matplotlib.pyplot as plt

    print("\n6. Generating diagnostic plots...")

    # Plot 1: Posterior distribution with true value
    print("   - Posterior distribution histogram...")
    svgd.plot_posterior(
        true_theta=true_theta,
        param_names=['θ (coalescence rate)'],
        save_path='examples/svgd_posterior.png'
    )

    # Plot 2: Trace plots showing convergence
    print("   - Trace plot showing particle evolution...")
    svgd.plot_trace(
        param_names=['θ (coalescence rate)'],
        save_path='examples/svgd_trace.png'
    )

    # Plot 3: Convergence diagnostics
    print("   - Convergence diagnostics...")
    svgd.plot_convergence(
        save_path='examples/svgd_convergence.png'
    )

    # Additional custom plot: Model predictions vs parameter
    print("   - Model predictions...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    theta_range = jnp.linspace(0.1, 2.0, 50)
    times_plot = jnp.array([1.0, 2.0, 3.0])

    for t in times_plot:
        probs = [model(jnp.array([th]), jnp.array([t]))[0] for th in theta_range]
        ax.plot(theta_range, probs, label=f't = {t}', linewidth=2)

    ax.axvline(true_theta[0], color='red', linestyle='--', linewidth=2,
                label=f'True θ = {true_theta[0]:.2f}')
    ax.axvline(svgd.theta_mean[0], color='blue', linestyle='--', linewidth=2,
                label=f'Inferred θ = {svgd.theta_mean[0]:.2f}')
    ax.set_xlabel('θ (coalescence rate)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Model Predictions vs Parameter', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/svgd_model_predictions.png', dpi=150, bbox_inches='tight')
    print("   Plot saved to: examples/svgd_model_predictions.png")

    plt.close('all')

except ImportError:
    print("\n   (matplotlib not available, skipping visualization)")

print("\n" + "=" * 70)
print("✅ SVGD inference complete!")
print("=" * 70)
print("\nKey takeaways:")
print("  • SVGD class provides object-oriented interface with built-in diagnostics")
print("  • Access results as attributes: svgd.theta_mean, svgd.theta_std, svgd.particles")
print("  • Built-in plotting methods: plot_posterior(), plot_trace(), plot_convergence()")
print("  • SVGD provides full posterior distribution, not just point estimates")
print("  • The posterior captures uncertainty in parameter estimates")
print("  • More particles and iterations generally improve accuracy")
print("  • Custom priors can be specified to incorporate domain knowledge")
print("=" * 70)
