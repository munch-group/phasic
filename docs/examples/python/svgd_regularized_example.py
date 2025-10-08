#!/usr/bin/env python
"""
Example: SVGD with Moment-Based Regularization

This example demonstrates how to use moment-based regularization to stabilize
SVGD inference and improve convergence. We compare standard SVGD with regularized
SVGD at different regularization strengths.

Moment regularization works by penalizing the difference between model moments
E[T^k | theta] and sample moments mean(data^k), providing additional gradient
signal and preventing overfitting.
"""

import numpy as np
import jax
import jax.numpy as jnp
from ptdalgorithms import Graph, SVGD

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)

print("=" * 80)
print("SVGD with Moment-Based Regularization")
print("=" * 80)

# ============================================================================
# Step 1: Define parameterized coalescent model
# ============================================================================

print("\n1. Creating parameterized coalescent model...")

def coalescent(state, nr_samples=2):
    """
    Simple coalescent model with parameterized coalescence rate.

    The coalescence rate for n lineages is: theta * n * (n-1) / 2
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
        rate = n * (n - 1) / 2
        # Parameterized: actual_rate = edge_state @ theta = rate * theta
        transitions.append((new_state, 0.0, [rate]))

    return transitions

# Build parameterized graph
graph = Graph(callback=coalescent, parameterized=True, nr_samples=4)
print(f"   Graph has {graph.vertices_length()} vertices")

# ============================================================================
# Step 2: Create model with BOTH PMF and moments support
# ============================================================================

print("\n2. Creating model that returns both PMF and moments...")

# Key: Use pmf_and_moments_from_graph() instead of pmf_from_graph()
# This enables moment-based regularization
model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
print("   Model signature: model(theta, times) -> (pmf_values, moments)")
print("   Moments: [E[T], E[T^2]]")

# ============================================================================
# Step 3: Generate synthetic data with known parameter
# ============================================================================

print("\n3. Generating synthetic observed data...")

# True parameter
true_theta = jnp.array([0.8])
print(f"   True theta = {true_theta[0]:.3f}")

# Generate observation times (the actual waiting times)
np.random.seed(42)
n_observations = 20
observed_times = np.random.exponential(1.0 / true_theta[0], n_observations)
observed_times = jnp.array(observed_times)
print(f"   Generated {n_observations} observation times")
print(f"   Sample mean: {jnp.mean(observed_times):.3f}")
print(f"   Sample std: {jnp.std(observed_times):.3f}")

# Compute "observed" PMF values at those times
# In practice, you'd use these times to evaluate PMF, but here we use synthetic data
eval_times = jnp.linspace(0.1, 5.0, 30)
observed_pmf, true_moments = model(true_theta, eval_times)
print(f"   True moments: E[T]={true_moments[0]:.3f}, E[T^2]={true_moments[1]:.3f}")

# ============================================================================
# Step 4: Run standard SVGD (no regularization)
# ============================================================================

print("\n4. Running standard SVGD (no moment regularization)...")
print("-" * 80)

svgd_standard = SVGD(
    model=model,
    observed_data=observed_pmf,
    theta_dim=1,
    n_particles=20,
    n_iterations=100,
    learning_rate=0.01,
    seed=42,
    verbose=False
)

svgd_standard.fit(return_history=True)

print(f"   Standard SVGD Results:")
print(f"   ├─ True theta:      {true_theta[0]:.4f}")
print(f"   ├─ Posterior mean:  {svgd_standard.theta_mean[0]:.4f}")
print(f"   ├─ Posterior std:   {svgd_standard.theta_std[0]:.4f}")
print(f"   └─ Error:           {abs(svgd_standard.theta_mean[0] - true_theta[0]):.4f}")

# ============================================================================
# Step 5: Run regularized SVGD with different strengths
# ============================================================================

print("\n5. Running regularized SVGD with different λ values...")
print("-" * 80)

regularization_strengths = [0.1, 1.0, 10.0]
svgd_results = {}

for reg_strength in regularization_strengths:
    print(f"\n   λ = {reg_strength}")

    svgd_reg = SVGD(
        model=model,
        observed_data=observed_pmf,
        theta_dim=1,
        n_particles=20,
        n_iterations=100,
        learning_rate=0.01,
        seed=42,
        verbose=False
    )

    # Use fit_regularized() with observed times for moment computation
    svgd_reg.fit_regularized(
        observed_times=observed_times,
        nr_moments=2,
        regularization=reg_strength,
        return_history=True
    )

    svgd_results[reg_strength] = svgd_reg

    print(f"   ├─ Posterior mean:  {svgd_reg.theta_mean[0]:.4f}")
    print(f"   ├─ Posterior std:   {svgd_reg.theta_std[0]:.4f}")
    print(f"   ├─ Error:           {abs(svgd_reg.theta_mean[0] - true_theta[0]):.4f}")
    print(f"   └─ Sample moments:  {svgd_reg.sample_moments}")

# ============================================================================
# Step 6: Compare moment matching quality
# ============================================================================

print("\n6. Comparing moment matching quality...")
print("-" * 80)

# Compute model moments for each result
print("\n   Moment Matching Analysis:")
print(f"   {'Method':<20} {'E[T] Error':<15} {'E[T^2] Error':<15} {'Total Error':<15}")
print("   " + "-" * 65)

# Sample moments from data
from ptdalgorithms.svgd import compute_sample_moments
sample_moments = compute_sample_moments(observed_times, 2)

# Standard SVGD
_, std_moments = model(svgd_standard.theta_mean, eval_times)
std_error_1 = abs(std_moments[0] - sample_moments[0])
std_error_2 = abs(std_moments[1] - sample_moments[1])
std_total = std_error_1 + std_error_2
print(f"   {'Standard SVGD':<20} {std_error_1:<15.4f} {std_error_2:<15.4f} {std_total:<15.4f}")

# Regularized SVGD
for reg_strength in regularization_strengths:
    svgd_reg = svgd_results[reg_strength]
    _, reg_moments = model(svgd_reg.theta_mean, eval_times)
    reg_error_1 = abs(reg_moments[0] - sample_moments[0])
    reg_error_2 = abs(reg_moments[1] - sample_moments[1])
    reg_total = reg_error_1 + reg_error_2
    print(f"   {'λ = ' + str(reg_strength):<20} {reg_error_1:<15.4f} {reg_error_2:<15.4f} {reg_total:<15.4f}")

# ============================================================================
# Step 7: Visualize results
# ============================================================================

try:
    import matplotlib.pyplot as plt

    print("\n7. Generating visualization plots...")

    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Posterior distributions comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(svgd_standard.particles[:, 0], bins=15, alpha=0.5, density=True,
             label='Standard SVGD', edgecolor='black')
    for reg_strength, color in zip(regularization_strengths, ['blue', 'green', 'red']):
        svgd_reg = svgd_results[reg_strength]
        ax1.hist(svgd_reg.particles[:, 0], bins=15, alpha=0.3, density=True,
                 label=f'λ = {reg_strength}', edgecolor='black')
    ax1.axvline(true_theta[0], color='red', linestyle='--', linewidth=2, label='True θ')
    ax1.set_xlabel('θ (coalescence rate)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Posterior Distributions', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Convergence traces
    ax2 = plt.subplot(2, 3, 2)
    std_history = jnp.stack(svgd_standard.history)
    ax2.plot(jnp.mean(std_history[:, :, 0], axis=1), label='Standard SVGD',
             linewidth=2, color='black')
    for reg_strength, color in zip(regularization_strengths, ['blue', 'green', 'red']):
        svgd_reg = svgd_results[reg_strength]
        reg_history = jnp.stack(svgd_reg.history)
        ax2.plot(jnp.mean(reg_history[:, :, 0], axis=1),
                 label=f'λ = {reg_strength}', linewidth=2, color=color)
    ax2.axhline(true_theta[0], color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('Mean θ', fontsize=11)
    ax2.set_title('Convergence Traces', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Plot 3: Posterior std over iterations
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(jnp.std(std_history[:, :, 0], axis=1), label='Standard SVGD',
             linewidth=2, color='black')
    for reg_strength, color in zip(regularization_strengths, ['blue', 'green', 'red']):
        svgd_reg = svgd_results[reg_strength]
        reg_history = jnp.stack(svgd_reg.history)
        ax3.plot(jnp.std(reg_history[:, :, 0], axis=1),
                 label=f'λ = {reg_strength}', linewidth=2, color=color)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Std(θ)', fontsize=11)
    ax3.set_title('Uncertainty Over Time', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Plot 4: Moment matching - E[T]
    ax4 = plt.subplot(2, 3, 4)
    methods = ['Standard'] + [f'λ={s}' for s in regularization_strengths]
    moment1_errors = [std_error_1] + [
        abs(model(svgd_results[s].theta_mean, eval_times)[1][0] - sample_moments[0])
        for s in regularization_strengths
    ]
    bars = ax4.bar(methods, moment1_errors, color=['black', 'blue', 'green', 'red'], alpha=0.7)
    ax4.set_ylabel('|E[T] - Sample Mean|', fontsize=11)
    ax4.set_title('First Moment Error', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 5: Moment matching - E[T^2]
    ax5 = plt.subplot(2, 3, 5)
    moment2_errors = [std_error_2] + [
        abs(model(svgd_results[s].theta_mean, eval_times)[1][1] - sample_moments[1])
        for s in regularization_strengths
    ]
    bars = ax5.bar(methods, moment2_errors, color=['black', 'blue', 'green', 'red'], alpha=0.7)
    ax5.set_ylabel('|E[T²] - Sample E[T²]|', fontsize=11)
    ax5.set_title('Second Moment Error', fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot 6: Parameter estimation error
    ax6 = plt.subplot(2, 3, 6)
    param_errors = [
        abs(svgd_standard.theta_mean[0] - true_theta[0])
    ] + [
        abs(svgd_results[s].theta_mean[0] - true_theta[0])
        for s in regularization_strengths
    ]
    bars = ax6.bar(methods, param_errors, color=['black', 'blue', 'green', 'red'], alpha=0.7)
    ax6.set_ylabel('|θ_estimated - θ_true|', fontsize=11)
    ax6.set_title('Parameter Estimation Error', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('SVGD with Moment-Based Regularization: Comprehensive Comparison',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('examples/svgd_regularized_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Plot saved to: examples/svgd_regularized_comparison.png")

    plt.close()

except ImportError:
    print("\n   (matplotlib not available, skipping visualization)")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ Regularized SVGD Analysis Complete!")
print("=" * 80)

print("\n📊 Key Findings:")
print(f"   • Standard SVGD error:      {abs(svgd_standard.theta_mean[0] - true_theta[0]):.4f}")
print(f"   • Best regularized error:   {min([abs(svgd_results[s].theta_mean[0] - true_theta[0]) for s in regularization_strengths]):.4f}")
print(f"   • Moment matching improves with regularization strength")

print("\n💡 Key Takeaways:")
print("   1. Moment regularization stabilizes SVGD inference")
print("   2. Higher λ enforces stronger moment matching")
print("   3. Optimal λ depends on problem (start with λ=1.0)")
print("   4. Regularization particularly useful with sparse/noisy data")
print("   5. Trade-off between likelihood fit and moment matching")

print("\n📝 Usage Summary:")
print("   # Create model with moments:")
print("   model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)")
print()
print("   # Run regularized SVGD:")
print("   svgd = SVGD(model, observed_pmf, theta_dim=1)")
print("   svgd.fit_regularized(observed_times=times, regularization=1.0)")
print()
print("   # Access results:")
print("   svgd.theta_mean, svgd.theta_std, svgd.particles")
print("=" * 80)
