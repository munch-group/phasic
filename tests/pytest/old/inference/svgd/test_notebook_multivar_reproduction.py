"""
Reproduce exact multivariate SVGD from simple_example.ipynb
"""

import phasic
import numpy as np
import jax.numpy as jnp


def _coalescent_transitions(state, nr_samples):
    """Coalescent transitions from a non-initial state (parameterized 3-tuple format)."""
    nr_samples = int(nr_samples)
    transitions = []
    for i in range(nr_samples):
        for j in range(i, nr_samples):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i+j+1] += 1
            transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
    return transitions


def test_multivariate_convergence():
    """Test multivariate SVGD convergence"""

    print("="*60)
    print("Testing multivariate SVGD (from notebook)")
    print("="*60)

    # Setup from notebook
    true_theta = np.array([10])
    nr_samples = 4

    print(f"\nTrue parameter: θ = {true_theta[0]}")
    print(f"Number of samples: {nr_samples}")

    # Build coalescent callback with @callback decorator (ipv depends on nr_samples)
    initial_state = [nr_samples] + [0] * nr_samples

    @phasic.callback(ipv=[(initial_state, 1.0)])
    def coalescent(state, nr_samples=None, **kwargs):
        return _coalescent_transitions(state, nr_samples)

    # Create graph
    graph = phasic.Graph(coalescent, nr_samples=nr_samples)
    print(f"Graph vertices: {graph.vertices_length()}")

    # Generate multivariate observations (exact notebook pattern)
    nr_observations = 1000  # Need sufficient data for convergence
    _graph = phasic.Graph(coalescent, nr_samples=nr_samples)
    _graph.update_weights(true_theta)

    # Rewards: (n_features, n_vertices) per v0.22.22+ shape convention
    rewards_old = _graph.states()[:, :-2]  # (n_vertices, n_features) - old layout
    rewards = jnp.array(rewards_old.T)     # (n_features, n_vertices) - new layout
    print(f"\nRewards shape: {rewards.shape}")
    print(f"Rewards (first 3 features):\n{rewards[:3]}")

    # Create sparse observation matrix
    n = nr_observations
    n_features = rewards.shape[0]
    a = np.empty((n_features, n * n_features))
    a[:] = np.nan

    print(f"\nGenerating sparse observations...")
    for i in range(n_features):
        samples = _graph.sample(n, rewards=np.asarray(rewards[i]))
        a[i, i*n:(i+1)*n] = samples
        print(f"  Feature {i}: mean = {np.mean(samples):.6f}, n = {len(samples)}")

    observed_data_dense = jnp.array(a).T

    print(f"\nObserved data shape: {observed_data_dense.shape}")
    print(f"NaN count: {jnp.isnan(observed_data_dense).sum()} / {observed_data_dense.size}")
    print(f"NaN percentage: {jnp.isnan(observed_data_dense).sum() / observed_data_dense.size * 100:.1f}%")

    # Multivariate observations now require SparseObservations format (NaN-padded dense rejected)
    from phasic import dense_to_sparse
    observed_data = dense_to_sparse(observed_data_dense)

    # Check sample moments
    from phasic.svgd import compute_sample_moments
    sample_moments = compute_sample_moments(observed_data_dense, nr_moments=2)
    print(f"\nSample moments: {sample_moments}")

    # Setup SVGD parameters (from notebook)
    def uninformative_prior(phi):
        """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
        mu = 0.0
        sigma = 10.0
        return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

    n_iterations = 300

    params = dict(
        observed_data=observed_data,
        bandwidth='median',
        theta_dim=len(true_theta),
        prior=uninformative_prior,
        n_particles=12,
        n_iterations=n_iterations,
        learning_rate=0.001,  # Use fixed learning rate
        seed=42,
        verbose=True,
        rewards=rewards,
    )

    print("\n" + "="*60)
    print("Running SVGD WITHOUT regularization")
    print("="*60)

    svgd_no_reg = graph.svgd(**params)

    print(f"\n\nResults WITHOUT regularization:")
    print(f"  Posterior mean: {svgd_no_reg.theta_mean}")
    print(f"  Posterior std:  {svgd_no_reg.theta_std}")
    print(f"  True value:     {true_theta}")
    print(f"  Error:          {abs(svgd_no_reg.theta_mean[0] - true_theta[0]):.4f}")

    # Now try WITH regularization
    print("\n" + "="*60)
    print("Running SVGD WITH regularization")
    print("="*60)

    params_reg = params.copy()
    params_reg['regularization'] = 1.0
    params_reg['nr_moments'] = 2

    svgd_with_reg = graph.svgd(**params_reg)

    print(f"\n\nResults WITH regularization:")
    print(f"  Posterior mean: {svgd_with_reg.theta_mean}")
    print(f"  Posterior std:  {svgd_with_reg.theta_std}")
    print(f"  True value:     {true_theta}")
    print(f"  Error:          {abs(svgd_with_reg.theta_mean[0] - true_theta[0]):.4f}")

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"\nTrue parameter:           θ = {true_theta[0]}")
    print(f"Without regularization:   θ̂ = {svgd_no_reg.theta_mean[0]:.4f} (error: {abs(svgd_no_reg.theta_mean[0] - true_theta[0]):.4f})")
    print(f"With regularization:      θ̂ = {svgd_with_reg.theta_mean[0]:.4f} (error: {abs(svgd_with_reg.theta_mean[0] - true_theta[0]):.4f})")

    if abs(svgd_no_reg.theta_mean[0] - true_theta[0]) > 2.0:
        print(f"\n⚠️  WARNING: Estimate without regularization is far from true value!")
        print(f"   Expected: {true_theta[0]}, Got: {svgd_no_reg.theta_mean[0]:.4f}")

    if abs(svgd_with_reg.theta_mean[0] - svgd_no_reg.theta_mean[0]) > 1.0:
        print(f"\n⚠️  WARNING: Regularization produces very different result!")
        print(f"   Without: {svgd_no_reg.theta_mean[0]:.4f}, With: {svgd_with_reg.theta_mean[0]:.4f}")


if __name__ == "__main__":
    print("Reproducing multivariate SVGD from notebook...\n")
    test_multivariate_convergence()
    print("\n" + "="*60)
    print("Complete")
    print("="*60)
