"""
Test which data layout gives correct SVGD convergence
"""

import phasic
import numpy as np
import jax.numpy as jnp


def coalescent(state, nr_samples=None):
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]
        return ipv
    else:
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


def test_layout(use_transpose, n_iterations=100):
    """Test SVGD convergence with different data layouts"""

    print("\n" + "="*70)
    print(f"Testing with {'TRANSPOSED' if use_transpose else 'NOTEBOOK'} layout")
    print("="*70)

    true_theta = np.array([10])
    nr_samples = 3

    # Create graph and rewards
    graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    _graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    _graph.update_parameterized_weights(true_theta)
    rewards = _graph.states()[:, :-2]

    print(f"Rewards shape: {rewards.shape}")

    # Create observed data
    n = 500  # Smaller for faster testing
    a = np.empty((rewards.shape[1], n * rewards.shape[1]))
    a[:] = np.nan
    for i in range(rewards.shape[1]):
        a[i, i*n:(i+1)*n] = _graph.sample(n, rewards=rewards[:, i])

    if use_transpose:
        observed_data = jnp.array(a).T
    else:
        observed_data = jnp.array(a)

    print(f"Observed data shape: {observed_data.shape}")
    print(f"NaN percentage: {jnp.isnan(observed_data).sum() / observed_data.size * 100:.1f}%")

    # Prior
    def uninformative_prior(phi):
        mu = 0.0
        sigma = 10.0
        return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

    # Run SVGD
    try:
        svgd = graph.svgd(
            observed_data=observed_data,
            bandwidth='median',
            theta_dim=len(true_theta),
            prior=uninformative_prior,
            n_particles=12,  # Fewer particles for speed
            n_iterations=n_iterations,
            learning_rate=0.001,
            seed=42,
            verbose=True,
            rewards=rewards,
        )

        print(f"\nResults:")
        print(f"  Posterior mean: {svgd.theta_mean}")
        print(f"  Posterior std:  {svgd.theta_std}")
        print(f"  True value:     {true_theta}")
        print(f"  Error:          {abs(svgd.theta_mean[0] - true_theta[0]):.4f}")

        return svgd.theta_mean[0], abs(svgd.theta_mean[0] - true_theta[0])

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("Testing data layouts for multivariate SVGD convergence")
    print("="*70)

    # Test notebook layout (n_features, n_observations)
    mean_notebook, error_notebook = test_layout(use_transpose=False, n_iterations=100)

    # Test transposed layout (n_observations, n_features)
    mean_transposed, error_transposed = test_layout(use_transpose=True, n_iterations=100)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"True parameter: θ = 10")

    if mean_notebook is not None:
        print(f"Notebook layout:    θ̂ = {mean_notebook:.4f} (error: {error_notebook:.4f})")
    else:
        print(f"Notebook layout:    FAILED")

    if mean_transposed is not None:
        print(f"Transposed layout:  θ̂ = {mean_transposed:.4f} (error: {error_transposed:.4f})")
    else:
        print(f"Transposed layout:  FAILED")

    if mean_notebook is not None and mean_transposed is not None:
        if error_transposed < error_notebook:
            print(f"\n✓ Transposed layout is better (error: {error_transposed:.4f} vs {error_notebook:.4f})")
        else:
            print(f"\n✓ Notebook layout is better (error: {error_notebook:.4f} vs {error_transposed:.4f})")
