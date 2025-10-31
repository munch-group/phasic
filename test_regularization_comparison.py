"""Test SVGD with different regularization levels"""
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

print("Testing different regularization levels")
print("=" * 70)

true_theta = np.array([10.0])
nr_samples = 5
nr_observations = 1000  # Smaller for faster testing

# Create data
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

# Sample data with all features (NO NaN)
np.random.seed(42)
observed_data = jnp.array([
    _graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T

print(f"Observed data shape: {observed_data.shape}")
print(f"Data means: {jnp.mean(observed_data, axis=0)}")

# Test configurations
configs = [
    {"regularization": 0, "nr_moments": 1, "name": "No regularization"},
    {"regularization": 1, "nr_moments": 1, "name": "Weak regularization (1)"},
    {"regularization": 10, "nr_moments": 1, "name": "Medium regularization (10)"},
    {"regularization": 100, "nr_moments": 1, "name": "Strong regularization (100)"},
    {"regularization": 1000, "nr_moments": 1, "name": "Huge regularization (1000)"},
]

n_iterations = 200  # Fewer iterations for faster testing
step_schedule = phasic.ExponentialDecayStepSize(
    first_step=0.001, last_step=0.0001, tau=n_iterations/5
)

def uninformative_prior(phi):
    return -0.5 * jnp.sum(((phi - 0.0) / 10.0)**2)

results_summary = []

for config in configs:
    print(f"\n{'=' * 70}")
    print(f"Testing: {config['name']}")
    print(f"  regularization={config['regularization']}, nr_moments={config['nr_moments']}")
    print('-' * 70)

    # Create fresh graph for each test
    graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

    params = dict(
        observed_data=observed_data,
        bandwidth='median',
        theta_dim=len(true_theta),
        prior=uninformative_prior,
        n_particles=12,  # Fewer particles for speed
        n_iterations=n_iterations,
        learning_rate=step_schedule,
        seed=42,
        verbose=False,  # Suppress output for cleaner comparison
        rewards=rewards,
        regularization=config['regularization'],
        nr_moments=config['nr_moments'],
    )

    svgd = graph.svgd(**params)
    results = svgd.get_results()

    final_theta = results['theta_mean']
    final_std = results['theta_std']
    error = final_theta[0] - true_theta[0]
    error_pct = abs(error) / true_theta[0] * 100

    print(f"  Estimated theta: {final_theta[0]:.2f} ± {final_std[0]:.2f}")
    print(f"  Error: {error:+.2f} ({error_pct:.1f}%)")

    results_summary.append({
        'config': config['name'],
        'regularization': config['regularization'],
        'theta': final_theta[0],
        'std': final_std[0],
        'error': error,
        'error_pct': error_pct
    })

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"True theta: {true_theta[0]:.2f}")
print(f"\n{'Config':<30} {'Theta':>10} {'Std':>8} {'Error':>10} {'Error %':>10}")
print('-' * 70)
for r in results_summary:
    print(f"{r['config']:<30} {r['theta']:>10.2f} {r['std']:>8.2f} {r['error']:>+10.2f} {r['error_pct']:>9.1f}%")

print(f"\n{'=' * 70}")
print("INTERPRETATION:")
print("  If estimates are consistent across regularization levels:")
print("    → Regularization is NOT the issue")
print("  If estimates vary widely:")
print("    → Regularization is affecting convergence")
print(f"{'=' * 70}")
