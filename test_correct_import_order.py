"""Test with correct import order: phasic BEFORE jax"""
import sys

print("Step 1: Import phasic FIRST")
sys.stdout.flush()

import phasic
print("  ✓ phasic imported")
sys.stdout.flush()

print("\nStep 2: Import numpy and jax AFTER phasic")
sys.stdout.flush()

import numpy as np
print("  ✓ numpy imported")
sys.stdout.flush()

import jax.numpy as jnp
print("  ✓ jax imported")
sys.stdout.flush()

print("\nStep 3: Define coalescent")
sys.stdout.flush()

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

print("  ✓ coalescent defined")
sys.stdout.flush()

print("\nStep 4: Create and test model")
sys.stdout.flush()

try:
    true_theta = np.array([10.0])
    nr_samples = 4
    graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    print("  ✓ Graph created")
    sys.stdout.flush()

    _graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    _graph.update_parameterized_weights(true_theta)
    print("  ✓ Second graph created and weights updated")
    sys.stdout.flush()

    rewards_raw = _graph.states().T[:-2]
    rewards = rewards_raw.T.astype(np.float64)
    print(f"  ✓ Rewards prepared: shape {rewards.shape}")
    sys.stdout.flush()

    nr_observations = 50
    observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards_raw])
    observed_data = observed_data_raw.T
    print(f"  ✓ Data sampled: shape {observed_data.shape}")
    sys.stdout.flush()

    def uninformative_prior(phi):
        mu = 0.0
        sigma = 10.0
        return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

    print("\nStep 5: Run SVGD")
    sys.stdout.flush()

    params = dict(
        observed_data=observed_data,
        bandwidth='median',
        theta_dim=1,
        prior=uninformative_prior,
        n_particles=8,
        n_iterations=3,
        seed=42,
        verbose=True,
        regularization=0,
        nr_moments=0,
        rewards=rewards,
    )

    svgd = graph.svgd(**params)
    print("\n✓ SUCCESS!")
    print(f"  True theta: {true_theta}")
    if hasattr(svgd, 'theta_mean'):
        print(f"  Posterior mean: {svgd.theta_mean}")

except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
