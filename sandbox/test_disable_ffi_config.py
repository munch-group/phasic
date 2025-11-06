"""Test with FFI disabled via configuration"""
import phasic

# Configure BEFORE doing anything else
print("Configuring phasic with ffi=False, openmp=False...")
phasic.configure(ffi=False, openmp=False)
config = phasic.get_config()
print(f"Config: jax={config.jax}, jit={config.jit}, ffi={config.ffi}, openmp={config.openmp}")

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

print("\nCreating graph...")
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

print("Creating data...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
nr_observations = 50
observed_data = jnp.array(_graph.sample(nr_observations))
print(f"Data shape: {observed_data.shape}")

print("\nRunning SVGD...")
def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

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
)

try:
    svgd = graph.svgd(**params)
    print("\n✓ SUCCESS!")
    print(f"  True theta: {true_theta}")
    if hasattr(svgd, 'theta_mean'):
        print(f"  Posterior mean: {svgd.theta_mean}")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
