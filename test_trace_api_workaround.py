"""WORKAROUND: Use trace API instead of model API to avoid segfault"""
import phasic
import numpy as np
import jax.numpy as jnp

from phasic.trace_elimination import (
    record_elimination_trace,
    trace_to_log_likelihood
)

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

print("=" * 70)
print("WORKAROUND: Using trace API instead of model API")
print("=" * 70)

print("\n1. Creating graph...")
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("   ✓ Graph created")

print("\n2. Recording elimination trace...")
trace = record_elimination_trace(graph, param_length=1)
print(f"   ✓ Trace recorded")

print("\n3. Creating data...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
nr_observations = 100
observed_times = jnp.array(_graph.sample(nr_observations))
print(f"   ✓ Data sampled: {len(observed_times)} observations")

print("\n4. Creating log-likelihood function from trace...")
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)
print("   ✓ Log-likelihood function created")

print("\n5. Testing log-likelihood evaluation...")
try:
    test_theta = jnp.array([10.0])
    ll_value = log_lik(test_theta)
    print(f"   ✓ Log-likelihood at true theta: {ll_value:.4f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n6. Running SVGD...")
def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

try:
    svgd = phasic.SVGD(
        log_likelihood=log_lik,
        observed_data=None,  # Already baked into log_lik
        theta_dim=1,
        prior=uninformative_prior,
        n_particles=16,
        n_iterations=10,
        seed=42,
        verbose=True,
        learning_rate=0.01,
    )
    result = svgd.fit()

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"\nTrue theta:       {true_theta[0]:.4f}")
    print(f"Posterior mean:   {result['theta_mean'][0]:.4f}")
    print(f"Posterior std:    {result['theta_std'][0]:.4f}")
    print(f"\nThis workaround avoids the model API segfault!")

except Exception as e:
    print(f"\n✗ SVGD FAILED: {e}")
    import traceback
    traceback.print_exc()
