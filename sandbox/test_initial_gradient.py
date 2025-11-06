"""Test what gradient is computed at initial particle value"""
import phasic
import numpy as np
import jax
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

true_theta = np.array([10.0])
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)

_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

np.random.seed(42)
observed_data = jnp.array([
    _graph.sample(10000, rewards=rewards[:, i])  # Use FULL dataset like multivar_test.py
    for i in range(rewards.shape[1])
]).T

print(f"Observed data shape: {observed_data.shape}")
print(f"Rewards shape: {rewards.shape}")

model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False, use_ffi=False, param_length=1
)

# Define log probability WITH PRIOR like SVGD does
def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

def log_prob(phi):
    """Full log probability: likelihood + prior"""
    # Transform to positive parameters
    theta = jax.nn.softplus(phi)
    # Compute likelihood
    pmf, _ = model(theta, observed_data, rewards=rewards)
    log_lik = jnp.sum(jnp.log(pmf + 1e-10))
    # Add prior
    log_pri = uninformative_prior(phi)
    return log_lik + log_pri

grad_fn = jax.grad(log_prob)

# Test at initial particle values
test_phis = [
    jnp.array([0.82]),
    jnp.array([-1.17]),
    jnp.array([1.19]),
]

print("\nGradients at initial particle positions:")
for phi in test_phis:
    grad = grad_fn(phi)
    log_p = log_prob(phi)
    theta = jax.nn.softplus(phi)
    print(f"  phi={float(phi[0]):6.2f} (theta={float(theta[0]):5.2f}): grad={float(grad[0]):12.2f}, log_prob={float(log_p):10.2f}")

print("\nWith step_size=0.05, the update would be:")
for phi in test_phis:
    grad = grad_fn(phi)
    phi_new = phi + 0.05 * grad
    theta_new = jax.nn.softplus(phi_new)
    print(f"  phi={float(phi[0]):6.2f} → phi_new={float(phi_new[0]):8.2f} (theta_new={float(theta_new[0]):8.2f})")
