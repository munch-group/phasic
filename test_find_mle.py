"""Find where gradient crosses zero"""
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
    _graph.sample(100, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T

model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False, use_ffi=False, param_length=1
)

def log_likelihood(phi):
    theta = jax.nn.softplus(phi)
    pmf, _ = model(theta, observed_data, rewards=rewards)
    return jnp.sum(jnp.log(pmf + 1e-10))

grad_fn = jax.grad(log_likelihood)

print("Finding where gradient crosses zero...")
for phi_val in [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50]:
    phi = jnp.array([float(phi_val)])
    theta = jax.nn.softplus(phi)
    grad = grad_fn(phi)
    log_lik = log_likelihood(phi)
    print(f"phi={phi_val:3.0f} (theta≈{float(theta[0]):6.1f}): grad={float(grad[0]):10.4f}, log_lik={float(log_lik):10.2f}")
