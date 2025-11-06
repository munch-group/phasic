import numpy as np
from phasic import Graph, set_theme
import jax.numpy as jnp

set_theme('dark')

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

true_theta = np.array([7])
nr_samples = 20

nr_observations = 100  # Reduced from 1000
_graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
observed_data = _graph.sample(nr_observations)

graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

from phasic import SVGD, ExponentialDecayStepSize

step_schedule = ExponentialDecayStepSize(first_step=0.001, last_step=0.00001, tau=500.0)

params = dict(
            bandwidth='median',
            observed_data=observed_data,
            prior=uninformative_prior,
            theta_dim=len(true_theta),
            n_particles=10,  # Reduced from 20
            n_iterations=3,  # Reduced for testing
            learning_rate=step_schedule,
            seed=42,
            verbose=True
)

model_pdf = Graph.pmf_and_moments_from_graph(graph)
svgd = SVGD(model_pdf, **params)

print("Starting fit_regularized...")
import time
start = time.time()
svgd.fit_regularized(nr_moments=2, regularization=1.0)
end = time.time()
print(f"fit_regularized took {end - start:.2f} seconds")
