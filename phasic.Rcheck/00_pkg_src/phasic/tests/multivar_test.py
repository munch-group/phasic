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



n_iterations = 1000
step_schedule = phasic.ExpStepSize(first_step=0.001, last_step=0.0001, tau=n_iterations/5)
#reg_schedule = phasic.ExpRegularization(first_reg=5.0, last_reg=100.0, tau=50.0)

# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# step_schedule.plot(n_iterations, ax=axes[0])
# reg_schedule.plot(n_iterations, ax=axes[1])
# plt.tight_layout()



true_theta = np.array([10])  
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)



nr_observations = 10000
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples) # should check using the graph hash if a trace of the graph is cached or available online
_graph.update_parameterized_weights(true_theta)



#rewards = _graph.states()[:, 1:2]
rewards = _graph.states()[:, :-2]
rewards


observed_data = jnp.array([
    _graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T  # Shape: (nr_observations, 5)

np.mean(observed_data, axis=0)


#observed_data = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards])



def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

params = dict(
            observed_data=observed_data,
            bandwidth='median', # bandwidth='local_adaptive',
            theta_dim=len(true_theta),
            prior=uninformative_prior, 
            n_particles=24,
            n_iterations=n_iterations,
            learning_rate=step_schedule, 
            seed=42,
            verbose=True,
            # regularization=reg_schedule, 
            # regularization=0,
            # nr_moments=2,
            rewards=rewards,
)


svgd = graph.svgd(**params)