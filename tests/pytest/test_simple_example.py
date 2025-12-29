import phasic
import numpy as np
import jax.numpy as jnp

nr_samples = 4

def coalescent(state, nr_samples=None):
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
            transitions.append([new, [state[i]*(state[j]-same)/(1+same)]])
    return transitions

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

step_schedule = phasic.ExpStepSize(first_step=0.01, last_step=0.001, tau=50.0)
reg_schedule = phasic.ExpRegularization(first_reg=10.0, last_reg=1.0, tau=50.0)

true_theta = [10]
nr_observations = 100  # Reduce for faster testing

params = dict(
            bandwidth='median',
            theta_dim=len(true_theta),
            prior=uninformative_prior,
            n_particles=10,  # Reduce for faster testing
            n_iterations=2,   # Just test a couple iterations
            learning_rate=step_schedule,
            seed=42,
            verbose=True,
            progress=True,
            discrete=False,
)

print("Creating graph...")
graph = phasic.Graph(coalescent, ipv=[[[4, 0, 0, 0], 1.0]], nr_samples=nr_samples)

print("Updating weights...")
graph.update_weights(true_theta)

print("Sampling observations...")
observed_data = np.array(graph.sample(nr_observations))
print(f"Observed data shape: {observed_data.shape}")
print(f"Observed data type: {type(observed_data)}")
print(f"First 10 values: {observed_data[:10]}")

print("\nRunning SVGD...")
svgd = graph.svgd(observed_data=observed_data, **params)

print("\nGetting results...")
result = svgd.get_results()["theta_mean"].item()
print(f"Estimated theta: {result}")
print(f"True theta: {true_theta[0]}")
