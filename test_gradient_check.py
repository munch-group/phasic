"""Test if gradients are correct for multivariate model"""
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

print("Testing gradient computation...")

true_theta = np.array([10.0])
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)

# Create test data
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
_graph.update_parameterized_weights(true_theta)
states = _graph.states()
rewards = states[:, :-1]

# Sample small dataset
observed_data = jnp.array([
    [np.array(_graph.sample(3, rewards=rewards[:, i])) for i in range(rewards.shape[1])]
]).T[0]  # Shape: (3, 5)

print(f"Observed data shape: {observed_data.shape}")
print(f"Observed data:\n{observed_data}")

# Create model
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=0, discrete=False, use_ffi=False, param_length=1
)

# Define log-likelihood function
def log_likelihood(theta_val):
    theta = jnp.array([theta_val])
    pmf, _ = model(theta, observed_data, rewards=rewards)
    return jnp.sum(jnp.log(pmf + 1e-10))

# Compute gradient at true theta
grad_fn = jax.grad(log_likelihood)
grad_at_true = grad_fn(true_theta[0])
print(f"\nGradient at TRUE theta={true_theta[0]}: {grad_at_true}")

# Compute gradient at theta=1
grad_at_1 = grad_fn(1.0)
print(f"Gradient at theta=1.0: {grad_at_1}")

# Compute gradient at theta=50
grad_at_50 = grad_fn(50.0)
print(f"Gradient at theta=50.0: {grad_at_50}")

# Check numerical gradient
eps = 0.01
log_lik_plus = log_likelihood(true_theta[0] + eps)
log_lik_minus = log_likelihood(true_theta[0] - eps)
numerical_grad = (log_lik_plus - log_lik_minus) / (2 * eps)
print(f"\nNumerical gradient at TRUE theta: {numerical_grad}")
print(f"JAX gradient at TRUE theta: {grad_at_true}")
print(f"Difference: {abs(numerical_grad - grad_at_true)}")

# Test if gradient points toward MLE
print(f"\nGradient direction test:")
print(f"  At theta=1 (too low): grad={grad_at_1:.6f} (should be positive)")
print(f"  At theta=10 (true): grad={grad_at_true:.6f} (should be ~0)")
print(f"  At theta=50 (too high): grad={grad_at_50:.6f} (should be negative)")

