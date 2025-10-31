"""Test if gradients are correct for multivariate rewards case"""
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

print("Testing gradient computation for multivariate model...")

true_theta = np.array([10.0])
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)

# Create test data
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

# Sample dataset
np.random.seed(42)
observed_data = jnp.array([
    _graph.sample(100, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T

print(f"Observed data shape: {observed_data.shape}")
print(f"Rewards shape: {rewards.shape}")

# Create model
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False, use_ffi=False, param_length=1
)

# Define log-likelihood function (with softplus like SVGD does)
def log_likelihood(phi):
    """phi is unconstrained, theta = softplus(phi)"""
    theta = jax.nn.softplus(phi)
    pmf, _ = model(theta, observed_data, rewards=rewards)
    return jnp.sum(jnp.log(pmf + 1e-10))

# Compute gradient at various phi values
# If phi=1, theta=softplus(1)≈1.3
# If phi=3, theta=softplus(3)≈3.05
# If phi=6, theta=softplus(6)≈6.00
# If phi=700, theta=softplus(700)≈700

grad_fn = jax.grad(log_likelihood)

test_phis = [
    (jnp.array([0.0]), "phi=0 (theta≈0.7)"),
    (jnp.array([1.0]), "phi=1 (theta≈1.3)"),
    (jnp.array([2.0]), "phi=2 (theta≈2.1)"),
    (jnp.array([3.0]), "phi=3 (theta≈3.0)"),
    (jnp.array([6.0]), "phi=6 (theta≈6.0)"),
    (jnp.array([10.0]), "phi=10 (theta≈10.0)"),
    (jnp.array([50.0]), "phi=50 (theta≈50)"),
    (jnp.array([700.0]), "phi=700 (theta≈700)"),
]

print("\nGradient check (should be positive for low values, negative for high values):")
for phi, label in test_phis:
    theta = jax.nn.softplus(phi)
    grad = grad_fn(phi)
    log_lik = log_likelihood(phi)
    print(f"  {label:25s}: grad={grad[0]:10.4f}, log_lik={log_lik:10.2f}")

print("\nExpected behavior:")
print("  - At phi<2 (theta<2): gradient should be POSITIVE (move toward higher theta)")
print("  - At phi≈2-3 (theta≈2-3): gradient should be near zero (at optimum)")
print("  - At phi>10 (theta>10): gradient should be NEGATIVE (move toward lower theta)")
print("  - At phi=700: gradient should be VERY NEGATIVE")
