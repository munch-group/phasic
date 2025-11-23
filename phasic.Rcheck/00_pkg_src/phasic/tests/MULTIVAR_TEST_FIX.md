# Fix for tests/multivar_test.py

## Issues Found

1. **Array shapes need transposing**
   - `rewards` is created as `(n_features, n_vertices)` but needs `(n_vertices, n_features)`
   - `observed_data` is created as `(n_features, n_observations)` but needs `(n_observations, n_features)`

2. **Incompatible default parameters**
   - Default `regularization=10` but `nr_moments=0`
   - Can't do moment regularization with 0 moments!

3. **Model auto-detection**
   - Added: `graph.svgd()` now auto-detects 2D rewards and uses multivariate model

## Required Changes

```python
# In tests/multivar_test.py

# Change 1: Transpose rewards
rewards = _graph.states().T
rewards = rewards[:-2]
rewards = rewards.T  # ADD THIS LINE: (n_features, n_vertices) -> (n_vertices, n_features)

# Change 2: Transpose observed_data
observed_data = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards])
observed_data = observed_data.T  # ADD THIS LINE: (n_features, n_observations) -> (n_observations, n_features)

# Change 3: Fix parameters
params = dict(
            observed_data=observed_data,  # Now transposed
            bandwidth='median',
            theta_dim=len(true_theta),
            prior=uninformative_prior,
            n_particles=24,
            n_iterations=n_iterations,
            seed=42,
            verbose=False,
            regularization=0,  # ADD THIS: Set to 0, or set nr_moments=2+
            nr_moments=0,      # Or change this to 2+ if using regularization
            rewards=rewards,   # Now transposed
)
```

## Complete Fixed Version

```python
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


true_theta = np.array([10.0])  # Make sure it's float
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

nr_observations = 1000
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)

# Get rewards and TRANSPOSE
rewards = _graph.states().T
rewards = rewards[:-2]
rewards = rewards.T  # TRANSPOSE: (n_features, n_vertices) -> (n_vertices, n_features)

# Sample data and TRANSPOSE
observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards.T])
observed_data = observed_data_raw.T  # TRANSPOSE: (n_features, n_observations) -> (n_observations, n_features)

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

n_iterations = 20

params = dict(
    observed_data=observed_data,  # Transposed!
    bandwidth='median',
    theta_dim=len(true_theta),
    prior=uninformative_prior,
    n_particles=24,
    n_iterations=n_iterations,
    seed=42,
    verbose=True,
    regularization=0,  # FIXED: Set to 0 (or use nr_moments=2+)
    nr_moments=0,      # Or set to 2+ and keep regularization=10
    rewards=rewards,   # Transposed!
)

# graph.svgd() now auto-detects 2D rewards and uses multivariate model
svgd = graph.svgd(**params)

print(f"True theta: {true_theta}")
print(f"Posterior mean: {svgd.theta_mean}")
print(f"Posterior std: {svgd.theta_std}")
```

## Why These Changes?

### Shape Requirements

Multivariate model expects:
- **rewards**: `(n_vertices, n_features)` - each column is a reward vector for one feature
- **observed_data**: `(n_observations, n_features)` - each row is one multivariate observation

Your code creates:
- **rewards**: `(n_features, n_vertices)` - transposed!
- **observed_data**: `(n_features, n_observations)` - transposed!

### Parameter Fix

The defaults in `graph.svgd()` are:
- `regularization=10` - enables moment regularization
- `nr_moments=0` - but with 0 moments!

This is incompatible! Either:
- Set `regularization=0` (no regularization), OR
- Set `nr_moments=2` or higher (compute moments for regularization)

## Testing

After making these changes, the test should run without segfaults and return posterior estimates.
