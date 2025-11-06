# Multivariate SVGD Convergence Issue

## Problem Statement

Multivariate SVGD with 3 features converges to θ ≈ 11 instead of true θ = 10.

- Univariate SVGD: 3.1% error (correct)
- Multivariate 1 feature: 3.1% error (correct)
- Multivariate 3 features: 11.0% error (BUG!)

## Test Setup

### Model
- Coalescent model with nr_samples=4
- Single parameter θ scaling all transition rates
- True θ = 10

### Data Generation
```python
graph.update_parameterized_weights(np.array([10.0]))
rewards = graph.states()[:, :-2]  # Shape: (6, 3)

# Rewards matrix:
# [[0, 0, 0],
#  [4, 0, 0],
#  [2, 1, 0],
#  [0, 2, 0],
#  [1, 0, 1],
#  [0, 0, 0]]

# Generate 10,000 samples per feature
for i in range(3):
    observed_data[:, i] = graph.sample(10000, rewards=rewards[:, i])
```

### Observed Means
- Feature 0: 0.199 (reward=[0,4,2,0,1,0], E[X]=0.200)
- Feature 1: 0.101 (reward=[0,0,1,2,0,0], E[X]=0.100)
- Feature 2: 0.068 (reward=[0,0,0,0,1,0], E[X]=0.067)

All observed means match theoretical expectations at θ=10.

## Investigation Results

### 1. Model Implementation is Correct

The multivariate model (`pmf_and_moments_from_graph_multivariate`) correctly:
- Extracts each feature's reward vector
- Computes PMF for each feature using its reward
- Returns 2D PMF matrix (n_times, n_features)

### 2. PMF Values Differ Across Features

At t=0.15 with θ=10:
- Feature 0 PMF: 3.534
- Feature 1 PMF: 1.146
- Feature 2 PMF: 1.500

This is EXPECTED and CORRECT - different reward vectors produce different marginal distributions.

### 3. Log-Likelihood is Computed Correctly

```python
log_lik = jnp.sum(jnp.where(mask, jnp.log(pmf_vals + 1e-10), 0.0))
```

Sums over all elements: `Σᵢⱼ log(PMF[i,j])`

This is the correct log-likelihood for independent observations.

### 4. `graph.svgd()` Auto-Detects 2D Rewards

```python
if rewards_arr.ndim == 2:
    model = Graph.pmf_and_moments_from_graph_multivariate(...)
```

The convenience method correctly selects the multivariate model.

## Hypothesis: Likelihood Weighting Issue?

When combining likelihoods from features with different PMF scales:
- Feature 0 has higher PMF values (range: 0-4)
- Features 1 & 2 have lower PMF values (range: 0-2)

In log-space:
- Feature 0: log(3.5) ≈ 1.25
- Feature 1: log(1.1) ≈ 0.10
- Feature 2: log(1.5) ≈ 0.41

Feature 0 dominates the log-likelihood sum, potentially biasing the inference.

### Test This Hypothesis

If Feature 0 has 3x more observations (30k vs 10k each for features 1&2):
- Total observations: 30k + 10k + 10k = 50k
- Feature 0 contribution: ~3x larger
- Could explain bias toward Feature 0's optimal θ

But in our test, each feature has exactly 10k observations, so this shouldn't happen.

## Alternative Hypothesis: Data Generation Artifact?

The data is generated with a NaN pattern:
```python
observed_data = [
    [t00, NaN, NaN],  # Observations 0-9999
    [NaN, t11, NaN],  # Observations 10000-19999
    [NaN, NaN, t22],  # Observations 20000-29999
]
```

This creates 30k "observations" where each has only 1 non-NaN value.

The log-likelihood computation:
```python
mask = ~jnp.isnan(pmf_vals)
log_lik = jnp.sum(jnp.where(mask, jnp.log(pmf_vals), 0.0))
```

Correctly handles NaNs by masking them out. So 30k observations with NaN pattern should be equivalent to 3 separate datasets of 10k each.

## Next Steps

1. **Verify likelihood computation manually**
   - Compute log-likelihood at θ=10 and θ=11
   - Check which has higher likelihood
   - If θ=11 has higher likelihood, then SVGD is correct (data was mislabeled?)

2. **Test with equal expected values**
   - Generate data where all 3 features use rewards that give SAME E[X]
   - This would eliminate any distribution shape differences
   - SVGD should converge perfectly to θ=10

3. **Test with independent datasets**
   - Instead of 30k observations with NaN pattern
   - Use 3 separate SVGD runs with 10k observations each
   - Compare convergence

## Current Status

- Implementation appears correct
- Convergence to θ≈11 instead of θ=10 is unexplained
- Need further investigation to identify root cause
