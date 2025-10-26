# Default Rewards Fix

**Date**: 2025-10-26
**Issue**: Rewards were required even when user wanted neutral (ones) transformation
**Status**: ✅ FIXED

## Problem

The trace-based reward transformation implementation required explicit `rewards` parameter even when users wanted the default behavior (neutral rewards = vector of ones). This created unnecessary boilerplate:

```python
# Before fix - required boilerplate
trace = record_elimination_trace(graph, param_length=2, enable_rewards=True)
rewards = np.ones(trace.n_vertices)  # User had to create this
graph_inst = instantiate_from_trace(trace, params=theta, rewards=rewards)
```

Additionally, sparse observations with NaNs (common in multivariate models) needed to be supported properly.

## Solution

### 1. Default Rewards to Ones

Modified `evaluate_trace()` and `evaluate_trace_jax()` to automatically default `rewards=None` to `ones(n_vertices)`:

**File**: `src/phasic/trace_elimination.py`

**Changes**:
- Lines 814-818: Changed from raising error to defaulting to ones
- Lines 1256-1262: Same for JAX version (using `jnp.ones`)
- Updated docstrings to reflect optional nature of rewards parameter

**Before**:
```python
if trace.reward_length > 0:
    if rewards is None:
        raise ValueError("Rewards required for reward-transformed trace")
```

**After**:
```python
if trace.reward_length > 0:
    if rewards is None:
        # Default to ones (neutral rewards)
        rewards = np.ones(trace.n_vertices, dtype=np.float64)
```

### 2. NaN Handling Verified

Confirmed that `_log_prob_unified()` in `svgd.py` already properly handles NaN observations via masking:

```python
# Log-likelihood term (handle missing data via NaN)
mask = ~jnp.isnan(pmf_vals)
log_lik = jnp.sum(jnp.where(mask, jnp.log(pmf_vals + 1e-10), 0.0))
```

This enables sparse observation patterns like:
```python
# Multivariate observations with NaNs
observed_data = jnp.array([
    [1.5, nan, nan],  # Only feature 0 observed
    [nan, 2.3, nan],  # Only feature 1 observed
    [nan, nan, 0.8],  # Only feature 2 observed
])
```

## API Changes

### evaluate_trace()

**Signature**: No change
```python
def evaluate_trace(trace, params=None, rewards=None)
```

**Behavior Change**:
- **Before**: Raises `ValueError` if `rewards=None` and `trace.reward_length > 0`
- **After**: Defaults to `np.ones(trace.n_vertices)` if `rewards=None`

### evaluate_trace_jax()

**Signature**: No change
```python
def evaluate_trace_jax(trace, params, rewards=None)
```

**Behavior Change**:
- **Before**: Raises `ValueError` if `rewards=None` and `trace.reward_length > 0`
- **After**: Defaults to `jnp.ones(trace.n_vertices)` if `rewards=None`

### instantiate_from_trace()

**Signature**: No change
```python
def instantiate_from_trace(trace, params=None, rewards=None)
```

**Behavior Change**:
- Inherits default from `evaluate_trace()` (calls it internally)
- No explicit change needed

## New Usage Patterns

### Simple Case (Neutral Rewards)

```python
# After fix - clean and simple
trace = record_elimination_trace(graph, param_length=2, enable_rewards=True)
graph_inst = instantiate_from_trace(trace, params=theta)  # rewards default to ones
pdf = graph_inst.pdf(1.0)
```

### Custom Rewards

```python
# Custom rewards still work as before
trace = record_elimination_trace(graph, param_length=2, enable_rewards=True)
custom_rewards = np.array([1.0, 2.0, 0.5, 1.5])
graph_inst = instantiate_from_trace(trace, params=theta, rewards=custom_rewards)
```

### Multivariate with Sparse Observations

```python
# NaN handling for sparse multivariate observations
model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2)

# Create sparse observations (most values are NaN)
n_observations = 10
n_features = 3
observations = np.full((n_observations * n_features, n_features), np.nan)
for i in range(n_features):
    observations[i*n_observations:(i+1)*n_observations, i] = data_for_feature_i

# SVGD handles NaNs automatically via masking
svgd = SVGD(model, observed_data=observations, theta_dim=2, rewards=rewards_2d)
svgd.optimize()
```

## Testing

Created comprehensive tests in `tests/`:

### test_default_rewards.py

Tests that verify:
1. Default rewards work in NumPy mode
2. Default rewards work in JAX mode
3. Default rewards work in graph instantiation
4. Scaled rewards produce different results than defaults
5. Backward compatibility maintained (traces without rewards)

**Results**: ✅ All tests pass

### test_nan_observations.py

Tests that verify:
1. Model evaluation with NaN observations
2. SVGD optimization with sparse observations (NaNs)
3. Proper handling of 2D rewards with multivariate data

**Results**: ✅ All tests pass

Example output:
```
✓ Model evaluation successful!
  PMF shape: (30, 3)
  Number of NaN PMF values: 0
  Number of non-NaN PMF values: 90

✓ SVGD optimization completed!
  Posterior mean: [0.64067555]
  Posterior std: [0.65207293]
```

## Performance Impact

**None** - Default creation is O(n) where n = number of vertices, negligible compared to trace evaluation O(n²) or recording O(n³).

## Backward Compatibility

✅ **Fully backward compatible**

- Existing code that provides explicit rewards continues to work
- Traces without rewards (`enable_rewards=False`) continue to work
- No API signature changes

## Files Modified

1. **src/phasic/trace_elimination.py**:
   - `evaluate_trace()`: Default rewards to ones (lines 814-818)
   - `evaluate_trace_jax()`: Default rewards to ones (lines 1256-1262)
   - Updated docstrings for all three functions

2. **TRACE_REWARDS_IMPLEMENTATION.md**:
   - Updated API documentation to reflect optional nature of rewards

3. **tests/test_default_rewards.py** (new):
   - Comprehensive test suite for default rewards behavior

4. **tests/test_nan_observations.py** (new):
   - Test suite for NaN handling in multivariate SVGD

## Conclusion

Default rewards simplify the API by eliminating boilerplate while maintaining full flexibility for custom reward transformations. NaN handling enables efficient sparse multivariate inference patterns.

**Key Achievement**: Users can now omit `rewards` parameter for neutral transformation, reducing code verbosity while maintaining correctness.

---

*Implementation completed: 2025-10-26*
