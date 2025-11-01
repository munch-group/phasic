# Bug Fix: PDF Incorrectly Computed from Reward-Transformed Graph

**Date**: 2025-01-01
**Version**: 0.22.0
**Status**: ✅ Fixed

## Problem

User reported: "it converges at about 20, rather than the correct value 10"

This indicated SVGD inference was converging to incorrect parameter values.

## Root Cause

**File**: `src/cpp/parameterized/graph_builder.cpp`
**Function**: `compute_pmf_and_moments()`

The bug was in how PDF and moments were computed when rewards were provided:

### Incorrect Implementation (BEFORE)

```cpp
// WRONG: Compute PDF from reward-transformed graph
Graph g_transformed = g.reward_transform(rewards_2d[j]);
pmf_2d[t][j] = g_transformed.pdf(times_vec[t], granularity);
moments_2d[j] = compute_moments_impl(g_transformed, nr_moments, rewards_2d[j]);
//                                                             ^^^^^^^^^^^^^^
//                                                             Also passed rewards again!
```

This had TWO bugs:
1. **PDF computed from transformed graph** → Wrong! PDF should represent P(T=t|θ), not some transformed distribution
2. **Rewards passed twice** → Moments were computed on already-transformed graph with rewards applied again

## Fix

### Correct Implementation (AFTER)

```cpp
// CORRECT: Compute PDF from original graph
double pdf_val = g.pdf(times_vec[t], granularity);
// Broadcast to all features (PDF is same for all)
for (size_t j = 0; j < n_features; j++) {
    pmf_2d[t][j] = pdf_val;
}

// Compute moments per feature from transformed graph
for (size_t j = 0; j < n_features; j++) {
    Graph g_transformed = g.reward_transform(rewards_2d[j]);
    // Pass empty vector - transformation already applied
    moments_2d[j] = compute_moments_impl(g_transformed, nr_moments, std::vector<double>());
}
```

## Why This Matters

In SVGD inference:

1. **PDF is used for likelihood**: log P(data | θ)
   - Must be computed from the ORIGINAL graph
   - Represents actual probability of observing the data
   - Reward transformation changes this → wrong likelihood → wrong inference

2. **Moments are used for regularization**: E[R·T^k]
   - Computed from REWARD-TRANSFORMED graph
   - Provides additional constraints on parameter estimates
   - Different rewards → different moment expectations

## Test Results

### Before Fix

```python
# Exponential(rate=2.0) with reward R=5
PMF with rewards:    [0.327, 0.268, 0.220]  # ✗ WRONG
PMF without rewards: [0.736, 0.271, 0.099]  # ✓ Baseline

# Moments
E[R*T] = 12.5  # ✗ WRONG (R² / rate = 25/2)
Expected: 2.5  # (R / rate = 5/2)
```

### After Fix

```python
# Exponential(rate=2.0) with reward R=5
PMF with rewards:    [0.736, 0.271, 0.099]  # ✓ CORRECT
PMF without rewards: [0.736, 0.271, 0.099]  # ✓ Same!

# Moments
E[R*T] = 2.5  # ✓ CORRECT
Expected: 2.5
```

## Multivariate Behavior

With 2D rewards, the **correct** behavior is:

```python
rewards_2d = [[0, 0, 0],      # Vertex 0
              [1, 5, 10],     # Vertex 1 (transient)
              [0, 0, 0]]      # Vertex 2

pmf, moments = model(theta, times, rewards=rewards_2d)

# PMF: identical for all features (broadcast from original graph)
assert pmf[:, 0] == pmf[:, 1] == pmf[:, 2]  # ✓

# Moments: differ by feature (reward-transformed)
assert moments[0, 0] == 0.5   # R=1, E[R*T] = 1/2
assert moments[1, 0] == 2.5   # R=5, E[R*T] = 5/2
assert moments[2, 0] == 5.0   # R=10, E[R*T] = 10/2
```

## Files Modified

1. **`src/cpp/parameterized/graph_builder.cpp`** (lines 422-477):
   - Separated PDF and moment computation
   - PDF computed once from original graph, broadcasted to all features
   - Moments computed per-feature from reward-transformed graphs
   - Fixed double-application of rewards bug

2. **`MULTIVARIATE_REWARDS_IMPLEMENTATION.md`**:
   - Added "Correct Behavior" section
   - Documented PDF invariance principle
   - Added bug fix summary

## Related Issues

- Multivariate rewards implementation (v0.22.0)
- SVGD convergence to incorrect values
- Reward transformation semantics

## Verification

Run `test_final_verification.py` to verify:
- ✓ PDF invariant to reward transformation
- ✓ Moments correctly transformed
- ✓ Multivariate: PMF broadcasted, moments differ

All tests pass.
