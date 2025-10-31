# Sparse Rewards Bug: PDF Normalization Issue

## Date
2025-10-27

## Executive Summary

**CONFIRMED BUG**: Phase-type distributions with sparse reward vectors have PDFs that are **not properly normalized**, causing SVGD to converge to incorrect parameter values with ~20% bias.

## Symptoms

With coalescent model (nr_samples=6, true θ=10):

| Feature | Sparsity | Non-zero | SVGD Estimate | Bias | PDF Integral | KS Test |
|---------|----------|----------|---------------|------|--------------|---------|
| 0       | 42%      | 7/12     | θ=10 ✓        | 0%   | 1.00 ✓       | p=0.81 ✓|
| 1       | 58%      | 5/12     | θ=10 ✓        | 0%   | ?            | ?       |
| 2       | 75%      | 3/12     | θ=12 ❌       | +20% | ?            | ?       |
| 3       | 83%      | 2/12     | θ=12 ❌       | +20% | ?            | ?       |
| 4       | 92%      | 1/12     | θ=12 ❌       | +20% | 0.41 ❌      | p=0.00 ❌|

**Pattern**: Features with >75% sparsity (≤3 non-zero rewards) have:
- PDFs that don't integrate to 1.0
- PDFs that don't match samples (KS test fails)
- SVGD estimates biased +20%

## Root Cause

The bug is in **PDF computation after reward_transform**, not in reward_transform itself:

### What Works Correctly ✓

1. **Moment computation**: E[T|θ] scales correctly (∝ 1/θ) for all reward vectors
2. **Sampling**: Sample means scale correctly when rewards are doubled/tripled
3. **Reward scaling**: When rewards are multiplied by k, E[T] multiplies by k (within 5%)
4. **Dense rewards**: PDFs are properly normalized and match samples

### What's Broken ❌

**Sparse reward PDFs**:
- Feature 4 (1 non-zero): PDF integral = 0.41 (should be 1.0)
- KS test p-value = 0.000 (PDF doesn't match samples)
- Likelihood peaks at wrong θ value

## Evidence

### Test 1: PDF Normalization
```python
# Dense reward (Feature 0: 7/12 non-zero)
PDF integral: 1.00 ✓
KS test: p=0.805 ✓

# Sparse reward (Feature 4: 1/12 non-zero)
PDF integral: 0.41 ❌
KS test: p=0.000 ❌
```

### Test 2: Likelihood Peaks
```python
# True θ = 10, 1000 observations per feature

Dense rewards (Features 0-1):
  Peak at θ=10 ✓

Sparse rewards (Features 2-4):
  Peak at θ=12 ❌ (bias: +20%)
```

### Test 3: Multivariate SVGD
```python
# Combined all 5 features
SVGD estimate: θ=12 ❌ (should be θ=10)

# This is because 3/5 features are biased,
# and their combined likelihood dominates
```

## Technical Details

### reward_transform Implementation

From `src/c/phasic.c` lines 2029-2038:
```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (__rewards[original_indices[i]] <= REWARD_EPSILON) {
        rewards[i] = 0;
    } else {
        rewards[i] = __rewards[original_indices[i]];
    }

    if (graph->starting_vertex == vertices[i] || vertices[i]->edges_length == 0) {
        rewards[i] = 1;
    }
}
```

Then lines 2152-2154:
```c
if (rewards[i] != 0) {
    continue;  // Skip elimination for non-zero rewards
}

// For zero-reward vertices: eliminate via graph reduction
```

**Hypothesis**: When reward vector has many zeros, too many vertices are eliminated, causing the PDF to lose mass and not normalize to 1.0.

### Why Moments Are Correct But PDFs Are Wrong

Moments use **expected_waiting_time()** which works correctly even with sparse rewards.

PDFs use **graph.pdf()** after **reward_transform()** which appears to have issues with sparse rewards.

## Impact

This bug affects:
- **Multivariate SVGD**: When using different reward vectors per feature
- **Sparse observations**: Site frequency spectra, coalescent models
- **Any model**: Where reward vectors have many zeros

**Severity**: HIGH - causes 20% parameter estimation bias

## Proposed Fix

Need to investigate why `reward_transform()` produces non-normalized PDFs for sparse rewards. Possible causes:

1. **Graph elimination bug**: Vertices with zero rewards are eliminated, but the resulting graph isn't properly re-normalized
2. **PDF computation bug**: The `pdf()` method doesn't handle reward-transformed graphs correctly
3. **Rate computation bug**: After reward transformation, rates might not be correctly computed

## Workaround

Until fixed, avoid using sparse reward vectors (>75% zeros). Use dense rewards or aggregate features differently.

## Next Steps

1. Debug reward_transform to understand why PDFs aren't normalized
2. Check if the issue is in graph elimination or PDF computation
3. Add normalization check after reward_transform
4. Add tests for sparse reward vectors

## Related Issues

- NaN handling bug (fixed): Caused 100% error with multivariate data
- Parameterized weight bugs (fixed): Missing base_weight and serialization issues

This sparse rewards bug is independent of those fixes.
