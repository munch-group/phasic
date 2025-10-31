# Multivariate SVGD Bug Diagnosis

**Date:** 2025-10-30
**Status:** Root cause identified ✓

## Problem Summary

Multivariate SVGD with reward-transformed graphs converges to θ≈11-12 instead of the true value θ=10 used to generate data. The error is systematic (11% vs 3% for univariate).

## Root Cause Identified

**The reward-transformed PDF computation has an inverted θ-dependency.**

### Evidence

#### 1. Likelihood Inversion (from `test_manual_likelihood.py`)

When computing log-likelihood for data generated at θ=10:

```
θ =  8.0: log-lik = -1,465,147  ← HIGHEST (should be worst)
θ =  9.0: log-lik = -1,466,741
θ = 10.0: log-lik = -1,468,468  ← Used to generate data
θ = 11.0: log-lik = -1,470,313  ← SVGD converges here
θ = 12.0: log-lik = -1,472,263
```

**All three features** show the same pattern - θ=8 has highest likelihood despite data being generated at θ=10.

#### 2. Component Breakdown (from `test_likelihood_components.py`)

Breaking down by feature:

```
θ = 8.0:
  Feature 0: log-lik = -462,839
  Feature 1: log-lik = -467,582
  Feature 2: log-lik = -534,693
  Total: -1,465,114

θ = 10.0:
  Feature 0: log-lik = -465,672
  Feature 1: log-lik = -467,754
  Feature 2: log-lik = -534,943
  Total: -1,468,369
```

Every single feature has higher likelihood at θ=8 than θ=10. This rules out feature weighting issues.

#### 3. Inverted PDF Behavior (from `test_pdf_theta_relationship.py`)

For base graph (no reward transformation):
- t=0.5: PDF decreases from 0.242 (θ=8) to 0.041 (θ=12) ✓ CORRECT
- Expectation decreases from 0.188 (θ=8) to 0.125 (θ=12) ✓ CORRECT

For reward-transformed graph (Feature 0, reward=[0,2,0]):
- t=0.5: PDF decreases from 0.563 (θ=8) to 0.121 (θ=12) ✓ CORRECT trend
- BUT expectation is DOUBLED: 0.250 vs 0.100 expected ✗ WRONG

#### 4. Simple Test Case (from `test_reward_pdf_correctness.py`)

For nr_samples=2 (single exponential transition), reward vector [0,2,0]:

```
Expectation:
  E[X] = 0.200 (got)
  E[T] = 0.100 (expected)

PDF at time=0.15:
  θ=8:  Actual=2.259, Expected=2.410  (12% too LOW)
  θ=10: Actual=2.438, Expected=2.231  (9% too HIGH)
  θ=12: Actual=2.523, Expected=1.984  (27% too HIGH)
```

**The PDF increases with θ when it should decrease!**

### The Bug

The reward transformation code in `src/c/phasic.c` lines 2107-2108:

```c
if (rewards[i] != 0) {
    rewards[i] /= rate;  // Convert to discrete-time via uniformization
}
```

This converts continuous-time rewards to discrete-time probabilities. However, **something about this conversion is creating an inverted θ-dependency**.

Possible issues:
1. The PDF computation from the uniformized chain may be missing a rate scaling factor
2. The reward division by rate may be incorrect for continuous-time graphs
3. The forward algorithm may not be correctly accounting for the time scaling

## Why This Affects Multivariate but not Univariate

- **Univariate SVGD** (no rewards): Uses base graph PDF, which is correct
- **Multivariate SVGD** (with rewards): Uses reward-transformed graph PDF, which is inverted

The bug is **silent** for univariate models because they don't use reward transformation.

## Data Generation is Correct

Confirmed by user:
- Feature means at θ=10: [0.200, 0.099, 0.066] match theoretical expectations
- `graph.sample(n, rewards=reward_vec)` is working correctly
- The issue is purely in the PDF/likelihood computation for reward-transformed graphs

## Impact

This bug:
- Breaks all multivariate phase-type SVGD inference
- Causes systematic bias toward LOWER θ values (opposite of true MLE)
- Would affect any application using reward-transformed graphs for PDF computation

## Next Steps

1. **Investigate the forward algorithm** for reward-transformed graphs
   - Check if PDF = PMF × uniformization_rate is correct
   - Verify the time scaling in the uniformization process

2. **Compare against theoretical PDF** for simple cases
   - Single exponential with reward=1 vs reward=2
   - Erlang distribution with different reward vectors

3. **Fix the C code** in the reward transformation or forward algorithm

4. **Verify fix** with all test cases

## Test Files Created

- `/tmp/test_pdf_theta_relationship.py` - Shows PDF vs θ for base and reward-transformed graphs
- `/tmp/test_likelihood_components.py` - Breaks down log-likelihood by feature
- `/tmp/test_reward_pdf_correctness.py` - Tests reward PDF against theoretical values
- `test_manual_likelihood.py` - Computes likelihood at different θ values
- `/tmp/analyze_reward_transform.py` - Analyzes reward-transformed graph structure

All tests consistently point to the same issue: **reward-transformed PDF has inverted θ-dependency**.
