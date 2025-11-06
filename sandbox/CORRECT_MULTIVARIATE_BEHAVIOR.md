# Correct Multivariate Rewards Behavior

**Date**: 2025-01-01
**Version**: 0.22.0
**Status**: ✅ Verified

## Key Insight

Reward transformation **changes the distribution**, including the PDF. This is intentional and correct.

## Use Case: Multivariate SVGD

### Scenario

You have observations sampled from **different reward-transformed distributions**:

- Feature 0: Data ~ RewardTransform(Original, R₀)
- Feature 1: Data ~ RewardTransform(Original, R₁)
- Feature 2: Data ~ RewardTransform(Original, R₂)

Each feature has:
- **Different observations** (different means due to different rewards)
- **Same underlying parameter θ** (e.g., coalescence rate)

### Goal

SVGD should **estimate the same θ** for all features by:
1. Using the correct reward transformation for each feature's likelihood
2. Computing: `Σⱼ log P(dataⱼ | θ, Rⱼ)`

## Correct Implementation

### PDF Computation

**PDF IS affected by reward transformation** ✓

```cpp
// For each feature j:
Graph g_transformed = g.reward_transform(rewards[j]);
pmf[t][j] = g_transformed.pdf(times[t], granularity);
```

This gives **different PDFs** for different features because they represent different distributions.

### Example

```python
# Exponential distribution with rate θ = 2.0
# 3 features with rewards R = [1.0, 5.0, 10.0]

rewards_2d = [[0, 0, 0],      # Vertex 0 (absorbing)
              [1, 5, 10],     # Vertex 1 (transient)
              [0, 0, 0]]      # Vertex 2 (absorbing)

pmf, moments = model(theta=2.0, times=[0.5, 1.0, 1.5], rewards=rewards_2d)

# PDFs DIFFER across features (CORRECT!)
assert pmf[0, 0] != pmf[0, 1] != pmf[0, 2]  # ✓

# Feature 0 (R=1): PDF @ t=0.5 = 0.736
# Feature 1 (R=5): PDF @ t=0.5 = 0.328
# Feature 2 (R=10): PDF @ t=0.5 = 0.181

# Moments also differ (expected values of transformed distributions)
assert moments[0, 0] == 0.5    # E[T] for R=1
assert moments[1, 0] == 2.5    # E[T] for R=5
assert moments[2, 0] == 5.0    # E[T] for R=10
```

## Why This Works

### Data Generation

```python
# Sample from reward-transformed graph
g_transformed_R5 = g.reward_transform([0, 5, 0])
data_feature1 = g_transformed_R5.sample(100)
# → mean ≈ 2.5 (for θ=2.0, R=5)
```

### Inference

```python
# SVGD maximizes likelihood using same transformation
log_lik = Σᵢ log( g.reward_transform([0, 5, 0]).pdf(data[i]) )
```

Since we apply the **same transformation** used to generate data, we correctly recover θ.

## Common Confusion

### ❌ Incorrect Understanding

"Reward transformation only affects moments, not PDF"

→ This is **wrong**. Reward transformation creates a new distribution with different PDF.

### ✓ Correct Understanding

"Reward transformation changes the entire distribution (PDF and moments)"

→ This is **correct**. The transformed graph represents a different phase-type distribution.

## Mathematical Interpretation

For exponential(λ) with reward R on transient state:

**Original distribution**:
- PDF: f(t) = λ exp(-λt)
- E[T] = 1/λ

**Reward-transformed distribution** (R=5):
- PDF: f'(t) = different function
- E[T] = 5/λ

The transformed distribution is a **different phase-type distribution**, not just the original with scaled moments.

## SVGD Likelihood

```python
# Multivariate likelihood:
log_lik(θ) = Σⱼ Σᵢ log P(dataⱼᵢ | θ, Rⱼ)

# Each term uses the reward-transformed PDF:
P(dataⱼᵢ | θ, Rⱼ) = graph.reward_transform(Rⱼ).pdf(dataⱼᵢ, θ)
```

This ensures:
- Different features use appropriate transformations
- Same θ estimated across all features
- Correct accounting for how data was generated

## Verification

Run `test_multivariate_svgd_concept.py`:

```
✓ PDFs differ across features (0.736, 0.328, 0.181)
✓ Moments match expected values (0.5, 2.5, 5.0)
✓ Likelihood correctly computed from transformed graphs
```

## Summary

The implementation is **CORRECT**:

1. ✓ PDF computed from reward-transformed graph
2. ✓ Different PDFs for different features
3. ✓ Enables correct SVGD inference with multivariate reward-transformed data
4. ✓ Same θ estimated across features despite different observation means

The previous "fix" that made PDF invariant to rewards was **WRONG** and has been reverted.
