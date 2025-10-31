# Complete Diagnosis: Vertex Bypass Bug in reward_transform

**Date**: 2025-10-27
**Status**: 🔴 CRITICAL BUG CONFIRMED

---

## Executive Summary

**CONFIRMED**: The vertex bypass algorithm in `_ptd_graph_reward_transform()` (src/c/phasic.c:2151-2230) has a critical bug that causes PDF normalization failure when **3 or more consecutive vertices** have zero rewards.

**Impact**:
- Sparse reward features with long zero-runs have PDFs with integral ≈ 0 (should be 1.0)
- SVGD parameter estimates biased by ~20% for sparse features
- Coalescent model with nr_samples=6: Features 2-4 peak at θ=12 instead of θ=10

**Root Cause**: Single-pass processing doesn't update parent-child relationships between vertex eliminations.

---

## Evidence Summary

### Test 1: Simple Linear Chain

**Setup**: Chain of 5 vertices (0→1→2→3→4→absorbing)

| Test Case | Rewards | Bypassed | PDF Integral | Status |
|-----------|---------|----------|--------------|--------|
| No bypass | [1,1,1,1,1] | 0 | 1.000 | ✓ |
| 1 bypass | [1,1,0,1,1] | 1 | 0.999 | ✓ |
| 2 bypasses | [1,1,0,0,1] | 2 | 0.992 | ✓ |
| **3 bypasses** | [1,0,0,0,1] | 3 | **0.000** | **❌** |

**Conclusion**: Bug triggers at 3+ consecutive bypasses.

### Test 2: Coalescent Model (nr_samples=6, θ=10)

**12 vertices, 5 reward features**:

| Feature | Sparsity | Non-zero | Consecutive Zeros | SVGD Peak | PDF Integral | Status |
|---------|----------|----------|-------------------|-----------|--------------|--------|
| 0 | 42% | 7/12 | ~2 | θ=10 | 1.00 | ✓ |
| 1 | 58% | 5/12 | ~2-3 | θ=10 | ? | ✓ |
| 2 | 75% | 3/12 | ~4 | θ=12 | ? | ❌ (+20%) |
| 3 | 83% | 2/12 | ~5 | θ=12 | ? | ❌ (+20%) |
| 4 | 92% | 1/12 | ~6 | θ=12 | 0.41 | ❌ (+20%) |

**Pattern**: Features with ≥4 consecutive zeros have biased estimates.

### Test 3: Reward Scaling

**All features tested with 2× reward multiplication**:

| Feature | Sparsity | Sample Mean Ratio (2×/1×) | Expected | Status |
|---------|----------|---------------------------|----------|--------|
| 0 | 42% | 2.045 | 2.0 | ✓ |
| 1 | 58% | 1.999 | 2.0 | ✓ |
| 2 | 75% | 2.052 | 2.0 | ✓ |
| 3 | 83% | 2.015 | 2.0 | ✓ |
| 4 | 92% | 2.037 | 2.0 | ✓ |

**Conclusion**:
- ✅ **Sampling** respects rewards correctly (all ratios ≈ 2.0)
- ✅ **Moments** work correctly with sparse rewards
- ❌ **PDFs** are broken for sparse rewards with long zero-runs

---

## The Bug: Line-by-Line Analysis

### Location: src/c/phasic.c lines 2151-2230

```c
// Line 2151: Process ALL zero-reward vertices in SINGLE pass
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) {
        continue;  // Skip non-zero rewards
    }

    // Lines 2156-2230: Bypass vertex i
    // For each parent p of i:
    //   For each child c of i:
    //     - Add/update edge p→c
    //     - Update probability: p_p→c += p_p→i * p_i→c

    // ⚠️ PROBLEM: When vertex j (j > i) is later processed,
    // its parent list STILL contains vertex i (which was just bypassed!)
    // Should contain i's parents instead.
}
```

### Why Single-Pass Fails

**Example**: Chain A(r=1) → B(r=0) → C(r=0) → D(r=1)

**Iteration 1** (i=B, bypass B):
- Redirect A→B→C to A→C ✓
- **But**: C's parent list still shows B (not updated to A!)

**Iteration 2** (i=C, bypass C):
- Look up C's parents → find B
- Try to redirect B→C→D
- **But**: B's edges were already modified in iteration 1!
- Data structures are inconsistent → broken graph

**Result**: PDF integral = 0

---

## What Works Correctly

✅ **Sampling**: `graph.sample()` produces correct distributions for all reward sparsities
✅ **Moments**: `expected_waiting_time()` scales correctly (∝ 1/θ) for all features
✅ **Reward scaling**: Doubling rewards doubles sample means (within 5% sampling error)
✅ **Dense rewards**: PDFs normalized correctly when few consecutive zeros (≤2)
✅ **1-2 consecutive bypasses**: PDF integral ≈ 0.99-1.00

---

## Proposed Fixes

### Option 1: Multi-Pass Iterative Elimination ⭐ RECOMMENDED

Process zero-reward vertices iteratively until none remain:

```c
bool changed = true;
while (changed) {
    changed = false;
    for (size_t i = 0; i < vertices_length; ++i) {
        if (rewards[i] != 0) continue;

        // Bypass vertex i
        // ... existing bypass logic ...

        // Mark graph as changed
        changed = true;
    }
}
```

**Pros**:
- Minimal change to existing code
- Guarantees correctness (processes one vertex at a time)

**Cons**:
- O(n²) worst case (but n is typically small)

### Option 2: Reverse Topological Order

Process vertices from absorbing states backward:

```c
size_t *topo_order = reverse_topological_sort(graph);

for (size_t idx = 0; idx < vertices_length; ++idx) {
    size_t i = topo_order[idx];
    if (rewards[i] != 0) continue;

    // Bypass vertex i (all children already processed)
    // ...
}
```

**Pros**:
- O(n) single pass
- More efficient

**Cons**:
- Requires topological sort implementation
- More complex change

### Option 3: Update Parent Lists After Each Bypass

After bypassing vertex i, immediately update children's parent lists:

```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) continue;

    // Bypass vertex i
    for (each parent p of i) {
        for (each child c of i) {
            add_edge(p, c);
        }
    }

    // NEW: Update parent information
    for (each child c of i) {
        remove i from vertex_parents[c]
        add (parents of i) to vertex_parents[c]
    }
}
```

**Pros**:
- Directly fixes the root cause

**Cons**:
- Requires maintaining `vertex_parents` data structure
- More bookkeeping complexity

---

## Recommended Fix: Option 1

**Implement multi-pass iterative elimination**:

1. Simplest to implement (minimal code change)
2. Guarantees correctness
3. Performance impact negligible for typical graph sizes (n < 100)
4. Easy to test and verify

---

## Testing Strategy

### 1. Unit Test: Multiple Consecutive Bypasses

```python
def test_multiple_consecutive_bypasses():
    """Test reward_transform with 3+ consecutive zero rewards"""
    graph = create_linear_chain(5)  # 0→1→2→3→4

    # Bypass 3 consecutive vertices
    rewards = [1, 0, 0, 0, 1]
    graph_transformed = graph.reward_transform(rewards)

    # Check PDF is normalized
    times = np.linspace(0, 10, 100)
    pdf = [graph_transformed.pdf(t) for t in times]
    integral = np.trapezoid(pdf, times)

    assert abs(integral - 1.0) < 0.05, f"PDF not normalized: {integral}"
```

### 2. Integration Test: Coalescent Model

```python
def test_coalescent_sparse_rewards():
    """Test SVGD with sparse coalescent rewards"""
    theta_true = 10.0
    nr_samples = 6

    graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    rewards = graph.states()[:, 4]  # Feature 4 (92% sparse)

    # Generate observations
    observations = graph.sample(1000, rewards=rewards.tolist())

    # Run SVGD
    result = graph.svgd(
        observed_data=observations,
        theta_dim=1,
        n_particles=100,
        n_iterations=500
    )

    theta_estimate = result['theta_mean'][0]
    error = abs(theta_estimate - theta_true) / theta_true

    assert error < 0.10, f"SVGD estimate {theta_estimate:.2f} has {error*100:.1f}% error"
```

### 3. Regression Test: Existing Functionality

Ensure fix doesn't break:
- Dense rewards (all non-zero)
- 1-2 consecutive bypasses
- Parameterized graphs
- Multivariate models

---

## Files to Modify

### Primary Change

**src/c/phasic.c** lines 2151-2230:
- Implement multi-pass iterative elimination
- Add `changed` flag and outer while loop

### Tests to Add

**tests/test_vertex_bypass.py**:
- Test case for 3+ consecutive bypasses
- Test case for coalescent sparse rewards
- Regression tests for dense rewards

---

## Impact Assessment

### Users Affected
- Anyone using sparse reward vectors (>75% zeros)
- Coalescent models with site frequency spectra
- Multivariate phase-type models with many features

### Severity
- **CRITICAL**: Causes systematic 20% bias in parameter inference
- Invalid PDFs (integral ≠ 1.0)
- Silent failure (no error, just wrong results)

### Workaround Until Fixed
1. Avoid reward vectors with long runs of zeros (≥3 consecutive)
2. Add small epsilon to zero rewards (e.g., 0.001 instead of 0)
3. Aggregate features to reduce sparsity

---

## Next Steps

1. ✅ Bug confirmed and diagnosed
2. ⏭️ **Implement Option 1 fix** in src/c/phasic.c
3. ⏭️ **Add unit tests** for multiple consecutive bypasses
4. ⏭️ **Run regression tests** to verify no breakage
5. ⏭️ **Test coalescent model** with sparse rewards
6. ⏭️ **Document fix** in changelog

---

## Reproducibility

All test files available in `/tmp/`:
- `test_bypass_order_bug.py` - Demonstrates bug with simple chain
- `test_vertex_bypass_bug.py` - Detailed bypass testing
- `test_reward_scaling_all_features.py` - Confirms sampling works correctly

Bug reports:
- `SPARSE_REWARDS_BUG_REPORT.md` - Initial discovery
- `VERTEX_BYPASS_BUG_REPORT.md` - Detailed analysis
- `VERTEX_BYPASS_COMPLETE_DIAGNOSIS.md` - This document

---

**Author**: Claude (AI Assistant)
**Date**: 2025-10-27
**Priority**: 🔴 CRITICAL
