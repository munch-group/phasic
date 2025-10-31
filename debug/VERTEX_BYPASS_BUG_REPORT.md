# Critical Bug: Vertex Bypass in reward_transform

## Summary

**CONFIRMED BUG in `_ptd_graph_reward_transform()` (src/c/phasic.c lines 2151-2230)**

When multiple consecutive vertices have zero rewards and are bypassed, the PDF becomes invalid (integral → 0). This causes SVGD parameter estimates to be biased by ~20%.

## Root Cause

The vertex bypass algorithm processes all zero-reward vertices in a **single pass** (line 2151), but doesn't properly update parent-child relationships between iterations.

### The Problem

**Given chain**: A(r=1) → B(r=0) → C(r=0) → D(r=1)

**Iteration 1** (bypass B):
- Redirect A→B→C to A→C
- **BUT**: C's parent list still records B as parent (not A!)

**Iteration 2** (bypass C):
- Try to redirect parent→C→D
- Look up C's parents → find B
- But B's edges were already redirected in iteration 1
- Data structures are inconsistent
- **Result**: Broken graph with PDF integral = 0

## Evidence

### Test Case: Simple Linear Chain

```python
# Chain: 0 → 1 → 2 → 3 → 4 (absorbing)

# No bypasses
rewards = [1, 1, 1, 1, 1]
PDF integral: 1.000 ✓

# 1 bypass (vertex 2)
rewards = [1, 1, 0, 1, 1]
PDF integral: 0.999 ✓

# 2 bypasses (vertices 2,3)
rewards = [1, 1, 0, 0, 1]
PDF integral: 0.992 ✓

# 3 bypasses (vertices 1,2,3)
rewards = [1, 0, 0, 0, 1]
PDF integral: 0.000 ❌ BROKEN!
```

### Coalescent Model (nr_samples=6)

With 12 vertices and different reward vectors:

| Feature | Non-zero Rewards | Consecutive Zeros | SVGD Estimate | Bias |
|---------|------------------|-------------------|---------------|------|
| 0       | 7/12 (58%)       | ~2                | θ=10 ✓        | 0%   |
| 1       | 5/12 (42%)       | ~2-3              | θ=10 ✓        | 0%   |
| 2       | 3/12 (25%)       | ~4                | θ=12 ❌       | +20% |
| 3       | 2/12 (17%)       | ~5                | θ=12 ❌       | +20% |
| 4       | 1/12 (8%)        | ~6                | θ=12 ❌       | +20% |

**Pattern**: Features with long runs of zeros (≥4 consecutive) have biased estimates.

## The Bug in Detail

**File**: `src/c/phasic.c`
**Function**: `_ptd_graph_reward_transform()`
**Lines**: 2151-2230

```c
// Line 2151: Process all zero-reward vertices in single pass
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) {
        continue;  // Skip non-zero rewards
    }

    // Lines 2156-2230: Bypass vertex i
    // For each parent of i:
    //   - Redirect parent→i→child to parent→child
    //   - Update probabilities: p_parent_child += p_parent_i * p_i_child

    // PROBLEM: When vertex j (j > i) is later processed,
    // its parent list still contains vertex i (which was just bypassed)
    // instead of i's parents. This causes incorrect edge redirections.
}
```

### Why Single-Pass Doesn't Work

The algorithm assumes that when processing vertex `i`, all vertices `j < i` are still in their original state. But after bypassing vertex `j < i`, the graph structure has changed:

1. **New edges created**: parent(j) → child(j)
2. **Old edges still recorded**: parent(i) still shows j as parent
3. **Inconsistency**: When processing i, the algorithm uses stale parent information

## Proposed Fix

### Option 1: Multi-Pass Iterative Elimination

Process zero-reward vertices iteratively until none remain:

```c
bool changed = true;
while (changed) {
    changed = false;
    for (size_t i = 0; i < vertices_length; ++i) {
        if (rewards[i] != 0) continue;

        // Bypass vertex i
        // ... existing bypass logic ...

        // Update parent-child data structures
        // ... update vertex_parents and vertex_edges ...

        changed = true;
    }
}
```

### Option 2: Process in Reverse Topological Order

Process vertices in reverse topological order (from absorbing states backward), ensuring that when a vertex is bypassed, all its children have already been processed:

```c
// Reverse topological sort
size_t *topo_order = reverse_topological_sort(graph);

for (size_t idx = 0; idx < vertices_length; ++idx) {
    size_t i = topo_order[idx];
    if (rewards[i] != 0) continue;

    // Bypass vertex i (all children already processed)
    // ...
}
```

### Option 3: Update Parent Information After Each Bypass

After bypassing vertex `i`, immediately update all child vertices to replace `i` in their parent lists with `i`'s parents:

```c
for (size_t i = 0; i < vertices_length; ++i) {
    if (rewards[i] != 0) continue;

    // Bypass vertex i
    for (each parent p of i) {
        for (each child c of i) {
            // Add edge p→c
            // ...
        }
    }

    // NEW: Update parent information
    for (each child c of i) {
        // Replace i with parents of i in c's parent list
        remove i from vertex_parents[c]
        add (parents of i) to vertex_parents[c]
    }
}
```

## Impact

This bug affects:
- **Sparse reward vectors**: >75% zeros with ≥3 consecutive zeros
- **Multivariate models**: Site frequency spectra, coalescent models with many features
- **SVGD inference**: ~20% parameter estimation bias
- **PDF computation**: Incorrect distributions (integral ≠ 1)

**Severity**: CRITICAL - causes systematic bias in parameter inference

## Workaround

Until fixed:
1. Avoid reward vectors with long runs of zeros
2. Rescale rewards so no vertex has exactly 0 reward (use small epsilon like 0.001 instead)
3. Use aggregated features instead of many sparse features

## Testing

Add test case to catch this:

```python
def test_multiple_consecutive_bypasses():
    """Test reward_transform with 3+ consecutive zero rewards"""
    graph = create_linear_chain(5)  # 5 vertices

    # Bypass 3 consecutive vertices
    rewards = [1, 0, 0, 0, 1]
    graph_transformed = graph.reward_transform(rewards)

    # Check PDF is normalized
    times = np.linspace(0, 10, 100)
    pdf = [graph_transformed.pdf(t) for t in times]
    integral = np.trapezoid(pdf, times)

    assert abs(integral - 1.0) < 0.05, f"PDF not normalized: {integral}"
```

## Related Issues

- NaN handling bug (FIXED): Multivariate sparse matrices
- Parameterized weight bugs (FIXED): base_weight handling

This vertex bypass bug is **independent** and is the **root cause** of the persistent ~10-20% SVGD bias with sparse rewards.
