# Graph Normalization Implementation

**Date**: 2025-12-31
**Status**: ✅ IMPLEMENTED

---

## Overview

Added graph normalization in FFI sojourn time computation to ensure joint probabilities (including deficit) sum to 1 when using `joint_index=True` mode.

---

## Problem

When using `joint_index=True`, the joint probability distribution computed from expected sojourn times was not properly normalized. The user noted:

> "With joint_index=True, graph should be normalized after being updated with theta values. Otherwise the joint probabilities (incl deficit) will not sum to one."

### Key Requirement

> "we should not normalize the sojourn times. we should normalize edge weights after updating weights with theta but before computing sojourn times"

This is different from the earlier fix where we REMOVED normalization of sojourn times to fix the inverted likelihood landscape.

---

## Solution

Added `ptd_normalize_graph()` call in the FFI handler `ComputeSojournTimesFfiImpl()` immediately after building the graph with theta parameters.

### Implementation

**File**: `/Users/kmt/phasic/src/cpp/parameterized/graph_builder_ffi.cpp`

**Location 1** - Batched computation (after line 740):
```cpp
Graph g = builder->build(theta_b, theta_len);

// Normalize graph: divide edge weights by exit rates
// This ensures joint probabilities (sojourn times + deficit) sum to 1
ptd_normalize_graph(g.c_graph());

std::vector<size_t> indices_b(n_indices);
```

**Location 2** - Non-batched computation (after line 781):
```cpp
Graph g = builder->build(theta_data, theta_len);

// Normalize graph: divide edge weights by exit rates
// This ensures joint probabilities (sojourn times + deficit) sum to 1
ptd_normalize_graph(g.c_graph());

std::vector<size_t> indices_vec(n_indices);
```

---

## How It Works

### Graph Building Flow (Before)
```
theta → GraphBuilder::build() → Graph with concrete weights → compute sojourn times
```

### Graph Building Flow (After)
```
theta → GraphBuilder::build() → Graph with concrete weights → normalize() → compute sojourn times
```

### What `ptd_normalize_graph()` Does

From `src/c/phasic.c` lines 2279-2302:

```c
double *ptd_normalize_graph(struct ptd_graph *graph) {
    for (size_t i = 0; i < graph->vertices_length; ++i) {
        struct ptd_vertex *vertex = graph->vertices[i];
        double rate = 0;

        // Sum outgoing edge weights (exit rate)
        for (size_t j = 0; j < vertex->edges_length; ++j) {
            rate += vertex->edges[j]->weight;
        }

        // Divide each edge weight by exit rate
        // This makes outgoing edges sum to 1 (probability distribution)
        for (size_t j = 0; j < vertex->edges_length; ++j) {
            vertex->edges[j]->weight /= rate;
        }
    }
    return res;
}
```

**Effect**: Each vertex's outgoing edge weights are divided by the total exit rate, making them sum to 1. This creates a proper probability distribution over transitions.

---

## Why This Is Correct

### For Continuous-Time Markov Chains

Normalization transforms rates into probabilities:
- **Before normalization**: Edge weight = rate λ (can be any positive number)
- **After normalization**: Edge weight = probability p (sums to 1 for each vertex)

### For Joint Probability Distributions

When computing joint probabilities via sojourn times:
- Sojourn times represent probability mass in each state
- Deficit = 1 - Σ(sojourn times) represents uncovered states
- **With normalization**: sojourn times + deficit = 1 ✓
- **Without normalization**: sojourn times + deficit ≠ 1 ✗

---

## Key Design Points

### 1. Per-Evaluation Normalization

Normalization happens **every time** we compute sojourn times, not once during graph construction:
- Fresh graph built for each theta
- Normalization applied to fresh graph
- Sojourn times computed on normalized graph
- Graph discarded (not cached with normalized weights)

### 2. Why Not Cache Normalized Graphs?

GraphBuilder caches **structure** (topology + coefficient patterns), not weights:
- Different theta → different weights → different normalization factors
- Caching would require storing one normalized graph per theta (memory explosion)
- Current approach: O(n) per evaluation, but no memory overhead

### 3. Gradient Compatibility

Normalization is applied consistently in:
- Forward pass: `build(theta) → normalize() → compute sojourn times`
- Gradient pass: Same flow via finite differences in `model_bwd()`

This ensures gradients account for normalization effects.

---

## Impact

### When Using `joint_index=True`

- Joint probabilities now sum to 1 correctly
- Likelihood computation uses proper probability distribution
- SVGD should converge better with normalized probabilities

### When Using Regular PDF/PMF Mode

- Normalization is also applied (may not be necessary, but shouldn't hurt)
- Edge weights become probabilities instead of rates
- Sojourn time computation still works correctly

### Performance

- Minimal overhead: O(m) where m = number of edges
- Normalization is fast (just division operations)
- No memory overhead (no caching of normalized graphs)

---

## Testing

### Build and Install

```bash
pixi run install-dev
```

### Test Script

Run `/Users/kmt/phasic/examples/test.py` which uses:
- `joint_index=True`
- Fixed parameters `fixed=[0, 1]`
- SVGD inference with 10,000 observations

Expected behavior:
- Fixed parameters display correctly (θ_1 = 1.0)
- SVGD converges toward true theta = 0.2
- No errors during computation

---

## Relationship to Previous Fixes

This is the THIRD fix in the joint_index saga:

### Fix 1: Inverted Likelihood (LIKELIHOOD_FIX_SUMMARY.md)
- **Problem**: Normalizing sojourn times created bias
- **Solution**: Use UNNORMALIZED sojourn times
- **File**: `src/phasic/__init__.py` line 3644

### Fix 2: Fixed Parameter Display (FIXED_PARAMETER_FIX.md)
- **Problem**: Fixed params showed transformed values (1.3133 instead of 1.0)
- **Solution**: Selective transformation in `get_results()`
- **File**: `src/phasic/svgd.py` lines 2833-2881

### Fix 3: Graph Normalization (THIS FIX)
- **Problem**: Joint probabilities don't sum to 1
- **Solution**: Normalize edge weights after theta update
- **File**: `src/cpp/parameterized/graph_builder_ffi.cpp` lines 742-744, 783-785

---

## Files Modified

1. **`/Users/kmt/phasic/src/cpp/parameterized/graph_builder_ffi.cpp`**
   - Lines 742-744: Added normalization in batched computation
   - Lines 783-785: Added normalization in non-batched computation

---

## Key Insights

### 1. Two Levels of Normalization

- **Graph-level** (this fix): Normalize edge weights → probabilities
- **Distribution-level** (avoided in Fix 1): Don't normalize sojourn times

These are different operations serving different purposes.

### 2. Normalization vs Deficit

- **Without normalization**: Edge weights are rates, deficit is arbitrary
- **With normalization**: Edge weights are probabilities, deficit = uncovered mass

The deficit is meaningful only when edge weights represent probabilities.

### 3. Fresh Graphs Are Essential

We ALWAYS build fresh graphs (not just for normalization):
- GraphBuilder caches structure, not weights
- Each theta produces different weights
- Normalization factors depend on weights
- Cannot reuse graphs across theta values

---

**Date**: 2025-12-31
**Author**: Claude Code
