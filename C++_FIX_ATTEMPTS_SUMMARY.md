# C++ Fix Attempts Summary - Multivariate PDF Bug

## Problem

SVGD estimates θ with ~12% error instead of expected 1-2% with 10K observations.

**Root Cause** (from MULTIVARIATE_ROOT_CAUSE.md):
- Sampling multiplies waiting times by rewards: `outcome += waiting_time * rewards[i]`
- Samples come from distribution of R·T (reward-transformed)
- PDF is computed on BASE graph (without rewards)
- PDF gives distribution of T, not R·T
- Mismatch causes systematic estimation errors

## Attempted Fixes

### Attempt 1: Graph Wrapper Approach (FAILED - Segfault)

**Implementation**:
```cpp
if (!rewards_vec.empty()) {
    Graph g_for_pdf = created_transformed_graph
        ? Graph(g_for_pdf_ptr, g.c_avl_tree())
        : g;
}
```

**Result**: Segmentation fault on module import or first test run

**Why it failed**:
- Graph class has complex reference counting system
- Creating Graph wrapper from `ptd_graph*` pointer conflicts with manual cleanup
- Double-free or invalid pointer dereference

### Attempt 2: C API Direct Approach (FAILED - 84% Error)

**Implementation**:
```cpp
if (!rewards_vec.empty() && !discrete) {
    struct ptd_graph* g_transformed = ptd_graph_reward_transform(
        g.c_graph(),
        rewards_vec.data()
    );

    // Create PDF context, step forward, cache values
    struct ptd_probability_distribution_context* ctx =
        ptd_probability_distribution_context_create(g_transformed, granularity);

    // Step and cache PDF values
    while (ctx->time < max_time) {
        ptd_probability_distribution_step(ctx);
        pdf_cache.push_back(ctx->pdf);
    }

    // Extract PDF for each time
    pmf_vec[i] = pdf_cache[granularity * time];

    // Cleanup
    ptd_probability_distribution_context_destroy(ctx);
    ptd_graph_destroy(g_transformed);
}
```

**Result**:
- Baseline (no fix): θ = 10.0 true, 11.19 ± 1.79 estimated (11.9% error)
- With C API fix: θ = 10.0 true, 18.39 ± 2.84 estimated (83.9% error)
- **ERROR GOT WORSE** by 7x

**Why it failed** (hypothesis):
1. `ptd_graph_reward_transform()` does complex graph transformations:
   - Finds strongly connected components (SCC)
   - Performs topological sort
   - Normalizes edge weights by dividing by vertex rate
   - Divides rewards by rate: `rewards[i] /= rate`

2. This transformation may NOT compute the PDF of R·T as we expect

3. The reward_transform function might be designed for a different purpose (computing moments with rewards, not PDF with rewards)

4. Looking at reward_transform implementation (lines 2063-2071 in phasic.c):
   ```c
   for (size_t j = 0; j < vertex->edges_length; ++j) {
       vertex->edges[j]->weight /= rate;
   }
   if (rewards[i] != 0) {
       rewards[i] /= rate;
   }
   ```
   This normalization is not what we need for PDF(R·T)

## Why C++ Approaches Are Failing

**Key Insight**: `ptd_graph_reward_transform()` is NOT designed to create a graph whose PDF represents the distribution of R·T.

Looking at how moments are computed (which DO work correctly with rewards), they use `expected_waiting_time(rewards)` - a different code path that doesn't require graph transformation.

The reward_transform function seems designed for graph elimination and moment computation, not for forward algorithm PDF computation.

## Recommended Next Step: Alternative B

From MULTIVARIATE_FIX_PLAN.md:

### Alternative B: Implement PDF-with-Rewards in C (Most Robust)

Add new C function that computes PDF(R·T) directly:

```c
double ptd_graph_pdf_with_rewards(
    struct ptd_graph *graph,
    double time,
    double *rewards,
    size_t granularity
)
```

**Advantages**:
- No graph transformation needed
- Works like `ptd_vertex_expected_waiting_time` (which correctly handles rewards)
- Modifies forward algorithm (uniformization) to multiply probabilities by rewards during computation
- Most efficient implementation
- No memory/ownership issues

**Implementation Strategy**:
Modify the forward algorithm in the probability distribution context to:
1. At each step, when computing probability mass at each vertex
2. Multiply the probability by the corresponding reward
3. This gives the distribution of R·T directly

**Effort**:
- Moderate (1-2 days)
- Requires understanding uniformization algorithm
- Need to modify C PDF computation code

## Test Results Comparison

| Approach | Error | Estimated θ | True θ | Status |
|----------|-------|-------------|--------|--------|
| Baseline (no fix) | 11.9% | 11.19 ± 1.79 | 10.0 | ❌ Wrong |
| Attempt 1 (Graph wrapper) | N/A | Segfault | 10.0 | ❌ Crash |
| Attempt 2 (C API direct) | 83.9% | 18.39 ± 2.84 | 10.0 | ❌ Worse |
| **Expected with correct fix** | **<2%** | **10.0 ± 0.2** | **10.0** | ✅ Target |

## Files Modified (Reverted)

- `src/cpp/parameterized/graph_builder.cpp` - reverted to baseline

## Next Actions

1. Implement Alternative B: `ptd_graph_pdf_with_rewards()` in C
2. Modify forward algorithm to handle rewards during uniformization
3. Update `GraphBuilder::compute_pmf_and_moments` to call new function when rewards provided
4. Test with single feature
5. Test with all features

## Related Documentation

- `MULTIVARIATE_ROOT_CAUSE.md` - Detailed root cause analysis
- `MULTIVARIATE_FIX_PLAN.md` - Original fix plan with alternatives
- `UNIFIED_ARCHITECTURE_PLAN.md` - Long-term architectural plan
