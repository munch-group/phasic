# Plan: Fix Multivariate Phase-Type PDF Computation

## Root Cause
PDF is computed on base graph instead of reward-transformed graph. This causes systematic estimation errors (12% instead of expected 1-2%).

**Key finding**: When sampling with rewards, the C code multiplies waiting times by rewards (`waiting_time *= rewards[vertex->index]`), so samples come from the distribution of R·T. However, the PDF is computed on the base graph, giving the distribution of T. These are different distributions!

## Solution Strategy
Modify `GraphBuilder::compute_pmf_and_moments` to compute PDF on reward-transformed graph when rewards are provided.

**Important**: Rewards CANNOT be "zipped" into the trace because:
- Trace operations handle parametric changes (θ values)
- Reward transformation causes structural changes (graph topology)
- Reward transform does SCC finding and topological sort - not simple arithmetic

The fix belongs in `GraphBuilder::compute_pmf_and_moments`, not in the trace system.

## Implementation Plan

### Step 1: Try C API Direct Call (Safest)
Modify `src/cpp/parameterized/graph_builder.cpp:compute_pmf_and_moments` (lines 391-417):

```cpp
// After building base graph (line 398)
Graph g = build(theta_vec.data(), theta_len);

// If rewards provided, use C API for transformation
struct ptd_graph* g_for_pdf_ptr = g.c_graph();  // Base graph pointer

if (!rewards_vec.empty()) {
    // Call C API reward_transform (returns new ptd_graph*)
    g_for_pdf_ptr = ptd_graph_reward_transform(
        g.c_graph(),
        rewards_vec.data()
    );

    if (g_for_pdf_ptr == NULL) {
        throw std::runtime_error("Reward transformation failed");
    }
}

// Compute PDF using (potentially transformed) graph
// Create temporary Graph wrapper if transformed
Graph g_for_pdf = (g_for_pdf_ptr == g.c_graph())
    ? g
    : Graph(g_for_pdf_ptr, false);  // Wrap C pointer, don't own

// Compute PDF
for (size_t i = 0; i < n_times; i++) {
    pmf_vec[i] = discrete
        ? g_for_pdf.dph_pmf(static_cast<int>(times_vec[i]))
        : g_for_pdf.pdf(times_vec[i], granularity);
}

// Clean up if we created transformed graph
if (g_for_pdf_ptr != g.c_graph()) {
    ptd_graph_free(g_for_pdf_ptr);
}

// Moments still use base graph with rewards (current code is correct)
moments = compute_moments_impl(g, nr_moments, rewards_vec);
```

### Step 2: Rebuild and Test with Single Feature
```bash
pip install -e . --no-build-isolation
python test_nan_single_feature_proper.py
```

**Expected**: θ = 10.00 ± 0.20 (2% error)
**Current**: θ = 11.19 ± 1.79 (12% error)

### Step 3: Test with Multiple Features
```bash
python test_one_feature_per_obs.py
```

**Expected**: All features → same θ estimate
**Current**: Feature 0 → 7.51, Feature 1 → 12.97 (inconsistent)

### Step 4: Full SVGD Test
```bash
python test_all_features_10k.py
```

- 10,000 observations
- All 4 features with proper NaN masking
- Error should be 1-2%, not 12%

## Fallback Plans (if Step 1 fails)

### Alternative A: Use Graph Pointer Method
If direct C API call causes issues, try the C++ pointer version:

```cpp
if (!rewards_vec.empty()) {
    Graph* g_transformed_ptr = g.reward_transform_p(rewards_vec);
    // Use pointer, clean up explicitly with delete
}
```

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

This would work like `ptd_vertex_expected_waiting_time` - no separate transformation needed, just modified forward algorithm.

**Advantages**:
- No graph copying/ownership issues
- Matches how moments are computed
- Most efficient implementation

**Implementation**: Modify forward algorithm to multiply probabilities by rewards during uniformization.

## Why Previous Attempts Failed

1. **C++ reward_transform (by value)**: Segfault
   - Likely cause: Copy semantics with C struct wrappers
   - Graph owns `ptd_graph*`, copying may invalidate pointers

2. **Python reward_transform on parameterized graph**: Returns NaN
   - Cause: reward_transform doesn't work on parameterized graphs
   - Must instantiate concrete graph first

3. **Trace + instantiate + reward_transform**: "Multiple edges" error
   - Cause: Graph elimination creates duplicate edges
   - reward_transform can't handle this structure

## Success Criteria

1. ✅ Single feature estimates θ = 10.0 ± 0.2 (2% error) with 10K observations
2. ✅ All features converge to SAME estimate (consistent across reward vectors)
3. ✅ Error DECREASES with more data (not increases)
4. ✅ No segfaults or memory errors
5. ✅ Backward compatible (no rewards → same behavior as before)

## Files to Modify

- `src/cpp/parameterized/graph_builder.cpp` (lines 391-417 in `compute_pmf_and_moments`)

## Test Files

- `test_nan_single_feature_proper.py` - Single feature, 10K obs
- `test_one_feature_per_obs.py` - One feature per obs with NaN
- `test_all_features_10k.py` - All features, 10K obs
- `test_reward_diagnostics.py` - Verify reward transformation correctness

## Additional Notes

- **No trace changes needed**: The trace system works correctly as-is
- **Moments are correct**: They use `expected_waiting_time` which handles rewards properly
- **Only PDF needs fixing**: It's the only computation that ignores rewards
- **Low risk**: Changes isolated to one function, easily reversible

## Related Documentation

- `MULTIVARIATE_ROOT_CAUSE.md` - Detailed root cause analysis
- `MULTIVARIATE_FIXES_SUMMARY.md` - Summary of dtype and NaN fixes already applied
- `CLAUDE.md` - Updated with multivariate API documentation
