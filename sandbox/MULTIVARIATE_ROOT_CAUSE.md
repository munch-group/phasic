# Multivariate Phase-Type Distribution Root Cause Analysis

## Problem

SVGD fails to correctly estimate parameters for multivariate phase-type distributions with ~12% error instead of expected 1-2% with 10K observations.

Different features give systematically different estimates:
- Feature 0 (reward=[0,5,3,1,2,0,1,0]): estimates θ=7.51 (25% under)
- Feature 1 (reward=[0,0,1,2,0,1,0,0]): estimates θ=12.97 (30% over)
- Combined: estimates θ=11.19 (12% over)

## Root Cause

**The PDF is computed on the base graph instead of the reward-transformed graph.**

### How Sampling Works

From `src/c/phasic.c:4035-4080`, `ptd_random_sample(graph, rewards)`:
1. Samples a path through the Markov chain
2. At each vertex i, samples waiting time T_i ~ Exp(rate_i)
3. **Multiplies** T_i by rewards[i]: `waiting_time *= rewards[vertex->index]`
4. Returns outcome = Σ(rewards[i] * T_i)

This IS reward transformation - samples come from the distribution of R·T.

### How PDF is Currently Computed

From `src/cpp/parameterized/graph_builder.cpp:354-432`, `compute_pmf_and_moments(...)`:
- Line 398: `Graph g = build(theta_vec.data(), theta_len)` - builds base graph
- Line 408-410: `pmf_vec[i] = g.pdf(times_vec[i], granularity)` - computes PDF on **BASE graph**
- Line 414: `moments = compute_moments_impl(g, nr_moments, rewards_vec)` - computes moments **WITH rewards**

**Inconsistency**: Samples are from R·T distribution, but PDF is computed for T distribution!

### Why This Causes Errors

When SVGD evaluates log P(data | θ):
1. Observed data comes from sampling with rewards → distribution of R·T
2. PDF is computed without rewards → distribution of T
3. These are DIFFERENT distributions!
4. The likelihood is systematically wrong
5. Gradients point in wrong direction
6. θ converges to wrong value

Different reward vectors → different R·T distributions → different wrong estimates.

## Attempted Fixes

### Fix 1: C++ reward_transform (FAILED - Segfault)

Attempted to modify `compute_pmf_and_moments` to:
```cpp
if (!rewards_vec.empty()) {
    Graph g_for_pdf = g.reward_transform(rewards_vec);
}
```

**Result**: Segmentation fault when calling reward_transform.

**Why it failed**:
- Calling `g.reward_transform(rewards_vec)` returns new Graph by value
- May have memory/ownership issues in this context
- Caused immediate crash on module import or first use

### Fix 2: Python reward_transform on parameterized graph (FAILED - NaN)

Tested `graph_param.reward_transform(rewards)` on parameterized graph.

**Result**: Returns NaN for PDF.

**Why it failed**: reward_transform doesn't work on parameterized graphs.

### Fix 3: Trace + instantiate + reward_transform (FAILED - Duplicate edges)

Attempted to:
1. Record trace from parameterized graph
2. Instantiate concrete graph with theta
3. Apply reward_transform
4. Compute PDF

**Result**: Error "Multiple edges to the same vertex"

**Why it failed**: Graph elimination creates duplicate edges that reward_transform can't handle.

## Required Solution

The PDF computation MUST use the reward-transformed graph to match what sampling does.

### Option A: Fix C++ reward_transform

Investigate and fix the segfault when calling `g.reward_transform(rewards_vec)` in `compute_pmf_and_moments`.

Possible issues to check:
- Memory management (use pointer version `reward_transform_p`?)
- GIL state during transformation
- Graph ownership/lifetime

### Option B: Pre-transform in Python

Modify the multivariate wrapper to:
1. For each feature j with reward vector r_j
2. Create a transformed model that computes PDF(R·T) directly
3. Use trace-based approach but handle duplicate edges

### Option C: Implement PDF-with-rewards in C

Add new C function: `ptd_graph_pdf_with_rewards(graph, time, rewards, granularity)`

That computes P(R·T = t) directly without needing separate reward_transform step.

## Test Results

All tests confirm the diagnosis:

1. **Feature-specific bias**: Different rewards → different wrong estimates (test_nan_single_feature_proper.py, test_nan_feature1.py)
2. **Moments are correct**: `reward_transform` NaN but sampling works → moments use different (correct) code path
3. **Error increases with data**: More observations → stronger signal → converges more tightly to wrong answer

## Next Steps

1. **Immediate**: Investigate why C++ `reward_transform` causes segfault
   - Check Graph copy constructor
   - Check memory management in C reward_transform implementation
   - Try pointer version `reward_transform_p`

2. **Alternative**: If C++ fix too risky, implement Option C (new C function for PDF with rewards)

3. **Test**: Once fixed, verify:
   - Single feature estimates θ=10 ± 0.2 (2% error) with 10K obs
   - All features converge to same estimate
   - Error decreases (not increases) with more data
