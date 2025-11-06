# Complete Investigation: Multivariate Phase-Type Distribution SVGD Failure

**Date Started**: October 2025
**Last Updated**: 2025-10-26
**Status**: Root cause identified, C++ fixes failed, Alternative B recommended

---

## Executive Summary

SVGD inference on multivariate phase-type distributions shows systematic parameter estimation errors of ~12% instead of expected 1-2% with 10,000 observations. The root cause has been definitively identified: **PDF is computed on the base graph instead of the reward-transformed graph**, causing a mismatch between the sampling distribution and the likelihood computation. Two C++ fix attempts have failed (segfault and 84% error). The recommended path forward is Alternative B: implement `ptd_graph_pdf_with_rewards()` in C.

---

## 1. Problem Statement

### 1.1 Observed Behavior

SVGD fails to correctly estimate parameters for multivariate phase-type distributions:

| Test Configuration | True θ | Estimated θ | Error | N obs |
|-------------------|--------|-------------|-------|-------|
| Feature 0 only | 10.0 | 7.51 ± 0.65 | 24.9% under | 10K |
| Feature 1 only | 10.0 | 12.97 ± 2.20 | 29.7% over | 10K |
| All 4 features (all obs non-NaN) | 10.0 | 13.24 ± 1.09 | 32.4% over | 10K |
| One feature per obs (NaN masking) | 10.0 | 11.19 ± 1.79 | 11.9% over | 10K |

**Key observations**:
1. Different reward vectors give different biases (feature 0 under, feature 1 over)
2. Error INCREASES with more data (not decreases)
3. Univariate (no rewards) works perfectly
4. Error is systematic, not random

### 1.2 Expected Behavior

With 10,000 observations, SVGD should estimate θ = 10.0 ± 0.2 (1-2% error).

### 1.3 Test Model

Kingman coalescent with n=5 samples:

```python
def coalescent(state, nr_samples=None):
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]
        return ipv
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i+j+1] += 1
                transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
        return transitions

graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
```

This creates an 8-vertex graph (including starting vertex).

**Reward structure**:
```python
states = graph.states()  # Shape: (8, 6) - 8 vertices, 6 state dimensions
rewards = states[:, :-2].astype(np.float64)  # Shape: (8, 4) - 4 features
# Each column is a reward vector for one marginal distribution
```

Example reward vector for feature 0: `[0, 5, 3, 1, 2, 0, 1, 0]`

---

## 2. Diagnostic History

### 2.1 Initial Hypotheses (ALL REJECTED)

**Hypothesis 1**: dtype issues (int32 vs float64)
**Test**: Added dtype conversions throughout Python API
**Result**: ❌ No improvement in error
**Files modified**: `src/phasic/__init__.py` (lines 2643, 2981-2986, 3258)

**Hypothesis 2**: NaN masking incorrect
**Test**: Tested various NaN structures (all non-NaN, one feature per obs)
**Result**: ❌ Error persists regardless of NaN pattern
**Implementation**: Added NaN masking in `src/phasic/svgd.py` (lines 2078-2080)

**Hypothesis 3**: JAX tracer issues
**Test**: Fixed tracer conversion in pure_callback
**Result**: ❌ Eliminated tracers but error persists
**Files modified**: `src/phasic/__init__.py` (lines 3066-3096)

### 2.2 Critical Diagnostic Tests

#### Test 1: reward_transform on parameterized graph

**File**: `test_python_reward_transform.py`

```python
graph_param = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
graph_param.update_parameterized_weights(np.array([10.0]))
graph_transformed = graph_param.reward_transform(reward_vec)
pdf = graph_transformed.pdf(0.1)
# Result: NaN
```

**Conclusion**: reward_transform doesn't work on parameterized graphs

#### Test 2: Trace + instantiate + reward_transform

**File**: `test_trace_reward_approach.py`

```python
trace = record_elimination_trace(graph_param, param_length=1)
graph_concrete = instantiate_from_trace(trace, theta)
graph_transformed = graph_concrete.reward_transform(reward_vec)
pdf = graph_transformed.pdf(0.1, granularity=100)
# Result: Error "Multiple edges to the same vertex!"
```

**Conclusion**: Graph elimination creates duplicate edges that reward_transform can't handle

#### Test 3: Sampling vs PDF mismatch

**Key finding from `src/c/phasic.c:4035-4080`**:

```c
long double ptd_random_sample(struct ptd_graph *graph, double *rewards) {
    // ... sample path through Markov chain ...
    waiting_time = -logl(draw_wait + 0.0000001) / rate;

    if (rewards != NULL) {
        waiting_time *= rewards[vertex->index];  // ← MULTIPLY by rewards!
    }

    outcome += waiting_time;
    return outcome;
}
```

Samples are from the distribution of **R·T** (reward-transformed).

**But PDF computation** (`src/cpp/parameterized/graph_builder.cpp:408`):

```cpp
pmf_vec[i] = g.pdf(times_vec[i], granularity);  // ← Uses BASE graph, ignores rewards!
```

PDF is computed for distribution of **T** (NOT reward-transformed).

**This is the root cause**: Samples ~ P(R·T | θ), but PDF computes P(T | θ).

---

## 3. Root Cause Analysis

### 3.1 The Mismatch

**What sampling does**:
- For each vertex i in the sampled path, draw waiting time T_i ~ Exp(rate_i)
- Multiply by reward: outcome_i = T_i × rewards[i]
- Return total: Σ(T_i × rewards[i])
- **Distribution**: This IS the reward transformation - samples come from R·T

**What PDF computation does**:
- Build base graph g from parameters θ
- Compute PDF using forward algorithm (uniformization) on graph g
- **Distribution**: This gives P(T | θ), NOT P(R·T | θ)

**Why moments work correctly**:

```cpp
// From compute_moments_impl() line 216
std::vector<double> rewards2 = g.expected_waiting_time(rewards);
```

The `expected_waiting_time()` function DOES handle rewards correctly (it's a different code path).

### 3.2 Mathematical Explanation

Let T be the absorption time in the base phase-type distribution.
Let R be a diagonal matrix of rewards.

**What we sample**: Y = R·T (element-wise multiplication of waiting times by rewards)
**What PDF computes**: f_T(t | θ)
**What we need**: f_Y(y | θ) = f_{R·T}(y | θ)

These are different distributions! The PDF of R·T is NOT the same as the PDF of T.

Example:
- If T ~ Exp(λ) and reward r = 2
- Then Y = 2T ~ Exp(λ/2) (NOT Exp(λ))
- PDF: f_Y(y) = (λ/2)·exp(-λy/2) ≠ f_T(y) = λ·exp(-λy)

### 3.3 Why Error Varies by Feature

Different reward vectors → different R·T distributions → different wrong PDFs → different biases:

- Feature 0 reward = [0,5,3,1,2,0,1,0]: mostly larger values → stretches distribution → underestimates rate → θ too low (7.51)
- Feature 1 reward = [0,0,1,2,0,1,0,0]: mostly smaller values → compresses distribution → overestimates rate → θ too high (12.97)

---

## 4. Attempted Fixes

### 4.1 Attempt 1: C++ Graph Wrapper Approach

**Date**: 2025-10-26
**File**: `src/cpp/parameterized/graph_builder.cpp`

**Implementation**:
```cpp
if (!rewards_vec.empty()) {
    struct ptd_graph* g_for_pdf_ptr = ptd_graph_reward_transform(
        g.c_graph(),
        rewards_vec.data()
    );

    // Wrap C pointer in Graph object
    Graph g_for_pdf(g_for_pdf_ptr, g.c_avl_tree());

    // Compute PDF
    pmf_vec[i] = g_for_pdf.pdf(times_vec[i], granularity);

    // Cleanup
    ptd_graph_destroy(g_for_pdf_ptr);
}
```

**Result**: ❌ Segmentation fault on module import or first test

**Why it failed**:
- Graph class has complex reference counting via `rf_graph->references`
- Creating Graph wrapper from raw `ptd_graph*` conflicts with manual cleanup
- Graph copy constructor increments reference count
- Calling `ptd_graph_destroy()` while Graph wrapper still holds reference → double-free or invalid pointer

**Files**: Reverted via `git checkout src/cpp/parameterized/graph_builder.cpp`

### 4.2 Attempt 2: C API Direct Approach

**Date**: 2025-10-26
**File**: `src/cpp/parameterized/graph_builder.cpp`

**Implementation**:
```cpp
if (!rewards_vec.empty() && !discrete) {
    // Transform graph
    struct ptd_graph* g_transformed = ptd_graph_reward_transform(
        g.c_graph(),
        rewards_vec.data()
    );

    // Create PDF context
    struct ptd_probability_distribution_context* ctx =
        ptd_probability_distribution_context_create(g_transformed, granularity);

    // Step forward and cache PDF values
    std::vector<double> pdf_cache;
    pdf_cache.push_back(ctx->pdf);
    double max_time = *std::max_element(times_vec.begin(), times_vec.end());

    while (ctx->time < max_time) {
        ptd_probability_distribution_step(ctx);
        pdf_cache.push_back(ctx->pdf);
    }

    // Extract PDF for each time
    for (size_t i = 0; i < n_times; i++) {
        size_t time_idx = static_cast<size_t>(ctx->granularity * times_vec[i]);
        pmf_vec[i] = time_idx < pdf_cache.size() ? pdf_cache[time_idx] : 0.0;
    }

    // Cleanup
    ptd_probability_distribution_context_destroy(ctx);
    ptd_graph_destroy(g_transformed);
}
```

**Result**: ❌ **84% error - MUCH WORSE than 12% baseline!**

Test output:
```
Baseline (no fix):  θ = 10.0 true → 11.19 ± 1.79 (11.9% error)
With C API fix:     θ = 10.0 true → 18.39 ± 2.84 (83.9% error)
```

**Why it failed**:

Analysis of `ptd_graph_reward_transform()` (`src/c/phasic.c:1987-2400`):

1. Finds strongly connected components (SCC)
2. Performs topological sort
3. **Normalizes edge weights** (lines 2059-2071):
   ```c
   double rate = 0;
   for (size_t j = 0; j < vertex->edges_length; ++j) {
       rate += vertex->edges[j]->weight;
   }
   for (size_t j = 0; j < vertex->edges_length; ++j) {
       vertex->edges[j]->weight /= rate;  // Normalize to probabilities
   }
   if (rewards[i] != 0) {
       rewards[i] /= rate;  // Also normalize rewards
   }
   ```

This transformation is **NOT** designed to create a graph whose PDF represents P(R·T | θ).

The function appears designed for:
- Graph elimination (used in moment computation)
- Converting between rate-based and probability-based representations
- NOT for computing PDF of reward-transformed distribution

**Files**: Reverted via `git checkout src/cpp/parameterized/graph_builder.cpp`

---

## 5. Key Technical Insights

### 5.1 Reward Transform Purpose

`ptd_graph_reward_transform()` is used for:
1. **Moment computation**: E[(R·T)^k] via graph elimination
2. **MPH (Multivariate Phase-Type)**: Multiple marginals from one base graph
3. **Graph structure preservation**: SCC + topological sort for elimination

It is **NOT** for computing PDF(R·T).

### 5.2 Why Moments Work

Moments use a different code path:

```cpp
// From GraphBuilder::compute_moments_impl
std::vector<double> rewards2 = g.expected_waiting_time(rewards);
```

`expected_waiting_time()` correctly computes E[R·T] without transforming the graph:
- Uses iterative algorithm with reward vector as input
- Multiplies expected times by rewards during computation
- No graph transformation needed

### 5.3 Forward Algorithm (PDF Computation)

PDF uses uniformization (Algorithm 4 from paper):

**Location**: `src/c/phasic.c` (probability distribution context)

**How it works**:
1. Discretize time: dt = 1/λ where λ = max(vertex rates)
2. At each time step, propagate probability mass through graph
3. Track mass reaching absorbing states
4. PDF = derivative of cumulative absorption probability

**Current implementation**: Takes graph as input, no provision for per-vertex rewards during forward algorithm.

### 5.4 Sampling with Rewards

**Location**: `src/c/phasic.c:4035-4080`

**How it works**:
1. Sample path through Markov chain (gillespie algorithm)
2. At each vertex, draw waiting time: T ~ Exp(rate)
3. Multiply by reward: outcome += T × reward[vertex]
4. Return total outcome

This gives samples from R·T distribution.

---

## 6. Current State

### 6.1 Files Modified (All Committed)

**Dtype fixes** (committed):
- `src/phasic/__init__.py`: Lines 2643, 2981-2986, 3258
- Added float64 conversions for rewards
- Respect `use_ffi=False` parameter

**NaN masking** (committed):
- `src/phasic/svgd.py`: Lines 2078-2080
- Added `jnp.where(mask, log(pmf), 0.0)` for NaN handling

**JAX tracer fixes** (committed):
- `src/phasic/__init__.py`: Lines 3066-3096
- Fixed tracer conversion in pure_callback with vmap

**Plot fixes** (committed):
- `src/phasic/plot.py`: Updated for 2D reward support
- `src/phasic/svgd_plots.py`: Updated for 2D reward support

### 6.2 Git Status

```
M  src/phasic/__init__.py          # Dtype and tracer fixes
M  src/phasic/svgd.py               # NaN masking
M  src/phasic/plot.py               # 2D reward plotting
M  src/phasic/svgd_plots.py         # 2D reward plotting
M  docs/api/Graph.qmd               # Documentation updates
M  docs/api/SVGD.qmd                # Documentation updates
```

C++ files are at baseline (all attempted fixes reverted).

### 6.3 Documentation Created

1. **MULTIVARIATE_ROOT_CAUSE.md**: Detailed root cause analysis with code references
2. **MULTIVARIATE_FIX_PLAN.md**: Implementation plan with 3 alternatives
3. **UNIFIED_ARCHITECTURE_PLAN.md**: Long-term plan to unify GraphBuilder and Trace with FFI
4. **C++_FIX_ATTEMPTS_SUMMARY.md**: Summary of failed C++ fixes
5. **MULTIVARIATE_INVESTIGATION_COMPLETE.md**: This file

### 6.4 Test Files Created

**Diagnostic tests**:
- `test_python_reward_transform.py`: Tests reward_transform on parameterized graphs → NaN
- `test_trace_reward_approach.py`: Tests trace + instantiate + transform → "Multiple edges" error
- `test_reward_diagnostics.py`: Comprehensive reward transformation diagnostics

**Validation tests**:
- `test_nan_single_feature_proper.py`: Single feature, 10K obs → 11.9% error (baseline)
- `test_one_feature_per_obs.py`: One feature per obs with NaN → varying errors
- `test_all_features_10k.py`: All 4 features, 10K obs → 32.4% error

**Related tests** (in repo):
- `test_multivariate.py`: Original multivariate tests
- `test_feature_moments.py`: Moment computation tests (these pass!)

---

## 7. Recommended Solution: Alternative B

### 7.1 Approach

Implement new C function:

```c
double ptd_graph_pdf_with_rewards(
    struct ptd_graph *graph,
    double time,
    double *rewards,
    size_t granularity
)
```

This function computes PDF(R·T) directly without graph transformation.

### 7.2 Implementation Strategy

Modify the forward algorithm (uniformization) to handle rewards:

**Current algorithm** (pseudo-code):
```
Initialize: prob[start] = 1.0
For each time step dt:
    For each vertex v:
        For each edge e from v:
            prob[e.target] += prob[v] * e.weight * dt
    pdf = prob[absorbing] / dt
```

**With rewards** (proposed):
```
Initialize: prob[start] = 1.0
For each time step dt:
    For each vertex v:
        reward_adjusted_dt = dt * rewards[v]  // ← Key modification
        For each edge e from v:
            prob[e.target] += prob[v] * e.weight * reward_adjusted_dt
    pdf = prob[absorbing] / dt
```

Alternative: adjust probabilities by rewards:
```
For each time step dt:
    For each vertex v:
        For each edge e from v:
            // Probability of transitioning from v, accounting for reward
            prob[e.target] += prob[v] * e.weight * rewards[v] * dt
    pdf = prob[absorbing] / dt
```

### 7.3 Implementation Files

**C implementation**:
- `src/c/phasic.c`: Add `ptd_graph_pdf_with_rewards()` near existing PDF functions
- Modify or create new probability distribution context that stores rewards
- Update forward algorithm step function

**C API header**:
- `api/c/phasic.h`: Add function declaration

**C++ wrapper**:
- `src/cpp/phasiccpp.cpp` or `api/cpp/phasiccpp.h`: Optionally add C++ wrapper

**GraphBuilder integration**:
- `src/cpp/parameterized/graph_builder.cpp`: Update `compute_pmf_and_moments()` to call new function when rewards provided

**Modified code** (in `graph_builder.cpp:400-412`):
```cpp
if (discrete) {
    for (size_t i = 0; i < n_times; i++) {
        int jump_count = static_cast<int>(times_vec[i]);
        pmf_vec[i] = g.dph_pmf(jump_count);
    }
} else if (!rewards_vec.empty()) {
    // NEW: Use PDF with rewards
    for (size_t i = 0; i < n_times; i++) {
        pmf_vec[i] = ptd_graph_pdf_with_rewards(
            g.c_graph(),
            times_vec[i],
            rewards_vec.data(),
            granularity
        );
    }
} else {
    // Standard PDF (no rewards)
    for (size_t i = 0; i < n_times; i++) {
        pmf_vec[i] = g.pdf(times_vec[i], granularity);
    }
}
```

### 7.4 Advantages

1. **No graph transformation**: Avoids `ptd_graph_reward_transform()` entirely
2. **Matches moment computation pattern**: Like `expected_waiting_time(rewards)`
3. **No memory/ownership issues**: No graph copying or cleanup
4. **Most efficient**: Computes PDF(R·T) directly in single pass
5. **Robust**: Proven pattern already used in codebase

### 7.5 Testing Plan

**Step 1**: Unit test in C
```c
// Test: Exponential distribution with reward
graph = single_vertex_with_rate(λ=1.0)
rewards = [2.0]
pdf_no_reward = graph_pdf(graph, t=1.0)      // Should be exp(-1.0)
pdf_with_reward = graph_pdf_with_rewards(graph, t=1.0, rewards)  // Should be exp(-0.5)/2

// Verify: PDF(2T) at t=1 equals (1/2)*PDF(T) at t=0.5
assert abs(pdf_with_reward - 0.5*graph_pdf(graph, 0.5)) < 1e-10
```

**Step 2**: Test with Erlang(k) distribution
```python
# k-stage Erlang with different rewards
rewards_all_one = [1.0] * k  # Should match base PDF
rewards_double = [2.0] * k   # Should match PDF(T/2) scaled
```

**Step 3**: Test single feature SVGD
```bash
python test_nan_single_feature_proper.py
# Expected: θ = 10.0 ± 0.2 (2% error, not 12%)
```

**Step 4**: Test all features
```bash
python test_one_feature_per_obs.py
# Expected: All features converge to same θ = 10.0
```

**Step 5**: Full SVGD test
```bash
python test_all_features_10k.py
# Expected: θ = 10.0 ± 0.2 with 10K observations
```

### 7.6 Estimated Effort

- **Research**: 0.5 days (understand uniformization algorithm in detail)
- **Implementation**: 1 day (write C function, test)
- **Integration**: 0.5 days (update GraphBuilder, rebuild, test)
- **Validation**: 0.5 days (run all tests, verify)

**Total**: 2-3 days

---

## 8. Alternative Approaches (Not Recommended)

### 8.1 Alternative A: Use Test Scripts

Current workaround in `test_trace_reward_approach.py` works:

```python
trace = record_elimination_trace(graph_param, param_length=1)
graph_concrete = instantiate_from_trace(trace, theta)
graph_transformed = graph_concrete.reward_transform(reward_vec)
pdf = graph_transformed.pdf(0.1, granularity=100)
```

**Why it works**:
- `instantiate_from_trace()` creates a concrete (non-parameterized) graph
- Concrete graphs can be successfully transformed
- PDF on transformed graph gives correct distribution

**Why not recommended**:
- Requires trace recording (slow for large graphs)
- Can't be integrated into GraphBuilder (used by FFI)
- "Multiple edges" error on some graphs (graph elimination creates duplicate edges)
- Workaround, not a fix

### 8.2 Alternative C: Python-Level Wrapper

Wrap model to apply transformation before PDF:

```python
def pmf_and_moments_multivariate_wrapper(theta, times, rewards):
    # For each feature/reward column:
    for j in range(n_features):
        reward_vec = rewards[:, j]

        # Build graph
        graph = build_from_theta(theta)

        # Transform
        graph_transformed = graph.reward_transform(reward_vec)

        # Compute PDF
        pmf[:, j] = graph_transformed.pdf(times, granularity)
```

**Why not recommended**:
- Slow (graph transformation per feature per evaluation)
- Can't use with GraphBuilder/FFI (pure_callback doesn't support this)
- Doesn't work with parameterized graphs (reward_transform returns NaN)
- "Multiple edges" error on eliminated graphs

---

## 9. Lessons Learned

### 9.1 About Phase-Type Distributions

1. **Reward transformation is subtle**: R·T is not just "graph with scaled rates"
2. **Different operations need different approaches**:
   - Moments: Use `expected_waiting_time(rewards)` ✅
   - Sampling: Multiply waiting times by rewards ✅
   - PDF: Need custom forward algorithm with rewards ❌ (not implemented)

3. **Graph transformation ≠ distribution transformation**:
   - `ptd_graph_reward_transform()` changes graph structure for elimination
   - Does NOT create graph whose PDF represents P(R·T)

### 9.2 About the Codebase

1. **Multiple computational backends**:
   - GraphBuilder (C++): Used by FFI, fast, structured
   - Trace evaluation (Python/C++): Used by trace API, very fast for repeated evals
   - Direct graph API (Python): Used by tests, flexible but slower

2. **Reference counting is complex**:
   - Graph wrapper has `rf_graph->references` shared pointer
   - Creating Graph from raw `ptd_graph*` is dangerous
   - Better to use C API directly

3. **FFI is critical for performance**:
   - `jax.ffi.ffi_call` enables pmap (distributed computing)
   - `pure_callback` works but slower, no pmap support
   - Future: Unified TraceEvaluator via FFI (see UNIFIED_ARCHITECTURE_PLAN.md)

### 9.3 About SVGD

1. **SVGD is sensitive to likelihood errors**:
   - 12% error in PDF → 12-32% error in parameter estimates
   - Systematic bias in PDF → systematic bias in estimates
   - More data amplifies the error (converges to wrong answer)

2. **Different features → different biases**:
   - Each reward vector creates different wrong distribution
   - Explains why feature 0 underestimates, feature 1 overestimates

3. **NaN masking works correctly**:
   - `jnp.where(mask, log(pmf), 0.0)` properly excludes missing data
   - Error is not due to NaN handling

---

## 10. Open Questions

### 10.1 Mathematical

**Q1**: What is the exact relationship between PDF(R·T) and PDF(T)?

For exponential: If T ~ Exp(λ), then R·T ~ Exp(λ/R) with PDF f(t) = (λ/R)·exp(-λt/R)

For phase-type: More complex, depends on path through graph and per-vertex rewards.

**Q2**: Can we derive PDF(R·T) analytically?

Possibly, but complex. Forward algorithm approach is more practical.

**Q3**: Why does `ptd_graph_reward_transform()` normalize by rate?

Looking at code, it converts from rate-based to probability-based representation for graph elimination. The normalization `edge_weight /= rate` converts rates to transition probabilities.

### 10.2 Implementation

**Q1**: Which uniformization approach is correct for rewards?

Option A: Adjust time step by reward: `dt_effective = dt * reward[v]`
Option B: Adjust transition probability: `prob += ... * reward[v]`
Option C: Adjust after absorption: `pdf *= product(rewards on path)`

Need to derive mathematically or compare with sampling.

**Q2**: Should we cache reward-transformed graphs?

Probably not - computing PDF with rewards directly is likely faster than transform + cache + PDF.

**Q3**: Will this work with DPH (discrete phase-type)?

Probably need separate function: `ptd_graph_dph_pmf_with_rewards()`

### 10.3 Testing

**Q1**: What is the simplest test case?

Single Exp(λ) vertex with reward r:
- Sample: Y = r·T where T ~ Exp(λ), so Y ~ Exp(λ/r)
- PDF: f_Y(y) = (λ/r)·exp(-λy/r)
- Test: `pdf_with_rewards(graph, t, [r])` should equal `(λ/r)*exp(-λ*t/r)`

**Q2**: How to validate complex graphs?

Compare with Monte Carlo:
1. Sample 100K observations with rewards
2. Compute kernel density estimate (KDE)
3. Compare with `ptd_graph_pdf_with_rewards()`
4. Should match within sampling error

---

## 11. References

### 11.1 Code Locations

**Sampling with rewards**:
- `src/c/phasic.c:4035-4080`: `ptd_random_sample(graph, rewards)`
- Line 4067: `waiting_time *= rewards[vertex->index]`

**Reward transformation**:
- `src/c/phasic.c:1987-2400`: `_ptd_graph_reward_transform()`
- `src/c/phasic.c:2376-2387`: `ptd_graph_reward_transform()` (public API)
- Lines 2063-2071: Edge weight and reward normalization

**PDF computation**:
- `src/cpp/parameterized/graph_builder.cpp:391-417`: `compute_pmf_and_moments()`
- Line 408: `pmf_vec[i] = g.pdf(times_vec[i], granularity)` (BUG: ignores rewards)
- `api/cpp/phasiccpp.h:491-518`: `Graph::pdf()` wrapper
- `src/c/phasic.c`: Probability distribution context (forward algorithm)

**Moment computation (works correctly)**:
- `src/cpp/parameterized/graph_builder.cpp:205-260`: `compute_moments_impl()`
- Line 216: `std::vector<double> rewards2 = g.expected_waiting_time(rewards)`
- `src/c/phasic.c`: `ptd_expected_waiting_time()` (handles rewards correctly)

**FFI integration**:
- `src/phasic/ffi_wrappers.py:628-643`: FFI registration for pmap support
- Line 637: `rewards` parameter exists but not used correctly in C++

**Multivariate wrapper**:
- `src/phasic/__init__.py:3229-3300`: `pmf_and_moments_from_graph_multivariate()`
- Uses `jax.lax.scan` to loop over features
- Passes 2D rewards array

### 11.2 Documentation

**Papers**:
- Røikjer, Hobolth & Munch (2022): "phasic: Fast Phase-Type Inference via Graph-Based Algorithms"
- Algorithm 4: Forward algorithm for PDF computation (uniformization)

**Project docs**:
- `CLAUDE.md`: Quick reference, API patterns
- `docs/api/Graph.qmd`: Graph API documentation
- `docs/api/SVGD.qmd`: SVGD API documentation

**Investigation docs** (this session):
- `MULTIVARIATE_ROOT_CAUSE.md`: Root cause analysis
- `MULTIVARIATE_FIX_PLAN.md`: Fix implementation plan
- `MULTIVARIATE_FIXES_SUMMARY.md`: Summary of dtype/NaN fixes
- `C++_FIX_ATTEMPTS_SUMMARY.md`: Failed C++ fix attempts
- `UNIFIED_ARCHITECTURE_PLAN.md`: Future architecture (TraceEvaluator via FFI)
- `MULTIVARIATE_INVESTIGATION_COMPLETE.md`: This file

### 11.3 Test Files

**Diagnostic**:
- `test_python_reward_transform.py`: Tests reward_transform on parameterized graphs
- `test_trace_reward_approach.py`: Tests trace + instantiate + transform
- `test_reward_diagnostics.py`: Comprehensive diagnostics

**Validation**:
- `test_nan_single_feature_proper.py`: Main test (10K obs, single feature)
- `test_one_feature_per_obs.py`: NaN masking test (one feature per obs)
- `test_all_features_10k.py`: Full test (4 features, 10K obs)

**Related**:
- `tests/test_multivariate.py`: Original multivariate tests
- `tests/test_feature_moments.py`: Moment tests (pass!)

### 11.4 Git Commits

Key commits in this investigation:

```
c2f9f82 Fix reward dtype handling for multivariate phase-type distributions
1a7ef59 Add multivariate phase-type distribution support with 2D observations and rewards
0c6e669 Support any number of moments in SVGD regularization
1973397 Fix broken inference by adding missing base_weight to parameterized edges
```

---

## 12. Next Steps for Implementation

### 12.1 Immediate (Alternative B)

1. **Study uniformization algorithm** (0.5 days)
   - Read Algorithm 4 in paper
   - Understand current implementation in `src/c/phasic.c`
   - Determine how to incorporate rewards

2. **Implement `ptd_graph_pdf_with_rewards()`** (1 day)
   - Create C function in `src/c/phasic.c`
   - Add declaration to `api/c/phasic.h`
   - Write unit tests in C

3. **Integrate with GraphBuilder** (0.5 days)
   - Modify `src/cpp/parameterized/graph_builder.cpp:400-412`
   - Add conditional: if rewards provided, use new function
   - Rebuild and test

4. **Validate** (0.5 days)
   - Run `test_nan_single_feature_proper.py` → expect 2% error
   - Run `test_one_feature_per_obs.py` → expect consistent estimates
   - Run `test_all_features_10k.py` → expect 2% error

5. **Commit and document**
   - Update `CLAUDE.md` with multivariate PDF fix
   - Add tests to test suite
   - Commit with message referencing issue

### 12.2 Medium-Term (Architecture)

From `UNIFIED_ARCHITECTURE_PLAN.md`:

1. **Implement TraceEvaluator** (2-3 days)
   - C++ class that evaluates traces with reward support
   - Register as JAX FFI target
   - Support pmap for distributed computing

2. **Migrate GraphBuilder to use TraceEvaluator** (1 day)
   - Change `pmf_and_moments_from_graph()` to record trace
   - Use TraceEvaluator via FFI
   - Keep GraphBuilder as fallback

3. **Unify Model API** (1 day)
   - Single code path for parameterized models
   - Transparent FFI/trace switching
   - Deprecate GraphBuilder eventually

**Total effort**: 7-9 days

### 12.3 Long-Term (Phase 5 continuation)

From `CLAUDE.md`:

1. **JAX FFI gradients** (Week 4)
   - Integrate Phase 5 Week 3 gradient implementation with FFI
   - Full autodiff support for PDF+gradients via XLA

2. **Extend to reward-transformed graphs** (Week 5)
   - Add gradient support for `ptd_graph_pdf_with_rewards()`
   - Enable HMC and gradient-based samplers

3. **Benchmark** (Week 6)
   - Test on large models (100+ vertices)
   - Optimize performance
   - Document best practices

---

## 13. Success Criteria

Fix is complete when ALL of these are met:

1. ✅ Single feature estimates θ = 10.0 ± 0.2 (2% error) with 10K observations
2. ✅ All features converge to SAME estimate (consistent across reward vectors)
3. ✅ Error DECREASES with more data (not increases)
4. ✅ No segfaults or memory errors
5. ✅ Backward compatible (no rewards → same behavior as before)
6. ✅ Works with both FFI and non-FFI modes
7. ✅ Works with both continuous (PDF) and discrete (PMF) distributions
8. ✅ Passes all existing tests plus new multivariate tests

---

## 14. Contact and Continuation

**Primary investigator**: Claude (AI assistant)
**User**: Kasper Munch (kaspermunch@birc.au.dk)
**Date**: 2025-10-26

**To continue this work**:

1. Read this document (`MULTIVARIATE_INVESTIGATION_COMPLETE.md`)
2. Review root cause (`MULTIVARIATE_ROOT_CAUSE.md`)
3. Review fix plan (`MULTIVARIATE_FIX_PLAN.md`)
4. Check git status for latest code state
5. Implement Alternative B as described in Section 7
6. Test using files in Section 11.3
7. Validate against Success Criteria (Section 13)

**Key insight**: The fix must be in the C forward algorithm, not in C++ graph transformation. No shortcuts work - we need `ptd_graph_pdf_with_rewards()`.

---

**END OF INVESTIGATION REPORT**
