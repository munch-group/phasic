# GraphBuilder Model Correctness Investigation Plan

**Date**: 2025-10-30
**Objective**: Diagnose why SVGD with θ=10 and 10k observations produces 34% error instead of <2%
**Hypothesis**: GraphBuilder-compiled model returns incorrect probabilities (not SVGD convergence failure)
**Approach**: Systematic testing of GraphBuilder outputs vs empirical data

---

## Background

Previous investigation revealed:
- Direct θ=10 test: 44.68% error
- Scaled model (θ_scaled=2, rates×5): 33.88% error
- **Key insight**: Even with perfect prior initialization, systematic ~34% underestimation persists
- **Conclusion**: Likely a bug in GraphBuilder model outputs, not SVGD convergence

---

## Phase 1: Test 1D Model Correctness

**File**: `tests/test_graphbuilder_1d_correctness.py`

### Tests

**Test 1.1: First Moment vs Empirical Mean**
- Generate 100,000 samples using `graph.sample()`
- Compute empirical mean
- Compare to GraphBuilder first moment
- **Success**: < 0.1% relative error
- **Test for**: θ ∈ {1.0, 2.0, 5.0, 10.0}

**Test 1.2: PMF Normalization**
- Compute PMF over dense time grid (0.001 to 5.0, 1000 points)
- Integrate using trapezoid rule
- **Success**: ∫ PMF dt = 1.0 ± 0.001

**Test 1.3: PMF vs Empirical Distribution**
- Generate 100k samples
- Create histogram (50 bins)
- Compare to GraphBuilder PMF
- **Success**: Chi-square test p-value > 0.05

**Diagnostic Output**:
- Print serialized graph structure
- Inspect start/end states
- Verify edge rates and probabilities

---

## Phase 2: Test Reward Transformation

**File**: `tests/test_reward_transformation.py`

### Structural Invariants (CRITICAL)

**Invariant 1: Single Absorbing State**
```python
absorbing = [v for v in transformed.vertices() if len(list(v.outgoing_edges())) == 0]
assert len(absorbing) == 1
```

**Invariant 2: Start State Transitions Sum to 1.0**
```python
total_prob = sum(edge.probability() for edge in start.outgoing_edges())
assert abs(total_prob - 1.0) < 1e-10
```

**Invariant 3: No Stale Edges**
```python
all_vertex_ids = {v.id() for v in transformed.vertices()}
for v in transformed.vertices():
    for edge in v.outgoing_edges():
        assert edge.target().id() in all_vertex_ids
```

**Invariant 4: Vertex Count = Nonzero Rewards + 1**
```python
expected = np.count_nonzero(reward_vector) + 1  # +1 for absorbing
assert transformed.vertices_length() == expected
```

### Correctness Tests

**Test 2.1: Reward-Transformed Moments**
- For each reward feature from `_graph.states()[:, :-2]`
- Sample 100k observations with `_graph.sample(n, rewards=reward_col)`
- Compare GraphBuilder moment to empirical mean
- **Success**: < 0.1% relative error

**Test 2.2: Uniform Rewards = No Transformation**
- Rewards all 1.0 should match untransformed graph
- Verify PMF and moments identical

---

## Phase 3: Test 2D Multivariate Model

**File**: `tests/test_graphbuilder_2d_correctness.py`

### Sampling Pattern (Exact from User)

```python
rewards = _graph.states()[:, :-2]
a = np.empty((rewards.shape[1], n * rewards.shape[1]))
a[:] = np.nan
for i in range(rewards.shape[1]):
    a[i, i*n:(i+1)*n] = _graph.sample(n, rewards=rewards[:, i])
observed_data = jnp.array(a).T
```

### Tests

**Test 3.1: 2D PMF Independence**
- Each column of 2D PMF must match corresponding 1D PMF
- Use `np.testing.assert_allclose(pmf_2d[:, i], pmf_1d, rtol=1e-10)`

**Test 3.2: Sparse NaN Handling**
- Verify log-likelihood correctly skips NaN values
- Check only non-NaN values contribute to likelihood

**Test 3.3: Moment Aggregation**
- 2D moments shape: (n_features, nr_moments)
- Each row must match corresponding 1D moment

---

## Expected Outcomes

| Phase | Outcome | Interpretation |
|-------|---------|----------------|
| Phase 1 passes | GraphBuilder PMF/moments correct | Bug is in SVGD/gradients |
| Phase 1 fails | GraphBuilder model broken | Likely C++ forward algorithm or parameter scaling |
| Phase 2 fails (invariants) | Reward transformation structural bugs | Graph structure corrupted |
| Phase 2 fails (correctness) | Reward sampling/moments wrong | Reward application incorrect |
| Phase 3 fails | Multivariate handling broken | 2D implementation issues |

---

## Success Criteria Summary

- **Phase 1**: All 3 tests pass for all θ values
- **Phase 2**: All 4 invariants hold + moments match empirical
- **Phase 3**: 2D PMF independence + correct NaN handling + moment aggregation

---

## Timeline

- Phase 1: ~10 minutes
- Phase 2: ~10 minutes
- Phase 3: ~10 minutes
- **Total: ~30 minutes**

---

## Post-Investigation Actions

### If GraphBuilder is Correct (Phase 1 passes)
→ Investigate SVGD gradients, kernel function, or likelihood computation

### If GraphBuilder is Broken (Phase 1 fails)
→ Debug C++ forward algorithm, parameter application, or serialization

### If Specific θ Values Fail
→ Investigate numerical precision issues, granularity settings, or parameter scaling

---

**Prompt to Resume After Compaction**:

```
Please execute the investigation plan in GRAPHBUILDER_TEST_PLAN.md:

1. Create tests/test_graphbuilder_1d_correctness.py and run Phase 1 tests
2. Create tests/test_reward_transformation.py and run Phase 2 tests
3. Create tests/test_graphbuilder_2d_correctness.py and run Phase 3 tests

For each phase, report:
- Which tests pass/fail
- Error percentages where applicable
- Diagnostic output from failing tests

This will identify whether the θ=10 SVGD failure is due to incorrect GraphBuilder model outputs.
```
