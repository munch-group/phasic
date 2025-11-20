# Hierarchical SCC Caching - Harmonic Mean Approach Results

## Summary
Implemented harmonic mean approach based on user's insight, but **still getting 612% PDF error**.

## What Was Implemented

### 1. Enhanced Subgraph Building
- Connecting vertices are absorbing (no outgoing edges)
- Tried adding temp absorbing vertices but reverted (didn't help)

### 2. Rate Tracking
Compute incoming rate for each connecting vertex:
```python
# For connecting vertex V in upstream SCC:
incoming_rate = SUM(source_rate * edge_prob for all edges to V from internal vertices)
```

### 3. Harmonic Mean Computation
When processing vertex in its home SCC:
```python
# If vertex appeared in N upstream SCCs with rates r1, r2, ..., rN:
harmonic_mean = N / sum(1/r_i for all upstream rates)
merged.vertex_rates[v] = harmonic_mean
```

## Results
- **Sum of rates**: 612% error
- **Harmonic mean of rates**: 612% error (identical!)
- **Conclusion**: Both sum and harmonic mean produce the SAME wrong result

## Key Observation
The harmonic mean and sum give identical wrong results, which suggests:
1. Either the incoming rate computation is fundamentally wrong
2. OR replacing vertex rate (regardless of how) is the wrong approach
3. OR there's a deeper issue with how traces are stitched

## Operation Counts
- Direct: 25,027 operations
- Hierarchical (with harmonic mean): 14,018 operations
- The extra operations from harmonic mean computation don't fix the issue

## Hypotheses for Why This Fails

### Hypothesis 1: Edge Probabilities Need Scaling Too
Maybe we can't just replace vertex rate - we also need to scale edge probabilities so they remain normalized.

### Hypothesis 2: Initial Probability Vector
Maybe the initial probability distribution is affected by the SCC decomposition and needs adjustment.

### Hypothesis 3: Fundamental Misunderstanding
Maybe connecting vertices shouldn't have their rates replaced at all. Perhaps the stitching should work differently - e.g., by adjusting the edges BETWEEN SCCs rather than within them.

### Hypothesis 4: Elimination Order Matters
Maybe the issue is that elimination within each SCC changes the probability structure in a way that can't be simply corrected by adjusting rates.

## What's Still Wrong
Despite trying multiple approaches:
1. ❌ Replace rate with upstream rate (0% - total failure)
2. ❌ Replace rate with SUM of upstream flows (~10% error)
3. ❌ Scale edge probs + replace rate (~600% error)
4. ❌ Replace rate with SUM of incoming rates (612% error)
5. ❌ Replace rate with HARMONIC MEAN of incoming rates (612% error)

**None of these produce correct PDFs!**

## Next Steps to Consider

### Option A: Debug with Minimal Example
Create 2-SCC example by hand:
- Manually compute what merged trace should be
- Compare with what code produces
- Identify exact mismatch

### Option B: Review Phase-Type Theory
- Re-examine mathematical foundations
- Verify that rate replacement is valid operation
- Check if there's a normalization step missing

### Option C: Different Stitching Approach
Instead of modifying vertex rates, perhaps:
- Keep all SCC traces unchanged
- Add "glue" operations that connect SCCs
- Properly handle probability flow between SCCs

### Option D: Ask for Help
- Consult phase-type distribution literature
- Check if hierarchical SCC elimination is a known problem
- Look for existing algorithms

## Files Modified
- `src/phasic/hierarchical_trace_cache.py`: Lines 526-530 (enhanced subgraph), 922-1018 (stitching with harmonic mean)

## Test Files
- `test_trace_correctness.py`: PDF comparison test
- `test_simple_correctness.py`: Operation count test

## Conclusion
The harmonic mean approach, while mathematically sound for combining rates, does NOT fix the hierarchical caching bug. The problem likely lies elsewhere - either in how we're computing the rates to combine, or in a fundamental misunderstanding of how traces should be stitched together.

**The bug remains unsolved.**
