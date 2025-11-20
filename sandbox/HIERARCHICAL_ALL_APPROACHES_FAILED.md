# Hierarchical SCC Caching - All Vertex Rate Replacement Approaches Have Failed

## Summary of All Attempts

| Approach | PDF Error | Notes |
|----------|-----------|-------|
| 1. Replace rate with upstream `vertex_rates[connecting_v]` | 100% (PDFs = 0) | Connecting vertices have rate = 0 in elimination traces |
| 2. Replace rate with SUM of upstream flows | 10% | Better but still wrong |
| 3. Replace rate + scale edge probs | 600% | Worse than sum alone |
| 4. Replace rate with SUM of incoming rates (no temp absorbing) | 612% | Much worse |
| 5. Replace rate with HARMONIC MEAN of incoming rates (n/sum(1/rᵢ)) | 612% | Identical to sum |
| 6. Replace rate with SUM + renormalize edge probs | 27,200% | **Worst result yet!** |
| 7. Replace rate with 1/sum(1/rᵢ) + renormalize edge probs | 272% | Better than #6 but still very wrong |

## Key Observations

### 1. Sum vs Harmonic Mean Makes No Difference
Approaches 4 and 5 gave **identical** results (612% error), which is suspicious. This suggests:
- Either the incoming rate computation is wrong
- OR the fundamental issue isn't about sum vs harmonic mean

### 2. Renormalization Made It Worse
Adding edge probability renormalization (approach 6) increased the error from 612% to 27,200%. This suggests:
- The renormalization formula is wrong, OR
- Renormalizing edge probabilities while also replacing vertex rate creates a mathematical inconsistency

### 3. Temp Absorbing Vertices Don't Help
With or without temp absorbing vertices, we get the same errors. This tells us:
- The temp absorbing vertices don't fundamentally change anything
- They just make connecting vertices have rate = theta[0] instead of 0
- But we still compute incoming flow manually, so it doesn't matter

## The Fundamental Problem

**Hypothesis**: Replacing vertex rates is WRONG.

In a phase-type distribution:
- `vertex_rate` = total exit rate from the vertex
- `edge_prob[i]` = probability of taking edge i
- `edge_rate[i]` = `vertex_rate * edge_prob[i]`

The elimination trace records edge probabilities that are **normalized relative to the LOCAL vertex rate**. When we replace the vertex rate with something else (upstream flow), we break this relationship!

### Why Renormalization Failed

We tried: `new_edge_prob = old_edge_prob * (local_rate / total_incoming_flow)`

This should keep edge rates constant:
```
new_edge_rate = new_edge_prob * new_vertex_rate
              = (old_edge_prob * local_rate / total_incoming) * total_incoming
              = old_edge_prob * local_rate
```

But this gave 27,200% error! This suggests that **even keeping edge rates constant is wrong**.

## What This Means

The problem is NOT:
- ❌ How to compute incoming flow (we've tried multiple ways)
- ❌ Whether to use sum or harmonic mean (both fail)
- ❌ Whether to renormalize edge probabilities (makes it worse)

The problem IS:
- ✅ **The fundamental approach of modifying vertex rates is WRONG**
- ✅ **There's something deeper about how traces should be stitched**

## The Real Solution (Hypothesis)

Instead of modifying vertex rates or edge probabilities, maybe we need to:

### Option A: Don't Modify Anything
- Keep all SCC traces exactly as they are
- Somehow "connect" them without modifying internal structure
- Maybe add special "bridge" operations

### Option B: Different Decomposition
- Maybe enhanced subgraphs aren't the right approach
- Use isolated subgraphs instead?
- Or a completely different decomposition strategy

### Option C: Re-normalize At Evaluation Time
- Keep traces as-is
- When evaluating with concrete parameters, apply corrections
- Not in the trace itself, but in the evaluation logic

### Option D: Fundamental Rethinking
- Maybe hierarchical SCC caching for phase-type distributions is fundamentally incompatible with trace-based elimination
- The elimination process changes probability structure in ways that can't be simply "stitched" back together
- Need a completely different approach

## Recommendation

**STOP trying to modify vertex rates or edge probabilities.**

The next step should be:
1. Create a minimal 2-SCC example by hand
2. Manually compute what the correct merged trace should be
3. Compare with what any approach produces
4. Identify the ACTUAL mathematical relationship needed

Without understanding the correct mathematical relationship, we're just guessing at fixes.

## Files Currently Modified
- `src/phasic/hierarchical_trace_cache.py`: Lines 526-1062 (all broken approaches)

All these modifications should probably be reverted and we should start fresh with a different strategy.
