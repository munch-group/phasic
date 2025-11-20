# Hierarchical SCC Caching - Debugging Session 2

## Problem
Hierarchical trace produces PDF values of exactly 0.0, versus correct values from direct trace.

## Initial Attempt: Replace Vertex Rates
I tried replacing the vertex rate with the upstream absorption rate:
```python
merged.vertex_rates[merged_v_idx] = upstream_total_op
```

**Result**: Vertex 1 got rate_op=0 (operation index 0 from SCC 0), which replaced the correct rate_op=12158. This caused the hierarchical graph to have `rate=0.0` for vertex 1, breaking all PDF computations.

## Root Cause Analysis

### Key Finding
```
SCC 0:
- Operation offset: 0
- Copied 8 operations (indices 0-7)
- Vertex 1 saved with rate_op=0 (first operation)

When vertex 1 processed in its home SCC:
- upstream_total_op=0 (from SCC 0)
- old_rate_op=12158 (correct local rate)
- Replacement: 12158 → 0 causes vertex rate to become 0.0
```

### Why Operation 0?
SCC 0 is the FIRST SCC in topological order. It contains the starting vertex and possibly very few internal vertices. Vertex 1 appears as a **connecting vertex** (absorbing) in SCC 0's enhanced subgraph.

When recording the elimination trace for SCC 0's enhanced subgraph, vertex 1's rate is stored as operation 0, which is likely:
- CONST(0) if vertex 1 has no incoming edges in SCC 0, OR
- A very small value representing minimal absorption

This is NOT the value we should use!

## The Real Problem

**Key Insight**: The "rate" of a connecting vertex in an enhanced subgraph does NOT represent the total flow to that vertex - it's set during elimination and may be 0 for absorbing vertices!

What we ACTUALLY need is the sum of **edge rates** from internal vertices TO the connecting vertex, not the vertex's own rate in the elimination trace.

## Correct Fix (Not Yet Implemented)

### Option A: Track Edge Flows (Better Approach)
Instead of saving `vertex_rates[connecting_v]`, we should:
1. During SCC trace recording, sum all edge rates FROM internal vertices TO each connecting vertex
2. Save these sums as "exit flows" for each connecting vertex
3. Use these exit flows to scale edge probabilities in the home SCC

### Option B: Compute From Edges (Current Data)
Use the edge probability data already in the trace:
1. When processing connecting vertex in its home SCC:
   - Find all edges TO this vertex from internal vertices in upstream SCCs
   - Sum the edge rates (edge_prob * source_vertex_rate for each)
   - This gives total upstream flow
2. Scale edge probabilities: `scale = upstream_flow / local_rate`
3. Apply scale to all outgoing edge probabilities

### Option C: Fix Enhanced Subgraph Rate Computation
Modify `_build_enhanced_scc_subgraph` to compute and store the correct "exit rate" for connecting vertices:
- For each connecting vertex
- Sum all incoming edge rates from internal vertices
- Store this as a custom operation in the trace
- Use this for scaling

## Implementation Status
- ✅ Tracking system works (connecting_rates dict populated correctly)
- ✅ Detection works (vertices identified correctly)
- ❌ Rate values are WRONG (operation 0 instead of actual flow)
- ❌ Replacement approach doesn't work (breaks vertex rates)
- ⏳ Need to implement proper edge flow tracking

## Next Steps
1. Modify SCC trace recording to track edge flows to connecting vertices
2. Use these flows (not vertex rates) for scaling
3. Implement edge probability scaling with upstream_flow / local_rate
4. Test PDF correctness

## Files to Modify
- `src/phasic/hierarchical_trace_cache.py`:
  - `_build_enhanced_scc_subgraph()`: Add edge flow computation
  - `stitch_scc_traces()`: Use edge flows instead of vertex rates for scaling
