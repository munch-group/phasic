# Trace Stitching Implementation Status

**Date**: 2025-11-06
**Status**: ❌ Implementation blocked by fundamental issue

---

## Summary

Implemented trace stitching algorithm based on design in `TRACE_STITCHING_DESIGN.md`, but discovered a fundamental issue during testing that blocks the current approach.

## Implementation Completed

✅ **Core Algorithm** (`src/phasic/hierarchical_trace_cache.py`):
- `_build_vertex_mappings()`: Maps vertices between SCC traces and original graph
- `_remap_operation()`: Adjusts operation indices when merging traces
- `_add_boundary_edges()`: Adds edges crossing SCC boundaries
- `stitch_scc_traces()`: Main stitching function (343 lines)

✅ **Implementation Plan**: Detailed 15-day plan in `TRACE_STITCHING_IMPLEMENTATION_PLAN.md`

✅ **Test Suite**: Unit tests in `tests/test_trace_stitching.py`

---

## Fundamental Issue Discovered

### The Problem

**Error**: `RuntimeError: No such vertex found` when trying to map SCC trace vertices to original graph

**Root Cause**: SCC traces contain states from the ELIMINATED graph, not the original SCC subgraph.

### Example

Original graph vertices: `[3]`, `[2]`, `[1]`

After elimination on SCC subgraph:
- Elimination process creates new intermediate states (e.g., `[0]`)
- Trace records ALL states including these new ones
- State `[0]` doesn't exist in original graph → lookup fails

### Why This Happens

The `record_elimination_trace()` function records operations during graph elimination:
1. Takes input graph
2. Performs elimination (removes vertices, creates bypass edges)
3. Process can create new absorbing states
4. **Trace vertices ≠ Original graph vertices**

---

## Why Current Approach Fails

The algorithm assumes:
```python
# ASSUMPTION: SCC trace vertices correspond to original graph vertices
state = scc_trace.states[scc_v_idx]
orig_vertex = original_graph.find_vertex(state)  # FAILS!
```

But **reality**:
- `scc_trace.states` contains states from ELIMINATED graph
- Original graph doesn't have all these states
- State lookup fails for elimination-created states

---

## Possible Solutions

### Solution 1: Track Original Vertex Mapping in Trace

**Modify `record_elimination_trace()` to store mapping**:
```python
@dataclass
class EliminationTrace:
    # ... existing fields ...
    original_vertex_mapping: Dict[int, int]  # trace_v_idx → original_v_idx
```

**Pros**:
- Clean, direct solution
- Trace contains all info needed for stitching
- No changes to stitching algorithm logic

**Cons**:
- Requires modifying trace recording
- Changes core data structure
- Backward compatibility issues

### Solution 2: Record Traces Differently for Stitching

**Don't eliminate SCC subgraphs before recording**:
```python
# Instead of:
scc_subgraph = scc.as_graph()
trace = record_elimination_trace(scc_subgraph)  # Eliminates first!

# Do:
trace = record_trace_without_elimination(scc_subgraph, original_vertex_ids)
```

**Pros**:
- Traces preserve original vertex identity
- State lookup works as expected

**Cons**:
- Requires new trace recording function
- More complex trace format
- Defeats purpose of elimination-based traces

### Solution 3: Use SCC Subgraph Vertices Instead of Trace Vertices

**Map SCC subgraph vertices directly**:
```python
def _build_vertex_mappings_from_scc(scc_graph):
    """Map SCC subgraph vertices (not trace vertices) to original graph"""
    for scc_idx, scc in enumerate(sccs):
        scc_subgraph = scc.as_graph()

        for scc_v_idx in range(scc_subgraph.vertices_length()):
            scc_vertex = scc_subgraph.get_vertex(scc_v_idx)
            state = scc_vertex.state()

            # This WILL work - SCC vertices exist in original graph
            orig_vertex = original_graph.find_vertex(state)
            orig_v_idx = orig_vertex.index()

            vertex_to_original[(scc_idx, scc_v_idx)] = orig_v_idx
```

**Then somehow use this mapping when stitching traces...**

**Pros**:
- No changes to trace structure
- Uses existing SCC API

**Cons**:
- Not clear how to connect SCC vertex indices to trace vertex indices
- Trace vertices != SCC subgraph vertices (due to elimination)

### Solution 4: Different Caching Strategy

**Instead of stitching traces, cache at different level**:
- Cache full graph traces only (current Phase 3a approach)
- OR: Cache elimination operations separately and compose
- OR: Use symbolic DAG caching instead

**Pros**:
- Avoids trace stitching complexity entirely
- Simpler implementation

**Cons**:
- Less cross-graph reuse of SCC computations
- May not achieve desired performance goals

---

## Recommended Path Forward

### Short Term: Disable Trace Stitching

Keep Phase 3a implementation (full graph caching only):
```python
def get_trace_hierarchical(graph, param_length=None, min_size=50):
    # Just cache full graphs, no SCC subdivision
    hash_result = compute_graph_hash(graph)
    trace = _load_trace_from_cache(hash_result.hash_hex)
    if trace is None:
        trace = record_elimination_trace(graph, param_length)
        _save_trace_to_cache(hash_result.hash_hex, trace)
    return trace
```

**Status**: ✅ Already implemented and working

### Medium Term: Investigate Solution 1

**Add original vertex mapping to traces**:
1. Modify `EliminationTrace` to include `original_vertex_ids: List[int]`
2. Update `record_elimination_trace()` to populate this field
3. Use mapping in `_build_vertex_mappings()`

**Estimated effort**: 2-3 days

### Long Term: Consider Alternative Approaches

1. **Symbolic DAG stitching** instead of trace stitching
2. **Lazy trace evaluation** where traces reference other traces
3. **Graph-level caching** with operation-level reuse

---

## Lessons Learned

### 1. Traces Are Post-Elimination

Elimination traces represent the graph AFTER elimination, not before. This is fundamental to how they work but was not fully appreciated in the design phase.

### 2. State-Based Lookup Has Limits

Looking up vertices by state works for original graph vertices, but fails for elimination-created states.

### 3. Need Bidirectional Mapping

Stitching requires knowing:
- Which trace vertex corresponds to which original vertex (MISSING!)
- Which original vertex corresponds to which SCC (HAVE)

### 4. Design Documents vs Implementation

The design in `TRACE_STITCHING_DESIGN.md` made assumptions about trace structure that weren't validated against actual trace behavior.

---

## Files Modified

### Implemented
- `src/phasic/hierarchical_trace_cache.py`: +343 lines (stitching implementation)
- `tests/test_trace_stitching.py`: +153 lines (test suite)
- `TRACE_STITCHING_IMPLEMENTATION_PLAN.md`: Detailed implementation plan

### Working (Phase 3a)
- `src/phasic/trace_serialization.py`: Hybrid C/Python caching ✅
- Cache load/save functions ✅
- Full graph trace caching ✅

---

## Next Steps

**Immediate**:
1. Document this issue for future reference ✅ (this file)
2. Update `IMPLEMENTATION_ROADMAP.md` with findings
3. Decide on path forward (likely: defer trace stitching, keep Phase 3a)

**Future**:
1. Design solution for original vertex mapping in traces
2. Prototype Solution 1 (add mapping to trace structure)
3. Update trace stitching implementation once mapping available

---

## Conclusion

Trace stitching as designed requires information not currently in traces (mapping from trace vertices to original vertices).

**Current Phase 3a implementation (full graph caching only) is complete and working.**

Trace stitching requires either:
- Modifying trace structure to include vertex mappings, OR
- Finding a different approach to hierarchical caching

---

**Status**: Implementation paused pending design decision
**Risk**: High - requires core data structure changes or algorithmic rethink
**Recommendation**: Defer trace stitching, continue with Phase 3a (full graph caching)
