# Hierarchical SCC Sister Merging - Implementation Progress

**Date**: 2025-11-11
**Status**: ⚠️ Partial Success - Sister detection working, but duplicate edges remain

---

## Summary

Implemented hierarchical SCC trace caching with sister vertex merging according to user's design. The system now correctly identifies sister vertices and attempts to merge them, but duplicate edges are still being created. Progress from initial 600%+ PDF errors to current issues with trace structure.

---

## What Was Implemented

### 1. Helper Functions for Vertex Categorization (lines 439-609)

Created functions to identify 5 vertex categories in enhanced SCC subgraphs:

```python
def _find_upstream_vertices(original_graph, internal_indices, scc_graph):
    """Find vertices in upstream SCCs that connect to current SCC."""

def _find_upstream_connecting(internal_indices, upstream_vertices, original_graph):
    """Find internal vertices receiving from upstream."""

def _find_downstream_connecting(internal_indices, original_graph):
    """Find internal vertices connecting to downstream."""

def _find_downstream_vertices(original_graph, downstream_connecting, internal_indices):
    """Find vertices in downstream SCCs."""
```

### 2. Enhanced Subgraph Builder (lines 612-754)

Rewrote `_build_enhanced_scc_subgraph()` to:
- Return tuple `(Graph, Dict[metadata])`
- Implement 5-part vertex ordering: `{*upstream, *upstream_connecting, *internal, *downstream_connecting, *downstream}`
- Store full metadata with all vertex categories

### 3. First SCC Special Handling (lines 757-845)

Added `_build_first_scc_subgraph()` for special case:
- Contains only `{Starting_vertex, *downstream}`
- No upstream vertices (it's the first!)

### 4. Sister Vertex Detection (lines 848-932)

Added `_find_sister_vertices()`:
- Matches vertices by state vector between SCCs
- Returns list of (upstream_idx, downstream_idx) pairs
- Correctly identifies C and E as sisters in test graph

### 5. Metadata Inference from Cached Traces (lines 1195-1218)

**CRITICAL FIX**: When traces are loaded from cache, metadata must be reconstructed:

```python
# Infer ordered_vertices by matching states
ordered_vertices = []
for trace_state in scc_trace.states:
    orig_idx = find_vertex_by_state(original_graph, trace_state)
    ordered_vertices.append(orig_idx)

# Recompute vertex categories using same logic as _build_enhanced_scc_subgraph
internal_indices = scc.internal_vertex_indices()
upstream_vertices = _find_upstream_vertices(original_graph, internal_indices, scc_graph)
upstream_connecting = _find_upstream_connecting(internal_indices, upstream_vertices, original_graph)
downstream_connecting = _find_downstream_connecting(internal_indices, original_graph)
downstream_vertices = _find_downstream_vertices(original_graph, downstream_connecting, internal_indices)
```

### 6. Sister Merging Logic (lines 1303-1341)

Completely rewrote merging in `stitch_scc_traces()`:

```python
# SKIP upstream vertices - already processed in their home SCC
if orig_idx in upstream_in_current_scc:
    continue

# SKIP downstream vertices, but first attach edges to upstream sister if applicable
if orig_idx in downstream_in_current_scc:
    if is_sister:
        # Attach downstream sister's edges to upstream sister
        for edge in downstream_sister_edges:
            upstream_sister.add_edge(edge)
    continue
```

---

## Current Status

### ✅ Working Correctly

1. **SCC Decomposition**: Correctly identifies 2 cyclic SCCs
   - SCC2: {E, D, C} (vertices [5,6,4], states [6,5,4])
   - SCC3: {B, A} (vertices [3,2], states [3,2])

2. **Sister Detection**: Finds sister pairs
   - SCC 3: Found 1 sister pair (likely C or E)
   - SCC 5: Found 1 sister pair

3. **Metadata Inference**: Reconstructs vertex categories from cached traces

4. **Edge Reduction**: Reduced from 34 edges (100% duplicates) to 22 edges (partial duplicates)

### ⚠️ Remaining Issues

1. **Duplicate Edges**: Still present but reduced
   ```
   Vertex 0: 2 edges to [1] (should be 1)
   Vertex 1: 2 edges to [2] (should be 1)
   Vertex 2: 2 edges to [3] (should be 1)
   Vertex 3: 5 edges [2,4,5,4,5] (should be 3: to 2,4,5)
   Vertex 4: 4 edges [6,7,6,7] (should be 2: to 6,7)
   Vertex 6: 4 edges [5,7,7,4] (should be 2: to 5,7)
   ```

2. **PDF Still Zero**: Hierarchical PDF = 0 (direct PDF ~0.7)
   - Indicates incorrect trace structure
   - Likely due to remaining duplicate edges

3. **Skipping Logic Not Triggering**: DEBUG messages for "Skipping upstream/downstream vertex" don't appear
   - Suggests upstream/downstream detection may not be working as expected
   - Or vertices aren't being categorized correctly

---

## Test Graph Structure

Created `/Users/kmt/phasic/test_scc_graph.py` with user-specified structure:

**Graph**: Starting→A, A→B, B→A, B→C, B→E, C→D, C→F, D→E, D→F, E→C

**Vertices** (after fixing Graph API issues):
- Vertex 0: state=[0] (TrueStart, auto-created)
- Vertex 1: state=[1] (LogicalStart)
- Vertex 2: state=[2] (A)
- Vertex 3: state=[3] (B)
- Vertex 4: state=[4] (C)
- Vertex 5: state=[6] (E)  # Note: states [5] and [6] swapped in discovery order
- Vertex 6: state=[5] (D)
- Vertex 7: state=[7] (F, absorbing)

**Expected Sister Vertices**: C and E should appear in both:
- Enhanced {A,B}: downstream fake absorbing
- Enhanced {C,D,E}: internal

---

## Key Insights Learned

### 1. Graph Constructor API

The Graph constructor with callback + ipv:
- ALWAYS creates vertex 0 with state=[0] as true starting vertex
- `ipv` defines the FIRST reachable state from true starting vertex
- DO NOT use state=[0] in callback or ipv - it creates duplicates!

### 2. Metadata Must Be Cached or Reconstructed

Enhanced subgraph metadata contains critical information:
- `upstream`: Fake starting vertices (from previous SCCs)
- `downstream`: Fake absorbing vertices (for future SCCs)
- `internal`: True internal vertices of this SCC

Without this metadata, stitching doesn't know which vertices to skip, creating duplicates.

### 3. Sister Merging Requires Careful Ordering

- Process SCCs in topological order
- Skip upstream vertices (already processed)
- Skip downstream vertices BUT attach their edges to upstream sisters first
- Only process internal vertices in their home SCC

---

## Next Steps

### Immediate Debugging

1. Add detailed logging to show which vertices are being categorized as upstream/downstream
2. Check if `upstream_in_current_scc` and `downstream_in_current_scc` sets are correctly populated
3. Verify that the skipping logic is actually being executed

### Potential Issues to Investigate

1. **Are vertices being processed multiple times?**
   - Check if a vertex appears as "internal" in multiple SCCs
   - This would explain why edges are added multiple times

2. **Is the ordered_vertices list correct?**
   - Mismatch between trace vertex indices and original graph indices
   - Could cause wrong vertices to be skipped

3. **Are sister edges being attached correctly?**
   - Check if edge attachment logic is working
   - Verify target vertex indices are correct after remapping

### Alternative Approaches if Current Fix Fails

1. **Track processed vertices**: Maintain a set of already-processed vertex indices and skip any that have been processed

2. **Clear edges before adding**: When processing a vertex, clear its existing edges first

3. **Different stitching strategy**: Instead of merging traces, create a NEW trace from scratch using the SCC traces as input

---

## Files Modified

- `/Users/kmt/phasic/src/phasic/hierarchical_trace_cache.py`: Complete rewrite of stitching logic
  - Lines 439-609: Helper functions
  - Lines 612-754: Enhanced subgraph builder
  - Lines 757-845: First SCC handler
  - Lines 848-932: Sister detection
  - Lines 1195-1218: Metadata inference
  - Lines 1303-1341: Sister merging logic

- `/Users/kmt/phasic/test_scc_graph.py`: New test file with user-specified graph

- Backup: `/Users/kmt/phasic/src/phasic/hierarchical_trace_cache.py.backup_before_sister_merge`

---

## Performance

- Graph: 8 vertices, 11 edges
- Direct trace: 115 operations
- Hierarchical trace: 157 operations (should be ~same or less)
- Edge count: 22 (should be 11)

---

*This document tracks the implementation progress of hierarchical SCC trace caching with sister vertex merging. The approach is conceptually sound but execution has edge duplication bugs that need debugging.*
