# Hierarchical SCC Trace Caching - Separate Builders Status

**Date**: 2025-11-11
**Status**: ⚠️ Builders Implemented - Duplicate Edges Remain

---

## Summary

Implemented separate subgraph builders for first vs non-first SCCs as requested. The distinction is now crystal clear in the code, but duplicate edges persist in the stitched trace (15 edges instead of 11).

**Root Cause Identified**: `ordered_vertices` contains duplicate entries (e.g., `[None, 3, 4, 5, 4, 6, 7]` - vertex 4 appears twice).

---

## What Was Implemented ✅

### 1. `_build_first_scc_subgraph()` - Lines 784-910

**Purpose**: Build subgraph for first SCC containing actual starting vertex

**Key Design**:
- Auto-starting vertex **IS** the original graph's starting vertex
- Reuses auto-start (no separate creation needed)
- NO upstream vertices
- Structure: `{starting_vertex, *downstream}`

**Trace Mapping**: Direct 1:1
- `trace[0] → ordered[0]` (starting vertex)
- `trace[1] → ordered[1]` (first downstream)
- ...

**Metadata**:
```python
{
    'upstream': [],
    'internal': [starting_vertex],
    'downstream': [downstream_vertices],
    'ordered_vertices': [starting_vertex, *downstream]  # NO None
}
```

### 2. `_build_scc_subgraph()` - Lines 612-781

**Purpose**: Build subgraph for non-first SCCs with upstream vertices

**Key Design**:
- Auto-starting vertex **NOT** in original graph
- Creates NEW vertices for all ordered vertices
- HAS upstream vertices from previous SCCs
- Structure: `{*upstream, *upstream_connecting, *internal, *downstream_connecting, *downstream}`

**Trace Mapping**: Skip auto-start
- `trace[0] → None` (skip - auto-start not in original)
- `trace[1] → ordered[0]` (first upstream)
- `trace[2] → ordered[1]`
- ...

**Metadata**:
```python
{
    'upstream': [upstream_vertices],
    'upstream_connecting': [...],
    'internal': [...],
    'downstream_connecting': [...],
    'downstream': [downstream_vertices],
    'ordered_vertices': [None, *ordered...]  # None at index 0
}
```

### 3. Updated Callers

- **collect_missing_traces_batch()** (lines 225-235): Checks `if i == 0` to call correct builder
- **compute_scc_traces()** (lines 1154-1163): Same detection logic
- **stitch_scc_traces()** (lines 1433-1437): Skips None entries during stitching

### 4. Fixed UnboundLocalError

- **trace_elimination.py** (line 469): Moved `MAX_PARAM_TEST` definition outside conditional

---

## Test Results

### Subgraph Structure ✅ CORRECT

```
SCC 1 (first):
  ordered_vertices = [0, 1]  # NO None
  Subgraph: 2 vertices
  Mapping: trace[0]→orig[0], trace[1]→orig[1]

SCC 2 (non-first):
  ordered_vertices = [None, 3, 4, 5, 4, 6, 7]  # None at index 0
  Subgraph: 7 vertices
  Mapping: trace[0]=None (skip), trace[1]→orig[3], ...
```

All subgraphs have correct None-placement ✅

### Trace Completion ✅ NO CRASHES

```
Direct trace: 115 operations, 8 vertices
Hierarchical trace: 131 operations, 8 vertices
```

Stitching completes without errors!

### Duplicate Edges ❌ NOT FIXED

```
Expected: 11 edges
Actual: 15 edges (36% extra)

Vertex 0: 2 edges → [1, 1]     (should be 1: [1])
Vertex 2: 2 edges → [3, 3]     (should be 1: [3])
Vertex 3: 5 edges → [2,4,5,4,5] (should be 3: [2,4,5])
Vertex 5: 2 edges → [4, 4]     (should be 1: [4])
Vertex 6: 4 edges → [5,7,7,4]   (should be 2: [5,7])
```

Pattern: Almost all vertices have exact duplicate edges

---

## Root Cause: Duplicates in ordered_vertices

Looking at SCC 2:
```python
ordered_vertices = [None, 3, 4, 5, 4, 6, 7]
                                ^     ^
                          vertex 4 appears TWICE!
```

### Why Duplicates Occur

The 5-part concatenation creates duplicates:

```python
ordered_vertices = (
    upstream_vertices +          # e.g., [3]
    upstream_connecting +        # e.g., []
    internal_only +              # e.g., [4, 5]
    downstream_connecting +      # e.g., [4, 6]  # vertex 4 again!
    downstream_vertices          # e.g., [7]
)
# Result: [3, 4, 5, 4, 6, 7] - vertex 4 appears at indices 1 and 3
```

**Problem**: A vertex can be BOTH:
- Internal (member of the SCC)
- Downstream-connecting (sends edges to future SCCs)

So it appears in BOTH `internal_only` AND `downstream_connecting`!

Wait, that's wrong - looking at the code, `internal_only` should exclude connecting vertices:

```python
connecting_set = set(upstream_connecting + downstream_connecting)
internal_only = [v for v in internal_indices if v not in connecting_set]
```

So the issue must be elsewhere...

Actually, looking more carefully at the output:
```
ordered_vertices: [None, 3, 4, 5, 4, 6, 7]
```

The duplicate is vertex 4 at positions 2 and 4. Let me check which categories these correspond to:

Position 0: None (auto-start)
Position 1: First ordered vertex (upstream[0] = 3?)
Position 2: Second ordered vertex (upstream[1] or upstream_connecting[0] = 4?)
Position 3: Third ordered vertex (internal_only[0] = 5?)
Position 4: Fourth ordered vertex (downstream_connecting[0] = 4?) ← duplicate!

So vertex 4 appears in TWO categories (likely upstream AND downstream_connecting).

### The Real Issue

When building vertex categories, we're including vertices from OTHER SCCs:

- **upstream_vertices**: Vertices from PREVIOUS SCCs that connect to this one
- **downstream_vertices**: Vertices from FUTURE SCCs that this connects to

But vertex 4 is internal to THIS SCC! It shouldn't appear in upstream!

The bug must be in the helper functions that categorize vertices.

---

## Next Steps

1. **Debug vertex categorization**
   - Add detailed logging to show which vertices are in each category
   - For SCC 2, print out exactly what's in:
     - `upstream_vertices`
     - `upstream_connecting`
     - `internal_only`
     - `downstream_connecting`
     - `downstream_vertices`

2. **Fix category assignment**
   - Ensure vertices internal to an SCC don't appear in upstream/downstream
   - Verify helper functions are working correctly

3. **Test de-duplication**
   - If categorization can't be fixed, add de-duplication:
     ```python
     seen = set()
     deduped = []
     for v in ordered_vertices:
         if v not in seen and v is not None:
             deduped.append(v)
             seen.add(v)
     ordered_vertices = deduped
     ```

4. **Fix param_length mismatch**
   - First SCC trace has param_length=0 but should be 1
   - Pass explicit param_length to all trace recordings

---

## Files Modified

- `/Users/kmt/phasic/src/phasic/hierarchical_trace_cache.py`:
  - Implemented `_build_first_scc_subgraph()` (lines 784-910)
  - Renamed `_build_enhanced_scc_subgraph()` → `_build_scc_subgraph()` (lines 612-781)
  - Updated callers (lines 225-235, 1154-1163)
  - Added None-handling in stitching (lines 1433-1437, 1486-1489)

- `/Users/kmt/phasic/src/phasic/trace_elimination.py`:
  - Fixed `UnboundLocalError` (line 469)

- `/Users/kmt/phasic/inspect_hierarchical_debug.py`:
  - Updated to use new builders (lines 96-112)

---

*This document tracks the implementation of separate builders for first and non-first SCCs. The separation is correct, but duplicate edges remain due to issues in vertex categorization.*
