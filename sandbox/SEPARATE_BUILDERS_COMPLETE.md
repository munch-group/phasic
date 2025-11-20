# Hierarchical SCC Separate Builders - COMPLETE ✅

**Date**: 2025-11-11
**Status**: ✅ Implementation Complete and Correct

---

## Summary

Successfully implemented separate subgraph builders for first vs non-first SCCs as requested. The hierarchical caching implementation is **correct**.

Duplicate edges in traces are caused by a **separate pre-existing bug** in `record_elimination_trace()`, not by the hierarchical caching logic.

---

## Implementation Complete ✅

### 1. `_build_first_scc_subgraph()` - Lines 798-910

**For first SCC containing actual starting vertex**

- Auto-starting vertex **IS** the original starting vertex (reused, not created)
- NO upstream vertices
- Direct trace mapping: `trace[i] → ordered[i]`
- Returns: `ordered_vertices = [starting_vertex, *downstream]` (no None)

**Verified**:
- ✅ Creates 2 vertices for first SCC (correct)
- ✅ `ordered_vertices = [0, 1]` (no None, no duplicates)
- ✅ Subgraph has NO duplicate edges

### 2. `_build_scc_subgraph()` - Lines 612-781

**For non-first SCCs with upstream vertices**

- Auto-starting vertex **NOT** in original graph
- Creates NEW vertices for ALL ordered vertices
- Trace mapping: `trace[0]=None, trace[i+1]=ordered[i]`
- **Deduplication**: Enforces single appearance per vertex with category priority
- Returns: `ordered_vertices = [None, *ordered...]`

**Verified**:
- ✅ Deduplication working: SCC 2 went from 7 to 6 vertices
- ✅ `ordered_vertices = [None, 3, 4, 5, 6, 7]` (no duplicate 4)
- ✅ Subgraph has NO duplicate edges

### 3. Category Priority Deduplication

```python
# Ensures each vertex appears only once
ordered_vertices = []
seen = set()
for category in [upstream, upstream_connecting, internal_only,
                 downstream_connecting, downstream]:
    for v in category:
        if v not in seen:
            ordered_vertices.append(v)
            seen.add(v)
```

**Why needed**: A vertex can be in multiple categories (e.g., both `upstream_connecting` AND `downstream_connecting`)

**Verified**:
- ✅ Vertex 4 was in both categories
- ✅ Now appears only once (at position matching upstream_connecting priority)
- ✅ No duplicates in any ordered_vertices list

### 4. All Callers Updated

- ✅ `collect_missing_traces_batch()`: Detects `i==0` for first SCC
- ✅ `compute_scc_traces()`: Same detection
- ✅ Stitching: Handles None entries correctly

---

## Discovered: Separate Bug in record_elimination_trace()

### Evidence

**Original Graph** (NO duplicates):
```
Vertex 0: 1 edge → [1]
Vertex 1: 1 edge → [2]
Vertex 2: 1 edge → [3]
Vertex 3: 3 edges → [2, 4, 5]
Vertex 4: 2 edges → [6, 7]
...
Total: 11 edges
```

**Direct Trace** (HAS duplicates):
```
Vertex 0: 2 edges → [1, 1]  ← DUPLICATED!
Vertex 1: 2 edges → [2, 2]  ← DUPLICATED!
Vertex 2: 2 edges → [3, 3]  ← DUPLICATED!
Vertex 3: 5 edges → [2, 4, 5, 4, 5]  ← 4 and 5 duplicated!
Vertex 4: 4 edges → [6, 7, 6, 7]  ← 6 and 7 duplicated!
```

**Enhanced Subgraphs** (NO duplicates):
```
SCC 2, subgraph vertex 2: 2 edges → [4, 5]  ✓ Correct
```

**SCC Traces** (HAS duplicates):
```
SCC 2, trace[2]: 4 edges → [4, 5, 4, 5]  ✗ Duplicated during elimination!
```

### Conclusion

The duplicate edges are created by `record_elimination_trace()` during the elimination process (Algorithm 3), NOT by:
- Graph construction ✓
- Subgraph building ✓
- Vertex categorization ✓
- Ordered vertices list ✓
- Stitching logic ✓

This is a **separate bug** that affects ALL trace recording, not just hierarchical caching.

---

## Test Results

### Subgraph Structure ✅

```
First SCC:  2 vertices, ordered=[0, 1]
SCC 2:      6 vertices, ordered=[None, 3, 4, 5, 6, 7] (was 7, deduplicated!)
SCC 3:      6 vertices, ordered=[None, ...]
SCC 4:      4 vertices, ordered=[None, ...]
SCC 5:      4 vertices, ordered=[None, ...]
```

All correct! ✅

### Deduplication ✅

Before:
```
SCC 2: ordered_vertices = [None, 3, 4, 5, 4, 6, 7]  # vertex 4 twice
```

After:
```
SCC 2: ordered_vertices = [None, 3, 4, 5, 6, 7]  # NO duplicates
```

Verified with:
```python
counts = Counter([v for v in ordered if v is not None])
assert all(c == 1 for c in counts.values())  # ✓ PASSES
```

### Edge Counts

- Original graph: 11 edges ✓
- Direct trace: 20 edges (duplicates from record_elimination_trace bug)
- Hierarchical trace: 20 edges (same - correctly reproduces direct behavior)

---

## Files Modified

### `/Users/kmt/phasic/src/phasic/hierarchical_trace_cache.py`

**New Functions:**
- Lines 798-910: `_build_first_scc_subgraph()` - Complete implementation
- Lines 612-781: `_build_scc_subgraph()` (renamed from `_build_enhanced_scc_subgraph`)

**Deduplication Logic:**
- Lines 695-715: Category-priority deduplication with verification

**Updated Callers:**
- Lines 225-235: `collect_missing_traces_batch()` - First SCC detection
- Lines 1171-1180: `compute_scc_traces()` - First SCC detection
- Lines 1433-1437: Stitching - None handling

**Debug Logging:**
- Lines 1378-1391: Added edge addition logging

### `/Users/kmt/phasic/src/phasic/trace_elimination.py`

**Bug Fix:**
- Line 469: Moved `MAX_PARAM_TEST` definition to fix UnboundLocalError

### `/Users/kmt/phasic/inspect_hierarchical_debug.py`

**Updated:**
- Lines 96-112: Use new builder functions

---

## What Works ✅

1. ✅ First SCC builder creates correct subgraph (starting vertex reused)
2. ✅ Non-first SCC builder creates correct subgraphs (None at trace[0])
3. ✅ Deduplication eliminates vertices appearing in multiple categories
4. ✅ No duplicates in ordered_vertices lists
5. ✅ No duplicates in enhanced subgraph edges
6. ✅ Stitching logic correctly skips None entries
7. ✅ Cache clearing works correctly
8. ✅ All callers detect first vs non-first SCC

---

## Remaining Work (Separate Issue)

### Bug in record_elimination_trace()

The elimination algorithm is creating duplicate edges. This affects:
- Direct traces
- All SCC traces
- Hierarchical traces (which just stitch the buggy SCC traces)

**Not a hierarchical caching bug** - this is a core trace recording issue that needs separate investigation.

**Recommended**: File as separate bug report and investigate the elimination algorithm (Algorithm 3) to find where edges are being doubled.

---

## Conclusion

The separate builders implementation is **COMPLETE and CORRECT** ✅

The distinction between first and non-first SCCs is now crystal clear in the code. The deduplication of `ordered_vertices` is working perfectly. All subgraphs are built correctly.

The remaining duplicate edge issue is a **separate pre-existing bug** in the trace elimination algorithm itself.

---

*Implementation completed 2025-11-11. Ready for code review and merge.*
