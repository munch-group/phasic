# min_size Parameter Not Used for Individual SCCs

**Date**: 2025-11-11
**Status**: ⚠️ **DESIGN ISSUE IDENTIFIED**

---

## Problem

The `min_size` parameter in `get_trace_hierarchical()` is **only used to decide whether to subdivide at the top level**, but not to filter individual SCCs. This leads to processing many tiny SCCs even when `min_size` is large.

### Example

For a graph with 110 vertices and `min_size=30`:

```python
graph.compute_trace(hierarchical=True, min_size=30)
```

**Expected behavior**: Only subdivide if SCCs are reasonably large (≥30 vertices)

**Actual behavior**:
1. Check: `110 < 30`? No → proceed with subdivision
2. Decompose into 27 SCCs (many with < 30 vertices)
3. Process **all 27 SCCs separately**, even if each has only 2-4 vertices

**Result**: 27 separate trace recordings, even though the user requested `min_size=30`!

---

## Code Analysis

### Current Implementation

**File**: `src/phasic/hierarchical_trace_cache.py`

**Line 177-189**: Top-level size check (works correctly)
```python
if n_vertices < min_size:
    logger.debug("Graph too small for subdivision (%d < %d), recording directly",
                n_vertices, min_size)
    # Record full graph directly
    work_units[g_hash] = graph
    return work_units, all_scc_hashes, None
```

**Line 203-251**: Process ALL SCCs (no size filtering)
```python
for i, scc in enumerate(sccs):
    scc_hash = scc.hash()
    scc_size = scc.size()
    all_scc_hashes.append(scc_hash)

    # NO SIZE CHECK HERE - processes every SCC regardless of size!

    # Check cache
    cached = _load_trace_from_cache(scc_hash)
    if cached is not None:
        continue

    # Build subgraph and add to work_units
    enhanced_subgraph, metadata = _build_scc_subgraph(...)
    work_units[scc_hash] = enhanced_subgraph
```

### The Missing Logic

There should be a size check for individual SCCs:

```python
for i, scc in enumerate(sccs):
    scc_hash = scc.hash()
    scc_size = scc.size()

    # MISSING: Check if SCC is large enough to warrant caching
    if scc_size < min_size:
        logger.debug("SCC %d too small (%d < %d), will record with neighbors",
                    i, scc_size, min_size)
        # Should group small SCCs together or record directly
        continue

    # Rest of processing...
```

---

## Impact

### Performance Impact

For graphs with many small SCCs:
- **Extra overhead**: Each SCC requires subgraph construction, trace recording, cache operations
- **Fragmented caching**: Many tiny traces in cache instead of fewer larger ones
- **Slower execution**: More overhead from processing boundaries between SCCs

### Example: 110-vertex graph with 27 SCCs

Assuming average SCC size of 4 vertices:
- **Current**: 27 × (subgraph build + trace record + cache save) operations
- **Expected**: 1-3 larger traces (if small SCCs grouped together)
- **Overhead**: ~10-20x more operations than necessary

---

## Proposed Solutions

### Option 1: Filter Small SCCs (Group Together)

**Idea**: SCCs smaller than `min_size` should be grouped with their neighbors and recorded together.

**Implementation**:
```python
# Group consecutive small SCCs
scc_groups = []
current_group = []
current_size = 0

for scc in sccs:
    scc_size = scc.size()

    if scc_size >= min_size:
        # Large SCC - process separately
        if current_group:
            scc_groups.append(current_group)
            current_group = []
            current_size = 0
        scc_groups.append([scc])
    else:
        # Small SCC - add to current group
        current_group.append(scc)
        current_size += scc_size

        # If group is now large enough, finalize it
        if current_size >= min_size:
            scc_groups.append(current_group)
            current_group = []
            current_size = 0

# Add remaining group
if current_group:
    scc_groups.append(current_group)

# Process groups instead of individual SCCs
for group in scc_groups:
    if len(group) == 1:
        # Single large SCC - process as before
        process_scc(group[0])
    else:
        # Multiple small SCCs - build combined subgraph
        process_scc_group(group)
```

**Benefits**:
- Respects user's `min_size` intent
- Reduces number of trace operations
- Still maintains some caching benefit (large SCCs cached separately)

**Challenges**:
- More complex grouping logic
- Need to handle stitching for groups
- Cache keys need to be stable across groups

### Option 2: Recursive Subdivision with Size Check

**Idea**: Add size check before processing each SCC, and only subdivide if large enough.

**Implementation**:
```python
for i, scc in enumerate(sccs):
    scc_size = scc.size()

    # Skip small SCCs (will be included in larger trace)
    if scc_size < min_size:
        logger.debug("SCC %d too small (%d < %d), skipping separate processing",
                    i, scc_size, min_size)
        small_sccs.append(scc)
        continue

    # Large SCC - process separately
    # ... existing logic ...
```

**Benefits**:
- Simple to implement
- Clear semantics: only SCCs ≥ min_size are cached separately

**Challenges**:
- Need to handle small SCCs that aren't processed separately
- May need to record full graph if no large SCCs

### Option 3: Document Current Behavior

**Idea**: Keep current behavior but document it clearly.

**Documentation**:
```python
min_size : int, default=50
    Minimum **total graph size** to enable SCC subdivision.
    If graph has fewer than min_size vertices, it's recorded directly.

    NOTE: This does NOT filter individual SCCs. All SCCs are processed
    separately regardless of their individual sizes. To avoid processing
    many tiny SCCs, set min_size larger than your graph size.
```

**Benefits**:
- No code changes needed
- Users can work around by adjusting min_size

**Challenges**:
- Doesn't solve the performance issue
- Unintuitive behavior

---

## Recommendation

**Implement Option 1 (Filter Small SCCs)**

This provides the best balance of:
- Performance improvement (fewer trace operations)
- User expectations (min_size actually filters SCCs)
- Backward compatibility (only affects edge cases)

### Implementation Steps

1. Add SCC grouping logic in `collect_missing_traces_batch()`
2. Modify subgraph builder to handle SCC groups
3. Update stitching logic to handle grouped SCCs
4. Add tests for various SCC size distributions
5. Update documentation

---

## Workaround (Current Users)

Until fixed, users can work around by:

1. **Set min_size larger than graph size** to disable subdivision:
```python
n_vertices = graph.vertices_length()
graph.compute_trace(hierarchical=True, min_size=n_vertices + 1)
```

2. **Disable hierarchical caching** for graphs with many small SCCs:
```python
graph.compute_trace(hierarchical=False)
```

3. **Use cache clearing** to avoid accumulating many tiny traces:
```python
from phasic.hierarchical_trace_cache import clear_all_caches
clear_all_caches()
graph.compute_trace(hierarchical=True, min_size=30)
```

---

## Test Case

To verify the fix, create a test with:
- Graph with 110 vertices
- Decomposed into 27 SCCs (avg 4 vertices each)
- min_size=30

**Expected**:
- 1-3 trace operations (grouped SCCs)
- Log shows "grouped 25 small SCCs into 2 groups"

**Current**:
- 27 trace operations (one per SCC)
- Log shows "processing 27 SCCs"

---

*Issue identified 2025-11-11*
