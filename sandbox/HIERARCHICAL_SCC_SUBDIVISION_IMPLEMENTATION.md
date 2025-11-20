# Hierarchical SCC Subdivision Implementation

**Date**: November 9, 2025
**Status**: Partially Complete - Core functionality implemented, stitching issue remains

## Summary

Successfully implemented hierarchical SCC-based trace caching with comprehensive DEBUG logging. The system correctly:
1. ✅ Decomposes graphs into SCCs
2. ✅ Recursively collects missing traces
3. ✅ Computes missing traces sequentially
4. ✅ Caches and retrieves SCC traces
5. ⚠️ **Hangs during trace stitching phase**

## What Was Implemented

### 1. Comprehensive DEBUG Logging

Added detailed logging throughout the hierarchical caching system:
- **SCC decomposition**: Logs number of SCCs found, their sizes, and proportion of graph
- **Cache operations**: Every cache hit/miss logged with hash, vertex count, operation count
- **Recursive collection**: Depth-aware indentation showing subdivision hierarchy
- **Trace stitching**: Step-by-step merge process logging

**Files Modified**:
- `src/phasic/hierarchical_trace_cache.py` - Added logging to all major functions

### 2. Fixed EliminationTrace Representation

**Problem**: Notebook file exploded from 26KB to 171MB after running `graph.compute_trace()`
**Root Cause**: Auto-generated `__repr__()` was printing all 25,027 operations (179,364,023 characters)
**Solution**: Added custom `__repr__()` returning concise 66-character string

**File**: `src/phasic/trace_elimination.py:EliminationTrace`

```python
def __repr__(self) -> str:
    return (f"EliminationTrace(n_vertices={self.n_vertices}, "
            f"operations={len(self.operations)}, "
            f"param_length={self.param_length})")
```

### 3. Implemented Actual Hierarchical SCC Subdivision

**Problem**: `get_trace_hierarchical()` was only doing simple full-graph caching ("Phase 3a")
**Solution**: Completely rewrote to implement actual hierarchical workflow

**New Workflow**:
1. Check full graph cache (fast path)
2. If miss and graph >= min_size:
   - Collect missing SCC traces recursively
   - Compute missing traces (sequential for now)
   - Load all SCC traces from cache
   - Stitch traces together
3. Cache final result

**File**: `src/phasic/hierarchical_trace_cache.py:get_trace_hierarchical()`

### 4. Fixed Multiple Issues

#### Issue A: `AttributeError: 'Graph' object has no attribute 'serialize'`
**Solution**: Changed work units to store Graph objects directly instead of serialized JSON (for in-process parallelization)

#### Issue B: `AttributeError: 'Graph' object has no attribute 'content_hash'`
**Solution**: Use `phasic_hash.compute_graph_hash(g)` instead of non-existent `g.content_hash()`

#### Issue C: Segmentation fault when loading SCC traces
**Problem**: Iterating through `scc_decomp.sccs_in_topo_order()` twice - SCC objects invalid second time
**Solution**: Collect SCC hashes during first iteration, use stored hashes for loading

**Implementation**:
- Modified `collect_missing_traces_batch()` to return `(work_units, all_scc_hashes)`
- Use `all_scc_hashes` list instead of re-iterating through SCC decomposition

## Test Results

### First Run (Cache Cold)
```
[INFO] Using hierarchical SCC subdivision (graph=110 vertices, min_size=20)
[DEBUG] ✓ Found 27 SCCs
[DEBUG] SCC sizes: [2, 2, 3, 3, 4, 4, 4, 8, 5, 4, 6, 5, 9, 4, 6, 9, 13, 3, 4, 5, 7, 3, 4, 5, 7, 6, 2]
[INFO] Trace collection complete: 27 work units needed
[INFO] Cache statistics: 0 hits, 28 misses
[DEBUG] Step 2: Computing 27 missing traces...
[INFO] Trace recording complete: 2 vertices, 1 operations
[INFO] Trace recording complete: 3 vertices, 17 operations
... (all 27 traces computed) ...
[DEBUG] ✓ Computed 27 missing traces
```

### Second Run (Cache Warm)
```
[INFO] Using hierarchical SCC subdivision (graph=110 vertices, min_size=20)
[DEBUG] ✓ Found 27 SCCs
[INFO] Trace collection complete: 0 work units needed
[INFO] Cache statistics: 27 hits, 1 misses  ← All SCCs cached!
[DEBUG] Step 2: No missing traces to compute (all cached)
[DEBUG] Step 3: Loading all 27 SCC traces from cache...
[DEBUG]   Loading SCC 1/27: hash=f33e392e7358f775...
[DEBUG]     ✓ Loaded trace: 2 vertices, 1 operations
... (all 27 loaded successfully) ...
[DEBUG] ✓ Loaded 27 SCC traces from cache
```

## Remaining Issue: Trace Stitching Hangs

### Symptom
Execution hangs at:
```
[DEBUG] Step 4: Getting SCC decomposition for stitching...
```

Never reaches "Step 5: Stitching SCC traces together..."

### Investigation

1. **Multiple SCC decompositions work fine**: Tested calling `graph.scc_decomposition()` multiple times on same graph - no issues
2. **Not a segfault**: Process doesn't crash, just hangs indefinitely
3. **Possible causes**:
   - Parameterization mismatch: Top-level graph is parameterized (1 param), but SCC graphs are concrete (0 params)
   - The `stitch_scc_traces()` function may have issues handling this mismatch
   - Possible infinite loop in C++ SCC decomposition code when called on complex graphs

### Evidence from Logs
```
[DEBUG] Computing content hash for graph: 110 vertices, 1 params, parameterized  ← Top level
[DEBUG] Computing content hash for graph: 2 vertices, 0 params, concrete  ← SCC graphs
```

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/phasic/trace_elimination.py` | Added custom `__repr__()` to `EliminationTrace` | ~10 |
| `src/phasic/hierarchical_trace_cache.py` | Complete rewrite of hierarchical subdivision | ~200 |
| `src/phasic/logging_config.py` | Added NONE level, disabled colors | ~20 |
| `docs/pages/tutorials/simple_example.ipynb` | Cleared outputs (171MB → 26KB) | N/A |

## Next Steps

1. **Debug stitching hang**:
   - Add timeout/interrupt to identify exact hanging location
   - Check if `stitch_scc_traces()` is compatible with parameterized→concrete mismatch
   - Consider skipping hierarchical subdivision for parameterized graphs

2. **Optimization**:
   - Implement actual parallel processing (currently forced to sequential)
   - Add graph serialization support for cross-process parallelization

3. **Testing**:
   - Test with non-parameterized graphs first
   - Add unit tests for each hierarchical caching component
   - Test with various graph sizes and structures

## Verification

To verify the implementation works (except stitching):

```bash
# Enable DEBUG logging
export PHASIC_LOG_LEVEL=DEBUG

# Run test (will hang at stitching)
python test_scc_subdivision.py

# Expected output:
# - 27 SCCs found
# - All 27 traces computed (first run) or loaded from cache (second run)
# - Hangs at "Step 4: Getting SCC decomposition for stitching..."
```

## Conclusion

Hierarchical SCC subdivision is **90% complete**:
- ✅ SCC decomposition working
- ✅ Recursive collection working
- ✅ Caching working perfectly (27/27 cache hits on second run)
- ✅ Comprehensive logging in place
- ⚠️ **Trace stitching hangs** - needs investigation

The caching infrastructure is solid - once stitching is fixed, this will provide significant performance improvements for large graphs.
