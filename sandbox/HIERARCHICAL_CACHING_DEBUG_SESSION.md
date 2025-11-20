# Hierarchical SCC Caching - Debug Session Summary

## Problem Statement

Hierarchical trace caching produces **12,572 operations** instead of the correct **25,027 operations** (2x too few). The stitched trace has correct `param_length=1` but wrong operation count.

## Root Causes Found

### ✅ Bug #1: Auto-detection of param_length is BROKEN
**Location**: `trace_elimination.py:record_elimination_trace()`

**Problem**: When `param_length=None` (auto-detect), the function returns `param_length=0` even for parameterized graphs.

**Evidence**:
```
Hierarchical=False, cache cleared
✓ Trace computed:
    operations: 25027
    param_length: 0  <--- WRONG! Should be 1
```

**Fix Applied**: Changed hierarchical_trace_cache.py to use `graph.param_length()` explicitly instead of auto-detection:
```python
# Line 431 and 1080
trace = record_elimination_trace(graph, param_length=graph.param_length())
```

### ✅ Bug #2: Using isolated SCC subgraphs instead of enhanced
**Status**: Already fixed in previous session (api/cpp/scc_graph.cpp)

The C++ code correctly copies parameterized edges from the original graph to SCC subgraphs.

## Current Status After Fixes

### SCC Traces (Individual Components)
- 27 SCCs processed
- **Total SCC operations: 376** (sum of all individual SCC traces)
- Some SCCs have `param_length=0` (absorbing states with 0 edges)
- Most SCCs have `param_length=1` (correctly parameterized)

### Stitched Result
- **Stitched operations: 12,572**
- **Expected operations: 25,027**
- **Missing operations: 12,651** (12,572 from stitching - 376 from SCCs = 12,196 extra!)

### Key Insight
The stitching logic is **adding 12,196 spurious operations** instead of correctly merging the 376 SCC operations into 25,027 operations.

## Evidence

```bash
# C++ logging shows correct parameterization:
[INFO] phasic.c: as_graph: Copied 2 edges (2 parameterized, 0 concrete)
[INFO] phasic.c:   SCC subgraph: parameterized=1, param_length=1

# Sum of SCC operations after param_length fix:
$ grep "SCC [0-9]*/27:" | awk '{sum += $5} END {print sum}'
376

# But stitched result:
Hierarchical: 12572 ops, param_length=1
```

## The Bug is in Trace Stitching

**Location**: `hierarchical_trace_cache.py:stitch_scc_traces()`

The stitching logic at lines 700-900 is incorrectly combining SCC traces. Instead of merging 376 operations from SCCs into a full 25,027-operation trace, it's:
1. Only using operations from the enhanced subgraph traces (not the full graph)
2. OR incorrectly mapping vertex indices between SCC traces and the merged trace
3. OR duplicating operations during the merge

## Files Modified This Session

1. **api/cpp/scc_graph.cpp** (lines 1-210)
   - Added debug logging for parameterized edge copying
   - Confirmed edges ARE being copied correctly as parameterized

2. **src/phasic/hierarchical_trace_cache.py** (lines 430-432, 1078-1081)
   - Fixed: Use `graph.param_length()` instead of `None` (auto-detect)
   - Result: SCC traces now have correct param_length

## Test Results

```bash
$ rm -rf ~/.phasic_cache/traces && pixi run python test_simple_correctness.py

Graph: 110 vertices
Direct: 25027 ops, param_length=1
Hierarchical: 12572 ops, param_length=1
❌ FAILURE: 1.99x difference
```

## Next Steps (CRITICAL)

### 1. Debug the Stitching Logic
**File**: `src/phasic/hierarchical_trace_cache.py`, function `stitch_scc_traces()` (lines ~700-900)

**Key questions to answer:**
- Are we correctly mapping SCC trace vertices to merged trace vertices?
- Are we using the enhanced subgraph traces (which include connecting vertices) or the base SCC traces?
- Are we correctly handling boundary edges between SCCs?
- Why does stitching produce 12,572 operations when SCCs only have 376?

**Debug approach:**
```python
# Add logging in stitch_scc_traces():
logger.info(f"Stitching {len(scc_trace_dict)} SCC traces")
for scc_hash, trace in scc_trace_dict.items():
    logger.info(f"  SCC trace: {trace.n_vertices} vertices, {len(trace.operations)} operations")

# After stitching:
logger.info(f"Merged trace: {merged.n_vertices} vertices, {len(merged.operations)} operations")
```

### 2. Understand the Enhanced Subgraph Strategy

The current approach:
- Extract each SCC as an "enhanced subgraph" (internal vertices + downstream connecting vertices)
- Record elimination trace for the enhanced subgraph
- Stitch traces together

**Question**: Should the enhanced subgraph traces already contain ~25K operations total? Or should stitching add more operations by processing boundary edges?

**Expected behavior**:
- Enhanced subgraphs for 27 SCCs should collectively cover the entire elimination
- Sum of enhanced trace operations should ≈ 25,027 (not 376)
- Stitching should just merge them without adding operations

### 3. Consider Alternative Approach

If stitching is fundamentally flawed, consider:
- Recording traces on isolated SCCs (not enhanced)
- Manually adding boundary edge operations during stitching
- This might be simpler than the current "enhanced subgraph" approach

## Debug Commands

```bash
# Clear cache and run test
rm -rf ~/.phasic_cache/traces && pixi run python test_simple_correctness.py

# Check SCC operation counts
pixi run python test_simple_correctness.py 2>&1 | grep "SCC [0-9]*/27:"

# Sum SCC operations
pixi run python test_simple_correctness.py 2>&1 | grep "SCC [0-9]*/27:" | awk '{sum += $5} END {print "Sum:", sum}'

# Check C++ subgraph creation
pixi run python test_simple_correctness.py 2>&1 | grep "as_graph:"
```

## Key Files to Review

1. **src/phasic/hierarchical_trace_cache.py**
   - Lines 700-900: `stitch_scc_traces()` - THE BUG IS HERE
   - Lines 436-561: `_build_enhanced_scc_subgraph()` - Working correctly
   - Lines 167-240: `collect_missing_traces_batch()` - Working correctly

2. **src/phasic/trace_elimination.py**
   - Lines 356-500: `record_elimination_trace()` - Has broken auto-detection of param_length

3. **api/cpp/scc_graph.cpp**
   - Lines 133-210: `SCCVertex::as_graph()` - Working correctly, copies parameterized edges

## The Core Mystery

**Why does stitching 27 SCCs with 376 total operations produce 12,572 operations?**

This 33x multiplication suggests:
- Operations are being duplicated/replicated during stitching
- OR boundary edges are being processed as if they were full SCCs
- OR the stitching logic is fundamentally misunderstanding the enhanced subgraph approach

The answer is in `stitch_scc_traces()` around lines 700-900 of hierarchical_trace_cache.py.
