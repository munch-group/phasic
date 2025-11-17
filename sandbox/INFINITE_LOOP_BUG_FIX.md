# Infinite Loop Bug Fix in Hierarchical Caching

**Date**: 2025-11-11
**Status**: ✅ **FIXED**

---

## Problem

When calling `graph.compute_trace(hierarchical=True, min_size=30)` on the two_locus_arg model (110 vertices, 27 SCCs with sizes < 30), the code entered an **infinite loop**, causing:
- Jupyter kernel crashes
- Repeated log messages
- No progress/completion

---

## Root Cause

The infinite loop occurred due to a logic error in recursive subdivision detection:

**File**: `src/phasic/hierarchical_trace_cache.py` line 456-476

### The Bug

1. User calls `get_trace_hierarchical(graph, min_size=30)` with 110-vertex graph
2. `collect_missing_traces_batch()` checks SCC sizes:
   - All 27 SCCs are < 30 vertices
   - Decision: "All SCCs below min_size, record full graph directly"
   - Returns the full 110-vertex graph as a work unit
3. `compute_missing_traces_parallel()` receives the 110-vertex graph
4. Checks if graph `can_subdivide`:
   ```python
   if graph.vertices_length() >= min_size:  # 110 >= 30 → True
       # Check SCCs
       non_trivial_sccs = [s for s in sccs if s.size() > 1]  # 27 SCCs (all > 1)
       can_subdivide = len(non_trivial_sccs) > 1  # 27 > 1 → True
   ```
5. Since `can_subdivide == True`, **recursively calls** `get_trace_hierarchical()` again
6. → **Back to step 1**: Infinite loop!

### Why This Happened

The recursion check only verified:
- Multiple non-trivial SCCs exist
- Graph size ≥ min_size

But it **didn't check** if any individual SCC was ≥ min_size, which would prevent infinite recursion when all SCCs are small.

---

## The Fix

**File**: `src/phasic/hierarchical_trace_cache.py` lines 465-471

### Before (Buggy)

```python
# Can subdivide if there are multiple non-trivial SCCs
can_subdivide = len(non_trivial_sccs) > 1
```

### After (Fixed)

```python
# Can subdivide ONLY if:
# 1. There are multiple non-trivial SCCs AND
# 2. At least one SCC is >= min_size (otherwise we'll just loop forever)
large_sccs = [s for s in sccs_test if s.size() >= min_size]
can_subdivide = len(non_trivial_sccs) > 1 and len(large_sccs) > 0
```

### Key Insight

**Recursion should only happen if subdivision would actually help** (i.e., produce at least one large SCC that can be cached separately). If all SCCs are small, recursing will just repeat the same "all small" decision forever.

---

## Test Results

### Test 1: two_locus_arg (Reproduces Original Bug)

**Setup**:
- 110 vertices
- 27 SCCs (sizes 1-12, all < 30)
- `min_size=30`

**Before Fix**: Infinite loop (kernel crash)

**After Fix**:
```
[INFO] SCC grouping: 0 large SCCs (≥30 vertices), 27 small SCCs (<30 vertices, 110 total vertices)
[INFO] All SCCs are below min_size=30, recording full graph directly (no subdivision)
[INFO] Trace recording complete: 110 vertices, 9394 operations
✓ SUCCESS: Trace computed
```

**Time**: ~2 seconds (vs infinite before)

### Test 2: Simple Models (Regression)

All existing correctness tests pass:

```bash
pixi run python test_hierarchical_correctness_simple.py
# ✓✓✓ ALL TESTS PASSED! ✓✓✓
```

- PDF values: 0.00e+00 difference
- Moments: 0.00e+00 difference
- No regressions

### Test 3: min_size Filtering (Behavior)

```bash
pixi run python test_min_size_filtering.py
# All tests pass, no infinite loops
```

---

## Impact

### Fixed Issues

1. ✅ **two_locus_arg with hierarchical=True** now works
2. ✅ **No more kernel crashes** in Jupyter notebooks
3. ✅ **Graphs with all-small SCCs** handled correctly
4. ✅ **No performance regression** for normal cases

### Performance

For two_locus_arg (110 vertices, 27 small SCCs):

**Before**: Infinite loop (unusable)
**After**: 2 seconds (direct recording, as intended)

---

## Related Changes

This fix complements the earlier `min_size` filtering implementation:

1. **First fix** (earlier today): Added SCC grouping by size in `collect_missing_traces_batch()`
   - Result: Graphs with all small SCCs record directly (no subdivision)

2. **This fix**: Prevent infinite recursion in `compute_missing_traces_parallel()`
   - Result: Recursive subdivision only happens when at least one SCC is large enough

Together, these ensure `min_size` parameter works correctly at ALL levels of the recursion.

---

## Prevention

### Design Principle

**Recursive subdivision should only occur when it provides value**:
- Value = At least one SCC is large enough to cache separately
- If all SCCs < min_size: just record directly, don't recurse

### Code Comments

Added clear comments explaining the recursion guard:

```python
# Can subdivide ONLY if:
# 1. There are multiple non-trivial SCCs AND
# 2. At least one SCC is >= min_size (otherwise we'll just loop forever)
```

### Log Messages

Updated debug log to show why recursion decision was made:

```
# Before: "Recursively subdividing (multiple non-trivial SCCs)"
# After:  "Recursively subdividing (has large SCCs >= min_size)"

# Before: "Recording trace directly (can't subdivide further)"
# After:  "Recording trace directly (no large SCCs or can't subdivide)"
```

---

## Verification Checklist

- [x] two_locus_arg works with hierarchical=True, min_size=30
- [x] No infinite loops in any test case
- [x] All existing correctness tests pass
- [x] Performance is good (2s for 110-vertex graph)
- [x] Log messages are clear and informative

---

## Files Modified

- `src/phasic/hierarchical_trace_cache.py` (lines 465-471, 474, 483)

## Documentation

- `INFINITE_LOOP_BUG_FIX.md` (this file)
- `test_two_locus_crash.py` (reproduction test)

---

## Conclusion

The infinite loop bug in hierarchical caching has been **fixed and verified**. The two_locus_arg model now works correctly with `hierarchical=True, min_size=30`, completing in ~2 seconds instead of looping forever.

**Status**: Production ready! ✅

---

*Bug fixed 2025-11-11*
