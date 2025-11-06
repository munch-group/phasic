# Starting Vertex Edge Scaling Fix

**Date**: November 3, 2025
**Issue**: Starting vertex edges were being incorrectly scaled by `update_weights()`
**Status**: ✅ **FIXED**

## Problem Description

After the unified edge interface refactoring, starting vertex edges were being scaled along with other edges when `update_weights(theta)` was called. This broke the initial probability vector (IPV) semantics required for phase-type distributions.

### Example of the Bug

```python
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([0])

v0.add_edge(v1, [5.0])  # Starting vertex edge
v1.add_edge(v2, [3.0])  # Other edge

g.update_weights([2.0])

# BEFORE FIX:
#   v0→v1: 10.0  ❌ (incorrectly scaled from 5.0)
#   v1→v2: 6.0   ✓

# AFTER FIX:
#   v0→v1: 5.0   ✓ (correctly stays at 5.0)
#   v1→v2: 6.0   ✓
```

## Solution

Modified `ptd_graph_update_weights()` to skip edges from the starting vertex (src/c/phasic.c, lines 2573-2575).

## Test Results

### ✅ test_starting_vertex_fix.py
- Test 1-5: All pass
- Starting edges remain unchanged after `update_weights()` ✓
- Non-starting edges scale correctly ✓

### ✅ test_unified_edge_correctness.py
- All 9 tests pass ✓
- Test 8 specifically verifies starting edges unchanged ✓

## Files Modified

1. `/Users/kmt/phasic/src/c/phasic.c` (lines 2570-2575)
2. `/Users/kmt/phasic/tests/test_unified_edge_correctness.py` (Tests 2, 8, 9)
3. `/Users/kmt/phasic/test_starting_vertex_fix.py` (new test file)

**✅ FIX VERIFIED AND PRODUCTION-READY**
