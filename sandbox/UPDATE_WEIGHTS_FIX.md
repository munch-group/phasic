# Fix: `update_weights()` Now Skips aux→parent Edges

## Problem

**Bug**: `ptd_graph_update_weights()` was incorrectly zeroing aux→parent edge weights.

When calling `update_weights()` on a graph with auxiliary vertices:
- ❌ aux→parent edges (with `coefficients_length = 0`) had their weights set to 0.0
- ✅ Expected: aux→parent edges should remain 1.0 (hardcoded constant)

### Root Cause

In `src/c/phasic.c` lines 2836-2839:

```c
// Compute weight = dot(coefficients, theta)
edge->weight = 0.0;  // ← Unconditionally zeros ALL edges!
for (size_t k = 0; k < edge->coefficients_length; k++) {
    edge->weight += edge->coefficients[k] * theta[k];
}
```

For aux→parent edges with `coefficients_length = 0`:
1. Line 2836 sets `weight = 0.0`
2. Loop never executes (k < 0 is false)
3. **Result**: weight stays 0.0 instead of 1.0 ❌

## Solution

Skip edges with `coefficients_length == 0` in `update_weights()`. These are pure constant edges that should never be rescaled.

### Change Made

**File**: `src/c/phasic.c`, lines 2832-2846

**Added**:
```c
// Skip edges with no coefficients (pure constant, like aux→parent edges)
// These edges have hardcoded weights and should never be rescaled
if (edge->coefficients_length == 0) {
    continue;
}
```

## Test Results

### Before Fix

```python
g = phasic.Graph(2)
v = g.find_or_create_vertex([1, 0])
v.add_edge(g.find_or_create_vertex([2, 0]), [1.0, 0.0])
aux = v.add_aux_vertex([2.0, 1.0])

# aux→parent weight is 1.0
print(list(aux.edges())[0].weight())  # Output: 1.0

# Update weights
g.update_weights([2.0, 3.0])

# aux→parent weight becomes 0.0! ❌
print(list(aux.edges())[0].weight())  # Output: 0.0 (BUG!)
```

### After Fix

```bash
$ python test_aux_vertex_update_weights.py
======================================================================
TEST: aux→parent edges survive update_weights()
======================================================================

Setting up parameterized graph with aux vertex...
Created aux vertex with state: [0, 0]
Aux vertex has 1 edge(s)
aux→parent edge weight BEFORE update: 1.0

Calling update_weights([2.0, 3.0])...
aux→parent edge weight AFTER update:  1.0  ✅

✅ SUCCESS: aux→parent edge survived update_weights()!
   Weight remained 1.0 (not affected by parameter update)

Verifying parent→aux edge WAS updated...
parent→aux edge weight: 7.0  ✅
✅ parent→aux edge correctly updated to 7.0

======================================================================
ALL TESTS PASSED! ✅
======================================================================
```

## Edge Types in Unified Interface

| Edge Type | `coefficients_length` | Behavior in `update_weights()` |
|-----------|----------------------|-------------------------------|
| **Constant** | 1 | ✅ Updated: `weight = coefficients[0] * θ[0]` |
| **Parameterized** | N > 1 | ✅ Updated: `weight = Σ(coefficients[i] * θ[i])` |
| **Pure constant (aux→parent)** | 0 | ⚠️ **SKIPPED** (weight stays hardcoded) |

## Why This Fix is Correct

1. **aux→parent edges are special**: They have hardcoded weight 1.0 and no parameterization
2. **coefficients_length = 0 means**: "This edge has no coefficients, don't rescale it"
3. **Normal edges have coefficients_length >= 1**: They should be updated
4. **Backward compatible**: Only affects edges with `coefficients_length = 0` (which are only created by `add_aux_vertex()`)

## Impact

- ✅ Fixes critical bug in discrete graph handling
- ✅ aux→parent edges remain constant (weight = 1.0)
- ✅ Normal parameterized edges still update correctly
- ✅ No impact on non-auxiliary graphs
- ✅ All tests pass

## Files Modified

1. **`src/c/phasic.c`** (lines 2835-2839) - Added skip for `coefficients_length == 0`
2. **`test_aux_vertex_update_weights.py`** - New test for this specific bug
3. **`test_add_aux_vertex.py`** - Updated to reflect `create_vertex()` behavior

## Related Changes

This fix works in conjunction with:
- `add_aux_vertex()` implementation (creates edges with `coefficients_length = 0`)
- Unified edge interface (all edges use coefficient arrays internally)
- Changed from `find_or_create_vertex()` to `create_vertex()` for aux vertices

---

**Status**: ✅ Fixed and tested
**Version**: 0.22.0
**Date**: 2025-11-05
