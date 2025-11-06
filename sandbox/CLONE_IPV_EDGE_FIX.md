# Fix: Clone Support for @phasic.callback Decorator Pattern

## Problem

The `graph.clone()` method crashed when cloning graphs created with the `@phasic.callback` decorator:

```python
@phasic.callback([2, 0])
def rabbits(state):
    left, right = state
    transitions = []
    if left:
        transitions.append([[left - 1, right + 1], [1, 0, 0, 0]])
        transitions.append([[0, right], [0, 1, 0, 0]])
    if right:
        transitions.append([[left + 1, right - 1], [1, 0, 0, 0]])
        transitions.append([[left, 0], [0, 0, 1, 0]])
    return transitions

graph = phasic.Graph(rabbits)
cloned = graph.clone()  # ❌ CRASHED
```

**Error**: `RuntimeError: Failed to clone starting vertex edge: graph expects 4 parameters, got 1. All edges in a graph must have the same coefficient length.`

## Root Cause

The `@phasic.callback` decorator creates graphs with **mixed coefficient lengths**:

1. **IPV (Initial Probability Vector) edges** from starting vertex: empty arrays `[]` or single-element arrays
2. **Regular edges**: coefficient arrays like `[1, 0, 0, 0]` with 4 elements

The unified edge interface enforces uniform coefficient length via `param_length_locked` validation in `ptd_graph_add_edge()`. When `ptd_clone_graph()` tried to clone IPV edges using `ptd_graph_add_edge()`, the validation rejected the mixed lengths.

## Solution

**Bypass validation for IPV edges by cloning them directly** without calling `ptd_graph_add_edge()`.

### Implementation

Modified `/Users/kmt/phasic/src/c/phasic.c` lines 1091-1138 to:

1. **Manually allocate edge structure**:
```c
struct ptd_edge *new_edge = (struct ptd_edge *)malloc(sizeof(*new_edge));
```

2. **Copy edge fields**:
```c
new_edge->to = new_target;
new_edge->weight = old_edge->weight;
new_edge->coefficients_length = old_edge->coefficients_length;
```

3. **Clone coefficient array**:
```c
if (old_edge->coefficients != NULL && old_edge->coefficients_length > 0) {
    new_edge->coefficients = (double *)malloc(old_edge->coefficients_length * sizeof(double));
    memcpy(new_edge->coefficients, old_edge->coefficients,
           old_edge->coefficients_length * sizeof(double));
    new_edge->should_free_coefficients = true;
} else {
    new_edge->coefficients = NULL;
    new_edge->should_free_coefficients = false;
}
```

4. **Add edge to starting vertex's edge list** (bypass validation):
```c
struct ptd_edge **new_edges = (struct ptd_edge **)realloc(
    new_start->edges,
    (new_start->edges_length + 1) * sizeof(struct ptd_edge *)
);
new_start->edges = new_edges;
new_start->edges[new_start->edges_length] = new_edge;
new_start->edges_length++;
```

This approach:
- ✅ Bypasses `ptd_graph_add_edge()` validation for IPV edges
- ✅ Preserves coefficient arrays exactly as they are
- ✅ Maintains proper memory management (allocates new arrays, sets `should_free_coefficients`)
- ✅ Keeps regular edges using `ptd_graph_add_edge()` for consistency

## Test Results

### Before Fix

```bash
$ python test_callback_decorator.py
RuntimeError: Failed to clone starting vertex edge: graph expects 4 parameters, got 1.
All edges in a graph must have the same coefficient length.
```

### After Fix

```bash
$ python test_callback_decorator.py
Testing @phasic.callback decorator pattern...
Creating graph from callback...
✅ Graph created successfully: 7 vertices

Testing clone...
✅ Clone successful: 7 vertices

✅ SUCCESS: @phasic.callback pattern works!
```

### Comprehensive Tests

All existing tests still pass:

```bash
$ python test_clone_comprehensive.py
======================================================================
COMPREHENSIVE GRAPH.CLONE() TESTS
======================================================================

Test 1: Basic clone with starting vertex
  ✅ Test 1 PASSED: Basic clone is independent

Test 2: Clone with multiple vertices and edges
  ✅ Test 2 PASSED: Multi-vertex clone preserves structure

Test 3: Clone with multiple [0] state vertices
  ✅ Test 3 PASSED: Multiple [0] vertices handled correctly

Test 4: Python copy() method
  ✅ Test 4 PASSED: Python copy() method works

======================================================================
ALL TESTS PASSED! ✅
graph.clone() and graph.copy() create independent deep copies
======================================================================
```

## Design Rationale

### Why bypass validation for IPV edges?

1. **IPV edges are special**: They represent initial probabilities, not transitions, and may legitimately have different coefficient structures
2. **Existing graphs rely on this**: The `@phasic.callback` decorator creates graphs with this pattern
3. **Validation is for consistency**: The `param_length_locked` validation ensures all edges in a graph use the same parameter space, but IPV edges are semantically different
4. **Clone should preserve structure**: A clone should exactly replicate the original graph structure, including mixed coefficient lengths if present

### Why only for starting vertex edges?

Starting vertex edges are the only place where IPV edges appear (by definition of Initial Probability Vector). All other edges are regular transition edges that go through validation.

## Files Modified

- `/Users/kmt/phasic/src/c/phasic.c` (lines 1091-1138): Direct IPV edge cloning
- `/Users/kmt/phasic/test_clone_comprehensive.py`: Updated API usage (`Graph(1)` vs `Graph(state_length=1)`)

## Backward Compatibility

✅ **Fully backward compatible**
- All existing tests pass
- Manual graph construction: works
- Callback decorator pattern: works
- Mixed coefficient lengths: preserved correctly

## Summary

The fix enables `graph.clone()` to work with graphs created using the `@phasic.callback` decorator by bypassing coefficient length validation for IPV edges. This preserves the exact structure of the original graph while maintaining independence between original and clone.

**Status**: ✅ Complete and tested
**Commit**: Ready for git commit
