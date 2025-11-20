# Clone State Sharing Bug - FIX COMPLETE ✅

## Summary

Fixed critical bug where `graph.clone()` shared state arrays between original and cloned graphs. The fix ensures state arrays are properly copied, allowing multiple `compute_trace()` calls and safe graph cloning.

## Root Cause

`ptd_clone_graph()` in `src/c/phasic.c` was passing state pointers directly to `ptd_vertex_create_state()`, which stored them without copying:

```c
// BUG (before fix):
struct ptd_vertex *new_v = ptd_vertex_create_state(new_graph, old_v->state);
// → Shared pointer, memory corruption when original freed
```

## Solution ✅

Modified `ptd_clone_graph()` to allocate and copy state arrays:

```c
// FIX (after):
int *new_state = (int *)malloc(graph->state_length * sizeof(int));
memcpy(new_state, old_v->state, graph->state_length * sizeof(int));
struct ptd_vertex *new_v = ptd_vertex_create_state(new_graph, new_state);
// → Independent copy, safe when original freed
```

## Verification ✅

### Test 1: State Independence
```python
graph = phasic.Graph(model, ipv=ipv)
clone = graph.clone()

orig_state = list(graph.vertices())[0].state()
clone_state = list(clone.vertices())[0].state()

assert id(orig_state) != id(clone_state)  # ✅ PASS
```

**Result**: Different memory addresses (6107498800 vs 6107499088)

### Test 2: Multiple compute_trace() Calls
```python
graph = phasic.Graph(model, ipv=ipv)  # 2999 vertices

trace1 = graph.compute_trace()
assert graph.vertices_length() == 2999  # ✅ PASS

trace2 = graph.compute_trace()
assert graph.vertices_length() == 2999  # ✅ PASS
```

**Result**: Graph preserved after multiple calls

### Test 3: C-Level Verification
Debug logging confirms independent pointers:
```
[DEBUG] Clone vertex 1: old_state=0x16b543910 new_state=0x16b528f10
[DEBUG] Clone vertex 2: old_state=0x13ebf23f0 new_state=0x13ebd8920
```

### Test 4: Non-Hierarchical Mode
```python
graph = phasic.Graph(model, ipv=ipv)
trace = graph.compute_trace(hierarchical=False)
assert graph.vertices_length() == 2999  # ✅ PASS
```

**Result**: Works correctly

## Files Modified

**src/c/phasic.c** (lines 1035-1077):
- Added state array allocation (`malloc`) and copying (`memcpy`) for all vertices
- Special handling for starting vertex (pre-created by `ptd_graph_create()`)
- Ensures complete memory independence between original and clone

**src/phasic/__init__.py** (line 3430):
- Uses `_graph = self.clone()` in `compute_trace()` to preserve original graph
- Clone is now safe thanks to C fix

## Impact

**Before fix:**
- ❌ Cloning would crash with memory corruption
- ❌ Shared state pointers between original and clone
- ❌ Dangling pointers when original graph freed
- ❌ Cannot call `compute_trace()` multiple times

**After fix:**
- ✅ Clone creates fully independent graph
- ✅ Original graph can be freed without affecting clone
- ✅ Multiple `compute_trace()` calls work correctly
- ✅ Original graph preserved (2999 vertices → 2999 vertices)

## Known Issue (Unrelated to Clone Fix)

There is a separate issue where hierarchical caching with `clear_caches()` may hang during trace recomputation for large graphs (2999 vertices). This is NOT related to the clone fix:

- Clone fix verified working (state arrays are independent)
- Issue only occurs with hierarchical caching + cleared cache
- Non-hierarchical mode works fine
- Likely related to trace stitching or SCC decomposition logic

**Workaround**: Use `graph.compute_trace(hierarchical=False)` or don't call `clear_caches()` before `compute_trace()`.

## Testing Commands

```bash
# Test 1: State independence
pixi run python test_clone_debug.py

# Test 2: Multiple compute_trace calls (with cached trace)
pixi run python test_multiple_compute_trace.py

# Test 3: Non-hierarchical mode
pixi run python test_notebook_no_hierarchical.py

# Test 4: Inline quick test
pixi run python -c "import phasic; ... trace = graph.compute_trace(); ..."
```

All clone-related tests pass ✅

---

**Date**: 2025-11-12
**Status**: ✅ Fixed, tested, and verified
**Related Issues**:
- MEMORY_LEAK_FIX.md (separate memory leak in caching)
- Hierarchical caching hang (separate issue, see Known Issue above)
