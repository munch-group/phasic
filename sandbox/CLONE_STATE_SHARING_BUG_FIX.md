# Fix for Clone State Sharing Bug (Crash/Memory Corruption)

## Problem

When using `graph.clone()` in `compute_trace()` to preserve the original graph, the program would crash with memory corruption after printing the hierarchical cache warnings. This happened because cloned graphs shared state arrays with the original graph.

## Root Cause

The bug was in `ptd_clone_graph()` (src/c/phasic.c):

1. **Vertex cloning** called `ptd_vertex_create_state(new_graph, old_v->state)`
2. **`ptd_vertex_create_state()`** stored the state pointer directly: `vertex->state = state;` (line 2618)
3. **No copying** - both old and new vertices pointed to the SAME state array
4. **Memory corruption** when original graph freed: `free(vertex->state);` (line 2648)
5. **Crash** when cloned graph tried to access freed memory

### Timeline of Corruption

```
1. _graph = self.clone()          # Clone created
2. get_trace_hierarchical(_graph) # SCC subdivision creates subgraphs
3. Enhanced subgraphs reference vertices in _graph
4. compute_trace() returns
5. _graph goes out of scope
6. Python GC triggers → rf_graph destructor
7. ptd_graph_destroy() → ptd_vertex_destroy()
8. free(vertex->state) on SHARED state arrays  <-- Original freed
9. Cloned vertices now have DANGLING POINTERS
10. Access to cloned graph → SEGFAULT/corruption
```

## Solution

Modified `ptd_clone_graph()` to **allocate and copy** state arrays instead of sharing pointers:

```c
// OLD CODE (bug):
struct ptd_vertex *new_v = ptd_vertex_create_state(new_graph, old_v->state);
// Shares state pointer → memory corruption when original freed

// NEW CODE (fix):
int *new_state = (int *)malloc(graph->state_length * sizeof(int));
memcpy(new_state, old_v->state, graph->state_length * sizeof(int));
struct ptd_vertex *new_v = ptd_vertex_create_state(new_graph, new_state);
// Independent copy → safe when original freed
```

### Special Handling for Starting Vertex

The starting vertex was already created by `ptd_graph_create()`, so we:
1. Allocate new state array
2. Copy data from old state
3. Free the default state
4. Assign the copied state

```c
if (old_v == graph->starting_vertex) {
    int *new_state = (int *)malloc(graph->state_length * sizeof(int));
    memcpy(new_state, old_v->state, graph->state_length * sizeof(int));
    free(new_graph->starting_vertex->state);  // Free default
    new_graph->starting_vertex->state = new_state;
    vertex_map[i] = new_graph->starting_vertex;
}
```

## Impact

**Before fix:**
- Cloning would crash after SCC subdivision
- Memory corruption from shared state pointers
- Dangling pointers when original graph freed

**After fix:**
- ✅ Clone creates fully independent graph
- ✅ Original graph can be freed without affecting clone
- ✅ Multiple `compute_trace()` calls work correctly
- ✅ Original graph preserved (2999 vertices → 2999 vertices after calls)

## Testing

```python
graph = phasic.Graph(model, ipv=ipv)  # 2999 vertices

# First call
trace1 = graph.compute_trace()  # Uses clone internally
print(graph.vertices_length())  # Still 2999! ✓

# Second call
trace2 = graph.compute_trace()  # Uses clone again
print(graph.vertices_length())  # Still 2999! ✓
```

## Files Modified

**src/c/phasic.c** (lines 1035-1077):
- Added state array allocation and copying for all vertices
- Special handling for starting vertex state
- Ensures complete memory independence

**src/phasic/__init__.py** (line 3430):
- Uses `_graph = self.clone()` to preserve original
- Clone is safe now thanks to C fix

## Technical Details

### Memory Ownership

**Before fix:**
```
Original Graph          Cloned Graph
  vertices[0] --------→ state[0,1,2,...]  ←-------- vertices[0]
  vertices[1] --------→ state[4,0,0,...]  ←-------- vertices[1]
                        (SHARED - BUG!)
```

**After fix:**
```
Original Graph          Cloned Graph
  vertices[0] → state[0,1,2,...]     vertices[0] → state[0,1,2,...] (copy)
  vertices[1] → state[4,0,0,...]     vertices[1] → state[4,0,0,...] (copy)
              (INDEPENDENT - SAFE!)
```

### Why This Wasn't Caught Earlier

- Small graphs (< 50 vertices) don't use SCC subdivision
- They return cached trace before original graph freed
- Bug only manifested with:
  1. Large graphs (≥ 50 vertices)
  2. SCC subdivision enabled
  3. Enhanced subgraphs created
  4. Original graph freed while subgraphs alive

---

**Date**: 2025-11-12
**Status**: ✅ Fixed and tested
**Related Issues**:
- MEMORY_LEAK_FIX.md (separate memory leak in caching)
- COMPUTE_TRACE_RERUN_FIX.md (helpful error messages)
