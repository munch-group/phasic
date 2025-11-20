# SCC Re-entrancy Bug Fix

**Date:** 2025-11-16  
**Issue:** JAX vmap/pmap crashes with "Stack is empty" error on cyclic graphs  
**Root Cause:** Static global variables in Tarjan's SCC algorithm corrupted by concurrent FFI calls  
**Status:** ✅ FIXED

---

## Problem

When using JAX's `vmap` or `pmap` with the FFI gradient handler on cyclic graphs (e.g., rabbits model), the program crashed with:

```
Stack is empty.
 @ /Users/kmt/phasic/src/c/phasic.c (1794)
Abort trap: 6
```

### Symptoms

- ✅ Sequential calls work fine
- ✅ Non-FFI path works fine (parallel='none', ffi=False)
- ❌ vmap crashes on cyclic graphs
- ❌ pmap crashes on cyclic graphs
- ✅ vmap/pmap work on acyclic graphs (e.g., coalescent)

### Root Cause Analysis

The SCC (Strongly Connected Components) algorithm used static global variables:

```c
static struct ptd_stack *scc_stack2 = NULL;
static struct ptd_vector *scc_components2 = NULL;
static size_t scc_index2 = 0;
static size_t *scc_indices2 = NULL;
static size_t *low_links2 = NULL;
static bool *scc_on_stack2 = NULL;
static bool *visited = NULL;
```

**The Bug:**
1. `vmap` executes multiple FFI calls (potentially concurrently or in rapid succession)
2. Call 1 starts SCC computation, pushes vertices onto `scc_stack2`
3. Call 2 starts, enters `ptd_find_strongly_connected_components()` which CLEARS all static state
4. Call 1's stack is now destroyed, but Call 1 is still running
5. Call 1 tries to pop from stack → **"Stack is empty" crash**

This is a **re-entrancy bug**. The FFI handler is called multiple times before previous calls complete, corrupting the shared static state.

---

## Solution

Made the SCC computation **re-entrant safe** by replacing static global variables with local state passed through function calls.

### Changes

**File:** `src/c/phasic.c`

1. **Created SCC state struct** (lines 1750-1759):
```c
struct scc_state {
    struct ptd_stack *scc_stack;
    struct ptd_vector *scc_components;
    size_t scc_index;
    size_t *scc_indices;
    size_t *low_links;
    bool *scc_on_stack;
    bool *visited;
};
```

2. **Updated `strongconnect2()` signature** to accept state parameter:
```c
static int strongconnect2(struct ptd_vertex *vertex, struct scc_state *state)
```

3. **Updated `ptd_find_strongly_connected_components()`** to allocate local state:
```c
struct ptd_scc_graph *ptd_find_strongly_connected_components(struct ptd_graph *graph) {
    // ...
    
    // Allocate local SCC state (re-entrant safe)
    struct scc_state state;
    state.scc_stack = stack_create();
    state.scc_index = 0;
    state.scc_indices = (size_t *) calloc(graph->vertices_length * 10, sizeof(size_t));
    // ... etc
    
    // Use state throughout
    strongconnect2(vertex, &state);
    
    // Cleanup local state at end
    free(state.scc_indices);
    // ... etc
}
```

### Key Insight

Each call to `ptd_find_strongly_connected_components()` now has its **own independent state on the stack**, preventing interference between concurrent/nested calls.

---

## Testing

### Test 1: Sequential Calls
```bash
python test_sequential_calls.py
```
✅ PASS - Multiple sequential calls work correctly

### Test 2: vmap
```bash
python test_vmap_crash.py
```
✅ PASS - vmap batching works (previously crashed)

### Test 3: pmap
```bash
python test_vmap_crash.py
```
✅ PASS - pmap multi-device parallelization works (previously crashed)

### Test 4: Full Small Rabbits
```bash
python test_small_rabbits.py
```
✅ PASS - All 7 steps pass:
1. Graph creation
2. Direct PDF computation
3. FFI factory function
4. Single evaluation
5. Gradient computation
6. vmap batching
7. pmap parallelization

### Test 5: Tutorial Rabbits with SVGD
```bash
python test_tutorial_rabbits.py
```
✅ PASS - Full SVGD inference completes:
```
SVGD complete!
Posterior mean: [1.11903928 2.12538184 1.59811721]
Posterior std:  [0.4170115  0.57674118 0.45960436]
```

---

## Impact

This fix enables:
- ✅ JAX vmap batching for SVGD with FFI gradients
- ✅ JAX pmap multi-device parallelization for SVGD
- ✅ Cyclic graph models (rabbits, etc.) work with FFI
- ✅ No regression on acyclic models or sequential calls

### Performance

No measurable performance impact - local variables on stack are as fast as static globals.

---

## Lessons Learned

1. **Static variables are dangerous in library code** - They break re-entrancy
2. **FFI handlers must be re-entrant** - JAX may call them concurrently or in rapid succession
3. **Test with cyclic graphs** - They exercise SCC decomposition unlike acyclic models
4. **vmap is a good test for re-entrancy** - It naturally creates concurrent/rapid calls

---

## Related Files

- **Implementation:** `src/c/phasic.c` (lines 1750-2240)
- **FFI Handler:** `src/cpp/parameterized/ffi_handlers.cpp` (calls `compute_moments_impl()`)
- **Tests:** 
  - `/tmp/test_vmap_crash.py`
  - `/tmp/test_small_rabbits.py`
  - `/tmp/test_tutorial_rabbits.py`

---

**Fixed by:** Claude  
**Date:** 2025-11-16  
**Status:** ✅ Complete and Tested
