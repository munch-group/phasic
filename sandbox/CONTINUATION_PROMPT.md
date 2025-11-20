# Continuation Prompt: FFI Gradients for pmap

**Date:** 2025-11-16  
**Context:** Implementing FFI gradients to enable JAX pmap parallelization for SVGD  
**Current Status:** SCC bug fixed, but gradients returning zero

---

## What We're Doing

Enabling pmap parallelization for SVGD by implementing custom gradients via JAX FFI. The goal is to allow multi-device parallelization without regressions or fallbacks.

**Principles:**
- NO REGRESSIONS - Every fix must not break existing functionality
- NO QUICK FIXES - Proper architectural solutions only
- NO FALLBACKS - Code must fail clearly, not silently degrade

---

## Progress Summary

### ✅ COMPLETED

1. **SCC Re-entrancy Bug Fixed** (`src/c/phasic.c:1750-2234`)
   - **Issue**: Static global variables corrupted by concurrent vmap/pmap calls
   - **Fix**: Created `struct scc_state` with local variables passed through function calls
   - **Result**: vmap and pmap now work on cyclic graphs (rabbits model)
   - **Details**: See `SCC_REENTRANCY_BUG_FIX.md`

2. **Starting Vertex Index Bug Fixed** (`src/c/phasic.c:6217-6219`)
   - **Issue**: Hardcoded `prob[0] = 1.0` assumed starting vertex at index 0
   - **Fix**: Changed to `prob[graph->starting_vertex->index] = 1.0`
   - **Result**: Fixes one bug, but gradients still zero

3. **Factory Pattern for pmap** (`src/phasic/ffi_wrappers.py:763-920`)
   - **Issue**: JAX pmap inspected JSON strings in function closures
   - **Fix**: Factory pattern creates functions with JSON captured at creation time
   - **Result**: pmap no longer fails on tracer inspection

### 🐛 CURRENT BUG: Zero Gradients

**Problem**: FFI gradients return `[0. 0. 0.]` when they should be non-zero

**Test Case**:
```python
import phasic
import jax
import jax.numpy as jnp

phasic.configure(ffi=True)

@phasic.callback([2, 0])
def rabbits(state):
    left, right = state
    transitions = []
    if left:
        transitions.append([[left - 1, right + 1], [left, 0, 0]])
        transitions.append([[0, right], [0, 1, 0]])
    if right:
        transitions.append([[left + 1, right - 1], [right, 0, 0]])
        transitions.append([[left, 0], [0, 0, 1]])
    return transitions

graph = phasic.Graph(rabbits)
from phasic.ffi_wrappers import _make_pmf_and_moments_autodiff_function, _make_json_serializable
import json

structure_json_str = json.dumps(_make_json_serializable(graph.serialize(param_length=3)))
fn = _make_pmf_and_moments_autodiff_function(structure_json_str, nr_moments=2, discrete=False, granularity=100)

theta = jnp.array([1.0, 2.0, 4.0])
times = jnp.array([1.0])

# Forward pass WORKS
pmf, moments = fn(theta, times, None)
# Output: PMF=[0.30932152], Moments=[0.50830565, 0.4742994]

# Gradients are ZERO (WRONG!)
grad_fn = jax.grad(lambda t: fn(t, times, None)[0][0])
gradient = grad_fn(theta)
# Output: [0. 0. 0.]  <-- SHOULD BE NON-ZERO!
```

**Hypothesis**: GraphBuilder sets `coefficients_length = 1` for all edges, making gradient code treat them as "constant" edges.

**Critical Code Path**:
```
FFI Handler (ffi_handlers.cpp:364)
  → GraphBuilder::build(theta, n_params)
  → Graph g with concrete weights
  → struct ptd_graph* c_graph = g.c_graph()
  → ptd_graph_pdf_with_gradient(c_graph, ...)  // phasic.c:6402
    → compute_pmf_with_gradient(graph, ...)    // phasic.c:6190
      → Check: if (edge->coefficients_length > 1) {
          // Parameterized: compute gradients
        } else {
          // Constant: NO gradients
        }
```

**Key Question**: Does GraphBuilder preserve `coefficients[]` array and set `coefficients_length = n_params` for parameterized edges?

**From phasic.h**:
```c
struct ptd_edge {
    struct ptd_vertex *to;
    double weight;              // Current evaluated weight
    double *coefficients;       // ALWAYS non-NULL, length = graph->param_length
    size_t coefficients_length; // Always = graph->param_length
    bool should_free_coefficients;
};
```

**Verified Facts**:
- Graph has 12 parameterized edges (serialization shows coefficients)
- Forward pass computes correct PDF values
- Gradients are zero for all 3 parameters

---

## Next Actions

1. **INVESTIGATE GraphBuilder edge creation**
   - File: `src/cpp/parameterized/graph_builder_ffi.cpp`
   - Find where it sets `edge->coefficients` and `edge->coefficients_length`
   - Check if parameterized edges get `coefficients_length = n_params` or just `1`

2. **ADD DEBUG LOGGING if needed**
   - Print `edge->coefficients_length` in `compute_pmf_with_gradient()`
   - Verify parameterized edges have length > 1
   - Check if gradient loop is executed

3. **FIX the root cause**
   - If GraphBuilder is wrong: fix edge creation to preserve parameterization
   - If gradient code is wrong: fix the logic to detect parameterized edges
   - NO QUICK FIXES - must be proper architectural solution

4. **TEST gradients vs finite differences**
   - Verify gradient correctness after fix
   - Compare with numerical gradients

5. **TEST full SVGD with gradients**
   - Run tutorial rabbits example with pmap
   - Verify no regression on simple models

---

## Key Files

- `src/c/phasic.c:6190-6481` - Gradient computation functions
- `src/cpp/parameterized/ffi_handlers.cpp:318-433` - FFI gradient handler
- `src/cpp/parameterized/graph_builder_ffi.cpp` - GraphBuilder (TO INVESTIGATE)
- `src/phasic/ffi_wrappers.py:763-920` - Factory pattern for pmap
- `src/phasic/__init__.py:3458-3486` - SVGD integration
- `api/c/phasic.h` - C struct definitions

---

## Documentation

- `SCC_REENTRANCY_BUG_FIX.md` - SCC bug analysis and fix
- `GRADIENT_ZERO_BUG_INVESTIGATION.md` - Current bug investigation
- `PLAN_FFI_GRADIENTS_FOR_PMAP.md` - Overall implementation plan

---

## Command to Continue

```
Please continue implementing FFI gradients for pmap. The SCC bug is fixed and 
vmap/pmap work, but gradients are returning zero. Investigate GraphBuilder edge 
creation to find why coefficients_length is wrong. Remember: NO REGRESSIONS, 
NO QUICK FIXES, NO FALLBACKS.
```
