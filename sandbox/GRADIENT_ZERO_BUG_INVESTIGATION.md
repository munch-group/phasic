# Gradient Zero Bug Investigation

**Date:** 2025-11-16  
**Status:** IN PROGRESS - Root cause partially identified

---

## Problem

FFI gradients are returning zero for all parameters when they should be non-zero.

### Test Case

```python
import phasic
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

# Forward pass works
pmf, moments = fn(theta, times, None)
# Output: PMF=[0.30932152], Moments=[0.50830565, 0.4742994]

# Gradients are ZERO (WRONG!)
import jax
grad_fn = jax.grad(lambda t: fn(t, times, None)[0][0])
gradient = grad_fn(theta)
# Output: [0. 0. 0.]  <-- SHOULD BE NON-ZERO!
```

---

## Investigation Steps

### 1. ✅ Fixed: Starting Vertex Index Bug

**Found**: `compute_pmf_with_gradient()` at line 6217 hardcoded `prob[0] = 1.0` assuming starting vertex is at index 0.

**Reality**: Starting vertex can be at any index (e.g., rabbits has starting state [2, 0] which may not be at index 0).

**Fix**: Changed to:
```c
size_t starting_idx = graph->starting_vertex->index;
prob[starting_idx] = 1.0;
```

**Result**: This fixed ONE bug, but gradients still zero.

### 2. ❌ Graph Structure: Parameterized Edges Exist

**Verified**: The rabbits graph has 12 parameterized edges.

Example edge from serialization:
```
[1.0, 2.0, 2.0, 0.0, 0.0]
```
Meaning: From vertex 1 to vertex 2, weight = 2.0*theta[0] + 0.0*theta[1] + 0.0*theta[2]

So the graph structure is correct and has parameter dependencies.

### 3. 🔍 Architecture Issue: GraphBuilder.build() Flow

**FFI Handler Flow**:
1. `structure_json` passed to FFI handler
2. `GraphBuilder builder(json_str)` created
3. `Graph g = builder.build(theta_data, n_params)` - builds concrete graph
4. `struct ptd_graph* c_graph = g.c_graph()` - gets C struct
5. `ptd_graph_pdf_with_gradient(c_graph, ...)` - computes PDF + gradients

**Question**: Does `builder.build()` preserve the `coefficients[]` array for gradient computation?

**Answer from phasic.h**:
```c
struct ptd_edge {
    struct ptd_vertex *to;
    double weight;              // Current evaluated weight
    double *coefficients;       // ALWAYS non-NULL, length = graph->param_length
    size_t coefficients_length; // Always = graph->param_length
    bool should_free_coefficients;
};
```

The comment says "ALWAYS non-NULL", suggesting coefficients ARE preserved!

### 4. 🔍 NEXT: Check if coefficients_length is correctly set

**Gradient code logic** (phasic.c:6276-6284):
```c
if (edge->coefficients_length > 1) {
    // Parameterized edge: weight = Σ coefficients[i] * params[i]
    for (size_t i = 0; i < n_params && i < edge->coefficients_length; i++) {
        weight += edge->coefficients[i] * params[i];
        exit_rate_grad[i] += edge->coefficients[i];  // <-- Gradient!
    }
} else {
    // Constant edge: weight = coefficients[0]
    weight = edge->coefficients[0];
    // NO gradient for constant edges
}
```

**Hypothesis**: GraphBuilder might be setting `coefficients_length = 1` for all edges, making them look like "constant" edges even though they have parameter dependencies!

**To verify**: Need to check GraphBuilder code to see how it sets up `coefficients_length`.

---

## Possible Root Causes

1. **GraphBuilder sets coefficients_length = 1** (most likely)
   - Even parameterized edges get `coefficients_length = 1`
   - Gradient code sees them as "constant" and skips gradient computation

2. **GraphBuilder doesn't copy coefficients array** (less likely given struct comment)
   - `coefficients` pointer might be NULL or point to wrong data

3. **Parameter count mismatch**
   - `n_params` might not match `edge->coefficients_length`
   - But we're passing 3 params and graph has 3 params

---

## Next Steps

1. **Examine GraphBuilder edge creation code**
   - Find where it sets `edge->coefficients` and `edge->coefficients_length`
   - Verify parameterized edges get `coefficients_length = n_params`

2. **Add debug logging**
   - Print `edge->coefficients_length` for each edge in gradient function
   - Verify the gradient computation loop is actually executed

3. **Test with simple single-parameter model**
   - Simplify to eliminate variables
   - Single exponential distribution should have clear gradient

---

## Files Involved

- `src/c/phasic.c:6190-6400` - `compute_pmf_with_gradient()` function
- `src/c/phasic.c:6402-6481` - `ptd_graph_pdf_with_gradient()` wrapper
- `src/cpp/parameterized/ffi_handlers.cpp:318-433` - FFI handler
- `src/cpp/parameterized/graph_builder_ffi.cpp` - GraphBuilder implementation (TO INVESTIGATE)
- `api/c/phasic.h` - C struct definitions

---

**Status**: Need to investigate GraphBuilder to find why coefficients_length is wrong.
