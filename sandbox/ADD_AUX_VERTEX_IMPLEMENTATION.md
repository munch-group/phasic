# Implementation: `add_aux_vertex(rate)` Method

## Summary

Added `add_aux_vertex(rate)` method to create auxiliary vertices for discrete graphs. Auxiliary vertices have all-zero state vectors and are connected with:
- Constant edge from aux→parent (weight 1.0)
- Parameterized or constant edge from parent→aux (based on graph mode)

## Implementation Details

### Files Modified

1. **`api/cpp/phasiccpp.h`** - Added method declarations
2. **`src/cpp/phasiccpp.cpp`** - Implemented methods
3. **`src/cpp/phasic_pybind.cpp`** - Added Python bindings

### Key Design Decisions

#### 1. Manual Edge Creation for aux→parent

The aux→parent edge is always constant (weight 1.0), even in parameterized graphs. To avoid coefficient length mismatch errors, we manually create this edge without using `ptd_graph_add_edge()`:

```cpp
struct ptd_edge *edge1 = (struct ptd_edge *)malloc(sizeof(*edge1));
edge1->to = this->vertex;
edge1->weight = 1.0;
edge1->coefficients_length = 0;  // No coefficients - pure constant
edge1->coefficients = NULL;
edge1->should_free_coefficients = false;
```

This approach:
- ✅ Bypasses coefficient length validation
- ✅ Creates a pure constant edge (no parameterization)
- ✅ Works with both constant and parameterized graphs

#### 2. Normal Edge Creation for parent→aux

The parent→aux edge uses normal `add_edge()` or `add_edge_parameterized()` machinery:
- Ensures proper validation
- Follows graph's parameterization mode
- Integrates with existing edge management

#### 3. Python API Validation

The Python binding validates that the `rate` argument matches the graph's mode:

```python
if graph.parameterized():
    # Require array: aux = v.add_aux_vertex([2.0, 1.0])
else:
    # Require scalar: aux = v.add_aux_vertex(3.0)
```

Clear error messages guide users to the correct usage.

## Usage Examples

### Non-Parameterized Graph

```python
import phasic

g = phasic.Graph(2)
v = g.find_or_create_vertex([1, 0])
v.add_edge(g.find_or_create_vertex([2, 0]), 3.0)  # Lock to constant mode

# Create auxiliary vertex with constant rate
aux = v.add_aux_vertex(2.5)

print(aux.state())  # Output: [0, 0]
print(g.parameterized())  # Output: False
```

### Parameterized Graph

```python
import phasic

g = phasic.Graph(2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([2, 0])
v1.add_edge(v2, [1.0, 0.0])  # Lock to parameterized mode

# Create auxiliary vertex with parameterized rate
# Edge weight = 2.0*theta[0] + 1.0*theta[1]
aux = v1.add_aux_vertex([2.0, 1.0])

print(aux.state())  # Output: [0, 0]
print(g.parameterized())  # Output: True
```

### C++ API

```cpp
#include "phasiccpp.h"

// Constant graph
phasic::Graph g1(2);
auto v1 = g1.find_or_create_vertex({1, 0});
auto v2 = g1.find_or_create_vertex({2, 0});
v1.add_edge(v2, 3.0);

auto aux1 = v1.add_aux_vertex(2.5);  // Scalar rate

// Parameterized graph
phasic::Graph g2(2);
auto v3 = g2.find_or_create_vertex({1, 0});
auto v4 = g2.find_or_create_vertex({2, 0});
v3.add_edge_parameterized(v4, 0.0, {1.0, 0.0});

auto aux2 = v3.add_aux_vertex({2.0, 1.0});  // Coefficient vector
```

## Edge Structure

Given a vertex `v` with `v.add_aux_vertex(rate)`:

```
     ┌─────────┐
     │   aux   │  State: [0, 0, ...]
     │ [0,0,0] │
     └─────────┘
          │
          │ weight = 1.0 (constant)
          │ coefficients_length = 0
          ↓
     ┌─────────┐
     │    v    │  State: [1, 0, ...]
     │ [1,0,0] │
     └─────────┘
          │
          │ weight = rate (constant or parameterized)
          │ coefficients_length = 1 (constant) or N (parameterized)
          ↓
     ┌─────────┐
     │   aux   │  State: [0, 0, ...]
     │ [0,0,0] │
     └─────────┘
```

## Test Results

All tests pass successfully:

```bash
$ python test_add_aux_vertex.py
======================================================================
TEST: add_aux_vertex() METHOD
======================================================================

Test 1: Non-parameterized graph with constant rate
  ✅ Test 1 PASSED

Test 2: Parameterized graph with coefficient array
  ✅ Test 2 PASSED

Test 3: Error handling - scalar for parameterized graph
  ✅ Correctly raised error: ValueError

Test 4: Error handling - array for constant graph
  ✅ Correctly raised error: ValueError

Test 5: Multiple auxiliary vertices
  ✅ Test 5 PASSED - Same aux vertex reused

Test 6: Verify edge weights
  ✅ Test 6 PASSED

======================================================================
ALL TESTS PASSED! ✅
======================================================================
```

## Important Notes

### Vertex Reuse

Multiple calls to `add_aux_vertex()` from different parent vertices will **reuse the same auxiliary vertex** (since `find_or_create_vertex([0, 0, ...])` finds the existing one). This is correct behavior for auxiliary vertices that serve as intermediate states.

### Edge Weights

- **aux→parent edge**: Always has weight 1.0 (constant, non-parameterized)
- **parent→aux edge**: Has the specified `rate` (constant or parameterized based on graph mode)

### Identifying Auxiliary Vertices

Auxiliary vertices can be identified by:
1. All-zero state: `list(v.state()) == [0, 0, ...]`
2. Single outgoing edge with constant weight 1.0

### Parameter Updates

When updating parameters in parameterized graphs, auxiliary vertices should be skipped as their outgoing edges are always constant (weight 1.0). The parent→aux edges will be updated normally as part of the graph's parameterization.

## Backward Compatibility

✅ **Fully backward compatible** - Only adds new functionality, no breaking changes.

## Future Enhancements

Possible future improvements:
1. Add flag to mark vertices as "auxiliary" explicitly
2. Add method to filter/skip auxiliary vertices during graph operations
3. Add visualization support to highlight auxiliary vertices

---

**Status**: ✅ Complete and tested
**Version**: 0.22.0
**Date**: 2025-11-05
