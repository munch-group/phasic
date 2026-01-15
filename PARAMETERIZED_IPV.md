# Parameterized IPV Implementation

**Date:** 2026-01-15
**Status:** 🚧 In Progress - Adding `is_constant` flag to edge struct
**Goal:** Enable conditional updates of IPV (Initial Probability Vector) edges based on creation syntax

---

## Overview

This document tracks the implementation of parameterized IPV edge support, allowing users to:
- Create **constant IPV edges** (scalar syntax): `g.starting_vertex().add_edge(v1, 1.0)` → remains fixed
- Create **parameterized IPV edges** (array syntax): `g.starting_vertex().add_edge(v1, [0.4, 0.0])` → updates with `update_weights()`

### Key Constraint

**The fundamental challenge:** When `param_length=1`, both constant and parameterized edges have `coefficients_length=1`, making them indistinguishable at runtime.

**Solution:** Add a `bool is_constant` flag to the `ptd_edge` struct to explicitly track edge creation intent.

---

## Design Rationale

### Current System (Before Changes)

**Edge Mode Locking:**
```c
enum ptd_edge_mode {
    PTD_EDGE_MODE_UNLOCKED = 0,      // No non-IPV edges added
    PTD_EDGE_MODE_CONSTANT = 1,       // All non-IPV edges constant
    PTD_EDGE_MODE_PARAMETERIZED = 2   // All non-IPV edges parameterized
};
```

**Key Properties:**
1. ✅ All edges with coefficients must have `coefficients_length == graph->param_length`
2. ✅ All **non-IPV** edges must be same type (all constant OR all parameterized)
3. ✅ IPV edges DON'T participate in mode locking (can be either type)
4. ❌ IPV edges are ALWAYS skipped in `update_weights()` (old behavior)

### Problem with Coefficient-Length-Based Detection

Initial attempt used:
```c
if (edge->coefficients_length < graph->param_length) {
    continue;  // Skip constant edges
}
```

**Why this fails:**
- Constant edges created via `add_edge(v, 1.0)` → stored as `[1.0]` with `coefficients_length=1`
- When `param_length=1`, parameterized edges also have `coefficients_length=1`
- Result: Cannot distinguish constant from parameterized when `param_length=1`

**Test Evidence:**
```python
# Test: test_backward_compatibility_constant_ipv
g.starting_vertex().add_edge(v1, 1.0)  # Constant IPV
v1.add_edge(v_absorbing, [2.0])        # Parameterized graph (param_length=1)

g.update_weights([5.0])

# Expected: IPV weight = 1.0 (constant)
# Actual: IPV weight = 5.0 (incorrectly updated!)
```

---

## Implementation Plan

### Phase 1: Add `is_constant` Flag ✅ COMPLETED

**Modified Files:**

1. **`/Users/kmt/phasic/api/c/phasic.h`** (lines 134-141)
   ```c
   struct ptd_edge {
       struct ptd_vertex *to;
       double weight;
       double *coefficients;
       size_t coefficients_length;
       bool should_free_coefficients;
       bool is_constant;  // ← NEW FIELD
   };
   ```

2. **`/Users/kmt/phasic/api/c/phasic.h`** (lines 173-179)
   ```c
   struct ptd_edge *ptd_graph_add_edge(
       struct ptd_vertex *from,
       struct ptd_vertex *to,
       double *coefficients,
       size_t coefficients_length,
       bool is_constant  // ← NEW PARAMETER
   );
   ```

3. **`/Users/kmt/phasic/src/c/phasic.c`** (line 2850)
   ```c
   edge->is_constant = is_constant;  // ← SET FLAG
   ```

### Phase 2: Update C++ Callers 🚧 IN PROGRESS

**Completed:**

1. **`/Users/kmt/phasic/src/cpp/phasiccpp.cpp`** (line 275)
   ```cpp
   // Constant edge via scalar syntax
   double coeff = weight;
   struct ptd_edge *result = ptd_graph_add_edge(
       this->vertex, to.vertex, &coeff, 1,
       true /* is_constant */  // ← PASS TRUE
   );
   ```

2. **`/Users/kmt/phasic/src/cpp/phasiccpp.cpp`** (line 317)
   ```cpp
   // Parameterized edge via array syntax
   struct ptd_edge *result = ptd_graph_add_edge(
       this->vertex, to.vertex, state, state_length,
       false /* is_constant */  // ← PASS FALSE
   );
   ```

3. **`/Users/kmt/phasic/src/c/phasic_symbolic.c`** (line 799)
   ```c
   // Symbolic DAG edges (parameterized)
   ptd_graph_add_edge(v, to_vertex, &weight, 1, false);
   ```

4. **`/Users/kmt/phasic/src/c/phasic.c`** (line 1200)
   ```c
   // Clone edge - preserve is_constant flag
   struct ptd_edge *new_edge = ptd_graph_add_edge(
       new_v, new_target,
       old_edge->coefficients,
       old_edge->coefficients_length,
       old_edge->is_constant  // ← PRESERVE FLAG
   );
   ```

**Remaining Calls to Update:**

From `grep` results:
```
/Users/kmt/phasic/src/c/phasic.c:2326:  ptd_graph_add_edge(vertex, auxiliary_vertex, &weight1, 1);
/Users/kmt/phasic/src/c/phasic.c:2327:  ptd_graph_add_edge(auxiliary_vertex, vertex, &weight2, 1);
/Users/kmt/phasic/src/c/phasic.c:3786:  ptd_graph_add_edge(...);
/Users/kmt/phasic/src/c/phasic.c:4030:  ptd_graph_add_edge(...);
/Users/kmt/phasic/src/c/phasic.c:4044:  ptd_graph_add_edge(...);
/Users/kmt/phasic/src/c/phasic.c:4054:  ptd_graph_add_edge(...);
```

### Phase 3: Update `update_weights()` Logic ⏳ PENDING

**Target:** `/Users/kmt/phasic/src/c/phasic.c` (lines 3051-3076)

**Current Code (WRONG):**
```c
for (size_t j = 0; j < vertex->edges_length; j++) {
    struct ptd_edge *edge = vertex->edges[j];

    // Skip constant edges
    if (edge->coefficients_length < graph->param_length) {
        continue;  // ← WRONG: Doesn't work when param_length=1
    }

    // Update parameterized edges
    edge->weight = dot(edge->coefficients, theta);
}
```

**Planned Fix:**
```c
for (size_t j = 0; j < vertex->edges_length; j++) {
    struct ptd_edge *edge = vertex->edges[j];

    // Skip constant edges (created with scalar syntax)
    if (edge->is_constant) {
        continue;  // ← CORRECT: Check explicit flag
    }

    // Update parameterized edges (including parameterized IPV!)
    edge->weight = dot(edge->coefficients, theta);
}
```

### Phase 4: Testing ⏳ PENDING

**Test File:** `/Users/kmt/phasic/tests/pytest/test_parameterized_ipv.py`

**Test Coverage:**
1. ✅ `test_constant_ipv_remains_unchanged` - Constant IPV + parameterized graph
2. ✅ `test_parameterized_ipv_updates` - Parameterized IPV gets updated
3. ✅ `test_mixed_ipv_coefficients` - Multi-coefficient IPV edges
4. ✅ `test_trace_with_parameterized_ipv` - Trace recording/instantiation
5. ✅ `test_serialization_with_parameterized_ipv` - Cache persistence
6. ❌ `test_backward_compatibility_constant_ipv` - **FAILING** (param_length=1 case)
7. ✅ `test_coefficient_length_validation` - Coefficient length checking

**Current Test Results:**
```
6 passed, 1 failed
FAILED: test_backward_compatibility_constant_ipv
  Expected: IPV weight = 1.0 (constant)
  Actual:   IPV weight = 5.0 (incorrectly updated)
```

---

## Files Modified

### C/C++ Core
- ✅ `/Users/kmt/phasic/api/c/phasic.h` - Added `is_constant` field and parameter
- ✅ `/Users/kmt/phasic/src/c/phasic.c` - Updated function signature and set flag
- 🚧 `/Users/kmt/phasic/src/c/phasic.c` - Need to update `update_weights()` logic
- 🚧 `/Users/kmt/phasic/src/c/phasic.c` - Need to update remaining 6 callers
- ✅ `/Users/kmt/phasic/src/cpp/phasiccpp.cpp` - Updated C++ wrapper calls
- ✅ `/Users/kmt/phasic/src/c/phasic_symbolic.c` - Updated symbolic DAG call

### Python Layer
- ✅ `/Users/kmt/phasic/src/phasic/__init__.py` - Enabled parameterized IPV serialization
- ✅ `/Users/kmt/phasic/src/phasic/trace_elimination.py` - Updated comments

### Tests
- ✅ `/Users/kmt/phasic/tests/pytest/test_parameterized_ipv.py` - Created comprehensive test suite

---

## Next Steps

1. **Update remaining `ptd_graph_add_edge()` calls** in phasic.c (6 locations)
   - Lines: 2326, 2327, 3786, 4030, 4044, 4054
   - Determine correct `is_constant` value for each context

2. **Fix `update_weights()` logic** to check `edge->is_constant` instead of coefficient length

3. **Rebuild C extension:**
   ```bash
   pixi run install-dev
   ```

4. **Clear trace cache** (contains old serialization format):
   ```bash
   rm -rf ~/.phasic_cache/traces/*
   ```

5. **Run full test suite:**
   ```bash
   pixi run python -m pytest tests/pytest/test_parameterized_ipv.py -v
   ```

6. **Verify backward compatibility** with existing tests:
   ```bash
   pixi run python -m pytest tests/pytest/ -v
   ```

---

## Design Guarantees

After implementation, the system will guarantee:

1. ✅ **Coefficient length uniformity:** All edges must have `coefficients_length == param_length`
2. ✅ **Edge mode consistency:** All non-IPV edges must be same type (constant OR parameterized)
3. ✅ **IPV flexibility:** IPV edges can be constant or parameterized independently
4. ✅ **Correct updates:** Only parameterized edges (including parameterized IPV) are updated
5. ✅ **Backward compatibility:** Constant IPV edges remain fixed (traditional behavior)

---

## User API

### Creating Constant IPV
```python
g = Graph(1)
v1 = g.find_or_create_vertex([1])

# Constant IPV - will NOT change with update_weights()
g.starting_vertex().add_edge(v1, 1.0)

# Parameterized graph edge
v1.add_edge(v_absorbing, [2.0])

g.update_weights([5.0])
# IPV weight: 1.0 (unchanged)
# v1 edge weight: 10.0 (2.0 * 5.0)
```

### Creating Parameterized IPV
```python
g = Graph(1)
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([2])

# Parameterized graph edges first (locks mode)
v1.add_edge(v_absorbing, [1.0, 0.0])
v2.add_edge(v_absorbing, [0.0, 1.0])

# Parameterized IPV - WILL update with update_weights()
g.starting_vertex().add_edge(v1, [0.4, 0.0])
g.starting_vertex().add_edge(v2, [0.6, 0.0])

g.update_weights([2.0, 3.0])
# IPV edge 0 weight: 0.8 (0.4 * 2.0 + 0.0 * 3.0)
# IPV edge 1 weight: 1.2 (0.6 * 2.0 + 0.0 * 3.0)
```

---

## Implementation Notes

### Why Not Use Coefficient Values?

**Considered approach:** Detect constant edges by checking if all coefficients are identical.
```c
// REJECTED: Fragile heuristic
bool is_constant_by_value = true;
for (size_t i = 1; i < coefficients_length; i++) {
    if (coefficients[i] != coefficients[0]) {
        is_constant_by_value = false;
        break;
    }
}
```

**Why rejected:**
- User could legitimately create `[2.0, 2.0, 2.0]` as a parameterized edge
- Heuristics are fragile and error-prone
- Explicit intent is clearer and more robust

### Why Not Store Edge Mode Per-Edge?

**Considered:** Store `enum ptd_edge_mode` in each edge instead of `bool is_constant`.

**Why rejected:**
- Adds 4 bytes per edge (enum) vs 1 byte (bool)
- Edge mode is already tracked at graph level
- Only need to distinguish constant/parameterized, not unlocked state

### Memory Impact

**Added per edge:**
- 1 byte (`bool is_constant`)
- Minimal padding (struct alignment)

**Estimated impact:**
- Graph with 10,000 edges: +10 KB
- Negligible compared to coefficient arrays and other metadata

---

## References

- **Issue Discussion:** Previous conversation summary (context from session start)
- **Related Code:**
  - Edge mode locking: `/Users/kmt/phasic/src/cpp/phasiccpp.cpp` lines 254-305
  - Graph update logic: `/Users/kmt/phasic/src/c/phasic.c` lines 2954-3130
  - Trace recording: `/Users/kmt/phasic/src/phasic/trace_elimination.py` lines 540-577
