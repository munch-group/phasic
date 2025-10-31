# Refactoring Plan: Eliminate `base_weight` Field

**Date:** 2025-10-31
**Status:** READY FOR EXECUTION
**Version:** 0.21.3 → 0.22.0 (breaking change)

---

## Executive Summary

Remove the `base_weight` field from parameterized edges by ensuring starting vertex edges are NEVER created as parameterized. Starting edges will use regular `add_edge()` instead of `add_edge_parameterized()`, eliminating the need for a separate base_weight field to handle constant-weight edges.

### Rationale

**Current problem:**
- Callback returns `[state, weight, [coefficients]]` for ALL edges
- Starting edges have `[state, 1.0, []]` (empty coefficients)
- Graph builder calls `add_edge_parameterized()` for these → marked as `parameterized=true`
- `update_parameterized_weights()` tries to update them → needs `base_weight` to preserve constant weight

**Solution:**
- Check if coefficients array is empty in graph builder
- If empty: use `add_edge()` (non-parameterized)
- If non-empty: use `add_edge_parameterized()`
- Starting edges naturally have empty arrays → automatically non-parameterized
- No `base_weight` field needed

---

## Git History Context

### Commits Introducing base_weight

1. **`1973397`** (Oct 24, 2025) - "Fix broken inference by adding missing base_weight to parameterized edges"
   - Added `base_weight` to C++ GraphBuilder
   - Changed serialization format
   - Files: `src/cpp/parameterized/graph_builder.{hpp,cpp}`

2. **`fff1e6c`** (Oct 25, 2025) - "Fix base_weight handling and starting vertex for parameterized graphs"
   - Added `base_weight` to C API
   - Added skip-starting-vertex logic
   - Files: `src/c/phasic.c`, `api/c/phasic.h`, `src/phasic/trace_elimination.py`

### Why It Was Added

Original issue: Parameterized edges were computing weight as `dot(coeffs, params)` without any constant term. Starting edges with empty `[]` coefficients had weight = 0, breaking the distribution.

Our solution: Instead of adding `base_weight`, we prevent starting edges from being parameterized in the first place.

---

## Changes Required

### 1. Python Graph Builder (src/cpp/phasic_pybind.cpp)

#### Location: Lines 719-727 (starting vertex edges)

**Current:**
```cpp
long double weight = std::get<1>(tup);
std::vector<double> edge_state = std::get<2>(tup);

if (!graph) {
  graph = new phasic::Graph(child_state.size());
}
phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
graph->starting_vertex().add_edge_parameterized(child_vertex, weight, edge_state);
```

**New:**
```cpp
long double weight = std::get<1>(tup);
std::vector<double> edge_state = std::get<2>(tup);

if (!graph) {
  graph = new phasic::Graph(child_state.size());
}
phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);

// Starting edges: use add_edge() if coefficients empty, add_edge_parameterized() otherwise
if (edge_state.empty()) {
    graph->starting_vertex().add_edge(child_vertex, weight);
} else {
    graph->starting_vertex().add_edge_parameterized(child_vertex, weight, edge_state);
}
```

#### Location: Lines 748-760 (non-starting vertex edges)

**Current:**
```cpp
phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);
this_vertex.add_edge_parameterized(child_vertex, weight, edge_state);
```

**New:**
```cpp
phasic::Vertex child_vertex = graph->find_or_create_vertex(child_state);

// Use add_edge() if coefficients empty, add_edge_parameterized() otherwise
if (edge_state.empty()) {
    this_vertex.add_edge(child_vertex, weight);
} else {
    this_vertex.add_edge_parameterized(child_vertex, weight, edge_state);
}
```

---

### 2. C API Header (api/c/phasic.h)

#### Location: Line 134

**Current:**
```c
struct ptd_edge_parameterized {
    struct ptd_vertex *to;
    double weight;
    bool parameterized;
    double *state;
    size_t state_length;
    bool should_free_state;
    double base_weight;  // ← REMOVE THIS
};
```

**New:**
```c
struct ptd_edge_parameterized {
    struct ptd_vertex *to;
    double weight;
    bool parameterized;
    double *state;
    size_t state_length;
    bool should_free_state;
    // base_weight removed
};
```

---

### 3. C Implementation (src/c/phasic.c)

#### Location: Line 1588

**Current:**
```c
edge->weight = weight;
edge->base_weight = weight;  // ← REMOVE THIS
edge->parameterized = true;
```

**New:**
```c
edge->weight = weight;
// base_weight removed
edge->parameterized = true;
```

#### Location: Lines 1657-1665

**Current:**
```c
void ptd_edge_update_weight_parameterized(
        struct ptd_edge *edge,
        double *scalars,
        size_t scalars_length
) {
    // Start with base weight, then add parameterized component
    double weight = ((struct ptd_edge_parameterized *) edge)->base_weight;

    for (size_t i = 0; i < scalars_length; ++i) {
        weight += scalars[i] * ((struct ptd_edge_parameterized *) edge)->state[i];
    }

    edge->weight = weight;
    // ...
}
```

**New:**
```c
void ptd_edge_update_weight_parameterized(
        struct ptd_edge *edge,
        double *scalars,
        size_t scalars_length
) {
    // Compute weight as dot product only (no base_weight)
    double weight = 0.0;

    for (size_t i = 0; i < scalars_length; ++i) {
        weight += scalars[i] * ((struct ptd_edge_parameterized *) edge)->state[i];
    }

    edge->weight = weight;
    // ...
}
```

**KEEP THIS** (Line 1765-1769): Skip starting vertex in parameter updates (still needed as defense-in-depth)

---

### 4. C++ API (api/cpp/phasiccpp.h)

#### Location: Lines 932-934

**Current:**
```cpp
class ParameterizedEdge : public Edge {
public:
    // ...
    double base_weight() const;  // ← REMOVE THIS
};
```

**New:**
```cpp
class ParameterizedEdge : public Edge {
public:
    // ...
    // base_weight() method removed
};
```

---

### 5. Python Bindings (src/cpp/phasic_pybind.cpp)

#### Location: Lines 3062-3065

**Current:**
```cpp
.def("base_weight", &phasic::ParameterizedEdge::base_weight,
    "Get the base weight of the parameterized edge")
```

**New:**
```cpp
// Remove base_weight() method exposure
```

---

### 6. C++ GraphBuilder Header (src/cpp/parameterized/graph_builder.hpp)

#### Location: Line 164

**Current:**
```cpp
struct ParameterizedEdge {
    int from_idx;
    int to_idx;
    double base_weight;  // ← REMOVE THIS
    std::vector<double> coefficients;
};
```

**New:**
```cpp
struct ParameterizedEdge {
    int from_idx;
    int to_idx;
    // base_weight removed
    std::vector<double> coefficients;
};
```

---

### 7. C++ GraphBuilder Implementation (src/cpp/parameterized/graph_builder.cpp)

#### Location: Lines 60-74 (parse param_edges)

**Current:**
```cpp
// Format: [from_idx, to_idx, base_weight, x1, x2, ...]
for (const auto& edge_arr : param_edges_json) {
    ParameterizedEdge edge;
    edge.from_idx = edge_arr[0].get<int>();
    edge.to_idx = edge_arr[1].get<int>();
    edge.base_weight = edge_arr[2].get<double>();  // ← REMOVE
    edge.coefficients.reserve(param_length_);
    for (int i = 3; i < 3 + param_length_; i++) {  // ← Change to i=2
        edge.coefficients.push_back(edge_arr[i].get<double>());
    }
    param_edges_.push_back(edge);
}
```

**New:**
```cpp
// Format: [from_idx, to_idx, x1, x2, ...]
for (const auto& edge_arr : param_edges_json) {
    ParameterizedEdge edge;
    edge.from_idx = edge_arr[0].get<int>();
    edge.to_idx = edge_arr[1].get<int>();
    // No base_weight
    edge.coefficients.reserve(param_length_);
    for (int i = 2; i < 2 + param_length_; i++) {  // Start at index 2
        edge.coefficients.push_back(edge_arr[i].get<double>());
    }
    param_edges_.push_back(edge);
}
```

#### Location: Lines 78-92 (parse start_param_edges)

**Current:**
```cpp
// Format: [to_idx, base_weight, x1, x2, ...]
for (const auto& edge_arr : start_param_edges_json) {
    ParameterizedEdge edge;
    edge.from_idx = -1;
    edge.to_idx = edge_arr[0].get<int>();
    edge.base_weight = edge_arr[1].get<double>();  // ← REMOVE
    edge.coefficients.reserve(param_length_);
    for (int i = 2; i < 2 + param_length_; i++) {  // ← Change to i=1
        edge.coefficients.push_back(edge_arr[i].get<double>());
    }
    start_param_edges_.push_back(edge);
}
```

**New:**
```cpp
// Format: [to_idx, x1, x2, ...]
// NOTE: This should be EMPTY after refactoring (starting edges not parameterized)
for (const auto& edge_arr : start_param_edges_json) {
    ParameterizedEdge edge;
    edge.from_idx = -1;
    edge.to_idx = edge_arr[0].get<int>();
    // No base_weight
    edge.coefficients.reserve(param_length_);
    for (int i = 1; i < 1 + param_length_; i++) {  // Start at index 1
        edge.coefficients.push_back(edge_arr[i].get<double>());
    }
    start_param_edges_.push_back(edge);
}
```

#### Location: Lines 168-178 (build param_edges)

**Current:**
```cpp
for (const auto& edge : param_edges_) {
    Vertex* from_v = vertices[edge.from_idx];
    Vertex* to_v = vertices[edge.to_idx];

    // Compute weight: base_weight + dot product
    double weight = edge.base_weight;  // ← REMOVE
    for (int i = 0; i < param_length_; i++) {
        weight += edge.coefficients[i] * theta[i];
    }

    from_v->add_edge(*to_v, weight);
}
```

**New:**
```cpp
for (const auto& edge : param_edges_) {
    Vertex* from_v = vertices[edge.from_idx];
    Vertex* to_v = vertices[edge.to_idx];

    // Compute weight: dot product only
    double weight = 0.0;
    for (int i = 0; i < param_length_; i++) {
        weight += edge.coefficients[i] * theta[i];
    }

    from_v->add_edge(*to_v, weight);
}
```

#### Location: Lines 181-191 (build start_param_edges)

**Current:**
```cpp
for (const auto& edge : start_param_edges_) {
    Vertex* to_v = vertices[edge.to_idx];

    // Compute weight: base_weight + dot product
    double weight = edge.base_weight;  // ← REMOVE
    for (int i = 0; i < param_length_; i++) {
        weight += edge.coefficients[i] * theta[i];
    }

    start->add_edge(*to_v, weight);
}
```

**New:**
```cpp
for (const auto& edge : start_param_edges_) {
    Vertex* to_v = vertices[edge.to_idx];

    // Compute weight: dot product only
    double weight = 0.0;
    for (int i = 0; i < param_length_; i++) {
        weight += edge.coefficients[i] * theta[i];
    }

    start->add_edge(*to_v, weight);
}
```

---

### 8. Python Serialization (src/phasic/__init__.py)

#### Location: Lines 1538-1540 (docstring)

**Current:**
```python
- 'param_edges': Array [from_idx, to_idx, base_weight, x1, x2, ...] (n_param_edges, param_length+3)
- 'start_param_edges': Array [to_idx, base_weight, x1, x2, ...] (n_start_param_edges, param_length+2)
```

**New:**
```python
- 'param_edges': Array [from_idx, to_idx, x1, x2, ...] (n_param_edges, param_length+2)
- 'start_param_edges': Array [to_idx, x1, x2, ...] (n_start_param_edges, param_length+1)
  NOTE: start_param_edges should be empty (starting edges are not parameterized)
```

#### Location: Lines 1697-1700

**Current:**
```python
if any(x != 0 for x in edge_state):
    # Store: [from_idx, to_idx, base_weight, x1, x2, x3, ...]
    base_weight = edge.weight()
    param_edges_list.append([from_idx, to_idx, base_weight] + edge_state)
```

**New:**
```python
if any(x != 0 for x in edge_state):
    # Store: [from_idx, to_idx, x1, x2, x3, ...]
    param_edges_list.append([from_idx, to_idx] + edge_state)
```

#### Location: Lines 1702

**Current:**
```python
param_edges = np.array(param_edges_list, dtype=np.float64) if param_edges_list else np.empty((0, param_length + 3 if param_length > 0 else 0), dtype=np.float64)
```

**New:**
```python
param_edges = np.array(param_edges_list, dtype=np.float64) if param_edges_list else np.empty((0, param_length + 2 if param_length > 0 else 0), dtype=np.float64)
```

#### Location: Lines 1723-1726

**Current:**
```python
if any(x != 0 for x in edge_state):
    # Store: [to_idx, base_weight, x1, x2, x3, ...]
    base_weight = edge.weight()
    start_param_edges_list.append([to_idx, base_weight] + edge_state)
```

**New:**
```python
if any(x != 0 for x in edge_state):
    # Store: [to_idx, x1, x2, x3, ...]
    start_param_edges_list.append([to_idx] + edge_state)
```

#### Location: Line 1728

**Current:**
```python
start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, param_length + 2 if param_length > 0 else 0), dtype=np.float64)
```

**New:**
```python
start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, param_length + 1 if param_length > 0 else 0), dtype=np.float64)
```

**IMPORTANT:** After refactoring graph builder, `start_param_edges_list` should always be empty because starting edges won't be parameterized. The serialization code will still work but will produce empty array.

---

### 9. Trace Elimination (src/phasic/trace_elimination.py)

#### Location: Lines 231-254 (TraceBuilder.add_dot method)

**Current:**
```python
def add_dot(self, coefficients: np.ndarray, base_weight: float = 0.0) -> int:
    """
    Add dot product operation: base_weight + c₁*θ₁ + c₂*θ₂ + ... + cₙ*θₙ

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient vector [c₁, c₂, ..., cₙ]
    base_weight : float, default=0.0
        Base weight to add to dot product
    """
    idx = len(self.operations)
    self.operations.append(Operation(
        op_type=OpType.DOT,
        coefficients=np.array(coefficients, dtype=np.float64),
        const_value=base_weight  # Store base_weight in const_value
    ))
    return idx
```

**New:**
```python
def add_dot(self, coefficients: np.ndarray) -> int:
    """
    Add dot product operation: c₁*θ₁ + c₂*θ₂ + ... + cₙ*θₙ

    Parameters
    ----------
    coefficients : np.ndarray
        Coefficient vector [c₁, c₂, ..., cₙ]
    """
    idx = len(self.operations)
    self.operations.append(Operation(
        op_type=OpType.DOT,
        coefficients=np.array(coefficients, dtype=np.float64),
        const_value=None  # No longer used
    ))
    return idx
```

#### Location: Lines 390 (docstring)

**Current:**
```python
- Parameterized edges have weights: w = c₁*θ₁ + c₂*θ₂ + ... + cₙ*θₙ + base_weight
```

**New:**
```python
- Parameterized edges have weights: w = c₁*θ₁ + c₂*θ₂ + ... + cₙ*θₙ
- Non-parameterized edges (including starting edges) have constant weights
```

#### Location: Lines 530-542

**Current:**
```python
edge_state = param_edge.edge_state(param_length if param_length > 0 else MAX_PARAM_TEST)
edge_state = edge_state[:param_length]
coeffs = np.array(edge_state, dtype=np.float64)

# weight = dot(coeffs, params) + base_weight
base_weight = param_edge.base_weight()

if np.allclose(coeffs, 0.0):
    # No parameterization, just constant weight
    weight_idx = builder.add_const(base_weight)
else:
    # DOT product with base weight
    weight_idx = builder.add_dot(coeffs, base_weight)
```

**New:**
```python
edge_state = param_edge.edge_state(param_length if param_length > 0 else MAX_PARAM_TEST)
edge_state = edge_state[:param_length]
coeffs = np.array(edge_state, dtype=np.float64)

# weight = dot(coeffs, params)
# Note: Starting edges are never parameterized, so won't reach this code
weight_idx = builder.add_dot(coeffs)
```

#### Location: Lines 593-605

**Current:**
```python
edge_state = param_edge.edge_state(param_length if param_length > 0 else MAX_PARAM_TEST)
edge_state = edge_state[:param_length]
coeffs = np.array(edge_state, dtype=np.float64)
base_weight = param_edge.base_weight()

# Compute weight expression
if np.allclose(coeffs, 0.0):
    weight_idx = builder.add_const(base_weight)
else:
    weight_idx = builder.add_dot(coeffs, base_weight)
```

**New:**
```python
edge_state = param_edge.edge_state(param_length if param_length > 0 else MAX_PARAM_TEST)
edge_state = edge_state[:param_length]
coeffs = np.array(edge_state, dtype=np.float64)

# Compute weight expression (no base_weight)
weight_idx = builder.add_dot(coeffs)
```

#### Location: Lines 834-840 (NumPy evaluation)

**Current:**
```python
elif op.op_type == OpType.DOT:
    # Dot product WITH base weight
    base_weight = op.const_value if op.const_value is not None else 0.0
    values[i] = base_weight + np.dot(op.coefficients, params if params is not None else np.array([]))
```

**New:**
```python
elif op.op_type == OpType.DOT:
    # Dot product only (no base_weight)
    values[i] = np.dot(op.coefficients, params if params is not None else np.array([]))
```

#### Location: Lines 1282-1289 (JAX evaluation)

**Current:**
```python
elif op.op_type == OpType.DOT:
    # Dot product WITH base weight
    base_weight = op.const_value if op.const_value is not None else 0.0
    values = values.at[i].set(
        base_weight + jnp.dot(op.coefficients, params if params is not None else jnp.array([]))
    )
```

**New:**
```python
elif op.op_type == OpType.DOT:
    # Dot product only (no base_weight)
    values = values.at[i].set(
        jnp.dot(op.coefficients, params if params is not None else jnp.array([]))
    )
```

---

## Testing Strategy

### Phase 1: Unit Tests
1. Test graph construction with empty coefficients → should create non-parameterized edges
2. Test serialization format → verify no `base_weight` in JSON
3. Test deserialization → verify correct parsing

### Phase 2: Integration Tests
1. Run existing SVGD tests with univariate case (no rewards)
2. Run existing SVGD tests with multivariate case (with rewards)
3. Verify moments and PDF correctness

### Phase 3: Trace Cache
1. Clear trace cache: `rm -rf ~/.phasic_cache/`
2. Run tests to regenerate traces
3. Verify trace format doesn't include `base_weight` in DOT operations

### Phase 4: Correctness Verification
1. Run `test_svgd_correctness_10k.py` (both univariate and multivariate)
2. Run `test_reward_moments_pdf.py`
3. Verify SVGD convergence to correct θ values

---

## Migration & Compatibility

### Breaking Changes
- ✅ JSON serialization format changes
- ✅ Trace format changes (DOT operations)
- ✅ C API struct layout changes

### Version Bump
- Current: `0.21.3`
- New: `0.22.0` (major change in 0.x series)

### Cache Invalidation
Add to package initialization (`src/phasic/__init__.py`):
```python
import os
from .config import get_config

def _check_version_and_clear_cache():
    """Clear cache if version changed (handles refactoring)"""
    cache_dir = get_config()['trace_cache_dir']
    version_file = os.path.join(cache_dir, '.version')

    current_version = "0.22.0"

    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            cached_version = f.read().strip()

        if cached_version != current_version:
            # Version changed - clear cache
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)

    # Write current version
    os.makedirs(cache_dir, exist_ok=True)
    with open(version_file, 'w') as f:
        f.write(current_version)

# Run on import
_check_version_and_clear_cache()
```

### Release Notes (0.22.0)

```markdown
## Breaking Changes

- **Serialization format change:** Parameterized edge arrays no longer include `base_weight` field
  - Old: `[from, to, base_weight, c1, c2, ...]`
  - New: `[from, to, c1, c2, ...]`
- **Trace cache invalidated:** All cached elimination traces must be regenerated
- **Internal API:** `base_weight` field removed from C structs (does not affect user-facing Python API)

## Improvements

- **Simplified model:** Starting edges are never parameterized (clearer semantics)
- **Cleaner code:** Removed workaround for handling constant-weight parameterized edges
- **Better performance:** Slightly faster trace evaluation (fewer operations)

## Migration

1. **Upgrade package:** `pip install --upgrade phasic`
2. **Clear cache:** Cache is cleared automatically on first import
3. **Recompile:** If using C API directly, recompile your code

No changes to user-facing Python API required.
```

---

## Execution Order

1. **C/C++ changes first** (bottom-up):
   - C API header (`api/c/phasic.h`)
   - C implementation (`src/c/phasic.c`)
   - C++ API (`api/cpp/phasiccpp.h`)
   - C++ GraphBuilder (`src/cpp/parameterized/graph_builder.{hpp,cpp}`)
   - Python bindings (`src/cpp/phasic_pybind.cpp`)

2. **Python changes**:
   - Serialization (`src/phasic/__init__.py`)
   - Trace elimination (`src/phasic/trace_elimination.py`)
   - Version check logic

3. **Build & Test**:
   - Rebuild C++ extension
   - Clear trace cache
   - Run test suite

4. **Version bump**:
   - Update `pyproject.toml` version
   - Update version check code
   - Update CLAUDE.md

---

## Files to Modify

1. `api/c/phasic.h`
2. `src/c/phasic.c`
3. `api/cpp/phasiccpp.h`
4. `src/cpp/parameterized/graph_builder.hpp`
5. `src/cpp/parameterized/graph_builder.cpp`
6. `src/cpp/phasic_pybind.cpp`
7. `src/phasic/__init__.py`
8. `src/phasic/trace_elimination.py`
9. `pyproject.toml` (version)
10. `CLAUDE.md` (documentation)

---

## Risk Assessment

### Low Risk
- ✅ Well-defined changes
- ✅ Comprehensive test coverage
- ✅ Automatic cache invalidation
- ✅ Clear migration path

### Medium Risk
- ⚠️ Many files affected (coordinated change required)
- ⚠️ Serialization format change (must be consistent)

### Mitigation
- Execute changes in single commit (atomic)
- Test thoroughly before committing
- Document all changes in this plan

---

## Success Criteria

1. ✅ All existing tests pass
2. ✅ SVGD convergence correct (univariate & multivariate)
3. ✅ Moments and PDF accurate
4. ✅ Trace cache regenerates correctly
5. ✅ No `base_weight` references in codebase (except comments/docs)
6. ✅ Starting edges never marked as `parameterized=true`

---

## Prompt for New Session

```
I need to execute the refactoring plan in REFACTOR_PLAN_ELIMINATE_BASE_WEIGHT.md.

This refactoring eliminates the `base_weight` field from parameterized edges by ensuring starting vertex edges are never created as parameterized. The plan includes:
- Modifying graph builder to check for empty coefficient arrays
- Removing base_weight from C/C++ structs and APIs
- Updating serialization format
- Simplifying trace elimination operations
- Version bump to 0.22.0

Please execute the plan step-by-step, following the execution order specified. Make all changes in a single commit for atomicity.

Key constraints:
- Starting edges should use add_edge() if coefficients are empty
- Only edges with non-empty coefficients should be parameterized
- All serialization format changes must be coordinated
- Test after each major section

Start with the C/C++ changes (bottom-up approach).
```
