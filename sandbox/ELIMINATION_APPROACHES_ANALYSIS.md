# Elimination Approaches in phasic: Complete Analysis

**Date**: 2025-01-01
**Version**: 0.22.0
**Purpose**: Comprehensive analysis of three elimination approaches and unification strategy

---

## Executive Summary

The phasic library currently uses **three distinct approaches** to Gaussian elimination:

1. **Direct Calls (Traditional)**: In-memory symbolic/concrete compute graphs
2. **SVGD Context (Trace-Based)**: Persistent trace recordings with disk caching
3. **Manual DAG API**: Symbolic expression trees (rarely used)

This document analyzes all three approaches and proposes **unifying Direct Calls with the SVGD trace-based system** to eliminate redundancy and improve performance.

---

## Part 1: The Three Approaches

### Approach 1: Direct Calls (Traditional)

**Context**: User calls `graph.moments()`, `graph.pdf()`, `graph.expectation()` directly

**Implementation**: `src/c/phasic.c:566-648` (`ptd_precompute_reward_compute_graph()`)

#### How It Works

**Entry Point**:
```python
g = Graph(state_length=1, parameterized=True)
v0 = g.create_vertex([0])
v1 = g.create_vertex([1])
v0.add_edge_parameterized(v1, 0.0, [2.0, 0.5])
g.update_parameterized_weights([1.0, 3.0])

# This triggers precomputation internally
moments = g.moments(2)
```

**Internal Flow** (C code):
```c
double *ptd_expected_waiting_time(struct ptd_graph *graph, double *rewards) {
    // 1. Ensure elimination is performed (lazy)
    ptd_precompute_reward_compute_graph(graph);

    // 2. Use cached reward_compute_graph for O(n) evaluation
    for (size_t j = 0; j < graph->reward_compute_graph->length; ++j) {
        struct ptd_reward_increase cmd = graph->reward_compute_graph->commands[j];
        result[cmd.from] += result[cmd.to] * cmd.multiplier;
    }
    return result;
}
```

#### Data Structures

**Parameterized Symbolic** (persistent):
```c
struct ptd_desc_reward_compute_parameterized {
    size_t length;
    struct ptd_parameterized_reward_increase *commands;
};

struct ptd_parameterized_reward_increase {
    int from;
    int to;
    struct ptd_symbolic_expression *multiplier;  // Expression tree!
};
```

**Concrete** (ephemeral, rebuilt per parameter update):
```c
struct ptd_desc_reward_compute {
    size_t length;
    struct ptd_reward_increase *commands;
};

struct ptd_reward_increase {
    int from;
    int to;
    double multiplier;  // Concrete value
};
```

#### Storage Lifecycle

**Non-Parameterized**:
1. User calls `g.moments(2)`
2. `ptd_precompute_reward_compute_graph()` checks `graph->reward_compute_graph`
3. If NULL: Eliminate graph → build concrete compute graph
4. Store in `graph->reward_compute_graph` (in-memory only)
5. All future calls reuse this (no re-elimination)

**Parameterized**:
1. User calls `g.update_parameterized_weights([1.0, 2.0])`
2. Symbolic elimination performed if needed → `graph->parameterized_reward_compute_graph`
3. Symbolic expressions evaluated with parameters → `graph->reward_compute_graph`
4. User calls `g.moments(2)` → uses concrete compute graph
5. Next parameter update: Symbolic reused, concrete rebuilt

#### Performance

**Elimination**: O(n³) once per graph structure

**Per-Parameter Evaluation**:
- Symbolic → Concrete: O(S) where S = total expression size
- Moment computation: O(n) using concrete compute graph

**Storage**: In-memory only (no disk caching)

#### Advantages
- ✅ Automatic (user doesn't manage elimination)
- ✅ Lazy (only computed when needed)
- ✅ Transparent API (no explicit elimination calls)

#### Disadvantages
- ❌ Symbolic expressions slower than traces (tree traversal overhead)
- ❌ No cross-session persistence
- ❌ Redundant with trace-based system
- ❌ Expression tree memory overhead
- ❌ Recomputed every Python session

---

### Approach 2: SVGD Context (Trace-Based)

**Context**: SVGD inference or explicit trace API usage

**Implementation**:
- Python: `src/phasic/trace_elimination.py`
- C: `src/c/phasic.c` (trace evaluation functions)
- Cache: `src/phasic/trace_cache.py`

#### How It Works

**Entry Point**:
```python
from phasic.trace_elimination import (
    record_elimination_trace,
    instantiate_from_trace
)

# 1. Record trace ONCE (O(n³))
g = Graph(callback=model_callback, parameterized=True)
trace = record_elimination_trace(g, param_length=2)

# 2. Instantiate with parameters (O(n))
theta = np.array([1.0, 2.0])
concrete_g = instantiate_from_trace(trace, theta)

# 3. Use directly (already eliminated!)
moments = concrete_g.moments(2)  # Fast!
pdf = concrete_g.pdf(1.5)        # Fast!
```

**Internal Flow**:
```python
# record_elimination_trace() in Python
1. Build temporary graph with symbolic edges
2. Call C elimination algorithm
3. Record each operation: vertex elimination, edge creation, weight computation
4. Serialize to JSON: {"operations": [...], "vertex_map": {...}, ...}
5. Save to cache: ~/.phasic_cache/traces/{hash}.json

# instantiate_from_trace() in Python
1. Load trace from cache (or use in-memory)
2. Call ptd_evaluate_trace(trace, theta, param_length) in C
3. Replay operations with concrete parameters
4. Build concrete graph with eliminated edges
5. Store trace in graph->elimination_trace for future use
6. Return Graph object with reward_compute_graph already built
```

**C Implementation** (lines 588-617):
```c
int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    if (graph->elimination_trace != NULL && graph->current_params != NULL) {
        // NEW PATH: Use trace instead of symbolic!

        // Evaluate trace with current parameters (O(n))
        struct ptd_trace_result *trace_result = ptd_evaluate_trace(
            graph->elimination_trace,
            graph->current_params,
            graph->param_length
        );

        // Build reward_compute from trace result (O(n))
        graph->reward_compute_graph = ptd_build_reward_compute_from_trace(
            trace_result,
            graph
        );

        return 0;
    }

    // Fall back to traditional symbolic path
    goto traditional_path;
}
```

#### Data Structures

**Trace** (persistent, JSON-serializable):
```c
struct ptd_elimination_trace {
    size_t n_operations;
    struct ptd_trace_operation *operations;
    size_t n_vertices;
    int *vertex_map;  // Maps original → eliminated indices
    size_t param_length;
};

struct ptd_trace_operation {
    enum { OP_ELIMINATE, OP_ADD_EDGE, OP_UPDATE_EDGE } type;
    int vertex_idx;
    int from_idx;
    int to_idx;
    struct ptd_linear_expr weight;  // Linear combo: c₀θ₀ + c₁θ₁ + ... + base
};

struct ptd_linear_expr {
    double base_weight;
    size_t n_params;
    double *coefficients;  // [c₀, c₁, ..., cₙ]
};
```

**Trace Result** (ephemeral, used to build compute graph):
```c
struct ptd_trace_result {
    size_t n_vertices;
    double *vertex_rates;      // Rate out of each vertex
    double **edge_probs;       // Transition probabilities
    int **vertex_targets;      // Target vertices
    size_t *n_targets;         // # targets per vertex
};
```

#### Storage Lifecycle

1. **Recording** (once per graph structure):
   ```python
   trace = record_elimination_trace(g, param_length=2)
   # Saves to: ~/.phasic_cache/traces/{hash}.json
   ```

2. **Cache Lookup** (subsequent uses):
   ```python
   # Automatic in pmf_from_graph()
   cache_key = hash_graph_structure(g)
   trace = load_from_cache(cache_key)  # Instant!
   ```

3. **Instantiation** (per parameter vector):
   ```python
   concrete_g = instantiate_from_trace(trace, theta)
   # - graph->elimination_trace = trace (persistent pointer)
   # - graph->current_params = theta (stored)
   # - graph->reward_compute_graph = built from trace (concrete)
   ```

4. **Direct Usage**:
   ```python
   moments = concrete_g.moments(2)
   # Calls ptd_precompute_reward_compute_graph()
   # → Detects elimination_trace exists
   # → Uses trace path (O(n) rebuild of reward_compute_graph)
   # → No symbolic elimination!
   ```

#### Performance

**Recording**: O(n³) once per graph structure

**Cache Lookup**: O(1) disk I/O (~1ms)

**Evaluation**: O(n) linear replay

**Per-Parameter Instantiation**: O(n) total
- Trace replay: O(n)
- Reward compute graph building: O(n)

**Storage**:
- Memory: O(n) for trace
- Disk: JSON file (~KB to MB depending on graph size)

#### Advantages
- ✅ **5-10× faster** than symbolic expression evaluation
- ✅ Persistent across Python sessions (disk cache)
- ✅ Compact storage (linear sequence, not expression trees)
- ✅ Explicit API (user controls when to record)
- ✅ Cache sharing (same structure = same trace)
- ✅ **Integrated with C**: trace path automatically used if available

#### Disadvantages
- ❌ Requires explicit recording step
- ❌ Two elimination systems to maintain
- ❌ Redundant with symbolic system

---

### Approach 3: Manual DAG API (Symbolic Expressions)

**Context**: Advanced users needing symbolic expression trees

**Implementation**:
- Python: `src/phasic/__init__.py:3543-3697`
- C: `src/c/phasic_symbolic.c`
- Cache: `sandbox/symbolic_cache.py` (experimental)

#### How It Works

**Entry Point**:
```python
g = Graph(state_length=1, parameterized=True)
v0 = g.create_vertex([0])
v1 = g.create_vertex([1])
v0.add_edge_parameterized(v1, 0.0, [1.0, 2.0])

# Explicit symbolic elimination
dag = g.eliminate_to_dag()  # O(n³)

# Fast instantiation
g1 = dag.instantiate([1.0, 3.0])  # O(n)
g2 = dag.instantiate([2.0, 4.0])  # O(n)
```

**Internal Flow**:
```python
# g.eliminate_to_dag()
1. Call ptd_graph_symbolic_elimination(graph.c_graph())
2. Perform elimination, storing symbolic expression trees on edges
3. Return opaque pointer to symbolic DAG struct
4. Wrap in SymbolicDAG Python object

# dag.instantiate(params)
1. Call ptd_graph_symbolic_instantiate(dag._ptr, params)
2. Traverse expression trees, evaluate with params
3. Create new graph with concrete edge weights
4. Return Graph object
```

#### Data Structures

**Symbolic Edge**:
```c
struct ptd_symbolic_edge {
    struct ptd_vertex *to;
    struct ptd_symbolic_expression *weight;  // Expression tree
};

struct ptd_symbolic_expression {
    enum { CONST, PARAM, ADD, MUL, DIV } type;
    union {
        double const_value;
        int param_index;
        struct {
            struct ptd_symbolic_expression *left;
            struct ptd_symbolic_expression *right;
        } binary_op;
    } data;
};
```

**Example Expression Tree** for `2θ₀ + 3θ₁`:
```
       ADD
      /   \
    MUL   MUL
   /  \  /  \
  2   θ₀ 3  θ₁
```

#### Performance

**Elimination**: O(n³) symbolic operations

**Instantiation**: O(S) where S = total size of all expression trees

**Storage**: O(n·m·d) where m = avg edges, d = avg expression depth

#### Advantages
- ✅ Access to symbolic expressions (useful for analysis)
- ✅ Clean API for multiple instantiations
- ✅ Language-agnostic (C API)

#### Disadvantages
- ❌ **Slower than traces** due to tree traversal
- ❌ Larger memory footprint (expression trees)
- ❌ Not actively used by high-level API
- ❌ Experimental caching system (not production-ready)
- ❌ Redundant with other systems

#### Status
- **Available but not recommended**
- Not used by `pmf_from_graph()` or SVGD
- Useful for R/Julia bindings via C API
- Consider deprecating in favor of traces

---

## Part 2: Comparison Matrix

| Feature | Direct Calls (Traditional) | SVGD Context (Trace) | Manual DAG API |
|---------|---------------------------|---------------------|----------------|
| **Trigger** | Lazy (automatic) | Explicit (user calls record) | Explicit (eliminate_to_dag) |
| **Elimination** | Symbolic expression trees | Linear operation trace | Symbolic expression trees |
| **Evaluation** | O(S) tree traversal | O(n) linear replay | O(S) tree traversal |
| **Storage** | In-memory C struct | JSON file + cache | In-memory C struct |
| **Persistence** | Session-only | Cross-session (disk) | Session-only |
| **Cache** | No | Yes (~/.phasic_cache) | Experimental |
| **Performance** | Baseline | **5-10× faster** | Same as baseline |
| **Memory** | O(n·m·d) (trees) | O(n·m) (linear) | O(n·m·d) (trees) |
| **API** | Transparent | Explicit | Explicit |
| **Used By** | graph.moments(), graph.pdf() | SVGD, pmf_from_graph() | Direct API only |
| **Status** | ✅ Production | ✅ Production (preferred) | ⚠️ Available, rarely used |

---

## Part 3: Code Locations

### Direct Calls (Traditional)

**C Implementation**:
- `src/c/phasic.c:566-648` - `ptd_precompute_reward_compute_graph()`
- `src/c/phasic.c:3885-3910` - `ptd_expected_waiting_time()` (uses reward_compute_graph)
- `src/c/phasic.c:620-636` - Symbolic path (traditional)

**C++ Wrapper**:
- `api/cpp/phasiccpp.h:204-219` - `Graph::expected_waiting_time()`
- `src/cpp/phasic_pybind.cpp:369-428` - `_moments()` helper

**Python Binding**:
- `src/cpp/phasic_pybind.cpp:1277-1309` - `.def("moments", ...)`

**Data Structures**:
- `struct ptd_desc_reward_compute` - Concrete compute graph
- `struct ptd_desc_reward_compute_parameterized` - Symbolic compute graph
- `struct ptd_symbolic_expression` - Expression trees

### SVGD Context (Trace-Based)

**Python Implementation**:
- `src/phasic/trace_elimination.py:308-550` - `record_elimination_trace()`
- `src/phasic/trace_elimination.py:552-750` - `evaluate_trace_jax()`
- `src/phasic/trace_elimination.py:1450-1580` - `instantiate_from_trace()`

**C Integration**:
- `src/c/phasic.c:588-617` - Trace path in `ptd_precompute_reward_compute_graph()`
- `src/c/phasic.c` (search "trace") - Trace evaluation functions

**Cache Management**:
- `src/phasic/trace_cache.py:15-80` - Cache directory, stats, clearing

**Data Structures**:
- `struct ptd_elimination_trace` - Recorded operations
- `struct ptd_trace_operation` - Single operation (eliminate vertex, add edge)
- `struct ptd_linear_expr` - Linear parameter combination

### Manual DAG API (Symbolic)

**Python Implementation**:
- `src/phasic/__init__.py:3543-3597` - `Graph.eliminate_to_dag()`
- `src/phasic/__init__.py:3600-3697` - `SymbolicDAG` class

**C Implementation**:
- `src/c/phasic_symbolic.c` - Symbolic elimination algorithms

**Experimental Cache**:
- `sandbox/symbolic_cache.py` - SymbolicCache class (not integrated)

---

## Part 4: Current Integration Status

### How the Three Systems Interact

**C-Level Integration** (`ptd_precompute_reward_compute_graph`):

```c
int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    // Already have concrete? Return
    if (graph->reward_compute_graph != NULL) {
        return 0;
    }

    // Parameterized graph?
    if (graph->parameterized) {
        // ✅ NEW: Check for trace first
        if (graph->elimination_trace != NULL && graph->current_params != NULL) {
            // USE TRACE PATH (SVGD Context)
            trace_result = ptd_evaluate_trace(...);
            graph->reward_compute_graph = ptd_build_reward_compute_from_trace(...);
            return 0;
        }

        // ❌ FALLBACK: Traditional symbolic path
        if (graph->parameterized_reward_compute_graph == NULL) {
            graph->parameterized_reward_compute_graph =
                ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
        }
        graph->reward_compute_graph =
            ptd_graph_build_ex_absorbation_time_comp_graph_parameterized(...);
    } else {
        // Non-parameterized: traditional path
        graph->reward_compute_graph = ptd_graph_ex_absorbation_time_comp_graph(graph);
    }

    return 0;
}
```

**Key Insight**: Trace path is **already integrated** at C level! But it's only used when:
1. `graph->elimination_trace != NULL` (user explicitly recorded trace)
2. `graph->current_params != NULL` (parameters set)

**Currently**:
- Direct calls (`g.moments()`) → Symbolic path (no trace)
- SVGD / `instantiate_from_trace()` → Trace path (fast!)

---

## Part 5: Unification Strategy

### Goal: Make Direct Calls Use Trace System

**Vision**: User calls `g.moments()` → automatically uses trace if available

**Benefits**:
- ✅ 5-10× performance improvement for direct calls
- ✅ Cross-session caching (trace saved to disk)
- ✅ Eliminate redundant symbolic system
- ✅ Unified codebase
- ✅ Backward compatible API

### Implementation Approach

#### Option A: Automatic Trace Recording (Recommended)

**Idea**: Record trace automatically on first elimination

**Changes Required**:

1. **Modify `ptd_precompute_reward_compute_graph()`** (C):
```c
int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    if (graph->parameterized) {
        // NEW: Check cache for trace
        if (graph->elimination_trace == NULL) {
            char hash[65];
            ptd_graph_content_hash(graph, hash);
            graph->elimination_trace = load_trace_from_cache(hash);
        }

        // NEW: Record trace if not cached
        if (graph->elimination_trace == NULL) {
            graph->elimination_trace = ptd_record_elimination_trace(graph);
            save_trace_to_cache(hash, graph->elimination_trace);
        }

        // Use trace path (already exists!)
        if (graph->current_params != NULL) {
            // ... trace evaluation code (already implemented)
        }
    }
}
```

2. **Add C-level cache functions**:
```c
// src/c/phasic.c
static struct ptd_elimination_trace *load_trace_from_cache(const char *hash_hex);
static bool save_trace_to_cache(const char *hash_hex, const struct ptd_elimination_trace *trace);
```

3. **Implement C-level trace recording**:
```c
// Currently Python-only (trace_elimination.py)
// Port to C: ptd_record_elimination_trace()
```

**Result**:
```python
g = Graph(callback=model, parameterized=True)
g.update_parameterized_weights([1.0, 2.0])

# First call: records trace, saves to cache
moments = g.moments(2)  # Automatic trace recording!

# Subsequent calls: uses cached trace
moments2 = g.moments(3)  # Fast!

# Next Python session: trace loaded from cache
g2 = Graph(callback=model, parameterized=True)  # Same structure
g2.update_parameterized_weights([3.0, 4.0])
moments3 = g2.moments(2)  # Uses cached trace! No re-recording!
```

**Pros**:
- ✅ Fully automatic (no API changes)
- ✅ Backward compatible
- ✅ Cross-session caching works seamlessly
- ✅ Unified behavior across all entry points

**Cons**:
- ❌ More complex C code (port Python trace recording to C)
- ❌ Cache I/O on first call (small overhead)

#### Option B: Python-Level Wrapper (Quick)

**Idea**: Intercept Python API, record trace before C call

**Changes Required**:

1. **Modify Python `Graph` class** (`__init__.py`):
```python
class Graph(_Graph):  # Wraps C++ Graph
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_recorded = False

    def moments(self, power, rewards=None):
        # Auto-record trace if parameterized
        if self.is_parameterized() and not self._trace_recorded:
            self._ensure_trace()
        return super().moments(power, rewards)

    def pdf(self, time, granularity=0):
        if self.is_parameterized() and not self._trace_recorded:
            self._ensure_trace()
        return super().pdf(time, granularity)

    def _ensure_trace(self):
        from .trace_elimination import record_elimination_trace
        from .trace_cache import get_cache_dir
        import hashlib

        # Compute cache key
        serialized = self.serialize()
        cache_key = hashlib.sha256(str(serialized).encode()).hexdigest()
        cache_file = get_cache_dir() / f"{cache_key}.json"

        # Load or record trace
        if cache_file.exists():
            trace = json.load(open(cache_file))
            self._attach_trace(trace)  # New C++ binding
        else:
            trace = record_elimination_trace(self, self.param_length)
            json.dump(trace, open(cache_file, 'w'))
            self._attach_trace(trace)

        self._trace_recorded = True
```

2. **Add C++ binding** (`phasic_pybind.cpp`):
```cpp
.def("_attach_trace", [](phasic::Graph &graph, py::dict trace_json) {
    // Deserialize trace from JSON
    struct ptd_elimination_trace *trace = ptd_trace_from_json(trace_json);

    // Attach to graph
    graph.c_graph()->elimination_trace = trace;
})
```

**Result**: Same user experience as Option A

**Pros**:
- ✅ Less C code (reuse existing Python implementation)
- ✅ Faster to implement
- ✅ Backward compatible

**Cons**:
- ❌ Python overhead on every call (check `_trace_recorded` flag)
- ❌ Two code paths (Python wrapper + C fallback)
- ❌ More complex debugging

#### Option C: Explicit API (Minimal Changes)

**Idea**: Add explicit method, require user opt-in

**Changes Required**:

```python
# New method
g = Graph(callback=model, parameterized=True)
g.enable_trace_caching()  # NEW: Explicit opt-in
g.update_parameterized_weights([1.0, 2.0])
moments = g.moments(2)  # Uses trace

# Or: automatic detection
g.update_parameterized_weights([1.0, 2.0])
moments = g.moments(2)  # Check cache, use trace if available
```

**Pros**:
- ✅ Minimal changes
- ✅ User control

**Cons**:
- ❌ Not automatic (breaks vision)
- ❌ API change (backward incompatible if required)

---

### Recommendation: **Option A** (Automatic Trace Recording)

**Rationale**:
1. **Best user experience**: Fully automatic, no API changes
2. **Maximum performance**: All parameterized graphs benefit
3. **Unified codebase**: Single elimination system
4. **Cross-session caching**: Persistent traces benefit all users
5. **Future-proof**: Scales to large models

**Implementation Plan**: See Part 6

---

## Part 6: Implementation Plan for Unification

### Phase 1: Port Python Trace Recording to C

**Goal**: Make `ptd_record_elimination_trace()` available in C

**Tasks**:
1. Create `src/c/phasic_trace.c` with C implementation
2. Port `record_elimination_trace()` from Python to C
3. Add JSON serialization: `ptd_trace_to_json()`, `ptd_trace_from_json()`
4. Test: C-recorded traces match Python-recorded traces

**Estimated Effort**: 2-3 days

### Phase 2: Implement C-Level Trace Cache

**Goal**: Cache I/O in C (no Python dependency)

**Tasks**:
1. Add cache functions to `src/c/phasic.c`:
   - `load_trace_from_cache(const char *hash_hex)`
   - `save_trace_to_cache(const char *hash_hex, struct ptd_elimination_trace *trace)`
2. Use same directory: `~/.phasic_cache/traces/`
3. Test: C cache reads Python-written traces, vice versa

**Estimated Effort**: 1 day

### Phase 3: Modify `ptd_precompute_reward_compute_graph()`

**Goal**: Automatic trace recording in C

**Tasks**:
1. Add trace recording logic (see Option A code above)
2. Ensure backward compatibility (trace path optional)
3. Add debug logging (when trace used vs symbolic)
4. Test: Direct calls use trace when available

**Estimated Effort**: 1 day

### Phase 4: Python Integration Testing

**Goal**: Verify automatic trace usage from Python

**Tasks**:
1. Test: `g.moments()` with parameterized graph
2. Verify: Trace cached to disk on first call
3. Verify: Subsequent calls use cached trace
4. Verify: Cross-session caching works
5. Benchmark: Measure performance improvement

**Estimated Effort**: 1 day

### Phase 5: Deprecation of Symbolic Path

**Goal**: Mark symbolic system as legacy

**Tasks**:
1. Add deprecation warning when symbolic path used
2. Document: "Symbolic system will be removed in v0.23.0"
3. Update CLAUDE.md: Recommend trace-based approach
4. Keep symbolic code for backward compatibility (one release cycle)

**Estimated Effort**: 0.5 days

### Phase 6: Documentation and Examples

**Goal**: Update all documentation

**Tasks**:
1. Update `CLAUDE.md` with unified approach
2. Add examples: `docs/pages/optimization/trace_caching.ipynb`
3. Update API docs: Note automatic trace recording
4. Add migration guide: Symbolic → Trace

**Estimated Effort**: 1 day

### Total Estimated Effort: **6-7 days**

---

## Part 7: Risk Analysis

### Technical Risks

**Risk 1: C Trace Recording Complexity**
- **Description**: Porting Python trace recording to C may introduce bugs
- **Mitigation**: Extensive testing, cross-validation with Python implementation
- **Severity**: Medium

**Risk 2: Cache Corruption**
- **Description**: Disk cache could become corrupted or version-incompatible
- **Mitigation**: Add version checks, cache validation, automatic fallback to symbolic
- **Severity**: Low

**Risk 3: Performance Regression**
- **Description**: Cache I/O could slow down first call
- **Mitigation**: Async cache writing, lazy loading, benchmarking
- **Severity**: Low (cache I/O ~1ms)

**Risk 4: Backward Compatibility**
- **Description**: Existing code may break if trace recording fails
- **Mitigation**: Always fallback to symbolic path on trace errors
- **Severity**: Low (fallback already implemented)

### User Impact

**Positive**:
- ✅ 5-10× faster direct calls (moments, pdf)
- ✅ Cross-session caching (no re-computation)
- ✅ No API changes required
- ✅ Unified documentation (less confusion)

**Negative**:
- ⚠️ First call slightly slower (cache write)
- ⚠️ Disk usage (~KB-MB per unique graph structure)

### Mitigation Strategy

1. **Feature flag**: `PHASIC_ENABLE_AUTO_TRACE=1` (default enabled)
2. **Fallback**: Always revert to symbolic if trace fails
3. **Monitoring**: Log when trace vs symbolic path used
4. **Testing**: Extensive unit tests, integration tests, benchmarks
5. **Documentation**: Clear explanation in CLAUDE.md

---

## Part 8: Performance Projections

### Current State (Symbolic Path)

**67-vertex parameterized graph**:
- First call: ~50ms (symbolic elimination + evaluation)
- Subsequent calls (same params): ~5ms (concrete graph reused)
- Subsequent calls (new params): ~10ms (symbolic → concrete rebuild)

**No cross-session caching**: Every Python session repeats symbolic elimination

### After Unification (Trace Path)

**67-vertex parameterized graph**:
- First call (cold cache): ~50ms (trace recording) + 1ms (cache write)
- First call (warm cache): ~1ms (cache read) + ~2ms (trace evaluation)
- Subsequent calls (same params): ~0.5ms (concrete graph reused)
- Subsequent calls (new params): ~2ms (trace replay)

**Cross-session caching**: Second Python session uses cached trace (1ms vs 50ms)

### Expected Improvements

| Scenario | Before (Symbolic) | After (Trace) | Speedup |
|----------|------------------|---------------|---------|
| First call (cold) | 50ms | 51ms | 0.98× (negligible overhead) |
| First call (warm) | 50ms | 3ms | **17×** |
| New parameters | 10ms | 2ms | **5×** |
| SVGD (1000 evals) | 10s | 2s | **5×** |

**Key Benefit**: Cross-session speedup (50ms → 3ms) **every time** after first recording

---

## Part 9: Alternative: Keep Both Systems

### Rationale

Some users might prefer symbolic expressions for:
- Symbolic analysis (derivative computation, expression inspection)
- Language bindings (R, Julia) that don't use Python cache
- Debugging (inspect expression trees)

### Hybrid Approach

**Keep both systems, but**:
1. **Default to trace** for performance
2. **Symbolic available** via explicit opt-out
3. **Document tradeoffs** clearly

**API**:
```python
# Default: trace (fast)
g = Graph(callback=model, parameterized=True)
g.moments(2)  # Uses trace

# Opt-out: symbolic (slower, but access to expressions)
g = Graph(callback=model, parameterized=True, use_symbolic=True)
g.moments(2)  # Uses symbolic system
dag = g.eliminate_to_dag()  # Access symbolic DAG
```

**Implementation**:
```c
int ptd_precompute_reward_compute_graph(struct ptd_graph *graph) {
    if (graph->parameterized) {
        // Check user preference
        if (!graph->force_symbolic && graph->elimination_trace == NULL) {
            // Try trace path...
        }

        if (graph->force_symbolic || graph->elimination_trace == NULL) {
            // Use symbolic path...
        }
    }
}
```

**Pros**:
- ✅ No breaking changes
- ✅ Power users keep symbolic access
- ✅ Default users get best performance

**Cons**:
- ❌ Maintain two systems
- ❌ More complex codebase
- ❌ Documentation burden

---

## Part 10: Recommendations

### Short-Term (v0.22.1 - next patch)

1. ✅ **Document the distinction** between trace and symbolic paths
2. ✅ **Add Python wrapper** (Option B) for immediate benefit
3. ✅ **Benchmark and validate** performance improvements
4. ✅ **No breaking changes**

**Effort**: 2-3 days
**Risk**: Low
**Benefit**: Immediate 5-10× speedup for users who adopt explicit trace API

### Medium-Term (v0.23.0 - next minor)

1. ✅ **Implement Option A** (automatic trace recording in C)
2. ✅ **Deprecate symbolic path** (mark for removal)
3. ✅ **Comprehensive testing** (unit, integration, performance)
4. ✅ **Update documentation** and examples

**Effort**: 6-7 days
**Risk**: Medium
**Benefit**: Unified system, automatic performance for all users

### Long-Term (v0.24.0 - future major)

1. ✅ **Remove symbolic system** entirely (or keep as opt-in)
2. ✅ **Optimize trace format** (binary instead of JSON?)
3. ✅ **Distributed cache** support (shared team cache)
4. ✅ **Advanced features**: cache preloading, compression, etc.

**Effort**: 3-4 days
**Risk**: Low
**Benefit**: Simplified codebase, faster compilation

---

## Conclusion

The phasic library has **three elimination approaches** that evolved organically:
1. **Direct Calls** (symbolic) - Baseline, automatic, but slower
2. **SVGD Context** (trace) - Fast, cached, but requires explicit API
3. **Manual DAG** (symbolic) - Rarely used, experimental

**Unifying Direct Calls with SVGD's trace system** will:
- Eliminate redundancy
- Improve performance by 5-10×
- Enable cross-session caching
- Simplify the codebase

**Recommended path**: Implement **Option A** (automatic trace recording in C) in v0.23.0, achieving full unification while maintaining backward compatibility.

---

## Appendix A: File Reference

### Key Files for Unification

**C Implementation**:
- `src/c/phasic.c:566-648` - `ptd_precompute_reward_compute_graph()` (modify)
- `src/c/phasic_trace.c` - NEW: C-level trace recording
- `src/c/phasic.c` - ADD: `load_trace_from_cache()`, `save_trace_to_cache()`

**Python Implementation**:
- `src/phasic/trace_elimination.py` - Reference for C port
- `src/phasic/trace_cache.py` - Cache utilities (share with C)

**Headers**:
- `api/c/phasic.h` - ADD: `ptd_record_elimination_trace()` declaration

**Tests**:
- `tests/test_trace_recording.py` - Extend to test C implementation
- `tests/test_trace_cache.py` - ADD: Test C cache I/O

**Documentation**:
- `CLAUDE.md` - Update: Document unified approach
- `docs/pages/optimization/trace_caching.ipynb` - NEW: Tutorial

---

**End of Report**
