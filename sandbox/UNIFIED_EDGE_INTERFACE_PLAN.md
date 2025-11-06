# Universal Edge Parameterization Implementation Plan

## Executive Summary

Unify edge types by making ALL edges parameterized with coefficient arrays. Constant edges are represented as single-element arrays `[value]`, evaluated with default `theta=[1.0]`. This enables universal trace caching for all graphs, simplifies the codebase, and provides a foundation for SCC-level hierarchical caching.

## Core Design

### Principle
- **All edges store coefficient arrays** (minimum length 1)
- **First `add_edge()` call locks the mode** for the entire graph
- **Scalar edges:** `add_edge(v, 3.0)` → `coefficients=[3.0]`, `param_length=1`
- **Parameterized edges:** `add_edge(v, [2.0, 9.0])` → `coefficients=[2.0, 9.0]`, `param_length=2`
- **Default evaluation:** `theta=[1.0, 1.0, ...]` makes constants work automatically

### Benefits
1. ✅ Universal caching: All graphs traced and cached
2. ✅ Type safety: Cannot mix scalar/array edges in same graph
3. ✅ Early validation: Mismatches caught at edge creation
4. ✅ Simplified codebase: One edge type, one code path
5. ✅ Foundation for SCC caching: Uniform structure enables hierarchical caching

---

## Phase 1: Update C API Structures (3-4 hours)

### Files to Modify
- `api/c/phasic.h`
- `src/c/phasic.c`

### Changes

#### 1.1 Update Edge Structure (`api/c/phasic.h:121-135`)

**Remove:**
```c
struct ptd_edge {
    struct ptd_vertex *to;
    double weight;
    bool parameterized;
};

struct ptd_edge_parameterized {
    struct ptd_vertex *to;
    double weight;
    bool parameterized;
    double *state;
    size_t state_length;
    bool should_free_state;
};
```

**Replace with:**
```c
struct ptd_edge {
    struct ptd_vertex *to;
    double weight;              // Current evaluated weight
    double *coefficients;       // ALWAYS non-NULL, length = graph->param_length
    size_t coefficients_length; // Always = graph->param_length
    bool should_free_coefficients;
};
```

#### 1.2 Update Graph Structure (`api/c/phasic.h:105-119`)

**Add fields:**
```c
struct ptd_graph {
    size_t vertices_length;
    struct ptd_vertex **vertices;
    struct ptd_vertex *starting_vertex;
    size_t state_length;
    size_t param_length;        // NEW: Set by first add_edge() call
    bool parameterized;         // NEW: true if param_length > 1
    bool param_length_locked;   // NEW: true after first edge added

    struct ptd_desc_reward_compute *reward_compute_graph;
    struct ptd_desc_reward_compute_parameterized *parameterized_reward_compute_graph;
    bool was_dph;

    struct ptd_elimination_trace *elimination_trace;
    double *current_params;
};
```

#### 1.3 Update Function Signatures (`api/c/phasic.h:167-179`)

**Remove:**
```c
struct ptd_edge *ptd_graph_add_edge(
    struct ptd_vertex *from,
    struct ptd_vertex *to,
    double weight
);

struct ptd_edge_parameterized *ptd_graph_add_edge_parameterized(
    struct ptd_vertex *from,
    struct ptd_vertex *to,
    double weight,
    double *edge_state,
    size_t edge_state_length
);
```

**Replace with:**
```c
struct ptd_edge *ptd_graph_add_edge(
    struct ptd_vertex *from,
    struct ptd_vertex *to,
    double *coefficients,
    size_t coefficients_length
);
```

#### 1.4 Implement Unified Edge Creation (`src/c/phasic.c:~2324`)

**Replace both `ptd_graph_add_edge()` and `ptd_graph_add_edge_parameterized()` with:**

```c
struct ptd_edge *ptd_graph_add_edge(
    struct ptd_vertex *from,
    struct ptd_vertex *to,
    double *coefficients,
    size_t coefficients_length
) {
    if (coefficients == NULL || coefficients_length == 0) {
        snprintf((char*)ptd_err, sizeof(ptd_err),
            "ptd_graph_add_edge: coefficients cannot be NULL or empty");
        return NULL;
    }

    // VALIDATION: Check consistency with existing edges
    if (from->graph->param_length_locked) {
        if (coefficients_length != from->graph->param_length) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "Edge coefficient length mismatch: graph expects %zu parameters, got %zu. "
                "All edges in a graph must have the same coefficient length.",
                (unsigned long)from->graph->param_length,
                (unsigned long)coefficients_length);
            return NULL;
        }
    } else {
        // First edge: set graph mode
        from->graph->param_length = coefficients_length;
        from->graph->parameterized = (coefficients_length > 1);
        from->graph->param_length_locked = true;
    }

    // Create edge
    struct ptd_edge *edge = (struct ptd_edge *)malloc(sizeof(*edge));
    if (edge == NULL) {
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge");
        return NULL;
    }

    edge->to = to;
    edge->coefficients_length = coefficients_length;
    edge->coefficients = (double *)malloc(coefficients_length * sizeof(double));
    if (edge->coefficients == NULL) {
        free(edge);
        snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate edge coefficients");
        return NULL;
    }

    memcpy(edge->coefficients, coefficients, coefficients_length * sizeof(double));
    edge->should_free_coefficients = true;

    // Compute initial weight with default params (theta=[1,1,...])
    edge->weight = 0.0;
    for (size_t i = 0; i < coefficients_length; i++) {
        edge->weight += coefficients[i] * 1.0;
    }

    // Add to vertex
    from->edges_length++;
    from->edges = (struct ptd_edge **)realloc(
        from->edges,
        from->edges_length * sizeof(*from->edges)
    );
    from->edges[from->edges_length - 1] = edge;

    return edge;
}
```

#### 1.5 Update Edge Destruction (`src/c/phasic.c`)

**Find and update `ptd_edge_destroy()` or equivalent:**
```c
void ptd_edge_destroy(struct ptd_edge *edge) {
    if (edge == NULL) return;

    if (edge->should_free_coefficients && edge->coefficients != NULL) {
        free(edge->coefficients);
    }

    free(edge);
}
```

#### 1.6 Update Graph Initialization (`src/c/phasic.c:~2150`)

**In `ptd_graph_create()`:**
```c
struct ptd_graph *ptd_graph_create(size_t state_length) {
    struct ptd_graph *graph = (struct ptd_graph *)calloc(1, sizeof(*graph));
    if (graph == NULL) return NULL;

    graph->state_length = state_length;
    graph->param_length = 0;           // NEW
    graph->parameterized = false;      // NEW
    graph->param_length_locked = false; // NEW
    graph->elimination_trace = NULL;
    graph->current_params = NULL;
    // ... rest of initialization

    return graph;
}
```

#### 1.7 Update Weight Update Function (`src/c/phasic.c:~2539`)

**Rename `ptd_graph_update_weight_parameterized()` to `ptd_graph_update_weights()`:**

```c
void ptd_graph_update_weights(
    struct ptd_graph *graph,
    double *params,
    size_t params_length
) {
    double *theta;
    size_t theta_len;
    bool need_free = false;

    if (params == NULL || params_length == 0) {
        // Use default theta = [1, 1, ..., 1]
        theta_len = graph->param_length;
        if (theta_len == 0) {
            // No edges yet, nothing to do
            return;
        }
        theta = (double *)malloc(theta_len * sizeof(double));
        if (theta == NULL) {
            snprintf((char*)ptd_err, sizeof(ptd_err), "Failed to allocate default parameters");
            return;
        }
        for (size_t i = 0; i < theta_len; i++) {
            theta[i] = 1.0;
        }
        need_free = true;
    } else {
        // Validate parameter length
        if (params_length != graph->param_length) {
            snprintf((char*)ptd_err, sizeof(ptd_err),
                "Parameter length mismatch: graph expects %zu parameters, got %zu",
                (unsigned long)graph->param_length,
                (unsigned long)params_length);
            return;
        }
        theta = params;
        theta_len = params_length;
    }

    // Store current parameters
    if (graph->current_params == NULL && theta_len > 0) {
        graph->current_params = (double *)malloc(theta_len * sizeof(double));
    }
    if (graph->current_params != NULL && theta_len > 0) {
        memcpy(graph->current_params, theta, theta_len * sizeof(double));
    }

    // Record/load trace if needed (ALWAYS - for all graphs!)
    if (graph->elimination_trace == NULL) {
        struct ptd_hash_result *hash = ptd_graph_content_hash(graph);

        if (hash != NULL) {
            graph->elimination_trace = load_trace_from_cache(hash->hash_hex);
            if (graph->elimination_trace != NULL) {
                DEBUG_PRINT("INFO: loaded elimination trace from cache (%s)\n", hash->hash_hex);
            }
        }

        if (graph->elimination_trace == NULL) {
            DEBUG_PRINT("INFO: recording elimination trace...\n");
            graph->elimination_trace = ptd_record_elimination_trace(graph);

            if (graph->elimination_trace != NULL && hash != NULL) {
                save_trace_to_cache(hash->hash_hex, graph->elimination_trace);
            }
        }

        if (hash != NULL) {
            ptd_hash_destroy(hash);
        }
    }

    // Update all edge weights using direct computation or trace
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *vertex = graph->vertices[i];
        for (size_t j = 0; j < vertex->edges_length; j++) {
            struct ptd_edge *edge = vertex->edges[j];

            // Compute weight = dot(coefficients, theta)
            edge->weight = 0.0;
            for (size_t k = 0; k < edge->coefficients_length; k++) {
                edge->weight += edge->coefficients[k] * theta[k];
            }
        }
    }

    // Invalidate cached compute graphs
    if (graph->reward_compute_graph != NULL) {
        free(graph->reward_compute_graph->commands);
        free(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }

    if (graph->parameterized_reward_compute_graph != NULL) {
        ptd_parameterized_reward_compute_graph_destroy(
            graph->parameterized_reward_compute_graph
        );
        graph->parameterized_reward_compute_graph = NULL;
    }

    if (need_free) {
        free(theta);
    }
}
```

#### 1.8 Update All Edge Access Code

**Search for all occurrences of:**
- `edge->parameterized` → Check `edge->coefficients_length > 1` instead
- Casts to `ptd_edge_parameterized` → Remove, all edges are now uniform
- `ptd_graph_add_edge_parameterized()` calls → Update to `ptd_graph_add_edge()`

**Key locations:**
- Reward transformation functions
- Trace recording functions
- Graph cloning functions
- Serialization functions

### Testing Phase 1
```bash
# Compile C library
cd /Users/kmt/phasic
pixi run pip install -e . --no-build-isolation

# Test basic edge creation
python -c "
from phasic import Graph
g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v0.add_edge(v1, 3.0)
print('✓ Scalar edge creation works')
"
```

---

## Phase 2: Update C++ API Layer (2-3 hours)

### Files to Modify
- `api/cpp/phasiccpp.h`
- `src/cpp/phasiccpp.cpp`

### Changes

#### 2.1 Update Vertex Class (`api/cpp/phasiccpp.h:796-798`)

**Replace:**
```cpp
void add_edge(Vertex &to, double weight);
void add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state);
```

**With:**
```cpp
// Primary method - accepts both scalar and vector
void add_edge(Vertex &to, double weight) {
    double coeffs[] = {weight};
    struct ptd_edge *edge = ptd_graph_add_edge(vertex, to.vertex, coeffs, 1);
    if (edge == NULL) {
        throw std::runtime_error(std::string("Failed to add edge: ") + (char*)ptd_err);
    }
}

void add_edge(Vertex &to, const std::vector<double> &coefficients) {
    if (coefficients.empty()) {
        throw std::invalid_argument("Edge coefficients cannot be empty");
    }
    struct ptd_edge *edge = ptd_graph_add_edge(
        vertex, to.vertex,
        const_cast<double*>(coefficients.data()),
        coefficients.size()
    );
    if (edge == NULL) {
        throw std::runtime_error(std::string("Failed to add edge: ") + (char*)ptd_err);
    }
}

// Deprecated - keep for backward compatibility
void add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state) {
    add_edge(to, edge_state);  // Ignore weight parameter, just use coefficients
}
```

#### 2.2 Update Edge Classes (`api/cpp/phasiccpp.h:837-895`)

**Simplify Edge struct (remove ParameterizedEdge):**
```cpp
struct Edge {
private:
    Edge(struct ptd_vertex *vertex, struct ptd_edge *edge, Graph &graph) : graph(graph) {
        this->_edge = edge;
        this->_vertex = vertex;
    }

private:
    Graph &graph;
    struct ptd_vertex *_vertex;
    struct ptd_edge *_edge;

public:
    static Edge init_factory(struct ptd_vertex *vertex, struct ptd_edge *edge, Graph &graph) {
        return Edge(vertex, edge, graph);
    }

    Vertex to() {
        return Vertex(graph, _edge->to);
    }

    double weight() {
        return _edge->weight;
    }

    std::vector<double> coefficients() {
        std::vector<double> result(_edge->coefficients_length);
        for (size_t i = 0; i < _edge->coefficients_length; i++) {
            result[i] = _edge->coefficients[i];
        }
        return result;
    }

    void update_weight(double weight) {
        if (_edge->coefficients_length != 1) {
            throw std::runtime_error("Cannot update weight on parameterized edge");
        }
        _edge->coefficients[0] = weight;
        _edge->weight = weight;
    }

    friend class Vertex;
};

// Remove ParameterizedEdge entirely
```

#### 2.3 Update Vertex Methods (`src/cpp/phasiccpp.cpp`)

**Find implementations and update to call new C API:**
```cpp
void Vertex::add_edge(Vertex &to, double weight) {
    double coeffs[] = {weight};
    struct ptd_edge *edge = ptd_graph_add_edge(vertex, to.vertex, coeffs, 1);
    if (edge == NULL) {
        throw std::runtime_error(std::string("Failed to add edge: ") + (char*)ptd_err);
    }
}

void Vertex::add_edge_parameterized(Vertex &to, double weight, std::vector<double> edge_state) {
    if (edge_state.empty()) {
        throw std::invalid_argument("Edge coefficients cannot be empty");
    }
    struct ptd_edge *edge = ptd_graph_add_edge(
        vertex, to.vertex, edge_state.data(), edge_state.size()
    );
    if (edge == NULL) {
        throw std::runtime_error(std::string("Failed to add edge: ") + (char*)ptd_err);
    }
}

std::vector<Edge> Vertex::edges() {
    std::vector<Edge> result;
    for (size_t i = 0; i < vertex->edges_length; i++) {
        result.push_back(Edge::init_factory(vertex->edges[i]->to, vertex->edges[i], graph));
    }
    return result;
}

// Remove parameterized_edges() method - no longer needed
```

### Testing Phase 2
```bash
# Test C++ compilation
cd /Users/kmt/phasic
pixi run pip install -e . --no-build-isolation

# Run C++ tests if available
cd test
make testcpp
./testcpp
```

---

## Phase 3: Update Python Bindings (1-2 hours)

### Files to Modify
- `src/cpp/phasic_pybind.cpp`

### Changes

#### 3.1 Update Vertex Bindings (`src/cpp/phasic_pybind.cpp:~2864`)

**Replace:**
```cpp
.def("add_edge", &phasic::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(...)delim")
.def("ae", &phasic::Vertex::add_edge, py::arg("to"), py::arg("weight"), R"delim(...)delim")
.def("add_edge_parameterized", &phasic::Vertex::add_edge_parameterized, ...)
```

**With:**
```cpp
.def("add_edge", [](phasic::Vertex& self, phasic::Vertex& to, py::object weight_or_coeffs) {
    if (py::isinstance<py::float_>(weight_or_coeffs) || py::isinstance<py::int_>(weight_or_coeffs)) {
        // Scalar: convert to single-element array
        double weight = weight_or_coeffs.cast<double>();
        self.add_edge(to, weight);
    } else if (py::isinstance<py::list>(weight_or_coeffs) || py::isinstance<py::array>(weight_or_coeffs)) {
        // Array: pass as coefficients
        std::vector<double> coeffs = weight_or_coeffs.cast<std::vector<double>>();
        if (coeffs.empty()) {
            throw std::invalid_argument("Edge coefficients cannot be empty");
        }
        self.add_edge(to, coeffs);
    } else {
        throw std::invalid_argument(
            "add_edge() expects either a scalar (float/int) or array-like (list/ndarray) argument"
        );
    }
}, py::arg("to"), py::arg("weight_or_coeffs"), R"delim(
    Add an edge to another vertex with constant or parameterized weight.

    Parameters
    ----------
    to : Vertex
        Target vertex
    weight_or_coeffs : float or array-like
        If scalar: constant edge weight (e.g., 3.0)
        If array: coefficient vector for parameterized edge (e.g., [2.0, 9.0])

    Returns
    -------
    Edge
        The created edge

    Notes
    -----
    All edges in a graph must use the same form (all scalar or all array).
    The first call to add_edge() sets the mode for the entire graph.

    Examples
    --------
    >>> # Constant edge
    >>> v.add_edge(target, 3.0)

    >>> # Parameterized edge: weight = 2.0*theta[0] + 9.0*theta[1]
    >>> v.add_edge(target, [2.0, 9.0])
)delim")

.def("ae", [](phasic::Vertex& self, phasic::Vertex& to, py::object weight_or_coeffs) {
    // Alias for add_edge
    if (py::isinstance<py::float_>(weight_or_coeffs) || py::isinstance<py::int_>(weight_or_coeffs)) {
        double weight = weight_or_coeffs.cast<double>();
        self.add_edge(to, weight);
    } else {
        std::vector<double> coeffs = weight_or_coeffs.cast<std::vector<double>>();
        self.add_edge(to, coeffs);
    }
}, py::arg("to"), py::arg("weight_or_coeffs"), R"delim(Alias for add_edge)delim")

.def("add_edge_parameterized",
    [](phasic::Vertex& self, phasic::Vertex& to, double weight, std::vector<double> edge_state) {
        // DEPRECATED: Issue warning
        py::print("WARNING: add_edge_parameterized() is deprecated. Use add_edge() with array argument.");
        self.add_edge(to, edge_state);
    },
    py::arg("to"), py::arg("weight"), py::arg("edge_state"),
    R"delim(
    DEPRECATED: Use add_edge() with array argument instead.

    This method is kept for backward compatibility but will be removed in v1.0.
)delim")
```

---

## Phase 4-8: Additional Phases

See detailed implementation steps in sections above for:
- Phase 4: Update Trace Recording (2-3 hours)
- Phase 5: Update Normalization Functions (1-2 hours)
- Phase 6: Update Documentation & Tests (2-3 hours)
- Phase 7: Performance Validation (1 hour)
- Phase 8: Cache Verification (1 hour)

---

## Implementation Checklist

### Phase 1: C API ☐
- [ ] Update `ptd_edge` struct
- [ ] Update `ptd_graph` struct (add param_length, parameterized, param_length_locked)
- [ ] Implement unified `ptd_graph_add_edge()`
- [ ] Remove `ptd_graph_add_edge_parameterized()`
- [ ] Update edge destruction
- [ ] Update graph initialization
- [ ] Implement `ptd_graph_update_weights()`
- [ ] Remove `ptd_graph_update_weight_parameterized()`
- [ ] Update all edge access code
- [ ] Test compilation

### Phase 2: C++ API ☐
- [ ] Update `Vertex::add_edge()` methods
- [ ] Remove/deprecate `add_edge_parameterized()`
- [ ] Simplify Edge struct
- [ ] Remove ParameterizedEdge struct
- [ ] Update edge iteration methods
- [ ] Test C++ compilation

### Phase 3: Python Bindings ☐
- [ ] Update pybind11 `add_edge()` dispatcher
- [ ] Add deprecation warning for `add_edge_parameterized()`
- [ ] Simplify Edge bindings
- [ ] Remove ParameterizedEdge bindings
- [ ] Add `param_length()` and `is_parameterized()` methods
- [ ] Add `update_weights()` method
- [ ] Test Python import

### Remaining phases... (see full plan above)

---

## Timeline

**Total estimated time: 13-19 hours**

---

## Success Criteria

✅ Single `ptd_edge` struct (no separate parameterized type)
✅ `param_length_locked` prevents mixed edges
✅ Python `add_edge()` accepts scalar or array
✅ All existing tests pass
✅ Non-parameterized graphs get cached traces
✅ No performance regression (<10% overhead)
✅ Clean deprecation path for old API
✅ Documentation updated
