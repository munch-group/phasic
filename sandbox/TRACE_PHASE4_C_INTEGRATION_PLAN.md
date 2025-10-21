# Phase 4: C-Level Trace Integration Plan

**Date:** 2025-10-15
**Critical**: This plan integrates trace elimination at the **C level** to preserve API contracts across C++, R, and Python.

---

## Architecture Overview

### Current API Flow (Pre-Trace)

```
User calls: graph.update_parameterized_weights([3, 2])
    ↓
C function: ptd_graph_update_weight_parameterized(graph, scalars, length)
    ↓
Sets: graph->parameterized_reward_compute_graph = NULL
    ↓
On next pdf/moment call:
    ↓
Recomputes: graph->parameterized_reward_compute_graph (expensive)
    ↓
Uses cached compute graph for fast reward computations
```

### New API Flow (With Trace)

```
User calls: graph.update_parameterized_weights([3, 2])
    ↓
C function: ptd_graph_update_weight_parameterized(graph, scalars, length)
    ↓
If graph->elimination_trace == NULL:
    - Record elimination trace (ONE-TIME, O(n³))
    - Store in graph->elimination_trace
    ↓
Evaluate trace with scalars to get:
    - vertex_rates
    - edge_probs
    - vertex_targets
    ↓
Build graph->parameterized_reward_compute_graph from trace results
    ↓
On pdf/moment calls: Use cached compute graph (FAST)
```

### Key Insight

**The trace replaces symbolic elimination** while maintaining the same internal data structure (`parameterized_reward_compute_graph`) that existing APIs depend on.

---

## C Data Structures

### New: Elimination Trace Structures

Add to `api/c/ptdalgorithms.h`:

```c
/**
 * Operation types for trace elimination
 */
enum ptd_trace_op_type {
    PTD_OP_CONST = 0,   // Constant value
    PTD_OP_PARAM = 1,   // Parameter reference θ[i]
    PTD_OP_DOT = 2,     // Dot product: Σ(cᵢ * θᵢ)
    PTD_OP_ADD = 3,     // a + b
    PTD_OP_MUL = 4,     // a * b
    PTD_OP_DIV = 5,     // a / b
    PTD_OP_INV = 6,     // 1 / a
    PTD_OP_SUM = 7      // sum([a, b, c, ...])
};

/**
 * Single operation in elimination trace
 */
struct ptd_trace_operation {
    enum ptd_trace_op_type op_type;

    // For CONST
    double const_value;

    // For PARAM
    size_t param_idx;

    // For DOT (optimized linear combination)
    double *coefficients;      // Coefficient array
    size_t coefficients_length;

    // For binary/unary operations
    size_t *operands;          // Indices of operand operations
    size_t operands_length;
};

/**
 * Complete elimination trace
 */
struct ptd_elimination_trace {
    struct ptd_trace_operation *operations;
    size_t operations_length;

    // Mapping from vertex index to operation indices
    size_t *vertex_rates;       // vertex_idx → op_idx for rate
    size_t **edge_probs;        // vertex_idx → [op_idx] for edge probs
    size_t *edge_probs_lengths; // Length of each edge_probs[i]

    size_t **vertex_targets;    // vertex_idx → [target_vertex_idx]
    size_t *vertex_targets_lengths;

    int **states;               // Vertex states
    size_t states_length;
    size_t state_length;

    size_t starting_vertex_idx;
    size_t n_vertices;
    size_t param_length;
    bool is_discrete;
};
```

### Modified: ptd_graph Structure

Add to existing `struct ptd_graph`:

```c
struct ptd_graph {
    // ... existing fields ...
    size_t vertices_length;
    struct ptd_vertex **vertices;
    struct ptd_vertex *starting_vertex;
    size_t state_length;
    size_t param_length;
    bool parameterized;
    struct ptd_desc_reward_compute *reward_compute_graph;  // EXISTING
    bool was_dph;

    // NEW: Trace-based elimination
    struct ptd_elimination_trace *elimination_trace;  // NULL until first update
};
```

---

## C Function Implementations

### 1. Record Elimination Trace (NEW)

**File**: `src/c/ptdalgorithms.c`

```c
/**
 * Record elimination trace from parameterized graph
 *
 * This performs graph elimination while recording all operations
 * in a linear trace. The trace can then be efficiently replayed
 * with different parameter values.
 */
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
) {
    if (!graph->parameterized) {
        return NULL;  // Only for parameterized graphs
    }

    struct ptd_elimination_trace *trace = malloc(sizeof(*trace));
    if (trace == NULL) {
        return NULL;
    }

    // Initialize trace builder
    // Similar to Python TraceBuilder but in C

    // Allocate operations array (start with capacity)
    size_t operations_capacity = 1000;
    trace->operations = malloc(operations_capacity * sizeof(struct ptd_trace_operation));
    trace->operations_length = 0;

    // Extract graph structure
    trace->n_vertices = graph->vertices_length;
    trace->state_length = graph->state_length;
    trace->param_length = graph->param_length;
    trace->is_discrete = graph->was_dph;

    // Allocate vertex_rates, edge_probs, vertex_targets
    trace->vertex_rates = malloc(trace->n_vertices * sizeof(size_t));
    trace->edge_probs = malloc(trace->n_vertices * sizeof(size_t*));
    trace->edge_probs_lengths = calloc(trace->n_vertices, sizeof(size_t));
    trace->vertex_targets = malloc(trace->n_vertices * sizeof(size_t*));
    trace->vertex_targets_lengths = calloc(trace->n_vertices, sizeof(size_t));

    // Copy states
    trace->states = malloc(trace->n_vertices * sizeof(int*));
    for (size_t i = 0; i < trace->n_vertices; i++) {
        trace->states[i] = malloc(trace->state_length * sizeof(int));
        memcpy(trace->states[i], graph->vertices[i]->state,
               trace->state_length * sizeof(int));
    }

    // PHASE 1: Compute vertex rates
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];

        if (v->edges_length == 0) {
            // Absorbing state: rate = 0
            trace->vertex_rates[i] = add_const_to_trace(trace, 0.0);
        } else {
            // Rate = 1 / sum(edge_weights)
            // Need to sum weights from parameterized edges

            size_t *weight_indices = malloc(v->edges_length * sizeof(size_t));

            for (size_t j = 0; j < v->edges_length; j++) {
                struct ptd_edge *edge = v->edges[j];

                if (edge->parameterized) {
                    struct ptd_edge_parameterized *param_edge =
                        (struct ptd_edge_parameterized*)edge;

                    // Extract coefficients from edge->state
                    double *coeffs = param_edge->state;
                    size_t coeffs_len = graph->param_length;

                    // Add DOT operation: c₁*θ₁ + c₂*θ₂ + ...
                    size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);

                    // Add base weight if non-zero
                    if (param_edge->weight != 0.0) {
                        size_t base_idx = add_const_to_trace(trace, param_edge->weight);
                        weight_indices[j] = add_add_to_trace(trace, base_idx, dot_idx);
                    } else {
                        weight_indices[j] = dot_idx;
                    }
                } else {
                    // Regular edge
                    weight_indices[j] = add_const_to_trace(trace, edge->weight);
                }
            }

            // Sum all weights
            size_t sum_idx = add_sum_to_trace(trace, weight_indices, v->edges_length);

            // Rate = 1 / sum
            trace->vertex_rates[i] = add_inv_to_trace(trace, sum_idx);

            free(weight_indices);
        }
    }

    // PHASE 2: Convert edges to probabilities
    // Similar to Python implementation...

    // PHASE 3: Elimination loop
    // Similar to Python implementation...

    return trace;
}

/**
 * Helper: Add CONST operation to trace
 */
static size_t add_const_to_trace(struct ptd_elimination_trace *trace, double value) {
    // Check capacity, realloc if needed
    // Add operation
    size_t idx = trace->operations_length++;
    trace->operations[idx].op_type = PTD_OP_CONST;
    trace->operations[idx].const_value = value;
    trace->operations[idx].operands = NULL;
    trace->operations[idx].operands_length = 0;
    trace->operations[idx].coefficients = NULL;
    trace->operations[idx].coefficients_length = 0;
    return idx;
}

/**
 * Helper: Add DOT operation to trace
 */
static size_t add_dot_to_trace(
    struct ptd_elimination_trace *trace,
    const double *coefficients,
    size_t coefficients_length
) {
    size_t idx = trace->operations_length++;
    trace->operations[idx].op_type = PTD_OP_DOT;

    // Copy coefficients
    trace->operations[idx].coefficients = malloc(coefficients_length * sizeof(double));
    memcpy(trace->operations[idx].coefficients, coefficients,
           coefficients_length * sizeof(double));
    trace->operations[idx].coefficients_length = coefficients_length;

    trace->operations[idx].operands = NULL;
    trace->operations[idx].operands_length = 0;

    return idx;
}

// Similar helpers for ADD, MUL, DIV, INV, SUM...
```

### 2. Evaluate Trace (NEW)

```c
/**
 * Evaluate elimination trace with concrete parameter values
 *
 * Returns evaluated vertex rates and edge probabilities
 */
struct ptd_trace_result {
    double *vertex_rates;          // Array of rates (n_vertices)
    double **edge_probs;           // Array of arrays (n_vertices × n_edges)
    size_t *edge_probs_lengths;
    size_t **vertex_targets;       // Target vertex indices
    size_t *vertex_targets_lengths;
};

struct ptd_trace_result *ptd_evaluate_trace(
    const struct ptd_elimination_trace *trace,
    const double *params,
    size_t params_length
) {
    // Validate parameters
    if (params_length != trace->param_length) {
        return NULL;
    }

    // Allocate value array for all operations
    double *values = calloc(trace->operations_length, sizeof(double));

    // Execute operations in order
    for (size_t i = 0; i < trace->operations_length; i++) {
        struct ptd_trace_operation *op = &trace->operations[i];

        switch (op->op_type) {
            case PTD_OP_CONST:
                values[i] = op->const_value;
                break;

            case PTD_OP_PARAM:
                values[i] = params[op->param_idx];
                break;

            case PTD_OP_DOT:
                // Dot product: Σ(cᵢ * θᵢ)
                values[i] = 0.0;
                for (size_t j = 0; j < op->coefficients_length; j++) {
                    values[i] += op->coefficients[j] * params[j];
                }
                break;

            case PTD_OP_ADD:
                values[i] = values[op->operands[0]] + values[op->operands[1]];
                break;

            case PTD_OP_MUL:
                values[i] = values[op->operands[0]] * values[op->operands[1]];
                break;

            case PTD_OP_DIV:
                values[i] = values[op->operands[0]] / values[op->operands[1]];
                break;

            case PTD_OP_INV:
                values[i] = 1.0 / values[op->operands[0]];
                break;

            case PTD_OP_SUM:
                values[i] = 0.0;
                for (size_t j = 0; j < op->operands_length; j++) {
                    values[i] += values[op->operands[j]];
                }
                break;
        }
    }

    // Extract results
    struct ptd_trace_result *result = malloc(sizeof(*result));

    // Vertex rates
    result->vertex_rates = malloc(trace->n_vertices * sizeof(double));
    for (size_t i = 0; i < trace->n_vertices; i++) {
        result->vertex_rates[i] = values[trace->vertex_rates[i]];
    }

    // Edge probabilities
    result->edge_probs = malloc(trace->n_vertices * sizeof(double*));
    result->edge_probs_lengths = malloc(trace->n_vertices * sizeof(size_t));
    result->vertex_targets = malloc(trace->n_vertices * sizeof(size_t*));
    result->vertex_targets_lengths = malloc(trace->n_vertices * sizeof(size_t));

    for (size_t i = 0; i < trace->n_vertices; i++) {
        size_t n_edges = trace->edge_probs_lengths[i];
        result->edge_probs_lengths[i] = n_edges;
        result->vertex_targets_lengths[i] = n_edges;

        if (n_edges > 0) {
            result->edge_probs[i] = malloc(n_edges * sizeof(double));
            result->vertex_targets[i] = malloc(n_edges * sizeof(size_t));

            for (size_t j = 0; j < n_edges; j++) {
                result->edge_probs[i][j] = values[trace->edge_probs[i][j]];
                result->vertex_targets[i][j] = trace->vertex_targets[i][j];
            }
        } else {
            result->edge_probs[i] = NULL;
            result->vertex_targets[i] = NULL;
        }
    }

    free(values);
    return result;
}
```

### 3. Build Reward Compute Graph from Trace Result (NEW)

```c
/**
 * Build parameterized_reward_compute_graph from trace evaluation results
 *
 * This creates the internal data structure that existing APIs depend on
 */
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *trace_result,
    size_t n_vertices
) {
    // Allocate reward compute graph
    struct ptd_desc_reward_compute *compute_graph =
        malloc(sizeof(*compute_graph));

    compute_graph->vertices_length = n_vertices;

    // Convert trace_result to desc_reward_compute format
    // This matches the structure that pdf/moment computations expect

    // ... implementation details to match existing reward_compute_graph structure ...

    return compute_graph;
}
```

### 4. Modified: ptd_graph_update_weight_parameterized()

**File**: `src/c/ptdalgorithms.c`
**Existing function**: Find and modify

```c
void ptd_graph_update_weight_parameterized(
    struct ptd_graph *graph,
    double *scalars,
    size_t scalars_length
) {
    if (!graph->parameterized) {
        return;  // Nothing to do
    }

    // NEW: Record elimination trace if not already done (ONE-TIME)
    if (graph->elimination_trace == NULL) {
        graph->elimination_trace = ptd_record_elimination_trace(graph);

        if (graph->elimination_trace == NULL) {
            // Trace recording failed, fall back to old method
            goto fallback_symbolic;
        }
    }

    // NEW: Evaluate trace with new parameters
    struct ptd_trace_result *result = ptd_evaluate_trace(
        graph->elimination_trace,
        scalars,
        scalars_length
    );

    if (result == NULL) {
        goto fallback_symbolic;
    }

    // NEW: Build reward compute graph from trace result
    if (graph->reward_compute_graph != NULL) {
        ptd_desc_reward_compute_destroy(graph->reward_compute_graph);
    }

    graph->reward_compute_graph = ptd_build_reward_compute_from_trace(
        result,
        graph->vertices_length
    );

    ptd_trace_result_destroy(result);
    return;

fallback_symbolic:
    // OLD: Fallback to symbolic elimination if trace fails
    // Keep existing implementation as safety net

    // ... existing symbolic elimination code ...
}
```

---

## Implementation Order

### Phase 1: C Core Implementation (Week 1)

**Day 1-2**: Data structures
- [ ] Add trace structures to `api/c/ptdalgorithms.h`
- [ ] Add `elimination_trace` field to `struct ptd_graph`
- [ ] Define helper structures (`ptd_trace_result`, etc.)

**Day 3-4**: Trace recording
- [ ] Implement `ptd_record_elimination_trace()`
- [ ] Implement helper functions (`add_const_to_trace`, `add_dot_to_trace`, etc.)
- [ ] Test with simple parameterized graphs

**Day 5**: Trace evaluation
- [ ] Implement `ptd_evaluate_trace()`
- [ ] Test evaluation produces correct results

### Phase 2: Integration (Week 1-2)

**Day 6-7**: Reward compute graph builder
- [ ] Implement `ptd_build_reward_compute_from_trace()`
- [ ] Ensure output matches existing `reward_compute_graph` format

**Day 8-9**: Integrate into update function
- [ ] Modify `ptd_graph_update_weight_parameterized()`
- [ ] Add trace recording on first call
- [ ] Use trace for subsequent updates
- [ ] Keep symbolic as fallback

**Day 10**: Memory management
- [ ] Implement `ptd_elimination_trace_destroy()`
- [ ] Implement `ptd_trace_result_destroy()`
- [ ] Update `ptd_graph_destroy()` to free trace

### Phase 3: Testing (Week 2)

**Day 11-12**: C unit tests
- [ ] Create `test/test_trace_c.c`
- [ ] Test trace recording
- [ ] Test trace evaluation
- [ ] Test reward compute graph building
- [ ] Test with various graph sizes (22, 37, 67 vertices)

**Day 13**: API validation
- [ ] Test C++ API still works (pybind tests)
- [ ] Test Python API still works (pytest)
- [ ] Test R API still works (if applicable)

**Day 14**: Performance validation
- [ ] Benchmark trace vs symbolic
- [ ] Verify 5-10x speedup for repeated updates
- [ ] Ensure Phase 3 targets still met

---

## C Testing Strategy

### Test File: `test/test_trace_c.c`

```c
#include "ptdalgorithms.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

void test_trace_record_simple() {
    printf("Test: Record trace from simple parameterized graph\n");

    // Build simple 3-vertex graph
    struct ptd_graph *g = ptd_graph_create(1);
    g->parameterized = true;
    g->param_length = 1;

    // Add vertices manually for test
    struct ptd_vertex *v0 = g->starting_vertex;
    struct ptd_vertex *v1 = ptd_vertex_create_state(g, (int[]){1});
    struct ptd_vertex *v2 = ptd_vertex_create_state(g, (int[]){2});

    // Add parameterized edge: v0 -> v1 with weight = 2.0 * θ[0]
    double coeffs[] = {2.0};
    ptd_graph_add_edge_parameterized(v0, v1, 0.0, coeffs);

    // Add regular edge: v1 -> v2
    ptd_graph_add_edge(v1, v2, 1.0);

    // Record trace
    struct ptd_elimination_trace *trace = ptd_record_elimination_trace(g);

    assert(trace != NULL);
    assert(trace->n_vertices == 3);
    assert(trace->param_length == 1);
    assert(trace->operations_length > 0);

    printf("  ✓ Trace recorded: %zu operations\n", trace->operations_length);

    ptd_elimination_trace_destroy(trace);
    ptd_graph_destroy(g);
}

void test_trace_evaluate() {
    printf("Test: Evaluate trace with parameters\n");

    // Build and record trace
    struct ptd_graph *g = ptd_graph_create(1);
    g->parameterized = true;
    g->param_length = 1;

    struct ptd_vertex *v0 = g->starting_vertex;
    struct ptd_vertex *v1 = ptd_vertex_create_state(g, (int[]){1});

    double coeffs[] = {3.0};
    ptd_graph_add_edge_parameterized(v0, v1, 0.0, coeffs);

    struct ptd_elimination_trace *trace = ptd_record_elimination_trace(g);

    // Evaluate with θ = [2.0]
    double params[] = {2.0};
    struct ptd_trace_result *result = ptd_evaluate_trace(trace, params, 1);

    assert(result != NULL);
    assert(result->vertex_rates != NULL);

    // Vertex 0 should have rate = 1/(3.0*2.0) = 1/6
    double expected_rate = 1.0 / 6.0;
    assert(fabs(result->vertex_rates[0] - expected_rate) < 1e-10);

    printf("  ✓ Evaluation correct: rate = %.6f (expected %.6f)\n",
           result->vertex_rates[0], expected_rate);

    ptd_trace_result_destroy(result);
    ptd_elimination_trace_destroy(trace);
    ptd_graph_destroy(g);
}

void test_update_parameterized_weights_uses_trace() {
    printf("Test: update_parameterized_weights uses trace\n");

    // Build parameterized graph
    struct ptd_graph *g = ptd_graph_create(1);
    g->parameterized = true;
    g->param_length = 2;

    // ... build graph with parameterized edges ...

    // First update should record trace
    double params1[] = {1.0, 2.0};
    ptd_graph_update_weight_parameterized(g, params1, 2);

    assert(g->elimination_trace != NULL);
    assert(g->reward_compute_graph != NULL);

    printf("  ✓ First update recorded trace\n");

    // Second update should reuse trace
    struct ptd_elimination_trace *trace_ptr = g->elimination_trace;

    double params2[] = {3.0, 4.0};
    ptd_graph_update_weight_parameterized(g, params2, 2);

    assert(g->elimination_trace == trace_ptr);  // Same trace
    assert(g->reward_compute_graph != NULL);    // Updated compute graph

    printf("  ✓ Second update reused trace\n");

    ptd_graph_destroy(g);
}

int main() {
    printf("\n========================================\n");
    printf("Trace-Based Elimination C Tests\n");
    printf("========================================\n\n");

    test_trace_record_simple();
    test_trace_evaluate();
    test_update_parameterized_weights_uses_trace();

    printf("\n========================================\n");
    printf("All C tests passed!\n");
    printf("========================================\n\n");

    return 0;
}
```

---

## API Validation Tests

### Python API Test

```python
def test_python_api_still_works():
    """Verify Python API unchanged after C trace integration"""
    from ptdalgorithms import Graph
    import numpy as np

    # Build parameterized graph
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    start.add_edge_parameterized(v1, 0.0, [2.0])

    # Update weights (should use trace internally)
    g.update_weights_parameterized([3.0])

    # Compute PDF (should work as before)
    times = np.array([0.5, 1.0, 1.5])
    pdf = g.pdf(times, granularity=100)

    assert pdf.shape == (3,)
    assert np.all(pdf > 0)
    assert np.all(pdf < 10)

    # Update with different params
    g.update_weights_parameterized([1.5])
    pdf2 = g.pdf(times, granularity=100)

    # Results should differ
    assert not np.allclose(pdf, pdf2)

    print("✓ Python API works correctly")
```

### C++ API Test

```cpp
void test_cpp_api_still_works() {
    // Build graph
    ptdalgorithms::Graph g(1);
    auto start = g.starting_vertex_p();
    auto v1 = g.find_or_create_vertex_p({1});

    start->add_edge_parameterized(*v1, 0.0, {2.0});

    // Update weights
    g.update_weights_parameterized({3.0});

    // Compute PDF
    double pdf = g.pdf(1.0, 100);
    assert(pdf > 0 && pdf < 10);

    // Update again
    g.update_weights_parameterized({1.5});
    double pdf2 = g.pdf(1.0, 100);

    assert(fabs(pdf - pdf2) > 1e-6);  // Should differ

    std::cout << "✓ C++ API works correctly\n";
}
```

---

## Success Criteria

### C Implementation
- [ ] All trace structures defined in C headers
- [ ] `ptd_record_elimination_trace()` working
- [ ] `ptd_evaluate_trace()` working
- [ ] `ptd_build_reward_compute_from_trace()` working
- [ ] `ptd_graph_update_weight_parameterized()` using traces
- [ ] C unit tests passing (test/test_trace_c.c)
- [ ] No memory leaks (valgrind clean)

### API Preservation
- [ ] C++ API unchanged (all existing tests pass)
- [ ] Python API unchanged (all existing tests pass)
- [ ] R API unchanged (if applicable)
- [ ] `update_weights_parameterized()` signature unchanged
- [ ] PDF/moment computations produce same results

### Performance
- [ ] First call: trace recording ~5-10ms (67 vertices)
- [ ] Subsequent calls: 5-10x faster than symbolic
- [ ] Phase 3 targets still met (37v <5min, 67v <30min)

---

## Key Design Principles

1. **Preserve API Surface**: All existing function signatures unchanged
2. **Transparent Integration**: Users don't know trace is being used
3. **Backward Compatible**: Symbolic elimination as fallback
4. **Memory Safe**: Proper allocation/deallocation of trace structures
5. **Language Neutral**: C implementation works for C++, R, Python equally

---

## Build and Test Commands

```bash
# Build C library with trace support
mkdir build && cd build
cmake ..
make

# Run C tests
./test/test_trace_c

# Run C++ tests
./test/test_cpp_api

# Run Python tests
cd ..
pytest tests/ -v

# Check for memory leaks
valgrind --leak-check=full ./test/test_trace_c
```

---

**This approach preserves all existing APIs while integrating trace-based speedups at the C level.**
