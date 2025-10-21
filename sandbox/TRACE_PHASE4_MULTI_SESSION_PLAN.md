# Phase 4: Multi-Session Implementation Plan
# Trace-Based Elimination - C-Level Integration

**Date:** 2025-10-15
**Total Duration:** 2-3 weeks across 5 phases
**Architecture:** C-level integration preserving all API contracts

---

## Overview

This plan implements trace-based elimination at the **C level** to preserve existing API contracts across C++, R, and Python. Each phase is designed to run in a separate conversation with clear handoff documents.

### Key Architectural Points

1. **Trace replaces symbolic elimination** internally
2. **API surface unchanged** - all existing code continues to work
3. **Performance gain** - 5-10x speedup for repeated parameter updates
4. **Single implementation** - trace-based elimination only (symbolic elimination obsolete)

### Integration Point

```c
void ptd_graph_update_weight_parameterized(graph, scalars, length) {
    if (graph->elimination_trace == NULL) {
        // FIRST CALL: Record trace (one-time, O(n³))
        graph->elimination_trace = ptd_record_elimination_trace(graph);
    }

    // EVERY CALL: Evaluate trace with new parameters (O(n))
    result = ptd_evaluate_trace(graph->elimination_trace, scalars, length);

    // Build internal reward_compute_graph from trace result
    graph->parameterized_reward_compute_graph =
        ptd_build_reward_compute_from_trace(result);
}
```

---

# PHASE 1: C Data Structures and Headers

**Duration:** 2-3 days
**Conversation Focus:** Define trace structures in C headers

## Objectives

1. Add trace data structures to `api/c/phasic.h`
2. Add trace field to `struct ptd_graph`
3. Document all new structures
4. Compile successfully (no implementation yet)

## Detailed Tasks

### Task 1.1: Define Trace Operation Structures

**File:** `api/c/phasic.h`
**Location:** After existing data structures, before function declarations

```c
/**
 * Operation types for trace-based elimination
 *
 * These operations form a linear sequence that can be efficiently
 * replayed with different parameter values.
 */
enum ptd_trace_op_type {
    PTD_OP_CONST = 0,   /* Constant value */
    PTD_OP_PARAM = 1,   /* Parameter reference θ[i] */
    PTD_OP_DOT = 2,     /* Dot product: Σ(cᵢ * θᵢ) */
    PTD_OP_ADD = 3,     /* Addition: a + b */
    PTD_OP_MUL = 4,     /* Multiplication: a * b */
    PTD_OP_DIV = 5,     /* Division: a / b */
    PTD_OP_INV = 6,     /* Inverse: 1 / a */
    PTD_OP_SUM = 7      /* Sum: sum([a, b, c, ...]) */
};

/**
 * Single operation in elimination trace
 */
struct ptd_trace_operation {
    enum ptd_trace_op_type op_type;

    /* For CONST */
    double const_value;

    /* For PARAM */
    size_t param_idx;

    /* For DOT (optimized linear combination) */
    double *coefficients;           /* Coefficient array */
    size_t coefficients_length;     /* Length of coefficient array */

    /* For binary/unary operations */
    size_t *operands;               /* Indices of operand operations */
    size_t operands_length;         /* Number of operands */
};

/**
 * Complete elimination trace
 *
 * This structure records all operations needed to eliminate a graph,
 * enabling fast replay with different parameter values.
 */
struct ptd_elimination_trace {
    /* Operation sequence */
    struct ptd_trace_operation *operations;
    size_t operations_length;

    /* Vertex rate mappings (vertex_idx → operation_idx) */
    size_t *vertex_rates;

    /* Edge probability mappings (vertex_idx → [operation_idx]) */
    size_t **edge_probs;
    size_t *edge_probs_lengths;

    /* Target vertex mappings (vertex_idx → [target_vertex_idx]) */
    size_t **vertex_targets;
    size_t *vertex_targets_lengths;

    /* Vertex states (copied from graph) */
    int **states;
    size_t state_length;

    /* Metadata */
    size_t starting_vertex_idx;
    size_t n_vertices;
    size_t param_length;
    bool is_discrete;
};

/**
 * Result of trace evaluation
 *
 * Contains evaluated vertex rates and edge probabilities for
 * specific parameter values.
 */
struct ptd_trace_result {
    double *vertex_rates;              /* Array[n_vertices] */
    double **edge_probs;               /* Array[n_vertices][n_edges] */
    size_t *edge_probs_lengths;        /* Array[n_vertices] */
    size_t **vertex_targets;           /* Array[n_vertices][n_edges] */
    size_t *vertex_targets_lengths;    /* Array[n_vertices] */
    size_t n_vertices;
};
```

### Task 1.2: Add Trace Field to Graph Structure

**File:** `api/c/phasic.h`
**Modify:** `struct ptd_graph`

```c
struct ptd_graph {
    /* ... existing fields ... */
    size_t vertices_length;
    struct ptd_vertex **vertices;
    struct ptd_vertex *starting_vertex;
    size_t state_length;
    size_t param_length;
    bool parameterized;
    struct ptd_desc_reward_compute *reward_compute_graph;
    bool was_dph;

    /* NEW: Trace-based elimination (NULL until first parameter update) */
    struct ptd_elimination_trace *elimination_trace;
};
```

### Task 1.3: Declare New Functions

**File:** `api/c/phasic.h`
**Location:** After existing function declarations

```c
/**
 * Record elimination trace from parameterized graph
 *
 * Performs graph elimination while recording all arithmetic operations
 * in a linear sequence. The trace can be efficiently replayed with
 * different parameter values.
 *
 * @param graph Parameterized graph
 * @return Elimination trace, or NULL on error
 *
 * Time complexity: O(n³) one-time cost
 * Space complexity: O(n²) for trace storage
 */
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
);

/**
 * Evaluate elimination trace with concrete parameter values
 *
 * Executes the recorded operation sequence with given parameters
 * to produce vertex rates and edge probabilities.
 *
 * @param trace Elimination trace
 * @param params Parameter array
 * @param params_length Length of parameter array
 * @return Trace evaluation result, or NULL on error
 *
 * Time complexity: O(n) where n = number of operations
 */
struct ptd_trace_result *ptd_evaluate_trace(
    const struct ptd_elimination_trace *trace,
    const double *params,
    size_t params_length
);

/**
 * Build reward compute graph from trace evaluation result
 *
 * Converts trace evaluation results into the internal reward_compute_graph
 * structure used by pdf/moment computations.
 *
 * @param result Trace evaluation result
 * @param graph Graph structure (for vertex references)
 * @return Reward compute graph, or NULL on error
 */
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *result,
    struct ptd_graph *graph
);

/**
 * Destroy elimination trace and free all memory
 */
void ptd_elimination_trace_destroy(struct ptd_elimination_trace *trace);

/**
 * Destroy trace evaluation result and free all memory
 */
void ptd_trace_result_destroy(struct ptd_trace_result *result);
```

### Task 1.4: Verify Compilation

```bash
# Compile headers only (no implementation yet)
cd build
cmake ..
make -j4

# Should compile successfully with new structures defined
```

## Deliverables

- [ ] `api/c/phasic.h` updated with trace structures
- [ ] `struct ptd_graph` has `elimination_trace` field
- [ ] All new functions declared in header
- [ ] Code compiles successfully
- [ ] No warnings from compiler

## Status Document

At end of Phase 1, create: `TRACE_PHASE4_1_STRUCTURES_STATUS.md`

**Template:**
```markdown
# Phase 4.1: C Data Structures - Status Report

**Date:** [DATE]
**Status:** ✅ COMPLETE / ⚠️ ISSUES / ❌ BLOCKED

## Completed Tasks
- [ ] Trace operation structures defined
- [ ] Graph structure modified with trace field
- [ ] Function declarations added
- [ ] Code compiles successfully

## Issues Encountered
[List any issues]

## Changes Made
- File: api/c/phasic.h
  - Lines added: [ESTIMATE]
  - New structures: 4 (ptd_trace_op_type, ptd_trace_operation, ptd_elimination_trace, ptd_trace_result)
  - Modified structures: 1 (ptd_graph)
  - New functions: 5 declarations

## Verification
```bash
# Compilation test
cd build && cmake .. && make -j4
# Result: [SUCCESS/FAILURE]
```

## Next Phase
Phase 4.2: Trace Recording Implementation
See: TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase 2
```

---

# PHASE 2: Trace Recording Implementation

**Duration:** 3-4 days
**Conversation Focus:** Implement trace recording in C

## Prompt to Start Phase 2

```
I am implementing Phase 4.2 of the trace-based elimination system for PtDAlgorithms.

Context:
- Phase 4.1 is complete (see TRACE_PHASE4_1_STRUCTURES_STATUS.md)
- C data structures are defined in api/c/phasic.h
- Graph structure has elimination_trace field

Task:
Implement ptd_record_elimination_trace() and helper functions in src/c/phasic.c

Reference:
- See TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase 2
- See TRACE_PHASE4_C_INTEGRATION_PLAN.md for algorithm details

Please implement the trace recording functionality following the plan.
```

## Objectives

1. Implement `ptd_record_elimination_trace()` function
2. Implement helper functions for building trace
3. Test trace recording with simple graphs
4. Verify trace structure is correct

## Detailed Tasks

### Task 2.1: Implement Trace Builder Helpers

**File:** `src/c/phasic.c`
**Location:** Near other helper functions

```c
/**
 * Helper: Add CONST operation to trace
 * Uses constant caching to reduce operation count
 */
static size_t add_const_to_trace(
    struct ptd_elimination_trace *trace,
    double value
) {
    // TODO: Check if this constant already exists in trace
    // For now, just add it

    // Ensure capacity
    // ... realloc if needed ...

    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_CONST;
    op->const_value = value;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;
    op->operands = NULL;
    op->operands_length = 0;

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
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_DOT;

    // Copy coefficients
    op->coefficients = malloc(coefficients_length * sizeof(double));
    memcpy(op->coefficients, coefficients,
           coefficients_length * sizeof(double));
    op->coefficients_length = coefficients_length;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->operands = NULL;
    op->operands_length = 0;

    return idx;
}

/**
 * Helper: Add ADD operation to trace
 */
static size_t add_add_to_trace(
    struct ptd_elimination_trace *trace,
    size_t left_idx,
    size_t right_idx
) {
    size_t idx = trace->operations_length++;
    struct ptd_trace_operation *op = &trace->operations[idx];

    op->op_type = PTD_OP_ADD;

    op->operands = malloc(2 * sizeof(size_t));
    op->operands[0] = left_idx;
    op->operands[1] = right_idx;
    op->operands_length = 2;

    op->const_value = 0.0;
    op->param_idx = 0;
    op->coefficients = NULL;
    op->coefficients_length = 0;

    return idx;
}

// Similar for: add_mul_to_trace, add_div_to_trace, add_inv_to_trace, add_sum_to_trace
```

### Task 2.2: Implement Trace Recording

**File:** `src/c/phasic.c`

```c
struct ptd_elimination_trace *ptd_record_elimination_trace(
    struct ptd_graph *graph
) {
    if (!graph->parameterized) {
        return NULL;  // Only for parameterized graphs
    }

    // Allocate trace structure
    struct ptd_elimination_trace *trace = malloc(sizeof(*trace));
    if (trace == NULL) {
        return NULL;
    }

    // Initialize metadata
    trace->n_vertices = graph->vertices_length;
    trace->state_length = graph->state_length;
    trace->param_length = graph->param_length;
    trace->is_discrete = graph->was_dph;
    trace->starting_vertex_idx = 0;  // TODO: find actual starting vertex

    // Allocate operations array (start with capacity)
    size_t operations_capacity = 1000;
    trace->operations = malloc(operations_capacity * sizeof(struct ptd_trace_operation));
    trace->operations_length = 0;

    // Allocate vertex mappings
    trace->vertex_rates = malloc(trace->n_vertices * sizeof(size_t));
    trace->edge_probs = malloc(trace->n_vertices * sizeof(size_t*));
    trace->edge_probs_lengths = calloc(trace->n_vertices, sizeof(size_t));
    trace->vertex_targets = malloc(trace->n_vertices * sizeof(size_t*));
    trace->vertex_targets_lengths = calloc(trace->n_vertices, sizeof(size_t));

    // Copy vertex states
    trace->states = malloc(trace->n_vertices * sizeof(int*));
    for (size_t i = 0; i < trace->n_vertices; i++) {
        trace->states[i] = malloc(trace->state_length * sizeof(int));
        if (graph->vertices[i]->state != NULL) {
            memcpy(trace->states[i], graph->vertices[i]->state,
                   trace->state_length * sizeof(int));
        }
    }

    // PHASE 1: Compute vertex rates
    for (size_t i = 0; i < graph->vertices_length; i++) {
        struct ptd_vertex *v = graph->vertices[i];

        if (v->edges_length == 0) {
            // Absorbing state: rate = 0
            trace->vertex_rates[i] = add_const_to_trace(trace, 0.0);
        } else {
            // rate = 1 / sum(edge_weights)
            size_t *weight_indices = malloc(v->edges_length * sizeof(size_t));

            for (size_t j = 0; j < v->edges_length; j++) {
                struct ptd_edge *edge = v->edges[j];

                if (edge->parameterized) {
                    struct ptd_edge_parameterized *param_edge =
                        (struct ptd_edge_parameterized*)edge;

                    // Extract coefficients from param_edge->state
                    double *coeffs = param_edge->state;
                    size_t coeffs_len = graph->param_length;

                    // Check if all coefficients are zero
                    bool all_zero = true;
                    for (size_t k = 0; k < coeffs_len; k++) {
                        if (fabs(coeffs[k]) > 1e-15) {
                            all_zero = false;
                            break;
                        }
                    }

                    if (all_zero) {
                        // No parameterization, just use base weight
                        weight_indices[j] = add_const_to_trace(trace, param_edge->weight);
                    } else {
                        // DOT product: c₁*θ₁ + c₂*θ₂ + ...
                        size_t dot_idx = add_dot_to_trace(trace, coeffs, coeffs_len);

                        // Add base weight if non-zero
                        if (fabs(param_edge->weight) > 1e-15) {
                            size_t base_idx = add_const_to_trace(trace, param_edge->weight);
                            weight_indices[j] = add_add_to_trace(trace, base_idx, dot_idx);
                        } else {
                            weight_indices[j] = dot_idx;
                        }
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
    // TODO: Implement (similar to Python version)

    // PHASE 3: Elimination loop
    // TODO: Implement (similar to Python version)

    return trace;
}
```

### Task 2.3: Implement Destruction Functions

```c
void ptd_elimination_trace_destroy(struct ptd_elimination_trace *trace) {
    if (trace == NULL) {
        return;
    }

    // Free operations
    for (size_t i = 0; i < trace->operations_length; i++) {
        struct ptd_trace_operation *op = &trace->operations[i];
        if (op->coefficients != NULL) {
            free(op->coefficients);
        }
        if (op->operands != NULL) {
            free(op->operands);
        }
    }
    free(trace->operations);

    // Free vertex mappings
    free(trace->vertex_rates);

    for (size_t i = 0; i < trace->n_vertices; i++) {
        if (trace->edge_probs[i] != NULL) {
            free(trace->edge_probs[i]);
        }
        if (trace->vertex_targets[i] != NULL) {
            free(trace->vertex_targets[i]);
        }
    }
    free(trace->edge_probs);
    free(trace->edge_probs_lengths);
    free(trace->vertex_targets);
    free(trace->vertex_targets_lengths);

    // Free states
    for (size_t i = 0; i < trace->n_vertices; i++) {
        free(trace->states[i]);
    }
    free(trace->states);

    free(trace);
}
```

### Task 2.4: Simple Test

**File:** `test/test_trace_c.c` (create new file)

```c
#include "phasic.h"
#include <stdio.h>
#include <assert.h>

void test_record_trace_simple() {
    printf("Testing trace recording on simple graph...\n");

    // Build simple graph
    struct ptd_graph *g = ptd_graph_create(1);
    g->parameterized = true;
    g->param_length = 1;

    // Add vertices and edges manually
    // ... build simple test graph ...

    // Record trace
    struct ptd_elimination_trace *trace = ptd_record_elimination_trace(g);

    assert(trace != NULL);
    assert(trace->n_vertices > 0);
    assert(trace->operations_length > 0);

    printf("  ✓ Trace recorded: %zu vertices, %zu operations\n",
           trace->n_vertices, trace->operations_length);

    ptd_elimination_trace_destroy(trace);
    ptd_graph_destroy(g);
}

int main() {
    printf("\n=== Phase 4.2: Trace Recording Tests ===\n\n");
    test_record_trace_simple();
    printf("\n=== All tests passed ===\n\n");
    return 0;
}
```

## Deliverables

- [ ] Helper functions implemented (add_const, add_dot, add_add, etc.)
- [ ] `ptd_record_elimination_trace()` implemented (at least Phase 1)
- [ ] `ptd_elimination_trace_destroy()` implemented
- [ ] Simple test passes
- [ ] No memory leaks (basic valgrind check)

## Status Document

Create: `TRACE_PHASE4_2_RECORDING_STATUS.md`

**Template:**
```markdown
# Phase 4.2: Trace Recording - Status Report

**Date:** [DATE]
**Status:** ✅ COMPLETE / ⚠️ PARTIAL / ❌ BLOCKED

## Completed Tasks
- [ ] Helper functions implemented
- [ ] ptd_record_elimination_trace() Phase 1 (vertex rates)
- [ ] ptd_record_elimination_trace() Phase 2 (edge probs)
- [ ] ptd_record_elimination_trace() Phase 3 (elimination loop)
- [ ] Destruction function implemented
- [ ] Basic test passes

## Implementation Details
- Lines of code added: ~[ESTIMATE]
- Functions implemented: [COUNT]
- Test results: [PASS/FAIL]

## Known Issues
[List incomplete parts or issues]

## Memory Check
```bash
valgrind --leak-check=full ./test/test_trace_c
# Result: [SUMMARY]
```

## Next Phase
Phase 4.3: Trace Evaluation Implementation
```

---

# PHASE 3: Trace Evaluation Implementation

**Duration:** 2-3 days
**Conversation Focus:** Implement trace evaluation

## Prompt to Start Phase 3

```
I am implementing Phase 4.3 of the trace-based elimination system for PtDAlgorithms.

Context:
- Phase 4.1 complete: C structures defined
- Phase 4.2 complete: Trace recording implemented (see TRACE_PHASE4_2_RECORDING_STATUS.md)

Task:
Implement ptd_evaluate_trace() and ptd_build_reward_compute_from_trace() in src/c/phasic.c

Reference:
- See TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase 3
- See TRACE_PHASE4_C_INTEGRATION_PLAN.md for algorithm details

Please implement the trace evaluation functionality.
```

## Objectives

1. Implement `ptd_evaluate_trace()` function
2. Implement `ptd_build_reward_compute_from_trace()` function
3. Test evaluation produces correct results
4. Verify reward compute graph is correct format

## Detailed Tasks

### Task 3.1: Implement Trace Evaluation

**File:** `src/c/phasic.c`

```c
struct ptd_trace_result *ptd_evaluate_trace(
    const struct ptd_elimination_trace *trace,
    const double *params,
    size_t params_length
) {
    // Validate parameters
    if (trace == NULL || params == NULL) {
        return NULL;
    }

    if (params_length != trace->param_length) {
        // Parameter count mismatch
        return NULL;
    }

    // Allocate value array for all operations
    double *values = calloc(trace->operations_length, sizeof(double));
    if (values == NULL) {
        return NULL;
    }

    // Execute operations in order
    for (size_t i = 0; i < trace->operations_length; i++) {
        const struct ptd_trace_operation *op = &trace->operations[i];

        switch (op->op_type) {
            case PTD_OP_CONST:
                values[i] = op->const_value;
                break;

            case PTD_OP_PARAM:
                if (op->param_idx < params_length) {
                    values[i] = params[op->param_idx];
                }
                break;

            case PTD_OP_DOT:
                // Dot product: Σ(cᵢ * θᵢ)
                values[i] = 0.0;
                for (size_t j = 0; j < op->coefficients_length && j < params_length; j++) {
                    values[i] += op->coefficients[j] * params[j];
                }
                break;

            case PTD_OP_ADD:
                if (op->operands_length >= 2) {
                    values[i] = values[op->operands[0]] + values[op->operands[1]];
                }
                break;

            case PTD_OP_MUL:
                if (op->operands_length >= 2) {
                    values[i] = values[op->operands[0]] * values[op->operands[1]];
                }
                break;

            case PTD_OP_DIV:
                if (op->operands_length >= 2) {
                    double denominator = values[op->operands[1]];
                    if (fabs(denominator) > 1e-15) {
                        values[i] = values[op->operands[0]] / denominator;
                    } else {
                        values[i] = 0.0;  // Handle division by zero
                    }
                }
                break;

            case PTD_OP_INV:
                if (op->operands_length >= 1) {
                    double val = values[op->operands[0]];
                    if (fabs(val) > 1e-15) {
                        values[i] = 1.0 / val;
                    } else {
                        values[i] = 0.0;  // Handle inverse of zero
                    }
                }
                break;

            case PTD_OP_SUM:
                values[i] = 0.0;
                for (size_t j = 0; j < op->operands_length; j++) {
                    values[i] += values[op->operands[j]];
                }
                break;
        }
    }

    // Allocate result structure
    struct ptd_trace_result *result = malloc(sizeof(*result));
    if (result == NULL) {
        free(values);
        return NULL;
    }

    result->n_vertices = trace->n_vertices;

    // Extract vertex rates
    result->vertex_rates = malloc(trace->n_vertices * sizeof(double));
    for (size_t i = 0; i < trace->n_vertices; i++) {
        result->vertex_rates[i] = values[trace->vertex_rates[i]];
    }

    // Extract edge probabilities
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

void ptd_trace_result_destroy(struct ptd_trace_result *result) {
    if (result == NULL) {
        return;
    }

    free(result->vertex_rates);

    for (size_t i = 0; i < result->n_vertices; i++) {
        if (result->edge_probs[i] != NULL) {
            free(result->edge_probs[i]);
        }
        if (result->vertex_targets[i] != NULL) {
            free(result->vertex_targets[i]);
        }
    }

    free(result->edge_probs);
    free(result->edge_probs_lengths);
    free(result->vertex_targets);
    free(result->vertex_targets_lengths);

    free(result);
}
```

### Task 3.2: Implement Reward Compute Graph Builder

```c
struct ptd_desc_reward_compute *ptd_build_reward_compute_from_trace(
    const struct ptd_trace_result *result,
    struct ptd_graph *graph
) {
    // This function converts trace_result into the internal
    // ptd_desc_reward_compute structure that pdf/moment functions expect

    // TODO: Study existing reward_compute_graph structure
    // TODO: Map trace results to that structure

    // For now, return NULL (will implement after studying structure)
    return NULL;
}
```

### Task 3.3: Add Tests

**File:** `test/test_trace_c.c`

```c
void test_evaluate_trace() {
    printf("Testing trace evaluation...\n");

    // Build and record trace
    struct ptd_graph *g = ptd_graph_create(1);
    g->parameterized = true;
    g->param_length = 1;

    // ... build simple graph with known result ...

    struct ptd_elimination_trace *trace = ptd_record_elimination_trace(g);
    assert(trace != NULL);

    // Evaluate with test parameters
    double params[] = {2.0};
    struct ptd_trace_result *result = ptd_evaluate_trace(trace, params, 1);

    assert(result != NULL);
    assert(result->n_vertices == trace->n_vertices);
    assert(result->vertex_rates != NULL);

    // Verify result values (TODO: add specific checks)
    printf("  ✓ Evaluation successful: %zu vertices evaluated\n",
           result->n_vertices);

    ptd_trace_result_destroy(result);
    ptd_elimination_trace_destroy(trace);
    ptd_graph_destroy(g);
}
```

## Deliverables

- [ ] `ptd_evaluate_trace()` implemented
- [ ] `ptd_trace_result_destroy()` implemented
- [ ] `ptd_build_reward_compute_from_trace()` stubbed (full impl in Phase 4)
- [ ] Evaluation test passes
- [ ] No memory leaks

## Status Document

Create: `TRACE_PHASE4_3_EVALUATION_STATUS.md`

---

# PHASE 4: Integration with Update Function

**Duration:** 3-4 days
**Conversation Focus:** Integrate trace into ptd_graph_update_weight_parameterized()

## Prompt to Start Phase 4

```
I am implementing Phase 4.4 of the trace-based elimination system for PtDAlgorithms.

Context:
- Phase 4.1-4.3 complete (see status documents)
- Trace recording and evaluation working

Task:
Integrate trace system into ptd_graph_update_weight_parameterized() function.
This function should:
1. Record trace on first call (if graph->elimination_trace == NULL)
2. Evaluate trace with new parameters
3. Build reward_compute_graph from trace result

Reference:
- See TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase 4
- See existing ptd_graph_update_weight_parameterized() implementation

Please implement the integration.
```

## Objectives

1. Modify `ptd_graph_update_weight_parameterized()` to use traces
2. Implement `ptd_build_reward_compute_from_trace()` fully
3. Test with existing API (Python, C++)
4. Verify performance improvement

## Detailed Tasks

### Task 4.1: Study Existing Implementation

**File:** `src/c/phasic.c`

```c
// Find and study the current implementation
void ptd_graph_update_weight_parameterized(
    struct ptd_graph *graph,
    double *scalars,
    size_t scalars_length
) {
    // Study how it currently works
    // Study what reward_compute_graph structure looks like
    // Understand how pdf/moment functions use it
}
```

### Task 4.2: Implement Reward Compute Builder

(Full implementation based on understanding from Task 4.1)

### Task 4.3: Modify Update Function

```c
void ptd_graph_update_weight_parameterized(
    struct ptd_graph *graph,
    double *scalars,
    size_t scalars_length
) {
    if (!graph->parameterized) {
        return;  // Nothing to do
    }

    // Validate parameters
    if (scalars == NULL || scalars_length != graph->param_length) {
        fprintf(stderr, "Error: Invalid parameters\n");
        return;
    }

    // Record elimination trace if not already done (ONE-TIME, O(n³))
    if (graph->elimination_trace == NULL) {
        graph->elimination_trace = ptd_record_elimination_trace(graph);

        if (graph->elimination_trace == NULL) {
            fprintf(stderr, "Error: Failed to record elimination trace\n");
            return;
        }
    }

    // Evaluate trace with new parameters (O(n) per evaluation)
    struct ptd_trace_result *result = ptd_evaluate_trace(
        graph->elimination_trace,
        scalars,
        scalars_length
    );

    if (result == NULL) {
        fprintf(stderr, "Error: Failed to evaluate trace\n");
        return;
    }

    // Build reward compute graph from trace result
    if (graph->reward_compute_graph != NULL) {
        ptd_desc_reward_compute_destroy(graph->reward_compute_graph);
        graph->reward_compute_graph = NULL;
    }

    graph->reward_compute_graph = ptd_build_reward_compute_from_trace(
        result,
        graph
    );

    ptd_trace_result_destroy(result);

    if (graph->reward_compute_graph == NULL) {
        fprintf(stderr, "Error: Failed to build reward compute graph\n");
    }
}
```

### Task 4.4: Test API Preservation

**Python test:**
```python
def test_api_unchanged():
    from phasic import Graph
    import numpy as np

    # Build parameterized graph
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    start.add_edge_parameterized(v1, 0.0, [2.0])

    # Update weights (should use trace internally)
    g.update_weights_parameterized([3.0])

    # Compute PDF
    times = np.array([0.5, 1.0])
    pdf = g.pdf(times, 100)

    assert pdf.shape == (2,)
    assert np.all(pdf > 0)

    # Update again (should reuse trace)
    g.update_weights_parameterized([1.5])
    pdf2 = g.pdf(times, 100)

    assert not np.allclose(pdf, pdf2)

    print("✓ Python API works with trace system")
```

## Deliverables

- [ ] `ptd_build_reward_compute_from_trace()` fully implemented
- [ ] `ptd_graph_update_weight_parameterized()` modified
- [ ] Python API tests pass
- [ ] C++ API tests pass
- [ ] Performance improvement verified

## Status Document

Create: `TRACE_PHASE4_4_INTEGRATION_STATUS.md`

---

# PHASE 5: Performance Validation and Documentation

**Duration:** 2-3 days
**Conversation Focus:** Benchmark and document

## Prompt to Start Phase 5

```
I am implementing Phase 4.5 (final phase) of the trace-based elimination system.

Context:
- Phase 4.1-4.4 complete (see status documents)
- Trace system fully integrated at C level
- All APIs working

Task:
1. Benchmark trace-based elimination performance
2. Verify Phase 3 targets still met
3. Create comprehensive documentation
4. Create final status report

Reference:
- See TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase 5

Please implement benchmarking and documentation.
```

## Objectives

1. Benchmark trace-based elimination performance
2. Measure speedup for repeated parameter updates (trace reuse)
3. Verify Phase 3 targets (37v <5min, 67v <30min)
4. Create final documentation

## Detailed Tasks

### Task 5.1: Performance Benchmarks

**File:** `test/test_trace_performance.c`

```c
#include <time.h>

void benchmark_trace_performance() {
    // Build rabbit coalescent model (37 vertices, 67 vertices)
    // Time: first update (includes one-time trace recording O(n³))
    // Time: 100 subsequent updates (trace reuse, O(n) each)
    // Calculate speedup from trace reuse

    // Metrics to measure:
    // - Initial trace recording time
    // - Average trace evaluation time
    // - Total time for N parameter updates
    // - Memory usage

    // Print results table
}
```

### Task 5.2: API Validation

Run all existing test suites:
- Python: pytest
- C++: existing tests
- R: existing tests (if applicable)

### Task 5.3: Create Documentation

**Create:** `TRACE_PHASE4_FINAL_STATUS.md`

```markdown
# Phase 4: Trace-Based Elimination - Final Status Report

**Date:** [DATE]
**Status:** ✅ COMPLETE

## Overview
Implemented trace-based elimination at C level, preserving all API contracts.

## Implementation Summary
- Trace structures: [LINES] in api/c/phasic.h
- Trace functions: [LINES] in src/c/phasic.c
- Tests: [COUNT] tests passing

## Performance Results
[TABLE: trace recording time, evaluation time, speedup from reuse, memory usage]

## API Validation
- Python: ✅ All tests pass
- C++: ✅ All tests pass
- R: ✅ All tests pass

## Files Modified
- api/c/phasic.h
- src/c/phasic.c
- test/test_trace_c.c (new)

## Backward Compatibility
✅ All existing APIs work unchanged

## Next Steps
[Future enhancements]
```

## Deliverables

- [ ] Performance benchmarks complete
- [ ] All API tests passing
- [ ] Final status document created
- [ ] No memory leaks in any tests

---

# Summary of Phases

| Phase | Duration | Focus | Status Doc |
|-------|----------|-------|------------|
| 4.1 | 2-3 days | C data structures | TRACE_PHASE4_1_STRUCTURES_STATUS.md |
| 4.2 | 3-4 days | Trace recording | TRACE_PHASE4_2_RECORDING_STATUS.md |
| 4.3 | 2-3 days | Trace evaluation | TRACE_PHASE4_3_EVALUATION_STATUS.md |
| 4.4 | 3-4 days | Integration | TRACE_PHASE4_4_INTEGRATION_STATUS.md |
| 4.5 | 2-3 days | Validation & docs | TRACE_PHASE4_FINAL_STATUS.md |

**Total:** 12-19 days (2-3 weeks)

---

# Quick Start Templates

## Starting Any Phase

```
I am implementing Phase 4.[N] of the trace-based elimination system for PtDAlgorithms.

Previous phases complete: [LIST]
Status documents: [LIST FILES]

Current phase objectives: [FROM PLAN]

Please review the previous status and begin implementation following TRACE_PHASE4_MULTI_SESSION_PLAN.md - Phase [N].
```

## Checking Phase Status

```
Please review the status of Phase 4.[N]:
1. Read TRACE_PHASE4_[N]_*_STATUS.md
2. Summarize what's complete
3. Identify any issues
4. Confirm readiness for next phase
```

---

**Each phase is designed to be completed in a single conversation session with clear handoff via status documents.**
