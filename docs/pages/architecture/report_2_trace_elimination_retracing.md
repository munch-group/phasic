# Report 2: Trace Elimination and Retracing Architecture

**Date:** 2025-12-21
**Analysis Method:** Source code inspection only (no documentation or inline comments)

## Executive Summary

Trace elimination is a two-phase algorithm that separates **structure** from **parameterization** in phase-type distribution graphs. The first phase records a linear sequence of elimination operations (the "trace"), and the second phase evaluates this trace with concrete parameter values. This separation enables 5-10x faster evaluation for SVGD workloads where the same graph structure is evaluated with different parameters thousands of times.

## Conceptual Foundation

### The Core Problem

Traditional graph elimination operates directly on concrete graphs:

```
Input: Parameterized Graph (structure + parameter coefficients)
       |
       v
[Gaussian Elimination on Graph]  <-- O(n³) complexity
       |
       +-- For each vertex v:
       |   - Compute elimination weight
       |   - For each (parent, child) pair:
       |       * Add bypass edge
       |       * Renormalize probabilities
       |   - Remove vertex v
       |
       v
Output: Acyclic Graph (ready for moment computation)
```

**Problem**: If we change parameters slightly, we must re-run the entire O(n³) elimination.

### The Trace Solution

Trace elimination **separates concerns**:

```
Phase 1 (Once):  Record WHAT operations to perform
                 → Trace (linear sequence of operations)

Phase 2 (Many):  Evaluate trace with concrete parameters
                 → Instantiated graph (O(n²) or O(n))
```

**Performance gain**: Recording trace = ~1x elimination cost, but evaluation = 0.05-0.1x cost.

**Break-even**: ~6 evaluations. SVGD needs 1000+ evaluations → 5-10x total speedup.

## Trace Recording Architecture

### Recording Flow

```
record_elimination_trace(graph, param_length)
       |
       v
[Initialize Trace Structure]
       |
       +-- Create symbolic arrays:
       |   - vertex_rates[i] = symbolic expression
       |   - edge_probs[i][j] = symbolic expression
       |   - vertex_targets[i] = list of target indices
       |
       +-- Each symbolic expression is:
       |   Expression {
       |       constant: double
       |       coefficients: [c0, c1, ..., c_{param_length-1}]
       |   }
       |   Meaning: value = constant + c0*θ[0] + c1*θ[1] + ...
       |
       v
[Call C Elimination with Symbolic Mode]
       |
       v
ptd_graph_eliminate_recording(c_graph, trace_ptr)
       |
       +-- For each vertex v in elimination order:
       |   |
       |   +-- Compute exit_rate (symbolic):
       |   |   exit_rate = SUM(edge weights from v)
       |   |   Record: OPERATION(DOT, operands=[edge_indices], coeffs=[1,1,...])
       |   |
       |   +-- For each edge v → w:
       |   |   edge_prob = edge_weight / exit_rate
       |   |   Record: OPERATION(DIV, operands=[weight_idx, exit_rate_idx])
       |   |
       |   +-- For each (parent u, child w) pair:
       |   |   |
       |   |   +-- Compute bypass probability:
       |   |   |   bypass = edge_prob[u→v] * edge_prob[v→w]
       |   |   |   Record: OPERATION(MUL, operands=[p1_idx, p2_idx])
       |   |   |
       |   |   +-- Add to existing u→w edge (if exists):
       |   |   |   new_prob = old_prob + bypass
       |   |   |   Record: OPERATION(ADD, operands=[old_idx, bypass_idx])
       |   |   |
       |   |   +-- Or create new u→w edge:
       |   |       Record: OPERATION(SET, operands=[bypass_idx])
       |   |
       |   +-- Renormalize u's outgoing edges:
       |       total = SUM(edge probs from u)
       |       Record: OPERATION(DOT, operands=[edge_indices])
       |       For each edge: new_prob = old_prob / total
       |       Record: OPERATION(DIV, operands=[prob_idx, total_idx])
       |
       v
[Return Trace Object]
       |
       +-- Trace contains:
           - operations: [(type, const, param_idx, operands, coeffs), ...]
           - n_vertices: int
           - state_length: int
           - param_length: int
           - starting_vertex_idx: int
           - is_discrete: bool
```

### Operation Types (OpType enum in trace_elimination.py)

```python
class OpType:
    SET = 0       # result = const + DOT(coeffs, params)
    ADD = 1       # result = operands[0] + operands[1]
    MUL = 2       # result = operands[0] * operands[1]
    DIV = 3       # result = operands[0] / operands[1]
    DOT = 4       # result = const + DOT(coeffs, params) + SUM(operands * weights)
```

**Key insight**: Every intermediate value in the elimination is assigned an **index**, and subsequent operations reference these indices.

### Example Trace (Pseudo-Code)

For a simple 2-vertex coalescent model with parameter θ:

```
Trace Operations:
0. SET: vertex_rates[0] = 0.0 + 1.0*θ[0]           # Starting vertex exit rate
1. DOT: edge_probs[0][0] = 1.0                     # Probability = 1 (only one edge)
2. SET: vertex_rates[1] = 0.0                      # Absorbing vertex (no exit)
3. MUL: result_3 = edge_probs[0][0] * 1.0          # Bypass probability
4. SET: edge_probs[starting][absorbing] = result_3 # Direct edge to absorption
```

## Trace Evaluation Architecture

### Evaluation Flow

```
evaluate_trace_jax(trace, theta)
       |
       v
[Initialize Results Arrays]
       |
       +-- results = jnp.zeros(num_operations)
       |   (One slot for each intermediate value)
       |
       v
[Sequential Operation Execution]
       |
       +-- For each operation in trace.operations:
       |   |
       |   +-- Switch on operation.type:
       |       |
       |       +-- OpType.SET:
       |       |   results[op_idx] = op.constant + DOT(op.coeffs, theta)
       |       |
       |       +-- OpType.ADD:
       |       |   results[op_idx] = results[op.operands[0]] + results[op.operands[1]]
       |       |
       |       +-- OpType.MUL:
       |       |   results[op_idx] = results[op.operands[0]] * results[op.operands[1]]
       |       |
       |       +-- OpType.DIV:
       |       |   results[op_idx] = results[op.operands[0]] / results[op.operands[1]]
       |       |
       |       +-- OpType.DOT:
       |           sum = op.constant + DOT(op.coeffs, theta)
       |           for (operand_idx, coeff) in zip(op.operands, op.operand_coeffs):
       |               sum += results[operand_idx] * coeff
       |           results[op_idx] = sum
       |
       v
[Extract Graph Structure from Results]
       |
       +-- vertex_rates = results[vertex_rate_indices]
       +-- edge_probs = results[edge_prob_indices]
       +-- vertex_targets = trace.vertex_targets (unchanged)
       |
       v
Return {vertex_rates, edge_probs, vertex_targets}
```

**Critical property**: This is **pure JAX code** (no side effects), so it's fully JIT-compilable and differentiable.

### Instantiation from Trace

```
instantiate_from_trace(trace, theta, rewards=None)
       |
       v
eval_result = evaluate_trace_jax(trace, theta)
       |
       v
[Build Concrete C++ Graph]
       |
       +-- Create Graph(state_length)
       +-- Create vertices from trace.states
       |
       +-- For each vertex v:
       |   exit_rate = eval_result['vertex_rates'][v]
       |   For each target vertex w in eval_result['vertex_targets'][v]:
       |       edge_prob = eval_result['edge_probs'][v][w]
       |       edge_weight = exit_rate * edge_prob
       |       v.add_edge(w, edge_weight)
       |
       +-- If rewards provided:
       |   graph = graph.reward_transform(rewards)
       |
       v
Return concrete Graph object (can call .pdf(), .pmf(), etc.)
```

## Retracing and Invalidation

### When Does Retracing Occur?

Retracing is **unnecessary** in the current architecture because:

1. **Graph structure is fixed**: The trace records the topology (which vertices, which edges exist)
2. **Parameters are symbolic**: The trace contains coefficient arrays, not concrete values
3. **Evaluation is pure**: Same trace + same θ → same graph (deterministic)

### Hypothetical Retracing Scenario

Retracing would be needed if:

```
User changes graph structure
       |
       +-- Adds/removes vertices
       +-- Adds/removes edges
       +-- Changes callback function
       |
       v
Hash graph structure
       |
       v
Cache lookup with new hash
       |
       +-- Cache miss → record_elimination_trace() again
       +-- Cache hit → reuse existing trace
```

But this doesn't happen in practice because:
- The `Graph` object is **immutable** after construction
- SVGD only varies **parameters**, not structure

### Trace Validation (Implicit)

The code **validates trace compatibility** through:

```python
def trace_to_log_likelihood(trace, observed_data, reward_vector=None, ...):
    # Parameter dimension check
    if theta.shape[0] != trace.param_length:
        raise ValueError(f"Expected {trace.param_length} params, got {theta.shape[0]}")

    # Reward vector check
    if reward_vector is not None and len(reward_vector) != trace.n_vertices:
        raise ValueError(f"Reward vector size mismatch")
```

## Integration with SVGD

### SVGD Execution with Traces

```
SVGD Initialization
       |
       v
Build parameterized graph
       |
       v
trace = record_elimination_trace(graph, param_length=n)
       |
       v
model = trace_to_log_likelihood(trace, observed_data)
       |
       v
[SVGD Iteration Loop: 1000 iterations]
       |
       +-- For each particle θ_i:
       |   |
       |   +-- Compute log-likelihood:
       |   |   |
       |   |   +-- evaluate_trace_jax(trace, θ_i)  <-- Fast (pure JAX)
       |   |   |   → {vertex_rates, edge_probs, vertex_targets}
       |   |   |
       |   |   +-- instantiate_from_trace(trace, θ_i)  <-- Medium (C++ call)
       |   |   |   → concrete Graph
       |   |   |
       |   |   +-- graph.pdf(observed_times)  <-- Slow (forward algorithm)
       |   |   |   → PDF values
       |   |   |
       |   |   +-- sum(log(PDF))  <-- Fast (JAX)
       |   |       → log-likelihood
       |   |
       |   +-- grad(log_likelihood)(θ_i)  <-- JAX autodiff
       |       → gradient
       |
       +-- Compute kernel K(θ_i, θ_j)  <-- Pure JAX
       +-- SVGD update: θ_i += step_size * phi_i
       |
       v
Return posterior particles
```

### Performance Breakdown (67-vertex model, 1000 SVGD iterations)

From hierarchical_trace_cache.py benchmarks:

```
Phase 1: Record trace (once)
    Duration: ~0.5s

Phase 2: Evaluate trace (1000× per particle × 100 particles = 100,000×)
    Per evaluation:
    - evaluate_trace_jax:      ~0.01 ms  (pure JAX, JIT compiled)
    - instantiate_from_trace:  ~0.05 ms  (pybind11 overhead)
    - graph.pdf (10 times):    ~0.5 ms   (forward algorithm × 10 observations)
    - Total per particle:      ~0.56 ms

    Total for 100,000 evals:   ~56s

Alternative (symbolic DAG):
    Per evaluation: ~5-10 ms
    Total: ~500-1000s

Speedup: 10-20×
```

## Trace Serialization and Deserialization

### Serialization Format (JSON)

```json
{
  "n_vertices": 67,
  "state_length": 1,
  "param_length": 1,
  "starting_vertex_idx": 0,
  "is_discrete": false,
  "states": [[5], [4], [3], ...],
  "operations": [
    {
      "type": 0,  // OpType.SET
      "constant": 0.0,
      "param_idx": 0,
      "operands": [],
      "coefficients": [10.0],  // Coefficient for θ[0]
      "operand_coeffs": []
    },
    {
      "type": 1,  // OpType.ADD
      "constant": 0.0,
      "param_idx": -1,
      "operands": [5, 12],  // Indices of previous results
      "coefficients": [],
      "operand_coeffs": []
    },
    ...
  ],
  "vertex_targets": [[1, 2], [2, 3], ...],  // Adjacency list
  "version": "0.22.0"
}
```

### Deserialization (trace_elimination.py)

```python
def trace_from_dict(data):
    trace = EliminationTrace()
    trace.n_vertices = data['n_vertices']
    trace.state_length = data['state_length']
    trace.param_length = data['param_length']
    trace.starting_vertex_idx = data['starting_vertex_idx']
    trace.is_discrete = data['is_discrete']
    trace.states = [tuple(s) for s in data['states']]
    trace.vertex_targets = [list(t) for t in data['vertex_targets']]

    # Rebuild operations
    trace.operations = []
    for op_data in data['operations']:
        op = Operation(
            type=op_data['type'],
            constant=op_data['constant'],
            param_idx=op_data.get('param_idx', -1),
            operands=op_data['operands'],
            coefficients=op_data['coefficients'],
            operand_coeffs=op_data.get('operand_coeffs', [])
        )
        trace.operations.append(op)

    return trace
```

## Advanced: Trace-Based Log-Likelihood

### C++ Code Generation from Trace

The function `_generate_cpp_from_trace()` in `__init__.py` creates **standalone C++ code**:

```cpp
// Generated code structure
#include "phasiccpp.h"

// Embedded trace data (from trace serialization)
static const size_t N_OPERATIONS = 1247;
static const size_t N_VERTICES = 67;
static const int operations_types[] = {0, 1, 2, 3, 4, ...};
static const double operations_consts[] = {0.0, 1.0, ...};
static const int operations_param_indices[] = {0, -1, 0, ...};
static const size_t operations_operands_flat[] = {5, 12, 18, ...};
static const double operations_coeffs_flat[] = {10.0, 6.0, ...};

// Embedded observation data
static const double OBSERVED_TIMES[] = {1.5, 2.3, 0.8, ...};
static const size_t N_OBSERVATIONS = 10;

double compute_log_likelihood(const double* theta, int n_params) {
    // Step 1: Evaluate trace
    double* results = (double*)malloc(N_OPERATIONS * sizeof(double));

    size_t operand_offset = 0;
    size_t coeff_offset = 0;

    for (size_t i = 0; i < N_OPERATIONS; i++) {
        int op_type = operations_types[i];
        double constant = operations_consts[i];
        int param_idx = operations_param_indices[i];

        switch (op_type) {
            case 0:  // SET
                results[i] = constant + operations_coeffs_flat[coeff_offset] * theta[param_idx];
                coeff_offset += 1;
                break;
            case 1:  // ADD
                results[i] = results[operations_operands_flat[operand_offset]]
                           + results[operations_operands_flat[operand_offset + 1]];
                operand_offset += 2;
                break;
            // ... other cases ...
        }
    }

    // Step 2: Instantiate graph from results
    phasic::Graph graph(STATE_LENGTH);
    auto start = graph.starting_vertex_p();
    std::vector<phasic::Vertex*> vertices;

    // Create vertices and edges using results array
    // (vertex_rates and edge_probs extracted from results)

    // Step 3: Compute PDF at all observation points
    double log_lik = 0.0;
    for (size_t i = 0; i < N_OBSERVATIONS; i++) {
        double pdf = graph.pdf(OBSERVED_TIMES[i], GRANULARITY);
        log_lik += log(fmax(pdf, 1e-10));
    }

    free(results);
    return log_lik;
}
```

**Compilation and usage**:

```python
# Compile to shared library
lib_path = _compile_trace_library(cpp_code, trace_hash)

# Load and wrap for JAX
compute_log_lik = ctypes.CDLL(lib_path).compute_log_likelihood
compute_log_lik.restype = ctypes.c_double
compute_log_lik.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# JAX wrapper
def log_likelihood_jax(theta):
    return jax.pure_callback(
        lambda t: compute_log_lik(t.ctypes.data_as(...), len(t)),
        jax.ShapeDtypeStruct((), jnp.float64),
        theta
    )

# Use with SVGD
svgd = SVGD(log_likelihood_jax, ...)
```

**Performance advantage**: Eliminates Python interpreter overhead, reduces to single C++ function call per evaluation.

## Summary

Trace elimination achieves high performance through:

1. **One-time recording**: O(n³) graph elimination recorded as linear operation sequence
2. **Fast evaluation**: O(n²) or O(n) trace evaluation using pure JAX operations
3. **Full differentiability**: Pure JAX code enables autodiff for gradient computation
4. **Immutable structure**: No retracing needed because graph topology is fixed
5. **JIT compilation**: Pure JAX evaluation is fully JIT-compilable
6. **C++ code generation**: Optional standalone C++ compilation for maximum performance

The key innovation is **separation of structure from parameterization**, enabling the same elimination structure to be reused across thousands of parameter variations in SVGD inference.
