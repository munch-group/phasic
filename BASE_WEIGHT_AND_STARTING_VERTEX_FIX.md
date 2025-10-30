# Base Weight and Starting Vertex Fix

## Problem Summary

SVGD inference was producing 96-98% relative error on parameter estimates:
- True θ = 2.0 → SVGD estimated θ = 0.052-0.070 (30-40x underestimation)
- True θ = 1.0 → SVGD estimated θ = 0.020-0.024 (40-50x underestimation)

Additionally, trace-based PDF evaluation was returning 0.0 for all parameter values.

## Root Causes Identified

### 1. Missing Base Weight in Parameterized Edges

**User Directive**: "The starting state should NOT be parameterized, and should be treated as a special case so it is NOT affected by parameter updates."

**Implementation**: Use empty coefficient arrays `[]` for non-parameterized edges with `base_weight` as the constant.

**Problem**: Code was computing weight as `dot(coefficients, params)`, ignoring `base_weight`:
- For empty coefficients: `weight = dot([], params) = 0` ❌
- Starting edges had zero weight, breaking the entire distribution

**Fix**: Changed weight computation to `base_weight + dot(coefficients, params)`:
- For empty coefficients: `weight = base_weight + 0 = base_weight` ✅
- Non-parameterized edges now work correctly

### 2. Starting Vertex Not Skipped in Parameter Updates

**Problem**: Parameter updates were being applied to ALL edges, including the starting vertex.

**Fix**: Added check to skip starting vertex in `ptd_graph_update_weight_parameterized()`:
```c
if (graph->vertices[i] == graph->starting_vertex) {
    continue;  // Skip starting vertex edges
}
```

### 3. Wrong starting_vertex_idx in Trace Recording

**Problem**: Trace recording was using state-based lookup to find starting vertex:
```python
start_state = tuple(graph.starting_vertex().state())
starting_vertex_idx = state_to_idx[start_state]
```

Multiple vertices can have the same state after elimination (e.g., vertex 0 and vertex 2 both with state `[0]`), causing the wrong vertex to be identified as starting vertex.

**Additional Discovery**: `graph.starting_vertex()` returns a sentinel vertex NOT in the `graph.vertices()` list.

**Fix**: Use convention that vertex 0 in the vertices list is always the starting vertex:
```python
starting_vertex_idx = 0
```

### 4. State Collision in instantiate_from_trace()

**Problem**: `instantiate_from_trace()` was using state-to-vertex mapping, causing vertices with identical states to collide. Only one vertex was created for all vertices sharing a state.

**Fix**: Use index-to-vertex mapping instead:
```python
idx_to_vertex = {}
# Get or create starting vertex
start_idx = trace.starting_vertex_idx
start_vertex = graph.starting_vertex()
idx_to_vertex[start_idx] = start_vertex

# Create all other vertices by index
for i in range(trace.n_vertices):
    if i not in idx_to_vertex:
        v = graph.find_or_create_vertex(trace.states[i].tolist())
        idx_to_vertex[i] = v
```

### 5. Graph State Pollution in SVGD Tests

**Root Cause for SVGD Failure**: Tests were calling `update_parameterized_weights()` on the graph before passing it to SVGD for data generation and PDF verification. This polluted the graph state.

**Fix**: Use separate graph instances:
```python
# Data generation: separate graph
graph_for_data = build_exponential_graph()
data = generate_data(graph_for_data, true_theta, N_OBSERVATIONS)

# PDF verification: separate graph
graph_for_pdf = build_exponential_graph()
integral = verify_pdf_integration(graph_for_pdf, true_theta)

# SVGD: fresh graph with NO update_parameterized_weights called
graph_for_svgd = build_exponential_graph()
passed, error_pct, results = run_svgd_test(graph_for_svgd, data, true_theta)
```

## Test Results

### Direct PDF Evaluation
**Before fixes**: ❌ Not tested (assumed working)
**After fixes**: ✅ All errors < 1.01%
```
PDF evaluation at t=0.5:
     θ |       PDF(t) |     Expected |    Error %
--------------------------------------------------
  0.50 |     0.391112 |     0.389400 |      0.44%
  1.00 |     0.611117 |     0.606531 |      0.76%
  1.50 |     0.715264 |     0.708550 |      0.95%
  2.00 |     0.743203 |     0.735759 |      1.01%
  2.50 |     0.723046 |     0.716262 |      0.95%
  3.00 |     0.674429 |     0.669390 |      0.75%
```

### Trace-Based PDF Evaluation
**Before fixes**: ❌ All PDFs = 0.0 (100% error)
```
     θ |       Direct |        Trace |    Error %
------------------------------------------------------
  0.50 |     0.391112 |     0.000000 |    100.00%
  1.00 |     0.611117 |     0.000000 |    100.00%
  2.00 |     0.743203 |     0.000000 |    100.00%
```

**After fixes**: ✅ Perfect match (0.00% error)
```
     θ |       Direct |        Trace |    Error %
------------------------------------------------------
  0.50 |     0.391112 |     0.391112 |      0.00%
  1.00 |     0.611117 |     0.611117 |      0.00%
  2.00 |     0.743203 |     0.743203 |      0.00%
```

### SVGD Inference
**Before all fixes**: ❌ 96-98% error
```
Test 1: Simple Exponential (θ = 2.0)
  True θ:      2.000
  SVGD θ_mean: 0.052
  Relative error: 97.4%
```

**After all fixes**: ✅ 1.1% error
```
Test: Simple Exponential (θ = 2.0)
  True θ:      2.000
  SVGD θ_mean: 2.022
  Error:       1.1%
```

## Files Modified

### C Implementation
**`src/c/phasic.c`**
1. **Weight computation** (lines 1652-1665): Changed from `dot(coeffs, params)` to `base_weight + dot(coeffs, params)`
2. **Skip starting vertex** (lines 1765-1771): Added check to skip starting vertex in parameter updates
3. **Validation** (lines 1692-1722): Added validation for consistent coefficient lengths

### C++ API
**`api/cpp/phasiccpp.h`**
- Added `base_weight()` method to `ParameterizedEdge` class (lines 932-934)

### Python Bindings
**`src/cpp/phasic_pybind.cpp`**
- Exposed `base_weight()` method in Python API (lines 3062-3065)

### Python Implementation
**`src/phasic/trace_elimination.py`**
1. **TraceBuilder.add_dot()** (lines 231-254): Updated signature to accept `base_weight` parameter
2. **Trace recording** (lines 533, 596): Changed from `param_edge.weight()` to `param_edge.base_weight()`
3. **evaluate_trace()** (lines 840-845): Include base_weight in DOT operations
4. **evaluate_trace_jax()** (lines 1292-1299): Include base_weight in DOT operations
5. **starting_vertex_idx** (line 733): Use convention that starting vertex is always index 0
6. **instantiate_from_trace()** (lines 1145-1186): Use index-to-vertex mapping instead of state-to-vertex

### Tests
**`test_svgd_pdf_fix_verification.py`**
1. **Exponential graph** (line 71): Changed from `add_edge_parameterized(v1, 0.0, [1.0])` to `add_edge(v1, 1.0)`
2. **Coalescent callback** (line 86): Changed from `[0, [1]]` to `[1.0, []]`
3. **All test functions**: Use separate graph instances for data generation, PDF verification, and SVGD

## Summary of Changes

### What Was Necessary

1. ✅ **Base weight support**: Essential for non-parameterized edges with empty coefficient arrays
2. ✅ **Starting edge non-parameterization**: Per user directive, starting edges shouldn't depend on inferred parameters
3. ✅ **Skip starting vertex in updates**: Prevents parameter updates from affecting starting edges
4. ✅ **Trace fixes**: Fixed real bug in trace-based PDF evaluation (separate from SVGD issue)
5. ✅ **Separate graph instances**: Fixed SVGD test methodology that was polluting graph state

All changes were necessary and addressed real bugs in the codebase.

## Impact

- **Direct PDF evaluation**: Working correctly (<1.01% error)
- **Trace-based PDF**: Now works perfectly (0.00% error vs 100% error before)
- **SVGD inference**: Reduced error from 97% to 1.1%
- **Model correctness**: Starting edges now properly non-parameterized as intended

## Related Documents

- `BASE_WEIGHT_BUG_DIAGNOSIS.md` - Initial diagnosis
- `INSTANTIATE_FROM_TRACE_FIX.md` - Trace-specific fixes
