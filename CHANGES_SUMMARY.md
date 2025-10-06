# Summary of Changes: Parameterized Edges Implementation

## Date
October 6, 2025

## Overview
Fixed critical bugs in parameterized edge support and updated all examples to reflect the working implementation. Parameterized edges now fully support JAX transformations (JIT, grad, vmap) for gradient-based inference.

## Bug Fixes

### 1. Fixed `serialize()` param_length Detection
**File**: `src/ptdalgorithms/__init__.py` (lines 615-680)

**Issue**: Detection algorithm treated 0.0 as garbage, failing on edge states like `[1.0, 0.0]`

**Fix**: Changed heuristic to detect NaN/inf/extremely tiny values (< 1e-300) instead of using magnitude checks. This correctly handles edge states containing zeros.

**Result**: Now correctly detects `param_length=2` for edge states like `[1.0, 0.0]` and `[0.0, 1.0]`

### 2. Fixed `parameterized_edges()` C++ Method
**File**: `src/cpp/ptdalgorithmscpp.cpp` (lines 292-313)

**Issue**: Method returned ALL edges (including regular ones) as parameterized edges, with NULL state pointers for non-parameterized edges. This broke serialization detection.

**Fix**: Changed implementation to only return edges where `parameterized == true`

**Result**:
- Serialization correctly identifies parameterized vs non-parameterized graphs
- Regular edges no longer appear in parameterized_edges() results
- Empty edge_state() arrays eliminated

## Documentation Updates

### 1. Updated `examples/jit_pdf.py`
- Removed outdated warning: "Continuous mode only (discrete mode has a known bug)" (line 478)
- Updated status message: "Parameterized edges work in both continuous and discrete modes!" (line 530)

### 2. Updated `examples/README.md`
- Added section on parameterized edges feature
- Updated file names to reflect current structure:
  - `jax_compatible_example.py` → `jit_pdf.py`
  - `ffi_approach_example.py` → `ffi_pdf.py`
  - `approach_comparison.py` → `jit_or_ffi.py`
- Added performance comparison table with gradient support column
- Added new test script to Getting Started section

### 3. Updated `examples/python_graph_to_jax_example.py`
- Replaced outdated section 2 using non-existent `pmf_from_graph_parameterized()` function
- Added complete working example using `add_edge_parameterized()`
- Demonstrated full JAX support: JIT, gradients, and vmap
- Updated summary to reflect current capabilities

### 4. Created `examples/test_parameterized_edges.py`
- New verification script to test all parameterized edges features
- Tests: serialization, PMF computation, JIT, gradients, vmap, discrete mode
- Provides clear pass/fail feedback
- Includes usage examples for SVGD and optimization

## Features Now Working

✅ **Parameterized edge serialization** with automatic param_length detection
✅ **C++ code generation** from Python graphs with correct theta computations
✅ **JAX compatibility**: JIT compilation, gradients (autodiff), vmap
✅ **Correct API usage**: `model = Graph.pmf_from_graph(g, discrete=mode)`, then `model(theta, times)`
✅ **Mixed graphs**: Graphs with both regular and parameterized edges
✅ **Both modes**: Continuous (PDF) and discrete (PMF) phase-type distributions

## Testing

All test scripts pass successfully:
- `test_parameterized_edges.py` - Comprehensive feature verification
- `test_edge_inspection.py` - Edge state access verification
- `test_serialization_debug.py` - Serialization correctness
- `test_pmf_complete.py` - Full workflow with rabbit model

## API Examples

### Non-parameterized (fixed weights):
```python
g = Graph(state_length=2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([0, 1])
v1.add_edge(v2, weight=1.5)  # Fixed weight

model = Graph.pmf_from_graph(g)
pdf = model(times)  # No theta parameter needed
```

### Parameterized (gradient support):
```python
g = Graph(state_length=2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([0, 1])
v1.add_edge_parameterized(v2, weight=0.0, edge_state=[2.0, 0.5])
# Actual weight = 2.0*theta[0] + 0.5*theta[1]

model = Graph.pmf_from_graph(g)
theta = jnp.array([1.0, 3.0])
pdf = model(theta, times)  # Theta required for parameterized graphs

# Gradients work!
grad_fn = jax.grad(lambda t: model(t, times).sum())
gradient = grad_fn(theta)
```

## Use Cases Enabled

The parameterized edges feature enables:
- **SVGD (Stein Variational Gradient Descent)** for Bayesian inference
- **Maximum likelihood estimation** with gradient-based optimization
- **Sensitivity analysis** via automatic differentiation
- **Gradient-based model selection**
- **Parameter uncertainty quantification**

## Files Modified

1. `src/ptdalgorithms/__init__.py` - Core serialization and detection logic
2. `src/cpp/ptdalgorithmscpp.cpp` - C++ parameterized_edges() method
3. `examples/jit_pdf.py` - Updated comments about discrete mode
4. `examples/README.md` - Comprehensive update with parameterized edges section
5. `examples/python_graph_to_jax_example.py` - Complete rewrite of section 2
6. `examples/test_parameterized_edges.py` - New verification script

## Backward Compatibility

All changes are backward compatible:
- Non-parameterized graphs continue to work as before
- Existing C++ model files unaffected
- FFI approach unchanged

## Next Steps

The implementation is complete and ready for production use. Potential future enhancements:
- Performance optimization of C++ code generation
- Additional examples for SVGD inference
- Integration with popular Bayesian inference libraries
