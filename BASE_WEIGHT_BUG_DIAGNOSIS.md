# Base Weight Bug Diagnosis

## Problem Summary

SVGD inference failing with 96-98% relative error on parameter estimates. Tests show:
- True θ = 2.0 → SVGD estimates θ = 0.055-0.070 (30-40x underestimation)
- True θ = 1.0 → SVGD estimates θ = 0.022-0.024 (40-45x underestimation)

## Root Cause Identified

**CRITICAL BUG**: `instantiate_from_trace()` produces graphs that return PDF=0.0 for all parameter values and observation times.

### Evidence

1. **Direct graph PDF evaluation**: ✅ WORKS
   - θ=2.0, t=0.5: PDF = 0.743 (expected: 0.736, error <1%)
   - Parameterized edges correctly use base_weight + dot(coefficients, params)

2. **Trace-based PDF evaluation**: ❌ FAILS
   - θ=2.0, t=0.5: PDF = 0.000 (expected: 0.736, error 100%)
   - Graph created via `instantiate_from_trace()` returns PDF=0.0 for ALL parameter values

3. **Trace recording**: ✅ CORRECT
   - DOT operations correctly store base_weight in const_value field
   - Example: `{"op_type":2, "const_value":0, "coefficients":[1]}`

4. **Trace evaluation (NumPy/JAX)**: ✅ CORRECT
   - `evaluate_trace()` and `evaluate_trace_jax()` correctly compute: `weight = base_weight + dot(coeffs, params)`

## Implementation Changes Made

### 1. C Code (`src/c/phasic.c`)
- ✅ Updated `ptd_edge_update_weight_parameterized()` to use `base_weight + dot(coefficients, params)`
- ✅ Added skip for starting vertex edges in parameter updates
- ✅ Added validation for consistent coefficient lengths

### 2. C++ Wrapper (`api/cpp/phasiccpp.h`)
- ✅ Added `base_weight()` method to `ParameterizedEdge` class

### 3. Python Bindings (`src/cpp/phasic_pybind.cpp`)
- ✅ Exposed `base_weight()` method in Python API

### 4. Trace Evaluation (`src/phasic/trace_elimination.py`)
- ✅ Updated `evaluate_trace()` to include base_weight in DOT operations
- ✅ Updated `evaluate_trace_jax()` to include base_weight in DOT operations
- ✅ Updated `TraceBuilder.add_dot()` to accept and store base_weight parameter
- ✅ Updated trace recording to call `param_edge.base_weight()` instead of `param_edge.weight()`

### 5. Test Updates (`test_svgd_pdf_fix_verification.py`)
- ✅ Changed exponential graph to use non-parameterized starting edge
- ✅ Changed coalescent callback to use base_weight=1.0, empty coefficients for starting edge

## Remaining Issue

**`instantiate_from_trace()` Bug**: The function that creates a Graph object from an elimination trace is producing graphs with PDF=0.0.

### Diagnosis Steps Needed

1. Check if `instantiate_from_trace()` correctly handles base_weight when creating parameterized edges
2. Check if edges are being added with correct base_weight and coefficient values
3. Check if starting vertex edges are being created correctly (non-parameterized)
4. Check if the graph structure (vertices, edges, rates) matches the original graph

### Test Files

- `debug_pdf_param.py` - Tests direct PDF evaluation (PASSES)
- `test_trace_vs_direct_pdf.py` - Compares trace vs direct (SHOWS BUG)

## Next Steps

1. Debug `instantiate_from_trace()` in `src/phasic/trace_elimination.py`
2. Fix how parameterized edges are created from trace evaluation results
3. Ensure base_weight is preserved when instantiating graphs from traces
4. Re-run SVGD test suite to verify fix

## Files Modified

- `src/c/phasic.c` - Weight computation, validation, parameter updates
- `api/cpp/phasiccpp.h` - Added base_weight() method
- `src/cpp/phasic_pybind.cpp` - Python binding for base_weight()
- `src/phasic/trace_elimination.py` - Trace evaluation and recording fixes
- `test_svgd_pdf_fix_verification.py` - Test updates for non-parameterized starting edges

## Test Results

- Direct PDF evaluation: ✅ All errors <1.01%
- Trace PDF evaluation: ❌ Returns 0.0 (100% error)
- SVGD tests: ❌ 96-98% error (consequence of trace bug)
