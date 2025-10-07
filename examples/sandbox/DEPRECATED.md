# DEPRECATED EXAMPLES

**Date:** 2025-10-07

These examples in the sandbox directory are **DEPRECATED** and use an outdated API.

## What Changed

The `use_ffi` parameter has been removed from the API and replaced with clearer, separate functions:

### Old API (Deprecated)
```python
# OLD - Don't use
builder = Graph.pmf_from_cpp("model.cpp", use_ffi=True)
graph = builder(theta)
```

### New API (Use This)
```python
# NEW - Use this instead
from ptdalgorithms import load_cpp_builder

builder = load_cpp_builder("model.cpp")
graph = builder(theta)
```

## Affected Files

The following notebooks use the deprecated API:

- **ffi_approach_example.ipynb** - Uses `Graph.pmf_from_cpp(..., use_ffi=True)`
- **cpp_models_showcase.ipynb** - Uses `Graph.pmf_from_cpp(..., use_ffi=True)`

## Recommended Alternatives

Instead of these deprecated notebooks, use the updated Python examples:

### For JAX-Compatible Approach
- **examples/jit_pdf.py** - Comprehensive JAX-compatible demonstration
- Supports: JIT compilation, automatic differentiation, vmap, pmap

### For Direct C++ Approach
- **examples/ffi_pdf.py** - Comprehensive direct C++ demonstration
- Uses: `load_cpp_builder()` for efficient graph reuse

### For Comparison
- **examples/jit_or_ffi.py** - Side-by-side comparison of both approaches
- Shows performance characteristics and use cases

## Migration Guide

To migrate from the old API to the new API:

### Option 1: JAX-Compatible (if you need gradients)
```python
# Old
builder = Graph.pmf_from_cpp("model.cpp", use_ffi=True)

# New - for gradient-based inference
model = Graph.pmf_from_cpp("model.cpp")
# Now use with JAX: jax.grad, jax.jit, etc.
```

### Option 2: Direct C++ (if you need performance)
```python
# Old
builder = Graph.pmf_from_cpp("model.cpp", use_ffi=True)

# New - for direct C++ graph building
from ptdalgorithms import load_cpp_builder
builder = load_cpp_builder("model.cpp")
# Usage remains the same: graph = builder(theta)
```

## Why This Change?

The `use_ffi` parameter was misleading:
- It suggested "JAX Foreign Function Interface" but actually meant "skip JAX wrapper"
- It returned different types (Graph vs function) which was confusing
- It mixed two different use cases into one parameter

The new API provides:
- **Clear separation**: Different functions for different use cases
- **Better names**: `load_cpp_builder()` clearly indicates what you get
- **Type safety**: Each function has a consistent return type
- **Better documentation**: Clearer guidance on when to use each approach

## Status

These notebooks are kept for archival purposes only and may be removed in a future release.

For up-to-date examples, see the main `examples/` directory.
