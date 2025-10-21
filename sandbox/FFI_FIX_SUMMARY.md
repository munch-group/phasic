# FFI Type Mismatch Fix - Summary

## Problem

The FFI wrapper functions (`compute_pmf_ffi`, etc.) expected JSON string inputs but `graph.serialize()` returns a dictionary with numpy arrays. This caused a `TypeError` when trying to use the FFI functions with graph serialization output.

**Error:** `TypeError: __init__(): incompatible constructor arguments`
**Location:** `src/phasic/ffi_wrappers.py` line 136

## Solution

Updated the FFI wrapper to accept both `dict` and `str` inputs by:

1. **Added helper functions** (lines 75-127):
   - `_make_json_serializable()` - Recursively converts numpy arrays to lists
   - `_ensure_json_string()` - Converts dict to JSON string if needed, passes through strings unchanged

2. **Updated type hints** for all functions to accept `Union[str, Dict]`:
   - `compute_pmf_fallback()` (line 195)
   - `compute_moments_fallback()` (line 256)
   - `compute_pmf_and_moments_fallback()` (line 311)
   - `compute_pmf_ffi()` (line 379)
   - `compute_moments_ffi()` (line 431)
   - `compute_pmf_and_moments_ffi()` (line 466)

3. **Added automatic conversion** in each fallback function using `_ensure_json_string()`

## Test Results

✅ **Dict input works:**
```python
structure_dict = graph.serialize()  # Returns dict
pdf_values = compute_pmf_ffi(structure_dict, theta, times)
```

✅ **String input still works** (backward compatibility):
```python
structure_json = json.dumps(_make_json_serializable(structure_dict))
pdf_values = compute_pmf_ffi(structure_json, theta, times)
```

✅ **JIT compilation works:**
```python
jit_fn = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))
result = jit_fn(structure_dict, theta, times, False, 100)
```

❌ **Gradients don't work yet** (pre-existing limitation):
```python
grad_fn = jax.grad(compute_pmf_ffi)  # Raises: "Pure callbacks do not support JVP"
```

## Known Limitations

### Gradient Support (Pre-existing Issue)

The `pure_callback` approach used in the FFI wrapper doesn't support automatic differentiation. This requires custom VJP (Vector-Jacobian Product) rules to be implemented.

**Impact:** SVGD and other gradient-based inference methods cannot use `compute_pmf_ffi` directly.

**Workaround options:**
1. Use finite differences for gradients (slow but works)
2. Use trace-based evaluation with instantiated graphs (Phase 3 approach)
3. Implement custom VJP rules (Phase 4+ work)

**Code location:** `src/phasic/ffi_wrappers.py` lines 195-246

### Notebook Status

The tutorial notebook (`docs/pages/tutorials/rabbits_full_py_api_example.ipynb`) has incomplete SVGD cells:
- Cell 177: Calls undefined method `rabbit_model.svgd()`
- Cell 180: References undefined `log_posterior_fn`
- Trace caching cells (183-184): ✅ Work correctly

## Files Modified

- `src/phasic/ffi_wrappers.py` - Added dict/string compatibility
- Test files created:
  - `test_ffi_dict_input.py` - Verifies dict/string inputs work
  - `test_svgd_cells.py` - Tests log-likelihood evaluation (no gradients)

## Next Steps

1. **For immediate use:** The FFI fix enables all dict-based workflows that don't need gradients
2. **For gradient support:** Implement custom VJP rules in `ffi_wrappers.py` (Phase 4 work)
3. **For notebook:** Either fix Cell 180 to define `log_posterior_fn` or use trace-based approach

## Verification

Run tests with:
```bash
python test_ffi_dict_input.py    # Should pass all tests
python test_svgd_cells.py        # Passes except gradient test
```
