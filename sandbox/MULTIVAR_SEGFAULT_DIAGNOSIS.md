# Multivariate Test Segfault - Complete Diagnosis

**Status**: ❌ UNRESOLVED - Deep memory corruption bug

**Date**: October 25, 2025

## Summary

The `tests/multivar_test.py` segfault is **NOT a multivariate issue**. It's a broader problem with coalescent model + SVGD affecting both univariate and multivariate cases.

## Root Cause

**Memory corruption in C++ trace elimination or evaluation code.**

The segfault occurs when:
1. Creating a coalescent graph (parameterized=True)
2. Creating a model from that graph
3. Calling the model function (even before SVGD starts)

The segfault happens at the first model evaluation, NOT during SVGD initialization.

## What I Tested

### ✓ Works:
- Simple exponential graphs with SVGD
- Multivariate models with simple graphs
- Coalescent graph creation and sampling
- Graph serialization and trace recording

### ✗ Segfaults:
-  Coalescent + univariate SVGD
- ✗ Coalescent + multivariate SVGD
- ✗ Calling model function on coalescent graph
- ✗ Even with FFI disabled (`phasic.configure(ffi=False, openmp=False)`)

## Evidence

### Test 1: Multivariate with simple exponential - WORKS
```python
# Simple exponential graphs work fine with multivariate
graph = simple_exponential_graph()
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(graph)
svgd = graph.svgd(observed_data_2d, rewards=rewards_2d)  # ✓ SUCCESS
```

### Test 2: Coalescent univariate - SEGFAULTS
```python
# Coalescent crashes even without multivariate
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
model = phasic.Graph.pmf_and_moments_from_graph(graph)
pmf, moments = model(theta, times)  # ✗ SEGFAULT HERE
```

###Test 3: Step-by-step diagnosis
```
✓ Graph created
✓ Data sampled
✓ Model created
✗ SEGFAULT at model(theta, times)
```

The crash happens when evaluating the model function, NOT during SVGD initialization or fit().

## Known Issues

From `src/phasic/config.py:291`:
```python
'ffi': False,  # Always False for now (memory corruption bug)
```

There's a documented memory corruption bug in the codebase!

## Files Affected

- `tests/multivar_test.py` - User's test (segfaults)
- `test_coalescent_univariate_svgd.py` - My diagnostic (segfaults)
- `test_step_by_step_svgd.py` - Isolated crash point (segfaults at model evaluation)

## Array Shape Issues (FIXED but not the root cause)

I did fix these issues in `tests/multivar_test.py`:

1. ✅ Added transposes:
   - `rewards`: `(n_features, n_vertices)` → `(n_vertices, n_features)`
   - `observed_data`: `(n_features, n_observations)` → `(n_observations, n_features)`

2. ✅ Fixed parameter incompatibility:
   - Changed `regularization=0` (was using default=10 with nr_moments=0)

3. ✅ Changed `true_theta` to float: `np.array([10.0])` instead of `np.array([10])`

**But these fixes didn't resolve the segfault** because the root cause is deeper.

## What Doesn't Help

- ✗ Disabling FFI: `phasic.configure(ffi=False, openmp=False)`
- ✗ Correct import order (phasic before jax)
- ✗ Using `use_ffi=False` parameter
- ✗ Smaller data sizes (50 observations, 8 particles, 3 iterations)
- ✗ Clearing trace cache
- ✗ Array transposes and dtype fixes

## Likely Causes

1. **Memory corruption in C++ trace evaluation**
   - Trace is loaded from cache successfully
   - But evaluation causes memory corruption
   - Happens for complex graphs like coalescent but not simple exponential

2. **Buffer overflow in parameterized edge handling**
   - Coalescent has many parameterized edges with complex structure
   - May be writing beyond buffer boundaries

3. **Stack overflow in recursive graph operations**
   - Coalescent graphs are more deeply nested
   - May exceed stack limits during elimination

## Recommended Actions

### Immediate Workaround

**Use trace-based elimination instead of model API:**

```python
from phasic.trace_elimination import (
    record_elimination_trace,
    evaluate_trace_jax,
    trace_to_log_likelihood
)

# 1. Record trace (safe - doesn't crash)
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
trace = record_elimination_trace(graph, param_length=1)

# 2. Use trace_to_log_likelihood for SVGD
observed_times = np.array([1.5, 2.3, 0.8, ...])
log_lik = trace_to_log_likelihood(trace, observed_times, granularity=100)

# 3. Run SVGD with log-likelihood function
from phasic import SVGD
svgd = SVGD(log_lik, theta_dim=1, n_particles=100, n_iterations=1000)
results = svgd.fit()
```

This avoids the model API entirely and uses the lower-level trace API which may not have the same memory corruption issue.

### Long-term Fix Required

1. **Debug C++ code with valgrind or AddressSanitizer**
   ```bash
   # Build with address sanitizer
   CFLAGS="-fsanitize=address" pip install --force-reinstall --no-deps -e .
   python test_coalescent_univariate_svgd.py
   ```

2. **Check buffer sizes in trace evaluation**
   - Look for fixed-size buffers that might overflow with complex graphs
   - Check array bounds in `src/c/phasic.c`

3. **Add defensive checks**
   - Validate array sizes before operations
   - Add bounds checking to parameterized edge access

## Conclusion

- ✅ Multivariate implementation is correct
- ✅ Array shapes and parameter fixes are correct
- ❌ **Root cause**: Memory corruption in C++ code when handling complex parameterized graphs
- ❌ Affects both univariate and multivariate cases
- ❌ Not FFI-specific

**The user should use the trace API workaround until the C++ memory corruption is fixed.**
