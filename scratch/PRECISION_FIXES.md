# Precision Fixes for Large Graph Expectation Errors

**Date:** 2026-01-15
**Issue:** ~0.5% errors in expectations for large graphs
**Root Cause:** Mixed precision usage in C++ API

## Summary

Investigation revealed that the codebase uses 64-bit doubles correctly throughout most of the implementation, but had **one critical precision bug** in the C++ API that could contribute to cumulative errors in large graphs.

## Changes Made

### 1. C++ API Precision Fix (CRITICAL) ⚠️

**File:** `api/cpp/phasiccpp.h`

**Problem:** The `pdf()` and `cdf()` methods used `float` (32-bit) for the time parameter, truncating precision to ~7 decimal digits before passing to the underlying double-precision C code.

**Changes:**
```cpp
// Before
double pdf(float time, int granularity = 0)
double cdf(float time, int granularity = 0)

// After
double pdf(double time, int granularity = 0)
double cdf(double time, int granularity = 0)
```

**Impact:** For large graphs with many time steps, this precision loss could accumulate and contribute to errors in computed expectations.

### 2. JAX FFI Float64 Safeguards

**File:** `src/phasic/ffi_wrappers.py`

**Problem:** Output shapes inferred dtype from input arrays, which could theoretically allow float32 to propagate through the system.

**Changes:**
- Line 539: Force `jnp.float64` for `compute_pmf_ffi` output
- Line 707: Force `jnp.float64` for `compute_pmf_and_moments_ffi` PMF output

**Impact:** Ensures float64 precision regardless of input dtype, preventing accidental precision loss.

## What Was Already Correct ✓

1. **C Core Implementation:** Uses `double` (64-bit) throughout
2. **Python/JAX Input Conversion:** Explicitly converts to `float64` at entry points
3. **Moment Computation:** Uses `double` for all accumulations
4. **Forward Algorithm:** Uses `long double` for probability accumulation (better than double!)
5. **Trace Elimination:** Uses `np.float64` throughout

## Testing

Tested with exponential distribution:
- Input: `float64` times with 14-digit precision
- Output: `float64` PDFs
- Relative error: <1% (dominated by uniformization approximation, not precision)

## Next Steps

**The 0.5% expectation errors are likely NOT caused by float precision issues.**

The codebase uses appropriate precision throughout. The errors you're seeing are more likely caused by:

1. **Numerical convergence issues** in graph elimination
2. **Granularity settings** in the forward algorithm (too coarse)
3. **Algorithmic approximations** in moment computation
4. **Conditioning problems** in large sparse systems

### Recommended Investigations

1. **Increase granularity** in forward algorithm calls (currently auto-selected, try manual override)
2. **Check for numerical conditioning** in elimination trace (look for very large/small multipliers)
3. **Compare accumulated vs direct computation** for specific problematic graphs
4. **Profile error growth** as graph size increases to identify scaling issues

### Code to Investigate Further

- `src/c/phasic.c:5547` - Moment accumulation loop (uses double, but may need Kahan summation)
- `src/c/phasic.c:6275-6296` - Forward algorithm probability updates (uses long double)
- Granularity selection in `ptd_probability_distribution_context_create`

## Files Modified

- `api/cpp/phasiccpp.h` - Fixed float→double for time parameters
- `src/phasic/ffi_wrappers.py` - Added float64 safeguards in JAX FFI
