# Gradient Sign Bug Analysis

**Date:** 2025-11-16
**Status:** ⚠️ UNDER INVESTIGATION - Root cause not yet identified

---

## Problem Statement

FFI gradients have **incorrect sign and magnitude** compared to analytical formulas and numerical gradients.

**Test Case**: Single exponential distribution with rate λ=2.0 at time t=1.0

**Expected**:
- Analytical formula: ∂PDF/∂λ = exp(-λt)(1 - λt) = **-0.1353**
- Numerical gradient (finite differences): **-0.1381**

**Actual**:
- FFI gradient: **+0.0381** ❌ WRONG SIGN AND MAGNITUDE!

**Error**: 1.73e-01 (17% relative error)

---

## Investigation Progress

### ✅ Fixed: GraphBuilder Bugs

Successfully fixed two GraphBuilder bugs that were causing zero gradients:

1. **Parameterized edges created as constant edges**
   - GraphBuilder called `add_edge()` instead of `add_edge_parameterized()`
   - Fixed by using `add_edge_parameterized()` with coefficient arrays

2. **Edge weights not updated with theta values**
   - GraphBuilder didn't call `update_weights_parameterized()` after creating edges
   - Fixed by adding `g.update_weights_parameterized(theta_vec)` call

**Result**: Forward pass now works correctly, PMF varies with theta ✓

### ⚠️ Current Issue: Gradient Sign/Magnitude Bug

Even after GraphBuilder fixes, gradients still have wrong sign and magnitude. This is a SEPARATE bug in the C gradient computation code.

---

## Code Locations

### Gradient Computation

**File**: `src/c/phasic.c`

**Functions**:
- `compute_pmf_with_gradient()` - lines 6190-6389
- `ptd_graph_pdf_with_gradient()` - lines 6404-6483

**FFI Handler**:
- `ComputePmfAndMomentsWithGradientFfiImpl()` - `src/cpp/parameterized/ffi_handlers.cpp:318-433`

---

## Historical Context

### Phase 5 Week 3 (October 2025)

According to `sandbox/PHASE5_WEEK3_COMPLETE.md`:

- Implementation was **COMPLETE and TESTED**
- Test results: error = 0.00e+00 (machine precision!)
- Single exponential gradient: **-0.13533528** (CORRECT!)
- Code location: lines 4727-4918 (OLD line numbers)

### Current Status (November 2025)

- Same function exists at lines 6190-6483 (line numbers shifted)
- Contains same "CRITICAL" comment about zeroing absorbed probability
- But gradients now have WRONG sign!

**Question**: Was the code changed after Phase 5 Week 3? Or is there a different code path?

---

## Technical Analysis

### Gradient Computation Flow

1. **Uniformization**: Convert continuous-time to discrete-time jumps
   - Uniformization rate: λ = max exit rate
   - Discrete time step: dt = 1/λ

2. **Forward Algorithm**: Compute PMF via dynamic programming
   - Track probability at each vertex: P[v, k] at step k
   - Track probability gradients: ∂P[v, k]/∂θ

3. **PMF → PDF Conversion** (line 6474-6479):
   ```c
   *pdf_value = pmf * lambda;
   for (size_t i = 0; i < n_params; i++) {
       pdf_gradient[i] = pmf_grad[i] * lambda;
   }
   ```

### Potential Issues

#### Issue 1: Missing Product Rule Term

The PDF gradient formula should account for λ depending on parameters:

```
PDF(t; θ) = PMF(t; θ) · λ(θ)

∂PDF/∂θ = ∂PMF/∂θ · λ + PMF · ∂λ/∂θ  [Product rule]
```

**Current code** (line 6478):
```c
pdf_gradient[i] = pmf_grad[i] * lambda;  // Only first term!
```

**Missing**: `PMF · ∂λ/∂θ` term

For single exponential with λ = θ:
- ∂λ/∂θ = 1
- Missing term = PMF · 1 = 0.135

But adding this gives: 0.038 + 0.135 = 0.173 (still wrong sign!)

#### Issue 2: Lambda Dependencies

Lambda appears in multiple places:
1. Uniformization rate (dividing edge weights)
2. Poisson probabilities: Poisson(k; λt)
3. PDF conversion: PDF = PMF · λ

Gradients must account for ALL these dependencies via chain rule. This is complex!

#### Issue 3: Coefficient Array Interpretation

After GraphBuilder fix, edges now store full coefficient arrays with `coefficients_length = param_length`. The gradient code checks:

```c
if (edge->coefficients_length > 1) {
    // Parameterized edge
    for (size_t i = 0; i < n_params && i < edge->coefficients_length; i++) {
        weight += edge->coefficients[i] * params[i];
        exit_rate_grad[i] += edge->coefficients[i];  // Gradient
    }
}
```

**Question**: Is this gradient formula correct for how GraphBuilder now stores coefficients?

---

## Test Results

### Forward Pass (PDF) ✓

**Single Exponential**:
- Expected: 0.2706705665
- Computed: 0.2706521549
- Error: 1.84e-05 ✓ GOOD!

**Rabbits Model**:
- PMF varies correctly with all 3 parameters
- No caching issues

### Gradients ❌

**Single Exponential** (λ=2.0, t=1.0):
| Method | ∂PDF/∂λ | Error |
|--------|---------|-------|
| Analytical | -0.1353 | - |
| Numerical | -0.1381 | 2% |
| FFI | +0.0381 | 173% ❌ |

**Rabbits Model**:
- Gradients are non-zero (not all zeros anymore)
- But don't match numerical gradients
- Sign and magnitude errors

---

## Next Steps

### Option 1: Compare with Phase 5 Week 3 Code

1. Check git history to see if `compute_pmf_with_gradient()` was changed after Oct 2025
2. If yes, revert to working version
3. If no, investigate why the same code now produces different results

### Option 2: Fix Gradient Formula

1. Add missing product rule term: `PMF · ∂λ/∂θ`
2. Account for λ dependencies in Poisson probabilities
3. Verify chain rule through uniformization

### Option 3: Run Original Phase 5 Week 3 Tests

1. Find `test_single_exp_grad.c` test file
2. Compile and run to see if it still passes
3. If fails: regression was introduced
4. If passes: different code path being used

---

## Complexity Assessment

**Gradient computation for uniformization-based phase-type distributions is COMPLEX**:

- Chain rule through discrete-time DP
- λ appears in multiple places
- Poisson weighting of gradients
- Product rule for PDF = PMF · λ conversion

**Recommendation**:
- Use existing tests from Phase 5 Week 3 to validate
- Compare working vs broken code carefully
- Avoid quick fixes - need proper understanding of gradient math

---

## Principles

Following "NO QUICK FIXES, NO REGRESSIONS, NO FALLBACKS":

- ❌ Do NOT add arbitrary sign flip
- ❌ Do NOT add magic scaling factors
- ✓ DO understand the mathematical formula
- ✓ DO find root cause
- ✓ DO verify with analytical test cases

---

## Related Documents

- `GRAPHBUILDER_FIX_COMPLETE.md` - GraphBuilder fixes (forward pass now works)
- `sandbox/PHASE5_WEEK3_COMPLETE.md` - Original working implementation
- `CONTINUATION_PROMPT.md` - Context for this session

---

**Status**: Investigation ongoing. Need to identify why Phase 5 Week 3 code that was working is now producing wrong gradients.
