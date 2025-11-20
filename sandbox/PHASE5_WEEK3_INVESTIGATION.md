# Phase 5 Week 3 Code Investigation

**Date**: 2025-11-16
**Status**: Investigation COMPLETE - Proceeding with gradient fix implementation

---

## Question

Did Phase 5 Week 3 implementation (October 2025) have correct gradients, or was there an error in testing/documentation?

---

## Findings

###  1. Code is IDENTICAL Between October and November

Compared October commit (d8b5893) with current code:

**Gradient accumulation** (October line 4990 vs Current line 6367):
```c
pmf_gradient[p] += poisson_k * prob_grad[i][p];  // IDENTICAL
```

**PDF conversion** (October vs Current):
```c
pdf_gradient[i] = pmf_grad[i] * lambda;  // IDENTICAL
```

The gradient computation formula has NOT changed since October.

### 2. Phase 5 Week 3 Documentation Claims

From `sandbox/PHASE5_WEEK3_COMPLETE.md`:
```
Test Results:
  PDF: computed=0.27067057, analytical=0.27067057, error=0.00e+00
  Gradient: computed=-0.13533528, analytical=-0.13533528, error=0.00e+00
  ✓ PASSED
```

### 3. Current Test Results (Same Model)

From `/tmp/test_single_exp_gradient.py`:
```
Expected (analytical): ∂PDF/∂λ = -0.1353352832
Numerical:             ∂PDF/∂λ = -0.1380878341
FFI:                   ∂PDF/∂λ = +0.0381264085  ❌ WRONG SIGN!
Error: 1.73e-01 (173% error)
```

### 4. Test File Missing

The C test file `tests/test_single_exp_grad.c` mentioned in Phase 5 Week 3 docs does NOT exist in current codebase.

```bash
$ find . -name "test_single_exp_grad.c"
# No results
```

---

## Analysis

### Possible Explanations

**Hypothesis 1**: Test was incorrect/misleading
- Maybe the October test had a bug that made it appear to pass
- Perhaps it tested a different formula or model
- The "error=0.00e+00" claim might be an error

**Hypothesis 2**: Code path changed
- Different function was being called in October
- FFI wrapper behavior changed
- GraphBuilder changes affected how parameters flow through

**Hypothesis 3**: Mathematical formula was always incomplete
- The Phase 5 Week 3 implementation had the same mathematical bug
- The test never actually validated against analytical gradients correctly
- Documentation error: reported wrong expected values

### Evidence Supporting Hypothesis 3

1. **Code formula is mathematically incomplete**:
   Current code only computes Term 3 of the full derivative:
   ```
   ∂PDF/∂θ = ∂λ/∂θ · PMF                     [Term 1 - MISSING]
           + λ · Σ_k (∂Poisson/∂θ) · P_k      [Term 2 - MISSING]
           + λ · Σ_k Poisson · (∂P_k/∂θ)     [Term 3 - CURRENT CODE]
   ```

2. **Gradient has wrong sign**:
   For single exponential with θ=2.0, t=1.0:
   - Expected: negative (-0.135)
   - Computed: positive (+0.038)
   - This is not a small numerical error - it's fundamentally wrong

3. **Same code, different claims**:
   Identical formula claimed to work in October but fails now

### Most Likely Conclusion

The Phase 5 Week 3 gradient implementation was **mathematically incomplete** from the start. Either:
- The test had bugs that made it appear to pass
- The documentation overstated the accuracy
- There was confusion about what was actually being tested

The current mathematical analysis in `GRADIENT_FIX_PLAN.md` is sound and identifies the missing derivative terms.

---

## Decision

**Proceed with implementing the full derivative formula** as planned in GRADIENT_FIX_PLAN.md:

1. Add lambda gradient computation: `∂λ/∂θ`
2. Add Poisson gradient terms: `Σ_k (∂Poisson(k; λt)/∂θ) · P_k`
3. Add product rule term in PDF conversion: `PMF · ∂λ/∂θ`

**Rationale**:
- Mathematical analysis is clear and rigorous
- Gradient error (173%) is far too large to be numerical noise
- Following "NO QUICK FIXES" principle - need proper solution
- Historical code investigation didn't reveal a working alternative

---

## Key Insight

The uniformization-based gradient computation is COMPLEX:

```
PDF(t; θ) = λ(θ) · Σ_k Poisson(k; λ(θ)t) · P_k(θ, λ(θ))
```

Because λ depends on θ:
- λ = max exit rate = max_v Σ_edges c_ij · θ_j
- This means λ appears in THREE places, each requiring chain rule
- Cannot treat λ as constant during differentiation

The current code treats λ as if it's independent of θ, which gives completely wrong results.

---

## Next Steps

1. ✅ Investigation complete - no working alternative found
2. ⏭️ Implement gradient fix following GRADIENT_FIX_PLAN.md
3. ⏭️ Test with analytical formulas (single exp, Erlang)
4. ⏭️ Verify SVGD convergence

---

**Estimated Time**: 6-9 hours for implementation + testing (as per GRADIENT_FIX_PLAN.md)

---

**Conclusion**: Phase 5 Week 3 likely had the same bug. Proceeding with proper mathematical fix.
