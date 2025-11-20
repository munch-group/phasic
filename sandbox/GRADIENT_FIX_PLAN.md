# Gradient Fix Plan: Complete Mathematical Analysis

**Date**: 2025-11-16
**Status**: ANALYSIS COMPLETE - Ready for implementation
**Priority**: CRITICAL - Blocks pmap implementation

---

## Executive Summary

FFI gradients have incorrect sign and magnitude due to **missing derivative terms in the uniformization-based gradient computation**. The current implementation only tracks gradients through probability evolution but ignores:

1. **Poisson weight gradients**: ∂Poisson(k; λt)/∂θ where λ = λ(θ)
2. **Lambda gradient in PDF conversion**: ∂λ/∂θ term in PDF = PMF · λ

Both require implementing the full chain rule through the uniformization scheme.

---

## Root Cause: Mathematical Analysis

### Current Implementation

```c
// Current code (INCOMPLETE):
PDF(t; θ) = λ · PMF(t; θ, λ)

where PMF(t; θ, λ) = Σ_k Poisson(k; λt) · P_k(θ, λ)

// Current gradient computation:
∂PDF/∂θ = λ · Σ_k Poisson(k; λt) · ∂P_k/∂θ
```

**Missing Terms**:
1. Gradient of Poisson weights (λ depends on θ!)
2. Gradient of λ in PDF = PMF · λ conversion

### Correct Formula

```
PDF(t; θ) = λ(θ) · Σ_k Poisson(k; λ(θ)t) · P_k(θ, λ(θ))

∂PDF/∂θ = ∂λ/∂θ · PMF
        + λ · Σ_k [∂Poisson(k; λt)/∂θ · P_k + Poisson(k; λt) · ∂P_k/∂θ]
```

Breaking down the terms:

**Term 1**: `∂λ/∂θ · PMF`
- λ = max exit rate across all vertices
- For vertex v: exit_rate_v = Σ_edges c_ij · θ_j
- ∂λ/∂θ_i = c_ij if vertex with max rate has edge with coeff c_ij for θ_i, else 0
- This is the **product rule** term from PDF = λ · PMF

**Term 2**: `λ · Σ_k ∂Poisson(k; λt)/∂θ · P_k`
- Poisson(k; μ) = μ^k · exp(-μ) / k! where μ = λt
- ∂Poisson/∂λ = Poisson(k; λt) · t · (k/λt - 1) = Poisson(k; λt) · (k - λt) / λ
- ∂Poisson/∂θ_i = (∂Poisson/∂λ) · (∂λ/∂θ_i)
- This accounts for **parameter dependence in Poisson weights**

**Term 3**: `λ · Σ_k Poisson(k; λt) · ∂P_k/∂θ`
- This is what the **current code computes** ✓
- Gradient of absorption probability at step k
- Already implemented correctly

### Why Current Code is Wrong

**Current code** (line 6367):
```c
pmf_gradient[p] += poisson_k * prob_grad[i][p];  // Only Term 3!
```

**Current code** (line 6478):
```c
pdf_gradient[i] = pmf_grad[i] * lambda;  // Still missing Terms 1 & 2!
```

**Missing**:
- Term 1: `∂λ/∂θ · PMF`
- Term 2: `Σ_k (∂Poisson/∂θ) · P_k`

---

## Implementation Strategy

### Option A: Full Derivative (Mathematically Correct)

Implement all three terms explicitly.

**Advantages**:
- Mathematically correct
- Will match analytical gradients exactly

**Disadvantages**:
- Complex implementation
- Need to track which vertex achieves max rate
- Need to compute ∂Poisson/∂λ for each k

**Complexity**: HIGH (~6-8 hours implementation + testing)

### Option B: Fixed-Lambda Approximation (Phase 5 Week 3 Approach?)

Treat λ as a **fixed constant** independent of θ during gradient computation.

**Rationale**:
- If λ is chosen sufficiently large (e.g., λ = 1.5 · max exit rate), it stays constant even as θ varies slightly
- This eliminates Terms 1 and 2, keeping only Term 3
- Trade accuracy for simplicity

**Advantages**:
- Simple: Current code structure is mostly correct
- Fast: No additional gradient terms to compute
- May be "close enough" for optimization (SVGD cares about gradient direction, not exact magnitude)

**Disadvantages**:
- Not mathematically exact
- Approximation error depends on how much λ changes with θ
- May cause SVGD convergence issues

**Complexity**: LOW (~2 hours to verify and document)

### Option C: Reverse-Mode Auto-Diff (Clean Slate)

Rewrite gradient computation using reverse-mode automatic differentiation principles.

**Approach**:
1. Build computational graph for PDF(t; θ)
2. Implement backward pass that propagates adjoints
3. Handle uniformization as a differentiable operation

**Advantages**:
- Systematic approach
- Handles all dependencies automatically
- Can be verified against JAX auto-diff on small examples

**Disadvantages**:
- Requires significant refactoring
- Need to understand full computational graph
- Time-consuming

**Complexity**: VERY HIGH (~2-3 days)

---

## Recommended Approach: Option A (Full Derivative)

**Justification**:
1. Following "NO QUICK FIXES" principle
2. Phase 5 Week 3 achieved machine precision (error=0.00e+00), suggesting full derivative was implemented
3. SVGD requires accurate gradients for correct posterior inference
4. One-time implementation cost worth correctness guarantee

---

## Implementation Plan

### Step 1: Add Lambda Gradient Computation

**File**: `src/c/phasic.c`
**Location**: Inside `ptd_graph_pdf_with_gradient()` after line 6443

**Code**:
```c
// Compute ∂λ/∂θ (gradient of uniformization rate)
double *lambda_grad = (double *)calloc(n_params, sizeof(double));
if (lambda_grad == NULL) {
    return -1;
}

// Find vertex that achieves max exit rate (determines λ)
size_t max_vertex_idx = 0;
for (size_t i = 0; i < graph->vertices_length; i++) {
    struct ptd_vertex *v = graph->vertices[i];
    double exit_rate = 0.0;

    for (size_t j = 0; j < v->edges_length; j++) {
        struct ptd_edge *e = v->edges[j];
        double weight = 0.0;
        if (e->coefficients_length > 1) {
            for (size_t k = 0; k < n_params && k < e->coefficients_length; k++) {
                weight += e->coefficients[k] * params[k];
            }
        } else {
            weight = e->coefficients[0];
        }
        exit_rate += weight;
    }

    if (exit_rate >= lambda - 1e-10) {  // Account for floating point
        max_vertex_idx = i;
        break;  // Found the vertex that determines λ
    }
}

// Compute ∂λ/∂θ from edges of max_vertex
struct ptd_vertex *max_v = graph->vertices[max_vertex_idx];
for (size_t j = 0; j < max_v->edges_length; j++) {
    struct ptd_edge *e = max_v->edges[j];
    if (e->coefficients_length > 1) {
        for (size_t k = 0; k < n_params && k < e->coefficients_length; k++) {
            lambda_grad[k] += e->coefficients[k];
        }
    }
}
```

### Step 2: Add Poisson Gradient Terms

**File**: `src/c/phasic.c`
**Location**: Inside `compute_pmf_with_gradient()` loop at line 6360-6376

**Modify accumulation** (line 6366-6368):
```c
for (size_t p = 0; p < n_params; p++) {
    // Original term: Poisson · ∂P/∂θ
    pmf_gradient[p] += poisson_k * prob_grad[i][p];

    // NEW: Add Poisson gradient term: (∂Poisson/∂θ) · P
    // ∂Poisson(k; λt)/∂θ = Poisson · (k - λt) / λ · ∂λ/∂θ
    double poisson_grad_factor = poisson_k * ((double)k - lambda_t) / lambda;
    pmf_gradient[p] += poisson_grad_factor * lambda_grad[p] * prob[i];
}
```

**Note**: Need to pass `lambda_grad` as parameter to `compute_pmf_with_gradient()`.

### Step 3: Add PDF Conversion Gradient

**File**: `src/c/phasic.c`
**Location**: Line 6474-6479

**Modify**:
```c
// Convert PMF to PDF with full gradient
*pdf_value = pmf * lambda;

for (size_t i = 0; i < n_params; i++) {
    // Original: pdf_gradient[i] = pmf_grad[i] * lambda;

    // Correct (product rule): ∂(PMF·λ)/∂θ = ∂PMF/∂θ · λ + PMF · ∂λ/∂θ
    pdf_gradient[i] = pmf_grad[i] * lambda + pmf * lambda_grad[i];
}

free(lambda_grad);
```

### Step 4: Update Function Signatures

**File**: `src/c/phasic.c`

**Modify** `compute_pmf_with_gradient()` signature (line 6190):
```c
static int compute_pmf_with_gradient(
    struct ptd_graph *graph,
    double time,
    double lambda,
    double *lambda_grad,  // NEW parameter
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pmf_value,
    double *pmf_gradient
)
```

### Step 5: Testing Strategy

**Test 1: Single Exponential** (analytical solution available)
```
Model: v0 --(θ)--> v_abs
PDF(t; θ) = θ · exp(-θt)
∂PDF/∂θ = exp(-θt) · (1 - θt)

At t=1.0, θ=2.0:
Expected PDF = 0.2706705665
Expected gradient = -0.1353352832
```

**Test 2: Erlang Distribution** (analytical solution available)
```
Model: v0 --(θ₁)--> v1 --(θ₂)--> v_abs
PDF(t; θ₁, θ₂) = θ₁·θ₂/(θ₂-θ₁) · (exp(-θ₁t) - exp(-θ₂t))
```

**Test 3: Rabbits Model** (compare with numerical gradients)
```
Use finite differences with ε=1e-6
Verify gradient magnitude and direction match
```

### Step 6: Verification Criteria

✅ **PASS** if:
1. Single exponential: |gradient_error| < 1e-10 (machine precision)
2. Erlang: |gradient_error| < 1e-10
3. Rabbits: |gradient_error| < 1e-4 (compared to numerical)
4. All tests: Gradient has correct sign
5. SVGD: Converges to correct posterior on rabbits tutorial

❌ **FAIL** if:
- Any gradient has wrong sign
- Error > 1% on analytical test cases
- SVGD doesn't converge or converges to wrong posterior

---

## Estimated Effort

**Implementation**: 4-6 hours
- Step 1 (Lambda gradient): 1-2 hours
- Step 2 (Poisson gradient): 2-3 hours
- Step 3 (PDF conversion): 30 min
- Steps 4-5 (Signatures + cleanup): 30 min

**Testing**: 2-3 hours
- Create test cases: 1 hour
- Debug and verify: 1-2 hours

**Total**: 6-9 hours

---

## Risk Mitigation

**Risk 1**: Implementation complexity leads to bugs
- **Mitigation**: Test incrementally (add one term at a time)
- **Mitigation**: Compare each term's contribution against numerical estimate

**Risk 2**: Still doesn't match analytical gradients
- **Mitigation**: Document Phase 5 Week 3 code for comparison
- **Mitigation**: Run original tests if they exist

**Risk 3**: Performance degradation
- **Mitigation**: Profile before/after
- **Mitigation**: Lambda gradient computation is O(vertices), negligible overhead

---

## Alternative: Investigate Phase 5 Week 3 Code

**Before implementing**, check if Phase 5 Week 3 code already has the fix:

1. Search git history for October 2025 commits
2. Compare `compute_pmf_with_gradient()` between versions
3. If different, identify what changed
4. If same, investigate why it worked then but not now

**Time**: 1-2 hours investigation could save 6-9 hours implementation

---

## Decision Point

**Recommendation**:
1. Spend 1-2 hours investigating Phase 5 Week 3 code first
2. If that doesn't resolve it, implement Option A (full derivative)
3. Do NOT use approximations or quick fixes

**Rationale**: Following "NO QUICK FIXES" principle. Gradient correctness is critical for SVGD.

---

## References

- `sandbox/PHASE5_WEEK3_COMPLETE.md`: Original working implementation
- `GRAPHBUILDER_FIX_COMPLETE.md`: Forward pass fixes
- `GRADIENT_SIGN_BUG_ANALYSIS.md`: Initial investigation
- [Munch et al. 2024] Phase-type distributions paper (Section 3.2: Uniformization)

---

**Next Action**: Investigate Phase 5 Week 3 git history to see if simpler solution exists.
