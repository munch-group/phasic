# Phase 5 Week 3: Option 2 Implementation Plan
## Analytical Poisson Derivative Approach

**Status**: Future optimization (use after Option 1 is validated)
**Estimated effort**: 3-4 days
**Performance gain**: 2× faster than Option 1

---

## 1. Mathematical Derivation

### 1.1 Forward Algorithm PDF Formula

The forward algorithm computes PDF using uniformization:

```
PDF(t|θ) = Σₖ Poisson(λ(θ)·t, k) · Pₖ(absorbing|θ)
```

Where:
- `λ(θ)` = max exit rate (depends on parameters)
- `Pₖ(absorbing|θ)` = probability at absorbing states after k uniformized jumps
- `Poisson(μ, k) = μᵏ e⁻ᵘ / k!`

### 1.2 Gradient Derivation

Taking derivative w.r.t. parameter θᵢ:

```
∂PDF/∂θᵢ = Σₖ [∂Poisson(λt,k)/∂θᵢ · Pₖ(abs) + Poisson(λt,k) · ∂Pₖ(abs)/∂θᵢ]
```

This requires two terms:

#### Term 1: Poisson derivative

```
∂Poisson(λt,k)/∂θᵢ = ∂Poisson/∂(λt) · ∂(λt)/∂θᵢ
                    = ∂Poisson/∂(λt) · t · ∂λ/∂θᵢ
```

Using Poisson derivative identity:
```
∂Poisson(μ,k)/∂μ = Poisson(μ,k-1) - Poisson(μ,k)
```

Therefore:
```
∂Poisson(λt,k)/∂θᵢ = t · ∂λ/∂θᵢ · [Poisson(λt,k-1) - Poisson(λt,k)]
```

#### Term 2: Probability derivative

This is what we already compute: `∂Pₖ(abs)/∂θᵢ` via chain rule through DP.

### 1.3 λ(θ) Derivative

Since `λ(θ) = max_v exit_rate_v(θ)`, we need:

```
∂λ/∂θᵢ = {
    ∂exit_rate_v*/∂θᵢ   if v* achieves the max and is unique
    ?                     if multiple vertices achieve max (non-differentiable!)
}
```

For vertex v with edges having parameterized weights:
```
exit_rate_v(θ) = Σₑ [base_weight_e + Σⱼ coeff_e,j · θⱼ]

∂exit_rate_v/∂θᵢ = Σₑ coeff_e,i
```

---

## 2. Implementation Strategy

### 2.1 High-Level Algorithm

```c
int ptd_graph_pdf_with_gradient_analytical(
    struct ptd_graph *graph,
    double time,
    size_t granularity,
    const double *params,
    size_t n_params,
    double *pdf_value,
    double *pdf_gradient
) {
    // Step 1: Compute λ and ∂λ/∂θᵢ
    double lambda;
    double *lambda_grad = alloc(n_params);
    compute_lambda_with_gradient(graph, params, n_params, &lambda, lambda_grad);

    // Step 2: Precompute Poisson terms
    double *poisson = malloc(max_jumps);
    double *poisson_deriv = malloc(max_jumps);  // Poisson(k-1) - Poisson(k)
    precompute_poisson_terms(lambda * time, max_jumps, poisson, poisson_deriv);

    // Step 3: Forward algorithm with gradient tracking
    double *prob = calloc(n_vertices);
    double **prob_grad = alloc_2d(n_vertices, n_params);
    prob[0] = 1.0;  // Starting probability

    *pdf_value = 0.0;
    for (int i = 0; i < n_params; i++) pdf_gradient[i] = 0.0;

    // Step 4: DP loop
    for (size_t k = 0; k < max_jumps; k++) {
        // Update probabilities via edges (same as Option 1)
        step_forward_with_gradients(graph, params, n_params, lambda,
                                    prob, prob_grad, next_prob, next_prob_grad);

        // Accumulate PDF contribution
        double P_k_abs = next_prob[absorbing_idx];
        double *dP_k_abs = next_prob_grad[absorbing_idx];

        // Term 1: ∂Poisson/∂θᵢ · Pₖ
        for (int i = 0; i < n_params; i++) {
            double poisson_grad_i = time * lambda_grad[i] * poisson_deriv[k];
            pdf_gradient[i] += poisson_grad_i * P_k_abs;
        }

        // Term 2: Poisson · ∂Pₖ/∂θᵢ
        *pdf_value += poisson[k] * P_k_abs;
        for (int i = 0; i < n_params; i++) {
            pdf_gradient[i] += poisson[k] * dP_k_abs[i];
        }

        // Swap buffers
        swap(prob, next_prob);
        swap(prob_grad, next_prob_grad);
    }

    // Cleanup
    free(lambda_grad);
    free(poisson);
    free(poisson_deriv);
    // ... etc
}
```

### 2.2 Key Subroutines

#### compute_lambda_with_gradient()

```c
void compute_lambda_with_gradient(
    struct ptd_graph *graph,
    const double *params,
    size_t n_params,
    double *lambda,
    double *lambda_grad
) {
    *lambda = 0.0;
    size_t max_vertex = 0;

    // Find vertex with maximum exit rate
    for (size_t v = 0; v < graph->vertices_length; v++) {
        double exit_rate = 0.0;
        for (size_t e = 0; e < vertex->edges_length; e++) {
            exit_rate += evaluate_edge_weight(edge, params, n_params);
        }

        if (exit_rate > *lambda) {
            *lambda = exit_rate;
            max_vertex = v;
        }
    }

    // Compute gradient at maximizing vertex
    memset(lambda_grad, 0, n_params * sizeof(double));
    struct ptd_vertex *v_max = graph->vertices[max_vertex];

    for (size_t e = 0; e < v_max->edges_length; e++) {
        struct ptd_edge *edge = v_max->edges[e];
        if (edge->parameterized) {
            struct ptd_edge_parameterized *ep = (struct ptd_edge_parameterized *)edge;
            if (ep->state != NULL) {
                for (size_t i = 0; i < n_params; i++) {
                    lambda_grad[i] += ep->state[i];
                }
            }
        }
    }
}
```

**Critical issue**: When multiple vertices have the same max rate, λ is non-differentiable!
**Solution**: Use subgradient (average of gradients from all max vertices) or perturbation.

#### precompute_poisson_terms()

```c
void precompute_poisson_terms(
    double mu,  // lambda * time
    size_t max_k,
    double *poisson,
    double *poisson_deriv
) {
    // Poisson(μ, k) = μᵏ e⁻ᵘ / k!
    poisson[0] = exp(-mu);

    for (size_t k = 1; k < max_k; k++) {
        poisson[k] = poisson[k-1] * mu / k;
    }

    // Derivative: Poisson(μ, k-1) - Poisson(μ, k)
    poisson_deriv[0] = -poisson[0];  // Poisson(μ,-1) = 0
    for (size_t k = 1; k < max_k; k++) {
        poisson_deriv[k] = poisson[k-1] - poisson[k];
    }
}
```

---

## 3. Edge Cases and Special Handling

### 3.1 Non-differentiable λ(θ)

**Problem**: Multiple vertices with max exit rate → λ non-differentiable

**Solutions**:
1. **Subgradient approach**: Average gradients from all max vertices
   ```c
   if (exit_rate ≈ lambda) {  // within tolerance
       for (i = 0; i < n_params; i++) {
           lambda_grad_sum[i] += vertex_exit_grad[i];
       }
       n_max_vertices++;
   }
   // Then: lambda_grad[i] = lambda_grad_sum[i] / n_max_vertices
   ```

2. **Perturbation**: Add tiny random noise to break ties
   ```c
   exit_rate += 1e-12 * random();  // Break ties randomly
   ```

3. **Conservative gradient**: Use zero gradient (always valid but less informative)

**Recommendation**: Use subgradient (mathematically sound for SVGD).

### 3.2 Poisson(μ, -1) = 0

Handle k=0 case specially:
```c
poisson_deriv[0] = -poisson[0];  // Since Poisson(μ,-1) = 0
```

### 3.3 Numerical overflow in Poisson

For large k, use log-space computation:
```c
log_poisson[k] = -mu + k*log(mu) - lgamma(k+1);
poisson[k] = exp(log_poisson[k]);
```

---

## 4. Validation Strategy

### 4.1 Unit Tests

**Test 1: Verify Poisson derivatives**
```c
// Check: ∂Poisson(μ,k)/∂μ = Poisson(μ,k-1) - Poisson(μ,k)
double mu = 2.5;
for (k = 0; k < 50; k++) {
    double analytical = poisson_deriv[k];
    double fd = (poisson_at(mu + eps, k) - poisson_at(mu, k)) / eps;
    assert(|analytical - fd| < 1e-6);
}
```

**Test 2: Verify λ gradients**
```c
// Build graph with parameterized edges
// Compute ∂λ/∂θᵢ analytically
// Compare with finite differences
double lambda1 = compute_lambda(graph, params);
params[i] += eps;
double lambda2 = compute_lambda(graph, params);
double fd_grad = (lambda2 - lambda1) / eps;
assert(|lambda_grad[i] - fd_grad| < 1e-6);
```

**Test 3: Single exponential (analytical formula)**
```c
// PDF(t|θ) = θ·exp(-θ·t)
// ∂PDF/∂θ = exp(-θ·t) · (1 - θ·t)
double pdf, grad;
compute_pdf_with_grad_analytical(exponential_graph, t, θ, &pdf, &grad);
assert(|pdf - θ*exp(-θ*t)| < 1e-6);
assert(|grad - exp(-θ*t)*(1-θ*t)| < 1e-6);
```

**Test 4: Erlang distribution (analytical formula)**
```c
// Erlang(n, λ): n stages, each rate λ
// PDF = λⁿ·tⁿ⁻¹·exp(-λ·t) / (n-1)!
// ∂PDF/∂λ = PDF·[n/λ - t]
```

### 4.2 Comparison Tests

Compare Option 2 vs Option 1:
```c
// Both should give same answer (within numerical precision)
double pdf1, grad1[n_params];
option1_compute(graph, params, &pdf1, grad1);

double pdf2, grad2[n_params];
option2_compute(graph, params, &pdf2, grad2);

assert(|pdf1 - pdf2| < 1e-8);
for (i = 0; i < n_params; i++) {
    assert(|grad1[i] - grad2[i]| < 1e-8);
}
```

### 4.3 Regression Tests

Use Option 1 to generate "ground truth" for complex models:
```python
# Generate test cases with Option 1
for model in [coalescent_10, sfs_20, complex_graph_67]:
    for theta in test_parameter_sets:
        pdf_opt1, grad_opt1 = option1_compute(model, theta)
        save_test_case(model, theta, pdf_opt1, grad_opt1)

# Then verify Option 2 matches
for test_case in load_test_cases():
    pdf_opt2, grad_opt2 = option2_compute(test_case.model, test_case.theta)
    assert_close(pdf_opt2, test_case.pdf_opt1)
    assert_close(grad_opt2, test_case.grad_opt1)
```

---

## 5. Performance Analysis

### 5.1 Complexity

**Option 1 (CDF numerical)**:
- Time: 2 × O(k·m·p)
- Space: 2 × O(n·p)

**Option 2 (analytical)**:
- Time: O(k·m·p) + O(k·p) Poisson term
- Space: O(n·p) + O(k) for Poisson cache

**Speedup**: ~2× (ignoring Poisson overhead)

### 5.2 Benchmarks

Expected performance (67-vertex coalescent model, 2 parameters):

| Method | Time/eval | Time/1000 evals | Speedup |
|--------|-----------|-----------------|---------|
| Option 1 | ~10ms | ~10s | 1.0× |
| Option 2 | ~5ms | ~5s | 2.0× |

### 5.3 Memory

| Method | Memory/eval |
|--------|-------------|
| Option 1 | 2·n·p·8 bytes (two copies of prob_grad) |
| Option 2 | n·p·8 + k·16 bytes (prob_grad + Poisson cache) |

For n=67, p=2, k=1000:
- Option 1: 2·67·2·8 = 2.1 KB
- Option 2: 67·2·8 + 1000·16 = 17.1 KB

**Trade-off**: Option 2 uses 8× more memory for Poisson cache, but still negligible (<20KB).

---

## 6. Implementation Checklist

### Phase 1: Core Implementation (Day 1-2)
- [ ] Implement `compute_lambda_with_gradient()`
- [ ] Handle non-differentiable λ via subgradient
- [ ] Implement `precompute_poisson_terms()`
- [ ] Handle k=0 boundary case
- [ ] Implement log-space Poisson for large k
- [ ] Refactor main loop to accumulate both terms

### Phase 2: Testing (Day 3)
- [ ] Unit test: Poisson derivative formula
- [ ] Unit test: λ gradient vs finite differences
- [ ] Integration test: Single exponential
- [ ] Integration test: Erlang(3, λ)
- [ ] Integration test: 2-stage with different rates
- [ ] Comparison test: Option 2 vs Option 1 (10 test cases)

### Phase 3: Validation (Day 4)
- [ ] Run all Phase 3 benchmarks
- [ ] Verify 2× speedup achieved
- [ ] Test with 67-vertex coalescent model
- [ ] Profile memory usage
- [ ] Stress test: 500+ vertex model
- [ ] Verify gradients with SVGD convergence

### Phase 4: Documentation
- [ ] Add docstrings with mathematical formulas
- [ ] Document non-differentiable λ handling
- [ ] Add performance comparison to CLAUDE.md
- [ ] Update PHASE4_5_IMPLEMENTATION_PLAN.md

---

## 7. Potential Pitfalls

### 7.1 Numerical Issues

**Issue**: Poisson terms can underflow/overflow
**Solution**: Use log-space for extreme k values

**Issue**: Subtracting similar Poisson values loses precision
**Solution**: Use higher precision (long double) for Poisson computation

### 7.2 Implementation Bugs

**Common mistakes**:
1. Forgetting to multiply by `t` in Poisson derivative term
2. Off-by-one errors in k indexing (k=0 boundary)
3. Not handling λ=0 edge case
4. Forgetting negative sign in `∂exit_rate/∂θᵢ` for self-loop term

### 7.3 Mathematical Subtleties

**Issue**: λ(θ) discontinuous at ties
**Impact**: Gradients may "jump" when optimization crosses tie boundary
**Mitigation**: SVGD is robust to this (uses ensemble of particles)

---

## 8. Future Optimizations

Once Option 2 is working:

1. **Adaptive Poisson cutoff**: Stop when Poisson(λt, k) < ε·Poisson_max
2. **Sparse gradient tracking**: Only track ∂P/∂θᵢ for parameters affecting reachable edges
3. **Vectorized Poisson computation**: SIMD for Poisson array
4. **GPU acceleration**: Port to CUDA for massive parallelism

---

## 9. References

**Poisson derivative formula**:
- Ross, S. M. (2014). *Introduction to Probability Models*, 11th ed. Section 5.3.

**Forward algorithm for phase-type distributions**:
- Neuts, M. F. (1981). *Matrix-Geometric Solutions in Stochastic Models*. Chapter 2.

**Automatic differentiation through uniformization**:
- Original research by Røikjer, Hobolth & Munch (2022) - extended to gradients

**SVGD with non-differentiable functions**:
- Liu & Wang (2016). *Stein Variational Gradient Descent*. NIPS.
- Subgradients are valid for SVGD.

---

**End of Option 2 Plan**

*Created: 2025-10-16*
*Status: Ready for implementation after Option 1 validation*
*Estimated effort: 3-4 days*
