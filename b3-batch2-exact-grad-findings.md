# Batch-2 — exact first-moment gradient wired into pmf_and_moments_from_graph — PIN FLIPPED

Replaces the finite-difference d(moments[0])/dθ with the exact reverse-mode
θ-adjoint, as an OPT-IN with FD fallback. Default behaviour is unchanged.

## 2a — production C gradient (commit 85df40c0)
`ptd_moment0_grad_theta` / `Graph._moment0_grad_theta`: env-free / thread-safe.
Builds a LOCAL param-tape recorder (mem clean because the numeric executor never
runs on it) → converts to `_off` → runs the validated reverse tape adjoint for
dQ/d(edge weight) → contracts to dθ via the linear edge Jacobian
`dw_e/dθ_j = coefficients[j]` (input_specs `{kind=EDGE, v, e}`). Returns -1 (FD
fallback) for non-param / external / log-formula / non-finite.
Gate (`experiments/dr_moment0_theta_gate.py`): dθ == closed-form on the pin's own
two-stage model at EVERY scale incl. θ=[1,1e-8] → [-1, -1e16] exact; == θ-CD on
the 2-cycle.

## 2b — JAX wiring (this commit)
`pmf_and_moments_from_graph(..., exact_moment_grad=False)`: new opt-in kwarg. When
True + continuous + weight_mode='linear':
- captures a private `graph.clone()` and a host `_exact_moment0_grad_np(theta)`
  callback that sets weights and reads `_moment0_grad_theta` (1D and vmap-batched
  2D theta; returns NaNs → per-θ FD fallback);
- `model_bwd` computes `_exact0 = pure_callback(..., vmap_method='expand_dims')`
  once, then swaps ONLY the first moment's FD term:
  `grad_moments_i += g_moments[0]*(exact0[i] - moments_diff[0])`, guarded by
  `jnp.all(isfinite(exact0))` (elementwise FD fallback). pmf + higher moments stay
  FD. Default (kwarg False) → `_exact0 is None` → byte-identical to the old FD path.

## Result: the committed mixed-scale pin is FLIPPED
`tests/pytest/inference/test_fd_gradient_mixed_scale.py`:
- `test_fd_gradient_correct_at_mixed_scale` — was `xfail(strict=True)`; now PASSES:
  `jax.grad(moments[0])` with `exact_moment_grad=True` matches the oracle
  `[-1/t0^2, -1/t1^2]` to rtol=1e-9 at θ=[1,1e-8] (FD was 9% wrong).
- added `test_fd_gradient_still_broken_at_mixed_scale_without_fix` — documents the
  DEFAULT (FD) backward stays defective (opt-in, so default unchanged).
- 5 passed, 1 xfailed (the daisy-chain FD, out of scope — its own FFI impls).
- `jax.grad` AND `jax.vmap(jax.grad(...))` both exact (relerr ~1e-16) — the exact
  path works under SVGD-style batching.

## No regression (verified adversarially)
- default-path tests using pmf_and_moments (`test_gate_persistent_graph_reuse`,
  `test_svgd_fixed_fd_skip`): 7 passed, IDENTICAL on master and worktree.
- The 9 `test_jax_integration.py` failures are PRE-EXISTING: they fail at the
  existing `param_length == 0` check (:6913, before any B3 code) on the worktree's
  older fixtures — master SKIPS these same tests.

## Scope / follow-ups
- First moment only, continuous, weight_mode='linear', monolithic, no was_dph.
- Higher moments (seed chain), was_dph/discrete, log/formula, joint-index,
  hierarchical SCC, MPFR gate → Batch-3 (each its own de-risk).
- Making exact the DEFAULT (vs opt-in) is a later decision once the above land.
- A native FFI gradient handler (vs the host pure_callback) is a perf option; the
  callback is correct and already vmap-safe for the pin + SVGD batching.
