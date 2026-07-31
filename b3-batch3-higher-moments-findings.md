# Batch-3 (higher moments) — exact moment-VECTOR gradient — COMPLETE

Extends the exact reverse-mode gradient from moment[0] only (Batch-2) to the FULL
moment vector, fixing the mixed-scale FD defect for every moment.

## Algorithm: the moment-chain reverse (de-risked build-free, 230/230 vs JAX)
The recurrence (graph_builder.cpp:512) is a_1=ewt(ones), a_{j+1}=ewt(a_j),
m_k=(k+1)!·a_{k+1}[0], and every ewt replays the SAME numeric tape with a new
seed. So the reverse is a CHAIN: reversing replay j with output cotangent bar_a_j
yields (i) dm[c] += adj[from]·snap_to_j[c] into a SHARED dm[], and (ii) the
seed-adjoint adj[] = bar on a_{j-1}, which becomes replay j-1's output cotangent
(plus its own j!·ḡ_{j-1} at the start vertex). Stage-2 (param reverse → edge grads
→ edge→θ contraction) runs once per output moment on the accumulated dm[].
Verified in `experiments/dr_moment_chain_adjoint.py` (230/230 random cases vs
jax.jacobian, worst 2.72e-16).

## C (commit 9cdaa173): `ptd_moments_grad_theta` / `Graph._moments_grad_theta`
Env-free/thread-safe (same clean-`_off` acquisition as Batch-2a). Runs K replays
storing per-replay seeds + snap_to, then per output moment does the chain reverse
→ stage-2 → contraction → a row of the Jacobian d[m]/dθ (returned flat, row-major
nr_moments×param_length). Gate `experiments/dr_moments_jac_gate.py`: exact ==
closed-form dm1/dt on the two-stage model at EVERY scale incl θ=[1,1e-8]
([-2e8,-4e24] exact), == native θ-CD at benign scales; 2-cycle exact == θ-CD.

## JAX wiring (this commit): full moments Jacobian in model_bwd
Generalized the Batch-2b moment-0-only swap: model_bwd now computes the whole
Jacobian J (nr_moments × n_params) via one host `pure_callback`, and the exact
moments contribution to θ_bar is `J^T · g_moments`. The loop swaps the entire
moments FD term for `(J^T·g_moments)_i` (pmf stays FD), with FD fallback when the
C path is not-applicable (NaN). Default (`exact_moment_grad=False`) unchanged.

## Result: BOTH moment gradients exact at mixed scale
`tests/pytest/inference/test_fd_gradient_mixed_scale.py`:
- `test_fd_gradient_correct_at_mixed_scale` (moment 0) — still PASSES.
- `test_exact_second_moment_gradient_correct_at_mixed_scale` (NEW) — the exact
  d(E[T^2])/dθ matches the closed form to rtol=1e-9 at θ=[1,1e-8], where FD is
  ~359% wrong (`test_fd_second_moment_gradient_broken_without_fix` pins that).
- 7 passed, 1 xfailed (daisy — out of scope).
- jax.grad AND jax.vmap(jax.grad) of moment[1] exact (relerr ~1e-16) → SVGD-safe.

## No regression
Default-path pmf_and_moments tests identical master vs worktree (7 passed).

## Remaining Batch-3 items (each own de-risk; not done here)
was_dph/discrete (sibling-edge renorm quotient), log/formula Jacobian, joint-index
(reverse-over-reverse), hierarchical SCC, MPFR gate (cond>1e12 → FD fallback).
Plus the eventual make-exact-DEFAULT decision and an optional native FFI gradient
handler (perf).
