# B3 discrete/was_dph exact gradient — plan

Branch: `b3-discrete-theta-adjoint`, off master `a8906862` (continuous B3 exact
gradient, already merged & shipped opt-in via `exact_moment_grad=True`).

## Goal

Extend the exact reverse-mode θ-adjoint for phase-type moments
(`ptd_moments_grad_theta`, `src/c/phasic.c:10738`) to **discrete / `was_dph`**
graphs (created via `Graph.discretize()`), which currently fall back to FD
inside `pmf_and_moments_from_graph(..., exact_moment_grad=True)` via the
`not discrete` gate at `src/phasic/__init__.py:6931`.

Full math derivation: `b3-batch3-mpfr-and-discrete-derisk.md` (already on
master). Two independent pieces of new math, both θ-independent-structure
linear maps:

1. **Renorm edge→θ Jacobian** (sibling coupling). `was_dph` graphs
   renormalize `p_e = w_e/S_v` per vertex in `ptd_graph_update_weights`
   (`phasic.c:5789`), `S_v = Σ_{e'∈out(v)} w_{e'}`, `w_e = c_e·θ`. The
   existing tape adjoint gives `dQ/dp_e` (stage-1 reverse — **UNCHANGED**,
   provably renorm-agnostic: it only ever reads the current `edge->weight` as
   an opaque free variable, regardless of how that value relates to θ). The
   contraction step must become:
   `∂p_e/∂θ_j = (c_e^j − p_e·Σ_{e'∈out(v)} c_{e'}^j) / S_v`
   — this needs θ explicitly (to compute `S_v`), so the new function takes
   `theta` as an argument.
2. **Discrete moment correction**: `continuous_to_discrete_moments`
   (`graph_builder.cpp:694`) maps continuous raw power moments → discrete raw
   moments via a fixed (θ-independent) **linear** map built from
   factorial/binomial/Stirling-2 coefficients. Because it's linear,
   `d(discrete m)/dθ = C · d(continuous m)/dθ`, i.e. apply the *same* scalar
   recursion to each **column** (θ-index) of the continuous Jacobian
   independently.
3. Both `is_discrete` sub-cases must be handled: `was_dph=True`
   (`discretize()`, needs the quotient rule) vs `was_dph=False` (native DPH
   via `is_discrete=True` only, edge weight IS `c_e·θ` directly, no
   renormalization — same plain contraction as the continuous case). The
   discrete moment correction (piece 2) applies in **both** sub-cases. One C
   function branches internally on `graph->was_dph`.

## Batches (each gated before moving on)

**D0 — build-free math de-risk (Python/numpy/JAX only, no C, no rebuild)**
- D0.1 `experiments/dr_dph_renorm_jacobian.py`: verify the quotient-rule
  formula against `jax.jacobian` of `p_e(θ) = w_e/Σw` on synthetic
  multi-edge/multi-param vertices (several random seeds).
- D0.2 `experiments/dr_discrete_moment_correction.py`: Python port of
  `continuous_to_discrete_moments`; verify linearity and that per-column
  application to a Jacobian matches `jax.jacobian` of the composed map on
  random continuous-moment "functions of θ".
- Gate: both scripts print ALL PASS.

**D1 — C implementation**
- Port `d_factorial`/`d_binomial`/`d_stirling2` + the correction transform
  into `phasic.c`.
- New `int ptd_moments_grad_theta_dph(struct ptd_graph *graph, int
  nr_moments, const double *theta, size_t theta_len, double *J_out)`:
  reuses `ptd_moments_grad_theta`'s forward moment chain + reverse chain +
  stage-2 param-tape reverse verbatim (unchanged, renorm-agnostic), adds a
  precompute pass for per-vertex `S_v`/`ΣC_v[j]`, branches the final
  edge→theta contraction on `graph->was_dph`, then applies the discrete
  moment correction to each output column. Reuses the existing MPFR gate
  (`ptd_dbg_tape_needs_mpfr`).
- Declare in `api/c/phasic.h`.

**D1b — bindings**
- `Graph::moments_grad_theta_dph(nr_moments, theta)` in
  `api/cpp/phasiccpp.h` (mirrors `moments_grad_theta`).
- `_moments_grad_theta_dph` pybind binding in `src/cpp/phasic_pybind.cpp`.
- `pixi run install-dev` (production build, no validator flag needed — this
  is shipped code, not a validator).

**D2 — gate against native central-difference**
- `experiments/dr_dph_moments_jac_gate.py`: `_erlang().discretize(0.5)`
  (2 params) + a sibling-coupled multi-param variant (`_chain(2).discretize`,
  3 params) at several θ (benign + one mixed-scale), central-diff the
  **native** `graph.moments(K, discrete=True)` as oracle. Also both
  `was_dph` sub-cases (discretize()'d vs native DPH via `dph(...,
  set_discrete=True)` from `test_discrete_moments_and_reward.py`).
  NegBinomial closed form as a secondary cross-check if straightforward.
- No-regression: `dr_moments_jac_gate.py` + `dr_mpfr_gate_test.py`
  (continuous, untouched) + the mixed-scale pin file still pass.

**D3 — Python wiring**
- Relax the `_exact_grad_enabled` gate in `pmf_and_moments_from_graph`
  (`__init__.py:6931`) to use **effective discreteness**
  (`discrete or serialized.get('is_discrete', False)` — mirrors
  `GraphBuilder::compute_pmf_and_moments`'s `is_disc` dispatch exactly, so
  the exact-grad gate can't silently diverge from what the forward actually
  computes).
- `_exact_moments_jac_np` calls `_moments_grad_theta_dph` when effective
  discreteness is true, `_moments_grad_theta` otherwise. Verify
  `graph.clone()` preserves `is_discrete`/`was_dph`.
- Default (`exact_moment_grad=False`) stays byte-identical.

**D4 — tests**
- New `tests/pytest/inference/test_exact_grad_discrete.py`: exact grad ==
  native central-diff/closed-form for `was_dph` graphs incl. sibling
  coupling, grad+vmap (SVGD-safety), default-path unchanged, MPFR-decline
  fallback still triggers FD.
- Re-run `test_discrete_moments_and_reward.py` (forward unaffected) +
  `test_fd_gradient_mixed_scale.py` (continuous pin still flipped) — do NOT
  gate on full-suite-green (pre-existing unrelated failures in
  `test_jax_integration.py`, see `project_test_suite_state` memory).

**D5 — handoff**
- Write a merge-review doc (mirrors `B3-MERGE-REVIEW.md`); ask whether to
  squash-merge into master now.

## Constraints carried over from the continuous batch
- Never `git add -A` (rewrites deps + notebooks); commit explicit paths.
- `pixi run install-dev` after every C/Python source edit (install is a
  copy, not editable).
- Opt-in stays opt-in: `exact_moment_grad` default `False`, byte-identical
  FD.
- Don't gate on full-green pytest.
