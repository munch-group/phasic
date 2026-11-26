# B3 discrete/was_dph exact gradient — MERGE REVIEW & HANDOFF

Branch: **`b3-discrete-theta-adjoint`**, off master `a8906862` (the continuous
B3 exact gradient — `exact_moment_grad=True` — already shipped there).

---

## 1. TL;DR — what this delivers

Extends the exact reverse-mode θ-adjoint for phase-type moments to
**discrete graphs**: both `Graph.discretize()` output (`was_dph=True`,
renormalised/sibling-coupled edges) and native DPH graphs (`is_discrete=True`,
`was_dph=False`, edge weight is `c_e·θ` directly). Previously these fell back
to FD unconditionally (`not discrete` gate). Still opt-in via the same
`exact_moment_grad=True` kwarg; default off is byte-identical FD.

Full math: `b3-batch3-mpfr-and-discrete-derisk.md` (on master) +
`b3-discrete-theta-adjoint-plan.md` (this branch, the batch plan actually
executed). De-risked in pure Python/JAX **before** any C was written
(`experiments/dr_dph_renorm_jacobian.py`, `dr_discrete_moment_correction.py`
— both ALL PASS, build-free), then ported to C and gated against **native
central-difference** of `graph.moments(K, discrete=True)`
(`experiments/dr_dph_moments_jac_gate.py` — ALL PASS, incl. a sibling-coupled
multi-param case, a native-DPH case, and the MPFR decline).

---

## 2. What SHIPS

| Area | Symbol / change | File:line |
|---|---|---|
| C core | `ptd_moments_grad_theta_dph` — exact discrete-moment Jacobian, branches on `graph->was_dph` between the renorm quotient rule and the plain linear rule, then applies the discrete moment correction | `src/c/phasic.c:10946` |
| C core | `ptd_dph_correct_discrete_moment_grad` + `ptd_dph_factorial`/`_binomial`/`_stirling2` — the continuous→discrete moment correction, ported from `GraphBuilder::continuous_to_discrete_moments` | `src/c/phasic.c:10870-10923` |
| Header | decl of `ptd_moments_grad_theta_dph` | `api/c/phasic.h` |
| C++/pybind | `Graph::moments_grad_theta_dph` + `_moments_grad_theta_dph` binding | `api/cpp/phasiccpp.h`, `src/cpp/phasic_pybind.cpp` |
| Python | `_exact_grad_enabled` gate relaxed (drops `not discrete`); `_effective_discrete = discrete or serialized['is_discrete']` (mirrors `GraphBuilder`'s own dispatch); `_exact_moments_jac_np` calls the new `_dph` variant when effectively discrete | `src/phasic/__init__.py:6930-6975` |
| Tests | `tests/pytest/inference/test_exact_grad_discrete.py` (8 tests: was_dph incl. sibling coupling, native DPH, `is_discrete`-without-per-call-flag, grad+vmap, default-path byte-identical, MPFR decline→FD fallback) | new file |
| Experiments | `dr_dph_renorm_jacobian.py`, `dr_discrete_moment_correction.py` (build-free de-risk), `dr_dph_moments_jac_gate.py` (real-C gate, also re-runs the two continuous gates as a no-regression check) | new files |

**Algorithm.** Reuses `ptd_moments_grad_theta`'s forward moment chain +
reverse chain + stage-2 param-tape reverse **verbatim, unchanged** — the
stage-1 reverse only ever reads the *current* `edge->weight` as an opaque
free variable, so it's provably agnostic to whether that value is a direct
`w_e = c_e·θ` or a was_dph-renormalised `p_e = w_e/S_v`. The only new math:

1. **Renorm quotient-rule contraction** (was_dph only, sibling coupling):
   `∂p_e/∂θ_j = (c_e^j − p_e·Σ_{e'∈out(v)} c_{e'}^j) / S_v`, needing a
   per-vertex precompute of `S_v`/`ΣC_v[j]` from the caller-supplied `theta`
   (edge->weight after `update_weights` already holds `p_e`, not the raw
   `w_e`, so `S_v` can't be recovered from it — the new function takes
   `theta` explicitly for this).
2. **Discrete moment correction**: `continuous_to_discrete_moments` is a
   fixed *linear* map in the moment axis, so it commutes with
   differentiation — applying it to each **column** (θ-index) of the
   continuous-moments Jacobian equals the true chain rule (this is the
   entire justification and it's mechanically checked, not just argued, in
   `dr_discrete_moment_correction.py`).

---

## 3. Two non-obvious bugs found and fixed *only in the new function*

Both were invisible before because `exact_moment_grad` was never previously
exercised on a graph with a coefficient-less (constant) tape-input edge —
every was_dph graph (`discretize()`) has these (aux back-edges), so building
this feature exposed them immediately as a **segfault**, not a wrong answer.

1. **Every edge is a tape input, including constant ones.** The elimination
   tape registers *every* edge weight — including coefficient-less constant
   edges (`coefficients_length==0`, `coefficients==NULL`, e.g.
   `discretize()`'s aux back-edges, weight fixed 1.0) — as a free input,
   regardless of whether it varies with θ. The contraction loop must skip
   these (`if (e->coefficients_length == 0) continue;`): their weight never
   depends on θ, so their exact gradient contribution is provably 0, and
   skipping avoids dereferencing a NULL `coefficients` pointer.
2. **Starting-vertex edges must also be skipped in the was_dph branch.**
   `update_weights()` never recomputes a starting-vertex (IPV) edge's weight
   from θ, so its true `dp_e/dθ` is identically 0 — but in the was_dph
   quotient rule, if `S_v` is (correctly) never computed for the starting
   vertex, `(coeff − p·Σc)/0` can be `±inf`, and `binp[k]=0 · inf = NaN` in
   IEEE754 (not 0). Fixed by skipping any tape input at the starting vertex.

The first fix is an additive guard (`continue` statement) needed in
**both** the new function and (see §3.1) the already-shipped continuous one;
the second (starting-vertex skip) only matters where a division makes
`binp[k]=0` dangerous, i.e. only in the new function's was_dph branch.

### 3.1 A related latent bug found in the ALREADY-SHIPPED continuous path — FIXED (by request)

Bug 1 above is **not specific to discrete/was_dph graphs** — it's a general
property of the tape (any coefficient-less edge is a tape input). The
**already-merged** `ptd_moments_grad_theta` (continuous) made no such check
and would **segfault** on *any* continuous parameterized graph containing a
constant aux edge (`add_aux_vertex`/`add_aux_vertex_constant`) combined with
`exact_moment_grad=True`. `Graph.joint_stop_prob_graph()` uses exactly this
primitive (`add_aux_vertex_constant`, its "t-aux trapping loops",
`__init__.py:9514`) for an otherwise-continuous graph. Confirmed concretely
(not just reasoned) before applying the fix:

```python
g = phasic.Graph(1); s = g.starting_vertex()
a = g.find_or_create_vertex([2]); b = g.find_or_create_vertex([1])
s.add_edge(a, 1.0); a.add_edge(b, [1.0])
a.add_aux_vertex_constant(1.0)     # same primitive joint_stop_prob_graph uses
g.update_weights([0.5])
g._moments_grad_theta(2)          # segfaulted before this fix
```

This was **not known to be reachable** through any existing caller (no test
combines `joint_stop_prob_graph()` output with `exact_moment_grad=True`), so
it had never fired in practice — but was a real crash waiting for the first
caller who did. Per explicit user sign-off, the same one-line additive guard
(`if (e->coefficients_length == 0) continue;`) was applied directly to
`ptd_moments_grad_theta` (`src/c/phasic.c`, contraction loop) — it cannot
change any previously-correct behaviour (it only prevents a crash) and was
verified: the repro above now returns a finite Jacobian instead of
segfaulting, and both continuous gates (`dr_moments_jac_gate.py`,
`dr_mpfr_gate_test.py`) still ALL PASS after the change.

---

## 4. Correctness evidence

- **Build-free math de-risk (no C, no rebuild):** `dr_dph_renorm_jacobian.py`
  (quotient-rule formula vs `jax.jacobian`, incl. mixed-scale θ and
  single-edge/single-param degenerate zero-Jacobian cases) — ALL PASS.
  `dr_discrete_moment_correction.py` (linearity, the known K=2 closed
  identity `E[N²]=m₁−m₀`, and the per-column-application recipe vs
  `jax.jacobian` for K=1..6) — ALL PASS.
- **Real-C gate vs native central-difference**
  (`dr_dph_moments_jac_gate.py`): `_erlang().discretize(0.5)` (K=2,3),
  `_chain(2).discretize(0.3)` (3 params, sibling coupling), a moderately
  mixed-scale θ, native DPH (`was_dph=False`, 2- and 3-stage), the MPFR
  conditioning decline (θ=[1,1e-13] → empty, θ=[1,0.5] → size 4) — ALL PASS.
  Also re-runs `dr_moments_jac_gate.py` + `dr_mpfr_gate_test.py` (continuous
  — re-run again after applying the §3.1 guard to `ptd_moments_grad_theta`)
  as a no-regression check — ALL PASS both before and after that guard.
- **pytest** (`test_exact_grad_discrete.py`, 8 tests): was_dph erlang (K=2,3),
  sibling-coupled was_dph, native DPH, `is_discrete`-without-per-call-flag,
  `jax.grad`+`jax.vmap` (SVGD-safety), default-path byte-identical-to-FD,
  MPFR-decline-stays-finite — **8 passed**.
- **No regression:** `tests/pytest/test_discrete_moments_and_reward.py` +
  `tests/pytest/inference/test_fd_gradient_mixed_scale.py` — **34 passed, 1
  xfailed**, identical to pre-change (the forward path is untouched; only a
  Python gradient-path gate was relaxed, and it only takes effect when the
  caller opts in via `exact_moment_grad=True`).
- **`test_jax_integration.py`:** 9 failed / 7 passed / 10 skipped — the
  **exact same** pre-existing baseline documented in
  `B3-MERGE-REVIEW.md` §5 (all 9 fail at the `param_length == 0` check,
  before any B3 code runs; confirmed not a regression).
- **Broader sweep** (every test file touching `discretize`/`was_dph`/
  `is_discrete`): `test_model_selection.py`, `test_c_trace_reload_param.py`,
  `test_daisy_chain_c_path.py`, `test_epoch_sojourn_finalread.py`,
  `test_gate_moments_3way.py`, `test_gate_svgd_seams.py`,
  `test_graph_discretize.py`, `test_modeling_compose.py`,
  `test_reward_transform_discrete_flag.py` — all pass. `test_graph.py` — 110
  passed, 4 pre-existing failures (rate-validation error-message text,
  traced to master commit `c673be83` "Removed mistaken check for rate <= 1",
  unrelated to this work), 3 skipped.

---

## 5. Risks / non-negotiables carried over

- **Forward parity is sacred:** unaffected — no forward code was touched,
  only the gradient path and only behind the existing `exact_moment_grad`
  opt-in.
- **Opt-in default off:** unchanged; `exact_moment_grad` still defaults
  `False`.
- **Scope:** `weight_mode='linear'` only (same as continuous); log/formula
  modes are unaffected (still FD). Hierarchical SCC and joint-index are
  unaffected (still FD; `ptd_moments_grad_theta_dph` isn't invoked there).
- **Mixed-vertex decline:** a was_dph vertex mixing a constant and a
  parameterized out-edge declines (empty → FD fallback) rather than risk a
  wrong `S_v` (the constant edge's pre-renormalisation weight isn't
  recoverable from its current, already-divided `edge->weight`). Verified
  this declines cleanly (no crash) via a synthetic mixed-vertex graph; does
  not arise for any graph produced by `Graph.discretize()` (its only
  constant edges are lone aux back-edges, never mixed with a parameterized
  sibling at the same vertex).
- **`ptd_moments_grad_theta` (continuous, shipped) now carries one additive
  safety guard** (§3.1: skip coefficient-less tape-input edges) applied by
  explicit user request during this batch. It cannot change any previously-
  correct output — verified via both continuous gates re-passing ALL PASS
  after the change — it only prevents a segfault on a code path with no
  currently-known caller.

---

## 6. Review checklist

- [ ] `ptd_moments_grad_theta_dph`: the `was_dph` branch precompute
      (`Sv`/`SigmaCv`) matches the quotient-rule formula; the mixed-vertex
      decline; the two "skip" guards (§3) are present and applied identically
      in both the was_dph and plain branches (currently `continue` is common
      to both — check this is intentional: yes, a starting-vertex or
      constant edge contributes 0 regardless of was_dph).
- [ ] `ptd_dph_correct_discrete_moment_grad`: applied to every output column
      exactly once, after the edge→θ contraction, before the final
      `isfinite` check.
- [ ] Python wiring: `_effective_discrete` mirrors
      `GraphBuilder::compute_pmf_and_moments`'s `is_disc` dispatch exactly
      (check both still agree if that C++ code ever changes).
- [ ] §3.1 — confirm the guard applied to `ptd_moments_grad_theta` is a pure
      addition (re-diff against master to see just the one `continue` line).
- [ ] Decide: squash-merge this branch into master now (deferred for review
      as of this handoff).

---

## 7. Cross-cutting decisions carried over from the continuous batch (unchanged, still open)

- Make the exact path the DEFAULT (vs opt-in)?
- Native FFI gradient handler vs the current host `pure_callback` (perf
  only)?
- log/formula weight modes, joint-index, hierarchical SCC — still FD-only,
  each its own future de-risk (per `b3-batch3-mpfr-and-discrete-derisk.md`
  §"Other remaining Batch-3 items").
