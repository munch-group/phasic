# B3 log-weight-mode exact gradient — plan

Branch: none yet (working directly on master, per the pattern established for
the rewards/svgd_step fixes in this session — small, gated, adversarially
reviewed increments). Continues the B3 initiative
([[project_b3_analytic_gradient]]); see `CLAUDE.md` → "B3 exact moment
gradient — known gaps" for the currently-tracked follow-ups this does NOT
address (out of scope here).

## Goal

Extend `Graph.pmf_and_moments_from_graph`'s exact reverse-mode moment
gradient (currently `weight_mode in (None, 'linear')` only, continuous +
discrete/was_dph) to **`weight_mode='log'`, continuous only**. `'formula'`
and `'callback'` are explicitly OUT of scope (see "Non-goals" below).

## De-risk findings (already done, build-free / no C touched)

1. **Forward parity confirmed.** `pmf_and_moments_from_graph` on a
   `weight_mode='log'` graph matches an independent native oracle
   (`graph.update_weights(theta, log=True)` + `graph.moments(K)` /
   `graph.pdf(t)`) bit-for-bit, across uniform/asymmetric/mixed-scale
   (1e-3..1e6) theta and K=2,3
   (`experiments/dr_log_mode_forward_verify.py`, ALL PASS). This matters
   because the 2025-07 audit that fixed `moments_from_graph` and
   `pmf_from_graph_joint_index` to RAISE on `'log'` (commit `d69919f2`) only
   claimed by code-inspection that `pmf_and_moments_from_graph`'s GraphBuilder
   path "honours the mode" — it was never asserted against values. Verified
   now, not assumed.
2. **Current gate correctly declines log mode already** — logs
   `"exact moment gradient not available for weight_mode='log'"` and falls
   back to FD, exactly as designed. No existing bug; this is purely additive.
3. **Analytic contraction rule confirmed.** `weight_mode='log'` computes
   `w_e = exp(Σ_i log(c_e[i]·θ_i))` over **ALL** `i` in `0..param_length-1`
   (every edge, unconditionally — `src/c/phasic.c` `ptd_graph_update_weights`
   use_log branch; `graph_builder.cpp:481`). The C layer already requires
   every `c_e[i]·θ_i > 0` strictly (raises otherwise), so in any graph that
   reaches this code, no `θ_j` is ever exactly 0 and no `c_e[i]` is ever 0 —
   division by `θ_j` is safe by construction, not something this change needs
   to additionally guard.
   `∂w_e/∂θ_j = w_e / θ_j` (product rule), verified against `jax.jacobian` of
   the SAME log-space computation the production code actually performs
   (`exp(Σ log(c·θ))`, not an idealized literal product) across: uniform
   scale (P=1..5), mixed-scale (1e-6..1e6), extreme mixed-scale
   (1e-12..1e12), P=1 degenerate, and P=6 many-params-one-edge
   (`experiments/dr_log_mode_edge_jacobian.py`, ALL PASS, all violations
   negative i.e. within tolerance).
4. **MPFR gate and stage-1/stage-2 reverse tape need no changes.** The
   condition-number check (`ptd_dbg_tape_needs_mpfr`) operates on the
   elimination tape's numeric multipliers, which are recorded AFTER
   `update_weights` has already computed `edge->weight` via whichever mode —
   fully agnostic to linear vs log vs was_dph, exactly the same "stage-1
   reverse only reads the current edge->weight as an opaque free variable"
   property established for the was_dph batch. Same for
   `ptd_pcg_convert_to_offset`'s `input_specs`.
5. **log + discrete/was_dph is out of scope and already fails safely without
   any new code.** `discretize()`'s `_rebuild_with_wider_layout` adds ONE new
   coefficient slot filled with the discretize rate ONLY on aux edges (0.0 on
   others); combined with log mode's "every edge needs ALL param products
   > 0" rule, a non-aux edge would have a 0 in the new slot and
   `update_weights` would already RAISE (product ≤ 0) before any gradient
   code runs. So even without an explicit guard, attempting to combine
   `discretize()` output with `weight_mode='log'` fails loudly at the
   forward, not silently. The gate below adds an explicit decline anyway
   (belt-and-suspenders, and correct for a hypothetical native-DPH+log
   graph that wouldn't hit that specific raise) — never silently wrong.

## Non-goals (explicitly deferred, not attempted this batch)

- **`weight_mode='formula'`**: a separate compiled bytecode tape
  (`ptd_weight_tape`, `weight_formula.py`) — needs its own small adjoint
  over the *formula's* stack-machine operations, a genuinely different (if
  likely tractable) problem from log's closed-form product rule. Own future
  batch.
- **`weight_mode='callback'`**: an arbitrary Python function
  `(theta, coefficients) -> weight`. Not analytically differentiable in
  general (could be anything) — stays FD permanently, not a gap to close.
- **log + discrete/was_dph combination**: see point 5 above. If ever wanted,
  needs its own de-risk (the discrete moment correction + log contraction
  would need to compose, unverified).

## Design decision: clone vs refactor the reverse-tape skeleton

This will be the **third** near-identical (~150 line) copy of the stage-0
(forward moment chain + MPFR gate) / stage-1 (reverse chain) / stage-2
(param-tape reverse) skeleton in `phasic.c`, after `ptd_moments_grad_theta`
(linear) and `ptd_moments_grad_theta_dph` (discrete/was_dph) — only the
per-edge contraction step differs between all three. A bug (the
coefficient-length-0 tape-input skip) already had to be fixed identically in
two copies during the discrete batch.

**Decision: clone again, do not refactor the two existing shipped functions
in this batch.** Rationale: refactoring shipped, gate-verified functions
carries regression risk beyond what's needed for THIS feature, and doing it
in the same change as adding new functionality mixes two concerns (matches
"don't modify existing code beyond what's needed" — [[feedback_no_modify_existing]]).
Flagging the duplication as a follow-up (adding to CLAUDE.md alongside the
other known gaps) rather than deciding unilaterally to refactor — this is
exactly the kind of question the adversarial review of this plan should
weigh in on before implementation starts.

## Batches

**D0 — build-free de-risk (DONE, see above).**

**D1 — adversarial review of THIS PLAN** (not just the eventual fix), per
explicit instruction: is the `w_e/θ_j` formula complete? Is the log+discrete
exclusion actually airtight? Is cloning vs refactoring the right call? Any
edge case in P=1..6 / extreme scale missed?

**D2 — C implementation** (only after D1 is clean):
- New `int ptd_moments_grad_theta_log(struct ptd_graph *graph, int
  nr_moments, const double *theta, size_t theta_len, double *J_out)`:
  clones `ptd_moments_grad_theta`'s stage-0/1/2 verbatim (same MPFR gate,
  same coefficient-length-0 / starting-vertex skip guards for tape-input
  hygiene), contraction becomes
  `J_out[outk*P+j] += binp[k] * (e->weight / theta[j])` for ALL `j` (every
  param contributes to every log-mode edge, unlike linear's sparse
  dot-product) — no per-vertex precompute needed (no sibling coupling, no
  renormalization, unlike the was_dph case). Declines if `graph->is_discrete`
  or `graph->was_dph` (point 5 above).
- Declare in `api/c/phasic.h`, bind `Graph::moments_grad_theta_log` /
  `_moments_grad_theta_log` (mirrors the dph binding).

**D3 — gate against native central-difference**
(`experiments/dr_log_mode_moments_jac_gate.py`): the same dense-coefficient
fixture used in D0, at benign + mixed-scale theta, central-diff the NATIVE
`graph.update_weights(theta, log=True)` + `graph.moments(K)` as oracle (never
the code path under test). Re-run the THREE existing gates
(`dr_moments_jac_gate.py`, `dr_mpfr_gate_test.py`,
`dr_dph_moments_jac_gate.py`) as a no-regression check.

**D4 — Python wiring**: widen `pmf_and_moments_from_graph`'s scope check
from `_wm in (None, 'linear')` to also accept `_wm == 'log' and not
_effective_discrete`; `_exact_moments_jac_np`'s `_one(t)` dispatches to
`_moments_grad_theta_log` when in log-scope. Default
(`exact_moment_grad=True`, already the default per the prior batch) picks
this up automatically for existing log-mode callers with NO code change on
their part — verify this doesn't silently change any currently-passing
log-mode test's gradient VALUES beyond FD's own ~1e-9 tolerance (compare
before/after on the existing log-mode fixtures in
`test_gate_trace_ffi_equivalence.py`, `test_weight_formula_kwarg.py`'s
log-adjacent cases if any, `Graph.svgd`'s `weight_mode='log'` path if
exercised anywhere).

**D5 — tests + adversarial review of the FIX** (not just re-running the
plan's own gates): new
`tests/pytest/inference/test_exact_grad_log_weight_mode.py` mirroring
`test_exact_grad_discrete.py`'s shape (matches-native-CD, grad+vmap,
default-path picks it up automatically, MPFR-decline-still-finite, explicit
log-mode + discrete combo still declines correctly). Full regression sweep
(the same 3-way split used for the rewards/svgd_step fixes) before
considering this batch done. Submit the diff to adversarial review before
merging/calling it complete, per explicit instruction this session.
