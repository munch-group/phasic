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
5. **log + discrete/was_dph: the "already fails safely" premise was WRONG —
   REFUTED by direct repro; the explicit exclusion is MANDATORY, not
   belt-and-suspenders.** Originally reasoned (by analogy with the scalar-rate
   `discretize()` path, which DOES add a zero-filled coefficient slot that
   would trip log's positivity check) that this combination could never
   silently succeed. An adversarial review of this plan pointed out the
   scalar-rate widening is only ONE of `discretize()`'s two branches — the
   CALLABLE-rate branch (`self.parameterized() and callable(rate)`,
   `__init__.py:2807`) does NOT widen the layout at all; it just clones and
   lets the callable supply the aux edge's own coefficient vector. Confirmed
   by direct reproduction (not just re-reading the code): a graph discretized
   via a callable rate returning all-positive coefficients, then set to
   `weight_mode='log'`, calls `update_weights([...], log=True)` and
   **succeeds** (`was_dph=True`, no raise). So an explicit exclusion at BOTH
   the Python gate (`not _effective_discrete`) AND inside the new C function
   (decline on `graph->is_discrete` or `graph->was_dph`) is the ONLY thing
   preventing a silently-wrong exact gradient for this combination — treat
   both as load-bearing, not defensive redundancy.

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

**Decision (reaffirmed after adversarial review): clone again, do not
refactor the two existing shipped functions in this batch.** The review
argued for extracting the shared core NOW (in a separate, value-identical
commit verified by re-running the three existing gates), on the grounds that
"three copies that must stay in sync" is where duplication bugs like the
coefficient-length-0 one come from. That argument was taken seriously and
checked concretely — including re-testing its own supporting claim (a
starting-vertex tape-input gap it identified between the two existing
functions), which did NOT reproduce empirically (a directly-constructed
parameterized start edge is never registered as a tape input at all;
verified: the shipped linear function already returns exactly 0 for such a
parameter, matching an independent central-difference). So the specific
evidence for "the copies are already silently diverging" didn't hold up —
but the general architectural point (three near-identical unshared copies
are a standing invitation for exactly this kind of drift) remains valid.

Kept the original decision anyway: this repo has an explicit, repeatedly
stated standing preference for purely additive changes and never modifying
existing code as a side effect of new work ([[feedback_no_modify_existing]],
also applied to SVGD in [[feedback_no_change_svgd]]). Refactoring two
already-shipped, gate-verified functions is exactly the kind of change that
preference is meant to require asking about first, rather than a session
deciding it unilaterally because an adversarial review suggested it — even
a well-reasoned, low-risk, verifiable one. The duplication risk is recorded
in `CLAUDE.md` as an explicit follow-up (alongside the other known B3 gaps)
for the user to authorize whenever they want it done, same treatment as
every other flagged-not-fixed item this session.

The new function DOES still include the starting-vertex skip guard the
review flagged (matching `ptd_moments_grad_theta_dph`'s pattern) even though
it was not shown to be reachable — the guard costs one comparison and keeps
all three copies structurally consistent with each other, which is the
cheap partial mitigation available without a full refactor.

## Batches

**D0 — build-free de-risk (DONE, see above).**

**D1 — adversarial review of THIS PLAN (DONE).** Found two real issues
before any C was written (both independently confirmed by direct
reproduction, not just re-reading the review):

1. **CRITICAL — confirmed via `grep`.** `_exact_moments_jac_np`'s `_one(t)`
   (`__init__.py:6993`) calls `_exact_graph.update_weights(t)` with NO `log=`
   argument — it defaults `log=False`. If the scope gate were widened to
   admit `weight_mode='log'` WITHOUT also fixing this call, the private
   clone's edge weights would be computed via the LINEAR dot product
   (`Σc_iθ_i`) while the actual model computes via the LOG product
   (`Π(c_iθ_i)`) — a completely different function. This would have shipped
   a silently-wrong exact gradient for every log-mode caller, exactly the
   class of bug this session already fixed once (rewards) and once
   (gradient-clip majority-collapse) via review. **Must fix as part of D4**:
   `_exact_graph.update_weights(t, log=(_wm == 'log'))`.
2. **log + discrete/was_dph "already fails safely" — REFUTED**, see point 5
   above (now corrected in this doc). Confirmed via direct repro that
   `discretize()` via a callable rate does NOT raise when combined with
   `weight_mode='log'`. The explicit exclusion in both the Python gate and
   the new C function is mandatory, not optional.
3. A third claim (starting-vertex tape-input gap in the shipped linear
   function) was raised, checked, and did NOT reproduce — see the "Design
   decision" section above. The starting-vertex guard is still included in
   the new function anyway (free, consistent with the dph function's
   pattern), just not because it fixes an active bug.
4. MPFR gate scale-sensitivity under log mode (a spread-θ log-mode graph's
   effective multiplier ratio grows faster than the equivalent linear graph)
   is worth *quantifying*, not fixing — folded into D3 below as a measurement,
   not a blocker.
5. `tests/pytest/inference/test_exact_grad_discrete.py`'s
   `test_no_silent_fallback_logs_on_out_of_scope_weight_mode` uses
   `weight_mode='log'` as its example of an out-of-scope mode that must log
   the FD-fallback message. Once log is in-scope this test's premise
   inverts — **must fix as part of D5**: switch its example to `'formula'`
   or `'callback'` (still genuinely out of scope) and add a new positive
   test asserting log mode does NOT log the fallback message once it's live.

**D2 — C implementation** (now safe to start, both critical issues designed
around from the start rather than discovered after implementing):
- New `int ptd_moments_grad_theta_log(struct ptd_graph *graph, int
  nr_moments, const double *theta, size_t theta_len, double *J_out)`:
  clones `ptd_moments_grad_theta`'s stage-0/1/2 verbatim (same MPFR gate,
  same coefficient-length-0 skip guard for tape-input hygiene, PLUS the
  starting-vertex skip guard from the dph function for consistency).
  Contraction becomes `J_out[outk*P+j] += binp[k] * (e->weight / theta[j])`
  for ALL `j` (every param contributes to every log-mode edge, unlike
  linear's sparse dot-product) — no per-vertex precompute needed (no sibling
  coupling, no renormalization, unlike the was_dph case). Declines if
  `graph->is_discrete` or `graph->was_dph` (mandatory, see D1.2).
- Declare in `api/c/phasic.h`, bind `Graph::moments_grad_theta_log` /
  `_moments_grad_theta_log` (mirrors the dph binding).

**D3 — gate against native central-difference**
(`experiments/dr_log_mode_moments_jac_gate.py`): the same dense-coefficient
fixture used in D0, at benign + mixed-scale theta, central-diff the NATIVE
`graph.update_weights(theta, log=True)` + `graph.moments(K)` as oracle (never
the code path under test). Include an MPFR-decline check at a deliberately
spread-θ log-mode graph, reporting (not gating on) how readily it declines
relative to the equivalent linear-mode case. Re-run the THREE existing gates
(`dr_moments_jac_gate.py`, `dr_mpfr_gate_test.py`,
`dr_dph_moments_jac_gate.py`) as a no-regression check.

**D4 — Python wiring**: widen `pmf_and_moments_from_graph`'s scope check
from `_wm in (None, 'linear')` to also accept `_wm == 'log' and not
_effective_discrete`; fix `_exact_moments_jac_np`'s clone `update_weights`
call to pass `log=True` when in log-scope (D1.1); dispatch `_one(t)` to
`_moments_grad_theta_log` when in log-scope. Default
(`exact_moment_grad=True`, already the default per the prior batch) picks
this up automatically for existing log-mode callers with NO code change on
their part — verify this doesn't silently change any currently-passing
log-mode test's gradient VALUES beyond FD's own ~1e-9 tolerance (compare
before/after on the existing log-mode fixtures in
`test_gate_trace_ffi_equivalence.py`, `test_omp_exception_safety.py`,
`test_weight_mode_probe_and_guards.py`; none of these currently call
`jax.grad` on a log-mode model per the plan review, but confirm this stays
true).

**D5 — tests + adversarial review of the FIX** (not just re-running the
plan's own gates): fix `test_exact_grad_discrete.py`'s inverted test (D1.5);
new `tests/pytest/inference/test_exact_grad_log_weight_mode.py` mirroring
`test_exact_grad_discrete.py`'s shape (matches-native-CD, grad+vmap,
default-path picks it up automatically, MPFR-decline-still-finite, explicit
log-mode + discrete/was_dph combo still declines correctly at BOTH the
Python gate and the C level). Full regression sweep (the same 3-way split
used for the rewards/svgd_step fixes) before considering this batch done.
Submit the diff to adversarial review before merging/calling it complete,
per explicit instruction this session.
