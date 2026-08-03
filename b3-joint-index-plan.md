# B3 joint-index exact gradient — plan

Continues the B3 initiative ([[project_b3_analytic_gradient]]). Extends the
exact reverse-mode/forward-mode theta-adjoint machinery to
`Graph.pmf_from_graph_joint_index` — currently 100% finite-difference
(absolute `eps=1e-7`, the same mixed-scale-defect-prone pattern the whole B3
initiative exists to replace; see `src/phasic/__init__.py`'s
`pmf_from_graph_joint_index` → `model_bwd`, no `exact_moment_grad`-style
kwarg exists yet at all).

## What "joint-index" computes, precisely

`pmf_from_graph_joint_index`'s forward is NOT a PMF/PDF read — it computes
**expected sojourn (residence) times**, not densities:

```
obs_sojourn      = expected_sojourn_time_subset(observed vertex indices)
all_sojourn      = expected_sojourn_time_subset(all terminal vertex indices)
normalization    = sum(all_sojourn)
sojourn_probs    = obs_sojourn / normalization
```

`expected_sojourn_time_subset`'s **default production algorithm is itself an
adjoint** (`src/c/phasic.c:10206`, `ptd_expected_sojourn_time_subset`, "(A)
ADJOINT" branch — the "(B) FORWARD" branch is a legacy O(n·k)-memory
escape hatch, opt-in via `PHASIC_SOJOURN_FORWARD=1`, disabled by default
specifically because it OOMs on real joint-probability graphs where
`k ~ n`, per that function's own comment and
`tests/pytest/test_sojourn_subset_adjoint.py`'s docstring: a real coalescent
joint-prob graph over 8 samples has `n=684226, k=279936` and the O(n·k)
form asked for a 1.5 PB matrix). This is the "reverse-over-reverse" framing
noted in `CLAUDE.md`/memory: the primal itself is already a reverse-mode-style
computation (an adjoint of `expected_waiting_time`'s own linear map), so an
exact theta-gradient needs to differentiate an adjoint, not a plain forward
pass.

## Derivation (de-risked build-free, no C touched)

Let the elimination trace be `nc` commands `(a_c, b_c, m_c)` meaning
`x[a_c] += x[b_c] * m_c`, applied in order. In matrix form this is
`x_final = M @ x_seed` for `M = E_{nc-1}...E_0`, `E_c = I + m_c e_{a_c}e_{b_c}^T`.

- **Moment chain** (existing, shipped): seed `x_seed = ones` (or a reward
  vector), read `x_final[target]`. `m_0 = e_target^T M x_seed`.
- **Sojourn** (`ptd_expected_sojourn_time_subset`'s adjoint): seed
  `y_seed = e_target`, walk the SAME commands **in reverse index order with
  `a`/`b` swapped** (`y[b_c] += y[a_c]*m_c`, walked `c=nc-1..0`). This
  computes `y_final = M^T @ e_target`, i.e. `y_final[v] = M[target, v]`.
  Confirmed algebraically (`E_c^T = F_c`, the transpose of a single update
  IS the swapped-role update, and reversing the whole product order gives
  the transpose of the whole product) AND empirically:
  `Graph.expected_sojourn_time()[v] == Graph.expected_waiting_time(reward=e_v)[0]`
  for every `v`, checked directly against the existing pybind API
  (`experiments/dr_sojourn_fwdmode_adjoint.py`'s cross-check section, and a
  standalone spot-check during this session). So
  **`sojourn(v) = m_0` computed with reward-seed `e_v`** — this is not a new
  quantity, it is the existing moment-chain machinery's own output at a
  different seed.

### Why forward-mode, not reverse-mode, for the gradient

`ptd_moments_grad_theta` differentiates `m_0` (ONE scalar) w.r.t. `theta`
(`P` components, typically small) via reverse-mode: one backward pass gives
all `P` components at once — efficient because outputs (1) << inputs (P).

Sojourn's gradient is the OPPOSITE shape: MANY outputs (`sojourn(v)` for up
to `k~n` target vertices — confirmed `k` can be in the hundreds of thousands,
see above) but FEW inputs (`theta_dim`, typically 1–10). Naively calling the
existing reverse-mode moment-gradient once per target vertex (seed `e_v`)
would cost `O(k · nc)` — reintroducing exactly the `k`-scaling blowup
`ptd_expected_sojourn_time_subset`'s own adjoint algorithm exists to avoid
for the PRIMAL.

**Forward-mode** is the efficient choice here: for a GIVEN theta component
`j`, seed the param-tape tangent directly with the coefficient column
(`idot[k] = coefficients[k][j]` for every edge-weight input `k`
simultaneously — valid by linearity of the JVP operator, i.e. seeding a
linear combination of one-hot tangents is the SAME as combining their
individually-propagated results; verified as an explicit numerical
cross-check, not just asserted), propagate through the SAME param-tape
arithmetic ops (`P`/`PP`/`INV`/`ONE_MINUS`/`DIVIDE` — this is EXACTLY the
already-shipped-as-a-validator `ptd_dbg_run_tape` forward-mode logic,
`src/c/phasic.c:10362`, reused unchanged), then propagate the resulting
per-command multiplier tangent through the sojourn recurrence (same
reversed-index, swapped-role walk as the primal, interleaved with the
primal since forward-mode needs the LIVE evolving state, not a
pre-recorded snapshot — this removes the need for stage-1's snapshot
arrays entirely, a simplification relative to reverse-mode). This gives the
**FULL `(n,)` sojourn-gradient column for that theta component in one
`O(nc)` pass** — `P` total passes, `O(P·nc)` overall, independent of `k`.

### De-risk evidence (`experiments/dr_sojourn_fwdmode_adjoint.py`, all build-free)

1. Primal sojourn (manual reversed-index/swapped-role walk) vs JAX
   autodiff-free replay: 259/259 random two-tier tapes match exactly.
2. Forward-mode theta-adjoint (this batch's new algorithm, seeded via the
   coefficient column) vs `jax.jacobian` ground truth: 243/243 match to
   machine precision (worst rel. err 4.4e-16).
3. **Independent cross-check**: the SAME forward-mode result vs the SLOW
   alternative (reverse-mode moment-chain machinery, run once per target
   vertex with seed `e_v` — a completely different code path): 76/76 match
   to machine precision (worst rel. err 3.4e-16). Two independent
   derivations agreeing this precisely is strong evidence neither has a
   compensating pair of errors.

## Scope this batch: CONTINUOUS / `weight_mode='linear'` only

Mirrors the ORIGINAL B3 phasing exactly (continuous first, discrete/was_dph
as a separate follow-up batch; log/formula/callback each their own
follow-up) — each batch stays reviewable and each scope decision is
re-derived from the CURRENT forward behavior, not assumed:

- **`weight_mode='log'`**: already rejected at the FORWARD level for
  joint-index (`pmf_from_graph_joint_index` raises `ValueError` for `'log'`
  today — confirmed by reading the current code, `__init__.py:7746`), so
  this combination never reaches the gradient at all. Nothing to do.
- **`weight_mode='formula'`/`'callback'`**: honoured by the forward (routed
  through the FFI/GraphBuilder or a materialize-then-call path
  respectively), but out of scope for the exact gradient — matches the
  precedent from both the discrete and log-mode batches (`'formula'` needs
  its own bytecode-tape adjoint; `'callback'` is arbitrary Python, not
  analytically differentiable in general). Decline + log why, same
  no-silent-fallback pattern.
- **`is_discrete`/`was_dph` graphs**: joint-probability graphs are
  frequently discrete in practice (`test_sojourn_subset_adjoint.py`'s own
  fixture parametrizes `discrete=True/False`) — this is a real, deferred
  gap, not a corner case. Left for a dedicated follow-up because it needs
  the SAME was_dph renormalization quotient-rule contraction derived in the
  discrete/was_dph batch (`ptd_moments_grad_theta_dph`), combined with this
  batch's NEW forward-mode-through-the-sojourn-recurrence — a genuinely new
  combination that deserves its own de-risk pass, not a decision made
  unilaterally inside this already-large batch. Decline + log why.
- **`observed_indices` baked/dedup mode** (the `custom_vmap`-batched
  unique-index path used for large-repeat-count SVGD datasets): also
  deferred. Supporting it correctly requires scatter-adding the upstream
  cotangent by the inverse-index map before applying the quotient rule (the
  adjoint of a gather is a scatter-add) — mechanically not hard, but another
  independent piece worth its own de-risk/test rather than bundling.
  Decline + log why (falls back to FD, unaccelerated but correct, exactly
  as today).
- **`exclude_vertices`**: NOT deferred — this only affects which indices are
  IN the `all_terminal_indices` set at construction time (a static, already
  pre-filtered list by the time the gradient runs); no special handling
  needed beyond computing the Jacobian at whatever `all_terminal_indices`
  the model was already built with.
- **`fixed_mask`**: NOT deferred — zero the corresponding gradient columns
  in Python after computing the full Jacobian, mirroring the exact pattern
  used in every other B3 batch's `model_bwd`.

## C implementation design

New `int ptd_sojourn_grad_theta_subset(struct ptd_graph *graph,
const size_t *indices, size_t k, const double *theta, size_t theta_len,
double *J_out)`:

- Declines (-1, FD fallback) if `graph->was_dph` (mirrors the log-mode
  function's decline; `is_discrete` has no C-level `ptd_graph` field, so
  the Python wiring's `not _effective_discrete`-style gate is the load-bearing
  check for native-DPH graphs, exactly as established for the log-mode
  batch — the C-level `was_dph` check is a secondary net for the subset
  that sets it).
- Stage-0 (shared with `ptd_moments_grad_theta`, reused verbatim): build the
  parameterized off-tape, replay once at the current theta to record
  `(na[c], nb[c], nm[c])` and the param-tape's per-op snapshots; apply the
  SAME `ptd_dbg_tape_needs_mpfr` gate.
- NEW: for each of the `P` theta components, seed the param-tape TANGENT
  with `idot[input k] = coefficients[edge for input k][j]`, propagate
  through the param tape's arithmetic ops (P/PP/INV/ONE_MINUS/DIVIDE —
  identical formulas to the existing `ptd_dbg_run_tape` validator, ported
  into production rather than left validator-only) to get
  `mdot[c] = d(nm[c])/d(theta_j)` for every numeric command; then propagate
  `(y[], y_dot[])` TOGETHER through the sojourn recurrence (seed
  `y[target]=1, y_dot[v]=0`; walk `c=nc-1..0`:
  `y_dot[nb[c]] += y_dot[na[c]]*nm[c] + y[na[c]]*mdot[c]`, THEN
  `y[nb[c]] += y[na[c]]*nm[c]`); gather `y_dot[indices[r]]` into
  `J_out[r*P+j]` for the `k` requested indices.
- Coefficient-length-0 / starting-vertex tape-input skip guards: N/A here —
  those guards exist in the CONTRACTION step of the reverse-mode functions
  (`binp[k] * coefficients[j]`); this function seeds tangents INTO the param
  tape instead of reading gradients OUT of it, so a coefficient-less or
  starting-vertex edge simply gets `idot=0` naturally (its coefficient row
  is all-zero or it's never an `off->inputs[]` entry at all) — no separate
  guard needed, but this equivalence should be spot-checked in the gate,
  not assumed.

## Wiring: `pmf_from_graph_joint_index`'s `model_bwd`

`sojourn_probs[i] = obs_sojourn[i] / norm`, `norm = sum(all_sojourn)`. Given
`J_obs` (`k_obs × P`) and `J_all` (`k_all × P`) from two calls to the new C
function (one per index set — see "why two calls" below), the quotient rule
is applied directly in Python/JAX (no further C needed):

```
dnorm[j]        = sum_v J_all[v, j]
d(sojourn_probs[i])/d(theta_j) = (J_obs[i,j]*norm - obs_sojourn[i]*dnorm[j]) / norm**2
theta_bar[j]    = sum_i g_visits[i] * d(sojourn_probs[i])/d(theta_j)
```

**Two C calls, not one combined index set**: the forward already makes two
separate FFI calls (`obs_idx`, `all_terminal_idx`) today; mirroring that
structure exactly is simpler than deduplicating the two index sets in
Python, at the cost of 2×P forward-mode passes instead of P — a modest,
deliberate constant-factor cost (not the `O(n)`-vs-`O(n²)` scale of blowup
this initiative cares about), not a premature-optimization violation.

## Batches

- **D0 — build-free de-risk (DONE)**: math derivation +
  `experiments/dr_sojourn_fwdmode_adjoint.py` (3 independent checks, ALL PASS).
- **D1 — adversarial review of THIS PLAN** before any C is written (per
  standing instruction this session): is the forward-mode derivation
  actually complete? Is the "no discrete/was_dph this batch" exclusion
  airtight (does anything reach the new C function for a discrete graph
  without going through the Python gate first)? Is the "two separate calls"
  design actually correct (do `J_obs`/`J_all` need to share any per-call
  state, e.g. the parameterized off-tape build, that would be wasted work
  if rebuilt twice)? Any edge case (empty index sets, a single-vertex
  target, `P=1`) missed?
- **D2 — C implementation** (only after D1 is clean).
- **D3 — gate against native central-difference** of
  `Graph.expected_sojourn_time(indices)` (the same production PRIMAL used
  by the forward), at benign + mixed-scale theta, on continuous joint-prob
  fixtures built the same way `test_sojourn_subset_adjoint.py` does
  (`StateIndexer`/`with_ipv`/`joint_prob_graph`, `discrete=False`). Re-run
  the existing gates (`dr_moments_jac_gate.py`, `dr_mpfr_gate_test.py`,
  `dr_dph_moments_jac_gate.py`, `dr_log_mode_moments_jac_gate.py`) as a
  no-regression check (this batch touches no existing function).
- **D4 — Python wiring**: new `exact_moment_grad`-style kwarg (name to be
  finalized in review — precedent is `exact_moment_grad`, but this function
  has no "moments" concept; consider `exact_grad` or reuse the same name
  for consistency) on `pmf_from_graph_joint_index`, defaulting to `True`
  from the start (no separate default-flip step needed this time, since
  there is no existing opt-in behavior to preserve — this function has
  never had an exact path). No-silent-fallback logging for every excluded
  case (out-of-scope weight mode, discrete/was_dph, baked mode, MPFR
  decline).
- **D5 — tests + adversarial review of the FIX**: new
  `tests/pytest/inference/test_exact_grad_joint_index.py` (matches native
  central-diff, grad+vmap, default picks it up automatically, every
  exclusion declines+logs correctly, MPFR-decline-stays-finite). Full
  regression sweep. Submit the implemented diff to adversarial review
  before considering the batch complete — this exact rhythm caught two real
  bugs in the log-weight-mode batch's fix that a green test suite alone had
  missed.

## Cross-cutting notes

- This is the **first B3 C function to use forward-mode in production**
  (everything shipped so far — linear/dph/log moments — is reverse-mode).
  The param-tape forward-mode arithmetic itself is not new (it is the
  existing `PHASIC_B3_VALIDATORS`-guarded `ptd_dbg_run_tape`/
  `ptd_debug_fwdmode_grad`), but promoting it to an UNGUARDED production
  path is new — worth flagging explicitly in review.
- Follow-ups this batch documents but does not attempt: discrete/was_dph
  joint-index, `weight_mode='formula'` (shared with the general
  formula-adjoint follow-up), `observed_indices` baked-mode support.
