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

## Adversarial review findings (D1) — incorporated throughout

A math-stats-checker review of this plan (before any C was written, per
standing instruction) confirmed the core derivation (transpose duality,
`sojourn(v) = M[target,v]`) but found four problems, all fixed in the
sections below; a fifth was found by independently re-verifying the
review's own claims against source (I have Bash/execution access the
reviewer didn't in this session):

1. **Segfault risk** — "coefficient-length-0 edges need no seeding guard"
   was wrong (would dereference `NULL`). Fixed in "C implementation design"
   above (NULL-safe seeding, matching the discrete/was_dph batch's own
   already-fixed bug of the same class).
2. **Silent-wrong-gradient risk** — the primal and tangent sojourn-walk
   guards are NOT symmetric (diagonal `multiplier-1` storage means `m==0`
   can coexist with nonzero `mdot`). Fixed in "C implementation design"
   above; also requires extending `dr_sojourn_fwdmode_adjoint.py` to model
   this case before D2 (see Batches, D1.5).
3. **Overbroad scope** — deferring ALL `is_discrete` graphs (not just
   `was_dph`) was unnecessarily conservative. Corrected in "Scope this
   batch" above: only `was_dph` is deferred.
4. **Undesigned host-callback/vmap wiring.** Fixed in "Wiring" above.
5. **(found via my own independent source re-verification, not in the
   original review) Performance-regression risk from the "two calls"
   design was correctly flagged by the review, but the ROOT CAUSE is
   broader than call count**: `ptd_moments_grad_theta` — the function this
   batch's C code was drafted to mirror — rebuilds the entire `O(n^3)`
   parameterized tape from scratch on EVERY call, with no cache reuse
   (confirmed directly, `phasic.c:10743-10750`,`:10874-10879`), unlike the
   PRIMAL sojourn/moments paths (`ptd_expected_sojourn_time_subset`,
   `ptd_expected_waiting_time`) and the current FFI-based FD path
   (`ComputeSojournTimesFfiImpl`, `graph_builder_ffi.cpp:1060-1078`), both
   of which reuse a graph-level/per-thread cached tape and pay the `O(n^3)`
   cost once per graph lifetime, not once per call. Copying
   `ptd_moments_grad_theta`'s pattern verbatim — even collapsed to one call
   — would still rebuild `O(n^3)` on every SVGD gradient step, a severe
   regression specifically on the large joint-probability graphs (`n` up to
   ~7×10^5) this function targets, where the current FD path pays that cost
   only once. Fixed in "C implementation design" above: the new function
   calls `ptd_precompute_reward_compute_graph` and reuses
   `graph->parameterized_reward_compute_graph[_off]`, exactly like the
   primal sojourn function already does. (This is a pre-existing gap in the
   shipped moments/log/dph gradient functions too, but fixing THOSE is out
   of scope here — flagged as a CLAUDE.md follow-up instead, since "never
   modify existing code" unless asked.)

## Scope this batch: CONTINUOUS + native-DPH / `weight_mode='linear'` only

> **Revised after D1 adversarial review** (see "Adversarial review findings"
> below) — the original draft deferred ALL `is_discrete` graphs alongside
> `was_dph`. That was overbroad: `was_dph` is the only C-level exclusion this
> function actually needs.

Mirrors the ORIGINAL B3 phasing (log/formula/callback/baked-mode each their
own follow-up) — each scope decision is re-derived from the CURRENT forward
behavior, not assumed:

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
- **`was_dph` graphs ONLY are deferred** (native DPH, `is_discrete=True,
  was_dph=False`, IS in scope this batch). Verified three independent ways
  that native DPH needs zero special-casing for this quantity: (1)
  `ComputeSojournTimesFfiImpl` (`graph_builder_ffi.cpp:955`) has NO
  `is_discrete`/`discrete` branch anywhere in its body (grepped directly,
  zero matches); (2) `ptd_expected_sojourn_time_subset`
  (`src/c/phasic.c:10206`) likewise has no `is_discrete` branch — same
  reversed-walk code runs for both continuous and native-DPH graphs; (3)
  precedent from the shipped moments-gradient functions: `ptd_moments_grad_theta`
  declines ONLY on `graph->was_dph` (`phasic.c:10921`), never on
  `is_discrete` — the existing comment there states plainly "`is_discrete`
  (native DPH, `was_dph=False`) has NO C-level `ptd_graph` field" (`phasic.c:10907`),
  i.e. at the C level a native-DPH graph IS a continuous/linear graph; only
  `was_dph` (the discretize()-quotient-rule renormalization) is a genuinely
  different computation, requiring the SAME was_dph contraction rule derived
  in the discrete/was_dph batch (`ptd_moments_grad_theta_dph`) combined with
  this batch's forward-mode sojourn recurrence — a new combination deserving
  its own de-risk pass. Only `graph->was_dph` triggers the decline; the
  Python-level gate mirrors this (no separate `is_discrete` check needed,
  same as the log-mode batch's own gate). Decline + log why for `was_dph`.
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

> **Revised after D1 adversarial review + independent verification** (see
> "Adversarial review findings" below) — three corrections from the original
> draft: (1) the "no seeding guard needed" claim was wrong and would
> segfault; (2) the primal/tangent guards are NOT symmetric; (3) the tape
> must be obtained via the graph-level CACHE, not rebuilt from scratch (the
> original draft copied `ptd_moments_grad_theta`'s uncached pattern, which
> would be a severe regression on the large graphs this function targets).

New `int ptd_sojourn_grad_theta_subset(struct ptd_graph *graph,
const size_t *indices, size_t k, const double *theta, size_t theta_len,
double *J_out)`:

- Declines (-1, FD fallback) if `graph->was_dph` (see scope section above —
  this is the ONLY C-level exclusion needed; native DPH needs none).
- **Stage-0: obtain the tape via the CACHE, not a fresh rebuild.** Call
  `ptd_precompute_reward_compute_graph(graph)` first — the SAME entrypoint
  `ptd_expected_sojourn_time_subset` and `ptd_expected_waiting_time` already
  call (`phasic.c:10208`, `:10043`), which populates/reuses
  `graph->parameterized_reward_compute_graph` (or `..._off` on a Stage-A2
  on-disk cache hit) exactly once per graph lifetime (invalidated only by
  `dph_compute_invalidated`, e.g. `set_was_dph(true)`), guarded by the
  per-graph `compute_graph_lock` mutex. Then obtain the offset form: reuse
  `graph->parameterized_reward_compute_graph_off` if already populated
  (cache hit), else convert the cached raw tape once via
  `ptd_pcg_convert_to_offset(graph->parameterized_reward_compute_graph, graph, NULL, 0)`
  — this conversion is an `O(commands)` linear pass, NOT the `O(n^3)`
  elimination (which already ran, cached). **Do NOT call
  `ptd_graph_ex_absorbation_time_comp_graph_parameterized[_dyn]` directly**
  the way `ptd_moments_grad_theta` does — that call rebuilds the whole
  `O(n^3)` tape from scratch with no cache check, and is destroyed at the
  end of every invocation (confirmed directly, `phasic.c:10743-10750` +
  `:10874-10879`; this is a pre-existing, already-shipped inefficiency in
  the moments/log/dph gradient functions, tolerable there because moment
  graphs are typically modest-sized — NOT tolerable here, where target
  graphs are exactly the large joint-probability case, `n` up to ~7×10^5
  per `test_sojourn_subset_adjoint.py`). This is a deliberate DIFFERENCE
  from the pattern the other B3 gradient functions follow — not a bug in
  them, just the wrong pattern to copy for this one. (Fixing the existing
  functions' lack of caching is out of scope for this batch — see the new
  CLAUDE.md follow-up note.)
  With the cache reused, replay the tape once at the current theta to
  record `(na[c], nb[c], nm[c])` and the param-tape's per-op snapshots
  (same stage-0 arithmetic `ptd_moments_grad_theta` already has, lines
  10766-10779); apply the SAME `ptd_dbg_tape_needs_mpfr` gate.
- **Input-spec validation, done ONCE up front** (mirrors
  `ptd_moments_grad_theta`'s contraction-step check, `phasic.c:10852-10854`,
  but performed BEFORE seeding rather than skipped): for every tape input
  `k`, validate `off->input_specs[k]` is `PTD_PCG_PTR_EDGE`, `byte == 0`,
  and `v`/`e` in bounds — decline (-1) if not (out of scope, not a silent
  wrong answer).
- **Seeding — the corrected, NULL-safe version.** For each of the `P` theta
  components `j`: for every tape input `k`, look up
  `e = graph->vertices[sp.v]->edges[sp.e]`. **If
  `e->coefficients_length == 0`, set `idot[k] = 0` WITHOUT dereferencing
  `e->coefficients`** (it is `NULL` for such edges, not an all-zero row) —
  this is the exact same NULL-pointer class already found and fixed in the
  discrete/was_dph batch (`phasic.c:10856-10868`, its own comment explicitly
  names `add_aux_vertex`/`add_aux_vertex_constant` — i.e.
  `Graph.discretize()` and **`Graph.joint_stop_prob_graph()`** — as
  producers of exactly such edges; joint-prob graphs are THIS function's
  target workload, not a theoretical corner case). Otherwise
  `idot[k] = e->coefficients[j]`. Propagate `idot[]` through the param
  tape's arithmetic ops (P/PP/INV/ONE_MINUS/DIVIDE — identical formulas to
  the existing `ptd_dbg_run_tape` validator, `phasic.c:10362`, ported into
  production rather than left validator-only) to get
  `mdot[c] = d(nm[c])/d(theta_j)` for every numeric command.
- **Sojourn recurrence — asymmetric guards, NOT the primal's guards
  copy-pasted onto the tangent.** Propagate `(y[], y_dot[])` TOGETHER (seed
  `y[target]=1, y_dot[v]=0`; walk `c=nc-1..0`):
  ```
  for (c = nc; c-- > 0; ) {
      a = na[c]; b = nb[c]; m = nm[c]; md = mdot[c];

      // TANGENT: skip ONLY the inf-with-zero-operand case -- NOT m==0
      if (!(isinf(m) && y[a] == 0.0))
          y_dot[b] += y_dot[a]*m + y[a]*md;

      // PRIMAL: unchanged from production -- skips on EITHER m==0 OR inf-with-zero-operand
      if (m == 0.0) continue;
      if (isinf(m) && y[a] == 0.0) continue;
      y[b] += y[a]*m;
  }
  ```
  The primal keeps BOTH guards the existing `ptd_expected_sojourn_time_subset`
  adjoint uses (`m==0` skip; `isinf(m) && y[a]==0` skip). **The tangent
  update must NOT share the `m==0` skip.** Root cause: diagonal commands
  (`from==to`) store `multiplier - 1`, not `multiplier`
  (`phasic.c:10770`), so a diagonal edge whose CURRENT weight is exactly
  1.0 stores `nm[c] == 0` even though its derivative `mdot[c]` can be
  nonzero (the weight varies with theta near 1.0) — skipping the tangent
  update on `m==0` would silently drop that gradient term. This exact trap
  already bit the original moment-chain forward-mode validator (its own
  comment, `phasic.c:10439-10446`); `experiments/dr_sojourn_fwdmode_adjoint.py`'s
  random-tape generator does not model diagonal-minus-1 at all and must be
  extended with an explicit case before its "ALL PASS" covers this (see D1
  fix-up below, done before D2).
  Gather `y_dot[indices[r]]` into `J_out[r*P+j]` for the `k` requested
  indices (see "Wiring" for how `indices`/`k` are chosen — ONE call, not
  two).

## Wiring: `pmf_from_graph_joint_index`'s `model_bwd`

`sojourn_probs[i] = obs_sojourn[i] / norm`, `norm = sum(all_sojourn)`.

> **Revised after D1 adversarial review** — the original draft called the
> new C function TWICE (once per index set), reasoned as "a modest constant
> factor." That reasoning didn't account for the Stage-0 fix above: once the
> tape is cache-reused, a call's cost is `O(P·nc)`, genuinely small — so
> doubling it is not the dangerous part per se, but it's still unnecessary
> work with a strictly simpler alternative available. Collapsed to ONE call.

**Single call, over the union of both index sets.** `vertex_indices`
(observed) and `all_terminal_indices` are BOTH static — fixed once when the
model is built (`all_terminal_indices` is already sorted+deduped at
construction time today; `vertex_indices` is whatever the caller passed to
`pmf_from_graph_joint_index`). At model-construction time (not inside the
per-gradient-call hot path), compute once:

```python
union_indices = np.union1d(vertex_indices_np, all_terminal_indices_np)   # sorted, deduped
obs_pos  = np.searchsorted(union_indices, vertex_indices_np)             # static gather map
all_pos  = np.searchsorted(union_indices, all_terminal_indices_np)       # static gather map
```

Every gradient call then does ONE call to the new C function with
`indices = union_indices`, producing `J_union` (`k_union × P`), then two
`O(k)` static gathers (no repeated tape work): `J_obs = J_union[obs_pos]`,
`J_all = J_union[all_pos]`. This is simpler than the original two-call
design AND strictly cheaper (one forward-mode pass set instead of two,
`union_indices` computed once and closed over, not recomputed per call).
The quotient rule is then applied directly in Python/JAX (no further C
needed):

```
dnorm[j]        = sum_v J_all[v, j]
d(sojourn_probs[i])/d(theta_j) = (J_obs[i,j]*norm - obs_sojourn[i]*dnorm[j]) / norm**2
theta_bar[j]    = sum_i g_visits[i] * d(sojourn_probs[i])/d(theta_j)
```

### Host-callback / vmap wiring (undesigned in the original draft)

The original draft never specified how the new C function's output crosses
into JAX. Design, mirroring the established `pmf_and_moments_from_graph`
pattern (`_exact_graph` + `_one(t)`, `__init__.py:~7024-7039`):

- A persistent `_exact_graph` (a `Graph` built once via `GraphBuilder`/pybind
  at model-construction time, same object reused for the whole SVGD run) is
  closed over by the new gradient function, exactly as the moments/log-mode
  batches already do.
- `union_indices`/`obs_pos`/`all_pos` are plain NumPy arrays computed ONCE
  at model-construction time and closed over in the Python closure — they
  are NOT traced/batched JAX values and do NOT need to cross the
  `pure_callback` boundary as an operand (this is a simplification relative
  to the original draft's unresolved "how does `vertex_indices` cross as a
  runtime pure_callback argument" question — because the index sets here
  are static per-model, not per-particle or per-call, they behave exactly
  like `theta_dim`/`fixed_mask` in the existing pattern: closed-over
  constants, not callback operands).
- `_exact_graph.update_weights(t)` MUST be called inside the callback
  before every gradient read — this is the exact defect class the log-mode
  batch's own fix caught (a stale-weights bug from forgetting this call).
- `ndim` dispatch for batched theta under `vmap(grad(...))`: mirror
  `__init__.py:7035-7039` exactly — `if th.ndim == 1: return _one(th)` else
  loop the batch, stacking results.
- Decline path: the C function returns empty/`-1` for `was_dph`,
  MPFR-ill-conditioned, or out-of-scope input-spec cases. Python detects
  the empty return, logs an INFO message naming WHICH reason (distinguish
  `was_dph` / MPFR / input-spec-out-of-scope — same no-silent-fallback
  pattern as every other B3 batch), and falls back to the existing
  100%-FD `model_bwd` unchanged.

## Batches

- **D0 — build-free de-risk (DONE)**: math derivation +
  `experiments/dr_sojourn_fwdmode_adjoint.py` (3 independent checks, ALL PASS).
- **D1 — adversarial review of THIS PLAN (DONE)**: math-stats-checker
  review + independent source re-verification, 5 findings, all incorporated
  above (see "Adversarial review findings").
- **D1.5 — extend the de-risk script** (before D2, not after): add the
  diagonal-`multiplier-1` storage convention and BOTH guards (primal
  `m==0`-skip + tangent NOT skipping on `m==0`) to
  `dr_sojourn_fwdmode_adjoint.py`'s tape model, with a dedicated case where
  a diagonal command's current weight is exactly 1.0 at theta but varies
  with theta (`nm[c]==0`, `mdot[c]!=0`) — the current generator cannot
  produce this, so its "ALL PASS" doesn't cover it yet. Re-run to confirm
  ALL PASS under the corrected, production-faithful model before writing
  any C.
- **D2 — C implementation** (only after D1.5 is clean): includes the
  cache-reuse fix (`ptd_precompute_reward_compute_graph`, not a raw
  rebuild), the NULL-safe seeding, the asymmetric guards, and the
  single-call/union-indices design — all per "C implementation design"
  above.
- **D3 — gate against native central-difference** of
  `Graph.expected_sojourn_time(indices)` (the same production PRIMAL used
  by the forward), at benign + mixed-scale theta, on continuous AND
  native-DPH joint-prob fixtures built the same way
  `test_sojourn_subset_adjoint.py` does (`StateIndexer`/`with_ipv`/
  `joint_prob_graph`, `discrete=True/False`). Re-run the existing gates
  (`dr_moments_jac_gate.py`, `dr_mpfr_gate_test.py`,
  `dr_dph_moments_jac_gate.py`, `dr_log_mode_moments_jac_gate.py`) as a
  no-regression check (this batch touches no existing function). **New,
  required this batch**: benchmark wall-clock, exact-gradient (single call,
  cache-reused) vs current FD (2×P FFI calls, per-thread cached graph), on
  a realistic large joint-prob graph (aim for `n`/`k` in the
  `test_sojourn_subset_adjoint.py` range, not a toy fixture) — confirm the
  cache-reuse fix actually delivers a net win (expected: `O(P·nc)` exact vs
  `O(P·nc)` FD-but-2×-the-constant, i.e. exact should be faster, not just
  "not slower") before deciding the new kwarg defaults to `True`.
- **D4 — Python wiring**: new `exact_grad` kwarg (name finalized in
  review — NOT `exact_moment_grad`, since this function's second output is
  `dummy_moments = jnp.zeros(2)`, no "moments" concept exists here) on
  `pmf_from_graph_joint_index`. Default depends on the D3 benchmark result
  (`True` if it's a clear win, matching every other B3 batch's precedent of
  defaulting exact-on; otherwise default `False` with a documented reason
  and revisit). No-silent-fallback logging for every excluded case
  (out-of-scope weight mode, `was_dph`, baked mode, MPFR decline, input-spec
  out-of-scope).
- **D5 — tests + adversarial review of the FIX**: new
  `tests/pytest/inference/test_exact_grad_joint_index.py` (matches native
  central-diff, grad+vmap, default picks it up automatically, every
  exclusion declines+logs correctly, MPFR-decline-stays-finite, a
  diagonal-weight-exactly-1 fixture per finding 2 above). Full regression
  sweep. Submit the implemented diff to adversarial review before
  considering the batch complete — this exact rhythm caught two real bugs
  in the log-weight-mode batch's fix that a green test suite alone had
  missed.

## Cross-cutting notes

- This is the **first B3 C function to use forward-mode in production**
  (everything shipped so far — linear/dph/log moments — is reverse-mode).
  The param-tape forward-mode arithmetic itself is not new (it is the
  existing `PHASIC_B3_VALIDATORS`-guarded `ptd_dbg_run_tape`/
  `ptd_debug_fwdmode_grad`), but promoting it to an UNGUARDED production
  path is new — worth flagging explicitly in review.
- This is also the **first B3 gradient function designed to reuse the
  graph-level tape cache** (`ptd_precompute_reward_compute_graph`) instead
  of rebuilding from scratch per call — a deliberate departure from the
  `ptd_moments_grad_theta`/`_log`/`_dph` pattern, necessary because this
  function's target graphs are the large joint-probability case where an
  uncached `O(n^3)` rebuild per gradient call would be a severe regression.
  Worth a CLAUDE.md follow-up note that the existing moments/log/dph
  gradient functions have this same no-caching gap (tolerable there so far
  because moment-graphs are typically small — not yet benchmarked at scale,
  so "tolerable" is an assumption worth eventually checking too).
- Follow-ups this batch documents but does not attempt: `was_dph`
  joint-index (native DPH IS in scope, see Scope section),
  `weight_mode='formula'`/`'callback'` (shared with the general
  formula-adjoint follow-up), `observed_indices` baked-mode support.
