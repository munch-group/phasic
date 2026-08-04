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
   vertex with seed `e_v` — a completely different code path): 79/79 match
   to machine precision (worst rel. err 3.4e-16). Two independent
   derivations agreeing this precisely is strong evidence neither has a
   compensating pair of errors.
4. **(D1.5, added after plan review)** Diagonal-`multiplier-1` storage
   convention (`phasic.c:10770`) added to the tape model, plus the
   asymmetric primal/tangent guards. A crafted (non-random) case with a
   diagonal command at weight exactly 1.0 confirms: the corrected function
   matches `jax.jacobian` exactly; a deliberately-wrong variant that
   (incorrectly) shares the primal's `m==0` skip on the tangent produces a
   materially different, wrong answer at the same point — proving the
   check actually discriminates the bug rather than passing regardless.

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
- **D1.5 — extend the de-risk script (DONE)**: added the diagonal-
  `multiplier-1` storage convention and both guards (primal `m==0`-skip;
  tangent NOT skipping on `m==0`) to `dr_sojourn_fwdmode_adjoint.py`'s tape
  model, plus a crafted diagonal-weight-exactly-1 case and a deliberately-
  wrong guard variant to prove that case discriminates. All checks pass:
  259/259 primal, 243/243 forward-mode-vs-jax, 79/79 forward-vs-reverse
  cross-check, and the new crafted case (correct matches jax exactly, wrong
  variant diverges as expected).
- **D2 — C implementation (DONE)**: `ptd_sojourn_grad_theta_subset`
  (`src/c/phasic.c`, after `ptd_moments_grad_theta_dph`), declared in
  `api/c/phasic.h`, wrapped as `phasic::Graph::sojourn_grad_theta_subset`
  (`api/cpp/phasiccpp.h`) and exposed as `_sojourn_grad_theta_subset` via
  pybind (mirrors `_moments_grad_theta`'s exposure pattern exactly). Takes
  no `theta` parameter (matches `ptd_moments_grad_theta`'s own convention —
  theta is implicit via the graph's current `update_weights`-mutated edge
  weights; only the log/dph variants need explicit theta, for their own
  renormalisation-specific reasons). Includes the cache-reuse fix
  (`ptd_precompute_reward_compute_graph`, not a raw rebuild), the NULL-safe
  seeding, the asymmetric guards, and the single-call/union-indices design
  — all per "C implementation design" above. `pixi run install-dev` builds
  clean; a hand smoke-test matched native central-difference to ~3e-11.
- **D3 — gate against native central-difference (DONE)**:
  `experiments/dr_sojourn_grad_theta_gate.py`. ALL PASS across continuous
  chains/branching (incl. mixed-scale theta), native-DPH fixtures
  (`is_discrete=True, was_dph=False`), and realistic joint-probability
  graphs built the same way `test_sojourn_subset_adjoint.py` does
  (`StateIndexer`/`joint_prob_graph`, `discrete=False`); confirms the
  `was_dph` decline on both a `discretize()` fixture AND (found during this
  gate, not anticipated) `joint_prob_graph(..., discrete=True)` itself,
  which turns out to set `was_dph=True` internally (renormalising), NOT
  "native DPH" — confirmed via `g.get_was_dph()` directly, so this decline
  is CORRECT/expected, not a bug; confirms the MPFR decline and an empty-
  indices edge case. Re-ran the existing gates (`dr_moments_jac_gate.py`,
  `dr_mpfr_gate_test.py`, `dr_dph_moments_jac_gate.py`,
  `dr_log_mode_moments_jac_gate.py`) as a no-regression check — all still
  ALL PASS (this batch touches no existing function).

  **Benchmark result — corrects the plan's original expectation.** The
  original draft expected exact to be unconditionally faster once the
  cache-reuse fix landed. Measured reality: the cache-reuse fix eliminates
  the CATASTROPHIC `O(n^3)`-per-call regression (the thing that actually
  mattered for large joint-prob graphs) but a smaller `O(L)`
  `ptd_pcg_convert_to_offset` step (converting the cached RAW parameterized
  tape to offset form) is still redone on every call — this conversion is
  NOT itself cached (a further cache would need a new `ptd_graph` field
  with its own invalidation lifecycle; deliberately not added this batch,
  see "Cross-cutting notes"). Since FD pays no equivalent fixed cost (it
  reads the already-numeric `graph->reward_compute_graph` cache directly),
  both paths scale ~linearly in the number of theta parameters `P`, but
  exact carries a fixed per-call overhead (convert + stage-0) amortized
  over `P` tangent passes while FD's `2·P` calls carry none — so exact's
  RELATIVE advantage grows with `P`, crossing over around `P`≈10–20 for a
  representative 2000-vertex chain (`P`=2: FD 4× faster; `P`=10: roughly at
  parity; `P`=50: exact ~2.9× faster), while the joint-index-model's own
  native `P`=2 (coalescent rate + mutation rate) sits on the FD-favoured
  side of that crossover (FD ~2.6× faster on a realistic `n`=39603 graph).
  Exact-mode's value proposition for THIS function is therefore primarily
  **correctness** (removing FD's mixed-scale gradient defect — the reason
  the whole B3 initiative exists, per `CLAUDE.md`), not raw speed, for the
  small-`P` models this specific function is most used on today; it is
  also a clear speed win for richer (`P`≳10) models. See D4 for how this
  informs the default.
- **D4 — Python wiring (DONE)**: new `exact_grad` kwarg (name finalized in
  the D1 review — NOT `exact_moment_grad`, since this function's second
  output is `dummy_moments = jnp.zeros(2)`, no "moments" concept exists
  here) on `pmf_from_graph_joint_index`. `model_bwd` always pays the cheap
  union-based callback + two small FFI sojourn calls to determine
  applicability, then uses `jax.lax.cond(exact_ok, lambda: exact_tbm,
  _fd_theta_bar)` so the expensive `2*n_params`-call FD loop is (in
  principle) only paid when the exact path declines. **Originally
  defaulted to `True`** per this section's original reasoning below — see
  "Adversarial review of the FIX" for why that default was wrong and
  reversed to `False` (initially matching D3's own finding, exact is a
  correctness win always but a speed win only for `P`≳10-20, matching
  every other B3 batch's default-`True` precedent seemed reasonable — the
  additional problem the fix-review found, that `lax.cond` cannot actually
  skip anything under SVGD's real `vmap` usage, was not visible from the
  isolated D3 benchmark and only surfaced by reviewing the WIRED
  composition). No-silent-fallback logging for every excluded case
  (out-of-scope weight mode, `was_dph`, baked mode, a `theta_dim`
  overriding the graph's own `param_length`, MPFR decline, input-spec
  out-of-scope).
- **D5 — tests + adversarial review of the FIX (DONE)**: new
  `tests/pytest/inference/test_exact_grad_joint_index.py` (matches native
  central-diff on continuous branching + native-DPH fixtures, grad+vmap,
  default behavior, every exclusion declines+logs correctly including the
  new `theta_dim` check, MPFR-decline-stays-finite, an unsorted/
  duplicated/subset-indices fixture, an FFI-vs-clone primal-consistency
  check, `fixed_mask` zeroing). Full regression sweep (`tests/pytest/
  inference/`, 332 passed / 1 pre-existing unrelated flake / 45 skipped /
  12 xfailed — the failure, `test_svgd_correctness.py::
  test_basic_convergence`, is a stochastic posterior-mean tolerance check
  on an unrelated exponential-distribution model with no joint-index
  involvement, consistent with this project's documented pre-existing test
  flakiness, not caused by this batch).

  **Adversarial review of the FIX** (math-stats-checker, full report in
  session history) confirmed the C function's math, NULL-safety,
  cache-reuse ownership/lifecycle, and `was_dph`/native-DPH scoping are all
  correct (tried to break each and could not), and confirmed the Python
  wiring's union/searchsorted gather and quotient-rule re-derivation are
  correct. It found:
  - **(fixed)** The tangent guard's `m==0` fix (required, see the C design
    section above) had re-introduced a `0 * inf -> NaN` hazard: dropping
    the whole-update `m==0` skip means each of the update's two summands
    now needs its OWN `0*inf=0` guard, not just the outer
    `isinf(m) && y[a]==0` one — real on production graphs with trap/
    deficit-sink vertices (confirmed finite `y[v]`/`y_dot[v]` there is not
    guaranteed), though fail-soft today (the resulting NaN already trips
    the existing `isfinite` sweep and correctly declines to FD) — fixed
    with per-summand guards, a no-op on every currently-passing case.
  - **(fixed)** No allocation in the new C function was NULL-checked, and
    — because the offset-conversion is NOT itself cached (see the
    Cross-cutting note below) — this function allocates and frees several
    tape-length-sized arrays on EVERY call, unlike the FD path. On the
    large joint-probability graphs this function targets this could be a
    genuine multi-GB spike, not just a missing safety net. Fixed: NULL
    checks on every allocation (decline to FD rather than segfault), plus
    a conservative size guard (`L > 5e7` declines) as a safety net pending
    real measurement on production-scale graphs.
  - **(fixed — the significant one)** `jax.lax.cond` does not skip a
    branch when its predicate is *batched*, which it always is under
    SVGD's actual `vmap(grad(loss))(particles)` — JAX/XLA computes BOTH
    branches and selects. So the wiring's "only pay FD when exact
    declines" design does not hold under the primary real usage pattern:
    `exact_grad=True` cost FD **plus** exact on every call, not FD only
    when needed. Combined with D3's own finding (exact alone is ~2–4×
    slower than FD alone at this model's typical native `P`=2), defaulting
    to `True` was a real regression under `vmap` for the common case, not
    the "modest, bounded" trade-off this section originally concluded from
    the *isolated* C-level benchmark. Presented to the user as a genuine
    judgment call (fix the `lax.cond`/`vmap` composition to restore the
    full benefit at the cost of losing automatic per-call MPFR fallback
    under `vmap`, vs. flip the default, vs. accept the cost as the price of
    guaranteed correctness) — resolved by flipping the default to `False`,
    the lowest-risk option: the wiring itself is unchanged (still correct,
    still useful for direct non-`vmap`ped calls and as an explicit opt-in
    for richer, `P`≳10 models), only the kwarg's default changed. This is
    the only B3 exact-gradient kwarg in the codebase that defaults to
    `False` — documented explicitly in the docstring, unlike every sibling
    kwarg's default-`True` precedent.
  - **(fixed)** Two shape/dtype wiring fragilities that would turn a
    previously-working FD call into a hard `pure_callback` error once
    `exact_grad=True`: a non-1-D `vertex_indices` (the callback always
    returns a raveled shape; the declared `ShapeDtypeStruct` did not match
    unless the input was already 1-D) and a `theta_dim` overriding the
    graph's own `param_length` (the C function reads `param_length` off
    the clone, silently disagreeing with the wiring's `param_length_actual`
    used for every reshape). Both fixed: normalize `vertex_indices` via
    `jnp.atleast_1d` consistently before use, and add a static
    `theta_dim`-mismatch decline to the scope gate.
  - **(not changed — noted, no defect demonstrated)** The exact path's
    quotient rule mixes a primal from the FFI-rebuilt graph
    (`compute_sojourn_times_ffi`, used for `norm_exact`/`obs_sojourn_exact`)
    with a Jacobian from a separately-built `graph.clone()` (used for
    `J_obs`/`J_all`) — two independently-built elimination tapes the
    pure-FD path never had to keep consistent. Added a direct test
    (`test_exact_sojourn_jac_np_ffi_and_clone_agree_on_primal`) asserting
    the two representations' primal sojourn values agree, rather than only
    checking it indirectly through the end-to-end gradient (which routes
    through a THIRD representation, `graph.expected_sojourn_time`, as the
    CD oracle, and could mask a small systematic offset as CD error).
  - **(not changed — noted)** The MPFR-conditioning decline's rationale
    ("mirrors the continuous moments gate") does not actually transfer:
    `ptd_moments_grad_theta`'s gate protects against a genuine primal/
    gradient MPFR-representation mismatch that has no counterpart here
    (`ptd_expected_sojourn_time_subset` has no MPFR path at all), so the
    gate here is a pure, and build-dependent (inert without `HAVE_MPFR`),
    conservatism knob rather than a correctness necessity. Left as-is
    (declining to FD on an ill-conditioned tape is still a reasonable,
    conservative default) but the comment should eventually be corrected
    to not claim a rationale that doesn't apply — tracked as a follow-up,
    not blocking.
  - **(not changed — noted)** No fixture in the new test suite has an
    infinite/NaN primal sojourn value (a trap/deficit-sink vertex), so the
    `0*inf` decline path fixed above is untested and its real decline rate
    on production graphs is unmeasured.

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
  This reuse is PARTIAL: `graph->parameterized_reward_compute_graph` (the
  RAW tape) is cached and reused, but `ptd_pcg_convert_to_offset`'s
  `O(commands)` conversion to offset form is NOT itself cached (redone
  every call) — caching that too would need a new `struct ptd_graph` field
  with its own invalidation lifecycle (tied into
  `dph_compute_invalidated`/`ptd_precompute_reward_compute_graph`'s
  existing free/rebuild logic), which is exactly the kind of change to
  shared, already-shipped infrastructure this project's standing preference
  is to avoid without asking first ([[feedback_no_modify_existing]]).
  Deliberately not attempted this batch; see D3's benchmark for the
  resulting (bounded, characterized, not catastrophic) performance profile
  this leaves in place.
  Worth a CLAUDE.md follow-up note that the existing moments/log/dph
  gradient functions have this same no-caching gap (tolerable there so far
  because moment-graphs are typically small — not yet benchmarked at scale,
  so "tolerable" is an assumption worth eventually checking too).
- Follow-ups this batch documents but does not attempt: `was_dph`
  joint-index (native DPH IS in scope, see Scope section),
  `weight_mode='formula'`/`'callback'` (shared with the general
  formula-adjoint follow-up), `observed_indices` baked-mode support.

## D6 — `lax.cond`/`vmap` redesign (follow-up batch, user-requested)

Addresses the D5 fix-review's significant finding directly (see above):
`jax.lax.cond` cannot skip a branch once its predicate is batched, which
it always is under SVGD's `vmap(grad(loss))(particles)` — so the current
wiring pays FD *and* exact on every call whenever `exact_grad=True`, not
FD-only-when-needed. The interim fix (D5) was to flip the default to
`False`. This batch redesigns the wiring so the intended behavior — pay
FD only when exact genuinely can't be used — actually holds under `vmap`,
which would let the default become `True` again.

### The mechanism: move the choice out of the traced graph entirely

`vmap` can only skip a branch of a Python-level `if`/`else` when the
condition is a plain Python value fixed at TRACE time (not a JAX array
computed from a traced input) — confirmed directly, build-free, in
`experiments/dr_lax_cond_vmap_derisk.py` (D6.0, all three claims below
CONFIRMED with call-counting spies, not just from documentation):

1. `lax.cond` under `vmap` with an always-True batched predicate traces
   AND executes both branches (reproduces the bug).
2. A Python `if <static bool>: … else: …` under `vmap` traces (and
   therefore executes, at any batch size) only the chosen branch.
3. Raising a Python exception inside a `jax.pure_callback` host function
   propagates as a real exception at the call site, both for a plain
   `grad(...)` call and under `vmap(grad(...))` (the vmap case wraps the
   message in a `"CpuCallback error…"` preamble but the original
   exception type and message survive intact inside it).

This means: decline reasons that are **static** (fixed once, never vary
per call) can be resolved with a plain `if`/`else` and cost nothing when
unused, exactly as `_jix_exact_enabled` (weight_mode/`was_dph`/baked/
`theta_dim`) already does today. Decline reasons that are **dynamic**
(vary per theta, i.e. per SVGD particle/iteration) cannot be resolved
this way without giving up the automatic per-call fallback — there is no
way to have BOTH "skip the unneeded branch under vmap" AND "gracefully
fall back to FD for whichever specific particles need it" for a
genuinely per-call condition; this is a hard JAX/XLA limitation, not an
engineering gap in this wiring.

### Which decline reasons are actually dynamic?

Re-examining the three C-level decline reasons
(`ptd_sojourn_grad_theta_subset`, see "C implementation design" above):

- **`was_dph`**: a graph-level flag, invariant across every call for a
  given model. Already resolved statically via `_jix_was_dph`.
- **Out-of-scope tape input** (an edge spec that isn't
  `PTD_PCG_PTR_EDGE`/byte-0/in-range): purely a function of graph
  TOPOLOGY, not of theta's numeric value — if it declines at one theta it
  declines at every theta, and vice versa. Currently only detected
  dynamically (by calling the C function and getting an empty result) —
  this can be resolved STATICALLY too, once, since it never varies.
- **MPFR conditioning**: the ONLY genuinely theta-dependent decline
  reason — the elimination tape's condition number depends on the actual
  edge weights, i.e. on theta.

So a single **construction-time probe** — call
`_jix_exact_graph._sojourn_grad_theta_subset([0])` once, at a reference
theta — resolves BOTH the topology-only reason (its result generalizes to
every future theta) AND gives a reasonable signal for the MPFR reason (at
least at that reference point). It cannot guarantee the MPFR gate will
also hold at every OTHER theta the optimizer visits later — that
residual risk is the trade-off the user decision below is about.

**Reference theta for the probe: `theta = ones(param_length)`.** Simple,
deterministic, reproducible (matches the reference point already used
elsewhere in this session's de-risk work, e.g. D1's `edge_weights_at_theta0
= coeff_matrix @ ones`). A single index (`[0]`) is enough — the decline
reasons the probe checks are graph-wide or theta-wide, not
index-specific, so probing more indices adds cost without adding
information. Cost: ONE extra `_sojourn_grad_theta_subset` call at
construction time only (not per gradient step) — for a large joint-prob
graph this pays the one-time cache-build cost the FIRST use of ANY
gradient method already pays regardless (per D2's cache-reuse design), so
it is not a new category of cost, just possibly moved slightly earlier.

### Failure mode once committed: raise (user decision, asked directly)

Once the probe succeeds, `model_bwd` commits unconditionally to the exact
path for the rest of that model's lifetime — no per-call `lax.cond`, no
implicit FD fallback. If some LATER theta (a specific SVGD particle at a
specific iteration) hits the MPFR gate, the C function returns empty and
the host callback **raises a `RuntimeError`** naming the theta and the
reason, rather than silently falling back or crashing with an opaque
shape-mismatch error. This was presented to the user as an explicit
three-way choice (raise / keep the per-call fallback and accept no `vmap`
speedup / — a third "accept the current cost" option had already been
superseded by flipping the default in D5) — the user chose **raise**,
for the reasons already given in the AskUserQuestion: no silent fallback,
consistent with this codebase's stated principle, and a hard stop is
preferable to an unnoticed problem for a Bayesian inference tool. Claim 3
above (raising propagates correctly, including under `vmap`) is exactly
what makes this failure mode viable rather than a debugging dead end.

### Design

```python
_exact_sojourn_jac_np = None      # unchanged in spirit: the host callback
_jix_probed_ok = False            # NEW: latched once at construction

if _jix_exact_enabled:
    _jix_exact_graph = graph.clone()
    _jix_param_length = param_length_actual
    _jix_all_terminal_np = all_terminal_indices_np

    def _exact_sojourn_jac_np(theta_np, vertex_indices_np):
        th = np.asarray(theta_np, dtype=np.float64)
        vi = np.asarray(vertex_indices_np, dtype=np.int64).ravel()
        _jix_exact_graph.update_weights(th.tolist())
        union_idx = np.union1d(vi, _jix_all_terminal_np)
        raw = _jix_exact_graph._sojourn_grad_theta_subset(union_idx.tolist())
        if not raw:
            raise RuntimeError(
                "pmf_from_graph_joint_index: exact sojourn gradient "
                f"(exact_grad=True) declined at theta={th.tolist()} -- an "
                "ill-conditioned elimination tape at this specific theta "
                "(the construction-time probe at theta=ones succeeded, so "
                "this is a THETA-SPECIFIC decline, not a structural one). "
                "No automatic finite-difference fallback is available once "
                "the exact path has been committed to for this model -- "
                "pass exact_grad=False, or investigate why this theta is "
                "ill-conditioned."
            )
        J_union = np.asarray(raw, dtype=np.float64).reshape(
            union_idx.shape[0], _jix_param_length)
        obs_pos = np.searchsorted(union_idx, vi)
        all_pos = np.searchsorted(union_idx, _jix_all_terminal_np)
        return J_union[obs_pos], J_union[all_pos]

    # Construction-time probe -- theta=ones, single index. Resolves the
    # topology-only decline reason for ALL future theta, and the MPFR gate
    # at this reference point. A probe failure is treated exactly like a
    # static structural exclusion (pure FD, no wiring overhead) -- it is
    # NOT re-tried per call.
    _probe_theta = np.ones(_jix_param_length, dtype=np.float64)
    _jix_exact_graph.update_weights(_probe_theta.tolist())
    _probe_raw = _jix_exact_graph._sojourn_grad_theta_subset([0])
    if _probe_raw and len(_probe_raw) == _jix_param_length:
        _jix_probed_ok = True
    else:
        _jix_grad_logger.info(
            "pmf_from_graph_joint_index: exact sojourn gradient declined "
            "at the construction-time probe (theta=ones) -- an "
            "out-of-scope tape input, or an ill-conditioned elimination "
            "tape even at this benign reference point -- using finite "
            "differences for the whole model."
        )
```

`model_bwd` becomes a plain, non-traced Python `if` on
`_jix_exact_enabled and _jix_probed_ok` (both ordinary Python bools fixed
before any tracing happens) instead of a per-call `lax.cond` on a traced
`exact_ok`:

```python
def model_bwd(res, g):
    theta, vertex_indices = res
    g_visits, g_moments = g
    n_params = theta.shape[0]

    def _fd_theta_bar():
        ...  # UNCHANGED

    if not (_jix_exact_enabled and _jix_probed_ok):
        theta_bar = _fd_theta_bar()
    else:
        _vi_norm = jnp.atleast_1d(vertex_indices).astype(jnp.int32)
        all_sojourn_exact = compute_sojourn_times_ffi(structure_dict, theta, all_terminal_indices)
        norm_exact = jnp.sum(all_sojourn_exact)
        obs_sojourn_exact = compute_sojourn_times_ffi(structure_dict, theta, _vi_norm)
        J_obs, J_all = jax.pure_callback(
            _exact_sojourn_jac_np,
            (jax.ShapeDtypeStruct(_vi_norm.shape + (n_params,), jnp.float64),
             jax.ShapeDtypeStruct(all_terminal_indices.shape + (n_params,), jnp.float64)),
            theta, _vi_norm, vmap_method='sequential',
        )
        dnorm_exact = jnp.sum(J_all, axis=0)
        d_probs_exact = (J_obs*norm_exact - obs_sojourn_exact[:,None]*dnorm_exact[None,:]) / norm_exact**2
        theta_bar = d_probs_exact.T @ g_visits
        if fixed_indices_set:
            theta_bar = theta_bar * _fixed_keep   # unchanged

    return theta_bar, None, None
```

No `exact_ok`/`jnp.isfinite` check remains in the committed path — the
callback's own `if not raw: raise` replaces it (matching D1.5/D2's
"decline via empty return" C-level contract, just surfaced as an
exception instead of a NaN sentinel once committed).

### What does NOT change

- `_jix_exact_enabled`'s static gate (weight_mode/`was_dph`/baked/
  `theta_dim`) — unchanged, still resolved once, still logged.
- The C function, the quotient-rule math, the union/`searchsorted`
  gather, `fixed_mask` handling — all unchanged (D5's fixes stand).
- `exact_grad=False` behavior — unchanged (still 100% FD, byte-identical
  to before any of this work).

### Should the default flip back to `True`?

Deliberately deferred to a SEPARATE decision after this redesign ships
and is gated/tested — not bundled into this batch. Even with the `vmap`
issue fixed, forward-mode's cost still scales with `P` (D3's benchmark:
crossover around `P`≈10–20), so `exact_grad=True` would still be a
mild-to-moderate slowdown at this model's typical native `P`=2, now
WITHOUT an automatic per-call FD fallback for a later ill-conditioned
theta. Whether that trade (guaranteed-correctness-with-a-hard-stop vs.
FD's documented mixed-scale defect, at small `P`) is worth defaulting to
is exactly the kind of judgment call this session has been bringing to
the user rather than deciding unilaterally — raised again at D9 below,
not assumed here.

### Batches

- **D6.0 — build-free de-risk (DONE)**: `experiments/dr_lax_cond_vmap_derisk.py`,
  3/3 claims CONFIRMED.
- **D6.1 — adversarial review of THIS ADDENDUM** before any wiring change
  (per standing instruction): is the probe genuinely sufficient to
  generalize the topology-only decline reason to every theta? Is there a
  scenario where the probe SUCCEEDS at theta=ones but the graph is
  actually out of scope for some OTHER structural reason not caught by a
  single-index, single-theta check? Does raising inside `model_bwd`'s
  `pure_callback` interact safely with `jax.custom_vjp`'s bookkeeping
  (e.g. does a raised exception during the backward pass leave any global
  state — the private `_jix_exact_graph` clone's weights, any cache —
  inconsistent for a SUBSEQUENT, non-raising call)? Is the raised
  exception's message actually reachable/legible to an end user running
  full SVGD (not just a bare `jax.grad` call), given the wrapping observed
  under `vmap`?
- **D6.2 — implement**: the wiring change above.
- **D6.3 — tests**: extend `test_exact_grad_joint_index.py` with a
  probe-success case (confirms `vmap`'s FD branch is genuinely never
  traced — e.g. via a call-counting spy on `_fd_theta_bar`, mirroring
  `dr_lax_cond_vmap_derisk.py`'s claim-2 methodology, not just a value
  check, since a value-only check cannot distinguish "skipped" from
  "computed but discarded"), a probe-failure case (graph with an
  out-of-scope tape input — reuse `Graph.discretize()`'s aux-edge pattern
  or a purpose-built fixture — falls back to FD with no wiring overhead,
  exactly like a static exclusion), and a post-probe MPFR-decline case
  (raises `RuntimeError` with the theta and reason in the message, both
  under plain `grad` and under `vmap(grad(...))`).
- **D6.4 — adversarial review of the FIX** before considering this batch
  complete, per the same rhythm as every prior batch.
