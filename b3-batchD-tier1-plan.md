# Batch D Tier-1 plan — D.1 vmap-crash fix · D.2 validator guards · D.4 `Graph.svgd()` plumbing

**Status: v2, 2026-08-12 — adversarial plan review folded in. Reviewer A
(technical, D.1+D.2): SOUND-WITH-CORRECTIONS. Reviewer B (D.4/API): BROKEN
as v1 — D.4 fully rewritten below to the reviewer's prescription, which is
also the master-plan-exact scope; the rewritten D.4 section goes back to
Reviewer B for a verdict flip before its code is written. D.1 and D.2 are
cleared for implementation now.** Design authority: master plan §6. Branch:
`b3/batchD-tier1`, worktree `../phasic-batchD` with isolated pixi env.
Baseline: `b3-test-baseline.md` (ledger stamped at `cadf1ca4`; working HEAD
`19b86d71` — the ledger's alignment commit; regenerate stamp at merge).

The three items are mutually independent (master plan §6); shared
branch/merge purely for review-bandwidth economy. If any item stalls, the
others merge without it (recorded in the merge review).

---

## D.1 — `Graph.moments_from_graph` vmap crash (review-cleared)

**Bug (verified):** `_compute_moments_pure` (`src/phasic/__init__.py:6767-6779`)
has no `ndim` check; `len(theta_np)` at `:6774` returns the batch size for
2-D theta; wrapper uses `vmap_method='expand_dims'` (`:6782-6786`).

**Fix:** inside `_compute_moments_pure`: `ndim == 2` → loop rows,
`np.stack` to `(B, nr_moments)`; `ndim == 1` unchanged; **`ndim > 2` →
raise a clear `ValueError`** (nested-vmap is unhandled by all three
siblings too — loud beats garbage; reviewer finding 7i). Output contract
confirmed by review against the repo's own battle-tested precedents (all
three loop-pattern siblings sit under `expand_dims` and return
`(B, *declared_shape)`; `_exact_moments_jac_np` `:7035-7040`/`:7568-7571`
is exercised under SVGD's `vmap(grad(...))`). No `ShapeDtypeStruct` change.
The FD backward routes every probe through `_compute_pure` (`:6799-6816`),
so the single-callback fix covers forward and backward; `use_ffi`'s only
reference is `:6653` — the one ctypes callback is the whole surface
(reviewer finding 7iii).

**Bounded side-probes (triage per process §6 — not fixes):**
1. `pmf_from_cpp` sibling probe (`:4174-4183`: `expand_dims`, no visible
   ndim handling — probe well-founded per review): confirm/deny under
   `vmap`; if real → Class B ledger entry.
2. Master-plan risk 11 (`moments_fn_bwd` vmap-transparency): covered by
   the `vmap(grad(...))` test (execution, not reasoning).

**G1 gates (`tests/pytest/inference/test_moments_from_graph_vmap.py`):**
- `vmap` forward == per-row loop forward, values, ≥2 batch sizes;
- `vmap(grad(...))` finite and == per-row `grad` (tolerance-based, FD bwd);
- non-vmap path value-identical to native `Graph.moments`;
- `ndim > 2` raises the new clear error.

## D.2 — `ptd_moment0_grad_theta` guard backport (review-corrected mechanics)

**Target (verified):** `src/c/phasic.c:10678-10726`, inside
`#ifdef PHASIC_B3_VALIDATORS` (`:10666`/`:10726`), zero production
exposure (gated C++ caller `phasiccpp.h:513-555`, gated pybind
`:1904-1914`, CMake default OFF).

**Guard 1 — `coefficients_length == 0` skip:** the unguarded
`e->coefficients[j]` read is at `:10714-10715`; backport the successor's
`continue` pattern (`:10868`). Trivial.

**Guard 2 — MPFR gate (mechanics corrected per Reviewer A finding 1;
the v1 "call the gate locally" was NOT executable — `nm`/`nc` are locals of
the shared helper `ptd_dbg_reverse_tape` (`:10538-10542`, freed `:10607`),
and `ptd_moment0_grad_theta` delegates the whole tape walk to it
(`:10702`)):** without touching the shared helper, add inside
`ptd_moment0_grad_theta`, before `:10702`: copy `off->mem_base`/inputs into
**fresh scratch buffers** (distinct from the `mem0`/`inp0` later handed to
`ptd_dbg_reverse_tape`, which requires clean pre-execution state), replay
the ~18-line stage-0 forward walk (the `:10547-10563`/`:10766-10779`
pattern, including the diagonal `*rm - 1.0` convention), record
`nm_local[]`, call `ptd_dbg_tape_needs_mpfr(nm_local, nc_local)` — which
is a `static` but **un-gated** function (`:10643-10664`, deliberately
outside the ifdefs; visible in both build modes) — and on decline route
through the existing single-exit cleanup to the `-1`/not-applicable
return. ~25-30 lines, one extra O(L) pass; acceptable for a validator.

**G1 gates (per Reviewer A findings 3-6):**
- **Pre-change baseline (finding 5):** in the validators-build worktree
  (`CMAKE_ARGS="-DPHASIC_B3_VALIDATORS=ON" pixi run install-dev`), run the
  five validator gates from `B3-MERGE-REVIEW.md:141-146` BEFORE touching C
  — a bit-rotted gate must not be misattributed to D.2. Also confirm
  finding 4's interpolation with one run: `dr_moment0_theta_gate.py` stays
  applicable (non-empty grad) at θ=[1,1e-8] (cond ~1e8 < 1e12) after the
  gate lands.
- **Post-change:** all five gates green again; plus a two-assertion repro
  with the guards' OPPOSITE expected behaviors kept separate (finding 3):
  (i) coefficient-less-constant-edge fixture at **benign θ → succeeds**,
  non-empty gradient matching central-diff (guard 1 = skip, not decline);
  (ii) ill-conditioned θ (e.g. `[1, 1e-13]`, mirroring
  `experiments/dr_mpfr_gate_test.py:21-23`, or `PHASIC_FORCE_MPFR=1`) →
  **declines** (guard 2). The decline assertion depends on `HAVE_MPFR`
  being compiled in — verify the worktree build has it; skip (loudly) if
  not.
- Default build: compiles; G3 unaffected (code compiled out). The G2
  jac-gate run's real value is **anti-spillover** (finding 6): the edit
  region sits 12 lines above production `ptd_moments_grad_theta` and next
  to the un-gated `ptd_dbg_tape_needs_mpfr`; the shared-helper
  second-caller (`ptd_debug_reverse_grad`) is guarded by
  `dr_reverse_adjoint_gate.py` in the validators build.

## D.4 — `Graph.svgd(exact_moment_grad=...)` (REWRITTEN per Reviewer B; leaf-5-only)

**v1's defects, for the record (Reviewer B, verdict BROKEN):** the
"verified" four-site mapping was false — `:6359`/`:6364` are inside
`Graph.mcmc` (`def svgd` `:5241`, `def mcmc` `:6220`; re-verified
first-hand); the real `Graph.svgd` moments sites are `:6111` (rewards
1-D) and `:6117` (no rewards), with rewards-2D going to
`pmf_and_moments_from_graph_multivariate` (`:6104`) which has **no**
passthrough kwarg (documented CLAUDE.md gap); v1's rewards-leaf forwarding
also exceeded master plan §9's explicit Batch-A gate; and v1's
`_exact_grad_enabled` introspection mechanism does not exist on the built
model.

**v2 scope — exactly master plan §6-D.4, nothing more:**
- New kwarg `exact_moment_grad: bool | None = None` on `Graph.svgd`
  (additive; `Graph.mcmc` untouched).
- `None` (default): not forwarded at all — byte-identical behavior.
- Explicit `True`/`False` on **leaf 5 only** (no-rewards moments leaf,
  site `:6117`): forwarded to `pmf_and_moments_from_graph`.
- Explicit non-`None` on ANY other leaf → `ValueError` naming the leaf and
  its gating batch: rewards 1-D (`:6111`) and rewards 2-D (`:6104`) →
  "blocked until Batch A (rewards support in the exact adjoint)";
  daisy/`epoch_starts` → "no exact variant exists yet (Batch H/G)";
  joint-index → "use the joint-index `exact_grad` story (Batch F/D.3)".
  No inert kwargs; no silent behavior.
- Leaf classification is decidable at the single validation choke point
  (Reviewer B verified: joint ⇔ `_joint_prob_base_graph_indexer is not
  None` `:5940`; daisy ⇔ that + `epoch_starts` `:6017`, guaranteed by
  svgd_config rule R1; rewards by presence/ndim `:6101-6102`).
- **Implementation points (finding 4):** a new rule in the centralized
  svgd_config validator + an **explicit** parameter threaded into
  `from_svgd_call` (its `**_unused` swallows unknowns — do not rely on
  it) + a `LEDGER_OPTION_ORDER` entry so `effective_options()` surfaces
  the kwarg.
- **Docstring (finding 6 + re-review finding 2):** document the leaf map;
  explicitly distinguish `exact_moment_grad` (moments leaves; callee
  default True; reverse-mode) from the joint-index `exact_grad` (default
  False; forward-mode; arriving via D.3) to pre-empt conflation; and state
  that explicit `True` on leaf 5 is honored within the callee's documented
  weight-mode scope (formula/callback graphs still take the callee's
  INFO-logged dynamic decline — inherited contract, not new plumbing).

**G1 gates (mechanisms corrected per finding 3):**
- (a) **Determinism pre-check then bit-identity:** run the unmodified
  build's fixed-seed leaf-5 fit TWICE (run-twice probe); if bit-stable,
  capture golden particles from the `19b86d71` build; post-change
  default-`None` fit must reproduce them exactly. If the probe is NOT
  bit-stable, that is a Class-D discovery (ledger entry) and the gate
  falls back to tolerance-based identity — stated here so a probe failure
  doesn't trigger an ad-hoc gate rewrite (re-review finding 1).
- (b) **Forwarding probe (the only discriminating direction):** attach a
  handler directly to `logging.getLogger('phasic')` at INFO (the
  `tests/pytest/inference/test_exact_grad_discrete.py:218-222` pattern —
  `caplog` is blind, `propagate=False`), run a leaf-5 fit with explicit
  `False` and `n_iterations=1`, assert the construction-time message
  `"exact_moment_grad=False -- using finite differences"` (`:6975-6978`).
- (c) `ValueError` tests for each rejected leaf (rewards-1D, rewards-2D,
  `epoch_starts`, joint-index) — raises fire at validation time,
  pre-construction, so fixtures are cheap (joint fixtures per
  `test_joint_index_callback.py:39-71`).
- (d) `effective_options()` surfaces the new kwarg (ledger-entry test).
- (v1's vacuous rewards-decline gate is dropped with the scope.)

## Gate ladder (process §4; placements corrected per Reviewer B finding 5)

- **G0:** ledger @ `cadf1ca4` (empty; working HEAD `19b86d71`).
- **G1:** per-item gates above.
- **G2:** `inference/test_jax_integration.py` (ledger-clean subset),
  `inference/test_fd_gradient_mixed_scale.py` (correct path per Reviewer A
  finding 7ii), jac-gates anti-spillover run.
- **G3:** full suite (literal green; xfail map 24 intact; skips
  explained) — includes `test_svgd.py` here, once (process G2-table rule).
- **G4:** adversarial diff review — two refuters (technical; API/process).
- **G5:** merge-review section appended here; squash-merge; tracker/
  master-plan/baseline updates; CLAUDE.md line for the new svgd kwarg.

## Risks (v2)

1. `expand_dims` contract — retired by review (precedent-verified); tests
   still assert values.
2. D.2's scratch-buffer walk must not perturb the state later handed to
   `ptd_dbg_reverse_tape` — hence fresh buffers, and the five-gate
   pre/post runs.
3. D.4 mis-classification — single choke point + per-leaf ValueError
   tests; `from_svgd_call`'s `**_unused` swallow is defused by explicit
   threading.
4. Line adjacency with Batch 0's extraction region: exact ranges verified
   (`:10678-10726` vs `:10738+`, comment block between) — disjoint;
   trivial rebase.
5. Validators build isolation: only ever in the worktree env; the main
   checkout's install is never switched.

## Adversarial plan-review record (2026-08-11/12)

Reviewer A (technical): SOUND-WITH-CORRECTIONS — findings 1 (MPFR-gate
mechanics, folded), 3 (repro semantics split, folded), 4 (gate-survival
interpolation, folded as a confirm-run), 5 (pre-change gate baseline,
folded), 6 (anti-spillover framing, folded), 7 (ndim>2, path prefix,
scope confirmations, folded). D.1's fix pattern, bug claim, and gates
survived attack outright.
Reviewer B (D.4/API): BROKEN (v1) — findings 1 (CRITICAL, false site
mapping/mcmv boundary), 2 (scope exceeded master plan §9), 3 (fictional
introspection + vacuous gate), 4 (validator threading + options ledger),
5 (G2/G3 placement + baseline stamp), 6 (kwarg-conflation docstring),
7 (bit-identity procedure). All folded; the rewritten D.4 returns to
Reviewer B for verdict before its implementation starts. Reviewer B also
cleared: None-not-forwarded byte-identity reasoning, accidental-forward
loudness (no `**kwargs` sinks), decline-guard/log existence, leaf-5
default-True reachability, fixture economy.

## Merge review (G5) — 2026-08-13

**All gates green; squash-merged to master.**

- **G0/baseline:** ledger @ `cadf1ca4` (empty known-failure list), working
  base `19b86d71`.
- **G1:** 14 new tests green (5 vmap + 8 kwarg/R29 + 1 direct-SVGD error);
  D.2 guards gate ALL PASS under a fresh validators build (executed after
  its final edit); golden-particle bit-identity **PASS, max-abs-diff 0.0**
  (run-twice determinism probe BIT-STABLE first).
- **G2:** jac-gates ×3 PASS; targeted suites green; the 9 sources-on
  `test_jax_integration` failures differentially confirmed pre-existing on
  untouched master and ledgered (baseline addendum 2026-08-12).
- **D.2 validator gates:** 5/5 PRE-change baseline @ `19b86d71`, 5/5
  POST-change — no bit-rot, no regression.
- **G3:** full suite, chunked (machine-sleep kills forced 12 sub-runs;
  identical coverage): **1885 passed / 0 failed / 84 skipped / 24 xfailed**
  — exact baseline arithmetic (+6 new no-source tests, +8 new source-gated
  skips, xfail map intact).
- **G4:** technical half **SOUND** (0 critical/major); D.4 half
  **SOUND-WITH-CORRECTIONS** — all findings folded and re-tested, incl. the
  R29 `joint_stop_prob` hole (a real contract breach caught by review) and
  the strictly-additive signature position.

**Deviations / notes:**
1. **Commit entanglement (G4-D.4 finding 2, accepted):** the D.4
   `__init__.py` hunks rode inside the D.1 commit (`5c2c55cb`) — a staging
   mistake. Accepted because no item stalled and the squash-merge erases
   intermediate granularity; process lesson: when items share a file,
   commit after each item's edits (folded into practice, not the process
   doc, to avoid churn).
2. **svgd.py one-token change** (helpful error on the direct `SVGD(...)`
   path) explicitly user-approved 2026-08-13; covered by a test.
3. **`pmf_from_cpp` sibling suspicion** ledgered (master plan §16b item 7):
   statically confirmed same-bug-shape, execution probe needs a C++
   model-file fixture — own micro-batch.
4. **G3 execution mode:** background full-suite runs are killed by machine
   sleep on this host; chunked foreground/short-background runs are the
   working pattern (recorded for future batches).
