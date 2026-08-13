# Batch D.3 plan — `Graph.svgd()` leaf-2b plumbing (joint-index + exposure)

**Status: DRAFT v1, pending the mandated two-refuter adversarial plan
review.** Design of record: master plan §6 (D.3) + §16 risk 4 (the
exposure-wrapper interaction "needs its own de-risk pass, not assumed
safe by transitivity") + the Batch F merge review's cross-batch note
(D.3 "inherits F0's jit findings AND must state the raise blast
radius"). Branch: `b3/batchD3-leaf2b`, worktree `../phasic-batchD3`
(Python-only batch, but the worktree keeps the main checkout's install
as the golden pre-D.3 reference; the D0 de-risk experiment runs first,
on the same branch). Baseline: ledger fourth stamp @ `ecd708fc`
(1899/0/84/24, ledger empty; next full run expected 1900 — one test
added at Batch H's G4 fold, post-G3). Master HEAD `cfa1d95a` is
docs-only above the stamp (close-out docs) — G0 records this delta
explicitly. Findings: `b3-batchD3-findings.md`. Experiment:
`experiments/dr_batchD3_exposure_derisk.py`.

## What D.3 is (plain language)

`Graph.svgd()` on a joint-probability graph WITHOUT epochs builds its
model at `src/phasic/__init__.py:6416-6421`:
`_bake_obs = observed_data if exposure is None else None`, then
`Graph.pmf_from_graph_joint_index(self, theta_dim=..., fixed_mask=...,
observed_indices=_bake_obs)` — **never forwarding `exact_grad`**, so
SVGD always gets that callee's default (False, user-decided 2026-08-13,
recorded `04775b63`). When `exposure` IS set, the model is built
NON-baked (`observed_indices=None`) — exactly the mode Batch F's
probe-and-commit exact path supports — and SVGD then applies
`_wrap_model_with_exposure` (`src/phasic/svgd.py:274-340`: a
`jax.lax.map` over per-obs `(theta_i, time_i)` pairs with
`theta[exposure_param_index]` scaled by alpha_i) on top. D.3 plumbs an
`exact_grad` kwarg from `Graph.svgd` to THIS leaf — the one leaf where
the exact joint-index path is structurally reachable today with no
other batch — with R29-style validation everywhere else.

## Scope decisions (explicit; no silent fallbacks)

1. **Accepted leaf (the ONLY one):** joint-prob graph, `epoch_starts is
   None`, `exposure is not None` (leaf 2b). Forwarding: `None` (default)
   = kwarg omitted, callee default governs — byte-identical to today;
   `True`/`False` = forwarded explicitly.
2. **Explicit `exact_grad` is REJECTED (`SvgdConfigError`, new rule
   R30) on every other leaf**, mirroring R29's discipline
   (`svgd_config.py:1111-1140`):
   - `epoch_starts` set → the daisy path's exact machinery is the
     SEPARATE `exact_final_grad` (Batch H, internal; public plumbing =
     Batch G leaf 1) — message says so;
   - moments/rewards leaves → different machinery (R29's domain);
   - joint-index BAKED variant (`exposure is None`) → the exact wiring
     statically excludes baked mode until Batch E; an explicit True
     would be structurally inert (probe-fail → whole-model FD + INFO
     log). Reject with a message naming Batch E. (An explicit False is
     ALSO rejected there for symmetry with R29's "None = not forwarded"
     contract — rejecting only True would make False silently
     equivalent to the default in a way users can't distinguish.)
   - R28 (`joint_index=False` incompatible with joint-prob) and R27
     interplay: R30 must not contradict them — the rule runs on the
     same config-classifier fields; the review checks for collisions.
3. **Blast-radius statement (F merge-review obligation), verbatim in
   the `Graph.svgd` docstring:** with `exact_grad=True`, the committed
   model RAISES on a per-theta decline (no FD fallback, the recorded
   user decision); under SVGD this means ONE particle whose theta
   declines halts the ENTIRE cloud mid-optimization. The F0 experiment
   (`dr_batchF_jit_raise_derisk.py`) proved the message survives
   `vmap(jit(grad))`; D0 below extends that proof through the exposure
   wrapper's `lax.map`.
4. **The svgd docstring paragraph at `__init__.py:5744-5751`** (which
   currently describes `exact_grad` as `pmf_from_graph_joint_index`'s
   separate kwarg) is updated to document the new plumbing + R30 scope
   + blast radius.

## D0 — de-risk experiment (mandated by §16 risk 4; runs BEFORE implementation)

`experiments/dr_batchD3_exposure_derisk.py`, on the current (post-H)
install. Fixture: small coalescent joint-prob graph (the Batch-H test
family, nr_samples=3) + a distinct-valued exposure array. Checks:

- **D0(i) correctness through the wrapper:** build
  `pmf_from_graph_joint_index(jpg, exact_grad=True)` directly
  (non-baked), wrap with `_wrap_model_with_exposure`, and compare
  `jax.grad` of an SVGD-shaped loss against (a) the same construction
  with `exact_grad=False` (FD), and (b) a dense-JAX oracle (per-obs
  scaled theta → subset-sojourn solve — the Batch-H oracle machinery,
  simplified: the joint-index model returns sojourn values directly).
  Target: exact-vs-oracle ≤1e-9; record FD-vs-oracle for the
  improvement statement.
- **D0(ii) the committed path really runs under the wrapper:**
  trace-time spy on `phasic.ffi_wrappers.compute_sojourn_times_ffi`
  (patched BEFORE construction — the F2(a)/H-test-8 seam lesson),
  counting emitted ops under `vmap(jit(grad))` THROUGH the wrapper's
  `lax.map`: the committed model must trace NO FD perturbation ops;
  assert exact expected counts with a nonzero-floor guard (the H G4
  dead-spy lesson).
- **D0(iii) raise legibility through `lax.map`:** force a committed
  per-theta decline (`PHASIC_CONDITION_THRESHOLD≈0` set AFTER
  construction, the F2(c) env mechanism; MPFR-build skip) and assert
  the diagnostic message is findable in the raised error under
  `vmap(jit(grad))` of the WRAPPED model — the blast-radius claim,
  demonstrated.
- **D0(iv) cost note:** wall-clock of exact vs FD backward through the
  wrapper at n_obs ∈ {5, 20} (the wrapper is lax.map-sequential either
  way; the exact path is forward-mode P-scaled — at this model's P=2
  the F-batch D3 benchmark says FD is cheaper; RECORD the numbers, the
  kwarg stays opt-in default-None/False so cost is user-chosen).
- **GO/NO-GO:** D0(i) ≤1e-9 AND D0(ii) zero FD ops AND D0(iii) legible
  → implement. Any failure → STOP, findings to the user (the
  exposure-wrapper interaction was explicitly never verified — a
  failure here is a real discovery, not a plan bug).

## Implementation (after D0 GO)

- **I1 — svgd_config:** `exact_grad: Optional[bool] = None` field +
  `'exact_grad'` in VALID_KEYS + resolver passthrough + rule R30
  (`_check_R30_exact_grad_leaf_scope`) per scope decision 2. Rule
  message quality: each rejection names the leaf it fired on and the
  batch that will unlock it (E for baked, G for daisy).
- **I2 — `Graph.svgd`:** accept the kwarg, thread through config,
  forward at the leaf-2 call site (`:6417-6421`) ONLY when not None;
  docstring updates (scope decision 3/4).
- **I3 — tests** (`tests/pytest/inference/test_svgd_exact_grad_leaf2b.py`):
  1. R30 rejections: epoch_starts / rewards / moments-leaf / baked
     (no-exposure) each raise `SvgdConfigError` with the documented
     message; `None` never raises anywhere.
  2. Accepted leaf, `exact_grad=True`: the svgd-built model commits
     (trace-time spy — no FD ops through the wrapper), and a short
     `SVGD(...)` construction runs its first gradient step finite.
  3. Accepted leaf, `exact_grad=False` explicitly: gradient
     BITWISE-equal to the no-kwarg default (same code path).
  4. End-to-end gradient parity vs the D0 oracle (productionized
     D0(i)).
  5. Blast radius: forced decline (env mechanism, MPFR skip) raises
     legibly through the wrapped model under `vmap(jit(grad))`.
  6. Golden: the no-kwarg default svgd gradient BITWISE-equal to the
     pre-D.3 install (cross-install golden via the worktree pattern —
     dump under the main checkout's install, check under the branch
     install; the H micro-gate (a2) template).
- **Existing-test fate table: NO existing test changes state.** The
  kwarg defaults to None (not forwarded); R30 fires only on explicit
  values. All joint-index files, svgd files, and config-rule tests must
  keep passing unchanged. Any deviation = G1 failure.

## Gates

- **G0:** recorded above (ledger @ `ecd708fc`, master `cfa1d95a`
  docs-only above it).
- **G1:** D0 experiment gates + the I3 suite + fate table holds.
- **G2:** `inference/test_exact_grad_joint_index.py`,
  `test_joint_index_callback.py`, `inference/test_optimized_joint_index.py`,
  `test_gate_daisy_chain_joint_probs.py` (process joint-index row);
  the svgd config/validation test files (exact filenames verified at
  implementation from the repo — the plan review checks the G2 map);
  always-run `inference/test_fd_gradient_mixed_scale.py`.
- **G3:** chunked full suite vs ledger @ `ecd708fc` (expect 1900 + I3's
  new tests).
- **G4:** two diff refuters (config-rule/wiring fidelity; tests/process).
- **G5:** merge review; ledger re-stamp; tracker (D.3 row → merged;
  Phase-1b complete); master §15 tick; CLAUDE.md note; memory; install
  rebuild. No §16b items are owed by this batch (risk-4 is discharged
  by D0; if D0 finds a real defect it goes to the user first).

## Risks

1. The exposure wrapper's `lax.map` × `pure_callback`(sequential) ×
   committed-raise composition is exactly what D0 exists to test —
   implementation only starts on GO.
2. R30 vs R27/R28/R29 interplay: the classifier fields must agree; a
   wrong classification order was a REAL bug class before (R29's
   joint_stop_prob hole, caught at D.4's G4). The plan review + G4
   both check it.
3. Line anchors in this plan are post-H (`ecd708fc`); the plan review
   verifies them.
4. The F-batch probe runs at model CONSTRUCTION inside `Graph.svgd` —
   construction happens once per svgd call, before particles exist;
   probe-failure (structural) → whole-model FD with INFO log, which is
   the documented Batch-F contract, NOT an R30 concern (R30 gates
   where the kwarg is structurally meaningless, not where the probe
   declines).

## Adversarial plan-review record (2026-08-13) — VERDICT: BROKEN, plan HALTED pre-D0

Both refuters independently: **BROKEN**. The core premise — leaf 2b
"structurally reachable today" — is FALSE in shipped code:

- **Rule R9** (`svgd_config.py:805-820`, shipped 2026-05-15 `f6fcbce7`,
  test-pinned at `test_svgd_config.py:185-198`) statically rejects
  exposure + joint-prob + no-epochs BEFORE model construction, with a
  still-valid cost rationale (the exposure wrapper = O(n_obs) full
  evaluations per gradient) and a message directing users to
  `epoch_starts=[0.0]` (the daisy route). The `else None` arm of
  `_bake_obs` at `__init__.py:6416` is DEAD CODE for joint_prob graphs
  under `Graph.svgd`. Master plan §6 D.3 and §16 risk 4 rest on the
  same false assumption (the atlas's "not structurally blocked" was a
  model-builder-level statement, not an svgd-entry-point one) — dated
  Class-D amendment filed in the master plan.
- The ONLY configuration reaching `:6416` with exposure is a
  `joint_stop_prob` graph, via an R9 classifier hole of exactly the
  R29 bug class (R9 tests only `graph_kind == 'joint_prob'`;
  jsp graphs carry the base-graph indexer and enter the same branch).
  No test composes joint_stop_prob + exposure anywhere; intent
  undeterminable — LEDGERED as §16b item 9.
- D0 as designed builds the model DIRECTLY and would have returned GO
  on an unshippable batch (process refuter MAJOR 2: no front-door
  svgd smoke test).
- Everything else verified sound by review (spy seam, env decline
  mechanism post-H, wrapper composition mechanics — live-probed:
  lax.map over a pure_callback custom_vjp under vmap(jit(grad)) works
  and raises legibly; R29 False-rejection symmetry confirmed; G0
  claims; naming). The mechanics survive; the PREMISE does not.

**USER DECISION REQUIRED (options per both reviews):**
(i) relax R9 for the exact-grad case — modifies shipped, test-pinned
    validation whose cost rationale still holds (the exact path does
    not remove the O(n_obs) wrapper structure);
(ii) re-scope the leaf to joint_stop_prob + exposure — rides the
    probable R9 classifier hole; needs the hole settled first;
(iii) FOLD D.3 INTO BATCH G: R9's own documented fix
    (`epoch_starts=[0.0]`) is now backed by Batch H — the daisy route
    has INTERNAL exposure (one OpenMP-batched FFI call, no lax.map)
    and `exact_final_grad` covers the WHOLE model when n_epochs==1
    (Batch H test 10). D.3's user value (exact gradients for
    exposure-bearing joint-prob SVGD) ships via G leaf 1 with no
    validation change and a strictly better cost profile.

Minor corrections recorded for whichever successor plan: front-door
D0(v); joint_stop_prob cases in any R-rule design; ledger/
effective_options tests (D.4 precedent); golden = single model-grad
call; fixed_mask/scalar-exposure variants; R29-message staleness
decision; derisk/* branch naming; G2 = test_svgd_config.py +
inference/test_svgd_exact_moment_grad_kwarg.py +
inference/test_svgd_exposure.py + inference/test_svgd_api_parity.py +
test_svgd_assumptions.py (+ a proposed process-doc G2 row).
