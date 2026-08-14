# Batch E plan — joint-index BAKED-mode exact gradient + svgd leaf-2 plumbing

**Status: v2, 2026-08-13 — two-refuter review DONE (both
SOUND-WITH-CORRECTIONS; one CRITICAL: the DEFAULT `joint_prob_graph()`
is `was_dph=True`, so R31 must pre-empt was_dph or the public kwarg is
silently inert on default jpgs — folded; all other findings folded
below; review record at the end). The R31 pre-emption questions are
RESOLVED by the reviews (pre-empt weight-mode AND was_dph at
validation). One fold-time user question: the svgd.py one-token guard
(third of its kind). Cleared for E0 then implementation.** Design of record: master plan §7 (the scatter-add derivation,
verified there by direct derivation; sequencing option (i) happened —
Batch F merged first, so E builds directly into the probe-and-commit
wiring) + the F merge review's requirement ("E probes the ACTUAL baked
union") + master §9 leaf 2's gate ("ship leaf-2's `Graph.svgd()` kwarg
pass-through **alongside Batch E**" — satisfied by doing both in this
batch). Branch: `b3/batchE-baked-exact`, worktree `../phasic-batchE`
(Python-only; worktree keeps the pre-E install as golden reference).
Baseline: ledger fifth stamp @ `0c052cfe` (1917/0/84/24, empty; next
full run 1919 — two tests added at G.1's G4 fold, post-G3). G0
enumerates the master delta at branch time. Findings:
`b3-batchE-findings.md`. Experiments: `experiments/dr_batchE_*.py`.

## What E delivers (plain language)

`Graph.svgd()` on a CONTINUOUS joint-probability graph
(`joint_prob_graph(..., discrete=False)`) with NO epochs and NO
exposure bakes the
observations into the model (`observed_indices`), which today
STATICALLY excludes the exact gradient (`__init__.py:8287-8293`: INFO
log → FD). E implements the baked exact backward (no new C, no new FFI
shape) and plumbs a public `Graph.svgd(exact_grad=...)` kwarg to this
leaf. **CRITICAL caveat found by review: the DEFAULT
`joint_prob_graph()` is DISCRETE (`discrete: bool = True` sets
`was_dph`, `__init__.py:9465-9472, 9844-9845`) and the exact path
excludes was_dph — so the kwarg applies to CONTINUOUS jpgs; on a
default jpg, R31 REJECTS an explicit value with a "rebuild with
joint_prob_graph(..., discrete=False)" message (loud, never
silently-inert; the repo's tutorials build default jpgs, so this
message is the user's migration path).**

## The mathematics (from master §7, verified there; restated)

Baked forward — **anchor corrected by review: there are TWO baked
`_compute_pure` closures; the LIVE one for the exact path is the
linear-mode FFI + custom_vmap variant at `__init__.py:8526-8593`
(gather at `:8588`); the `:8463-8497` variant is the callback-
weight-mode branch, statically excluded from exact and UNTOUCHED by
E** — computes `uniq_sojourn` at the STATIC unique indices;
`uniq_probs = uniq_sojourn / sum(all_sojourn)`; `sojourn_probs =
uniq_probs[_inverse_idx_jnp]` — a GATHER. Its VJP is a SCATTER-ADD:
`g_uniq = jnp.zeros(n_unique, dtype).at[_inverse_idx_jnp].add(g_visits)`
(one JAX primitive). The EXISTING quotient rule (`:8701-8727`) then
applies verbatim at unique granularity: replace the runtime `_vi_norm`
with the static `_uniq_idx_jnp` in the two FFI forwards and the
Jacobian callback, and contract `exact_tbm = d_uniq_probs_exact.T @
g_uniq`. `_fd_theta_bar` needs ZERO changes (it treats the whole
forward, gather included, as a black box).

## Scope

1. **Model level (`pmf_from_graph_joint_index`):**
   - Remove the `elif _baked:` static exclusion (`:8287-8293`); baked
     mode joins the `_jix_exact_enabled = True` arm (the OTHER
     exclusions — was_dph, weight-mode, theta_dim-override — keep
     firing first, unchanged; post-E ladder-order regression tests
     assert was_dph+baked and log+baked still decline/raise with the
     EARLIER arm's message).
   - **Probe index set = `union(_uniq_idx_np, all_terminal_np)`** (the
     F merge-review requirement): both operands are construction-known,
     so the probe is EXACT for baked mode — there are NO runtime index
     rows, hence no residual "index-dependent decline the probe cannot
     see" class at all. (The F probe's `[0]` member existed for the
     runtime-rows residual; for baked it is subsumed — stated, and the
     review checks the reasoning.) Probe length check
     `len(raw) == len(probe_union) * P`.
   - Backward baked branch: scatter-add + quotient at unique
     granularity as above. **Construction-time bounds validation of the
     baked indices is a HARD I1 deliverable (review finding): today the
     public builder does NO range validation (`:8226-8240`), the sojourn
     FFI NaN-fills bad indices silently, and E's per-call defense
     removal would otherwise leave garbage indices to surface as a
     probe-decline → silent-FD → NaN forward. A dedicated
     construction-raise test ships with it.** The committed-decline RAISE semantics and message
     are IDENTICAL to non-baked (same callback, same causes); the
     conditioning gate stays DEFAULT-ON for this path (the Batch-H
     opt-out was the daisy caller's, user-scoped; jpg start IPVs are
     clean vectors, and the H de-risk's boundary mapping showed the
     gate's realistic-theta declines need WIDE-dynamic-range IPVs —
     the joint-index fixtures probe fine at ones AND typical theta;
     the E de-risk experiment measures this on the real fixture before
     implementation — PRE-MEASURED by the design refuter on the G.1
     `_base_graph` continuous jpg: COMPUTES at theta=ones and
     theta=[1e-4,1e-4]; proposed backward matches native CD to 2.2e-7;
     note uniq ⊆ all_terminal for svgd-mapped obs, so the baked probe
     union == all_terminal in practice).
   - The "free bonus" hoist (master §7): in baked mode the callback's
     `union_idx`/`obs_pos`/`all_pos` are static — hoisted to
     construction. NON-baked callback code path stays byte-identical.
   - Docstring: the baked-exclusion sentence in the `exact_grad`
     docstring (written at F) becomes stale — updated.
2. **svgd leaf-2 plumbing (`Graph.svgd(exact_grad: Optional[bool] =
   None)`):**
   - Forwarded at the leaf-2 call site (the `pmf_from_graph_joint_index`
     call, post-G.1 `__init__.py:6448-6453`) only when not None
     (None = not forwarded, byte-identical default — the
     G.1/D.4 contract, incl. the forwarding-discrimination test).
   - **New rule R31** (`_check_R31_exact_grad_leaf_scope`), R29/R30
     discipline (any non-None rejected off-leaf): accepted =
     joint-prob-KIND graph (`graph_kind in ('joint_prob',
     'joint_stop_prob')` — jsp graphs route to the same leaf; the
     R29/R9 classifier-hole lesson applied on day one), no
     `epoch_starts` (message points at `exact_final_grad`), no
     exposure (unreachable anyway — R9 rejects exposure+no-epochs on
     both kinds; R31's presence-check is belt-and-braces and its
     message defers to R9's). Rejected: epochs (→ exact_final_grad);
     moments/rewards leaves. **Pre-emption, RESOLVED by both reviews
     concordantly (the v1 REVIEW QUESTION is closed): R31 pre-empts
     (a) `effective_weight_mode != 'linear'` (the R30 precedent; NOTE
     the corrected mechanism narrative — a formula-mode jpg is
     STATICALLY excluded by the builder's ladder before any probe
     exists, and log mode hard-raises at construction independent of
     exact_grad, so v1's "probe-fail→FD" story was wrong) and (b)
     `was_dph` (the CRITICAL finding: DEFAULT jpgs are was_dph;
     `from_svgd_call` captures `graph.get_was_dph()` into a new
     config field; the message prescribes
     `joint_prob_graph(..., discrete=False)`; native DPH — is_discrete
     without was_dph — is NOT rejected, matching the C scope). The
     residual builder-level logged declines (theta_dim-override;
     structural probe failure) keep the F contract: explicit True →
     FD + INFO log, DOCUMENTED in the R31/svgd docstrings and TESTED
     (I3 item 8's explicit-True theta-dim-override cell).**
   - Ledger entry (`LEDGER_OPTION_ORDER`), `from_svgd_call` named
     param, `SvgdConfig` field.
   - `svgd.py` token (`_GRAPH_SVGD_ONLY_KWARGS` + `'exact_grad'`):
     same one-token shape the user approved for D.4 and G.1 — ASKED
     at fold time again (the file carries a standing do-not-modify
     rule; two precedents, but each approval was explicit).
   - Shipped-text updates (flagged): R30's no-epochs message clause
     "svgd plumbing for that leaf is not available" becomes FALSE →
     points at `exact_grad`; the `Graph.svgd` docstring paragraph
     (post-G.1) gains the new kwarg's parameter entry (blast-radius
     statement included — same committed-raise semantics; plus the
     probe-fail→FD+log nuance and the forward-mode P-scaling cost
     note from the F docstring, so svgd users see the same trade the
     model-level docstring states).
3. **Out of scope, stated:** leaves 3/4 (rewards — Batch A); the
   joint-index `exact_grad` DEFAULT stays False at both levels
   (user decision `04775b63`; svgd None = not forwarded); non-baked
   behavior byte-identical; `Graph.mcmc` untouched.

## E0 — de-risk experiment (pre-implementation)

`experiments/dr_batchE_baked_derisk.py` on the current install:
Fixture: the G.1 `_base_graph` coalescent jpg (NR=3, reward_limit=4,
mutation_rate=1e-4) built with **`discrete=False`** (the CRITICAL
finding: a default/discrete jpg would measure the was_dph decline, not
the gate). The design refuter PRE-RAN the core: GO expected.
- (i) **Oracle check of the scatter-add derivation on real numbers**:
  build the baked model's forward pieces directly (pybind sojourn at
  uniq + normalization + gather), compute the proposed backward
  (scatter-add + quotient at uniq via `_sojourn_grad_theta_subset`),
  and compare against `jax.jacobian` of a dense-JAX replica AND
  against tight central FD of the baked forward. Target ≤1e-9.
- (ii) **Probe/gate reality on the real fixture**: a DYNAMIC-RANGE
  SWEEP in the H boundary-mapping style (not two points): theta over
  {ones, 1e-2, 1e-4, mixed-scale} × the fixture jpg; plus a
  trap/deficit-sink DISPOSITION — the CLAUDE.md-flagged class E makes
  reachable as a committed raise: attempt a small trap-bearing jpg
  fixture (the H micro-gate (c) manual-graph technique); if
  inconstructible at jpg level, record that with reasoning. STOP →
  user decision if ANY sweep point declines (incl. theta=ones —
  wording fixed: a ones-decline is the same disposition, not generic
  NO-GO).
- (iii) **Front-door smoke**: `Graph.svgd(obs)` (no kwarg) on the
  fixture reaches the baked leaf (spy on
  `phasic.ffi_wrappers.compute_sojourn_times_ffi`) — the D.3 lesson.
- GO/NO-GO: (i) ≤1e-9 AND (ii) computes at both thetas AND (iii)
  reaches the leaf.

## Implementation (after E0 GO)

- **I1** model-level baked exact branch + probe union + hoist +
  docstring; **I2** svgd_config (R31, field, param, ledger, R30
  message update) + `Graph.svgd` (kwarg, threading, forwarding,
  docstring) + svgd.py token (post-approval); **I3** tests
  (`tests/pytest/inference/test_svgd_exact_grad_kwarg.py` (pattern-consistent name) for
  the svgd level, additions to
  `inference/test_exact_grad_joint_index.py`'s file? NO — new file
  `inference/test_exact_grad_joint_index_baked.py` for the model
  level, keeping the F file untouched per the fate table):
  1. Baked exact-vs-oracle parity (E0(i) productionized; benign +
     mixed-scale theta; ≥1e3 improvement floor vs FD).
     Degenerate shapes included: n_unique==1 (all obs identical),
     n_obs==1, and UNSORTED/PERMUTED duplicated observations (mirror
     the non-baked `..._unsorted_duplicated_subset_indices` test).
  2. Baked exact-vs-FD parity at tolerance anchored to measured
     actuals at authoring time (the G.1 G4 discipline — measure
     FIRST, then set tolerance with ~100x headroom, comment the
     measured number truthfully).
  3. Trace-time spy: committed baked model traces ZERO FD ops under
     `vmap(jit(grad))`; ABSOLUTE delta pinned vs the FD model
     (measured at authoring; nonzero-floor guard).
  4. Duplicated observations: scatter-add correctness with heavy
     duplication (n_obs >> n_unique) — gradient equals the
     dedup-weighted expectation (oracle).
  5. Probe exactness: a committed baked model's first real call
     cannot hit an index-dependent decline (the probe set == the
     call set; assert via a spy that the probe and call unions are
     identical).
  6. Committed-decline raise legibility under `vmap(jit(grad))`
     (monkeypatched decline, F test-8(b) mechanism).
  7. fixed_mask × baked exact (fixed slots exactly 0.0).
  8. R31 rejections (epochs → names exact_final_grad; moments;
     rewards 1-D/2-D; WAS_DPH default-jpg cell with the rebuild
     message; weight-mode cells; `joint_index=False` cell — R28's
     raise shadows R31, stated); the accepted-leaf None-sweep (jsp
     kind VALIDATION-LEVEL only — no jsp fit exists, reason in-code
     as at G.1); forwarding discrimination (explicit False arrives,
     None absent); ledger default/user; constructor-guard test
     (post-approval); front-door spy test (svgd-built CONTINUOUS-jpg
     baked model commits; spy asserts the spied index-array length ==
     n_unique < n_obs to prove the BAKED leaf specifically; patched
     BEFORE Graph.svgd); explicit-True + theta-dim-override →
     FD + INFO log, completes, no raise (the F-contract residual).
  9. Cross-install golden: svgd-built baked model gradient (no
     kwarg), dump pre-E / check post-E, bitwise
     (`experiments/dr_batchE_golden.py`).
- **Existing-test fate table (names/counts corrected by review):**
  `inference/test_exact_grad_joint_index.py::
  test_observed_indices_baked_mode_declines_and_logs` (`:385-399`)
  **BREAKS BY DESIGN** — baked no longer declines; rewritten as
  `test_observed_indices_baked_mode_commits` (probe succeeds at the
  baked union, INFO decline-log GONE, exact engaged; PRESERVES the
  original's second half: finite gradient, runtime vidx ignored) IN
  THE F FILE — the batch's only existing-test change, byte-diff
  discipline on the file's other **16** tests (the file has 17, no
  parametrization). Everything else — the G.1 svgd suite (19
  collected), daisy/epoch files, svgd-config files — must pass
  unchanged. Any other deviation = G1 failure.

## Gates

- **G0:** E0 GO + ledger @ `0c052cfe` + enumerated master delta.
- **G1:** I3 suites + the fate table holds exactly (one enumerated
  rewrite, nothing else).
- **G2:** joint-index/sojourn row (process map, as amended at G.1:
  `test_joint_index_callback.py`, `inference/test_optimized_joint_index.py`,
  `test_gate_daisy_chain_joint_probs.py`,
  `inference/test_exact_grad_joint_index.py`), the svgd-config row
  (all six files incl. G.1's), always-run
  `inference/test_fd_gradient_mixed_scale.py`; sources-on runs treat
  the 9 ledgered `test_jax_integration` failures per the ledger
  addendum.
- **G3:** chunked (`-rf` per the adopted amendment) vs ledger @
  `0c052cfe` (expect 1919 + new).
- **G4:** two diff refuters. **G5:** merge review (shipped-text
  statement: the F-file test rewrite, R30 message clause
  `svgd_config.py:1213-1215`, F docstring baked sentence
  `__init__.py:8093-8095`); ledger re-stamp; tracker (+ ready-to-push
  note); master §7/§9-leaf-2/§15 ticks + **§16 risk item 5 tick**
  (the "scatter-add must be numerically gated" clause — discharged by
  E0(i)/I3-1) + **§16b PHASE-BOUNDARY review** (E completes Phase 2;
  process §6); **process-map amendment** adding the two new suites to
  their G2 rows (the G.1 precedent, `600c4b84`); CLAUDE.md
  (`:235-236` — the deferral sentence couples baked with
  formula/callback: the edit removes ONLY baked, preserving the
  weight-mode deferrals); memory; install rebuild.

## Risks

1. The R31 pre-emption questions (scope 2) — RESOLVED by this plan
   review (weight-mode AND was_dph pre-empted), not mid-batch.
2. The E0(ii) gate-decline measurement is the one genuine unknown
   (the H finding's transfer surface); NO-GO path defined.
3. The F-file single-test rewrite is the batch's only existing-test
   change — byte-diff discipline on the rest of that file.
4. Non-baked byte-identity: golden = the F suite's own
   `exact_grad=False`/non-baked tests + the cross-install golden.


## Adversarial plan-review record (2026-08-13; v1 → v2)

Design/wiring refuter: SOUND-WITH-CORRECTIONS — CRITICAL 1 (default
jpg is was_dph → R31 must pre-empt was_dph or the kwarg is silently
inert on default jpgs; headline corrected; E0/tests must use
discrete=False) → folded throughout; MAJOR 2 (wrong forward anchor:
the live baked forward is `:8526-8593`, the `:8463-8497` variant is
callback-mode, exact-unreachable; patch-before-construction spy
requirement) → folded; MAJOR 3 (R31 weight-mode pre-emption RESOLVED
yes, with the mechanism narrative corrected: static decline, not
probe-fail; log hard-raises) → folded; MAJOR 4 (E0 fixture named +
PRE-MEASURED: computes at both thetas, backward matches CD to 2.2e-7;
uniq ⊆ all_terminal in practice) → folded; MINORs 5-9 (fate-table
name `test_observed_indices_baked_mode_declines_and_logs` + 17/16
counts; construction bounds validation = hard deliverable; jsp cell
validation-only; call-site `:6448-6453`) → folded. Process/tests
refuter: SOUND-WITH-CORRECTIONS — MAJORs 1-6 (fate-table name/count;
two-forward duality; bounds-check migration test; E0(ii) sweep +
trap disposition + fixture naming; explicit-True-builder-declines
cell test; G5 process-map amendment) → folded; MINORs 7-14
(call-site drift; degenerate scatter shapes; ladder-order regression
tests; joint_index kwarg cell; E0(iii) baked discrimination via
index length; G5 §16-risk-5 tick + phase-boundary review +
ready-to-push; G2 map-match statement; naming pattern; CLAUDE.md
`:235-236` location + coupling caveat) → folded. R31 pre-emption
question: RESOLVED by both reviews concordantly (pre-empt weight-mode
AND was_dph); no user escalation needed. Remaining fold-time user
question: the svgd.py one-token guard (third approval of the same
shape). **DECIDED (user, fold-time): YES — add the token; the guard
test ships.**

## Dated amendment (2026-08-14) — the HALT, the user decision, and `exact_grad_decline`

During G1, running REAL svgd fits fired the plan's E0(ii) STOP clause
late: SVGD's log-scale particle init (sd=5.0) routinely creates theta
ratios ≥1e8 where the conditioning gate declines — the committed-raise
contract killed first fits, and the gate-lifted answers were measured
34-144% OFF tight FD exactly there (the gate's genuine job; probe table
in the findings doc). **USER DECISION (2026-08-14): host-side
per-particle FD fallback + WARNING for the svgd entry.** Implemented as
a new model kwarg `exact_grad_decline={'raise','fd'}` (default 'raise'
preserves the F hard-stop contract at the model level; the svgd leaf
forwards 'fd'). The fallback computes the declined call's Jacobian rows
by relative-step central FD on the raw sojourn values on the private
clone, then the exact quotient rule downstream — value-correctness
verified by the G4 wiring refuter (1.8e-10 vs tight FD at a REAL gate
decline) and pinned by a mixed-batch value test at the G4 fold.
Process lesson recorded: de-risk sweeps for svgd-facing features must
sample from the ACTUAL particle-init distribution, not hand-picked
scales.

## G4 review record (2026-08-14)

Both diff refuters SOUND-WITH-CORRECTIONS; numerics comprehensively
confirmed by independent probes (fallback values 1.1e-10-1.8e-10 vs
tight FD both arms incl. a REAL decline; per-particle isolation exact:
1 WARN, computable rows bitwise-stable; scatter-add under both
jit/vmap orders 5e-11; R31 NOT inert — get_was_dph probed live on
default and continuous jpgs; svgd-level exact_grad_decline pass-through
impossible — TypeError). Folded: the CRITICAL stale svgd docstring
(still promised the raise the decision replaced); exact_grad_decline
documented (Parameters entry, exact_grad block, RuntimeError remedy
mention); baked-probe sentence fixed; fallback comment de-garbled +
step floor 1e-10; ndim>1 observed_indices rejected; I3-2's err_fd
bound restored (1e-4, measured ~1e-6); delta==14 comment de-garbled;
value-level mixed-batch fallback test + exact_grad_decline ValueError
test + 2-D-rejection test + jsp kind-regression guard added; R31's
rewards pointer message qualified. Recorded-not-fixed (merge review
deviations): WARN volume unbounded per fit (measured 21 lines/5-iter
forced-decline fit — follow-up candidate); F-file rewrite's oracle
assert compares exact zeros (honest NOTE in-code; strong parity lives
in the baked suite); ladder log-cell shipped as formula-cell (log
hard-raises at construction); svgd-level theta-dim cell model-level
only; the chunk-runner initially tail-1'd away -rf names (fixed
mid-run; process amendment wording strengthened at G5).

## Merge review (G5) — 2026-08-14

**All gates green; squash-merged to master.**

- **G0:** E0 GO (findings: backward vs shipped exact 1.4e-12; gate
  computes at all moderate scales; traps outside baked index sets;
  front-door dedup proven); ledger fifth stamp `0c052cfe`
  (1917/0/84/24; 1919 expected on this tree); master delta docs-only.
- **G1:** 45 at implementation → **49 after the G4 fold** (13-file
  count: baked 16, svgd 16, F file 17 with ONE fate rewrite).
- **G2:** **156 / 3 / 1** — zero flips across the 12-file surface.
- **G3:** chunked (31 groups, `-rf`): **1947 / 0 / 84 / 24** = ledger
  1919 + 28 new; skips/xfails ledger-identical. One ab-group
  first-pass transient re-ran green twice; the runner initially
  discarded `-rf` names (fixed mid-run; amendment strengthened below).
- **G4:** two refuters SOUND-WITH-CORRECTIONS, zero live numeric
  defects; probes confirmed fallback values (1.1-1.8e-10 incl. a REAL
  gate decline), per-particle isolation, both jit/vmap orders, R31
  non-inertness (get_was_dph live-probed). All corrections folded
  (`818bc911`).

**Shipped-text statement:** the F-file single-test fate rewrite; the
R30 no-epochs clause; the F docstring baked sentence + probe-set
sentence; the svgd exact_grad docstring failure-mode paragraph
(REWRITTEN post-decision); NEW public model kwarg `exact_grad_decline`
(+ docs + validation + RuntimeError remedy mention); R31 (new rule) +
its rewards-pointer qualification; one user-approved svgd.py token.

**Deviations recorded:** WARN volume unbounded per fit (follow-up
candidate, not ledgered — derivable); F-file rewrite's oracle assert
compares exact zeros (honest in-code NOTE; strong parity in the baked
suite); ladder log-cell shipped as formula-cell (log hard-raises at
construction — the plan's own corrected narrative); svgd-level
theta-dim cell model-level only; E0(ii)'s sweep under-sampled the
SVGD init distribution (process lesson recorded in the amendment).

**Delivered:** the most common CONTINUOUS joint-prob SVGD case gets
opt-in exact gradients (`svgd(obs, exact_grad=True)` on a
`discrete=False` jpg — baked/dedup, probe-exact index sets); default
(discrete) jpgs get a loud rebuild-with-discrete=False message; a
declined particle falls back to host-side FD with a WARNING instead of
killing the cloud (user decision); §16 risk item 5 discharged; Phase 2
complete.
