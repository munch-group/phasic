# Batch H plan — daisy-chain FINAL-epoch exact gradient (de-risk first)

**Status: v3, 2026-08-13 — de-risk phase COMPLETE with a clean GO
(`b3-batchH-findings.md`: all H0 gates pass, composed gradient 3.6e5×
more accurate than FD at 7.4% of the FD backward's cost; §16b item 3
declined with evidence). Both queued user decisions are TAKEN
(2026-08-13): (1) the conditioning gate gets an ADDITIVE C-side opt-out
for this caller — default behavior unchanged everywhere else; (2) the
joint-index `exact_grad` default stays False (recorded separately,
master `04775b63`). The v3 implementation plan below is the plan of
record, pending its OWN two-refuter adversarial review before any
implementation code.**

*(v2 header, superseded but retained for the record: the v1→v2 plan
review found 2 CRITICAL + 5 MAJOR math/design and 1 CRITICAL + 3 MAJOR
process defects, folded as the amendments below; v2 cleared the de-risk
phase only.)* Branch: `derisk/batchH-final-epoch` (then
`b3/batchH-final-epoch`). Worktree + own pixi env required the moment any
native change enters (H2's FFI-extension option); pure-`experiments/`
de-risk may run in the main checkout on the de-risk branch. Baseline:
ledger @ `eaf86e82` (third stamp, 1888/0/84/24, known-failure ledger
EMPTY; next full run expected 1889 — one test added post-G3). Master HEAD
`e06ea0c6` is docs-only above the stamp: G0 for the de-risk branch records
this explicitly rather than assuming docs-only counts as fresh.
Findings doc: `b3-batchH-findings.md`. De-risk scripts:
`experiments/dr_batchH_*.py`.

## Goal

`final_read='sojourn'` (the shipped default since `9a80ac45`) already
reads the final epoch via `joint_sojourn_graph()` — an exact,
granularity-free ELIMINATION solve — but the gradient still bulk-FDs the
whole chain. This batch gives the FINAL epoch's theta slots an exact
gradient, leaving earlier epochs' slots on FD. No granularity-pinning and
no backprop-through-time (master plan §10) — those stay Deferred-2's.
The payoff target is ACCURACY (the FD mixed-scale defect), not speed;
H0(iv) quantifies it and the go/no-go below gates on it.

## The mathematical structure (as corrected by review; VERIFIED items cited)

The shipped final read is `final_jp[c] = r_v · sojourn(v; theta_final,
ipv_in) · handoff_mass` (verified: `graph_builder_ffi.cpp:2132`, docstring
`__init__.py:10010-10015`). Corrections to v1's derivation:

1. **`r_v` is theta-dependent** [math-review CRITICAL 1]. It is read
   per-vertex from the live graph AFTER `ptd_graph_update_weights(gs,
   theta_final)` — the sum of the t-vertex's parameterized exit-edge
   weights (`graph_builder_ffi.cpp:2095, 2119-2132`; typically the
   mutation slot). So the exact block is the PRODUCT RULE:
   `d(final_jp)/dθ = [d(r_v)/dθ]·soj·mass + r_v·[d(soj)/dθ]·mass`.
   The shipped adjoint provides only `d(soj)/dθ`; for `weight_mode=
   'linear'` the missing term is `d(r_v)/dθ_j = Σ exit-edge
   coefficients[j]`, computable in Python from the clone — still no new C.
   H0(iii)'s oracle must be constructed so the fixture does NOT mask this
   term: `epoch_tied_free_pair` pins the mutation slot
   (`inference/test_lrt_at.py:145`, `fixed=[(1, mu)]`) — exactly where the
   term lives — so H0 runs an additional unpinned-mutation variant.
2. **The handoff IPV exists nowhere outside the fused FFI handler**
   [math-review CRITICAL 2]. The whole chain (intermediate epochs AND
   final read) runs inside one fused C++ call
   (`graph_builder_ffi.cpp:2043-2134`; `ipv_work` is a local); the Python
   custom_vjp sites see only `theta → final_jp`. EVERY variant needs the
   handoff produced somewhere: either (i) an additive FFI output
   extension (new native surface — in the same handler family as §16b
   item 8's silent-NaN swallow, so its loud-failure story must be stated),
   or (ii) Python replication of the intermediate-epoch loop
   (`EpochContext.update_ipv`/`cumulative_probs`,
   `__init__.py:10852-10900`; `stop_probability` `:2543`), which must
   reproduce the handler's aux-folding gather
   (`graph_builder_ffi.cpp:2068-2077`) and granularity EXACTLY, and whose
   cost (duplicated intermediate uniformization per backward call, per
   particle, per unique exposure) is modelled, not assumed. H2 decides
   (i) vs (ii) with numbers; the H0 pure-JAX oracle produces reference
   handoff values to validate whichever is picked.
3. **Primary variant: final-epoch slots exact, earlier slots keep
   FULL-CHAIN FD** [math-review MAJOR 3]. v1's "the existing bulk-FD
   keeps computing d(chain-up-to-handoff)/d(theta)" was FALSE — both
   shipped backwards (`_autodiff_bwd` `__init__.py:4676-4690`,
   `_per_obs_bwd` `:4907-4930`) central-difference the FULL chain
   including the final read; no chain-up-to-handoff forward exists. In
   the flat `(n_epochs·P)` slot convention (verified `:4589-4591,
   :4660`), final-epoch slots do not influence earlier epochs, so the
   split is well-defined: those slots leave the FD perturbation set and
   get the amendment-1 exact block (at the amendment-2 handoff); earlier
   slots' FD is unchanged. Under this variant **J_ipv is not needed at
   all**. The ipv_bar-composition variant (earlier slots FD only up to
   the handoff + exact chaining through it) is CONTINGENT: it needs
   amendment-2 machinery PLUS an FD-of-the-truncated-chain forward that
   doesn't exist — priced in H2 only if H0(iv) shows the primary variant
   leaves substantial final-read error in earlier slots.
4. **Tied theta composes cleanly** (verified, math-review): `_apply_tying`
   runs BEFORE the custom_vjp boundary (`:4701-4702, :4941-4942`); the
   scatter VJP sums slave→master automatically. H0(iv) runs BOTH the free
   and tied fixtures; the exact block emits per-slot partials in the flat
   convention.

Review-verified foundations that HOLD (recorded so v3 doesn't re-litigate):
clone + `update_ipv` preserves the symbolic parameterized tape — only the
concrete trace is freed (`phasic.c:5919-5928`), tape inputs are re-read
per call (`:11412`) with zero tangent for constant IPV edges — so the
shipped adjoint at a fresh handoff needs NO O(n³) re-elimination;
`joint_sojourn_graph` output is continuous (`was_dph` never set,
`__init__.py:10135-10137`) so the adjoint's `was_dph` exclusion does not
bite, and the sojourn graph is non-trapping (`:10005-10006,
:10095-10106`), shrinking the infinite-sojourn decline class; the
final-read index set (`sojourn_t_indices`, `:4617-4620, :4817-4820`) is
construction-known; SVGD reaches the two `_daisy_chain_svgd_model`
custom_vjp sites, not `daisy_chain_joint_probs`'s wrapper (`:4669,
:4900`; built at `:5157, :6072`).

## Scope restrictions (explicit, no silent fallbacks) [math-review MAJOR 7]

- **`weight_mode='linear'` ONLY.** Formula mode deliberately propagates
  into the sojourn graph for C-side evaluation (`__init__.py:10141`), and
  the C adjoint's contraction is linear-only (`phasic.c:11448`) with NO
  C-side weight-mode guard — the exclusion lives in Python wiring (the
  joint-index precedent: `__init__.py:7941-7947`). Batch H's wiring must
  replicate it loudly. Whether `Graph.svgd`'s epoch branch already rejects
  log/formula upstream is CHECKED in H2, not assumed.
- Exposure branch in scope, with its cost multiplier stated [math-review
  MINOR 12]: theta pre-scaling is outside the FFI (verified `:4870,
  :4886-4888`) and `scale_per_unique` changes BOTH the final-epoch slice
  and each handoff, so the exact path costs K_unique × P(articles)
  clone-updates + handoff extractions per iteration. H2 records this in
  the cost model.
- Dispatch/decline semantics: **Batch F's probe-and-commit does NOT
  transfer as-is** [math-review MAJOR 5]. The decline-relevant state here
  is (theta_final, handoff ipv): the MPFR gate (`phasic.c:11432`, reading
  `PHASIC_CONDITION_THRESHOLD` per call) sees the IPV constant-edges,
  and the handoff CHANGES EVERY SVGD ITERATION by construction — a
  construction-time probe proves nothing about later handoffs, and
  committed-raise semantics would introduce a mid-optimization
  RuntimeError class Batch F never had. Additional degenerate mode:
  handoff_mass → 0 (mass fully absorbed before the final epoch at extreme
  particles). The failure-mode design (raise vs per-call fallback vs
  hybrid) is an EXPLICIT v3 design question, informed by H2's measured
  decline rate along a real SVGD trajectory, and is expected to go to the
  user with numbers (the joint-index raise decision was made for a
  static-commit regime; this is a different regime).
  Related: **CC-2 pinning AFFECTS this path** (v1 said the opposite)
  [math-review MINOR 10] — the same `ptd_dbg_tape_needs_mpfr` gate serves
  both; a Deferred-4 pin changes when this path declines. Pin note
  recorded for CC-2.
  **[SUPERSEDED by decision 1, dated 2026-08-13 — process-review MAJOR
  5: with the gate opt-out, the `exact_final_grad` path no longer reads
  `PHASIC_CONDITION_THRESHOLD` at all, so Deferred-4 pinning does NOT
  affect it; the pin note now applies only to the default (gated)
  sojourn/joint-index path. The H0 gate-decline evidence is routed to
  Def-4/CC-2 via a dated note at G5 (Def-4's scope currently excludes
  the sojourn slice — flagged to its owner).]**

## De-risk phase (branch-only, `experiments/dr_batchH_*.py`)

- **H0 — pure-JAX one-hop oracle** (`dr_batchH_oracle.py`). Fixture:
  `tests/pytest/inference/test_lrt_at.py:112-153` (`epoch_tied_free_pair`,
  continuous joint-prob, `epoch_starts=[0, 0.5]`) in free, tied, AND
  unpinned-mutation variants (amendment 1). Implement the full chain
  densely in JAX (which yields the handoff by construction — the
  reference for H2). Verify: (i) value parity vs production
  `final_read='sojourn'` (target ≤1e-10; a NaN row from production is a
  SWALLOWED C failure, §16b item 8 — debug, don't average) ;
  (ii) LINEARITY of final_jp in the raw handoff ipv (exact, construction
  check — FIRST check, load-bearing); (iii) `jax.jacobian` w.r.t.
  (theta_final, ipv_in) as the oracle for the exact block INCLUDING the
  r_v product-rule term; (iv) the composed chain gradient (primary
  variant: exact final slots + full-chain-FD earlier slots) vs full-FD
  and vs the full-JAX oracle — quantifying how much of FD's error the
  final epoch owned, on a mixed-scale theta.
- **H1 — cost/instrument study** (`dr_batchH_cost.py`).
  (a) MANDATORY regardless of variant [§16b item 3, assigned to Batch H's
  design by master plan — process-review CRITICAL 1]: profile the
  offset-tape conversion (`ptd_pcg_convert_to_offset`, O(commands) fresh
  per call, `phasic.c:11337`; `L > 5e7` guard `:11363`) as a SHARE of one
  exact-block call on the H0 fixture and one larger synthetic graph;
  record an evaluate/decline decision on caching, with evidence,
  referencing §16b item 3. Note: at production scale the size guard
  becomes a decline — feeds the failure-mode design above.
  (b) CONTINGENT (only if the ipv_bar variant stays live after H0(iv)):
  J_ipv instruments. v1's options were both wrong in detail [math-review
  MAJOR 4]: the batched sojourn FFI has NO IPV input
  (`ffi_wrappers.py:936-937` — IPV baked into structure JSON), so
  basis-vector reads need serial live-clone `update_ipv` (O(L) concrete
  replay each, and n_ipv = O(n) interior vertices, `__init__.py:
  10115-10127` — NOT "tens-to-hundreds" [math-review MAJOR 6]); the
  correct instrument is the TRANSPOSED solve via the EXISTING
  `ptd_expected_waiting_time(graph, rewards)` (`phasic.c:9980`,
  `api/c/phasic.h:423`) seeded with r⊙g and gathered at the ipv target
  indices — no new C. Measure it only if needed.
- **H2 — wiring-point study** (`dr_batchH_wiring.py` + findings doc).
  (a) Handoff extraction decision (amendment 2): FFI-output extension vs
  Python replication; validate the chosen route against H0's reference
  handoffs (exact match modulo fp); state the loud-failure story if the
  FFI route is picked. (b) Composition point inside the two custom_vjp
  sites — modification of the shipped custom_vjp INTERNALS behind the
  unchanged external VJP shape is authorized by master plan §10 (signed
  off 2026-08-11); default-off byte-identity is the guard
  [process-review 10]. (c) Exposure-branch cost model (K_unique
  multiplier). (d) Decline-rate measurement: run a short SVGD trajectory
  on the fixture, count would-be declines (MPFR gate at per-iteration
  handoffs; handoff_mass→0 events) — the empirical input to the v3
  failure-mode design. (e) Check whether the epoch branch already rejects
  non-linear weight modes upstream.

## Go / no-go (process-review MAJOR 4 — decision rules, not vibes)

- **GO** to v3 implementation planning requires ALL of: H0(i) ≤1e-10;
  H0(ii) linearity holds to fp round-off; H0(iii) exact blocks (incl.
  r_v term) match `jax.jacobian` ≤1e-10; H0(iv) composed gradient reduces
  final-epoch-slot error vs full-FD by ≥10× on the mixed-scale variant
  with NO accuracy regression on earlier slots; H1(a) offset-conversion
  decision recorded; H2(a) handoff route validated against oracle.
- **NO-GO / park** (report + tracker `parked`, feeds Deferred-2's §1
  value test): H0(ii) linearity FAILS (cost model collapses — present to
  user); or H0(iv) improvement is negligible (<2× on final slots — the
  exactness doesn't buy enough); or handoff extraction is infeasible
  without non-additive changes (present to user).
- **"Prohibitive" for any instrument** (H1/H2 cost): the exact path may
  not add more than ~0.5× of one full-FD backward per particle-iteration
  on the fixture; anything above goes to the user with the numbers.

## v3 IMPLEMENTATION PLAN (post-de-risk; the plan of record)

**v3.1, 2026-08-13: the mandated v3 adversarial review is DONE (two
refuters, both SOUND-WITH-CORRECTIONS; all findings folded inline below
— see the v3 review record at the end of this file). CLEARED FOR
IMPLEMENTATION.**

Inputs: `b3-batchH-findings.md` (H0-H2 evidence), the two user decisions
above, master §10's authorization to modify the two daisy custom_vjp
INTERNALS behind an unchanged external VJP shape. Branch:
`b3/batchH-final-epoch` cut **from `derisk/batchH-final-epoch`** (a
strict superset of master via merge `cc5a936d` — cutting from master
would orphan the findings doc, de-risk scripts, and this plan's v3,
which live only on the de-risk branch [process-review CRITICAL 1]);
worktree `../phasic-batchH` with its OWN pixi env (REQUIRED — this
batch changes native code); the main checkout's install stays the golden
pre-H reference. G0: re-record the ledger stamp (`eaf86e82`,
1888/0/84/24; master since then is docs/docstring-only — verified by
review: the only source-touching commit is `04775b63`, docstring text
only — state the delta explicitly at branch time).

### I1 — C/C++/pybind: the additive conditioning-gate opt-out

- `ptd_sojourn_grad_theta_subset` (src/c/phasic.c) becomes a THIN WRAPPER
  over a new static core taking `int skip_condition_gate`; the wrapper
  passes 0 — output byte-identical (micro-gate below). **Justification
  chain, stated precisely [process-review MAJOR 4]: the opt-out itself
  is user-sanctioned (decision 1, master `04775b63`); the wrapper+core
  TECHNIQUE is Batch-0 precedent, byte-identity-gated — decision 1 does
  not itself mention wrapper+core, and its "joint-index untouched"
  wording means behavior, not bytes: the restructured shared function,
  the new public C symbol, and the C++/pybind default-arg additions are
  recorded as an explicit surface-change statement in G5's merge
  review.** New PUBLIC additive entry
  `ptd_sojourn_grad_theta_subset_nogate(graph, indices, k, J_out)`
  (the REAL four-arg signature, `api/c/phasic.h:514-515` [C-review
  MINOR 5]) = core with 1: identical in every respect EXCEPT the
  `ptd_dbg_tape_needs_mpfr` conditioning check (`phasic.c:11432`, one
  self-contained line — split feasibility verified by review) is
  skipped. Everything else stays: was_dph exclusion, `L > 5e7` size
  guard, allocation NULL-checks, the final per-requested-row isfinite
  sweep (`:11501` — verified by review to run INDEPENDENT of the gate:
  the LIVE defense once the gate is off; non-finite rows still decline).
  The new entry is DECLARED in `api/c/phasic.h` (additive — required:
  `api/cpp/phasiccpp.h:610` calls the C symbol directly).
- **§16b item 2 closure [process-review MINOR 9]:** the known-wrong
  MPFR-rationale comment sits inside the function I1 rewrites; the new
  core CORRECTS it (citing the H0 evidence), closing item 2 at G5 —
  recorded as an in-scope deviation from Batch F's "no C edits" decline
  (that reason no longer applies: I1 edits this C under user sanction).
- C++ `phasic::Graph::sojourn_grad_theta_subset` gains a default-valued
  `skip_condition_gate = false` argument (source-compatible); pybind
  `_sojourn_grad_theta_subset` mirrors it (`py::arg("skip_condition_gate")
  = false`, edited in the SAME change as the C++ method) — existing
  callers (Batch F wiring, its tests; the only call sites, verified:
  `phasiccpp.h:606-610`, `phasic_pybind.cpp:1934`, `__init__.py:8029/
  :8066`) see identical BEHAVIOR (the signature gains the defaulted arg
  — stated in G5's surface-change note).
- Micro-gates (`experiments/dr_batchH_i1_gate.py`): (a) default path
  byte-identity pre/post wrapper-ization on the joint-index fixtures AND
  the H0 sg fixture (flag=0 == old build, exact array equality);
  (b) flag=1 at H0's documented declining point (realistic theta + real
  handoff) COMPUTES and matches the H0 oracle Jacobian ≤1e-10;
  (c) flag=1 at a genuinely non-finite row (trap fixture if
  constructible, else skip-with-note) still declines via the isfinite
  sweep — proving the gate-skip did not disable the real defense;
  (d) NEW [C-review MINOR 6]: a subnormal-mass probe — the exact-block
  path at a handoff with tiny NONZERO mass (e.g. scaled to ~1e-300):
  measure whether the C computes finite values or declines; the
  measured behavior decides whether I2's zero-branch stays exact-0.0
  only or widens to a documented threshold, recorded in the findings
  doc. **Red micro-gate action [process-review MINOR 10]: a failed
  byte-identity gate = red gate under the standing ruling — STOP, no
  Python wiring; the named fallback is the strictly-additive
  duplicate-function shape (no wrapper-ization), taken back to the
  user before proceeding.**

### I2 — Python wiring in `_daisy_chain_svgd_model`

- New INTERNAL kwarg `exact_final_grad: bool = False` on
  `_daisy_chain_svgd_model` only. Public `Graph.svgd` plumbing stays
  Batch G leaf-1's job (master §9) — G is next after H and wires it with
  the R29-style cannot-be-silently-inert rule. Tests reach the kwarg via
  `_daisy_chain_svgd_model` directly (the SVGD-reachable site, verified
  in H2(b)).
- Construction-time (only when `exact_final_grad=True`):
  - LOUD scope guards (no silent fallbacks): raise `ValueError` unless
    `weight_mode == 'linear'` (H2(e): formula reaches this path today),
    `final_read == 'sojourn'`, continuous graph (`is_discrete`/`was_dph`
    excluded upstream already — re-assert), no `exposure_param_index`
    restriction (exposure IS in scope). `n_epochs == 1` IS supported
    (handoff := initial ipv; empty FD perturbation set).
  - Build the PRIVATE clones: a jsp clone for handoff extraction and an
    sg clone for the exact block (joint-index `_jix_exact_graph`
    precedent — never mutate the shared model graphs from a callback).
    Precompute from `sg.serialize(theta_dim=P)`: exit-edge structure
    (`r_const`, `r_coeff` per t-index) and `sojourn_t_indices` /
    `sojourn_jsp_gather` (already computed by the builder — reuse the
    locals, don't re-derive).
- Backward composition (both custom_vjp sites; external VJP shape
  unchanged):
  - Handoff extraction = the VALIDATED Python-replication route (H0
    tier-1 2.2e-16): private jsp clone → per intermediate epoch
    `update_ipv` / `update_weights` / `stop_probability(dt,
    granularity=<model's value>)` / aux-collapse / gather. A NaN result
    raises Python-side (loud by construction; §16b item 8 confound noted
    in the message).
  - The exact block runs in a HOST CALLBACK via `jax.pure_callback`
    [C-review MINOR 4 — plumbing named explicitly]: result declared as
    `ShapeDtypeStruct((n_t, P), float64)` per particle
    (`(n_unique, n_t, P)` on the exposure branch — all
    construction-known shapes), `vmap_method='sequential'` so
    `vmap(jit(grad))` over particles unstacks it (the H1 per-particle
    cost model's premise). The callback does per-call ATTRIBUTE LOOKUP
    of `_sojourn_grad_theta_subset` on the clone (not a captured bound
    method) — this is the monkeypatch seam I3 test 6 relies on
    [C-review MINOR 7].
  - Exact block at the extracted handoff (INSIDE the callback, on
    concrete values — `mass` is traced upstream, so the branch cannot
    be Python-level in traced code [C-review MINOR 4]):
    `mass = alpha.sum()`; if `mass == 0.0` return the zero block
    WITHOUT calling C. **Corrected rationale [C-review MAJOR 1]: zeros
    are the correct LINEAR-LIMIT Jacobian; production's FORWARD
    NaN-fills at a zero handoff (0-mass normalization is 0/0 — verified
    empirically by review), a pre-existing behavior H does not change,
    so the particle's loss is already NaN there; this branch exists to
    return the limit Jacobian without tripping the C decline→raise
    path (the C adjoint DECLINES at a zero IPV — verified).** Whether
    the branch widens to a tiny-mass threshold is decided by micro-gate
    (d)'s subnormal probe. Else: sg-clone `update_ipv(alpha)` +
    `update_weights(theta_final)` + forward subset sojourn +
    `_sojourn_grad_theta_subset(idx, skip_condition_gate=True)` +
    the r_v PRODUCT-RULE term:
    `J[k,:] = r_coeff[k,:]*soj[k]*mass + r[k]*J_soj[k,:]*mass`
    (line-identical to the validated oracle block, confirmed by
    review).
  - theta_bar: final-epoch slots from `cotangent @ J`; earlier slots
    from the EXISTING FD loop with final-epoch slots removed from the
    perturbation set. **Slot-precedence rules, explicit [C-review
    MAJOR 3]: `fixed ∩ final-epoch → 0.0` (fixed wins — mirroring the
    joint-index re-mask at `__init__.py:8391-8396`; NB the flagship
    fixture's pinned mutation slot IS such a slot under the default
    `bake_fd_skip=True`); `tied-slave ∩ final-epoch → exact value` (the
    `_apply_tying` scatter-VJP needs it to sum slave→master, per the H0
    tied check); under `bake_fd_skip=False` (`epoch_model`) the fixed
    set is empty and EVERY final slot gets the exact value — correct,
    since exact values are correct gradients.**
    `exact_final_grad=False` path byte-identical.
  - A residual C decline RAISES a diagnostic `RuntimeError` listing the
    ACTUAL residual causes [C-review MINOR 6]: allocation failure, the
    `L > 5e7` size guard, non-finite Jacobian rows, AND the
    mmap-loaded Stage-A2 descriptor with NULL input_specs
    (`phasic.c:11324-11336` — env-dependent via
    `PHASIC_REWARD_COMPUTE_CACHE=1`, reachable in the field) /
    out-of-scope tape inputs / precompute failure (Batch-F message
    discipline; F0's jit-raise legibility finding transfers — same
    pure_callback-inside-custom_vjp shape, cite
    `dr_batchF_jit_raise_derisk.py`, no new experiment).
  - Exposure branch: per unique alpha value, the theta batch is
    pre-scaled OUTSIDE the FFI (H2(c)); the exact block runs per unique
    alpha at the correspondingly-scaled theta_final and per-unique
    handoff. **Chain rule, slot-specific [C-review MAJOR 2]: multiply
    the block's columns element-wise by
    `scale_per_unique[u, final_slice]` — which is 1.0 everywhere
    EXCEPT the `exposure_param_index` column, where it is α_u
    (`__init__.py:4757-4761`, `:4840-4846`). A blanket all-columns
    scaling would corrupt the P−1 non-exposure final slots by a factor
    α_u (~1e3 on realistic exposures).** K_unique × cost, ratio
    K-invariant (H1).

### I3 — tests (`tests/pytest/inference/test_exact_grad_daisy_final.py`)

1. Composed-gradient oracle parity (productionized H0(iv)): benign +
   mixed-scale theta vs a dense-JAX oracle fixture, final-slot cols
   ≤1e-9 abs-rel vs oracle, improvement factor ≥1e3 asserted (loose
   floor under the measured 3.6e5).
2. `exact_final_grad=False` golden byte-identity vs the pre-H master
   install (worktree pattern; FD is deterministic).
3. Tied + free fixtures end-to-end (`epoch_tied_free_pair` shape):
   gradient through `_apply_tying` matches oracle column-sums.
4. Exposure branch parity (duplicated exposure values → K_unique <
   n_obs) vs per-obs oracle.
5. Construction-time loud declines: formula-mode graph, `final_read=
   'stopprob'`, each raises with the documented message.
6. mass→0 branch [respecified per C-review MAJOR 1]: a fixture whose
   intermediate epoch absorbs all mass. The FORWARD is NaN there
   (pre-existing production behavior — asserted explicitly so the
   confound is pinned, not hidden); therefore the test drives the
   custom_vjp BACKWARD directly with a synthetic FINITE cotangent and
   asserts: final-epoch block = exact 0.0, NO C adjoint call (spy via
   the per-call attribute-lookup seam on the clone), no raise.
7. I1 micro-gates as pytest: default-path byte-identity; flag=1
   computes at the declining point; joint-index suite unaffected.
8. SVGD-shape composition: `vmap(jit(grad(loss)))` over particles runs
   the exact path and a forced residual decline raises legibly.
   **Spy mechanics [C-review MINOR 7]: the FD-side spy patches
   `phasic.ffi_wrappers.compute_daisy_chain_sojourn_ffi` BEFORE calling
   `_daisy_chain_svgd_model` (the builder from-imports it at
   construction — patching after intercepts nothing, silently); the
   assertion is an EXACT expected call count (2 per remaining FD slot
   per backward + the forward), not an upper bound (which would pass
   vacuously on a dead spy).**
9. NaN-handoff raise [process-review MINOR 11]: monkeypatch the private
   jsp clone's `stop_probability` to return a NaN vector → the backward
   raises loudly with the §16b-item-8 confound message (no silent NaN
   gradient).
- **Existing-test fate table: NO existing test changes state.** Default
  off ⇒ byte-identical behavior; the daisy/epoch files
  (`test_epoch_sojourn_finalread.py`, `inference/test_lrt_at.py`,
  `inference/test_epoch_model.py`, `test_gate_daisy_chain_joint_probs.py`),
  the joint-index files (`inference/test_exact_grad_joint_index.py` — 17
  tests, `test_joint_index_callback.py`,
  `inference/test_optimized_joint_index.py` — shared C function
  [process-review MAJOR 3]) must ALL keep passing unchanged. Any
  deviation = G1 failure.

### Gates

G1 = I3 suite + micro-gates + fate table holds. G2 = the daisy/epoch row
(`test_epoch_sojourn_finalread.py`, `inference/test_lrt_at.py`,
`inference/test_epoch_model.py`) PLUS the joint-index/sojourn row per the
process map [process-review MAJOR 3]: `test_joint_index_callback.py`,
`inference/test_optimized_joint_index.py`,
`test_gate_daisy_chain_joint_probs.py`, plus
`inference/test_exact_grad_joint_index.py` (shared C function; also
proposed as a dated process-doc amendment to the joint-index G2 row —
the map predates it), always-run
`inference/test_fd_gradient_mixed_scale.py`. G3 = chunked full suite vs
ledger @ `eaf86e82` (docstring-only master delta noted). G4 = two diff
refuters (C wrapper fidelity + wiring/tests). G5 = merge review
INCLUDING the explicit surface-change statement against decision 1's
"untouched" phrasing (restructured shared C function; new public C
symbol + header declaration; C++/pybind default-arg) [process-review
MAJOR 4]; **baseline-ledger re-stamp at the merge commit (fourth stamp;
expect 1889 + I3's new tests) [process-review MAJOR 2]**; tracker
(incl. the §16b snapshot row-3 declined-with-evidence edit); master §15
tick + §16b item 3 closure (declined-with-evidence) + item 2 CLOSURE
(comment corrected in I1) + a dated Def-4/CC-2 routing note for the H0
gate-decline evidence (Def-4's scope currently excludes the sojourn
slice — flagged to its owner, not silently absorbed) [process-review
MAJOR 5]; the §15 E/H conflict-matrix cell annotation (re-verified
against I1: C signature unchanged, E's Python consumer unaffected —
recorded per process-review MAJOR 6); CLAUDE.md Batch-H paragraph;
memory update; main-checkout install rebuild.

### v3 risks

1. The C wrapper-ization must be byte-identical — micro-gate (a) is the
   proof; run BEFORE any Python wiring lands. A red byte-identity gate =
   STOP + the named strictly-additive fallback + back to the user
   (see I1).
2. Private-clone thread/reentrancy posture = joint-index precedent (host
   callbacks serialize on the clone); pmap caveat documented, not solved.
   Clone starting-edge ORDER preservation (needed for `update_ipv`
   correspondence) is asserted from the joint-index production use of
   clones but was never directly measured [C-review "not checked"] —
   I3 test 1's oracle parity catches any violation (parity would fail
   grossly), noted here so a failure is diagnosed correctly.
3. The handoff-extraction cost (~half an FFI forward per backward per
   particle) is the add-on's dominant term (H1) — if SVGD wall-time
   regresses more than the predicted net win on a real run, that's a
   finding for G5, not a silent accept.
4. Granularity: the replication MUST read the model's granularity value
   (structure metadata), not a default — H0 used the same source; test 1
   covers it implicitly (parity fails otherwise).
5. Cosmetic anchor drift found by review (fixture at
   `inference/test_lrt_at.py:114-152`, `fixed=[(1, mu)]` at `:147`;
   formula propagation at `__init__.py:10144`) — all constructs
   verified present; earlier sections keep their original cites for the
   record [both reviews, cosmetic].

## v3 adversarial plan-review record (2026-08-13; v3 → v3.1)

C/wiring refuter: SOUND-WITH-CORRECTIONS — MAJOR 1 (mass==0 forward
claim FALSE in production: verified all-NaN forward + C adjoint decline
at zero IPV; branch re-rationalized as linear-limit Jacobian +
raise-avoidance; test 6 respecified with finite synthetic cotangent) →
folded; MAJOR 2 (exposure chain rule is slot-specific, not
all-columns) → folded; MAJOR 3 (fixed ∩ final precedence undefined;
fixed wins per joint-index re-mask; tied-slave gets exact; bake_fd_skip=
False case stated) → folded; MINOR 4 (pure_callback plumbing:
ShapeDtypeStruct shapes, vmap_method='sequential', mass check inside
the host callback) → folded; MINOR 5 (real 4-arg C signature; phasic.h
declaration named) → folded; MINOR 6 (decline-cause list extended incl.
mmap NULL-input_specs; subnormal-mass probe = micro-gate (d)) → folded;
MINOR 7 (both spy seams named; patch-before-construction; exact call
counts) → folded. Verified sound: I1 split feasibility, gate-independent
isfinite sweep, no other C call sites, formula fidelity vs oracle,
final-slot separability, n_epochs==1 reachability, loud NaN-IPV path,
clone precedent, scope-guard necessity, FD-skip semantics, clean golden
baseline, cost-model consistency, anchor audit. Process refuter:
SOUND-WITH-CORRECTIONS — CRITICAL 1 (branch cut point: from the
de-risk branch, NOT master) → header; MAJOR 2 (G5 ledger re-stamp) →
gates; MAJOR 3 (G2 joint-index row + fate table + process-map
amendment) → gates/I3; MAJOR 4 (decision-1 attribution reword + G5
surface-change statement) → I1; MAJOR 5 (CC-2/Def-4 supersession +
evidence routing) → scope §; MAJOR 6 (E/H re-verification recorded) →
gates; MINORs 7-12 (tracker status + snapshot edit; findings-doc dated
appendix; item-2 comment fate = CLOSED in I1; red-gate action; NaN-
handoff test 9; cosmetic cites) → folded. Verified sound: G0 delta
claim (git-verified docstring-only), G1/G3/G4 shape, worktree
commitment, decision-2 fidelity, Batch-G/§9 consistency, no kwarg
collision, Def-2 boundary, §16b item 3/8 handling, naming conventions.

## Gate surfaces (provisional; bound in v3)

- **G1** = the de-risk oracle gates (H0(i)-(iv) + H1(a) + H2(a) recorded
  in `b3-batchH-findings.md`) for the de-risk phase; the I3 suite for
  implementation.
- **G2** (process §4 Daisy/epoch row): `tests/pytest/
  test_epoch_sojourn_finalread.py`, `inference/test_lrt_at.py`,
  `inference/test_epoch_model.py`, + always-run
  `inference/test_fd_gradient_mixed_scale.py`.
- **G3** chunked full suite vs ledger @ `eaf86e82`.
- **G4** two diff refuters. **G5** merge review; tracker/master §15 tick;
  CC-2 pin note lands with Deferred-4.

## Risks / notes (v2)

1. Linearity claim: load-bearing, cheap to verify — H0(ii) FIRST; failure
   → NO-GO branch (user decision).
2. Exposure: pre-scaling outside the FFI (verified) — but the multiplier
   is K_unique × particles, stated in H2's cost model.
3. E/H interface (master plan §10 Dependencies + §15 conflict matrix —
   NOT "§16 risk 6", which is E/F sequencing, de-facto resolved by F's
   merge; a RESOLVED tick for it is proposed as a docs item)
   [both reviews]: primary variant + H1(b)-as-existing-C means no
   collision with Batch E in any surviving option; H1 records the
   decision + evidence.
4. Mid-optimization declines are a NEW failure regime (ipv-dependent) —
   v3's central design question, taken to the user with H2(d) numbers.
5. §16b items honored here: item 3 (offset conversion — H1(a), mandated);
   item 8 (NaN swallow — H0(i) confound note + H2(a) loud-failure story).

## Adversarial plan-review record (2026-08-13; v1 → v2)

Math/design refuter: SOUND-WITH-CORRECTIONS — CRITICAL 1 (r_v
product-rule term missing) → amendment 1; CRITICAL 2 (handoff IPV
machinery unbuilt) → amendment 2; MAJOR 3 (chain-up-to-handoff FD
nonexistent; primary variant re-based) → amendment 3; MAJOR 4 (both J_ipv
instruments wrong; transposed solve = existing `ptd_expected_waiting_time`)
→ H1(b); MAJOR 5 (probe-and-commit non-transfer; ipv changes per
iteration; mass→0) → scope §; MAJOR 6 (n_ipv = O(n)) → H1(b); MAJOR 7
(weight-mode silence) → scope §; MINORs 8-12 (stale baseline; risk-6
mis-cite; CC-2 actually affected; §16b item 3; exposure multiplier) →
folded. Explicitly verified sound: tape survival under `update_ipv`,
not-was_dph, non-trapping, tied-theta composition, fixture fit, D2
boundary. Process refuter: SOUND-WITH-CORRECTIONS — CRITICAL 1 (§16b
item 3 unevaluated) → H1(a); MAJOR 2 (stale baseline/G0) → header; MAJOR
3 (missing G1/G2/fate-table/findings-doc/worktree/v3-review commitments)
→ header + gate surfaces + sketch; MAJOR 4 (no go/no-go) → §above; MINORs
5-10 (risk-6 cite; D.3/G dangling reference + open-decision input; NaN
confound + ipv_bar FFI surface; fixture path + tied variant; tracker
hygiene [fixed in the same docs pass as this v2]; master-§10
authorization cite) → folded. Both reviews' PASSED-check lists retained in
the review transcripts; key positives promoted into "Review-verified
foundations" above.

## Merge review (G5) — 2026-08-13

**All gates green; squash-merged to master.**

- **G0:** implementation branch `b3/batchH-final-epoch` cut from
  `derisk/batchH-final-epoch` at `6acbc455` (the de-risk corpus is a
  strict superset of master via `cc5a936d`); ledger third stamp
  `eaf86e82` (1888/0/84/24; +1 post-stamp test = 1889 expected); master
  delta above the stamp verified docs/docstring-only (`04775b63`).
- **G1:** I3 suite **11 passed** (10 functions / 11 collected — test 1
  parametrized ×2; test 10 = n_epochs==1 added at the G4 fold). I1
  micro-gates: (a) cross-install byte-identity **6/6** (sojourn +
  joint-index-graph cases, goldens from the pre-H install), (a2) daisy
  False-path gradient cross-install golden **bitwise identical**,
  (b) nogate-at-declining-point vs dense-JAX oracle 1.5e-16, (c) trap
  declines with the gate skipped, (d) subnormal mass declines (branch
  stays exact-0.0-only).
- **G2:** **58 passed, 1 xfailed** across the daisy/epoch row + the
  joint-index/sojourn process-map row + always-run mixed-scale file —
  independently re-run in full by the G4 C/wiring refuter (same tally).
- **G3:** chunked full suite (31 groups, foreground/background pattern):
  **1899 / 0 / 84 / 24** = 1889 baseline + 10 new tests, skips/xfails
  identical to the ledger. (Run pre-G4-fold; the fold added test 10 →
  next full run expects 1900.)
- **G4:** two diff refuters, both SOUND-WITH-CORRECTIONS, **zero
  correctness defects in shipped code**. The C/wiring refuter textually
  diffed the extracted core vs the old body (identical except the three
  planned changes), verified CRLF integrity, and ran independent
  numeric probes on the two regimes the suite lacks (n_epochs=3 slot
  arithmetic incl. bitwise-FD middle epochs; exposure under
  `vmap(jit(grad))`) — both clean. All corrections folded (`43567b50`):
  dead-spy guard, tied-test tolerances anchored to actuals + explicit
  precondition, 'precompute failure' in the raise, index-space assert +
  constant_edges in the block helper, BOTH narrowed plan gates RESTORED
  (jix byte-identity cases; daisy False-path golden), test 10, findings
  rewording (micro-gate (c) proves the decline, not the mechanism).

**Surface-change statement (owed against decision 1's "joint-index
untouched" phrasing):** the shared C function's BODY was restructured
into wrapper+core (behavior byte-identical, gated by micro-gates (a)/(a2)
against fresh pre-H goldens); ONE new public C symbol
(`ptd_sojourn_grad_theta_subset_nogate`) + its `api/c/phasic.h`
declaration; the C++ method and pybind binding gained a default-valued
`skip_condition_gate=false` argument (source-compatible; all existing
callers unchanged, verified: the joint-index call sites are
content-identical, line-shifted only). "Untouched" holds for BEHAVIOR at
every layer.

**Deviations / notes recorded:**
1. Plan I3 test-7 mapping: sub-item (i) = in-suite flag-inertness at a
   benign point (weaker than cross-install identity, which lives in
   micro-gate (a)); (ii) declining-point semantics in-suite; (iii)
   "joint-index suite unaffected" delegated to G2 (re-run
   independently). The two G4-flagged gate narrowings were RESTORED
   pre-merge, so no standing deviation on them.
2. Suite gaps (recorded, not owed here): no pytest trap/deficit-sink
   fixture (the CLAUDE.md-flagged class; narrowed by micro-gate (c),
   still open); no exposure+tied combination test; the was_dph
   construction guard is shielded by the stronger upstream is_discrete
   rejection (record-only).
3. §16b item 2 CLOSED early (the corrected MPFR-rationale comment lands
   inside the function I1 rewrites — in-scope deviation from Batch F's
   "no C edits" decline, whose reason no longer applies). §16b item 3
   CLOSED (offset-conversion caching declined with H1(a) evidence:
   the adjoint call containing the conversion is 1.0-1.3% of the FD
   backward across a 37× size range).
4. Def-4/CC-2 routing: the H0 gate-decline evidence (100% decline at
   realistic coalescent scales; lifted answers accurate to ~1e-13; the
   default threshold unreachable below ~1e300 at real handoffs) is an
   INPUT to Deferred-4's threshold-semantics work — Def-4's scope
   currently EXCLUDES the sojourn slice, so this is flagged to its
   owner via the master-plan §16b routing note, not silently absorbed.
   Post-decision-1, Def-4 pinning no longer affects the
   `exact_final_grad` path at all (it skips the gate).
5. E/H interface re-verified against the shipped I1: the C signature is
   unchanged (wrapper keeps it; the new symbol is additive), so Batch
   E's planned consumer is unaffected — §15 conflict-matrix cell
   annotated at this merge.
6. Unblocks: Batch G leaf 1 (public `Graph.svgd` plumbing of
   `exact_final_grad`) and Deferred-2's activation gate ("Batch H
   shipped" is now true; its §1 value test + user authorization remain).
