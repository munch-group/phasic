# Batch H plan — daisy-chain FINAL-epoch exact gradient (de-risk first)

**Status: v2, 2026-08-13 — the mandated adversarial plan review is DONE
(two refuters, both SOUND-WITH-CORRECTIONS; 2 CRITICAL + 5 MAJOR from the
math/design refuter, 1 CRITICAL + 3 MAJOR from the process refuter, all
folded below as amendments). Cleared for the DE-RISK PHASE ONLY: after
H0-H2 the implementation section is re-detailed into a v3 that gets its
OWN adversarial plan review before any implementation code (standing
ruling: every phase's plan is reviewed; the sketch below is by definition
a material change pending).** Branch: `derisk/batchH-final-epoch` (then
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

## Implementation sketch (RE-DETAILED into v3 after H0-H2; v3 gets its own review)

I1: host callback for the final-epoch exact block (clone + `update_ipv`
+ shipped sojourn adjoint + r_v product-rule term; dispatch semantics per
the v3 failure-mode design — NOT assumed probe-and-commit); I2:
composition inside the daisy custom_vjps (static dispatch, no lax.cond;
opt-in kwarg default False — the conservative choice compatible with the
OPEN joint-index default-flip user decision, `b3-batchF-plan.md` merge
review item 1, which is an INPUT to v3 naming/semantics along with master
§6 (D.3) and §9 (leaf 1) [process-review 6]); I3: tests (oracle parity
incl. mixed-scale + tied; exposure branch parity; FD-only byte-identity;
decline cases per the v3 design), with an existing-test fate table due IN
v3 [process-review 3]; gates G0-G5 per process, chunked G3.

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
