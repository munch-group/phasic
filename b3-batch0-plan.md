# Batch 0 plan — reverse-tape skeleton extraction (the shared prerequisite)

**Status: v2, 2026-08-13 — adversarial plan review folded in (two refuters,
both SOUND-WITH-CORRECTIONS; the v1 `binp_out` pre-design was refuted as
CRITICAL/unusable and is REMOVED). Cleared for implementation.**
Design authority: master plan §2 (refactor of shipped C explicitly
authorized there, gated by value-identical gate outputs). Branch:
`b3/batch0-skeleton`, worktree `../phasic-batch0` + isolated pixi env.
Baseline: `b3-test-baseline.md` @ `164e2758` (1885/0/84/24; the
`164e2758..315a53b4` delta is docs-only — verified — so the ledger is
fresh; re-confirm at implementation start per G0).

## Scope

Extract the code-identical stage-0/1/2 core shared by
`ptd_moments_grad_theta` (linear, `src/c/phasic.c:10787-10930`),
`ptd_moments_grad_theta_log` (`:10966-11112`), and
`ptd_moments_grad_theta_dph` (`:11191-11387`) into one static helper;
convert all three into thin wrappers. Zero user-facing change; zero
numerical change. NOT touched: `ptd_sojourn_grad_theta_subset`, the
validators (incl. D.2's guarded `ptd_moment0_grad_theta`),
`ptd_dbg_tape_needs_mpfr`.

**Review-verified premise:** the stage-0/1/2 regions are CODE-identical
linear-vs-log (comments differ only — extract from the comment-richest
linear copy) and identical-up-to-ownership for dph (the exact divergence
list below). "Byte-for-byte" was v1 overstatement; the M-step diffs are
pure moves *modulo the listed divergences*, each of which is explicit.

## Design (v2 — corrected per both reviews)

**Policy (uniform, replaces v1's speculative `binp_out`):** the core is
`static` with all consumers in this file. Each consumer batch extends the
core's signature WHEN IT LANDS — cheap by design, and cheaper than
maintaining speculative parameters that reviews showed were unusable
(`ni`/`input_specs` don't exist caller-side). Concretely sanctioned now:
Batch A will add a `(rewards, rewards_len)` parameter (its 2-line hook
edits the seed line `:10844`-pattern and reverse-seed line `:10876`-pattern
— kept comment-marked so A's diff stays ~2 lines + threading); Batch B will
add a `PTD_B3_FORMULA` enum case whose per-input `dw/dθ` precompute runs
CORE-INTERNALLY as a kind-gated pre-outk stage (the k-ordering only exists
after tape conversion — a caller-side ctx cannot carry it; formula
feasibility doc agrees: "once per tape-input edge, before the per-outk
loop"); Batch C will add its pre-contraction exit THEN (reviewer-supplied
options recorded: core-allocated out-params + spec exposure, or
wrapper-side contraction — decided in C's plan).

**Structure:**

```c
enum ptd_b3_contract { PTD_B3_LINEAR, PTD_B3_LOG, PTD_B3_DPH };

struct ptd_b3_dph_ctx { const double *Sv; const double *SigmaCv; };

/* Wrappers build ptape+off and own their destruction; the core consumes a
 * built tape. (Review option (i): ~8 duplicated lines per wrapper; makes
 * ni/input_specs wrapper-visible for future consumers, and preserves
 * dph's mixed-edge decline BEFORE the expensive tape build, exactly as
 * today at :11227 vs :11230.) */
static int ptd_b3_moments_core(
    struct ptd_graph *graph,
    const struct ptd_desc_reward_compute_parameterized_off *off,
    int nr_moments,
    const double *theta, size_t theta_len,   /* NULL/0 for linear */
    enum ptd_b3_contract kind,
    const struct ptd_b3_dph_ctx *dph_ctx,    /* non-NULL iff DPH */
    double *J_out);
```

**Wrapper responsibilities (the definitive divergence list — review-derived,
mechanically diffed, exhaustive):**
- *All three:* param/nr_moments prechecks; build `ptape` (dyn-ordering
  dispatch) + `off`; call core; destroy `off` + `ptape` on every path.
- *linear:* + a NEW early `graph->was_dph` DECLINE (return -1 — NOT
  assert; master §2 item 4's opportunistic close, placed in the WRAPPER so
  it declines before paying the O(n³) tape build). Deliberate behavior
  addition; own commit (M4); own micro-gate (below). Today's safety is
  Python-routing-only (review-verified: no C/C++/pybind guard; only
  `_effective_discrete` dispatch).
- *log:* + `was_dph` decline (`:10970`) and `theta_len != P` check
  (`:10972`). **No positivity check — v1's "positivity precondition" was a
  phantom (review finding): `update_weights(theta, log=True)` enforces
  positivity upstream by construction; adding one would be a behavior
  change.** The log contraction reads `e->weight` LIVE from the edge
  (`:11099`), not the `inv[k]` snapshot — the move keeps the live read
  verbatim.
- *dph:* + PRE-pass (Sv/SigmaCv precompute + mixed-edge decline, before
  tape build) and POST-pass = `ptd_dph_correct_discrete_moment_grad`
  **+ the SECOND isfinite sweep of J_out after the correction + ok-flip**
  (`:11375-11377` — three lines, not one; v1 missed the re-sweep and no
  gate fixture would have caught its omission). **Ownership rule:** the
  wrapper allocates Sv/SigmaCv and frees them UNCONDITIONALLY after the
  core returns (success or decline); the core NEVER frees ctx. This
  replaces dph's four in-body free sites (`:11234`, `:11238`, `:11272`
  inside the MPFR-decline cleanup, `:11381`) — expected M3 divergences,
  each visible in the diff.
- *Deferred-1 note (master §5 third-consumer):* the reverse-chain seeding
  block — INCLUDING the `target = 0` selection (`:10802`) and the
  factorial seed (`:10865-10867`) — is kept as one comment-marked section.
  Stage-1 seeding (input side) and any stage-2 exit are orthogonal and
  compose: a future cotangent-seed parameter changes only this section,
  not the contraction. Declined for now (different output semantics:
  E[T]-vector VJP vs K moments); G5 adds a dated decline note at master
  plan §5/risk 12.

## Method — M-steps, one commit + full gate re-run each

- **M0:** worktree + build; run the byte-identity reference set TWICE
  (protocol-validity pre-check: M0-vs-M0 must byte-match on
  stdout+exit-code before M0 is trusted as a reference; if it doesn't,
  fall back to parsed-value comparison at ≤1 ULP). Reference set: the 3
  jac-gates + `dr_mpfr_gate_test.py`. Comparison is **stdout + exit code
  only** (stderr may carry JAX/absl noise). `pixi run install-dev` before
  EVERY M-step gate run (copy install).
- **M1:** create the core from the linear body (pure move + parameterized
  contraction); convert linear to a wrapper. Gates vs M0. Commit.
- **M2:** convert log. Gates (all four — cross-check linear untouched).
  Commit.
- **M3:** convert dph (divergence list above). Gates. Commit.
- **M4:** the linear-wrapper `was_dph` decline + micro-gate: direct
  `_moments_grad_theta` binding call on a `discretize()`'d graph asserts
  decline (-1/empty), and on a continuous graph asserts no decline
  (the was_dph-vs-is_discrete latch distinction). Commit.
- Validators build once at the end: the SIX validator-relevant gates
  (`dr_realtape_validator`, `dr_reverse_adjoint_gate`,
  `dr_moment0_theta_gate`, `dr_moments_jac_gate`, `dr_mpfr_gate_test`,
  `dr_d2_moment0_guards_gate`) all pass — enumerates v1's vague "five
  gates" (review finding).

## Gate ladder

- **G0:** re-confirm the ledger references master HEAD at branch time
  (docs-only deltas acceptable, recorded).
- **G1:** the M0-M4 protocol above.
- **G2:** `test_gate_moments_3way.py`,
  `inference/test_jax_integration.py` (ledger-clean subset — process-map
  mandated, v1 omission), `inference/test_exact_grad_discrete.py`,
  `inference/test_exact_grad_log_weight_mode.py`,
  `inference/test_exact_grad_rewards.py`,
  `inference/test_fd_gradient_mixed_scale.py`,
  `inference/test_moments_from_graph_vmap.py`,
  `inference/test_svgd_exact_moment_grad_kwarg.py`.
- **G3:** full suite, chunked (~6-12 short runs), vs the ledger.
- **G4:** multi-agent diff review; the pure-move reviewer's mandate
  EXPLICITLY includes cleanup/ownership on every early-return path across
  the new wrapper/core boundary (leaks/double-frees are gate-invisible —
  review finding; ~14 allocations + ptape/off cross the boundary). Second
  reviewer: interface-vs-consumer-needs + the M4 behavior addition.
- **G5:** merge review; tracker/master-plan/baseline/CLAUDE.md updates
  (mark the "reverse-tape skeleton duplication" follow-up RESOLVED; dated
  Deferred-1 decline note at §5/risk 12).

## Sequencing

- Batch F / Batch H: no file overlap (joint-index wiring / sojourn fn).
- **CC-2 (Deferred-4 sweep): must pin to a fixed commit or wait out
  Batch 0's implementation window** — it exercises
  `ptd_moments_grad_theta` and a sweep straddling M-merges would measure a
  moving target (review finding).
- Batch A lands next on the finished core (Phase 3), extending the
  signature per the policy above.
- This plan file is committed at review sign-off (its freeze point) so the
  batch worktree contains it.

## Risks

1. Silent semantic drift during the move — byte-identity per M-step +
   pure-move diff discipline + the exhaustive divergence list.
2. The M4 decline — smallest possible surface (wrapper early-guard), own
   commit, own micro-gate; reviewers ruled KEEP.
3. Interface lock-in — resolved by the static-core extend-on-demand
   policy; nothing speculative remains.
4. Ownership-transfer bugs (dph ctx frees; ptape/off moves) — the
   explicit ownership rules above + G4's cleanup mandate.

## Adversarial plan-review record (2026-08-13)

Reviewer A (C fidelity): SOUND-WITH-CORRECTIONS — finding 1 CRITICAL
(`binp_out` unusable: ni unknowable pre-call, specs destroyed in-core) →
REMOVED, replaced by wrapper-built-tape design + extend-on-demand policy;
2 (dph free-site divergences) → definitive list + ownership rule; 3 (the
second post-correction isfinite sweep) → wrapper post-pass spec; 4 (phantom
log positivity precheck) → deleted; 5 (was_dph decline placement → wrapper,
return -1, keep) → adopted; 6 (byte-compare protocol: stdout+exit only,
M0-twice validity check, per-step rebuild) → adopted; 7 (G2 omission) →
added; 8 (anchor imprecisions) → fixed; 9 (validator gates enumerated) →
six named. Reviewer B (consumers/process): SOUND-WITH-CORRECTIONS —
findings 1-3 (B ctx infeasible caller-side; C exit needs edge identity; A's
rewards param asymmetry) → all resolved by the uniform policy + core-internal
formula precompute + A sanctioned; 4 (was_dph decline gate-invisible) →
M4 micro-gate; 5 (G2) → added; 6 (Deferred-1 orthogonality + target in the
marked region + G5 dated note) → adopted; 7 (G0 + commit the plan) →
adopted; 8 (G4 cleanup mandate) → adopted; 9 (CC-2 pinning) → adopted.
Anchors: both reviewers verified all cited lines at `315a53b4`; the
stage-0/1/2 code-identity premise was mechanically diffed and CONFIRMED.

## Merge review (G5) — 2026-08-13

**All gates green; squash-merged to master.**

- **G0:** branch base `b4960fe8` == master HEAD; `164e2758..b4960fe8`
  docs-only — ledger fresh (review-verified independently).
- **G1 / M-protocol:** M0 ran every reference gate TWICE with byte-compare
  (all four "PASS + byte-stable" — the protocol-validity pre-check the
  interface reviewer flagged as unevidenced; log artifacts in the session
  scratchpad, `m0-*-r{1,2}.out`); M1-M4 each byte-identical on all four
  gates after `pixi run install-dev`; M5 (CRLF reinstatement) re-verified;
  M6 (review fold-ins) re-verified. Validators build: 6/6 gates PASS.
  M4 micro-gate final form: 3/3 (continuous applicable; native-DPH
  applicable — the was_dph-vs-is_discrete latch cell added per G4
  finding 2; discretize()'d declines).
- **G2:** targeted suites green — exactly the 9 ledgered sources-on
  `test_jax_integration` failures, nothing else (55 passed).
- **G3:** chunked full suite **1885 / 0 / 84 / 24** — exact ledger match.
- **G4:** pure-move reviewer **SOUND** (0 critical/major; re-proved the
  code-identity premise mechanically; every divergence classified as
  intended; cleanup balanced on every path). Interface/process reviewer
  **SOUND-WITH-CORRECTIONS** — both majors folded in M6 (comment-marking
  promises; native-DPH micro-gate cell) plus the const/stale-comment
  minors and this record's evidence carries.

**Deviations / notes:**
1. **M5 line-ending incident:** the M1-M3 line surgeries normalized the
   CRLF file to LF; caught at diff-stat review, repaired wholesale in M5,
   gates re-verified. Process lesson: byte-level (newline-preserving) file
   surgery for CRLF sources.
2. Net diff: ~447 lines on `phasic.c` (−123 net; the three near-identical
   copies now share one core) + the 34-line micro-gate.
3. Batch A is now unblocked (Phase 3); its hook lines are comment-marked
   in the core.
