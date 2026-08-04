# Loose-ends inventory — repo-root planning/findings docs vs CLAUDE.md

Cross-references every `/Users/kmt/phasic/*.md` planning/findings/handoff document (as of
2026-08-04) against `/Users/kmt/phasic/CLAUDE.md` (the canonical, currently-maintained summary),
to find OPEN loose ends / deferred items / known gaps not currently reflected in CLAUDE.md — with
an explicit flag for plausible intersection with the "B3" initiative (replacing finite-difference
gradients with exact analytic gradients; see CLAUDE.md's "trace-based elimination" and "Disabled
paths / follow-ups" sections).

Built incrementally, batch by batch, per explicit instruction (a prior attempt lost all work by
holding everything until one final write).

**Note on CLAUDE.md version used:** CLAUDE.md was re-read directly from disk mid-way through this
audit (rather than relying on the copy pasted into this conversation's system context) and turned
out to contain a THIRD "Disabled paths / follow-ups" subsection — "B3 joint-index exact sojourn
gradient (`exact_grad`, default `False`) — known gaps, flagged not fixed" — describing
`ptd_sojourn_grad_theta_subset` in detail, which was NOT present in the system-context copy. All
cross-referencing below uses the on-disk file (227 lines as of this audit) as canonical.

**B3 relevance key:** none / possible / likely — erring toward flagging any plausible connection
(numerical-stability, caching, SVGD-divergence, test-coverage gaps all count as "possible" even if
not directly about gradients).

---

## Summary table

| File | One-line summary | Status | B3/exact-FD relevance | Notes |
|---|---|---|---|---|
| audit-differential-findings.md | Differential-audit findings log (F-001..F-011): 11 verified findings from executing cross-path comparisons against independent oracles | **mostly DONE** — verified against live code (see correction note); only Q10/#9 (SCC ordering) remains open | possible (Q10 only; not gradient-related) | F-001 captured in CLAUDE.md. F-007/Q7.1, N1/N2/N3/N4, Q5, Q6a/Q6b have ALL been fixed by later commits (`cc2d76ea`, `98f18e2a`, `be6a6ed5`, `66a5976d`/`bd751e03`, Q5/Q6 gate updates) not reflected in CLAUDE.md — but since they're DONE this is not a gap, just an undocumented changelog. Only Q10 (SCC topological order) remains genuinely open — see detail section |
| audit-situation-map.md | Compute-path atlas + gap list for the differential audit; baseline `3082ebc6`, full route table, 15-item unguarded-gap list, known-defect breadcrumbs | open (atlas itself is a snapshot; many listed gaps still open) | likely | Companion to audit-differential-findings.md; several gap-list items (esp. #4 log-mode, #11 gradient-equivalence-across-paths) directly bear on B3 — see detail section |
| b3-batch0-realtape-findings.md | B3 historical: forward-mode tangent validator confirms real C elimination tape is differentiable (precursor to reverse adjoint) | done (superseded by shipped B3 work) | likely (historical B3) | Fully superseded — table entry only |
| b3-batch1-reverse-adjoint-findings.md | B3 historical: reverse-mode θ-adjoint over real C tape confirmed to machine precision on cyclic fixtures | done (superseded) | likely (historical B3) | Fully superseded — table entry only |
| b3-batch2-exact-grad-findings.md | B3 historical: first production exact first-moment gradient wired into `pmf_and_moments_from_graph` as opt-in (`exact_moment_grad`) | done (superseded — now default True per CLAUDE.md) | likely (historical B3) | Fully superseded — table entry only |
| b3-batch3-higher-moments-findings.md | B3 historical: extends exact gradient from moment[0] to full moment vector (moment-chain reverse) | done (superseded — shipped, now part of default-True path) | likely (historical B3) | Its own "remaining items" list (was_dph, log/formula, joint-index, hierarchical, MPFR) is now captured in CLAUDE.md (was_dph/MPFR since shipped; log/formula/joint-index/hierarchical explicitly listed as still FD-only) |
| b3-batch3-mpfr-and-discrete-derisk.md | B3 historical: MPFR conditioning-safety gate shipped; discrete/was_dph math derived but NOT yet implemented at time of writing | done (MPFR shipped; discrete/was_dph superseded by later B3-DISCRETE-MERGE-REVIEW.md) | likely (historical B3) | Table entry only |
| b3-c-theta-adjoint-plan.md | The original detailed B3 Tier-3 C-reverse-adjoint implementation plan (batches 0-4), adversarially reviewed with 9 amendments | done (superseded — plan executed) | likely (historical B3) | One NOT-superseded item: unwired `ptd_graph_pdf_with_gradient` (phasic.c:11805) landmine — see detail section |
| b3-derisking-strategy.md | Pre-implementation de-risking strategy (DR-A..DR-F experiments) that decided Tier-3 (C θ-adjoint) over Tier-1 (JAX trace-replay) as the production B3 approach | done (superseded — decision made, matches memory note `project_fd_gradient_b3`) | likely (historical B3) | Repeats the `ptd_graph_pdf_with_gradient` landmine; DR-D (scale-aware FD stopgap) status unclear — see detail section |
| B3-DISCRETE-MERGE-REVIEW.md | Merge review/handoff for the discrete/was_dph exact-gradient batch (now merged: matches git log `c0cb9de1`) | done (merged, matches CLAUDE.md's "continuous + discrete/was_dph" B3 scope) | likely (historical B3, but see below) | Two follow-ups NOT in CLAUDE.md: "mixed-vertex decline" scope boundary, and open perf decision (native FFI gradient handler vs pure_callback) — see detail section |
| b3-discrete-theta-adjoint-plan.md | The plan actually executed to produce the discrete/was_dph batch (see B3-DISCRETE-MERGE-REVIEW.md) | done (superseded — plan executed as written) | likely (historical B3) | Table entry only |
| b3-experiment-findings.md | Make-or-break experiments: DR-D (FD scale-aware step) proven a dead end and dropped; DR-A proves exact AD is feasible+cyclic-correct+far better than FD, but flags an inherent float64 "conditioning floor" corrupting the sub-dominant gradient component at extreme mixed scale | done (decision made) but flags an UNRESOLVED characterization item | likely | The "conditioning floor" characterization/pinned-regime recommendation appears NOT to have been completed/documented — see cross-cutting note below |
| b3-joint-index-plan.md | CURRENT, actively-worked B3 doc for `pmf_from_graph_joint_index`'s exact sojourn gradient (`ptd_sojourn_grad_theta_subset`) | open (in progress; per instructions, skimmed lightly) | likely (directly B3) | Its follow-up items ARE now captured in CLAUDE.md's own "B3 joint-index exact sojourn gradient" section (added since — see note); not detailed further here per task scope (parallel research stream covers this file's code directly) |
| b3-log-weight-mode-plan.md | The plan for the shipped log-weight-mode B3 batch (continuous only) | done (superseded — matches CLAUDE.md's "log-weight-mode batch" description) | likely (historical B3, directly cited by CLAUDE.md) | One possibly-unresolved item: whether the planned "quantify MPFR scale-sensitivity under log mode" measurement (D3) was ever actually performed/reported — no findings doc for this batch exists in the repo root to confirm — see detail section |
| B3-MERGE-REVIEW.md | Merge review/handoff for the FIRST (continuous-only) B3 exact-gradient batch | done (merged; superseded by later batches) | likely (historical B3) | Confirms `PHASIC_B3_VALIDATORS` debug scaffolding still exists (compile-guarded OFF by default, verified present in current CMakeLists.txt/phasic.c) and was deliberately "kept, not stripped ... strip in the final B3 cleanup once coverage is complete" — that cleanup does not appear to have happened; native-FFI-vs-pure_callback perf decision also repeated here, still open |
| b3-prototype-findings.md | Cyclic-graph θ-adjoint/trace prototype validating Tier-3 (and Tier-1) against an analytic oracle | done (superseded — Tier-3 shipped) | likely (historical B3) | Repeats the same "conditioning floor" finding as b3-experiment-findings.md — see cross-cutting note |
| README.md | One line: "# Phasic" | n/a | none | No content |
| sojourn-fix.md | Detailed writeup of the `expected_sojourn_time_subset` forward→reverse-adjoint rewrite (`c340bedc`) fixing a 1.5TB OOM on large joint-prob graphs | done (shipped, matches CLAUDE.md/memory `project_sojourn_solve_memory`) | likely (historical, but architecturally the direct precedent B3 extended) | Two self-flagged follow-ups not in CLAUDE.md: (1) `joint_prob_graph` construction itself is a separate `O(n²)` Python bottleneck (`np.append`-in-a-loop) untouched by this fix; (2) the no-arg full `ptd_expected_sojourn_time` is still `O(n²)` and could reuse the same adjoint if ever needed at scale — see detail section |
| stage2-coverage-safety-net-handoff.md | Stage-2 handoff: coverage measurement + 7 equivalence gates + risk-ranked gap report, pre-Stage-3 | done (delivered; superseded by Stage-3 and later fixes) | possible | Nearly all concrete bugs it flagged (Q5/Q6a/Q6b/Q7.1) are now fixed (see category-A-fix-plans.md entry); `svgd.py`'s "no dedicated equivalence gate" and the SCC composer's "0% native coverage, no gate" (#9) items are the two it ranked HIGHEST-risk — see detail section for current status |
| stage3-execution-handoff.md | Stage-3 as-built record: 8 commits (equivalence gates, dead-code purge, first WS-C god-object extractions) | done (delivered; partial — only the "clean" WS-C clusters were finished) | possible | The notebook-checkout data-loss incident (~24 notebooks) is referenced again here; "remaining for the next pass" list (`svgd.py` module split, WS-D/E/F, joint-probability cluster) mostly still open per code inspection — see detail section |
| stage3-refactor-plan.md | The full Stage-3 architectural plan: 33-agent audit, doctrine violations, a 20-step commit sequence, human decision gate | done (decisions recorded; execution only partially carried out — see stage3-execution-handoff.md) | likely | Largest source of NOT-in-CLAUDE.md items: the still-open `defect()`/`cdf()` inconsistency (verified live, native assert still disabled), the SCC composer split-brain (#9/Q10, confirmed still open), the sanctioned-but-fragile FD-gradient carve-out for daisy-chain/reward-visit-prob (already captured by CLAUDE.md for daisy; reward-visit-prob's FD path is NOT explicitly named in CLAUDE.md's B3 section — cross-referenced from audit-situation-map.md above), and an unconfirmed `bffg.py` duplication-risk flag — see detail section |
| svgd-divergence-robustness-plan.md | The actual implementation plan for `deferred-svgd-divergece-fix.md`'s fix A (fail-soft callback) + conditional B.1 (per-step φ trust-region) | **DONE** — matches live code verified under `deferred-svgd-divergece-fix.md` above (`_is_rate_blowup`/`_rate_blowup_penalty` = fix A; `_GRAD_NORM_CLIP_MULT` = fix B.1) | likely (already covered) | No new items beyond what's already noted under `deferred-svgd-divergece-fix.md` / CLAUDE.md's B3 gaps |
| svgd-lrt-fix-a-plan.md | Implementation plan for making canonical `likelihood_ratio_test` accept tied-vs-free (any provably-equivalent) pairs, by delegating to `likelihood_ratio_test_at` | **DONE** — matches memory `project_svgd_lrt_canonical` and live `model_selection.py` (docstring confirms "tied-vs-free epoch pattern... also works") | possible | Repeats the model-selection.ipynb doc-follow-up gap (already flagged under deferred-notebook-triage.md above): "Updating that notebook to the supported pattern... is a doc follow-up, out of Batch 1's core scope" — reinforces that this notebook update was deliberately deferred and its current status is unverified |
| svgd-lrt-model-reuse-plan.md | Implementation plan for fix C (`likelihood_ratio_test_at`) + fix D (SVGD kwarg guard); explicitly rejects option A (move tying out of the model) | **DONE** — matches live code (`likelihood_ratio_test_at` exists) and memory ("A-move rejected") | possible | No new items |
| tmp.md | An informal glossary/decoder for acronyms (FD/AD/VJP/FFI/DPH/CTMC/SVGD/MPFR/ULP/SNR) and colliding stage-label schemes (Phase 1-4, Batch A, an OLDER B0-B4 numbering with its own 3 tiers, F1-F7/P2-1.../G1-G2/N1-N12 finding labels) from an early, now-superseded round of this audit | historical / superseded — the labels it defines (old B0-B4, N1-N12, G1/G2) predate and were later replaced by the cleaner F-00x/N1-N4/Q-labels used in audit-differential-findings.md and the current b3-*.md naming | none directly (meta-documentation) | Confirms `audit-fd-step-remediation-plan.md` and `numerical-refactor-handoff-plan.md` (referenced here) no longer exist in the repo root — consistent with audit-situation-map.md's own note that this plan doc "is absent from the tree; this document supersedes it." No actionable open item beyond what's tracked elsewhere under clearer names |
| tree_toplogy_encoding.md | Short domain-design note: an ancestor/parent-vector canonical encoding for coalescent tree topologies (alternative to `state_indexing.py`'s mixed-radix approach), for two-locus/recombination models | reference note, not a bug/gap tracker; already cross-linked from CLAUDE.md's opening paragraph | none | Not a loose end in the sense this audit targets (no bug, no deferred fix) — a population-genetics domain-modeling idea; not verified whether/where implemented, out of scope for a gradient-focused audit |
| batch2-free-epoch-model-plan.md | Plan for a public `Graph.epoch_model()` / `FreeEpochModel` builder for reusable free daisy-chain epoch models, enabling same-model LRTs | **DONE** — verified live: `src/phasic/epoch_model.py` + `Graph.epoch_model` exist (commit `fc6040aa` "feat(epoch): public Graph.epoch_model ... (Batch 2)") | possible | Not itself gradient work, but touches the daisy-chain FD-gradient model machinery CLAUDE.md flags as still FD-only; no further detail needed (fully shipped) |
| category-A-fix-plans.md | Adversarially-reviewed fix plans for the 3 reachable Category-A bugs: A1 (tied-slave sentinel export = F-010/N3), A2 (FFI reward-transform = F-007/Q7.1), A3 (trace-replay log footgun = F-004) | **DONE, all 3 shipped** — verified live (see detail) | possible (A3 directly; A1/A2 adjacent) | See detail section |
| deferred-notebook-triage.md | Triage of 24 tutorial notebooks pre-Stage-3; found 2 genuine bugs (`state-space` fixed same-pass, `model-selection` open) + 2 environment/robustness issues (`time-inhomogeneous` graphviz, `distributed` kernel SIGABRT) | mixed: `state-space` DONE; `model-selection`'s `epoch_starts` cause is now fixed upstream (epoch_model/LRT work) but notebook itself and the LRT-inversion (convergence) failure not verified updated; graphviz env fix and the `distributed.ipynb` GIL/`dec_ref` crash NOT verified fixed | possible (distributed SIGABRT only, as a crash-hardening pattern) | See detail section |
| deferred-svgd-divergece-fix.md | Diagnosis + fix menu (A: fail-soft compute path, B: SVGD-level clipping/guards, C: bounded transforms, D: diagnostics) for a single-diverged-particle hard-crashing all of SVGD | **DONE (A + B.1/B.2 shipped)** — verified live: `_is_rate_blowup`/`_rate_blowup_penalty` (fail-soft, A) and `_GRAD_NORM_CLIP_MULT`/gradient-norm clipping (B.2) both present in `src/phasic/__init__.py`/`svgd.py`; matches CLAUDE.md's own references to these exact symbols | likely (directly referenced by CLAUDE.md's B3 known-gaps section) | CLAUDE.md documents residual gaps in how these guards interact with the exact-gradient path (fwd/bwd rate-blowup inconsistency, unguarded slow band) — already captured, no new gap found. Option C (bounded/capped rate transform) was NOT found implemented — minor, "principled longer-term fix", not required |
| deferred-svgd-lr-bug.md | Diagnosis of why a tied-vs-free epoch-model nested LRT couldn't be run (3 blockers) + 4 fix options (A: SVGD-level tying + public builder, B: `SVGD.from_fitted`, C: one-model-two-theta LRT, D: ergonomics) | **DONE (effectively C + D + the public builder from batch2-free-epoch-model-plan.md)** — verified live: `likelihood_ratio_test_at` exists in `model_selection.py` (fix C) and its docstring explicitly documents "the tied-vs-free epoch pattern (different callables) also works" for the canonical `likelihood_ratio_test`, and `Graph.epoch_model`/`FreeEpochModel` (fix-A's public builder) ships | possible | Matches memory note `project_svgd_lrt_canonical` ("closed (C+D+A)"); no further detail needed |

*(All 31 files accounted for above; rows are in the order each batch was processed, not
alphabetical.)*

---

## Detail sections

### audit-differential-findings.md

Log of a "differential audit" comparing post-refactor behaviour against baseline `3082ebc6` and
independent oracles. 11 findings (F-001..F-011). Cross-checked against CLAUDE.md:

- **F-001** (discrete-PMF `normalize()` bug in `pmf_from_graph_parameterized`) — **captured in
  CLAUDE.md**, verbatim as "bug F-001" under "`Graph.pmf_from_graph_parameterized` — disabled,
  needs revival" (fix item 3 of the revival checklist). No further action needed here.
- **F-002** (sojourn-subset reverse adjoint, `c340bedc`) — verified CORRECT to machine precision
  against an independent dense oracle. Not a loose end itself, but flags a **gap**: "a permanent
  gate for this belongs in the gap list (§4.7: sojourn has no cross-path equivalence gate)" — this
  gate does not appear to exist yet and is not mentioned in CLAUDE.md. **B3 relevance: possible** —
  this is a *different* reverse-mode adjoint (for `expected_sojourn_time(subset)`, unrelated to the
  θ-gradient B3 work) but is architecturally the same technique (reverse pass over the elimination
  trace) and sits in the same C file family; a coverage gap here is adjacent risk.
- **F-003** (WS-C verbatim-relocation claim) — CLEARED, no bug. Historical, not a loose end.
- **F-004** (trace-vs-FFI gate + `use_log` footgun) — Gate built (`test_gate_trace_ffi_equivalence.py`).
  Found a LATENT footgun: `instantiate_from_trace`/`evaluate_trace_jax` default `use_log=False`,
  so replaying a log-mode graph's trace without `use_log=True` silently gives the LINEAR answer
  (measured divergence up to 3.9e16 relative error). Verified **not reachable** by any current
  production path (evaluate_trace_jax has zero production callers; the `instantiate_from_trace`
  pdf/moments code is commented out/dead). **However it explicitly flags forward risk:** "The
  deferred exact-AD gradient plan (Tier 1) proposes using `evaluate_trace_jax` as the differentiable
  backward. On a LOG-mode graph that path MUST thread use_log or it will silently differentiate the
  LINEAR function." **This is NOT mentioned anywhere in CLAUDE.md.** CLAUDE.md's B3 section documents
  the C θ-adjoint (Tier 3) approach that shipped instead of the Tier 1 JAX-trace-replay approach, and
  the memory note `project_weight_mode_log_semantics` covers a related-but-different silent-log-ignore
  bug (`moments_from_graph`/`joint_index`). If Tier 1 (trace-replay-as-backward) is ever revisited —
  e.g. for `weight_mode='formula'`, which per CLAUDE.md is the next planned B3 variant — this footgun
  would need to be re-surfaced. **B3 relevance: likely.** Also notes a separate lead (auto-granularity
  accuracy) later closed by F-005.
- **F-005** (auto-granularity `max_rate=512` floor) — root-caused, NOT a bug, pre-existing quirk.
  Closed; worth a doc note only. B3 relevance: none (affects PDF discretization accuracy, not
  gradients).
- **F-006** (strict-xfail map intact after refactor) — confirms no regression; explicitly lists
  several xfails as "REAL deferred bugs, not mere engine-capability gaps": **Q7.1** (reward-PMF
  cross-path divergence, detailed in F-007), **Q5** (`Graph.moments(discrete=True)` → missing
  `moments_discrete` pybind binding → AttributeError), **Q6a/Q6b** (`from_serialized` drops
  `constant_edges` / merges duplicate-state vertices vs `GraphBuilder`), **Q10/#9** (SCC topological
  order divergence). None of Q5/Q6a/Q6b/Q10/#9 appear in CLAUDE.md. **B3 relevance for Q5: possible**
  — discrete moments being entirely unreachable via one pybind path is adjacent to B3's
  discrete/was_dph exact-gradient work (CLAUDE.md: "continuous + discrete/was_dph for
  weight_mode='linear'"); worth checking whether B3's discrete gradient path depends on the same
  binding. Q6a/b/Q10/#9: B3 relevance none (serialize/SCC-ordering, not gradient-related).
- **F-007** (Q7.1 detail) — **REAL bug, unfixed**: `pmf_and_moments_from_graph(..., use_ffi=True)`
  with rewards returns the PMF computed on the **untransformed** graph (FFI
  `ComputePmfAndMomentsFfiImpl` never calls `reward_transform`), while the pybind default path and
  the multivariate FFI handler correctly transform. Verified against an oracle: PMFs differ by 1.38×
  relative. Reachable only via explicit `use_ffi=True` opt-in (no production caller does this
  today — SVGD always uses the default). Pre-existing, not refactor-introduced. Fix identified
  (mirror `graph_builder.cpp:741`'s `reward_transform` call in the FFI handler) but **not applied**;
  owned by "Stage-3" (an older initiative, itself apparently unfinished — see stage3 docs below).
  **Not in CLAUDE.md.** **B3 relevance: likely** — CLAUDE.md documents that B3's own reward-handling
  had a real bug ("rewards silently ignored by the exact Jacobian", fixed commit `315ce9c8`). This
  is a *second*, still-open reward+FFI-handler correctness bug in the same neighborhood
  (`ComputePmfAndMomentsFfiImpl`/rewards), raising the question of whether the exact-gradient Jacobian
  is differentiating a correct forward PMF when `use_ffi=True` + rewards are combined.
- **F-008 (N1)** — **REAL, pre-existing, unfixed**: `discrete=True` + rewards silently applies the
  **continuous** reward transform instead of `reward_transform_discrete`, in
  `ComputePmfMultivariateFfiImpl` (`graph_builder_ffi.cpp:791,840`) and the default pybind
  `compute_pmf_and_moments` path (`graph_builder.cpp`) — both give a silently wrong DPH PMF when
  rewards are supplied. Verified with a closed-form NegBinomial comparison (P(N=2): correct 0.0000
  vs library's 0.0450). **Not in CLAUDE.md.** **B3 relevance: likely** — this is the exact forward
  primal (discrete PMF/moments with rewards) that a discrete/was_dph exact-gradient implementation
  would need to differentiate correctly; if the *forward* value is wrong when rewards are present,
  an "exact" gradient computed elsewhere could either (a) be differentiating a different, correct
  internal representation (if B3's discrete path bypasses this handler) or (b) inherit the same
  silent error. Needs checking against which code path `exact_moment_grad` actually calls for
  discrete+rewards.
- **F-009 (N2)** — **REAL, pre-existing, unfixed, closed-form-proven**: DPH second moment computed
  with the **continuous** phase-type formula, overcounting by exactly `E[N]` (proven via
  `continuous_m2 − discrete_m2 = (I−P)⁻¹` and `α(I−P)⁻¹·1 = E[N]`). Root cause:
  `compute_moments_impl` (`graph_builder.cpp:479`) has **no `discrete` parameter at all** — discrete
  moments (beyond the mean) have never been implemented correctly. Mean is correct (masks the bug in
  mean-only checks); `E[N²]` is wrong by a large, provable margin. **Not in CLAUDE.md.** **B3
  relevance: likely, and high-priority** — this is a forward-primal bug in the *moments* computation
  itself, in the discrete case, which is precisely the quantity CLAUDE.md's B3 section says has a
  shipped exact discrete/was_dph gradient (`ptd_moments_grad_theta_dph`, default True). If
  `compute_moments_impl`/`ComputePmfAndMomentsFfiImpl` (C++ layer, this bug's location) is a
  *different* code path from the C `phasic.c` `ptd_moments_grad_theta_dph` machinery B3 actually
  uses, this bug may not contaminate B3's gradient — but that separation is NOT verified anywhere in
  these documents, and given B3's gradient is supposed to be the Jacobian of *some* forward moments
  computation, this deserves explicit tracing to confirm which forward computation the discrete exact
  gradient is differentiating, and whether it's this broken one.
- **F-010 (N3)** — **REAL, substantive (escalated from "cosmetic" by adversarial review),
  pre-existing, unfixed, off-limits to modify** (touches SVGD → `feedback_no_change_svgd` applies —
  investigate only). Tied SLAVE parameters in daisy-chain SVGD are correctly handled *inside* the
  model (fit is correct) but exported as a raw `0.0` sentinel in `get_results()`,
  `map_estimate_from_particles()`, `plot_posterior()` — ~8000× wrong rate reported with no "tied"
  flag anywhere except `summary()`. **Not in CLAUDE.md.** **B3 relevance: possible** — not a gradient
  bug per se (the gradient/fit is verified correct), but it's in the tied-parameter daisy-chain SVGD
  machinery, adjacent to the daisy-chain FD gradient that CLAUDE.md explicitly still lists as
  FD-only/out of B3 scope ("log/formula/joint-index/hierarchical still FD-only"). Relevant mainly as
  a correctness trap for anyone consuming exported posteriors from a model that also touches B3
  gradient paths.
- **F-011 (N4)** — **REAL memory-safety bug, pre-existing, unfixed**: a reward vector shorter than
  `n_vertices` causes an out-of-bounds heap read in `_ptd_graph_reward_transform`
  (`src/c/phasic.c:6189-6193`), with **no length validation** on the `pmf_and_moments_from_graph`
  path (contrast: `Graph.reward_transform` does validate). Demonstrated non-deterministic across
  processes (both forward value and FD gradient), i.e. genuinely UB, not just "surprising". **Not in
  CLAUDE.md.** **B3 relevance: likely** — this OOB read is in `src/c/phasic.c`, the SAME file that
  hosts every B3 gradient function (`ptd_moments_grad_theta`, `_dph`, `_log`, per CLAUDE.md). Even
  though the bug's trigger (malformed rewards) is a validation gap rather than a gradient-algorithm
  defect, undefined/nondeterministic reads in this file are exactly the kind of thing that could
  silently corrupt an exact-gradient computation if it shares the same reward-buffer plumbing. Given
  `feedback_no_silent_fallbacks`, this should raise on `len(rewards) != n_vertices` rather than read
  OOB.

**Net assessment (superseded by direct verification below):** the paragraphs above describe the
findings AS WRITTEN in the audit doc — 5 concrete pre-existing bugs (F-007, N1, N2, N3, N4). Live
verification against the current codebase (grep + `git log -S`) shows **every one of these has
since been fixed**, in commits not mentioned by CLAUDE.md (which is fine — CLAUDE.md is not a
changelog):

- **F-007/Q7.1** (FFI reward-PMF divergence) — fixed, `cc2d76ea "fix(ffi): reward-transform the
  graph in ComputePmfAndMoments (F-007/Q7.1)"`; the gate test file now says "Q7.1 (FIXED)".
- **N1/N2** (discrete reward-transform + DPH 2nd-moment bug) — fixed, `98f18e2a "fix(discrete): FFI
  parity for is_discrete dispatch (Batch 3b, N1/N2)"`.
- **N3/F-010** (tied-slave sentinel export) — fixed, `be6a6ed5 "fix(svgd): resolve tied slave
  parameters in every exported surface (F-010)"` (touches SVGD; presumably done with explicit
  sign-off given `feedback_no_change_svgd`).
- **N4** (reward-vector OOB read) — fixed (after one revert-and-redo cycle: `452ebe1d` → reverted
  `26db7f03` "convention underspecified" → re-fixed `66a5976d`/`bd751e03` "reward-length backstops").
- **Q5** (discrete moments pybind binding) — fixed/bound; `test_gate_moments_3way.py` now has
  `test_g2_moments_discrete_now_bound` ("Q5 resolved").
- **Q6a/Q6b** (serialize constant_edges / duplicate-vertex merge) — fixed; `test_gate_serialize_
  roundtrip.py` comments say "Q6a (FIXED)" / "Q6b (FIXED)", commit `2f60ea37` referenced.
- **Q10/#9** (SCC `sccs_in_topo_order` non-topological / `_expected_scc_filenames` tuple bug) —
  **STILL OPEN**: `tests/pytest/test_gate_scc_ordering.py:200` still carries a live
  `xfail`/skip reason citing "#9/Q10 ... trusts the [stored Tarjan order]". This is the one item
  from this file that remains a genuine, currently-open loose end. **B3 relevance: none directly**
  (SCC topological ordering for the composer, not a gradient computation) but **possible**
  indirectly since hierarchical SCC is the one B3 moment-gradient extension CLAUDE.md still lists as
  entirely unaddressed ("hierarchical SCC ... still FD-only") — an SCC-ordering correctness bug in
  the composer is adjacent risk for whoever eventually attempts that extension.

This is a good example of why "read the doc" alone is insufficient for an accurate loose-ends
inventory — most of this file's headline findings are resolved, and would be false positives if
reported as open without checking the live tree.

### audit-situation-map.md

The companion "atlas" to audit-differential-findings.md: full compute-path routing table (which
backend/exactness each `Graph` method uses under which flag combination), a list of what's already
gated (12 gate files, only 5 doing real cross-path comparison), a 15-item **gap list** of unguarded
routes, "intended vs accidental" behaviour-change classification, and a breadcrumb list of
known-but-unfixed defects. This is a point-in-time engineering map (dated to baseline `3082ebc6`,
pre-dating the current B3 work), not itself a set of "todo" items with independent status — but
several of its gap-list entries and breadcrumbs are directly germane and NOT reflected in CLAUDE.md:

- **Gap #4: "`weight_mode='log'` has NO gate anywhere"** — combined with "the known silent-ignore
  bug (`moments_from_graph` & `joint_index` ignore log)". The silent-ignore bug IS captured in
  CLAUDE.md/memory (`project_weight_mode_log_semantics`), but the **absence of any gate** for
  log-mode generally is not itself flagged in CLAUDE.md. **B3 relevance: likely** — CLAUDE.md says
  log-mode gradient support (continuous only) was added in the log-weight-mode batch; a gate gap
  here is directly adjacent to that shipped B3 work's correctness surface.
- **Gap #11: "Gradient equivalence across paths ungated. ... the mandated FD→analytic swap has no
  cross-path oracle."** — **B3 relevance: likely, directly on-topic.** This says there is no gate
  comparing the exact-gradient path against an independent cross-path gradient oracle (beyond the
  specific unit gates B3 itself built, e.g. `dr_moments_jac_gate.py`). Worth noting alongside
  CLAUDE.md's own list of B3 gates.
- **Gap #7: "Quantities with no cross-path gate at all: CDF, expected_sojourn_time/expected_waiting_time
  ... Laplace transform, plain (non-daisy) joint_prob_graph/joint_index."** — B3 relevance: possible,
  since `joint_index` is explicitly named in CLAUDE.md as still FD-only and under active work
  (b3-joint-index-plan.md, "most current" doc per task instructions).
- **Known-defect breadcrumb: "FD→analytic gradient removal deferred... three FD sites stay: moments
  use_ffi=False; reward-visit-prob central-diff custom_vjp (ffi_wrappers.py:1278); daisy-chain
  joint-probs FD custom_vjp (`__init__.py:11293`)."** Also notes an on-branch audit concluding the FD
  gradient is "broken in BOTH step regimes ... no central-difference step can fix this" at mixed
  parameter scales — this is the **origin claim** of the whole B3 initiative (matches
  memory `project_fd_gradient_b3`). CLAUDE.md's "Disabled paths" section does mention `reward_visit_probability`
  is not itself covered by B3 (it isn't named at all, actually) and daisy-chain FD is mentioned as
  out of scope. **`reward_visit_probability`'s FD custom_vjp (ffi_wrappers.py:1278) is NOT mentioned
  anywhere in CLAUDE.md's B3 section** — this is a third FD-gradient site (besides
  moments_from_graph/method_of_moments, already flagged in CLAUDE.md) that remains completely
  FD-based and un-flagged. **B3 relevance: likely — new gap for CLAUDE.md's follow-up list.**
- **Contradiction noted in §8**: "Daisy-chain gradient correctness not independently verified — the
  gradient is FD custom_vjp (`__init__.py:11293`); the daisy gate pins it only against a loose golden
  (5e-2)... Whether this is a defect or an accepted approximation is unresolved." Reinforces that
  daisy-chain gradient quality is an open, not merely "out of scope", question. **B3 relevance:
  likely.**
- Everything else (SCC ordering divergence, `_expected_scc_filenames` bug, `defect()`/`cdf()`
  inconsistency, notebook-checkout incident, native CTest dormancy, `wf_*` C++ compiler
  unreachability) is **not gradient-related** (B3 relevance: none) and mostly overlaps with Stage-3
  handoff docs covered below — not re-detailed here to avoid duplication.

---

### b3-c-theta-adjoint-plan.md / b3-derisking-strategy.md — one shared open landmine

Both documents (the pre-implementation de-risking strategy and the resulting detailed
implementation plan) are otherwise fully superseded by shipped B3 work, but both independently flag
the same **not-yet-addressed landmine**, in near-identical words:

> `ptd_graph_pdf_with_gradient` (`phasic.c:11805`, declared `api/c/phasic.h:1397`) is an **unwired,
> forward-mode PDF gradient** with a red-flag comment: "the lambda gradient term should be
> SUBTRACTED ... the minus sign gives correct results" — zero callers. **Re-derive + independently
> test before any PDF adjoint relies on it**; forward-mode also doesn't scale (O(n·n_params)).

This function is **not mentioned anywhere in CLAUDE.md**. It matters because CLAUDE.md's B3 section
covers only **moments** (continuous + discrete/was_dph); the continuous **PDF/PMF gradient is still
entirely FD** (unlisted explicitly, but implied — `pmf` stays FD per every B3 batch doc, e.g.
"pmf stays FD" in b3-batch2/3 findings). If/when B3 is ever extended to the PDF itself, there is a
pre-existing, seemingly-buggy, completely untested forward-mode gradient function sitting in the
same C file, with a comment actively warning about a possibly-wrong sign convention. **B3 relevance:
likely** — this is exactly the kind of numerical-stability/correctness trap the task explicitly asks
to surface, and it would be easy for a future implementer to find and reuse this function without
realizing it was already flagged as suspect.

b3-derisking-strategy.md additionally leaves **DR-D (Tier-2 scale-aware/clamped FD step, a stopgap
meant to stop the FD-driven SVGD-killing abort on daisy-chain gradients)** in an ambiguous state: the
doc frames it as a cheap, independent experiment to run, and separately notes that a *relative* FD
step was tried and explicitly reverted (commit `12a30a78`, "roll back the relative FD step to
master's absolute step"), with the doc telling the reader to "find out WHY" as a first sub-task. It's
unclear from these documents alone whether DR-D was ever properly executed as a clamped/scale-aware
(not simply relative) step, or abandoneed in favor of going straight to Tier-3. **This bears directly
on the still-open "SVGD hard crash on one bad particle" issue** flagged in audit-situation-map.md's
breadcrumbs (a θ implying rate ~1e31 crossing the `pure_callback` boundary and killing the whole SVGD
run) — worth checking against `deferred-svgd-divergece-fix.md` (read in a later batch) for current
status. **B3 relevance: likely.**

### B3-DISCRETE-MERGE-REVIEW.md — two items not in CLAUDE.md

This is the (now-merged, per git log `c0cb9de1`) handoff for the discrete/was_dph exact-gradient
batch — CLAUDE.md's B3 section already captures its headline result. Two narrower items from its
"carried over" / risk sections are not reflected in CLAUDE.md's text:

- **"Mixed-vertex decline"**: a `was_dph` vertex that mixes a constant edge and a parameterized edge
  as siblings causes the new discrete Jacobian to decline (return empty → FD fallback) rather than
  risk computing a wrong renormalization constant `S_v`. Verified safe (no crash) and confirmed this
  never arises from `Graph.discretize()`'s own output — but it IS a real scope boundary for anyone
  constructing a was_dph graph manually rather than via `discretize()`. Not documented in CLAUDE.md's
  B3 scope description. **B3 relevance: likely (direct scope boundary of shipped B3 discrete code).**
- **Open perf decision, carried over unresolved**: "Native FFI gradient handler vs the current host
  `pure_callback` (perf only)?" — still an open question per the doc's own "cross-cutting decisions
  carried over ... still open" list. Not mentioned in CLAUDE.md. **B3 relevance: likely** (a direct,
  acknowledged-open follow-up to the shipped B3 work, but purely a performance question, not
  correctness).
- (Also documents, as already-fixed history not an open item: a segfault in the **already-shipped
  continuous** `ptd_moments_grad_theta` when a graph contains a coefficient-less/constant tape-input
  edge — e.g. `joint_stop_prob_graph()`'s `add_aux_vertex_constant` "t-aux trapping loops" — fixed by
  a one-line additive guard, applied by explicit user request during this batch, both continuous
  gates re-verified passing after. This is DONE, not a loose end, but is a third distinct
  B3-adjacent bug beyond the two CLAUDE.md already lists from the "default-flip adversarial review" —
  worth noting for completeness since CLAUDE.md's bug count could be read as exhaustive when it isn't.)

---

### Cross-cutting: the "conditioning floor" — NOT reflected in CLAUDE.md

Three independent documents (`b3-experiment-findings.md` DR-A, `b3-prototype-findings.md`,
referenced again in `b3-derisking-strategy.md` DR-E) converge on the same finding: at extreme
mixed-scale θ (e.g. one rate ~1e-8 dominating `E[T]`), the **sub-dominant** gradient component is
corrupted **identically** for exact reverse-mode AD, forward-mode AD, and an independent
linear-solve oracle (relerr ~3 in the cited example) — a genuine **float64 precision floor of the
underlying linear algebra** (near-singular sub-generator, condition ~1/θ_small), not a defect of
any gradient method. FD is far worse in the same regime (2.2e7 relerr — actual garbage), so this is
NOT a reason to prefer FD; it's a residual limitation of "exact" AD itself. Every one of these docs
recommends the same follow-up: **characterise + document this regime, and add it to
`test_fd_gradient_mixed_scale.py` as a pinned (not xfail) documented conditioning limit** so the
gradient's real accuracy envelope is measured rather than assumed. Searching CLAUDE.md's B3
sections (moment gradient, joint-index gradient) turns up no mention of this — the MPFR
conditioning gate (`ptd_dbg_tape_needs_mpfr` / the joint-index MPFR-decline discussion) governs a
related but distinct concern (declining to FD above a tape condition-number threshold), and given
FD is *worse* in this exact regime, an MPFR-gate decline in the conditioning-floor regime would
trade a moderately-wrong exact gradient for a badly-wrong FD one — worth explicit verification.
**B3 relevance: likely — this is a direct, still-apparently-open numerical-accuracy question about
the exact gradient itself**, flagged repeatedly by name across three historical documents but
absent from the current canonical summary.

### b3-joint-index-plan.md — one cross-cutting gap not yet in CLAUDE.md (skimmed lightly per instructions)

This is the current, actively-worked B3 document (modified today; a parallel research stream covers
its code directly, so only its cross-cutting/follow-up sections were skimmed here). CLAUDE.md's
"B3 joint-index exact sojourn gradient" section already captures its headline gaps (the
`lax.cond`/`vmap` regression forcing `exact_grad` default `False`; the offset-tape-conversion
per-call cost being unmeasured at production scale; the `was_dph` exclusion; the MPFR-decline
rationale mismatch; the untested trap/deficit-sink fixture). One item from the plan's own
"Cross-cutting notes" section is broader than what CLAUDE.md captured and is worth surfacing
explicitly: the doc notes that the offset-tape-conversion caching gap is **not unique to
joint-index** — "the existing moments/log/dph gradient functions [`ptd_moments_grad_theta`,
`_log`, `_dph`] have this same no-caching gap (tolerable there so far because moment-graphs are
typically small — not yet benchmarked at scale, so 'tolerable' is an assumption worth eventually
checking too)." CLAUDE.md's moment-gradient section (the largest B3 section) does not mention any
per-call tape-conversion cost concern at all. **B3 relevance: likely, directly** — this is an
explicit, self-flagged gap the plan itself asks to be added to CLAUDE.md but which does not appear
to have made it into the moment-gradient section (only into the joint-index section, about its own
function). A live open design question (D6, "`lax.cond`/`vmap` redesign") is actively being planned
in this same document to fix the `vmap` regression and let `exact_grad` default back to `True`.

### sojourn-fix.md — two self-flagged follow-ups, neither in CLAUDE.md

This is the detailed writeup of the reverse-mode adjoint that replaced the O(n·k) forward
`expected_sojourn_time_subset` computation (commit `c340bedc`) — the same technique B3 later
extended to a θ-gradient (per `b3-c-theta-adjoint-plan.md`'s explicit citation of this exact commit
as "the reference exemplar"). Fully shipped and verified (F-002 in audit-differential-findings.md
independently re-confirmed it correct). Its own "§8 Caveats & follow-ups" section flags two items
that don't appear in CLAUDE.md:

- **`joint_prob_graph` construction is a separate O(n²) Python bottleneck** (`np.append`-in-a-loop
  growing child-state arrays, `__init__.py:10677` at time of writing) — untouched by the sojourn
  fix. For the largest joint-prob models (hundreds of thousands of vertices) this could dominate
  wall-time even though the sojourn *read-out* is now fast. **B3 relevance: possible** — any future
  B3 extension to joint-index/hierarchical models inherits this construction-time cost regardless of
  how fast the gradient itself becomes.
- **The no-arg full `ptd_expected_sojourn_time` (the whole residence-time vector, not a subset) is
  still O(n²).** The doc notes the same adjoint pass "already computes all n components" and "could
  replace the n×n forward there too" if ever needed at scale — an identified, easy, but unimplemented
  follow-up. **B3 relevance: possible** (same adjoint machinery family).

### stage2-coverage-safety-net-handoff.md / stage3-execution-handoff.md / stage3-refactor-plan.md — large historical architecture docs; most concrete bugs now fixed, three items remain genuinely open

These three documents (Stage-2 safety net, Stage-3 as-built record, Stage-3 full plan) total ~70KB
and represent a large historical refactoring initiative. Cross-checking their concrete findings
against the live tree:

- **Already fixed** (verified above under audit-differential-findings.md / category-A-fix-plans.md):
  Q5 (discrete moments unbound), Q6a/Q6b (serialize divergences), Q7.1 (FFI reward-transform), and
  the `moments_from_graph` `pow()`-recurrence forward bug (verified fixed — the current source has a
  comment "This previously did `rewards3[j] = rewards2[j] * pow(rewards2[j], i)`", i.e. the buggy
  recurrence is gone).
- **STILL OPEN, verified live, NOT in CLAUDE.md: `defect()`/`cdf()` inconsistency.** For a graph
  whose initial distribution puts mass directly on an absorbing vertex, `defect()` returns `0.0`
  while `cdf()` reflects the instant-absorption mass — an internal semantic inconsistency, flagged
  as a "production-behavior question for Stage-3" that was never resolved. Verified: the disabling
  comment and commented-out assertion are STILL present verbatim in `tests/cpp/testcpp.cpp` (~line
  619, "STAGE-2 DISABLED... Re-enable once Stage-3 resolves it") — i.e. it was never re-enabled,
  meaning the underlying semantic question was never settled. **B3 relevance: possible** — `defect()`
  and the reward/moment machinery share the same C engine notion of "mass that never leaves the
  start vertex"; an ambiguous semantic here is a latent trap for any future exact-gradient work on
  quantities touching absorption-at-start (e.g. `reward_visit_probability`, which CLAUDE.md's own
  gaps note is one of the still-FD-only quantities).
- **STILL OPEN, verified live, NOT in CLAUDE.md: #9/Q10 SCC composer split-brain.** Already detailed
  above (cross-referenced from `audit-differential-findings.md`'s Q10/#9 xfail) — `tests/pytest/
  test_gate_scc_ordering.py` still carries a live xfail for this. The Stage-3 plan calls this "the
  sharpest risk" and says a cross-provider parity gate "must exist before #9 or #12 is touched" —
  that gate exists only as an xfail-pinning test, not a passing equivalence check; the underlying
  divergence between the C Kahn-based composer and the C++ `SCCGraph::sccs_in_topo_order` (which
  trusts a stored, non-topological Tarjan order) is unresolved. **B3 relevance: possible** —
  hierarchical SCC is the one B3 moment-gradient dimension CLAUDE.md explicitly lists as entirely
  unaddressed; an ordering correctness bug in the SCC composer is squarely in the path of that
  eventual work.
- **UNCONFIRMED, not in CLAUDE.md: possible `bffg.py` duplication of sojourn/reward accounting.**
  `stage3-refactor-plan.md` §4.6 flags `bffg.py` (612 lines, backward-forward-backward-Gibbs /
  importance-weighting on sampled paths) as a "possible low-grade duplication of sojourn/reward
  accounting vs the C sampler," explicitly "unconfirmed... one confirmation pass recommended before
  touching it (not a blocker)." No evidence was found (in any later doc, commit message, or
  CLAUDE.md) that this confirmation pass was ever done. **B3 relevance: possible, weak** — if
  `bffg.py` independently re-implements sojourn/reward semantics in pure Python rather than calling
  the shared C engine, it could silently diverge from whatever exact-gradient-informed behavior
  changes land in the C core over time; this is speculative (unconfirmed even at the "is there a
  duplication at all" stage) but exactly the kind of indirect risk the task asks to surface rather
  than omit.
- **`reward_visit_probability`'s FD-only gradient** (`ffi_wrappers.py:1278`, central-difference
  `custom_vjp`, flagged as "deferred, analytical eventually via an adjoint through
  `ptd_backward_probabilities`" in `stage3-refactor-plan.md` §4a) — cross-referenced from
  audit-situation-map.md's detail section above; repeating here because `stage3-refactor-plan.md` is
  where the *reasoning* for keeping it FD (a JAX dual-path sanctioned-carve-out argument) lives.
  **Still not in CLAUDE.md's B3 gaps list. B3 relevance: likely.**
- **Lower-priority / not B3-adjacent**, not detailed further: the `x64`-not-enabled-at-import
  contradiction (§4 item 10 of stage3-refactor-plan.md) appears to have since been addressed (a
  defensive `jax_enable_x64` re-assertion with a warning exists in current `__init__.py`, though this
  was not exhaustively re-verified against the exact historical claim); the large god-object
  decomposition backlog (WS-D/E/F, `svgd.py` module split, `phasic.c` 9-way TU split, pybind
  thinning) is pure code-organization work, not correctness or gradient-relevant.

### b3-log-weight-mode-plan.md — one possibly-unfinished item

This plan is otherwise fully captured by CLAUDE.md (which cites it by name for the shipped
continuous-only log-mode gradient, and separately captures its "reverse-tape skeleton duplication"
finding almost verbatim). One item from the plan's own batch list does not have a visible
resolution anywhere in the repo-root docs: **D3** calls for the MPFR gate's scale-sensitivity under
log mode to be **quantified** ("a spread-θ log-mode graph's effective multiplier ratio grows faster
than the equivalent linear graph ... worth quantifying, not fixing ... reporting (not gating on)
how readily it declines relative to the equivalent linear-mode case"). No `b3-log-weight-mode-
findings.md` or merge-review document exists in the repo root (unlike every other shipped B3 batch,
which has a companion findings/merge-review doc) to confirm this measurement was actually taken or
what it showed. This may simply mean the log-mode batch's findings were folded directly into
CLAUDE.md's prose without a separate doc — but the specific quantitative question (how much more
readily does log-mode decline to FD under the MPFR gate vs linear-mode, for the same θ-spread) is
not answered anywhere available. **B3 relevance: likely**, low urgency (a measurement/reporting gap,
not a known defect).

### category-A-fix-plans.md — all 3 fixes verified shipped

Adversarially-reviewed implementation plans (each caught real errors in its own first draft via
review before any code was written) for the three reachable Category-A bugs identified by
audit-differential-findings.md. Live verification (git log + grep, matching commit messages that
explicitly cite the plan's bug IDs):

- **A1** (tied-slave 0.0-sentinel export, F-010/N3) → `be6a6ed5 "fix(svgd): resolve tied slave
  parameters in every exported surface (F-010)"`. The plan's own noted residual gap — the export
  fix does NOT cover `map_estimate_with_optimization()`'s post-hoc gradient-ascent refinement, which
  can let the master drift from the frozen slave copy — is explicitly flagged in the plan as
  "out of scope (pre-existing, not addressable here)"; not verified whether this residual gap was
  separately closed. **Minor open item, not in CLAUDE.md, low severity** (a narrow post-refinement
  export-drift edge case, not a gradient-correctness issue). B3 relevance: none (SVGD export
  bookkeeping, not gradient computation).
- **A2** (FFI reward-transform bug, F-007/Q7.1) → `cc2d76ea`, plus the plan's own escalation (fixing
  A2 ALSO silently fixed a second latent bug: discrete moments were being computed via a
  mathematically-inconsistent continuous-reward-weighting-then-discrete-correction sequence) —
  matches the N1/N2 fix (`98f18e2a`, "Batch 3b"). The plan called for "a discrete + integer-rewards
  FFI-vs-pybind MOMENT gate" since none existed — not independently verified whether this specific
  gate was added (a gap-in-the-gap, low priority to chase further).
- **A3** (trace-replay log footgun, F-004) → `fc5458c7 "fix(trace): require explicit use_log on
  trace replay (F-004)"`. The plan's **recommended, smaller "Option A"** (make the low-level footgun
  loud instead of silent — raise when `use_log` is omitted rather than silently defaulting to
  linear) is exactly what shipped: `_require_explicit_use_log` in `trace_elimination.py` now raises
  on `use_log=None`. **This directly closes the same landmine flagged as forward-looking risk in
  audit-differential-findings.md F-004** ("the deferred exact-AD gradient plan (Tier 1) proposes
  using `evaluate_trace_jax` as the differentiable backward... MUST thread use_log or it will
  silently differentiate the LINEAR function") — since it's now a hard `raise` rather than a silent
  wrong answer, this specific B3-adjacent risk is CLOSED, not open. **Correction to the batch-1
  detail section above**: the Tier-1 trace-replay `use_log` footgun is fixed, not merely latent.
  **B3 relevance: likely, but now RESOLVED**, not open.

### deferred-notebook-triage.md — mixed status, partially unverifiable without running notebooks

- **`state-space.ipynb`**: fixed same-session per the doc itself (a `skip-execution` tag fix). DONE.
- **`model-selection.ipynb` (b) `epoch_starts` TypeError**: root-caused as exactly
  `deferred-svgd-lr-bug.md`'s blocker, which is now fixed upstream (see that file's entry above) —
  the underlying API gap is closed. Whether the notebook itself was updated to use the new
  `epoch_model`/`likelihood_ratio_test_at` API (rather than still hitting the old `TypeError` path)
  was **not verified** — no commit touching `docs/pages/tutorial/model-selection.ipynb` was found
  that references `epoch_model` or the LRT fix. **Possibly still open as a doc/tutorial gap even
  though the underlying library bug is fixed.**
- **`model-selection.ipynb` (a) LRT likelihood inversion** (non-convergence from tiny tutorial
  fit budgets, not a library bug): status unverified; would need actually re-running the notebook.
  Plausibly still open (it's a tutorial-tuning issue, not something a code commit would fix).
- **`time-inhomogeneous.ipynb`** (graphviz `dot -c` env registration): no evidence of a fix (env
  configuration, not a code change — would not show in git log). **Likely still open.**
- **`distributed.ipynb`** (OpenMP thread-exhaustion → pybind11 `dec_ref`-with-invalid-GIL →
  interpreter `SIGABRT`, recommended to be filed as "a failed thread spawn should raise, not abort"):
  a `gil-release-moment-bindings` merge (`4d8c49a5`/`0730a42d`, "Release the GIL during
  moment-family computations") exists but addresses GIL release during normal moment computation
  (likely a performance/threading feature), not specifically the thread-spawn-failure crash path
  described here. **Not verified fixed; likely still open.** **B3 relevance: possible, weak** — this
  is a crash-vs-raise robustness pattern in the same spirit as the already-fixed rate-blowup
  fail-soft guard (deferred-svgd-divergece-fix.md, option A) but for a different failure mode
  (thread creation, not numerical blowup); worth noting as an open hardening gap in case a future
  distributed/parallel B3 hierarchical-SCC gradient extension exercises the same OpenMP thread path
  under memory/thread pressure.

---

## Prioritized list — open items most likely to matter for exact/FD gradient work

Ranked by (a) how directly they touch B3's own gradient code or its immediate forward-primal
dependencies, (b) whether they are confirmed still open (not superseded by a later fix), and (c)
severity if they do bite.

1. **The "conditioning floor" — inherent float64 precision limit on the sub-dominant gradient
   component at extreme mixed scale, identical for exact AD, forward-mode AD, and an independent
   oracle.** Flagged by name in three separate historical B3 documents (`b3-experiment-findings.md`,
   `b3-prototype-findings.md`, `b3-derisking-strategy.md`) with an explicit, repeated recommendation
   to characterise it and pin it as a documented regime in `test_fd_gradient_mixed_scale.py`. No
   evidence this was ever done, and it is absent from CLAUDE.md entirely. This is a direct,
   still-open accuracy question about the exact gradient's own reliability envelope — not a bug, but
   an un-quantified limitation.
2. **`ptd_graph_pdf_with_gradient` (`src/c/phasic.c:11805`) — an unwired, forward-mode PDF gradient
   with a self-documented suspicious sign-convention comment, zero callers, never re-derived or
   independently tested.** Flagged in two historical planning docs as something that must be
   re-verified before any future PDF/PMF gradient work (the one B3 dimension — pmf/pdf, as opposed to
   moments — that remains completely untouched) relies on it. A latent trap sitting in the same file
   as every shipped B3 gradient function.
3. **The shared "no-caching gap" for the offset-tape conversion (`ptd_pcg_convert_to_offset`), which
   the CURRENT `b3-joint-index-plan.md` explicitly says also applies to the shipped
   `ptd_moments_grad_theta`/`_dph`/`_log` functions ("tolerable there so far... not yet benchmarked at
   scale"), and explicitly asks to be added to CLAUDE.md.** CLAUDE.md's moment-gradient section does
   not mention any per-call tape-conversion cost concern; only the joint-index section (about its own,
   different function) carries this caveat.
4. **`reward_visit_probability`'s finite-difference `custom_vjp` (`ffi_wrappers.py:1278`)** — a third
   FD-gradient site (besides `moments_from_graph`/`method_of_moments`, which CLAUDE.md does list) that
   is not named anywhere in CLAUDE.md's B3 gaps, despite being explicitly called out as
   B3-in-principle-scope ("analytical eventually via an adjoint through `ptd_backward_probabilities`")
   in `stage3-refactor-plan.md`.
5. **The `defect()`/`cdf()` semantic inconsistency** — still verifiably open (disabled native assert,
   never re-enabled). Foundational ambiguity about "probability mass that never leaves the start
   vertex," in the same C engine as every reward/moment/gradient computation; a latent trap for future
   work on start-absorption-adjacent quantities.
6. **Gate/coverage gaps rather than known bugs**: no cross-path oracle for the exact-gradient-vs-FD
   swap generally (audit-situation-map.md gap #11); no gate for `weight_mode='log'` generally (gap
   #4, beyond the specific silent-ignore bug already tracked); the Q10/#9 SCC-composer ordering
   split-brain sits directly upstream of B3's one still-fully-unaddressed dimension (hierarchical
   SCC).
7. **Lower-confidence/speculative**: `bffg.py`'s unconfirmed possible duplication of sojourn/reward
   accounting; the `distributed.ipynb` OpenMP-thread-exhaustion→interpreter-abort crash (a
   crash-vs-raise robustness gap in the same spirit as the already-fixed rate-blowup fail-soft guard,
   but for a different C-layer failure mode that could plausibly surface again under a future
   parallel/distributed hierarchical-SCC gradient path).

Everything else surfaced by this audit (F-001/F-007/N1-N4/Q5/Q6a/Q6b, the tied-slave export bug, the
epoch-model/LRT gaps, the SVGD-divergence crash, the log-mode trace-replay footgun) has been verified
**already fixed** in the live tree, or is **already captured** in CLAUDE.md's existing "Disabled
paths / follow-ups" sections (moment-gradient gaps and joint-index gaps) almost as thoroughly as this
audit would restate it.

---

*(Batch 1 of ~6 complete: audit-differential-findings.md, audit-situation-map.md,
b3-batch0-realtape-findings.md, b3-batch1-reverse-adjoint-findings.md,
b3-batch2-exact-grad-findings.md. Batch 2 complete: b3-batch3-higher-moments-findings.md,
b3-batch3-mpfr-and-discrete-derisk.md, b3-c-theta-adjoint-plan.md, b3-derisking-strategy.md,
B3-DISCRETE-MERGE-REVIEW.md. Batch 3 complete: b3-discrete-theta-adjoint-plan.md,
b3-experiment-findings.md, b3-joint-index-plan.md (skimmed per instructions),
b3-log-weight-mode-plan.md, B3-MERGE-REVIEW.md, b3-prototype-findings.md. Batch 4 complete (with
live-code verification, which corrected several "open" findings to "done"): batch2-free-epoch-
model-plan.md, category-A-fix-plans.md, deferred-notebook-triage.md, deferred-svgd-divergece-
fix.md, deferred-svgd-lr-bug.md. Batch 5 complete: README.md, sojourn-fix.md,
stage2-coverage-safety-net-handoff.md, stage3-execution-handoff.md, stage3-refactor-plan.md.
Batch 6 complete (final): svgd-divergence-robustness-plan.md, svgd-lrt-fix-a-plan.md,
svgd-lrt-model-reuse-plan.md, tmp.md, tree_toplogy_encoding.md.

**All 31 repo-root markdown planning/findings/handoff documents read** (every `*.md` in
`/Users/kmt/phasic/` except `CLAUDE.md` itself). Audit complete.)*
