# Deferred 1 — hierarchical/SCC two-level adjoint: de-risk & activation plan

**Status: v2 — adversarially reviewed 2026-08-11 (dedicated technical refuter:
SOUND-WITH-CORRECTIONS; cross-plan conflict refuter: COMPATIBLE-WITH-CORRECTIONS);
all findings folded in, see §10. Planning-only — no code changes of any kind are
authorized by this document.** This plan does NOT change Deferred 1's
"not scheduled" status in `b3-exact-gradient-master-plan.md` §11/§15; it defines
(a) the activation conditions under which the unit would be un-deferred, (b) the
de-risk experiments that must precede any implementation plan, and (c) a
conditional implementation sketch whose batches will be re-detailed from de-risk
findings (per `feedback_derisk_and_reevaluate`) before any code is written.

**Grounding.** Primary source: `atlas/plan-feasibility-hierarchical-scc.md`
(2026-08-05, fresh source re-derivation; cited below as "HSCC §n") plus master
plan §11 and its risk-register items 7 and 13. Claims below marked [HSCC §n] are
read-from-source findings of that document; claims marked [reasoned] are this
plan's own inferences and are exactly what the de-risk phase must confirm.

---

## 1. Goal and non-goals

**Goal (conditional):** an exact theta-Jacobian for quantities computed via the
hierarchical/SCC composition path (`parallel_elimination=True` /
`PHASIC_HIERAR_ELIMINATION=1`), so that large graphs which benefit from (or
require) SCC decomposition are not permanently excluded from exact gradients.

**Non-goals:**
- No change to the monolithic exact-gradient functions beyond what Batch 0
  (skeleton extraction) already produces. This unit *consumes* the Batch-0 core.
- No change to the numeric hierarchical composer's default behavior. Any
  retention of per-SCC artifacts is additive/opt-in ([[feedback_no_modify_existing]]).
- No `was_dph`/discrete support: the composer itself has zero discrete awareness
  today [HSCC §3] — extending that is value-side work, out of gradient scope.
- **`weight_mode='linear'` only [review F4].** The composer re-derives every
  weight with `use_log=false` unconditionally (`scc_compose.c:187`, `:315` —
  VERIFIED by review) and never copies the formula `weight_tape` — the numeric
  hierarchical path is silently linear-only today. Any adjoint inherits that
  scope; non-linear modes must be declined loudly at the entry point.
  Separately, this is a pre-existing silent-wrong-answer hazard in the
  NUMERIC path (a log-mode parameterized graph under `parallel_elimination=True`
  gets linear-weight results with no error) — recorded in the master plan's
  §16b follow-ups ledger as its own guard/doc item, not gated on this unit.

## 2. Why this is deferred (restated, so the activation logic is explicit)

1. The parent-level artifact of the hierarchical path is a plain numeric
   `double*`; per-SCC tapes are destroyed after each composition call; the
   cross-SCC phantom-edge weight `1.0 / parent_result[target]` is not
   representable in the existing tape-input encoding. A correct adjoint needs a
   genuinely new two-level reverse-mode structure [HSCC §2, §4].
2. Nothing is blocked today: the four exact-gradient C functions are
   monolithic-only under every configuration, and no tested workflow depends on
   the combination [HSCC §1; master plan §11.2 with the two-tutorials caveat].
3. **The primal itself limits the payoff:** the hierarchical gate requires
   `rewards == NULL`, and every moment beyond the first re-invokes
   `expected_waiting_time` with a concrete rewards vector, falling back to
   monolithic — so for `nr_moments >= 2` (the normal mean+variance case) even
   the *forward* computation gets no hierarchical benefit [HSCC §2]. An adjoint
   alone cannot fix this; see P3 below.

## 3. Activation conditions (gate A — decides whether the unit exists)

Un-defer only if at least one of the following holds:

- **A1 (forcing model):** a real model is found where monolithic elimination is
  intractable (memory or time), not merely slower, so hierarchical mode is
  *required* — upgrading this unit from parity nice-to-have to blocking gap
  (master-plan risk 7). Evidence source: de-risk experiment E0 below.
- **A2 (deliberate parity decision):** the user decides hierarchical/exact
  parity is wanted regardless of A1 (e.g. for the distributed/SLURM story).

If neither holds after E0, the unit stays parked; the only deliverables are the
E0 measurement note and the small documentation fix already tracked separately
(master-plan risk 16: `distributed.ipynb`/`profile.py` overstating
`parallel_elimination`'s benefit — that fix is NOT gated on this plan).

## 4. Prerequisites (gate P — must all hold before any implementation batch)

- **P1 — Batch 0 landed and gate-verified.** The inner level reuses the shared
  stage-0/1/2 core; building it against pre-extraction code would create the
  4th/5th near-duplicate the master plan exists to prevent [HSCC §6, §7].
- **P2 — exit-point design coordinated with Batch C.** Batch C's recommended
  shape is a "return the pre-contraction per-tape-input adjoint" exit on the
  shared core (master plan §5). The inner per-SCC VJP needs **more than the
  same class of exit** [cross-plan review F5]: it needs C's *output-side*
  pre-contraction exit PLUS **caller-supplied cotangent seeding** (an
  *input-side* entry — the shipped functions hard-code a one-hot seed at
  vertex 0, `phasic.c:10753`/`:10816-10818`). [reasoned — confirm in E1]
  The master plan's §5 and risk 12 now carry an explicit third-consumer note
  (amended 2026-08-11) so Batch C's implementer sees this requirement instead
  of designing blind to it. This remains a design-conversation dependency, not
  a code dependency; Deferred 1 must not block Batch C.
- **P3 — a value-side decision about moments.** Because the composer computes
  only the rewards-free first-moment vector [HSCC §2], the unit must choose
  its differentiated quantity explicitly before implementation:
  - **P3(a) scope to E[T]-class quantities only** (expectation / first-moment
    vector; what `ptd_compose_scc_prcs` actually produces). Smallest scope;
    honest but narrow — `pmf_and_moments_from_graph`'s moment regularization
    (K>=2) would remain monolithic-only.
  - **P3(b) first extend the composer to rewards-bearing replays** (value-side
    initiative, its own de-risk + review, out of B3 scope), then differentiate
    the extended composer. Much larger; only justified under A1.
  The choice is a user decision at activation time; this plan's default
  recommendation is P3(a) for a first increment.

## 5. De-risk phase (experiments; branch-only, no shipped-code changes)

Run on a branch per [[feedback_derisk_and_reevaluate]]. Experiments write only
new `experiments/dr_*.py` / scratch files; they do not modify `src/`. E0 is
independent of P1/P2 and can run anytime after master-plan sign-off; E1-E4
presuppose nothing has changed in the composer and can also run pre-P1 (they
test today's code), but their *conclusions* feed the post-P1 implementation
design.

- **E0 — scale/necessity measurement (answers gate A1).** Build (or reuse) at
  least one production-scale fixture (coalescent/two-locus class, n >= 1e4-1e5)
  plus all `toy_model.BUILDERS` fixtures (**6** — the docstring's "five
  variants" undercounts, `toy_model.py:332-339` [review F8]), all
  `weight_mode='linear'` (§1 scope). Measure across sizes: (a) the primal —
  monolithic `expected_waiting_time`/`expectation()` vs. hierarchical —
  wall-clock + peak RSS; **and (b) the shipped exact-gradient pipeline
  itself** (`_moments_grad_theta(1)` / the `pmf_and_moments_from_graph`
  exact path) [review F3] — it builds a private tape per call plus O(L)
  snapshot arrays (`phasic.c:10743-10761`) and does NOT share the primal's
  precompute/cache pipeline, and gate A1's real question is whether *exact
  gradients* are blocked: a model whose primal fits monolithically can still
  OOM/timeout in the gradient function. Pin the config matrix explicitly
  (`PHASIC_DYN_ORDERING` on/off, Stage-A2 cache cold/warm, `OMP_NUM_THREADS`)
  — these change the answer. Output: a table answering "is hierarchical ever
  *required* (OOM/timeout) vs. merely faster, and at what n — **separately
  for the primal and gradient pipelines**". Also record per-graph tape length
  L and SCC count/size distribution (feeds E1's cost model and Deferred-4's
  L-statistics incidentally). *Gate: A1 answered with numbers, not
  impressions.*
- **E1 — Python reference two-level adjoint vs. monolithic oracle (the core
  mathematical de-risk).** In pure Python/numpy (via `serialize()` +
  `SCCGraph` bindings where convenient):
  1. Reproduce the composition (sink-first, per-SCC eliminate, inject Type-C
     and phantom weights) on all 6 fixtures — value parity vs.
     `ptd_compose_scc_prcs` to ~1e-12 (the existing numeric gate's tolerance).
  2. Implement the two-level reverse pass: per-SCC VJP (cotangent-seeded
     reverse over the per-SCC elimination; seeds are the incoming cotangents on
     that SCC's internal-vertex results) + outer source-first accumulation over
     the condensation DAG, applying the composer's **actual piecewise boundary
     rule** at each phantom boundary [review F6]: `w = 1/x`, derivative
     `-1/x^2` for `x > 0`, but the composer injects the CONSTANT `1e300` when
     `parent_result[target] == 0.0` (every channel targeting an absorbing
     vertex; `scc_compose.c:213-223`) — there the derivative is 0 and the
     cotangent is dropped. Route Type-C contributions to the parent edge's
     real coefficients.
  3. Oracle — BOTH of the following, mandatory (not either/or) [review F1]:
     (a) a dense per-vertex-cotangent VJP reference (JAX autodiff or analytic
     on `alpha @ (-S)^{-1}`), evaluated at >=5 **random per-vertex cotangent
     seeds** per fixture — the shipped `_moments_grad_theta` binding is
     seeded exclusively one-hot at vertex 0 (`phasic.c:10753`, `:10816-10818`)
     and so validates only the e_start direction; and (b) the shipped binding
     itself as the e_start cross-check. Gate: machine-precision match
     (<=1e-12 rel) across all 6 fixtures × >=4 theta vectors — **restricted
     to the well-conditioned regime where the shipped oracle answers**; in
     mixed-scale regimes the shipped adjoint deliberately declines via its
     MPFR gate (`phasic.c:10783-10788`), so conditioning behavior is E3's
     question, not this gate's [review F10].
  This experiment resolves, before any C design: (i) whether cotangent-seeded
  reverse over a per-SCC tape is exactly the existing stage-1 mechanism with a
  different seed [reasoned — this is the load-bearing assumption; review
  inspected `phasic.c:10811-10829` and found the stage-1 accumulator
  seed-agnostic, so this survived as designed]; (ii) the precise bookkeeping
  of which parent vertices' cotangents feed which SCC; (iii) whether any
  quantity other than Type-C/phantom weights couples SCCs (HSCC §2 says no;
  E1 confirms empirically by perturbation testing); and (iv) that Type-A
  source-edge placeholders — which carry the same `[1.0, 0, ...]` placeholder
  coefficients and are never overwritten by the composer
  (`scc_synthetic.c:844-857`) — receive exactly zero adjoint or are
  confirmed-skipped by the tape, asserted by perturbation, not assumed
  [review F7].
- **E2 — landmine demonstration + guard design.** Empirically confirm HSCC
  §4(b)'s silent-wrong-answer hazard: run a shipped gradient function on a
  post-composition synthetic SCC graph and show it returns a plausible-looking
  wrong Jacobian (placeholder coefficients read as real). Output: the exact
  guard the implementation must add so this can never happen accidentally
  (e.g. a synthetic-graph marker checked by the exact-grad entry points), plus
  a pinned test-to-be. *This also protects the master plan's Batches A/B/C
  from a user mis-wiring them onto synth graphs — worth pinning regardless of
  whether Deferred 1 ever activates.* **Cost honesty [review F5 /
  cross-plan F13]:** no synthetic-graph marker field exists on
  `struct ptd_graph` today, and the guard's natural check sites are the four
  shipped exact-gradient functions — so the guard's eventual *implementation*
  is a modification of shipped code requiring explicit user approval, and its
  cheapest landing vehicle is Batch 0's shared core (one insertion, landing
  with an already-approved touch of that code). E2's deliverable here is the
  guard's design + demonstration only; implementation is routed to Batch 0 or
  a user-approved micro-batch (recorded in §7's matrix), never assumed free.
- **E3 — per-SCC vs. whole-graph conditioning (master-plan risk 13a).** On
  fixtures engineered to be ill-conditioned across SCC boundaries (rate ratios
  spanning >=1e12 *between* SCCs but benign within each), compare per-SCC
  condition numbers (what the composer's inner MPFR escalation sees) with the
  whole-graph condition number (what the monolithic gate sees). Output: does a
  per-SCC gate under-detect? Feeds the future adjoint's decline-gate design;
  also independently useful to the numeric path's own trustworthiness.
- **E4 — reentrancy survey (master-plan risk 13b).** Static read of what an
  in-loop per-SCC adjoint would share under `omp parallel for`
  (`scc_compose.c:517` region), plus a decision note: run the adjoint inside
  the existing parallel loop vs. as a separate (initially serial) pass over
  retained per-SCC artifacts. Default recommendation: **separate serial pass
  first** — correctness before parallelism; parallelizing the adjoint is a
  later optimization. [reasoned]

**Re-evaluation checkpoint:** after E0-E4, re-write §6 as a real implementation
plan (or park the unit), per [[feedback_derisk_and_reevaluate]]. Do not proceed
on the sketch below as-is.

## 6. Conditional implementation sketch (to be re-detailed post-de-risk)

Numbered for reference only; each batch gets its own detailed plan + adversarial
review at activation time.

- **B1 — additive retention mode in the composer.** A new entry point (or
  opt-in flag threaded from a new Python kwarg) that makes composition retain,
  per SCC: its PRC/offset tape, its channel-edge metadata (which parent edge /
  which parent target vertex each injected weight came from), and the
  `parent_result` snapshot. Default path byte-identical (retention off);
  no existing signature changes ([[feedback_no_modify_existing]]).
  Retention must handle BOTH tape forms [review F9]: a fresh build (raw PRC +
  offset copy) and a rev-3 cache HIT, where `prc == NULL` and only the offset
  tape exists — possibly mmap-backed with buffer lifetime tied to the mapping
  (`scc_compose.c:164-175`, `scc_synthetic.c:1166-1183`; the same region as
  the `bc071d84` segfault fix). Simplest v1: force fresh builds under
  retention mode. (Correction to §2's wording: per-SCC tapes are destroyed
  *in memory* after composition, but the rev-3 *on-disk* cache already
  persists per-SCC tapes — B1 extends a partially-existing mechanism rather
  than inventing persistence.)
- **B2 — per-SCC VJP primitive.** Cotangent-seeded reverse over one retained
  per-SCC tape, returning adjoints w.r.t. (i) the SCC's theta-linear edges
  (contracted via real coefficients), (ii) the injected channel-edge weights
  (returned uncontracted), and (iii) Type-A source-edge placeholders —
  asserted zero / confirmed-skipped and excluded from contraction (E1(iv));
  a nonzero Type-A adjoint is a hard error, never silently contracted
  [review F7]. Built on the Batch-0 shared core + the P2 exit
  (pre-contraction output + cotangent-seeded input).
- **B3 — outer DAG reverse pass.** Source-first traversal of the condensation:
  seed with the caller's cotangent on `parent_result`; at each SCC, call B2;
  route phantom-edge adjoints through the piecewise rule of E1.2 (`-1/x^2`
  for x>0; derivative 0 at the `1e300` clamp) into the producing SCC's
  result-cotangent; route Type-C adjoints into the parent edge's coefficient
  contraction; accumulate `J_out`. New code, C or C++ — no existing analogue.
- **B4 — wiring + gates.** Python entry (explicit opt-in kwarg, decline →
  logged FD per [[feedback_no_silent_fallbacks]]); gates: hierarchical adjoint
  == monolithic adjoint (machine precision) on all E1 fixtures + the E0
  production-scale fixture — **all `weight_mode='linear'`** (§1 scope; the
  primals themselves already disagree on non-linear fixtures, review F4);
  the E2 guard test; determinism under OpenMP.

**Sizing honesty:** B1-B4 constitute a large unit (new math + new
infrastructure), comparable to or larger than the discrete/was_dph batch. The
de-risk phase (E0-E4) is deliberately front-loaded so most of the risk is
retired before any C is written.

## 7. Interaction / conflict analysis

| Against | Interaction | Resolution |
|---|---|---|
| Batch 0 | Hard prerequisite (P1) — inner level consumes the shared core; also the E2 guard's implementation vehicle | Sequence strictly after Batch 0; the E2 guard lands with Batch 0 or a user-approved micro-batch [review F5]; no line overlap before then (this plan writes no code) |
| Batch A (rewards) | No line conflict (different functions once Batch 0 lands). Semantic note: composer is rewards-free; A does not change that [HSCC §6(a)] | None needed now; P3 records the moments limitation |
| Batch B/C (formula/callback) | Shared-core co-consumers; P2 shares Batch C's exit-point design | Design conversation at Batch-C time; Deferred 1 must not block C |
| Batch D/E/F/G/H | No overlap — those operate on monolithic tape paths / SVGD wiring / daisy chain [HSCC §6(c),(d)] | None |
| Deferred 2/3 | Orthogonal (daisy/JSP graphs are monolithic elimination targets in every path read) [HSCC §6(d), PDF-doc §8(d)] | None |
| Deferred 4 | If a full MPFR adjoint ever rewrites the shared core, a hierarchical inner level built on that core inherits it — sequencing only, no design coupling now. E0's L-statistics also size Deferred-4's Phase-2 memory cost (cross-referenced in that plan's matrix) [review F11] | Revisit if both units ever activate (unlikely per both plans' own recommendations); share E0's L data |
| `feedback_no_modify_existing` | B1 is additive-only; default composition path unchanged | Respected by construction |
| `feedback_no_silent_fallbacks` | Exact-vs-FD dispatch must be explicit + logged; the E2 guard converts the silent landmine into a loud error | Respected |
| Master-plan Phase 1 cheap check (co-occurrence grep) | E0 **extends** it with measurements (not "supersedes" — the grep retains its ongoing role, re-run as new tutorials are added [cross-plan F10]); not double-work — the grep is a 5-minute check, E0 is the quantitative follow-on §11.4 anticipated | Master plan §11 now names this plan as the unit's design-of-record; tick/annotate there when E0 runs |

## 8. Risks specific to this plan

1. The cotangent-seeded-reverse assumption (E1(iii)'s load-bearing claim) could
   fail in some structural corner (e.g. self-loop normalization inside an SCC
   interacting with the phantom edge) — that is precisely what E1's
   perturbation testing must probe adversarially, not confirm.
2. E0 may show hierarchical mode is *never* required at reachable scales — in
   which case the correct outcome of this plan is **parking the unit**, and
   that outcome must be reported as success, not failure.
3. Retention mode (B1) changes composer memory behavior when enabled;
   production-scale memory cost is unmeasured until E0/E1 quantify tape sizes.
4. The `PTD_PCG_PTR_EXTERNAL` kind exists only in the legacy on-disk format and
   is NOT used by live composition [HSCC §4(b)] — any design that assumed
   EXTERNAL anchors could carry cross-SCC derivatives would be building on the
   dead/broken WP-3 path ([[project_distributed_scc_arch]]). The implementation
   must not touch it.

## 9. Handoff (for a fresh session picking this up)

**State snapshot (2026-08-11):** master = `cadf1ca4`, 42 ahead of origin,
unpushed. All B3 shipped work is on master (continuous/discrete/log moments
adjoints default-on; joint-index sojourn adjoint opt-in). Master plan
`b3-exact-gradient-master-plan.md` awaits user sign-off; Batch 0 NOT yet
implemented. This unit is parked pending gate A. Feasibility source:
`atlas/plan-feasibility-hierarchical-scc.md`. Nothing in `src/` may change
until master-plan sign-off; this unit additionally requires gate A + P1-P3.

**Copy-paste prompt for the executing session** (rewritten per review F2 —
the v1 prompt deadlocked: it required gate A before E0, but E0 is the
experiment that answers gate A):
> Read `/Users/kmt/phasic/deferred-1-hierarchical-scc-adjoint-plan.md` and
> `/Users/kmt/phasic/atlas/plan-feasibility-hierarchical-scc.md` in full.
> Confirm (a) the master plan has been signed off and (b) the user has
> explicitly authorized this unit's de-risk phase; if either fails, stop and
> report. Then run E0 FIRST on a branch (`derisk/hierarchical-scc-adjoint`),
> writing experiments to `experiments/`, modifying nothing in `src/`.
> Evaluate gate A from E0's numbers (or an explicit user A2 decision): if A
> fails, park the unit and report that as the successful outcome. If A holds,
> run E1-E4 (they do not require Batch 0; P1-P3 gate *implementation
> planning*, not experiments). Then re-write §6 as a detailed batch plan from
> the findings, put that plan through adversarial review, and present it for
> sign-off before any implementation. Honor `feedback_no_modify_existing`,
> `feedback_no_silent_fallbacks`, `feedback_batch_plan`.

## 10. Adversarial review record (2026-08-11)

One dedicated technical refuter (verdict: **SOUND-WITH-CORRECTIONS**; 5 MAJOR
F1-F5, 6 MINOR F6-F11) + one cross-plan conflict refuter (this plan:
**COMPATIBLE-WITH-CORRECTIONS**). All findings folded into v2:

- **F1** dense per-vertex-cotangent oracle made mandatory, shipped binding
  demoted to e_start cross-check (§5-E1.3 — the shipped binding seeds one-hot
  at vertex 0 only).
- **F2 / cross-F6** handoff-prompt deadlock fixed: E0 runs before gate A is
  evaluated; P1-P3 gate implementation planning only (§9).
- **F3** E0 now measures the gradient pipeline (private per-call tape + O(L)
  snapshots), not just the primal, and pins the env-knob matrix (§5-E0).
- **F4** `weight_mode='linear'` scoping added (§1, §5, §6-B4); the composer's
  unconditional `use_log=false` recorded as a pre-existing NUMERIC-path
  silent-wrong-answer hazard → master-plan §16b ledger.
- **F5 / cross-F13** E2 guard costed honestly: new struct field + edits in
  four shipped functions; landing vehicle = Batch 0 or user-approved
  micro-batch (§5-E2, §7).
- **F6** piecewise phantom rule incl. the `1e300` clamp at
  `parent_result==0` (§5-E1.2, §6-B3).
- **F7** Type-A source-edge placeholder category added (§5-E1(iv), §6-B2).
- **F8** fixture count corrected to the 6 `BUILDERS` (§5).
- **F9** B1 retention handles fresh-build AND rev-3/mmap tape forms; §2
  "destroyed" wording corrected (in-memory only) (§6-B1).
- **F10** E1's 1e-12 gate restricted to the oracle's answering regime; the
  shipped oracle declines (MPFR gate) at mixed scale (§5-E1.3).
- **F11 / cross-F12** Deferred-4 L-statistics cross-reference added (§7).
- **cross-F5** P2 sharpened: the exit-point requirement is pre-contraction
  output PLUS caller-supplied cotangent seeding; master plan §5 + risk 12
  amended to name this third consumer (§4-P2).
- **cross-F10** "supersedes" → "extends" for the master-plan grep (§7).

Survived attack (verified by the reviewer at source): the P3 first-moment-only
claim; the Type-C/phantom-only cross-SCC coupling claim (with the F7 Type-A
caveat); E2's read-only runnability via existing pybind surface
(`scc_decomposition`/`as_synthetic_graph`/`Edge.update_weight`/
`_moments_grad_theta`); the load-bearing seed-agnostic stage-1 assumption;
all [HSCC §n] citations; the handoff snapshot.
