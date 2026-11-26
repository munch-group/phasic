# B3 execution tracker — grand overview

> **STOCK-TAKE 2026-08-16 — READ `b3-stocktake-2026-08-16.md` FIRST.**
> The user stated the library's real scale targets for the first time
> (20k-60k vertices TYPICAL; 200k-600k required) and a memory-capped
> measurement showed the shipped exact-gradient programme does not
> serve them: the moments gradient DECLINES via the conditioning gate
> on real two-locus graphs at 1k vertices, and no gradient call of
> either family completes within 15 minutes at 22.6k vertices. Every
> batch below is correct and gate-verified at the sizes it was tested
> at; none was ever tested at production size. Plans are re-opened
> pending a Q&A on intended use — treat the per-unit statuses below as
> accurate history, not as an agreed forward plan.

## Decoder (plain language — what the codes mean)

- **B3** — the whole program: replace finite-difference (FD) gradients with
  exact analytic ones throughout the library. (Historical label from the
  original de-risking doc.)
- **Batches** (units of work, defined in `b3-exact-gradient-master-plan.md` §1):
  **0** = refactor the three near-identical C gradient functions into one
  shared core · **A** = make the exact gradient work with `rewards` ·
  **B** = formula weight mode · **C** = callback weight mode ·
  **D.1** = fix the real bug where `moments_from_graph` crashed under JAX
  `vmap` · **D.2** = backport two safety guards to a debug-only C function ·
  **D.3/D.4/G** = wire `Graph.svgd()` options through to the exact paths ·
  **E** = exact gradient for "baked" joint-index models · **F** = replace a
  JAX `lax.cond` that wastefully computes both branches under `vmap` ·
  **H** = exact gradient for the final epoch of multi-epoch (daisy-chain)
  models.
- **Deferred 1-4** — four big investigations parked with their own plan
  files (`deferred-*.md`): hierarchical/SCC gradients, multi-epoch interior
  gradients, PDF-at-time-t gradients, high-precision (MPFR) gradients.
- **Gates G0-G5** (the checklist every batch passes, `b3-execution-process.md` §4):
  **G0** baseline up to date · **G1** the batch's own new tests ·
  **G2** targeted test files for the touched area · **G3** the FULL test
  suite compared to the pinned baseline (`b3-test-baseline.md`; currently
  zero known failures) · **G4** independent reviewer agents attack the code
  diff · **G5** written merge record, then squash-merge to master.
- **R29** — rule #29 in `Graph.svgd`'s argument validator
  (`svgd_config.py`): rejects the new `exact_moment_grad=` option wherever
  it cannot take effect, so it can never be silently ignored.
- **Worktree `../phasic-batchD`** — a second checkout holding the batch
  branch `b3/batchD-tier1`, so the main checkout's installed package stays
  untouched as a reference.

**Read this first each session.** Live status for every unit of the B3
exact-gradient program. Design: `b3-exact-gradient-master-plan.md` (signed
off 2026-08-11, as amended). Process: `b3-execution-process.md`. Baseline:
`b3-test-baseline.md` — **re-stamped 2026-08-14 (eighth) at `c6cc38b9`
(Batch B merged): measured post-merge tallies in the ledger.**
Batches D Tier-1, 0, F, H, G.1, E, A, and B are MERGED; D.3
closed-folded into G.1. **ALL PHASES (1, 1b, 2, 3, 4) ARE COMPLETE — every planned batch of
the B3 program is shipped** (tenth stamp `f73d0650`). The moments
adjoint is exact across all four weight modes with rewards on every
applicable leaf, and `exact_moment_grad` is honored on every moments
leaf. Remaining (unscheduled): CC-2 (Deferred-4 Phase-0 MPFR sweep),
the four deferred units, and the open §16b items. The core's
contraction switch has FIVE kinds (incl. the Batch-C binp exit).
Master plan §16b (authoritative list, now items 1-10): items 2 and 3
CLOSED at the H merge (comment corrected in situ; conversion caching
declined with evidence); item 9 (R9 jsp classifier hole) CLOSED at the
G.1 merge; item 10 ADDED at the A merge (pre-existing 2-D-rewards
FORWARD shape defect on the 1-D leaf — vehicle: G.2 or standalone);
items 1/4/5/6/7/8 open, statuses unchanged; item 11 ADDED at C's
close-out (analytic-derivative-callback opt-in + joint-index callback
exit + batched-vmap W, all ledgered). This note refreshed at C's
close-out (2026-08-14).
Status vocabulary: `not-started | de-risk | plan-review | implementing |
gate | diff-review | merged | parked | blocked(<on>)`.

## Phase 1 (independent; may start once baseline ledger is recorded)

| id | what | status | branch/worktree | plan doc | findings | review record | merge |
|---|---|---|---|---|---|---|---|
| D.1 | `moments_from_graph` vmap-crash fix | **merged** | branch kept for now (worktree deletable) | `b3-batchD-tier1-plan.md` v2 + merge review | all gates green; G3 1885/0/84/24 | plan 2 refuters + diff 2 refuters (SOUND / SOUND-W-C) | `164e2758` |
| D.2 | `ptd_moment0_grad_theta` validator guards | **merged** | same | same | validators 5/5 pre + 5/5 post + guards gate ALL PASS | same | `164e2758` |
| D.4 | `Graph.svgd(exact_moment_grad=)` leaf-5 plumbing | **merged** | same | same | golden bit-identity 0.0; R29 incl. joint_stop_prob | same (D.4 v1 BROKEN → v2 flip) | `164e2758` |
| 0 | Reverse-tape skeleton extraction | **merged** | branch kept (worktree deletable) | `b3-batch0-plan.md` v2 + merge review | M0-M6 byte-identity; validators 6/6; G2 (9 ledgered); G3 1885/0/84/24 | plan 2 refuters (S-W-C ×2) + diff 2 refuters (SOUND / S-W-C, folded M6) | `d2cca7ab` |
| F | D6 `lax.cond`/`vmap` static-dispatch redesign | **merged** | branch kept (worktree deletable) | `b3-batchF-plan.md` v2 + merge review | F0 GO; 17/17; golden 0.0; G2 (9 ledgered); G3 1888/0/84/24 | plan 2 refuters (D6.1) + diff 2 refuters, all S-W-C folded | `eaf86e82` |
| H | Daisy final-epoch exact gradient | **merged** (branches/worktree deletable) | derisk/batchH-final-epoch → b3/batchH-final-epoch (worktree `../phasic-batchH`) | b3-batchH-plan.md v3.1 + merge review | `b3-batchH-findings.md` — H0 oracle all-pass (composed grad 3.6e5× vs FD @ 7.4% cost); gate-decline finding; I1 micro-gates 6/6 + a2 bitwise; G1 11/11; G2 58/1; G3 1899/0/84/24 | de-risk plan 2 refuters (v1→v2) + v3 plan 2 refuters (→ v3.1) + G4 diff 2 refuters (S-W-C ×2, zero shipped-code defects, folded `43567b50`) | `ecd708fc` |
| D1-E2 | Synthetic-SCC-graph guard (exact-gradient cores decline placeholder-coefficient graphs) | **merged** | `b3/d1-e2-synthetic-guard` (in-tree, deviation recorded) | `b3-d1-e2-guard-plan.md` v2 (2 plan refuters folded) | `b3-d1-derisk-findings.md` §E2 (+ dated correction) | G1 8/8; G2 targeted 407+1x, SCC row 407+3x, SLURM row 81+2x; G3 2012/0/84/24 (11th stamp); G4 2 refuters S-W-C — refuter A found the distributed serialize BYPASS (fixed, pinned), refuter B found 6 stale C decline-cause lists + G0 staleness (both fixed) | `dac5fe8c` |
| CC-1 | Cheap check: `parallel_elimination` co-occurrence grep | done-once (during master-plan review); re-run as tutorials are added | — | master plan §15 | — | — | — |
| CC-2 | Cheap check: Deferred-4 Phase 0 sweep (= D4 design-of-record §2) | **DONE 2026-08-15 — verdict CLEAN, Deferred 4 PARKED (user sign-off: pin test + CLAUDE.md note + park, all shipped)**. Exact-rational oracle (mpmath substitution recorded) calibrated 1e-16-class; 64 points, 0 GAP; gate conservative 3-4 decades on swept fixtures; large-θ FD failure served by the exact path | `derisk/d4-mpfr-sweep` (branch deletable after merge) | `deferred-4-mpfr-conditioning-floor-plan.md` §2 | `b3-d4-sweep-findings.md` | D4 plan §8 | — |

## Phase 1b / 2 / 3 / 4 (gated)

| id | what | status | gate | plan doc |
|---|---|---|---|---|
| D.3 | leaf-2b plumbing (joint-index + exposure) | **CLOSED — folded into Batch G (user decision 2026-08-13)** after a BROKEN plan-review verdict (shipped rule R9 statically rejects the target leaf; master §6 premise false, Class-D amendment filed). D.3's user value ships via G leaf 1 (epoch-route exact gradients); R9 jsp hole (§16b item 9) fixed in G; review corrections transferred to G's plan | — | b3-batchD3-plan.md v1 + review record + disposition |
| E | Joint-index baked-mode scatter-add + svgd leaf-2 `exact_grad` | **merged** (branches/worktree deletable) — E0 GO; mid-batch user decision 2026-08-14 (`exact_grad_decline='fd'` per-particle fallback for svgd); G1 49/49; G2 156/3/1; G3 1947/0/84/24; G4 2 refuters S-W-C folded | b3/batchE-baked-exact | b3-batchE-plan.md v2 + amendment + merge review; b3-batchE-findings.md | `c475a78c` |
| G.2 | Multivariate/2-D leaf `exact_moment_grad` + §16b item 10 | **MERGED 2026-08-15** (squash `f73d0650`, fold text-only; TWO user decisions: full symmetry + uniform rejection; R32 added, predicate narrowed at G1 — the plan-review sparse-only claim refuted by five shipped tests, defect-avoidance recorded; G3 2001/0/84/24 = ledger+9; two G4 refuters SOUND-WITH-CORRECTIONS, zero shipped-code defects; 10th stamp; **BATCH G CLOSED, PHASE 4 COMPLETE — every planned batch shipped**) | b3/batchG2-multivariate (branch deletable) | b3-batchG2-plan.md v1+v2; b3-batchG2-findings.md |
| C | Callback-mode exact gradient (Job A) | **MERGED 2026-08-14** (squash `35a17364`, fold `88e5cc68`; 9th stamp = measured post-merge run; **PHASE 3 COMPLETE** — all four weight modes exact) (plan v1+v2 two-refuter review — headline: the ALIGNED-theta-dim restriction REFUTED for callback, decoupled graphs SUPPORTED; deployed-transform jit(grad) probe; D-C5 was_dph evidence load-bearing; impl `ff0e9d91`+`e6d3b613`: binp exit 5th core consumer + Python matmul contraction; micro-gates ALL PASS incl. 7-golden byte-identity + exit==linear-exact BITWISE at mixed scale + LAZY-decoupled engagement; 12/12 new tests + discrete-file fate rework; G2 17-file map 200/31/3 zero fail) | b3/batchC-callback (worktree ../phasic-batchC) | b3-batchC-plan.md v1+v2; b3-batchC-findings.md |
| B | Formula-mode exact gradient | **MERGED 2026-08-14** (squash `c6cc38b9`, G4 fold `d6bb0c99`; Wengert-list wf-tape autodiff + PTD_B3_FORMULA core kind, ALIGNED-scope static gates; micro-gates ALL PASS incl. rewards-bearing byte-identity goldens + formula==linear-exact BITWISE at mixed scale; G3 1975/0/84/24 = ledger+12 pre-fold, 15 tests post-fold; two G4 refuters SOUND / SOUND-WITH-CORRECTIONS — the wiring refuter's independent oracle 24/24 at ≤3.3e-15, zero memory drift; 8th ledger stamp = measured post-merge run; lazy-decoupled formula class stays FD-with-log, ledgered follow-up; **Batch C unblocked** — strictly-serial satisfied) | b3/batchB-formula (worktree deletable) | b3-batchB-plan.md v1+v2+merge review; b3-batchB-findings.md |
| A | Rewards support in moments adjoint + bundled svgd opt-out | **MERGED 2026-08-14** (squash `798ddcaa`, impl `47cb980b` + G4-fold `1ee12b3f`; micro-gates ALL PASS incl. G4-fold log leg + dph sub-kind contract; pre-fold G3 1957/0/84/24 = ledger+6; two G4 refuters SOUND-WITH-CORRECTIONS, zero shipped-code defects, corrections folded; 7th ledger stamp = measured post-merge run; dph rewards REFUTED by computation, permanent static decline; svgd 1-D-rewards leaf forwards exact_moment_grad = G leaf 3 delivered, G.2 shrinks to 2-D/multivariate; B/C unblocked) | b3/batchA-rewards (worktree deletable) | b3-batchA-plan.md v2 + G4 + G5 merge review; b3-batchA-findings.md |
| G | SVGD plumbing Tier 3 | **leaf 1 MERGED as Batch G.1 (`0c052cfe`)** — public `exact_final_grad` + R30 + R9 jsp fix (§16b item 9 closed) + R29 update; D.3's user value delivered (exposure + epoch_starts=[0.0] = fully-exact batched route); gates G0-G5 all green (G3 1917/0/84/24, flakes closed); branch/worktree deletable. All four leaves DELIVERED: leaf 1 = G.1+H, leaf 2 = E, leaf 3 = G.2 (2-D), leaf 4 = A (1-D) | per leaf | master plan §9 + b3-batchG1-plan.md v2 + merge review + b3-batchG1-findings.md |

## Deferred units (parked; activation gates in their plans)

**De-risk phases for Def-1/2/3 executed 2026-08-15** (consolidated
branch `derisk/d1-scc-adjoint`, squash-merged to master `a5bb71e1`;
serial + memory-capped after the 50GB incident). Findings:
`b3-d{1,2,3}-derisk-findings.md`. Three-way user checkpoint decided
2026-08-15 — statuses below.

| id | what | status | next step | design-of-record |
|---|---|---|---|---|
| Def-1 | Hierarchical/SCC two-level adjoint | **ACTIVATED** (de-risk complete: E0 forcing model EXISTS, E1 GO, E2 landmine+guard, E3 inverted, E4 forced-serial) | implementation plan **DRAFTED v3, awaiting user sign-off** (`b3-d1-implementation-plan.md`, both refuters folded); **E2 guard SHIPPED** (branch `b3/d1-e2-synthetic-guard`) | `deferred-1-hierarchical-scc-adjoint-plan.md` + `b3-d1-derisk-findings.md` + `b3-d1-implementation-plan.md` |
| Def-2 | Daisy intermediate-epoch exact gradient | de-risk (user: GO) | remaining E2 cost model (large case) + E3 oracle/custom_vmap probe → build-vs-park returns to user | `deferred-2-daisy-intermediate-epoch-plan.md` + `b3-d2-derisk-findings.md` (A2: benign ~6e-8; mixed-scale shipped backward CRASHES) |
| Def-3 | Exact PMF/PDF-term gradient | de-risk (user: GO; **route (ii) Poisson mixture selected, route (i) refuted**) | E0 value measurement on real fits + E5 dossier → build-vs-park returns to user | `deferred-3-pdf-gradient-revival-plan.md` + `b3-d3-derisk-findings.md` |
| Def-4 | MPFR conditioning floor (beyond Phase 0) | parked (Phase 0 = CC-2 above) | GAP outcome from Phase 0 + user approval | `deferred-4-mpfr-conditioning-floor-plan.md` (v2, reviewed) |

## §16b ledger snapshot (details in master plan §16b; triage per process §6)

| # | item | class | vehicle |
|---|---|---|---|
| 1 | `distributed.ipynb`/`profile.py` overstate `parallel_elimination` benefit | C/doc | standalone micro-task |
| 2 | Joint-index MPFR-comment correction | C/doc | **CLOSED @ H merge `ecd708fc`** (comment corrected in situ in the rewritten core, with H0 evidence) |
| 3 | Offset-tape conversion uncached (E/H hot paths) | C | **CLOSED @ H merge** (declined with H1(a) evidence: the adjoint call incl. conversion = 1.0-1.3% of the FD backward across 37× sizes) |
| 4 | Rate-blowup fwd/bwd inconsistency (moments path) | B | unscheduled |
| 5 | `moments_from_graph`/`method_of_moments` exact grads | C | explicitly out of scope |
| 6 | Composer silently linear-only (`use_log=false` unconditional) — silent wrong numeric answer for log-mode + `parallel_elimination` | **B** | own guard/doc micro-fix; pin candidate |
| 7 | `pmf_from_cpp` vmap sibling (2-D batch handling, same family as the D.1 fix) | C | unscheduled |
| 8 | Daisy FFI swallows failures as NaN + `Success()` (unlogged on default sojourn handler; + batched sojourn NaN-fill on bad indices, F merge review dev. 3) | **B** | small logging fix; H0(i) treats NaN as confound; full loud-path in Def-2 E1 |

## Non-B3 residuals (from the 2026-08-10 status report, so they aren't lost)

- `docs/pages/tutorial/model-selection.ipynb` still uses the fix-D-broken
  pattern; in the `pixi run test` conversion path. Doc rewrite pending.
- `epoch_model` degenerate-outcome-group RNG tie-break mapping untested.
- Legacy cache-format hardening (`PTDPRMC1`) + `pull_cache` parent-artifact
  promise broken (atlas finding) — unscheduled, not gradient work.
- F-005: 512 auto-granularity floor doc note.
- Working tree: `.gitignore`, `.vscode/settings.json`,
  `docs/pages/tutorial/discrete.ipynb` modified + uncommitted (user
  decision).
