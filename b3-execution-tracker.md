# B3 execution tracker — grand overview

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
`b3-test-baseline.md` — **re-stamped 2026-08-14 (seventh) at `798ddcaa`
(Batch A merged): see the ledger for the measured post-merge tallies.**
Batches D Tier-1, 0, F, H, G.1, E, and A are MERGED; D.3 closed-folded
into G.1. **Phases 1, 1b, and 2 are COMPLETE; Phase 3 is A-done,
B/C open.** Next up: Batch B (formula) / Batch C (callback) — both
unblocked by A (note: the shared core's contraction signature now
carries `(rewards, rewards_len)`); Batch G.2 (2-D/multivariate leaf
kwarg semantics; optional vehicle for master §16b item 10); CC-2.
Master plan §16b: items 2 and 3 CLOSED at the H merge (comment corrected
in situ; conversion caching declined with evidence) — the snapshot
table below mirrors master §16b, which stays the authoritative list.
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
| CC-1 | Cheap check: `parallel_elimination` co-occurrence grep | done-once (during master-plan review); re-run as tutorials are added | — | master plan §15 | — | — | — |
| CC-2 | Cheap check: Deferred-4 Phase 0 sweep (= D4 design-of-record §2) | not-started (~days; needs oracle) | `derisk/d4-mpfr-sweep` (planned) | `deferred-4-mpfr-conditioning-floor-plan.md` §2 | `b3-d4-sweep-findings.md` (future) | D4 plan §8 | — |

## Phase 1b / 2 / 3 / 4 (gated)

| id | what | status | gate | plan doc |
|---|---|---|---|---|
| D.3 | leaf-2b plumbing (joint-index + exposure) | **CLOSED — folded into Batch G (user decision 2026-08-13)** after a BROKEN plan-review verdict (shipped rule R9 statically rejects the target leaf; master §6 premise false, Class-D amendment filed). D.3's user value ships via G leaf 1 (epoch-route exact gradients); R9 jsp hole (§16b item 9) fixed in G; review corrections transferred to G's plan | — | b3-batchD3-plan.md v1 + review record + disposition |
| E | Joint-index baked-mode scatter-add + svgd leaf-2 `exact_grad` | **merged** (branches/worktree deletable) — E0 GO; mid-batch user decision 2026-08-14 (`exact_grad_decline='fd'` per-particle fallback for svgd); G1 49/49; G2 156/3/1; G3 1947/0/84/24; G4 2 refuters S-W-C folded | b3/batchE-baked-exact | b3-batchE-plan.md v2 + amendment + merge review; b3-batchE-findings.md | `c475a78c` |
| A | Rewards support in moments adjoint + bundled svgd opt-out | **MERGED 2026-08-14** (squash `798ddcaa`, impl `47cb980b` + G4-fold `1ee12b3f`; micro-gates ALL PASS incl. G4-fold log leg + dph sub-kind contract; pre-fold G3 1957/0/84/24 = ledger+6; two G4 refuters SOUND-WITH-CORRECTIONS, zero shipped-code defects, corrections folded; 7th ledger stamp = measured post-merge run; dph rewards REFUTED by computation, permanent static decline; svgd 1-D-rewards leaf forwards exact_moment_grad = G leaf 3 delivered, G.2 shrinks to 2-D/multivariate; B/C unblocked) | b3/batchA-rewards (worktree deletable) | b3-batchA-plan.md v2 + G4 + G5 merge review; b3-batchA-findings.md |
| B | Formula-mode exact gradient | blocked(A) — lands as PTD_B3_FORMULA core-internal stage | after A | master plan §4 |
| C | Callback-mode exact gradient (Job A) | blocked(A) — exit options recorded in b3-batch0-plan.md | after A | master plan §5 |
| G | SVGD plumbing Tier 3 | **leaf 1 MERGED as Batch G.1 (`0c052cfe`)** — public `exact_final_grad` + R30 + R9 jsp fix (§16b item 9 closed) + R29 update; D.3's user value delivered (exposure + epoch_starts=[0.0] = fully-exact batched route); gates G0-G5 all green (G3 1917/0/84/24, flakes closed); branch/worktree deletable. Leaves 3/4 still blocked(A) | per leaf | master plan §9 + b3-batchG1-plan.md v2 + merge review + b3-batchG1-findings.md |

## Deferred units (parked; activation gates in their plans)

| id | what | status | un-parks when | design-of-record |
|---|---|---|---|---|
| Def-1 | Hierarchical/SCC two-level adjoint | parked | gate A (E0 evidence or user A2 fiat) + user authorization of de-risk | `deferred-1-hierarchical-scc-adjoint-plan.md` (v2, reviewed) |
| Def-2 | Daisy intermediate-epoch exact gradient | parked | **"Batch H shipped" now TRUE (`ecd708fc`)** — remaining: §1 value test + user authorization | `deferred-2-daisy-intermediate-epoch-plan.md` (v2, reviewed) |
| Def-3 | Exact PMF/PDF-term gradient | parked | gate A1 (user confirms wanted; E0 measurement) | `deferred-3-pdf-gradient-revival-plan.md` (v2, reviewed) |
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
