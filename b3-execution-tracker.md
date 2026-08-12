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
`b3-test-baseline.md` — **re-stamped 2026-08-13 at `164e2758`
(Batch D Tier-1 merged): 1885 passed / 0 failed / 84 skipped / 24 xfailed.**
Master is ready to push (user pushes). Next up: Batch 0.
Ledger snapshot note: master plan §16b now has 8 items — new item 7 =
`pmf_from_cpp` probable vmap sibling (statically confirmed, probe needs a
cpp-file fixture); old item 7 (daisy FFI NaN swallow) renumbered to 8.
Status vocabulary: `not-started | de-risk | plan-review | implementing |
gate | diff-review | merged | parked | blocked(<on>)`.

## Phase 1 (independent; may start once baseline ledger is recorded)

| id | what | status | branch/worktree | plan doc | findings | review record | merge |
|---|---|---|---|---|---|---|---|
| D.1 | `moments_from_graph` vmap-crash fix | **merged** | branch kept for now (worktree deletable) | `b3-batchD-tier1-plan.md` v2 + merge review | all gates green; G3 1885/0/84/24 | plan 2 refuters + diff 2 refuters (SOUND / SOUND-W-C) | `164e2758` |
| D.2 | `ptd_moment0_grad_theta` validator guards | **merged** | same | same | validators 5/5 pre + 5/5 post + guards gate ALL PASS | same | `164e2758` |
| D.4 | `Graph.svgd(exact_moment_grad=)` leaf-5 plumbing | **merged** | same | same | golden bit-identity 0.0; R29 incl. joint_stop_prob | same (D.4 v1 BROKEN → v2 flip) | `164e2758` |
| 0 | Reverse-tape skeleton extraction | plan-review | `b3/batch0-skeleton` (planned) | `b3-batch0-plan.md` v1 | — | 2 refuters in flight (2026-08-13) | — |
| F | D6 `lax.cond`/`vmap` static-dispatch redesign | not-started (design pre-reviewed: `b3-joint-index-plan.md` D6) | — | `b3-joint-index-plan.md` D6 §§ | — | — | — |
| H | Daisy final-epoch exact gradient | not-started | — | master plan §10 (needs own plan + de-risk) | — | — | — |
| CC-1 | Cheap check: `parallel_elimination` co-occurrence grep | done-once (during master-plan review); re-run as tutorials are added | — | master plan §15 | — | — | — |
| CC-2 | Cheap check: Deferred-4 Phase 0 sweep (= D4 design-of-record §2) | not-started (~days; needs oracle) | `derisk/d4-mpfr-sweep` (planned) | `deferred-4-mpfr-conditioning-floor-plan.md` §2 | `b3-d4-sweep-findings.md` (future) | D4 plan §8 | — |

## Phase 1b / 2 / 3 / 4 (gated)

| id | what | status | gate | plan doc |
|---|---|---|---|---|
| D.3 | leaf-2b plumbing (joint-index + exposure) | blocked(F) | Batch F | master plan §6, §15 Phase 1b |
| E | Joint-index baked-mode scatter-add (+ leaf-2 plumbing) | blocked(F) | Batch F | master plan §7 |
| A | Rewards support in moments adjoint | blocked(0) | Batch 0 | master plan §3 |
| B | Formula-mode exact gradient | blocked(0, then after A) | Batch 0 | master plan §4 |
| C | Callback-mode exact gradient (Job A) | blocked(0, then after A) | Batch 0; exit-point design must honor §5's third-consumer note | master plan §5 |
| G | SVGD plumbing Tier 3 (leaves 3/4 need A; leaf 1 needs H) | blocked(A / H) | per leaf | master plan §9 |

## Deferred units (parked; activation gates in their plans)

| id | what | status | un-parks when | design-of-record |
|---|---|---|---|---|
| Def-1 | Hierarchical/SCC two-level adjoint | parked | gate A (E0 evidence or user A2 fiat) + user authorization of de-risk | `deferred-1-hierarchical-scc-adjoint-plan.md` (v2, reviewed) |
| Def-2 | Daisy intermediate-epoch exact gradient | parked | Batch H shipped + §1 value test + user authorization | `deferred-2-daisy-intermediate-epoch-plan.md` (v2, reviewed) |
| Def-3 | Exact PMF/PDF-term gradient | parked | gate A1 (user confirms wanted; E0 measurement) | `deferred-3-pdf-gradient-revival-plan.md` (v2, reviewed) |
| Def-4 | MPFR conditioning floor (beyond Phase 0) | parked (Phase 0 = CC-2 above) | GAP outcome from Phase 0 + user approval | `deferred-4-mpfr-conditioning-floor-plan.md` (v2, reviewed) |

## §16b ledger snapshot (details in master plan §16b; triage per process §6)

| # | item | class | vehicle |
|---|---|---|---|
| 1 | `distributed.ipynb`/`profile.py` overstate `parallel_elimination` benefit | C/doc | standalone micro-task |
| 2 | Joint-index MPFR-comment correction | C/doc | bundle with E/F docs pass |
| 3 | Offset-tape conversion uncached (E/H hot paths) | C | evaluate in Batch H design |
| 4 | Rate-blowup fwd/bwd inconsistency (moments path) | B | unscheduled |
| 5 | `moments_from_graph`/`method_of_moments` exact grads | C | explicitly out of scope |
| 6 | Composer silently linear-only (`use_log=false` unconditional) — silent wrong numeric answer for log-mode + `parallel_elimination` | **B** | own guard/doc micro-fix; pin candidate |
| 7 | Daisy FFI swallows failures as NaN + `Success()` (unlogged on default sojourn handler) | **B** | small logging fix; full loud-path in Def-2 E1 |

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
