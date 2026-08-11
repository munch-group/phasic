# B3 execution tracker — grand overview

**Read this first each session.** Live status for every unit of the B3
exact-gradient program. Design: `b3-exact-gradient-master-plan.md` (signed
off 2026-08-11, as amended). Process: `b3-execution-process.md`. Baseline:
`b3-test-baseline.md` — **ESTABLISHED 2026-08-11 at `cadf1ca4`: 0 failed /
1879 passed / 76 skipped / 24 xfailed** (4 stale `TestDiscretize` tests
aligned to the deliberate `c673be83` contract change; alignment edit
uncommitted pending git-policy approval). Phase 1 is clear to start.
Status vocabulary: `not-started | de-risk | plan-review | implementing |
gate | diff-review | merged | parked | blocked(<on>)`.

## Phase 1 (independent; may start once baseline ledger is recorded)

| id | what | status | branch/worktree | plan doc | findings | review record | merge |
|---|---|---|---|---|---|---|---|
| D.1 | `moments_from_graph` vmap-crash fix | not-started | — | master plan §6 (needs `b3-d1-vmap-plan.md` at start) | — | — | — |
| D.2 | `ptd_moment0_grad_theta` validator guards | not-started | — | master plan §6 | — | — | — |
| D.4 | `Graph.svgd()` leaf-5 plumbing | not-started | — | master plan §6 | — | — | — |
| 0 | Reverse-tape skeleton extraction | not-started | — | master plan §2 (needs `b3-batch0-plan.md`) | — | — | — |
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
