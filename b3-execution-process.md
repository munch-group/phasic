# B3 execution process — documentation, branching, gates, git, deferral, and doc-maintenance policy

**Status: APPROVED 2026-08-11 with three user rulings folded in:
(1) adversarial review is mandatory for EVERY phase — both the batch plan
before code and the diff before merge; no waivers (routine bookkeeping
edits to tracker/ledger are not "phases"); (2) standing authorization to
proceed across implementation phases WITHOUT per-batch sign-off as long as
gates are green AND the plans are unchanged — any material plan deviation,
red gate, or ledgered decision point stops work and goes back to the user;
(3) push policy: THE USER PUSHES — the assistant never runs `git push`;
local commits per §5 are authorized (implied by (2): merging requires
commits), including the plan corpus.**

This document is deliberately about *process*, not design — design lives in
`b3-exact-gradient-master-plan.md` (signed off 2026-08-11) and the per-unit
plan documents. If this document and a design document conflict, the design
document wins and the conflict is fixed here.

---

## 1. Document taxonomy — what lives where

| Document | Role | Mutability |
|---|---|---|
| `b3-exact-gradient-master-plan.md` | **Design-of-record**: batch inventory, sequencing (§15), risk register (§16), follow-ups ledger (§16b) | Amend only with dated notes (the established pattern); never rewrite history |
| `b3-execution-tracker.md` | **Grand overview / dashboard**: one row per unit of work with live status + links to every detail file. THE first file to read in any session | Updated at every state change (see §7) |
| `b3-execution-process.md` (this file) | Process policy | Amend with dated notes on user approval |
| `b3-test-baseline.md` | **Pinned baseline ledger**: exact suite state at a named master commit; known-failure list with causes | Regenerated at every merge to master (§4.1) |
| `b3-<batch>-plan.md` (e.g. `b3-batch0-plan.md`) | Per-batch implementation plan, written when the batch STARTS, adversarially reviewed before any code | Frozen at review sign-off; dated amendment notes only |
| `b3-<batch>-findings.md` | De-risk/experiment findings for that batch (existing convention: `b3-batch1-reverse-adjoint-findings.md` etc.) | Immutable after batch close; dated correction notes only |
| `B3-<BATCH>-MERGE-REVIEW.md` or a merge-review section in the findings doc | Pre-merge review record: gate results, adversarial-review verdicts, deviations (existing convention: `B3-MERGE-REVIEW.md`, `B3-DISCRETE-MERGE-REVIEW.md`) | Immutable after merge |
| `deferred-{1,2,3,4}-*-plan.md` | Deferred-unit design-of-record (reviewed 2026-08-11) | As master plan |
| `atlas/*.md`, historical `b3-*-findings.md` | Reference/feasibility corpus | Read-only |
| `CLAUDE.md` | Behavior contract for shipped code (defaults, scopes, known gaps) | Updated in the SAME batch that changes shipped behavior (§7.2) |

Naming: all new B3 documents at repo root, prefixed `b3-`; plans end in
`-plan.md` (standing convention). Everything above is committed to git
(§5.6).

## 2. The grand overview (`b3-execution-tracker.md`)

One table, one row per unit: Phase-1 items (D.1, D.2, D.4, Batch 0, F, H,
the two cheap checks), later batches (D.3, E, A, B, C, G-leaves), the four
deferred units, and §16b ledger items. Columns: **id · what · status ·
branch/worktree · plan doc · findings doc · review record · merge commit ·
gates passed**. Status vocabulary (only these):
`not-started | de-risk | plan-review | implementing | gate | diff-review |
merged | parked | blocked(<on>)`.

Rules: the tracker never contains design content — links only; a unit's row
is updated in the same working session as the state change; every merged row
links its merge commit and its baseline regeneration; parked/blocked rows
say why and what un-parks them.

## 3. Branches and worktrees

1. **master is the integration trunk.** No implementation commits directly
   on master. Documentation-only commits (tracker updates, plan docs,
   CLAUDE.md edits accompanying a merge) may go directly on master.
2. **One branch per unit of work**, named by role:
   - `b3/<batch-id>-<slug>` for implementation batches
     (e.g. `b3/batch0-skeleton-extraction`, `b3/batchF-static-dispatch`);
   - `derisk/<slug>` for de-risk phases (already the convention baked into
     the deferred plans' handoff prompts).
3. **Worktree + isolated pixi env whenever native C/C++ is touched** — the
   proven pattern from the original B3 batches:
   `git worktree add ../phasic-<slug> <branch>` then
   `pixi run --manifest-path <WT>/pyproject.toml install-dev`, so the main
   checkout's installed package remains a valid baseline reference at all
   times. Pure-Python or docs-only branches may work in-place (remember the
   copy-install: `pixi run install-dev` after every edit, and re-run it on
   master after switching back).
4. **Concurrency** is governed by the master plan's cross-batch conflict
   matrix (§15) — units listed as independent may run in parallel worktrees;
   units sharing the moments core (0/A/B/C/D4-Phase-1) are strictly serial.
5. Branch lifetime: keep the branch after squash-merge until its
   post-merge baseline regeneration is green, then delete (the two
   historical squash sources may also be deleted at the user's convenience).

## 4. Gates — the ladder every batch climbs, in order

**G0 — baseline freshness.** `b3-test-baseline.md` must reference the
current master HEAD. If stale (master moved), regenerate before starting.

**G1 — batch-specific gates.** Defined in the batch's own plan (e.g.
Batch 0: the three jac-gates `dr_moments_jac_gate.py` /
`dr_dph_moments_jac_gate.py` / `dr_log_mode_moments_jac_gate.py`
byte-identical before/after; Batch F: the D6 plan's parity + vmap-cost
checks; Batch H: its own de-risk oracle). A batch plan without explicit G1
gates fails plan-review.

**G2 — targeted suite map.** The pytest files owned by the touched area,
run green (not merely no-worse):

| Touched area | Suites |
|---|---|
| Moments core (`ptd_moments_grad_theta*`, shared skeleton) | the 3 jac-gates + `test_gate_moments_3way.py` + `inference/test_jax_integration.py` (non-pre-broken subset per ledger) |
| Joint-index / sojourn | `test_joint_index_callback.py`, `test_optimized_joint_index.py`, `test_gate_daisy_chain_joint_probs.py` |
| Daisy / epoch | `test_epoch_sojourn_finalread.py`, `inference/test_lrt_at.py`, `inference/test_epoch_model.py` |
| SCC / hierarchical | `test_scc_compose.py`, `test_hierar_elimination_env.py`, `test_scc_*` |
| FFI handlers | `test_gate_ffi_vs_pybind.py`, `test_gate_trace_ffi_equivalence.py` |
| SVGD-touching (any) | `test_svgd.py` (slow — run once at G3, not per-iteration) |
| svgd config/validation (rules, kwargs, ledger) | `test_svgd_config.py`, `inference/test_svgd_exact_moment_grad_kwarg.py`, `inference/test_svgd_exact_moment_grad_rewards.py`, `inference/test_svgd_exact_final_grad_kwarg.py`, `inference/test_svgd_exact_grad_kwarg.py`, `inference/test_svgd_exposure.py`, `inference/test_svgd_api_parity.py`, `test_svgd_assumptions.py` *(row added 2026-08-13, G.1 G5 — the map predated the rule suites; proposed by the D.3/G.1 reviews; rewards file added 2026-08-14, A G5)* |
| Moments-adjoint rewards (Batch A surface) | `inference/test_exact_grad_rewards.py` (+ `experiments/dr_batchA_i1_gate.py check` when the C core/wrappers change) *(row added 2026-08-14, A G5)* |
| Moments-adjoint formula mode (Batch B surface) | `inference/test_exact_grad_formula_mode.py`, `test_weight_formula_*.py`, `test_gate_weight_formula_conformance.py` (+ `experiments/dr_batchB_i1_gate.py check` when the core/wrappers/wf-tape change) *(row added 2026-08-14, B G5)* |
| Moments-adjoint callback mode (Batch C surface) | `inference/test_exact_grad_callback_mode.py`, `test_callback_svgd_kwarg.py`, `test_weight_mode_probe_and_guards.py` (+ `experiments/dr_batchC_i1_gate.py check` when the core/wrappers change) *(row added 2026-08-14, C G5)* |
| Multivariate / 2-D rewards (Batch G.2 surface) | `inference/test_exact_grad_multivariate_kwarg.py`, `inference/test_multivariate.py`, `inference/test_multivariate_correctness.py`, `inference/test_multivariate_length1.py`, `inference/test_notebook_multivar_reproduction.py` *(row added 2026-08-15, G.2 G5)* |
| Joint-index / sojourn (amended 2026-08-13; +E 2026-08-14) | + `inference/test_exact_grad_joint_index.py`, `inference/test_exact_grad_joint_index_baked.py` |
| Gradient defect regression (always, cheap) | `test_fd_gradient_mixed_scale.py` |

<!-- AMENDMENT 2026-08-13 (G.1 G5), STRENGTHENED 2026-08-14 (E G5): chunked G3 commands ALWAYS pass -rf, AND each chunk's full pytest output is preserved (file per chunk) until the merge review records the tallies -- summarizing/tailing a chunk's output before failure names are recorded voids the run. -->
<!-- AMENDMENT 2026-08-14 (A G4, the bf-chunk incident): the chunk-group list is enumerated from the split output ON DISK at run time -- never assumed or hardcoded. Before tallies are recorded: (1) the union of the group files must equal the collected test-file list; (2) an output file must exist for every group. A missing group voids the run. (Batch A added a 156th test file; split -l 5 made 32 groups but only aa..be were initially run; caught by tally arithmetic, 1954 != ledger+6.) -->
**G3 — full-suite differential vs. the baseline ledger.**
`pixi run pytest tests/pytest/` compared against `b3-test-baseline.md`:
**zero new failures, zero new errors**; any new XPASS or vanished XFAIL is
investigated (a silently-unified cross-path pin is a finding, not a win —
established by the F-006 audit); known ledger failures may stay failing.
This — not literal all-green — is the "clean baseline" gate, per the
standing project instruction that pre-existing failures exist and full-green
must never be the gate. (If the 2026-08-11 baseline run comes back fully
green, the ledger is empty and G3 becomes literal green.)

**G4 — adversarial review of the diff. MANDATORY FOR EVERY PHASE (user
ruling 2026-08-11) — no waivers.** Refute-tasked, sized to complexity:
multi-agent for C changes, default-flips, or anything touching shipped
behavior; at minimum one dedicated refuter for small pure-Python or
test-only changes. The batch PLAN gets the same treatment before any code
(§8 step 3 — also mandatory, also unwaivable). This project's track
record — every serious defect in the B3 program was found by review, not
by the author's own tests — is the reason.

**G5 — merge review.** A short record (doc or section): gate results
verbatim, review verdicts, deviations from the batch plan, deferrals
ledgered (§6), CLAUDE.md/tracker updates included.

De-risk phases (experiments only) run G0 + their own experiment gates and a
findings doc; G2-G5 apply only when code changes ship.

## 5. Git policy

1. **Commits**: conventional prefixes already in use —
   `feat|fix|test|docs|chore(scope): message`; body cites the batch and its
   plan doc. Small, reviewable commits on the branch; the branch tells the
   story, master gets the summary.
2. **Staging**: explicit paths ONLY — never `git add -A` / `git add .`
   (standing rule: `install-dev` rewrites `pyproject.toml` + `pixi.lock`;
   never commit those unless doing deliberate version work). Never commit
   `.ipynb` changes without an explicit decision (the notebook-safety hook
   stands). Never `git stash` as part of any baseline or gate procedure.
3. **Merging**: **squash-merge batch branches into master** (the
   established pattern: `git merge --squash <branch>` → one clean master
   commit; the branch keeps the detailed trail until deletion per §3.5).
   Docs-only work commits directly to master. No rebasing of master, no
   force-push, ever. **No GitHub PRs** — solo repository; the adversarial
   diff review (G4) + merge-review record (G5) replace PR review.
4. **Push policy — RESOLVED (user ruling 2026-08-11): the user pushes.**
   The assistant never runs `git push`; after each merged batch the tracker
   notes "ready to push" so the user can push at their convenience.
5. **Tags**: none per-batch; releases keep the existing
   `pixi run bump/release` workflow, untouched by B3.
6. **Committing the plan corpus — RESOLVED (2026-08-11): committed.** The
   master plan, the four deferred plans, this file, the tracker, the
   baseline ledger, and the older plan docs (`category-A-fix-plans.md`,
   `batch2-free-epoch-model-plan.md`, `svgd-lrt-fix-a-plan.md`, the 6
   `atlas/plan-feasibility-*.md`) are versioned as the design-of-record.

## 6. Deferral policy — what happens when something out-of-scope surfaces

Every mid-batch discovery is triaged into exactly one class, in the batch's
merge review — nothing is ever silently dropped:

- **Class A — blocks this batch's correctness**: fixed inside the batch,
  with its own test; scope expansion recorded in the merge review.
- **Class B — real defect in shipped code, out of this batch's scope**
  (e.g. the two bugs found 2026-08-11): entry in master-plan **§16b
  ledger** (what, evidence file:line, suggested vehicle) + a pinning
  strict-xfail test *if cheap and non-invasive*; never fixed opportunistically
  in the same batch (`feedback_no_modify_existing` + scope discipline).
- **Class C — improvement/idea**: §16b ledger, one line, no test.
- **Class D — new information invalidating another plan's assumption**:
  dated amendment note in the affected plan document + tracker note; if it
  changes sequencing, a master-plan amendment proposed to the user before
  acting on it.

The ledger is reviewed at every phase boundary (not per-batch) to decide if
anything graduates into scheduled work — a user decision each time.

## 7. Documentation update policy

1. **Tracker**: updated in the same session as any state change (§2).
2. **CLAUDE.md**: updated in the SAME batch/merge that changes shipped
   behavior, defaults, scopes, or known-gap status (the established B3
   pattern) — never retroactively "when someone remembers".
3. **Master plan**: at each merge, tick/annotate the affected §15 item with
   the merge commit (dated); amendments only as dated notes.
4. **Findings/merge-review docs**: immutable after close; corrections as
   dated appendices.
5. **Baseline ledger**: regenerated at every merge to master (G3's
   reference for the *next* batch); the regeneration is part of the merge,
   not a separate chore.
6. **Memory** (assistant-side): updated end-of-session per its own rules;
   never a substitute for the repo documents above.

## 8. Standard session loop (the operational summary)

1. Read `b3-execution-tracker.md`; pick the next unit per master plan §15 +
   the conflict matrix.
2. G0 baseline freshness.
3. If the unit lacks a batch plan: write `b3-<batch>-plan.md` (with G1
   gates + de-risk steps), adversarial plan-review, present for user
   sign-off if it deviates from the master plan.
4. De-risk on `derisk/*` branch → findings doc → re-detail the plan if
   findings demand it (`feedback_derisk_and_reevaluate`).
5. Implement on `b3/*` branch (worktree + isolated env for native code).
6. Climb G1→G5.
7. Squash-merge; update CLAUDE.md/tracker/master-plan/baseline; (push, if
   §5.4 is approved).
8. Ledger any deferrals (§6); update memory.
