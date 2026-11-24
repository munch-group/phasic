# Bug-fix series handoff (2026-11-24)

Written so a fresh session can continue without the earlier conversation.

## Copy-paste prompt

Continue the bug-fix series described in `bugfix-handoff-2026-11-24.md`.
Work one thing at a time (no reviewers alongside test runs), run
reviewers on Opus not Fable, one reviewer per round, and merge a bug
when its suite comparison shows no new failures. Do not change
notebooks. Start with whatever the "Next step" line of the first
unfinished bug says.

## Recipe (user's, as scaled down on 2026-11-24)

Per bug, in its own worktree: plan -> adversarial review until sign-off
-> implement -> ONE adversarial implementation review -> suite comparison
against the previous bug's run -> merge to master if no new test fails ->
`pixi run install-dev` in the main tree. Scaled down: plans already
signed off are not re-reviewed; new plan rounds use ONE reviewer; reviewers
run on Opus (`model: "opus"`); nothing heavy runs in parallel.

Standing rules: never edit notebooks (list cells for the user); cap every
run's memory (`.git/claude-scratch/runcap.py`); stay inside the repo and
its worktrees; never inspect the real `~/.phasic_cache`; plans live in the
worktree root; `pixi run install-dev` after every source change; commits
end with the Co-Authored-By line; the user pushes.

## Tooling (`/Users/kmt/phasic/.git/claude-scratch/`)

- `gate2.sh <tree> <tag>`: one pytest process per file, writes
  `<tree>/.gate_<tag>/`. It WIPES that directory: never re-invoke on a
  finished tag.
- `gate_rerun.sh <tree> <tag> <cap_mb> files...`: rerun files, no wipe.
- `gate_compare.py <baseline_dir> <candidate_dir> --new-file F`: prints
  GATE PASS / GATE FAIL.
- When filtering pytest output, never pipe through `head` (it cut the
  summary line once); use `tail`.

## State per bug

1. **Compiled-gradient cache shared between fits** — MERGED (`974d6422`).
   Suite run: `/Users/kmt/phasic-bug1/.gate_bug1`.
2. **Fixed parameter values altered before reaching the model** — MERGED
   (`39154082`); main environment rebuilt. Suite run (baseline for bug 3):
   `/Users/kmt/phasic-bug2/.gate_bug2`.
3. **Single BetaPrior uses mismatched transforms** — MERGED (`b1203728`);
   main environment rebuilt. Suite run (baseline for bug 4):
   `/Users/kmt/phasic-bug3/.gate_bug3` (GATE PASS; the 4 test_plot_ci
   failures are on both sides, deterministic). Slow tier of the new file
   run once: 44/44. Found by the implementation review, pre-existing, not
   fixed: MCMC with positive_params=False, a single BetaPrior and a fixed
   value outside (0, 1) freezes (the hard wall sees the fixed value;
   found-defects item A territory).
4. **Discrete models accept row sums above 1 / negative probabilities** —
   MERGED (`195ab396`); main environment rebuilt. Suite run (baseline for
   bug 5): `/Users/kmt/phasic-bug4/.gate_bug4` (GATE PASS vs bug 3: 2912
   passed, the same 4 test_plot_ci failures; 64 new tests). Gate evidence
   in `experiments/bug4_discrete_rowsum/` (bit-identity, pdf-only pin
   refresh, timing pdf +3-7% / sampler +5-6%). Before merging, the branch
   was rewritten (unpushed) to drop ~95 MB of plan-review-3 scratch
   (`review3/nw/negw.jsonl`, `review3/patched/`); both remain on disk,
   untracked, in the bug 4 worktree.
5. **On-disk graph cache returns stale graphs** —
   `/Users/kmt/phasic-bug5`, branch `fix/graph-cache-stale`. Plan
   `bug5-graph-cache-plan.md` v1 REJECTED by reviewer 1 (evidence in
   `experiments/bug5_graph_cache/review1/`): the dependency fingerprint
   still gives stale hits (helpers behind `lru_cache`/`np.vectorize`,
   objects pickled by reference, two lambdas on one line, source edited
   after import) and refuses any callback using a `StateIndexer` (not
   picklable). Reviewers 2 and 3 were cut off; partial evidence may be in
   `review2/`, `review3/`. USER DECISIONS 2026-11-24: D1 caching only via
   a user-named entry `Graph(cb, ipv, cache_key="...")` (key = user string
   + callback AST + kwargs/ipv by value incl. `to_dict()` objects + phasic
   build id); `graph_cache=` removed. D2 the global switch
   (`configure(graph_cache=)`, `PHASIC_DISABLE_GRAPH_CACHE`) removed.
   Branch rebased on master (`5ef0d76c`), env rebuilt. Plan v2 committed
   (`d94cef2c`), including the notebook cell list (§5); one Opus
   reviewer running (evidence to `experiments/bug5_graph_cache/review4/`).
   Gate baseline: `/Users/kmt/phasic-bug4/.gate_bug4`. Plan now v2.2
   (`abeb620f`) after two REVISE rounds (review4, review5), not yet
   re-reviewed. User decided (2026-11-24) to keep the full v2.2 design
   (inference features rely on closures). USER RULES: once a reviewer
   signs off with no issues, run two more independent reviewers; DO NOT
   IMPLEMENT until the user says go.

## To report to the user at the end

- Decisions taken: bug 2 — any finite fixed value (incl. 0, negative)
  reaches the model exactly, only non-finite raises; `FreeEpochModel.fit`
  with a slot fixed at 0 now raises the negative-rate error. Bug 3 — SVGD
  refuses BetaPrior with `positive_params=False`, MCMC keeps accepting it.
  Bug 4 — negative/non-finite weights and row sums above 1 + 1e-9 raise;
  native discrete fits raise within 1e-7 of a bound. Bug 5 — user-named
  `cache_key=` replaces `graph_cache=`; global switch removed (D1, D2).
- Defects found but not approved for fixing: `found-defects-2026-11-23.md`
  (plus the out-of-scope lists in the bug 4 plan §2.3).
- Notebook cells needing the user's edits (model-selection.ipynb cell 13;
  bug 5 list in `bug5-graph-cache-plan.md` §5).
- Parked: `fd-step-and-slow-band-guard-plan.md` v5, to be split into Plan A
  (gradient step) and Plan B (forward replacement) after the bugs.
