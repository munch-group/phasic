# Bug-fix series handoff (2026-11-24)

Written so a fresh session can continue without the earlier conversation.

## Copy-paste prompt

> Continue the bug-fix series described in `bugfix-handoff-2026-11-24.md`.
> Work one thing at a time (no reviewers alongside test runs), run
> reviewers on Opus not Fable, one reviewer per round, and merge a bug
> when its suite comparison shows no new failures. Do not change
> notebooks. Start with whatever the "Next step" line of the first
> unfinished bug says.

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
3. **Single BetaPrior uses mismatched transforms** —
   `/Users/kmt/phasic-bug3`, branch `fix/single-betaprior-transform`. Plan
   `bug3-betaprior-transform-plan.md` v4 signed off by three reviewers.
   Next step: rebase on master after bug 2 merges, install-dev, implement,
   one implementation review, suite comparison (baseline = bug 2's run).
4. **Discrete models accept row sums above 1 / negative probabilities** —
   `/Users/kmt/phasic-bug4`, branch `fix/discrete-negative-pmf`. Plan
   `bug4-discrete-rowsum-plan.md` v3.2 signed off by three reviewers.
   Next step: rebase after bug 3, implement (native rebuild), follow the
   plan's gate section (bit-identity dump, pin procedure, timing), one
   implementation review, suite comparison.
5. **On-disk graph cache returns stale graphs** —
   `/Users/kmt/phasic-bug5`, branch `fix/graph-cache-stale`. Plan
   `bug5-graph-cache-plan.md` v1 REJECTED by reviewer 1 (evidence in
   `experiments/bug5_graph_cache/review1/`): the dependency fingerprint
   still gives stale hits (helpers behind `lru_cache`/`np.vectorize`,
   objects pickled by reference, two lambdas on one line, source edited
   after import) and refuses any callback using a `StateIndexer` (not
   picklable). Reviewers 2 and 3 were cut off; partial evidence may be in
   `review2/`, `review3/`. Next step: redesign more simply — candidate:
   drop the callback-keyed cache and offer an explicit user-supplied
   `cache_key=` (or explicit save/load of a built graph) — write plan v2,
   one reviewer. Also from review 1: `config.py:465` encodes the default
   too; the graph cache ignores `cache_dir`; compute the key once per
   construction; tests must monkeypatch `graph_cache.DEFAULT_CACHE_DIR`.
   Open user decision: what explicit `graph_cache=True` does when the cache
   is disabled globally (raise / warn / silent).

## To report to the user at the end

- Decisions taken: bug 2 — any finite fixed value (incl. 0, negative)
  reaches the model exactly, only non-finite raises; `FreeEpochModel.fit`
  with a slot fixed at 0 now raises the negative-rate error. Bug 3 — SVGD
  refuses BetaPrior with `positive_params=False`, MCMC keeps accepting it.
  Bug 4 — negative/non-finite weights and row sums above 1 + 1e-9 raise;
  native discrete fits raise within 1e-7 of a bound. Bug 5 — cache becomes
  opt-in.
- Defects found but not approved for fixing: `found-defects-2026-11-23.md`
  (plus the out-of-scope lists in the bug 4 plan §2.3).
- Notebook cells needing the user's edits (model-selection.ipynb cell 13;
  bug 5 list still to be produced).
- Parked: `fd-step-and-slow-band-guard-plan.md` v5, to be split into Plan A
  (gradient step) and Plan B (forward replacement) after the bugs.
