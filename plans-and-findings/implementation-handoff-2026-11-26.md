# Handoff: start of implementation (2026-11-26)

## Copy-paste prompt for a fresh session

> Read `OPEN-ITEMS.md` (the Priorities section first), then
> `plans-and-findings/fd-domain-steps-plan.md` (Plan A, v2.9, signed off)
> and `plans-and-findings/core-engine-ssot-plan.md` (the refactor master
> plan, v2.17, signed off). Start Plan A, Batch 0: the census, step-size
> measurements and the fresh master baseline, with no source changes, on
> branch `fix/fd-domain-steps` in a worktree `../phasic-fdsteps`. Record
> the results in `plans-and-findings/fd-domain-steps-findings.md`. Follow
> the plan's §0 prompt and every standing rule below.

## Where things stand

- **master** is at `2d11292c` or later; many commits are unpushed (the user
  pushes).
- **Plan A (domain-aware finite-difference steps), v2.9, SIGNED OFF** at
  review 13 of 13. Batches: 0 census (no code), 1 helper/builders/crash
  fallback, 2 Graph.svgd declares domains. User decision D-NZ: plain step
  for `'unconstrained'` near zero, crash prevention on refusing models,
  the limit documented; ties are counted and logged, not repaired.
- **Refactor master plan, v2.17, SIGNED OFF** at review 20 of 20. Plan A
  runs first, then phases 0a-0d, 1, 2, ..., 16. Every phase needs its own
  sub-plan, adversarially reviewed before any code.
- **Decisions recorded** (all in §2 of the refactor plan):
  - R1-R14;
  - D15: `-ffp-contract=off` build-wide;
  - D16: R13 scope;
  - D17: the trace recorder is kept as a tests-only oracle until phase 13;
  - D18: the binary hand-off format;
  - D19: `load_cpp_builder` raises `NotImplementedError` from phase 0d
    until phase 12;
  - D20: the cppimport route is kept, guarded by G1-G8 in phase 3.
- **Review evidence:** `experiments/fd_domain_steps/review1..13/` and
  `experiments/core_ssot/review1..20/`, each with a `REPORT.md`.
- **New defects found during the reviews** (in `OPEN-ITEMS.md`):
  - OI-088: moments wrong for k >= 13;
  - OI-089 and OI-116: `load_cpp_builder` does not compile, and it
    normalizes every graph;
  - OI-117: the exact moments gradient is off near rate poles;
  - OI-118: a `phasic::Vertex` assignment segfault.

## Standing rules (from the user and the assistant's memory)

- Never edit notebooks (`.ipynb`); list affected cells for the user.
- Memory-cap every run (`.git/claude-scratch/runcap.py --mb N --timeout S
  -- cmd`); one heavy run at a time; never run reviewers alongside test
  runs.
- Stay inside `/Users/kmt/phasic` and its worktrees; never inspect
  `~/.phasic_cache`.
- Plans and findings go in `plans-and-findings/`; open items go into
  `OPEN-ITEMS.md` first.
- Run `pixi run install-dev` after every source change; never
  `git add -A` (install-dev rewrites `pyproject.toml` and `pixi.lock`).
- Commits end with the Co-Authored-By line; the user pushes.
- Gates: `.git/claude-scratch/gate2.sh <tree> <tag>` (it WIPES the tag
  directory, so never re-run it on a finished tag) and `gate_compare.py`;
  the gate is no new failures, with regressions rerun three times in
  isolation.
- Reviews: one Opus reviewer per round, sequential. Stop a review series
  at the first round with no blocker or major.
- Ask before decisions that are hard to reverse; present options.
