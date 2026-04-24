# Using coverage-kit

This guide is for people working in a repo where coverage-kit is already
installed (you see a `.claude/commands/coverage*.md` tree). If you're setting
up a fresh repo, run `/coverage-init` first — it configures the kit against
your project and is described briefly at the end of this document.

---

## The mental model

The kit is two workflows wired together:

1. **Measure** (`/coverage <target>`) — the coverage-analyst subagent runs
   the configured coverage command, parses the JSON it produces, and walks
   every uncovered region. For each region it assigns a risk tag
   (🔴 public API / 🟡 internal helper or branch / 🟢 dunder/getter/trivial)
   and a one-sentence hypothesis for why it's uncovered. A coordinator writes
   a single prioritized markdown report to `.claude/coverage-reports/`.
2. **Apply** (`/coverage-apply`) — that report is parsed, grouped by source
   file into small batches, and tests are written one batch at a time.
   After every batch the test suite runs AND coverage is re-measured; a
   batch is only committed if tests stay green AND the batch's target lines
   are actually hit.

The split exists so you get:
- a measurement phase that's cheap to re-run and safe to read (nothing is edited),
- a test-writing phase that's safe to leave running (every commit is
  test- and coverage-verified).

Every commit Part 2 creates lists the findings it resolved, so `git log` is
a readable audit trail and `git revert <sha>` cleanly undoes any one batch.

**Coverage-kit only writes tests.** It never edits source code. If a finding
turns out to be unreachable without a source change, Part 2 marks it
`skipped` with a note and moves on.

---

## A complete walkthrough

Suppose you want to improve coverage for `src/mylib/`.

### 1. Start from a clean tree

```bash
git status          # should be clean; commit or stash in-progress work
```

Part 2 refuses to run on a dirty tree because mixing auto-edits with
in-progress work is a recipe for losing both.

### 2. Generate a coverage report

```
/coverage src/mylib
```

(With no argument, the default target from `/coverage-init` is used.)

What happens:
- The orchestrator verifies the target exists.
- It spawns the `coverage-analyst` subagent.
- The analyst runs the configured coverage command (e.g.
  `pytest --cov=src/mylib --cov-report=json:.coverage.json`), parses the
  JSON, and walks every uncovered region.
- Each region gets a severity tag, a kind (function / branch / error-path /
  module-top-level / dunder), a why-uncovered hypothesis, and a one-sentence
  suggested-test description.
- Suspected dead code (zero hits AND zero references elsewhere) is
  flagged separately and is **not** used as a test-writing target.
- The orchestrator writes a single markdown report to:

  ```
  .claude/coverage-reports/src-mylib-20260423-151200.md
  ```

- You get back a summary: overall coverage %, threshold, counts by severity,
  dead-code candidate count, and a reminder that `/coverage-apply` is next.

**You can stop here.** The report is useful on its own — skim it, pick
findings to tackle by hand, share it. `/coverage-apply` is only needed if
you want to add tests in bulk.

### 3. Read the report

Open the newest file under `.claude/coverage-reports/`. Structure:

```
# Coverage — src/mylib

Overall coverage: 62% (threshold: 80%). 34 uncovered regions across 7 files.

[🔴] [a3f9c2d1] parse_config — src/mylib/parser.py:45-78
Signature: def parse_config(path: Path) -> Config:
Kind: function
Why uncovered: no test exists
Suggested test: Call with a minimal valid config file and assert the
  returned Config has the expected fields.

[🔴] [...]
...

[🟡] [...]
...

[🟢] [...]
...

### Per-file summary
| File                       | Stmts | Miss | Cover% |
|----------------------------|-------|------|--------|
| src/mylib/parser.py        |   240 |  102 |    58% |
| ...

### Dead-code candidates (do not test)
- src/mylib/legacy.py:12-30 _old_loader — 0 hits, 0 references outside definition

### Cross-cutting themes
- All error paths in `parser.py` are unexercised.
- No module under `io/` has a single test.
```

The 8-char id next to each finding (`[a3f9c2d1]`) is stable across re-runs
of `/coverage` on the same code. Part 2 uses those ids to track which
findings have already been handled.

### 4. Apply the report

```
/coverage-apply
```

With no arguments, this picks up the newest report in
`.claude/coverage-reports/`. What happens:

1. **Safety checks.** Clean git tree? Test and coverage commands configured?
   Then it runs the test suite once and records baseline coverage on the
   clean tree. If the baseline tests are red, it stops — there's no point
   writing new tests on top of a broken suite.
2. **Batch plan.** Findings are grouped into batches of ≤ 3, by source file.
   Dead-code candidates and skipped findings are dropped. It prints the plan
   and starts applying.
3. **For each batch:**
   - Identifies the test file (`src/pkg/foo.py` → `test/test_foo.py`).
   - Reads the existing test file (if any) to match its style.
   - Writes new tests for each finding in the batch.
   - Runs the test command — green gate 1.
   - Re-runs coverage and verifies every target line is now hit — green gate 2.
   - On both green: commits the batch.
   - On either red: stops immediately.
4. **A sidecar state file** (`<report>.apply-state.json`) records which
   findings are done, skipped, or failed, plus the baseline and current
   coverage percentages. If a batch fails and you fix it by hand, re-running
   `/coverage-apply` picks up where it left off.

### 5. Review the commits

```bash
git log --oneline -20
```

You'll see one commit per batch, each titled something like:

```
coverage-apply: src/mylib/parser.py batch 1 — +4% (62% → 66%)
```

Each commit body lists the specific findings it resolved (symbol, file,
line_range, id). If a batch introduced a flaky test, `git revert <sha>`
reverses just that batch.

---

## Filtering what gets applied

Defaults for `/coverage-apply`:
- severity: `🔴,🟡` — skip nits
- batch-size: `3` — test-writing is denser work than bug-fix edits

Override with flags:

| Flag | Meaning | Example |
| ---- | ------- | ------- |
| `--severity` | Comma-separated severities to include | `--severity 🔴` (critical only) |
| `--only-files` | Restrict to these source files | `--only-files src/mylib/parser.py,src/mylib/io.py` |
| `--batch-size` | Max findings per batch (default 3) | `--batch-size 1` |
| `--max-commits` | Stop after N green commits | `--max-commits 5` |
| `--stop-at-threshold` | Stop once coverage reaches threshold | `--stop-at-threshold` |
| `--dry-run` | Show the batch plan, make no edits | `--dry-run` |
| `--no-verify-coverage` | **Escape hatch** — skip coverage-verification gate (tests still run). Requires interactive confirmation | `--no-verify-coverage` |

You can also pass an explicit report path as the first positional argument:

```
/coverage-apply .claude/coverage-reports/src-mylib-20260423-151200.md
```

Useful when you have multiple reports and want to apply an older one.

---

## Common workflows

**"Fill critical gaps only, then iterate."**

```
/coverage src/mylib
/coverage-apply --severity 🔴
# inspect the commits
/coverage-apply --severity 🟡 --only-files src/mylib/parser.py
```

Each invocation picks up the same report — findings committed in the first
run are marked `done` in the state file and skipped on the second run.

**"I just want the report, I'll write tests by hand."**

```
/coverage src/mylib
# open the .md under .claude/coverage-reports/ and work from there
```

No `/coverage-apply` required.

**"Reach 80% coverage, then stop."**

```
/coverage-apply --stop-at-threshold
```

It halts as soon as the running coverage percent reaches the threshold
configured at init time, even if more findings remain.

**"A batch failed. What now?"**

Part 2 stops on the first red batch and prints:
- what failed (test output or "target lines still missing"),
- which findings were in the failed batch,
- three options:
  1. Fix by hand, then re-run `/coverage-apply <report>` to continue.
  2. Revert (`git restore <files>`) and re-run with `--only-files` adjusted
     to skip the problematic source file.
  3. Mark the failed batch as `failed` in the state file and re-run — those
     findings are then skipped.

The state file survives across runs, so you can stop and resume days later.

**"I want to re-measure after adding tests."**

Just run `/coverage <target>` again. It writes a new timestamped report —
previous reports and state files are kept.

---

## What the coverage-analyst looks at

| Severity | Examples | Why it's ranked there |
| -------- | -------- | --------------------- |
| 🔴 Critical | Public API functions with 0% coverage (exported from `__init__.py`, top-level names without leading underscore) | Highest risk for shipping bugs |
| 🟡 Medium | Internal helpers with at least one caller; unexercised branches inside otherwise-covered functions; validators and error paths | Real logic, some callers, worth testing |
| 🟢 Low | Dunder methods (`__repr__`, `__str__`), trivial getters, logging, defensive branches only reachable on programmer error | Tests here rarely catch real bugs |

Each finding also carries a **why-uncovered hypothesis**:
- `no test exists` — no test file for this module at all
- `test exists, branch unexercised` — `test_<module>.py` exists but only hits
  the happy path
- `fixture gap` — needs a fixture the suite lacks (DB, filesystem, network)
- `possibly dead` — zero hits AND no references elsewhere in the package.
  Flagged separately; **not** used as a test-writing target.

If you want to customize the rubric or the output format, the analyst is a
plain markdown file at `.claude/agents/coverage-analyst.md` — edit it
directly and the next `/coverage` picks up your changes.

---

## Troubleshooting

**"The kit says it's not yet configured."**
Run `/coverage-init`. It fills placeholders once and commits the setup.

**"`/coverage-apply` refuses to run because the tree is dirty."**
Commit or stash. The kit will not mix its edits with yours — that trade-off
is deliberate.

**"Baseline tests fail before any edits."**
Not a kit problem — your suite was already red. Fix it first, then re-run.

**"Coverage command failed."**
Usually means `pytest-cov` isn't installed. `/coverage-init` offers to add
it to your project's dependency file (detects `pyproject.toml` /
`requirements*.txt` / environment file automatically); if you skipped that,
install `pytest-cov` in whichever environment your test command runs in
(e.g. `pip install pytest-cov`, `pixi install`, `poetry install`,
`hatch env prune`, `uv sync`).

**"The analyst tagged something 🔴 but it's actually an internal helper."**
The analyst's public/private split is based on `__init__.py` exports and
underscore conventions. If your project has a different convention, edit
`.claude/agents/coverage-analyst.md` to match — severity assignment is
described in the "Classify" section and easy to adjust.

**"A batch passes tests but coverage didn't move."**
Part 2 catches this and marks the batch failed. Usually the test is hitting
a different code path than intended — e.g. it's asserting on a mock instead
of the real function, or the import path in the test is shadowing the real
module. Inspect the test, fix, re-run.

**"A test I wrote is flaky."**
`git revert <batch-sha>` to back it out, then re-run `/coverage-apply` with
`--only-files` excluding that source file so the problematic finding gets
left alone.

**"I want to stop mid-apply."**
Interrupt Claude Code as usual. The state file is written after every
committed batch, so resuming picks up from the last green commit — the
in-progress batch is not marked `done`, so it re-runs next time.

---

## The setup command (`/coverage-init`)

If you're in a freshly forked template and the `{{...}}` placeholders haven't
been filled yet, run:

```
/coverage-init
```

It will:

1. Detect what's in the repo: package layout under `src/` or top-level;
   pytest configuration; which dependency manager is in use (plain
   `pyproject.toml`, pixi, poetry, hatch, uv, conda, requirements files);
   whether `pytest-cov` is already a dependency; whether any other Claude
   Code kit in the repo has already configured a test command.
2. Ask a short question set: coverage target (if ambiguous), test command
   (confirm or supply), threshold (default 80%), and whether to add
   `pytest-cov` to the detected dependency file (if missing).
3. Fill placeholders in the commands and agent file and commit the setup
   as a single commit.

You can re-edit the files any time after — it's a one-shot bootstrap, not a
lock-in. `/coverage-init` refuses to run a second time against an
already-configured kit, so you can't accidentally overwrite your choices.
