# coverage-kit

A standalone Claude Code workflow for measuring test coverage of a Python
package and then filling the gaps with new tests, added in small,
test-verified batches.

Designed to drop into any Python template repo: copy the `.claude/` tree
into the template, fork, run `/coverage-init` once, and the workflow is
live. No machine-specific or author-specific state; no required external
tooling beyond `pytest` and `pytest-cov`; works with any Python packaging
/ dependency manager (plain `pip`, `pixi`, `poetry`, `hatch`, `uv`, `conda`).

## Which doc do you want?

- **[USAGE.md](USAGE.md)** — day-to-day use. Walkthrough, flag reference,
  common workflows, troubleshooting. Read this if the kit is already installed
  in a repo you're working in.
- **[TEMPLATE_AUTHORS.md](TEMPLATE_AUTHORS.md)** — how to package the kit
  into your own template repo and customize it for your house style. Read
  this if you maintain a library template.

## The three commands at a glance

| Command                | When to run                | What it does |
| ---------------------- | -------------------------- | ------------ |
| `/coverage-init`       | Once, after forking        | Detects defaults from the repo, asks a few setup questions, fills placeholders. |
| `/coverage <target>`   | Whenever you want a report | Runs coverage measurement, classifies every uncovered region by risk, writes one prioritized report to `.claude/coverage-reports/`. |
| `/coverage-apply`      | After a `/coverage`        | Adds tests for the report's findings in small, test- and coverage-gated batches, one commit per batch. |

## How the two parts fit together

**Part 1 — `/coverage`** spawns a single specialist subagent
(`coverage-analyst`) that runs the configured coverage command, parses the
JSON report, and walks every uncovered region. For each region it assigns a
risk tag (🔴 critical / 🟡 medium / 🟢 low) based on whether the symbol is
public API, has call sites, or is a trivial dunder/getter, and it flags
probable dead code separately. A single prioritized markdown report lands in
`.claude/coverage-reports/`. No files are edited; the report is the artifact.

**Part 2 — `/coverage-apply`** reads that report, groups findings into small
batches (default 3, by source file), and for each batch: writes new tests in
the matching `test/test_<module>.py` file, runs the test suite, re-runs
coverage and verifies the target lines are now hit, commits on green, stops
on red. Designed to be resumable — a sidecar state file lets you pick up
where you left off.

**Both gates are mandatory in Part 2.** Before any edits, it runs the test
suite on the clean tree to confirm a green baseline. After every batch it
runs the suite again AND re-runs coverage to verify the batch's target lines
actually got hit — tests that pass but don't move the needle are caught and
rejected. There's a `--no-verify-coverage` escape hatch that still runs
tests; there's no escape hatch that skips tests.

## Coexisting with other Claude Code kits

coverage-kit is fully self-contained — it ships only files under
`.claude/agents/coverage-*.md` and `.claude/commands/coverage*.md`, plus
its own documentation under `coverage-kit/`. It does not depend on any
other kit being present.

If another kit in the same repo has already filled a `{{TEST_COMMAND}}`
placeholder in one of its own command files under `.claude/commands/`,
`/coverage-init` will detect that value and reuse it, so the user doesn't
answer the same question twice. If no such kit is present, `/coverage-init`
detects the test command fresh from the repo (pytest configuration, packaging
manager, etc.).

**What coverage-kit does not do:**

- It does not edit source code. Part 2 only writes tests.
- It does not diagnose bugs or smells. That's a different workflow.
- It does not run in any test framework other than pytest.

## Kit layout

```
coverage-kit/
├── README.md               (this file)
├── USAGE.md                (user guide — read when using the kit)
├── TEMPLATE_AUTHORS.md     (for template maintainers packaging the kit)
└── .claude/
    ├── agents/
    │   └── coverage-analyst.md
    └── commands/
        ├── coverage-init.md  (setup — run once after fork)
        ├── coverage.md       (Part 1 — measure + classify)
        └── coverage-apply.md (Part 2 — batched test-writing + verify)
```

## Severity rubric (used by the analyst)

- 🔴 **Critical** — Public API (exported in `__init__.py`, or top-level names
  without a leading underscore). The gaps most likely to ship bugs.
- 🟡 **Medium** — Internal helpers with callers; unexercised branches;
  validators and error paths.
- 🟢 **Low** — Dunder methods, trivial getters, logging, defensive branches.

Findings are always anchored at `<file>:<line_range>`.

## Design notes

- The analyst uses `model: sonnet` and tools `Glob, Grep, Read, Bash`. The
  `Bash` permission is needed to run the coverage command and read the
  resulting JSON file; it does not edit.
- Part 1 uses a single specialist. Coverage is one concern measured with
  one tool; there's nothing to parallelize. The orchestrator / specialist
  split is kept anyway so the orchestrator's context stays small.
- Part 2 **batches by source file**, not by line proximity. The test edits
  all go to the matching `test/test_<module>.py`, so grouping by source
  file keeps each commit focused.
- Part 2 never edits source code. If a test can't be written without a
  source change, the finding is marked `skipped` with a note — fix the
  source by hand and re-run.
- `possibly dead` candidates are flagged in the report but **never fed to
  Part 2**. Writing a test for dead code cements it in place; the right
  next step is human review.
- Part 2 commits one batch per commit, so `git log` shows the full trail and
  you can revert any individual batch without unwinding the rest.
