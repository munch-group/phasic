---
description: Part 2 of the coverage kit. Reads a coverage report produced by /coverage, groups its findings into small batches, and for each batch writes tests, runs the test suite, verifies the target lines are now covered, commits on green, stops on red. Resumable.
argument-hint: [report-path] [--severity 🔴|🟡|🟢] [--only-files src/pkg/foo.py,...] [--batch-size 3] [--max-commits N] [--stop-at-threshold] [--dry-run] [--no-verify-coverage]
---

# /coverage-apply

You are executing **Part 2** of the coverage kit. Your job is to take a report
written by `/coverage` and turn its findings into **new tests**, added in
**small, test- and coverage-verified batches**, committing each successful
batch to git so the trail is visible and any batch can be reverted
independently.

**You edit test files only. You do not edit source files.** If a finding
points at a suspected bug in the source, flag it and skip — this kit is for
locking in existing behavior with tests, not for fixing source code.

## Configuration — fill these at template-instantiation time

- `pixi run -- pytest -x -q tests/pytest/` — **required**. e.g. `pytest -x -q`, `pixi run test`,
  `poetry run pytest -x -q`, `hatch run test`, `uv run pytest -x -q`.
  Used as the green gate after each batch. If still the literal placeholder,
  stop at Step 1 and route the user to `/coverage-init`.
- `pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/` — **required**. e.g.
  `pytest --cov=src/mylib --cov-report=json:.coverage.json`. Used to verify
  that each batch's target lines actually get hit. Same rule: if still the
  placeholder, route to `/coverage-init`.
- `src/phasic` — the package path the report covers.
- `80` — integer percent. Used only by `--stop-at-threshold`.

## Step 1: Safety and inputs

1. **Working tree must be clean.** Run `git status --porcelain`. If anything is
   listed other than the report file itself, stop and ask the user to commit
   or stash first. Reason: batched auto-edits mixed with in-progress work is
   a recipe for lost changes.

2. **Resolve the report path** from `$ARGUMENTS`:
   - If a path is given, use it.
   - If none, pick the newest file matching `.claude/coverage-reports/*.md`.
   - If none exists, tell the user to run `/coverage` first and stop.

3. **Parse flags** from `$ARGUMENTS`:
   - `--severity <levels>` — comma-separated subset of `🔴,🟡,🟢`. Default: `🔴,🟡`.
   - `--only-files <paths>` — comma-separated source file paths; restrict
     to findings in these files.
   - `--batch-size <n>` — cap findings per batch. Default: 3 (test-writing
     is denser than review fixes; keep batches small).
   - `--max-commits <n>` — safety cap for unattended runs; stop after N
     green commits even if work remains. Default: unlimited.
   - `--stop-at-threshold` — halt once overall coverage reaches
     `80%`.
   - `--dry-run` — print the batch plan without editing.
   - `--no-verify-coverage` — **escape hatch only.** Skip the per-batch
     coverage-verification step (still runs `pixi run -- pytest -x -q tests/pytest/`). Requires
     explicit acknowledgement: if passed, print a one-line warning and ask
     the user to confirm with "yes, apply without coverage verification"
     before proceeding. Commits land with a `[no-verify-coverage]` marker.

4. **Check commands are configured.** If `pixi run -- pytest -x -q tests/pytest/` or
   `pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/` is still the literal placeholder, stop with:
   > "Tests and coverage measurement are the safety gates for
   > /coverage-apply. Run `/coverage-init` to configure them for this repo."

5. **Baseline.** Before any edits:
   - Run `pixi run -- pytest -x -q tests/pytest/` once on the clean working tree. If it fails, stop —
     there is no point batching tests on top of a red suite, since we can't
     tell whether a batch made things worse.
   - Run `pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/` and parse the resulting JSON. Record the
     overall percent coverage as `baseline_coverage`.

6. **Load apply-state.** Sidecar file `<report-path>.apply-state.json`. Schema:
   ```json
   {
     "baseline_tests": "passed|failed",
     "baseline_coverage": 62.3,
     "current_coverage": 62.3,
     "findings": {
       "<finding-id>": {
         "status": "done|skipped|failed|pending",
         "commit": "<sha>",
         "note": "..."
       }
     }
   }
   ```
   Finding-ids come from the report (8-char hex that `/coverage` assigned).
   If the file does not exist, treat every finding as `pending` and set
   `current_coverage = baseline_coverage`.

## Step 2: Read and parse the report

Read the report file. Extract each finding into a structured record:

```
{
  id: <8-char id from the report>,
  severity: 🔴 | 🟡 | 🟢,
  file: <source file path>,
  line_range: <e.g. "45-78">,
  symbol: <function/method name>,
  signature: <one-line>,
  kind: function | branch | error-path | module-top-level | dunder,
  why_uncovered: <hypothesis>,
  suggested_test: <one sentence>
}
```

- **Drop** findings whose status in apply-state is already `done` or `skipped`.
- **Drop** any finding from the `Dead-code candidates` section — those are
  not for testing, regardless of filters.
- Apply the severity / `--only-files` filters from Step 1.

## Step 3: Batch the filtered findings

Group into batches with these rules (in order):

1. **Same source file first.** Findings from `src/pkg/foo.py` share a test
   file (`test/test_foo.py`) and share fixture needs — batch them together.
2. **Within a file, order by line_range ascending** so the test file grows
   in a readable order.
3. **Cap at `--batch-size` findings per batch** (default 3).
4. **Order batches across files by severity** — all 🔴 batches first, then
   🟡, then 🟢 — and by file within each severity tier.

Produce a **batch plan** and show it to the user:

```
Baseline coverage: 62% (threshold: 80%)
Batch plan (N batches, M findings):
  Batch 1 [🔴] src/mylib/parser.py — 3 findings (parse_config, validate, _tokenize)
  Batch 2 [🔴] src/mylib/io.py     — 2 findings (load_json, save_json)
  Batch 3 [🟡] src/mylib/parser.py — 3 findings (error branches)
  ...
```

If `--dry-run`, stop here.

## Step 4: Apply batches one at a time

For each pending batch, in order:

1. **Announce the batch.** One line: batch number, severity, source file,
   finding count.

2. **Identify the test file.** Convention: source `src/<pkg>/<mod>.py` →
   tests in `test/test_<mod>.py`. If the test file exists, `Read` it to
   understand existing fixtures and style. If not, create it.

3. **Read the source region(s)** you need to write tests against. Only the
   lines the findings point at — don't blindly reload the whole file.

4. **Write the tests.** For each finding in the batch:
   - Use the analyst's `suggested_test` as a starting point.
   - Match the project's existing test style (fixtures, naming, pytest
     conventions) based on what you read in step 2.
   - Each test should have a clear, independent purpose. One test per
     finding where reasonable; if two findings share setup, one test
     exercising both is acceptable.
   - **Do not edit the source file.** If the test can't be written without
     a source change (e.g. the function has no way to be called in
     isolation), mark the finding `skipped` with a note explaining why and
     move on.
   - **Do not mock the thing under test.** Mock external resources (HTTP,
     filesystem, DB), never the code the test is supposed to exercise.

5. **Run the tests — mandatory gate.** Run `pixi run -- pytest -x -q tests/pytest/` on the edited
   tree. On failure: go to step 8 (fail path).

6. **Verify coverage — mandatory gate** (unless `--no-verify-coverage`
   confirmed in Step 1). Run `pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/`, re-parse the JSON, and
   check that every finding's `line_range` is now outside the new
   `missing_lines` for its file. If any target line is still missing, the
   tests didn't actually exercise what they claimed to — go to step 8.

   Also record the new overall coverage percent. Compute the delta vs.
   `current_coverage`.

7. **On both gates green: commit.** Stage only the test files you touched
   (never `git add -A`) and commit with a HEREDOC message:

   ```
   coverage-apply: <source-file> batch <N> — +<Δ>% (<old>% → <new>%)

   Added tests for findings from <report-path>:
   - <symbol> (<source-file>:<line_range>) [id <id>]
   - <symbol> (<source-file>:<line_range>) [id <id>]
   ...

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

   If `--no-verify-coverage` was used, add `[no-verify-coverage]` to the
   subject line.

   Update apply-state:
   - Each finding in the batch → `status: "done"`, `commit: <new sha>`.
   - `current_coverage` → the new percent from step 6.
   Write the state file.

   If `--max-commits` reached, stop with a summary (don't mark remaining
   batches as failed). If `--stop-at-threshold` and
   `current_coverage >= 80`, also stop with a summary.

8. **On failure: stop.** Do not try another batch. Do not auto-revert — the
   user may want to inspect and fix by hand. Report:
   - Which batch failed and which gate (tests red, or coverage unchanged).
   - For test failures: first ~50 lines of output.
   - For coverage failures: which target lines are still missing.
   - Path to the report and the state file.
   - Three options for the user:
     - "Fix the tests manually, then re-run `/coverage-apply <report>` to
       continue."
     - "Revert this batch: `git restore <files>` then re-run with
       `--only-files` adjusted to skip this source file."
     - "Mark the batch as `failed` and continue: re-run — failed findings
       are skipped."

   Update apply-state for each finding in the failed batch → `status: "failed"`,
   with a short `note`.

## Step 5: Wrap up

After all batches are done (or on a stop), summarize:

- Coverage: `baseline_coverage`% → `current_coverage`% (Δ +N%).
- Batches applied / skipped / failed.
- Commits created (SHAs and titles).
- Findings still pending.
- The report path and state file path, so the user can resume.

## Rules

- **Tests and coverage both gate every batch.** Baseline tests must be green
  before any edits. Tests must be green after every batch AND the target
  lines must now be covered, before the commit lands. No exceptions unless
  the user explicitly opted in with `--no-verify-coverage` (and even then,
  tests still gate).
- **Never touch source files.** This command adds tests. Source changes —
  including "obvious" tweaks to make code testable — are out of scope.
  Handle those by hand or with a separate code-review workflow.
- **Never test `possibly dead` candidates.** They're flagged separately in
  the report for human review; writing a test for them cements dead code.
- **One batch, one commit.** Never squash. Never amend. If you need to undo,
  that's `git revert <sha>`.
- **Never bypass hooks or signing.** No `--no-verify`. If a pre-commit hook
  fails, treat it like a test failure.
- **Don't mock the code under test.** Mock external resources only.
- **Match the project's test style.** Read the existing test file (if any)
  first; the new tests should look like they belong next to the old ones.
- **State file is source of truth for resumability.** Update it after every
  commit and at every stop.
