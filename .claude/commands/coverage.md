---
description: Part 1 of the coverage kit. Runs coverage measurement on a target package via the coverage-analyst subagent, then writes a single prioritized report to .claude/coverage-reports/ classifying every uncovered region by risk.
argument-hint: [optional: path/to/target/package — defaults to src/phasic]
---

# /coverage

You are the orchestrator of the coverage kit's **Part 1**. You do NOT measure
coverage yourself or read source files. Your only job is to spawn the
`coverage-analyst` specialist, collect its report, and write it to disk in a
consistent location.

## Step 0: Check the kit is instantiated

Run:

```
L='{{'; R='}}'
grep -l "${L}COVERAGE_COMMAND${R}\|${L}COVERAGE_TARGET${R}\|${L}COVERAGE_THRESHOLD${R}" \
    .claude/commands/coverage*.md .claude/agents/coverage-*.md
```

(allow failure). If any file still contains any of these placeholders, the
kit has not been instantiated yet. Stop and tell the user:

> "This repo's coverage kit is not yet configured. Run `/coverage-init` first —
> it detects sensible defaults from the repo and asks a small set of setup
> questions."

## Step 1: Resolve the target

- If `$ARGUMENTS` is given, use it as the coverage target (overrides the
  baseline for this run).
- Otherwise, use the `src/phasic` baseline that `/coverage-init`
  wrote into the agent and command files.
- Run `ls "<target>"` to verify it exists. If missing, stop and tell the user.
- Compute a **slug** = target path with `/` and `.` replaced by `-`
  (e.g. `src/mylib` → `src-mylib`).
- Compute a **timestamp** = `YYYYMMDD-HHMMSS` from `date +%Y%m%d-%H%M%S`.
- Report path = `.claude/coverage-reports/<slug>-<timestamp>.md`. Ensure the
  directory exists (`mkdir -p .claude/coverage-reports`).

## Step 2: Spawn the coverage-analyst

Make one `Agent` call to `coverage-analyst`. The prompt should be minimal —
the agent's system prompt already contains the measurement command, scope,
and output format. A suitable prompt:

> "Measure coverage for `<target>` per your configured scope. Run the
> configured coverage command, parse the JSON, classify every uncovered
> region by risk, and return the structured report in the exact format your
> system prompt specifies."

Do not pass extra context. Do not re-run the coverage command yourself.

If the coverage run itself fails (the agent returns a "coverage run failed"
single-finding report), write that to the report file anyway and tell the
user to fix the underlying test/coverage-tool issue — there is nothing useful
to aggregate.

## Step 3: Write the report

Take the analyst's output and write it verbatim into the report path from
Step 1, prefixed with this header:

```
# Coverage — <target>

_Generated <timestamp> by /coverage. Threshold: 80%._

```

Then append the analyst's full output.

After the analyst content, append this footer:

```

---

_To add tests for these findings in test-verified batches, run:
`/coverage-apply <path-to-this-report>`._
```

## Step 4: Assign stable finding IDs

For each finding in the report, compute a stable `id` = first 8 chars of
SHA1(`<file>:<line_range>:<symbol>`). Insert the id at the start of each
finding line so `/coverage-apply` can reference it in the state file:

```
[🔴] [<id>] <symbol> — <file>:<line_range>
```

IDs are stable across re-runs of `/coverage` on the same code — important
for resumability.

## Step 5: Report back to the user

Tell the user:
- Where the report was written.
- Overall coverage % and the threshold.
- How many findings total, split by severity.
- Count of `possibly dead` candidates (flagged separately — these are not
  for testing).
- That Part 2 is `/coverage-apply <report-path>` (or just `/coverage-apply`
  to auto-pick the newest report).

## Rules

- Do **not** run the coverage command yourself. The analyst does that work.
- Do **not** invent findings. You are a writer-to-disk, not a reviewer.
- Do **not** re-classify the analyst's severities.
- **Do** assign stable finding IDs so `/coverage-apply` can track state.
- **Do** preserve `possibly dead` candidates as a separate section —
  `/coverage-apply` must not treat them as test-writing targets.
