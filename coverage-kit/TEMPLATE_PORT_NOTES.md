# Notes to port back to template repo

Running log of edits I had to make to `.claude/agents/coverage-*.md` or `.claude/commands/coverage*.md` while using the kit in phasic. Port these back to the template source.

## 2026-04-24 — guard grep in `.claude/commands/coverage.md`

**What:** The Step 0 guard grep in `coverage.md` was a one-liner:
```
grep -l '{{COVERAGE_COMMAND}}\|{{COVERAGE_TARGET}}\|{{COVERAGE_THRESHOLD}}' \
    .claude/commands/coverage*.md .claude/agents/coverage-*.md
```
Its own pattern string contained the `{{…}}` tokens, so even on a fully-instantiated kit, `grep` matched `coverage.md` itself on line 18 and reported the kit as unconfigured.

**Fix applied:** Split the braces via shell vars so the pattern literal no longer contains the token:
```
L='{{'; R='}}'
grep -l "${L}COVERAGE_COMMAND${R}\|${L}COVERAGE_TARGET${R}\|${L}COVERAGE_THRESHOLD${R}" \
    .claude/commands/coverage*.md .claude/agents/coverage-*.md
```

**Why this matters for the template:** Every repo that runs `/coverage-init` hits this same self-match. Suggest either this shell-var split, adding `--include` exclusions for `coverage.md`, or using a different sentinel in the guard.

## 2026-04-24 — ambiguity in `.claude/commands/coverage-init.md` Step 4

**What:** Step 4 says "In the three command files and the agent file", then lists `{{COVERAGE_COMMAND}}`, `{{TEST_COMMAND}}`, `{{COVERAGE_TARGET}}`, `{{COVERAGE_THRESHOLD}}`. But the three command files include `coverage-init.md` itself, which contains the same tokens in documentation/code blocks describing what to fill. Filling them there mangles the docs; *not* filling them breaks the Step 0 guard in `coverage.md` because the guard globs `coverage*.md`.

**Fix applied:** Filled the tokens in `coverage-init.md` too. After my fix to the Step 0 guard (above), this is less load-bearing, but the instruction in the init command is still confusing.

**Suggestion for the template:**
- Either rename the init file (e.g. `coverage-setup.md`) so the `coverage*.md` glob doesn't include it, or
- Add an explicit exclusion of `coverage-init.md` from the Step 0 guard, or
- Clarify in Step 4: "Fill placeholders in the three runtime files (coverage.md, coverage-apply.md, coverage-analyst.md). Leave `coverage-init.md` alone — it describes the init process."

## 2026-04-24 — `tests/pytest/` vs `test/` layout

**What:** coverage-apply.md:146–148 hard-codes the test-file convention: `src/<pkg>/<mod>.py → test/test_<mod>.py`. In phasic, tests live in `tests/pytest/`. The kit should infer this from `[tool.pytest.ini_options] testpaths`.

**Fix applied:** None yet — haven't run `/coverage-apply` so haven't hit this path. Flagging for when I do, or for template authors to generalize.

**Suggestion for the template:** Detect `testpaths` during `/coverage-init` and add a `{{TEST_DIR}}` placeholder that `/coverage-apply` uses to form the destination test file.
