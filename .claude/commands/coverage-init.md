---
description: One-time setup for the coverage kit in a freshly forked template repo. Detects un-instantiated placeholders across the coverage agent and commands, asks the user the minimum set of questions, and writes the answers back into the relevant files.
argument-hint: (no arguments)
---

# /coverage-init

You are bootstrapping the coverage kit into this repository. The kit ships with
`{{PLACEHOLDER}}` tokens that must be filled before `/coverage` and
`/coverage-apply` will work. Your job is to do that, once, with as few
questions as possible.

## Step 1: Detect the current state

Run these greps and report the counts:

```
grep -l "pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/"    .claude/commands/coverage*.md .claude/agents/coverage-*.md
grep -l "pixi run -- pytest -x -q tests/pytest/"        .claude/commands/coverage-apply.md
grep -l "src/phasic"     .claude/commands/coverage*.md .claude/agents/coverage-*.md
grep -l "80"  .claude/commands/coverage*.md .claude/agents/coverage-*.md
grep -l "{{COVERAGE_SCOPE}}"      .claude/agents/coverage-*.md
```

If every placeholder is already replaced, tell the user the kit is already
configured and stop — re-running init would overwrite their choices.

## Step 2: Detect sensible defaults

Before asking, try to detect each answer from the repo. **Do not assume any
particular packaging or dependency manager** — detect what the repo uses and
adapt. If you can't detect, ask; don't guess.

1. **Coverage target.** In order of preference:
   - A single directory under `src/` containing `__init__.py` → that's the
     default `src/phasic`.
     If multiple, list them and ask the user to pick.
   - No `src/` layout → any top-level directory containing `__init__.py`
     whose name isn't `test`, `tests`, `docs`, or similar.
   - If still ambiguous, ask the user for the package path.

2. **Test command.** Detect, in this order of specificity:
   - If **any other kit** in this repo has already filled a `pixi run -- pytest -x -q tests/pytest/`
     placeholder in a file under `.claude/commands/`, reuse that exact value
     verbatim — forkers shouldn't answer the same question twice. Find it
     with: `grep -rh "^- \`pixi run -- pytest -x -q tests/pytest/\`" .claude/commands/ 2>/dev/null`
     is not useful; instead grep for lines that look like a configured test
     command (e.g. look for `pytest`, `pixi run`, `poetry run pytest`,
     `hatch run`, `uv run pytest`, `npm test` in the commands directory
     where they're clearly documented as the filled-in test command). If in
     doubt, ask the user to confirm.
   - `pixi.toml` or `[tool.pixi.tasks.test]` in `pyproject.toml`
     → `pixi run test`
   - `[tool.poetry]` in `pyproject.toml` → `poetry run pytest -x -q`
   - `[tool.hatch.envs.*.scripts]` with a `test` script in `pyproject.toml`
     → `hatch run test`
   - `uv.lock` present → `uv run pytest -x -q`
   - `conda` / `environment.yml` present → `pytest -x -q` (user runs in
     their activated environment)
   - Plain `pyproject.toml` with pytest as a dev dep, or any `conftest.py`,
     or a `tests/` or `test/` directory → `pytest -x -q`
   - Otherwise → ask the user, no default.

3. **Coverage command.** Detect whether `pytest-cov` is already available by
   grepping the dependency declarations relevant to the detected ecosystem
   (only inspect what exists):
   - `pyproject.toml` — any of `[project.optional-dependencies]`,
     `[tool.poetry.group.*.dependencies]`, `[tool.hatch.envs.*.dependencies]`,
     `[tool.pixi.dependencies]`, `[tool.pixi.pypi-dependencies]`
   - `requirements*.txt`, `requirements/*.txt`
   - `environment.yml`, `pixi.lock`, `poetry.lock`, `uv.lock`
   - `setup.cfg` / `setup.py` (legacy)

   Default command (always usable once `pytest-cov` is installed):
   ```
   pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json
   ```

   If the detected test command wraps pytest in a runner (e.g. `pixi run`,
   `poetry run`, `hatch run`, `uv run`), offer a wrapped variant so the
   coverage command uses the same environment:
   - `pixi run test` → `pixi run -- pytest --cov=src/phasic --cov-report=json:.coverage.json`
   - `poetry run pytest ...` → `poetry run pytest --cov=src/phasic --cov-report=json:.coverage.json`
   - `hatch run test` → `hatch run pytest --cov=src/phasic --cov-report=json:.coverage.json`
   - `uv run pytest ...` → `uv run pytest --cov=src/phasic --cov-report=json:.coverage.json`
   - plain pytest → use the default above.

   If `pytest-cov` is **not** detected anywhere, offer to add it. This is
   the single edit to a non-kit file that `/coverage-init` is allowed to
   make, and **only** to the dependency file for the ecosystem actually in
   use — see Step 4. If the user declines, leave the default command in
   place and tell them they need to install `pytest-cov` themselves before
   `/coverage` will work.

4. **Coverage threshold.** Default to `80`. If `pyproject.toml` has
   `[tool.coverage.report] fail_under = N`, use that instead.

## Step 3: Ask the minimum set of questions

Ask the user — in a single batched question block — only for what couldn't be
detected, plus confirmation of any auto-detected defaults that affect more
than one file. Typically this is:

- **Coverage target** (if ambiguous or not detectable)
- **Test command** (confirm detected value, or ask if none detected)
- **Coverage threshold** (offer the default, `80`)
- **Add `pytest-cov` to dependencies?** (if missing — specify which file
  you'd add it to, based on the detected ecosystem)

Do NOT ask about `{{COVERAGE_SCOPE}}` — that's per-target and better set
lazily when `/coverage` is first run on a given package. The block in the
agent file stays as the "describe scope here" instruction.

## Step 4: Write the answers

In the three command files and the agent file:

- Replace `pixi run -- pytest --cov=src/phasic --cov-report=term-missing --cov-report=json:.coverage.json tests/pytest/` with the final, literal shell command
  (with `src/phasic` already expanded to the literal path — these
  are shell commands, not templated strings).
- Replace `pixi run -- pytest -x -q tests/pytest/` with the final, literal shell command.
- Replace `src/phasic` with the literal package path.
- Replace `80` with the integer (no `%` sign).
- Leave `{{COVERAGE_SCOPE}}` as-is.

Use `Edit` with `replace_all: true` for placeholders that appear multiple
times in a single file.

If the user said yes to adding `pytest-cov`, add it to the **single
dependency file that matches the ecosystem you detected** in Step 2. Use
one of (in priority order, first match wins):

- `pyproject.toml` has `[tool.pixi.dependencies]` → add `pytest-cov = "*"` there
- `pyproject.toml` has `[tool.poetry.group.dev.dependencies]` (or
  `[tool.poetry.dev-dependencies]` on older layouts) → add
  `pytest-cov = "*"` there
- `pyproject.toml` has `[tool.hatch.envs.default.dependencies]` or
  `[tool.hatch.envs.test.dependencies]` → add `"pytest-cov"` to that array
- `pyproject.toml` has `[project.optional-dependencies]` with a `test` or
  `dev` group → add `"pytest-cov"` to that group
- `requirements-dev.txt` or `requirements/dev.txt` exists → append `pytest-cov`
- No recognizable dependency declaration exists → **do not edit anything
  else.** Tell the user they'll need to add `pytest-cov` to whatever
  install flow they use, and continue with placeholder-fill only.

Never add the dependency to more than one file. Never create a new
dependency file.

## Step 5: Commit

Stage only the files you changed and commit:

```
chore: instantiate coverage-kit for this repo

Coverage target: <target>
Coverage command: <cmd>
Test command: <cmd>
Threshold: <N>%

Co-Authored-By: Claude <noreply@anthropic.com>
```

If the repo is not a git repo, skip the commit and tell the user.

## Step 6: Next steps

Tell the user:
- `/coverage <optional target>` to measure coverage and generate a report.
- `/coverage-apply` to add tests in test-verified batches.
- They can re-edit the agent file any time — this was a one-shot bootstrap,
  not a lock-in.
- If `pytest-cov` was just added to a dependency file, they need to
  install it before the first `/coverage` run — e.g. `pixi install`,
  `poetry install`, `hatch env prune`, `uv sync`, `pip install -e .[test]`,
  or whatever matches the ecosystem you detected.

## Rules

- **Ask as little as possible.** Detect defaults from the repo before asking.
- **Don't invent placeholders the kit didn't ship with.** Only the five
  listed above exist (`COVERAGE_COMMAND`, `TEST_COMMAND`, `COVERAGE_TARGET`,
  `COVERAGE_THRESHOLD`, `COVERAGE_SCOPE`).
- **Never overwrite an already-configured kit.** Step 1 is the guard.
- **Prefer leaving `{{COVERAGE_SCOPE}}` untouched** over asking the user to
  enumerate scope they haven't thought about yet. The first `/coverage` run
  is a better time for that.
- **Reuse any already-filled `pixi run -- pytest -x -q tests/pytest/` in `.claude/commands/`.**
  If another kit in this repo has already configured the test command, read
  and reuse that value — one question answered twice in one session is a
  bug. If nothing like that is present, detect from the repo as described
  in Step 2.
