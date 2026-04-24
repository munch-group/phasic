# Shipping coverage-kit with a template repo

This guide is for people who maintain a template repo (e.g. a cookiecutter,
a GitHub "template repository", or a starter fork) and want the coverage-kit
to be available out-of-the-box when someone forks or templates from it.

If you just want to **use** the kit, see `USAGE.md`.

---

## What to ship

Copy the `.claude/` directory from this kit into the root of your template
repo:

```
your-template/
├── ...
└── .claude/
    ├── agents/
    │   └── coverage-analyst.md
    └── commands/
        ├── coverage-init.md
        ├── coverage.md
        └── coverage-apply.md
```

If your template already has a `.claude/` directory (e.g. because it ships
other Claude Code kits), merge — don't overwrite. coverage-kit only adds
files under `agents/coverage-*.md` and `commands/coverage*.md`, so name
collisions with other kits are unlikely as long as those kits don't also
use a `coverage-` prefix.

Also ship `USAGE.md` (so forkers have a user guide), and optionally this
file. The kit's `README.md` is a landing page you can keep, adapt, or drop.

---

## What forkers experience

When someone forks your template and opens Claude Code, they see three slash
commands available: `/coverage-init`, `/coverage`, `/coverage-apply`.

If they run `/coverage` or `/coverage-apply` first, those commands detect
un-instantiated `{{PLACEHOLDER}}` tokens and route them to `/coverage-init`.
So the **expected first interaction** is always `/coverage-init`.

`/coverage-init` does a one-shot setup:

- Detects defaults from the repo: package layout under `src/` or top-level;
  which Python dependency manager is in use (plain `pyproject.toml`, pixi,
  poetry, hatch, uv, conda, `requirements*.txt`); whether `pytest-cov` is
  already a dependency; whether any other kit in the repo has already
  filled in a test command placeholder.
- Asks a short question set (coverage target if ambiguous, test command,
  threshold, whether to add `pytest-cov` to the detected dependency file).
- Fills placeholders in each file and commits the result as a single
  setup commit.

Forkers don't need to read any `.claude/` file by hand unless they want to
customize beyond the init flow.

---

## Coexisting with other Claude Code kits

coverage-kit is fully self-contained. It reads and writes only its own
files under `.claude/agents/coverage-*.md` and
`.claude/commands/coverage*.md`, plus (at init time only, with user
consent) a single existing dependency file to add `pytest-cov`.

If another kit in the same repo exposes a `{{TEST_COMMAND}}` placeholder
that has been filled in, `/coverage-init` will grep for it in
`.claude/commands/*.md` and reuse the value so the forker doesn't answer
the same question twice. If no such kit is present, `/coverage-init`
detects the test command fresh from the repo.

This coexistence is best-effort and unidirectional: coverage-kit reuses
other kits' answers, but does not require them to exist. It also does
not modify any other kit's files.

---

## Customizing the kit for your template

All of these are optional — the stock kit is designed to be useful as-is.

### Pre-instantiate placeholders

If your template always targets the same layout (e.g. every fork has its
package at `src/<name>/`), you can pre-fill `{{COVERAGE_TARGET}}` and
`{{COVERAGE_COMMAND}}` in the kit files before shipping. `/coverage-init`
still runs cleanly — it just has nothing to do for the pre-filled
placeholders.

### Pre-fill the threshold

If your shop has a standard coverage target (e.g. 70% for scientific code
with plotting paths, 85% for core libraries), pre-fill `{{COVERAGE_THRESHOLD}}`
in the kit files and `/coverage-init` will skip asking.

### Adjust severity rubric or scope

The severity rubric (🔴 / 🟡 / 🟢) and `## What to check` section are plain
markdown in `.claude/agents/coverage-analyst.md`. Edit them to match your
house style — e.g.:

- Re-classify module-level code to 🟡 instead of tagging it by default
- Add a fourth severity tier
- Adjust the dead-code detection criteria (e.g. require two missing
  references instead of zero)
- Tighten the `possibly dead` exclusion so fewer symbols get flagged

The orchestrator (`commands/coverage.md`) doesn't know the rubric details;
it just writes the analyst's output to disk. So rubric changes inside the
analyst propagate without any cross-file updates.

### Tighten the coverage command

If your template always uses the same test-runner pattern (e.g. every fork
runs `pixi run test-cov`, or every fork runs `poetry run pytest --cov=...`),
edit `.claude/commands/coverage-apply.md`, `.claude/commands/coverage.md`,
and `.claude/agents/coverage-analyst.md` to replace `{{COVERAGE_COMMAND}}`
(and optionally `{{TEST_COMMAND}}`) with the fixed value before shipping.
`/coverage-init` will skip asking about any placeholder that's no longer a
placeholder.

Do **not** remove the coverage-verification gate logic in Part 2 — the whole
safety story of coverage-kit depends on it. "Tests passed but coverage
didn't move" is the single failure mode that distinguishes real tests from
mocking-through tests.

### Add a specialist

Adding a second specialist (e.g. a C/C++ coverage analyst for mixed-language
projects) follows this convention:

1. Create `.claude/agents/coverage-<name>-analyst.md` with frontmatter:
   ```
   name: coverage-<name>-analyst
   description: "..."
   tools: Glob, Grep, Read, Bash
   model: sonnet
   ```
2. Use the same output format as the default analyst: overall summary,
   severity-tagged findings with stable anchors, per-file summary, dead-code
   section, cross-cutting themes.
3. Add it to the spawn list in `commands/coverage.md`.
4. Make sure `/coverage-init` knows how to detect when it's relevant (e.g.
   presence of `.c` / `.cpp` / `.pyx` files) and either asks or auto-enables.

Specialists are stateless markdown files — there's no registry, no index,
no import plumbing to update.

### Remove the kit for non-Python templates

If your template is for a language coverage-kit doesn't target, just delete
the coverage-kit files from your template's `.claude/` tree. There's no
cross-file dependency to clean up beyond that.

---

## Versioning the kit inside your template

The kit is a set of flat markdown files with no versioning mechanism of its
own. If you want to track which version of the kit your template carries,
the lightest option is to record the upstream commit hash in a comment at
the top of the agent file (or similar) and bump it when you pull in changes.

Forkers inherit whatever version you shipped. There's no auto-update path —
that's on purpose, because the files are meant to be edited per-project.

---

## Testing the kit before shipping

The kit ships with its own verification story: after running
`/coverage-init` on a fresh clone, the following should hold:

```
grep -r "{{" .claude/commands/coverage*.md .claude/agents/coverage-*.md
```

returns nothing — every placeholder has been replaced (except
`{{COVERAGE_SCOPE}}`, which is intentionally left as user-editable
instructions inside the agent file).

Running `/coverage` on the template's placeholder module should produce a
report file under `.claude/coverage-reports/` with at least an overall
coverage percent and a per-file table. Running `/coverage-apply --dry-run`
should print a batch plan without making any edits.

If all three pass, the kit is shippable.
