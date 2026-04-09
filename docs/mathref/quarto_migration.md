# Plan: Convert SOM from .md to .qmd with Quarto Native Theorems

## Context

The SOM (27 files, ~7200 lines) currently uses manually hardcoded numbering for ~168 definitions/theorems and ~154 cross-references in `(Definition 6.3 in [06])` format. Converting to Quarto's native theorem/definition environments with `@label` cross-references would:

- Eliminate manual numbering maintenance
- Enable clickable cross-references in HTML
- Auto-number definitions/theorems per chapter
- Allow adding/removing definitions without renumbering everything

## Key Constraint

The current Quarto setup is `type: website`. **Cross-document theorem references only work in `type: book`**. The SOM must be rendered as a Quarto book (separate from the main website, or as a sub-project).

## Scope Assessment

| Item | Count | Transformation |
|------|-------|---------------|
| Files to rename .md → .qmd | 28 | Mechanical |
| Definition blocks to wrap in `::: {#def-*}` | ~111 | Regex + manual label creation |
| Theorem/Lemma blocks to wrap in `::: {#thm-*}` / `::: {#lem-*}` | ~57 | Regex + manual label creation |
| Proof blocks to wrap in `::: {.proof}` | ~35 | Regex |
| Cross-references to convert | ~154 | Regex `Definition X.Y in [NN]` → `@def-label` |
| Extended Recall blocks to update | ~23 | Manual number → `@label` |
| Section cross-references to convert | ~10–20 | Add `{#sec-*}` labels, update refs |
| Equations with `\tag{N}` to convert to `{#eq-label}` | ~128 | Regex + manual label creation |
| Equation references to convert | ~50–80 | `equation (N)` → `@eq-label` |
| Algorithm headers to wrap in `::: {#alg-*}` (if using extension) | ~40 | Regex |
| 00_index.md registries to update | 1 | Regenerate from labels |

**Estimated total: ~700 individual edits across 28 files.**

## Recommended Approach

### Step 1: Create SOM as Quarto Book Sub-Project

Add a `docs/mathref/_quarto.yml` that defines the SOM as a standalone book:

```yaml
project:
  type: book
  output-dir: _build

book:
  title: "Phasic — Supplementary Online Material"
  chapters:
    - index.qmd           # was 00_index.md
    - 01_preliminaries.qmd
    - 02_graph_representation.qmd
    # ... all 27 files
  
crossref:
  thm-prefix: "Theorem"
  def-prefix: "Definition"
  lem-prefix: "Lemma"
  cor-prefix: "Corollary"
  prp-prefix: "Proposition"
  eq-prefix: "Equation"

format:
  html:
    html-math-method: katex
    theme: [cosmo, ../custom.scss]
```

### Step 2: Rename All Files .md → .qmd

Mechanical: `for f in docs/mathref/[0-2]*.md; do mv "$f" "${f%.md}.qmd"; done`

### Step 3: Convert Definition Blocks

**Before:**
```markdown
**Definition 1.2** (Continuous phase-type distribution). *Let...*
```

**After:**
```markdown
::: {#def-cph}
## Continuous Phase-Type Distribution
*Let...*
:::
```

Each definition needs a unique label. Convention: `#def-{short-descriptive-name}`.

### Step 4: Convert Theorem/Lemma Blocks

**Before:**
```markdown
**Theorem 1.1** (PDF of a continuous phase-type distribution). *Let...*

*Proof.* ... $\square$
```

**After:**
```markdown
::: {#thm-cph-pdf}
## PDF of a Continuous Phase-Type Distribution
*Let...*
:::

::: {.proof}
... $\square$
:::
```

### Step 5: Convert All Hardcoded Cross-References to Quarto `@label` System

**Principle:** No hardcoded numbers should remain for any cross-referenceable entity. Every definition, theorem, lemma, equation, and section that is referenced elsewhere must have a Quarto label, and every reference must use Quarto's `@label` syntax. Quarto handles all numbering automatically — the rendered output will show "Definition 6.3" etc. without any manual number appearing in the source.

**5a. Definition/Theorem/Lemma cross-references:**

| Before | After |
|--------|-------|
| `(see Definition 6.3 in [06])` | `(see @def-vertex-elimination)` |
| `(by Theorem 4.1 in [04])` | `(by @thm-tarjan-correctness)` |
| `(Lemma 8.1 in [08])` | `(@lem-avl-dedup-complexity)` |
| `(Algorithm 5 in [06])` | `(@alg-graph-elimination)` (if using extension) |

This requires building a **mapping table**: every hardcoded `Definition X.Y`, `Theorem X.Y`, `Lemma X.Y` → its new Quarto label. The table must be built before conversion begins and used consistently across all files.

**5b. Extended Recall blocks:** These currently restate a definition from another file in italic with a manual number. After migration, they become standard Quarto cross-references:

| Before | After |
|--------|-------|
| `> **Recall** (Definition 6.1, [06]). *Vertex elimination of $v$...*` | `> **Recall** (@def-vertex-elimination). *Vertex elimination of $v$...*` |

The recall text stays (it serves a pedagogical purpose), but the reference becomes a clickable Quarto link.

**5c. Section cross-references:** Any hardcoded section references like "see Section 2.3" or "in the Definitions section of [06]" should use Quarto section labels where possible:

```markdown
## Definitions {#sec-06-definitions}
```

Referenced as `@sec-06-definitions`. This is lower priority since section references are less common than definition/theorem references, but should be done for any section explicitly referenced from another file.

### Step 6: Convert Equations

**Before:** `$$ f_\tau(t) = \boldsymbol{\alpha} e^{\mathbf{S}t} \mathbf{s} \tag{1} $$`
**After:** `$$ f_\tau(t) = \boldsymbol{\alpha} e^{\mathbf{S}t} \mathbf{s} $$ {#eq-cph-pdf}`

Then references change: `equation (1)` → `@eq-cph-pdf`

Cross-file equation references like `equation (3) in [01]` become just `@eq-cph-pdf` — Quarto resolves the chapter and number automatically.

**No `\tag{}` should remain after migration.** Every numbered equation either gets a Quarto label (if referenced) or loses its number (if never referenced). The notation standard rule "only number equations that are referenced elsewhere" (Section 9.3) maps naturally to "only label equations that are referenced elsewhere."

### Step 7: Handle Algorithms

Quarto doesn't have built-in algorithm support. Options:
1. **Custom numbered blocks extension** (`quarto add ute/custom-numbered-blocks`) — adds `#alg-` prefix support with auto-numbering and `@alg-label` cross-references
2. **Keep algorithms as-is** in code blocks with manual headers (least risk)
3. **Custom Lua filter** — most flexible but most work

Recommend option 1 if the extension works reliably with the book format, option 2 as fallback. If option 2 is used, algorithm numbers remain the only hardcoded numbers in the SOM — document this as a known exception.

### Step 8: Update 00_index.qmd

The global registries become less critical since Quarto auto-generates a table of contents and cross-references are label-based. The index can be simplified.

### Step 9: Update notation_standard.md and _template.md

- Update cross-reference format rules (Section 5 of PRODUCTION_PLAN.md)
- Update _template.md with Quarto div syntax
- Update notation_standard.md Section 9 (numbering) and Section 10 (documentation structure)

## Feasibility Assessment

**This is feasible but large.** The transformation is mostly mechanical (regex-driven) but requires:
1. Creating ~168 unique labels (one per definition/theorem)
2. Building the old-number → new-label mapping table
3. Replacing ~154 cross-references using that table
4. Converting ~128 equation tags
5. Testing the build

**Risk:** If a label is misspelled or a cross-reference mismatched, Quarto will show a broken `@ref` — but these are visible and debuggable at build time.

**Usage estimate:** The conversion is ~700 edits. An agent can do batches of 3-5 files. With 28 files, that's ~6-10 agent calls for conversion + 1-2 for testing. This should be within remaining usage limits if we work efficiently.

## Execution Order

1. Create `docs/mathref/_quarto.yml` book config
2. Build the label mapping table (all 168 definitions/theorems → labels)
3. Convert files in batches of 3-5, starting from 01 (lowest dependencies):
   - Batch A: 01, 02, 03 (foundations — test cross-refs work)
   - Batch B: 04, 05, 06, 07 (core algorithms)
   - Batch C: 08, 09, 10, 11, 12, 13 (symbolic + trace)
   - Batch D: 14, 15, 16, 17 (distribution + sampling)
   - Batch E: 18, 19, 20, 21, 22 (inference)
   - Batch F: 23, 24, 25, 26, 27 (spatial + integration)
   - Batch G: 00_index.qmd (finalize)
4. Test build: `cd docs/mathref && quarto render`
5. Update notation_standard.md and _template.md

## Verification

- `quarto render` in docs/mathref/ produces no cross-reference warnings
- All `@def-*`, `@thm-*`, `@eq-*` references render as clickable links
- Auto-numbering matches the chapter structure (Def 1.1, 1.2, ... in chapter 1; Def 6.1, 6.2, ... in chapter 6)
- **No hardcoded numbers remain** in definition/theorem/lemma headers (grep for `**Definition \d`, `**Theorem \d`, `**Lemma \d` — should return zero)
- **No `\tag{}`** remains in any equation (grep for `\\tag\{` — should return zero)
- **No `in [NN]`** cross-reference format remains (grep for `in \[\d+\]` — should return zero)
- All Extended Recall blocks use `@label` syntax instead of manual numbers

## Critical Files

- `docs/mathref/_quarto.yml` (new)
- `docs/mathref/*.qmd` (28 renamed files)
- `docs/notation_standard.md` (update cross-ref format rules)
- `docs/mathref/_template.md` (update with Quarto div syntax)
- `docs/mathref/PRODUCTION_PLAN.md` (update cross-ref section)
