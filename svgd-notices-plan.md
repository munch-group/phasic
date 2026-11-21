# Plan: short SVGD notices that point into `effective_options`

Status: IMPLEMENTED on branch `fix/epoch-layout-notice` (2026-11-21), awaiting merge. See section 7.

## 1. Goal

The `SvgdAssumptionWarning` notices from `Graph.svgd` (the "phasic picked or
overrode a value for you" messages) print as one very long line in
notebooks. Make each notice one short sentence that names the option(s) it
concerns and says exactly where the full explanation is:
`effective_options('<option>')` on the returned SVGD object. That call is a
new single-option view that prints the full, wrapped, untruncated record of
the option, including the arguments that state the choice explicitly (and so
avoid the notice).

Also tidy the value column of the existing `effective_options()` table:
- `'auto -> probability-Jaco...` becomes `'auto' -> 'jacobian'`;
- raw object reprs such as `<phasic.svgd.GaussPrior object at 0x...>` become
  `svgd.GaussPrior`.

"Done" means:
1. Every notice is at most about 2 lines at 80 columns and ends with the
   exact call to run.
2. That call prints everything the old long notice said, including the
   remedy, wrapped at about 80 columns, with nothing truncated.
3. Nothing in the table is truncated in the cases found so far. Anything
   that still has to be cut is marked and points to the single-option view.
4. Tests pin all of the above.

## 2. Current state (read from source)

- Notices go through one function, `svgd_config.assume()`
  (`src/phasic/svgd_config.py:95`), which calls `warnings.warn`. There are
  three call sites, all in `Graph.svgd` (`src/phasic/__init__.py`):
  - **N1, discrete forced:** "forcing discrete=True: joint-index models read
    converged visit counts ... your discrete=False was overridden."
    (around line 6648)
  - **N2, preconditioner resolved:** "preconditioner='auto' resolves to the
    probability-Jacobian preconditioner ..." (around line 6678)
  - **N3, epoch layout:** prior/fixed expanded per epoch. On the unmerged
    branch `fix/epoch-layout-notice` (c389af2e) it is a long text built by
    `_epoch_layout_notice()`, with the remedy. On master it is the old
    "re-derived for the flat per-epoch layout" text.
- The ledger (`SvgdOptionsLedger`, `svgd_config.py:132`) stores
  `{value, status, user_value, reason}` per option. `render()` prints a
  table, cutting every value at 28 characters (`fmt()`, line 195), and uses
  `repr()`, which gives `<phasic.svgd.GaussPrior object at 0x...>` for
  priors, optimizers and schedules (none of them define `__repr__`).
- The preconditioner entry's value is the display string
  `"auto -> probability-Jacobian"`; the explicit-`'jacobian'` entry (merged
  today, 9b94ded8) is `"jacobian -> probability-Jacobian"`.
- In epoch fits the ledger marks `theta_dim` as `forced (was None; ...)`
  even when the user never passed it. That is misleading: it was inferred,
  not overridden.
- `effective_options(return_dict=True)` is used only by
  `tests/pytest/test_svgd_assumptions.py`: two assertions check
  `'probability-Jacobian' in d['preconditioner']['value']` (lines 141 and
  the test added today).

## 3. Design

### 3.1 Ledger entries carry the long text

Add two optional fields to each ledger entry (additive; `as_dict()` gains
the keys):
- `note`: the full explanation (what was done and why);
- `explicit`: the argument(s) that state this choice explicitly, e.g.
  `fixed=[(1, [0.012, 0.012, 0.012])]`.

Also add an optional `display` string: how the value is shown when the raw
value alone is not informative. This is used for the preconditioner:
- the value is `'jacobian'`;
- the display is `'auto' -> 'jacobian'`.

### 3.2 `effective_options(*names, file=None, return_dict=False)`

- No names: the table as now, with the formatting fixes from 3.3.
- One or more names: for each, a non-tabular block wrapped with `textwrap`
  at 80 columns and never truncated. For example:

  ```
  preconditioner
    value:     'auto' -> 'jacobian'
    status:    inferred (joint-probability model)
    details:   On a joint-probability model 'auto' and 'jacobian' build the
               probability-Jacobian preconditioner from the model's
               probability output; the moment Jacobian would have no effect
               here.
    explicit:  preconditioner='jacobian'
  ```

- An unknown name raises `ValueError` listing the valid names. Names that
  are tracked but have no note print value and status only.
- `return_dict=True` with names returns only those entries.
- The signature change is additive: existing calls `effective_options()`,
  `effective_options(file=...)` and `effective_options(return_dict=True)`
  behave as before.

### 3.3 Value formatting in the table

A formatter used by both views:
- **Objects with Python's default repr**, `<pkg.mod.Class object at 0x...>`,
  show as last-module-component.Class: `svgd.GaussPrior`, `svgd.Adam`,
  `svgd.ExpStepSize`. This applies inside lists and tuples too:
  `[svgd.GaussPrior, None]`.
- **Objects with their own repr** keep it.
- **`display`, if set**, wins.
- **Truncation:** the value column widens to fit the longest value, up to 40
  characters. Anything longer is cut with `...`, and a footer line says
  "values ending in ... are shown in full by effective_options('<name>')".

### 3.4 Notice text

Each notice keeps its warning class and `quiet_assumptions=True` behaviour.
Only the text changes, to `svgd: <one clause>. Details:
.effective_options('<name>'[, ...]) on the returned SVGD object.`

| notice | short text (draft) | names |
|---|---|---|
| N1 | discrete=False was overridden to True for this joint-probability model | `'discrete'` |
| N2 | preconditioner='auto' resolved to 'jacobian' | `'preconditioner'` |
| N3 | prior and fixed were applied to each of the 3 epochs (6 values) | `'prior', 'fixed'` (only those actually expanded) |

The old long texts move into the entries' `note`/`explicit` fields:
- N3's text from `_epoch_layout_notice()` is split per option, so `prior`
  and `fixed` each get their own note and explicit form;
- the per-epoch value-order explanation ("all parameters of epoch 1, then
  epoch 2, ...") goes on `theta_dim`.

N3 keeps the rules from the epoch branch: the notice appears only with 2+
epochs, and only for arguments actually expanded.

### 3.5 Small fix folded in

In epoch fits, record `theta_dim` as `inferred` when the user did not pass
it, and as `forced` only when they passed a different value.

## 4. Batches (test gate between each)

**B1, ledger and view** (`svgd_config.py`, `svgd.py` `effective_options`):
- fields `note`/`explicit`/`display`, the formatter, the single-option view
  and `*names`;
- tests: the formatter (default repr, lists, custom repr, `display`,
  truncation footer); the single-option view (wrapping at 80, no
  truncation, unknown name raises, several names, `return_dict` subset);
- old calls are unchanged.

**B2, notices** (`__init__.py`, three sites):
- short texts; notes/explicit forms recorded on `discrete`,
  `preconditioner`, `prior`, `fixed`, `theta_dim`; the `theta_dim` fix;
- update the two `'probability-Jacobian' in value` assertions to the new
  value `'jacobian'` plus a check of the note;
- tests:
  - each notice is short;
  - each names options whose entries have a note;
  - the epoch notice is absent for 1 epoch and for explicit per-epoch
    forms (kept from the epoch branch).

**B3, docs and gate:**
- docstrings of `effective_options`, `assume` and the `quiet_assumptions`
  parameter;
- one line in `PROJECT-OVERVIEW.md`;
- gate: the svgd assumption/config, epoch, daisy-chain and preconditioning
  test files (the 153 + 31 run today);
- an adversarial review of the diff before asking to merge.

## 5. Branching

Continue from `fix/epoch-layout-notice`, which is not merged. Its long N3
text becomes the `note`s, so merging it on its own first would only ship a
text that changes again. One merge at the end, after you approve.

## 6. Out of scope

- Other phasic warnings and log messages not routed through `assume()`
  (e.g. `logger.warning` in svgd.py, DataPrior messages). Same approach
  could follow later.
- Your notebooks (they call `effective_options()`, which keeps working).

## 7. Outcome (2026-11-21)

B1-B3 were done as planned. An adversarial review found 12 points; the
real ones were fixed and tested:
- values with line breaks (2-D numpy, JAX arrays) broke the table;
- a `__repr__` that raises crashed `effective_options()`;
- the explicit `fixed=` form rounded to 6 digits;
- an explicit `theta_dim` overridden on an epoch fit lost its notice;
- the default epoch prior was recorded as `DataPrior`;
- tuple priors were described as per-parameter lists;
- repeated option names were printed twice.

Left as is:
- the `...` footer can also fire on a value whose own repr contains `...`;
- `effective_options('x')` on an object with no ledger prints the
  no-ledger message instead of rejecting the name.

Found but out of scope: a tuple prior on an epoch fit fails with "Prior at
index 0 does not have a sample() method", because the epoch builder only
accepts lists. That was already the case.

Gate: 243 passed, 0 failed (the svgd, epoch, daisy-chain, preconditioning
and prior test files).
