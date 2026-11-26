# HPD / HDR naming consistency — plan (v1, 2026-10-23)

Status: IMPLEMENTED 2026-10-23 (uncommitted), Batches 0-5 + 7; Batch 6 skipped. See §8.

## Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/hpd-hdr-naming-plan.md` and execute it batch by
> batch. Resolve the open decisions in §3 with me first (present them as
> options). Each batch ends with its test gate; stop on red. Do not change
> behaviour beyond what a batch lists. Deprecated names must keep working
> and warn. Memory-cap every pytest run (RLIMIT_DATA in a child + RSS
> watchdog). Never `git add -A` (install-dev rewrites pyproject.toml and
> pixi.lock). Run `pixi run install-dev` after every `src/` edit — the
> package is installed as a copy, not editable.

## 1. State snapshot (at plan time)

- Branch `master`, HEAD `eac27077`, version 0.32.2.
- **The user has UNCOMMITTED edits** in `src/phasic/svgd.py` (+27/−25),
  `src/phasic/__init__.py`, and several tutorial notebooks
  (`svgd-basics`, `svgd-multi-feature`, `svgd-multi-param`,
  `svgd-learning-rate` (renamed), `svgd-priors` (new), `parametrization`,
  `svgd_tests.py`), plus `docs/_quarto-default.yml`. Do NOT stash, reset or
  overwrite these. See decision D1.
- Test suite has pre-existing failures: the gate is "no new failures vs. the
  Batch 0 baseline", never "full suite green".

## 2. Goal and naming rule

Goal: make the names say (a) whether a region is per-parameter (marginal)
or joint, and (b) use one word for "probability mass inside".

Rule:

| term | meaning |
|---|---|
| **HPD** / `hpd` | Highest posterior density *interval*, computed separately for each parameter (marginal, 1-D). |
| **HDR** / `hdr` | Highest density *region*, joint over 2 or more parameters. |
| `level` (`ci_level` when paired with `ci_method`) | Probability mass enclosed, e.g. 0.95. |
| `alpha` | Only ever matplotlib transparency. Never a credible level (in statistics alpha usually means 1 − level). |

### Rename table

All old names stay as deprecated aliases (§4).

| # | where | old | new | kind |
|---|---|---|---|---|
| R1 | `SVGD` method | `plot_hdr` | `plot_pairwise_hdr` | joint region for one pair (others marginalised) |
| R2 | `plot_hdr` kwarg | `alphas` | `levels` | |
| R3 | `plot_hdr` kwarg | `show_hpd` | `show_marginal_hpd` | |
| R4 | `plot_hdr` kwarg | `hpd_alpha` | `marginal_hpd_level` | |
| R5 | `SVGD` method | `estimate_hdr(alpha)` | `joint_hdr_particles(level, unconstrained=False)` | joint over ALL parameters |
| R6 | `plot_ci` kwarg | `ci` | `ci_level` | matches `summary`/`plot_posterior`/`plot.py` |
| R7 | `ci_method` value | `'kde'` | `'kde_hpd'` | `plot_ci` |
| R8 | `ci_method` value | `'percentile'` | `'equal_tailed'` | `plot_ci`, `plot_posterior`, `summary`, `phasic.plot.plot_posterior` |
| R9 | private helper | `_compute_kde_hdr` | `_compute_kde_hpd` | it is per-parameter |
| R10 | private helper kwarg | `_compute_hpd(alpha=)` | `_compute_hpd(level=)` | |
| R11 | `SVGD` method (optional, D4) | `plot_ci` | `plot_ci_trace` | sits next to `plot_trace` |

### Additions and fixes (not renames)

| # | change |
|---|---|
| A1 | New `SVGD.hpd_intervals(level=0.95, unconstrained=False) -> (lower, upper)` and the same on `MCMC` (§5). |
| A2 | `get_results()` (SVGD and MCMC) gains the key `'hpd_level': 0.95`. Additive only. |
| F1 | `phasic/plot.py` drops its own `_compute_hpd` and imports the one in `svgd.py`. **Behaviour change**: the `plot.py` copy has an off-by-one. For n=20 and level=0.95 it spans all 20 samples where `svgd._compute_hpd` spans 19. So MCMC's plotted HPD band currently disagrees with MCMC's own `get_results()['hpd_*']`. See D3. |
| F2 | The MCMC `summary()` header says `HPD 2.5%` / `HPD 97.5%` (`mcmc.py:970`), which is wrong (HPD endpoints are not quantiles). Change it to `HPD 95% lo` / `HPD 95% hi`, as in `SVGD.summary`. |
| F3 | Delete the dead, commented-out `estimate_hdr` contour block in `plot_hdr` (around `svgd.py:8810-8825`). |

## 3. Decisions

**RESOLVED 2026-10-23 by the user:** D1 (a) — the src edits were committed
as `742619e3` + `ad3cc690`, and work starts from `ad3cc690`; D2 (a); D3 (a); D4 (b);
D5 (a); **D6 (b), made stricter: do NOT change ANY notebook**, and also not
`docs/pages/tutorial/svgd_tests.py` (it is generated from notebooks and
carries uncommitted user edits). The aliases keep the notebooks working.

Original options, kept for the record:

- **D1 — the user's uncommitted `src/` edits.** Options: (a) the user
  commits them first and this work goes on a branch from that commit
  (recommended); (b) work on top in the same tree and commit only this
  plan's hunks with `git add -p`.
- **D2 — removal target for deprecated names.** Options: (a) "removed no
  earlier than 0.34.0", stated in every warning (recommended); (b) no
  stated version.
- **D3 — F1 behaviour change** (MCMC plot band narrows by one sample to
  match its own summary). Options: (a) apply it, as a correctness fix
  (recommended); (b) keep `plot.py`'s copy and only rename its kwarg.
- **D4 — rename `plot_ci` → `plot_ci_trace`** (R11). Options: (a) yes, with
  an alias; (b) keep `plot_ci` (recommended: the name is already accurate,
  since CI is the generic term that covers all three methods).
- **D5 — the `plot_ci` default `ci_method`** is `'kde'`, while `summary`
  and `plot_posterior` default to `'hpd'`, so the ribbon's final iteration
  does not match `summary()`. Options: (a) keep `'kde_hpd'` (it was chosen
  deliberately: sample HPD jumps with few particles) and state the
  difference in the docstring (recommended); (b) switch to `'hpd'`.
- **D6 — the docs notebooks** contain the user's uncommitted edits.
  Options: (a) Batch 6 updates the old names in place with NotebookEdit,
  touching only the affected cells (recommended); (b) leave the notebooks
  for the user, since the aliases keep them working.

## 4. Deprecation mechanics (applies to all batches)

- Warning class `DeprecationWarning`, `stacklevel=2`, which is the
  existing idiom (`svgd.py:5318`). Message format:
  `"plot_hdr() is deprecated; use plot_pairwise_hdr(). Will be removed no earlier than 0.34.0."`
  For kwargs: `"plot_ci(ci=...) is deprecated; use ci_level=..."`.
- Renamed **method**: the old name becomes a thin wrapper that warns,
  translates its kwargs, and calls the new method. Its signature and
  docstring stay as they are, with a `.. deprecated::` note added at the top.
- Renamed **kwarg**: keep the old kwarg in the signature with default
  `None`. If the old kwarg is given and the new kwarg differs from its
  default → `TypeError("pass only one of ci / ci_level")`. This follows the
  no-silent-fallbacks rule: never guess which one the caller meant.
  Documented limitation: an explicit new kwarg equal to its default, plus
  the old kwarg, is accepted and uses the old value.
- Renamed **`ci_method` value**: normalise the old value to the new one at
  the top of each function, with a warning. Error messages list only the
  new values.
- **R5 alias semantics:** `estimate_hdr()` returned UNCONSTRAINED particles.
  The alias must call `joint_hdr_particles(level, unconstrained=True)` so
  old callers get identical output. The new method defaults to
  `unconstrained=False`, matching the plots. State this in both docstrings.
- **R9** (private): the module-level `_compute_kde_hdr` becomes a wrapper
  that warns and calls `_compute_kde_hpd`.
- Put one small private helper in `svgd.py` for the kwarg case
  (`_renamed_kwarg(func, old, old_val, new, new_val, new_default)`) so
  the rule is written once.

## 5. New method A1 — `hpd_intervals`

```python
def hpd_intervals(self, level: float = 0.95,
                  unconstrained: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-parameter (marginal) HPD intervals at the given level.

    Returns (lower, upper), each shape (theta_dim,). Same particles and
    algorithm as get_results()['hpd_lower'/'hpd_upper'] (constrained space,
    tied parameters re-tied), so hpd_intervals(0.95) equals those exactly.
    These are separate intervals per parameter, not a joint region; for the
    joint region see joint_hdr_particles().
    """
```

- SVGD: `P = get_results()['particles']`, or `self.particles` if
  `unconstrained=True` (raise a `ValueError` like `plot_posterior` when
  `unconstrained=True` and there is no transform). Loop `_compute_hpd(P[:, i], level=level)`.
- MCMC: same, over the pooled samples that its `get_results` uses.
- Must raise `RuntimeError` if not fitted (same message style as the
  neighbouring methods). Validate `0 < level <= 1` → `ValueError`.
- `get_results` is NOT rewired to call it (that is out of scope); the
  equality test below protects agreement between the two.

## 6. Batches and gates

Every gate runs only the listed files (memory-capped), then compares
against the Batch 0 baseline. New tests go in a new file
`tests/pytest/inference/test_hpd_naming.py` (additive; existing tests are
not edited, so they double as proof that the aliases work).

**Batch 0 — baseline.** Resolve D1–D6. Run `test_hpd.py`,
`test_plot_ci.py`, `test_mcmc.py`, `inference/test_mcmc_accuracy.py`, and
any `test_*plot*` files. Record pass/fail per test in the plan.

**Batch 1 — helpers (R9, R10, F1 if D3a).** `_compute_hpd(samples,
level=0.95, alpha=None)`; `_compute_kde_hpd` + alias; `plot.py` imports
from `svgd`; `mcmc.py` import unchanged.
Gate: baseline files unchanged; new tests: (a) old `alpha=` and
`_compute_kde_hdr` warn and give identical values; (b) both `alpha` and
`level` → `TypeError`; (c) `phasic.plot._compute_hpd is
phasic.svgd._compute_hpd`; (d) n=20 regression for the F1 width.

**Batch 2 — A1, A2.** `hpd_intervals` on SVGD and MCMC; `'hpd_level'` key.
Gate: `hpd_intervals(0.95)` == `get_results()['hpd_lower'/'hpd_upper']`
bit-for-bit (SVGD with and without transform, MCMC); the 0.5 interval
nests inside the 0.95 one; the not-fitted and bad-level errors.

**Batch 3 — `ci_method`/`ci_level` (R6, R7, R8, D5).** `plot_ci`,
`plot_posterior`, `summary`, `phasic.plot.plot_posterior`.
Gate: each old value/kwarg warns and gives an identical figure/printout
(compare the axvspan/fill_between data and the captured stdout); conflicts
→ `TypeError`; `test_plot_ci.py` unchanged and green.

**Batch 4 — `plot_pairwise_hdr`, `joint_hdr_particles` (R1–R5, F3, R11 if
D4a).**
Gate: `plot_hdr(...)` and `plot_pairwise_hdr(...)` with translated kwargs
give identical contour levels and band extents (Agg backend);
`estimate_hdr()` equals `joint_hdr_particles(unconstrained=True)`
exactly; `joint_hdr_particles()` equals the transform applied to that.

**Batch 5 — F2.** MCMC summary header. Gate: capture stdout and check the
labels.

**Batch 6 — docs (D6 = b: NO notebook edits).** Only regenerate the API
reference, if `pixi run api` touches nothing beyond `docs/api/`. The text
below is the superseded (a) variant. Update the call sites (from the grep at plan time):
`plot_hdr` — svgd-multi-param, svgd-joint-prob, popgen/coalescent-derived;
`estimate_hdr` — svgd-multi-param, svgd_tests.py; `show_hpd` —
svgd-multi-param; `plot_ci(ci=...)` and `ci_method` — re-grep at batch
time (the `ci=` grep is noisy). Then regenerate the API reference with
`pixi run api`; never hand-edit `docs/api/*.qmd`.
Gate: re-grep shows no old names outside the alias definitions and their
tests; execute the edited notebooks with `-W error::DeprecationWarning`.

**Batch 7 — review.** Adversarial review subagent: does each alias give
identical output; is any old name missed (`grep -rn` over src, tests,
docs, experiments); does any warning fire from inside phasic itself
(internal calls must use the new names). Then run the final gate against
the Batch 0 baseline.

## 7. Out of scope

- Changing the HPD/HDR algorithms (beyond F1's alignment).
- Making `estimate_hdr`'s ranking a constrained-space density.
- Rewiring `get_results` onto `hpd_intervals`.
- Removing the aliases (that is a later release, per D2).

## 8. Execution record (2026-10-23)

- Base `ad3cc690`. Changed: `src/phasic/svgd.py`, `src/phasic/plot.py`,
  `src/phasic/mcmc.py`; new `tests/pytest/inference/test_hpd_naming.py`
  (34 tests). No notebooks, no `svgd_tests.py`, no existing tests touched.
  Not committed.
- **Baseline (Batch 0):** test_mcmc.py (0 tests, all commented out),
  test_plot_ci.py, test_hpd.py, test_svgd_tied_export.py,
  test_daisy_chain_c_path.py → 68 passed, 1 skipped, 4 FAILED in
  test_plot_ci.py. The failures are pre-existing: `ad3cc690` moved the "MAP ..
  HDR" text from plot_ci's y-label to its title, but the tests still read
  the y-label.
- **Final gate:** the same files + test_mcmc_accuracy.py + the new file → 34
  new passed; old files: the same 4 failures, nothing new. Peak RSS 2.5 GB.
- **Batch 6 skipped:** `pixi run api` runs a full `quarto render`
  (scripts/docs-build-render.sh), which is not confined to docs/api and
  involves the user's uncommitted `_quarto-default.yml`. docs/api/*.qmd
  still show the old names until the next `pixi run api`.
- **Behaviour changes, beyond pure renames** (found partly by the Batch 7
  review):
  1. F1: phasic.plot.plot_posterior uses the svgd HPD helper, so the MCMC
     band is one sample narrower. Like the svgd helper, it now also warns
     when n < 10.
  2. phasic.plot.plot_posterior now raises ValueError on an unknown
     ci_method. Before, anything other than 'hpd' silently became
     equal-tailed.
  3. The plot_ci title for the KDE method says "KDE HPD", not "HDR". The
     already-failing `test_ci_methods[kde-HDR]` would still fail on this
     once it is fixed to read the title.
  4. joint_hdr_particles validates level and keeps at least 1 particle.
     The estimate_hdr alias deliberately does NOT: it reproduces the old
     edge behaviour exactly (alpha < 1/n keeps all particles), as the
     review required.
  5. The MCMC summary header reads "HPD 95% lo/hi".
- **Known residual:** MCMC.plot_posterior(ci_method='percentile') forwards
  **kwargs, so its DeprecationWarning points at mcmc.py instead of the
  user's line. Cosmetic.
- **Fixed on request:** the `_compute_hpd` docstring claimed the HPD
  was "invariant to monotone reparametrisation". That was wrong; the
  equal-tailed interval is the invariant one.
