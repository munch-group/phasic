# Bug 2: fixed parameter values are not delivered to the model as given — fix plan

**Status: v3, 2026-11-23 — SIGNED OFF by all three reviewers (reviewers 1 and 3 on v2; reviewer 2 conditional on two test rewordings, applied here as written, section 8). v1 was reviewed by three independent agents
(diagnosis/scope; fix design; tests/impact). All confirmed the diagnosis
and the wrapper design (prototyped by two of them) and required the
changes made here (section 8). Branch `fix/fixed-value-under-positive`,
worktree `/Users/kmt/phasic-bug2`. No source changed yet. Backward
compatibility is not required (user decision).**

**Prerequisite (correctness, not convenience): this fix is implemented and
gated only on top of the bug-1 fix (no shared compiled gradients,
`fix/svgd-compiled-cache`).** Today the fixed values reach the compiled
gradient as runtime data; under this plan they are constants of the traced
closure, so with the old `id(model)`-keyed cache a second fit on the same
model object would run with the FIRST fit's fixed values (measured on a
prototype: model saw [0.3, 2.0] while `theta_mean` reported 2.0). The
branch is rebased onto master after bug 1 merges; implementation starts
after the rebase.

Evidence: `experiments/bug2_fixed_values/` (`b1`-`b3`; reviewers'
scripts/logs and wrapper prototypes in `review1/`, `review2/`, `review3/`).

## 1. The defect (read from source + measured, unmodified code, fresh model per fit)

`SVGD.__init__` (`src/phasic/svgd.py:6149-6236`) and `MCMC.__init__`
(`src/phasic/mcmc.py:321-356`) store a fixed value in the UNCONSTRAINED
space — `log(exp(v) - 1)` when `positive_params and v > 0`, otherwise the
raw value — and the model later receives `param_transform(stored)`. The
round trip is not the identity:

| given | setting | model receives |
|---|---|---|
| 0 | positive_params (default) | **0.693147** |
| <= 1e-10 | positive_params | **1e-9** (transform floor) |
| 1e-8 .. 1e-3 | positive_params | wrong in the 8th-9th digit (100% of values) |
| 1e-3 .. 1 | positive_params | off by rounding (88% of values; e.g. 0.3 → 0.30000000000000004) |
| >= 710 | positive_params | **inf** (a real model then fails validation; a permissive one runs at inf) |
| -0.5 | positive_params | **0.474** |
| 2.0 / 0 | custom `param_transform=exp` | **7.389** / **1.0** |
| binary mask `[0, 1]` | custom `param_transform=exp` | **2.718** (documented as "fixed at 1.0") |
| 0.25 | `[BetaPrior, None]`, slot 1 fixed | 0.24999999999999994 |
| any | no transform | exact |

Paths that inherit the distortion:

- **Exposure on a fixed parameter** (the common case: mutation rate fixed,
  exposure on it): the floor is applied BEFORE scaling, so
  `fixed=(1, 1e-10)` with alpha = [1, 10, 100, 1000] reaches the model as
  [1e-9, 1e-8, 1e-7, 1e-6] — 10x wrong per observation; fixed at 0 it is
  alpha x 0.693. Same on the epoch path (scaling inside the model after
  the transform).
- **User-supplied `theta_init`**: the fixed columns are only set when
  `theta_init is None` (`svgd.py:6187, 6226`); with a user column 3.0 and
  `fixed=(1, 0.25)` the constructor's probe evaluates the model at 3.0486
  and `history[0]` reports it.
- **`plot_pairwise_hdr`** re-inverts the constrained mean with a floor of
  1e-6 (`svgd.py:9341-9345`): a value fixed at 1e-8 is evaluated at 1e-6.
- **Likelihood-ratio guard**: `model_selection._check_nested_fixed`
  (`model_selection.py:218-245`) compares the stored unconstrained values;
  800 vs 900 is ACCEPTED today (both inf).
- **Tied epoch parameters** lock slave slots with the placeholder value 0.0
  through this mechanism; the model input there is 0.693 before the model
  overwrites it with the master's value. Verified harmless: `_apply_tying`
  is the first operation in the model; output is bit-identical for a slave
  input of 0, 0.693, 123, -5 or NaN.

Already exact (no change needed): `Graph.svgd`'s flattening of per-epoch
`fixed`; `FreeEpochModel.fit`'s merge; `DataPrior`, `method_of_moments`,
`probability_matching`, the default epoch prior (they use the user's
values directly).

## 2. Fix

**Principle: a fixed parameter never passes through a transform. The
model gets exactly the float the user gave, and every exported surface
reports exactly that float.**

1. **`fixed_theta`** (new attribute, SVGD and MCMC): float64 array, the
   user's values in parameter space; the single source of truth in every
   regime. Binary-mask form: 1.0.
2. **`fixed_values`** stays what the optimiser uses to pin columns:
   - no transform (`param_transform is None`): `fixed_values == fixed_theta`
     — nothing changes in this regime, and `param_transform` stays `None`
     (about 30 `is None` branches depend on it);
   - a transform exists: `fixed_values` is a constant placeholder (0.0)
     at fixed slots.
3. **Wrapper** (only when a transform exists and something is fixed),
   installed right after the fixed values are parsed and BEFORE the
   constructor's model probe (`svgd.py:~6239` / `6312-6322`):

   ```python
   base = self.param_transform
   def with_fixed(phi):
       phi = jnp.asarray(phi)
       if phi.shape[-1] != theta_dim: raise ValueError(...)
       return jnp.where(fixed_mask_bool, fixed_theta, base(phi))
   with_fixed._per_dim_fns = getattr(base, '_per_dim_fns', None)   # when present
   self._free_transform = base
   self.param_transform = with_fixed
   ```

   Census (review 2, verified by a strict prototype on every path):
   full-vector callers see the wrapper (`_log_prob*` 6455/6517/6697/6756,
   the model probe 6322, `_model_with_fixed` 7326, exports 7549/7603, MAP
   7712/7783, `mcmc.py:458, 848`); the preconditioner's non-composite
   learnable-subset branch (`svgd.py:7282`, consumed at 3895-4202) gets
   `self._free_transform`; priors keep the base transform (they are wired
   at `svgd.py:6015-6023` / `mcmc.py:281-287`, before fixed parsing). The
   shape check is load-bearing (an unchecked wrapper fails on the subset
   path by broadcasting).
4. **`theta_init`**: fixed columns are set to the pin vector
   unconditionally, including when the user supplies `theta_init`.
5. **Exports**: at user-fixed slots `theta_mean` is assigned `fixed_theta`,
   `theta_std` 0 and the HPD bounds `(v, v)` — not computed (the mean of n
   identical floats is generally not that float: v = 0.3, n = 40 gives
   0.29999999999999993). Sites: `SVGD.optimize` (`svgd.py:7485-7512`),
   `SVGD.get_results` (both branches), `MCMC.get_results`
   (`mcmc.py:855-870`). Tied slave slots are NOT assigned; they keep being
   re-tied to their master as today.
6. **`model_selection._check_nested_fixed`** compares `fixed_theta`
   (message says parameter space).
7. **Validation at construction** (no silent repair):
   - a non-finite fixed value raises `ValueError`;
   - any finite value, including 0 and negative values, is delivered
     exactly: the user fixed it explicitly, and positivity is a constraint
     on the OPTIMISED parameters. (Decision taken here, reported to the
     user: the alternative was to reject negative values under
     `positive_params`; that would also forbid fixing a signed auxiliary
     coefficient in a formula/callback model.) A model for which the value
     is invalid fails with its own error;
   - with a CUSTOM transform, the constructor probes `base` at two
     different placeholder values for the fixed slots and raises if any
     free-slot output changes (a transform that couples dimensions, e.g.
     softmax or cumsum, cannot be combined with `fixed`), and raises if
     `base` is non-finite at the placeholder.
8. **Docstrings** for `fixed` (SVGD, MCMC, `Graph.svgd`, `Graph.mcmc`,
   `FreeEpochModel.fit`) and the module docstring example at
   `model_selection.py:24` (it fixes a parameter at 0.0, which now means a
   true zero).

Documented limits (not changed here):

- A single full-vector prior callable is evaluated on the unconstrained
  vector including fixed slots (`svgd.py:6779`, `mcmc.py:487`); it now
  sees the placeholder there instead of the old inverse value. For
  separable priors (all built-in ones) this is a constant with no effect
  on free slots (bit-identical particles measured); a NON-separable custom
  prior sees a different constant. Stated in the `prior` docstring.
- A user-supplied preconditioner INSTANCE keeps its own transform.
- `map_estimate_with_optimization` already raises `TypeError` on master
  (with or without a transform); not repaired here.
- `MCMC(parallel='processes')` fails on master for graph-built models
  (`TypeError: cannot pickle ... GraphBuilder`); unrelated, not repaired
  here.
- Finite-difference backwards at sites that do not skip fixed slots
  (`pmf_from_graph`, `pmf_from_cpp`, `moments_from_graph`, reward-visit)
  probe ±1e-7 around a slot fixed at 0 (the discarded component only);
  `pmf_and_moments_from_graph` clamps its minus probe at 1e-9. Belongs to
  the gradient-step plan.

## 3. Tests (new file `tests/pytest/test_fixed_values_exact.py`)

Spy models (a Python callable recording theta via `jax.debug.callback`;
verified to fire under jit/vmap and to see construction, preconditioner
and optimisation evaluations) for "what the model receives"; REAL models
for "the fit is the fit at that value". Fresh model object per fit unless
a test says otherwise. Particle counts include 4, 40 and 50.

Model input and exports (fail on master except where noted):
1. SVGD, positive_params, v in {0, 1e-12, 1e-10, 1e-9, 1e-8, 0.3, 5.0, 800,
   1e6, -0.5}: every recorded model input at the fixed slot `== v`
   bitwise; constrained particles, history and HPD bounds `== v`;
   `theta_mean == v`, `theta_std == 0` (exact by assignment).
2. Custom transform `exp` (v = 2, 0, 0.3) and `[BetaPrior, None]`
   (v = 0.25, 0.3); binary mask with default, `exp` and BetaPrior
   variants: exactly 1.0.
3. No transform (v = 0, -0.5, 0.3): exact (guard; passes on master) and
   `param_transform is None` still.
4. MCMC: the rows of 1 and 2 (model input and results). One additional
   `parallel='processes'` row uses a picklable pure-Python `log_prob_fn`
   and asserts EXPORTED results only (fixed column, mean, std, HPD exact):
   a spy's record lives in the child process, and graph-built models
   cannot be used in this mode at all (see limits).
5. User-supplied `theta_init` with a different value in the fixed column:
   model input and `history[0]` are the fixed value.
6. Exposure on a fixed parameter, v in {1e-10, 0, 2}. SVGD-level exposure
   wrapper: a spy sees per-observation values `alpha * v` bitwise. Epoch
   path (`Graph.svgd(..., epoch_starts=..., exposure=...)`), where the
   scaling happens inside the model and a spy on the input sees only v:
   the model input is `== v`, and the fit's `log_likelihood` equals the
   epoch model evaluated directly with exactly v at the fixed slot (as in
   test 11).
7. `plot_pairwise_hdr` on a fit fixed at 1e-8: every model evaluation
   during the plot has the fixed slot at 1e-8.
8. Same model object, consecutive fits with different fixed VALUES and
   different fixed SETS: each fit's model input is its own (needs bug 1's
   fix; this is the prerequisite's test).

The fit is the fit at that value (real models):
9. A model with a competing edge, `fixed=(1, 0)`: free-slot particles and
   `log_likelihood` match those of the reduced one-parameter model
   (prototype: -18.36418179450782 both; master evaluates at 0.693 and
   gives -18.652). Array-equal if the implementation reproduces the
   prototype's bitwise agreement, otherwise rtol 1e-9 with the reason
   recorded (a zero-weight edge can change the elimination order). For the
   moments family and for a joint-probability model in which the rate
   fixed at 0 is a COMPETING rate (not the mutation rate, which would make
   the model degenerate); the reduced graph is built explicitly.
10. Same-build equivalence, v in {2.0, 0.3, 5.0}: fit A with
    `fixed=[(1, v)]` vs fit B on a model that ignores theta[1] and
    hard-codes v, B keeping two parameters with slot 1 fixed at another
    value (7.0) so initialisation matches: free-slot particles
    array-equal. (v = 0.3 and 5.0 fail on master by 2e-10 / 5e-10; 2.0 is
    the guard. Replaces v1's cross-build reference numbers.)
11. Epoch model built by `Graph.epoch_model`, one epoch's rate fixed at a
    value below the old floor (1e-10): `log_likelihood` at the fitted
    particles equals the model evaluated with exactly 1e-10 there.
12. Tied epochs: slave equals master on particles, constrained particles,
    history, `theta_mean`, MAP-from-particles and `summary()`; user-fixed
    values exact alongside.
13. Preconditioners 'jacobian' and 'fisher', default and per-dimension
    composite transform: the preconditioner's own model evaluations have
    the exact fixed value.
14. `map_estimate_from_particles` and `log_likelihood(refine=True)` keep
    fixed slots exact.

Validation and guards:
15. Non-finite fixed value raises (SVGD and MCMC); negative and zero are
    accepted and delivered exactly.
16. Custom coupled transform (softmax-like) with `fixed` raises; a custom
    transform that is non-finite at the placeholder raises.
17. `_check_nested_fixed`: two real fits fixing a shared slot at 2.0 vs
    3.0, and at 800 vs 900, are refused; equal values pass.
18. Free slots unaffected by the placeholder: separable prior, fixed value
    2.0 (round-trips exactly today) — particles bit-identical to the same
    fit with a different placeholder constant (the placeholder is a module
    constant the test patches).

## 4. Existing tests and documents that change (each listed, none edited just to pass)

Must be rewritten (they assert the OLD stored unconstrained value):
- `tests/pytest/test_theta_mean_constrained.py:48` (`particles[:, [1,3]] < -4`)
  → constrained value equals the fixed value; unconstrained column equals
  the placeholder.
- `tests/pytest/inference/test_preconditioning.py:337` (unconstrained
  `particles[:, 0]` ≈ 5.0) → same rewrite.
- `tests/pytest/test_exposure_daisy_chain.py:544-557` (unconstrained
  columns equal `log(exp(1e-8) - 1)`) → same rewrite.
- `tests/pytest/inference/test_model_selection.py:524-560`
  (`test_lrt_shared_fixed_value_mismatch_raises` mutates
  `svgd_nested.fixed_values`) → builds the mismatch through `fixed=` /
  `fixed_theta`.
- `tests/pytest/test_svgd_tied_export.py:34, 154`: the "not the sentinel
  image ln 2" assertions lose meaning (the slave's pre-tie input is now
  0.0); the constant and comments are updated, the slave == master
  assertions stay.

Must keep passing unchanged: `tests/pytest/test_refine_respects_fixed.py`
(builds SVGD via `__new__` with `param_transform=None` and parameter-space
`fixed_values`; the refine paths keep working from `fixed_mask` /
`fixed_values`).

Numeric shifts expected: fits are chaotic in the fixed value (delivering
exactly 5.0 instead of 4.999999999999999 moved free particles by up to
1.25 after 30 iterations in a prototype), so EVERY existing fit whose
fixed value does not round-trip today (1e-8, 1e-4, 1.2e-4, 0.3, 0.7, 5.0,
...) changes numerically. Affected test files found by review 3 (loose
assertions, expected to stay green; the gate judges each):
`test_epoch_normalization.py`, `test_epoch_layout_notice.py`,
`test_epoch_default_prior.py`, `test_svgd_config.py`,
`test_daisy_chain_c_path.py`, `inference/test_model_selection.py`,
`inference/test_epoch_model.py`, `test_weight_formula_daisy.py`,
`test_weight_formula_kwarg.py`, `test_daisy_chain_svgd_kernel.py`,
`test_notebook_model_recovery.py`,
`inference/test_svgd_exact_final_grad_kwarg.py`, `inference/test_lrt_at.py`,
`test_svgd_tied_export.py`, `test_exposure_daisy_chain.py`.

For the user (not edited by me):
- `docs/pages/tutorial/model-selection.ipynb` cell 13 fixes migration at
  0 for "svgd_no_mig". Today that is evaluated at migration 0.693; after
  the fix it is a true zero-migration model, so its likelihood-ratio
  results change (or the cell raises if that model is degenerate).
- Every notebook fit with a non-round-tripping fixed value changes
  numerically (svgd-joint-prob cells 26, 34, 40, 45, 50, 63;
  model-selection cells 7, 26, 30; method-of-moments cells 35, 40; ghost
  cell 11; coalescent-derived cell 14; moran-coalescent-derived cell 19).

## 5. Gate

After rebasing onto master-with-bug-1: `pixi run install-dev` → new test
file → full gate with `.git/claude-scratch/gate2.sh`, compared by
`gate_compare.py <baseline> <candidate> --new-file
tests/pytest/test_fixed_values_exact.py`, where the baseline is bug 1's
gate run (the same code as this branch's new base). All gate rules from
bug 1's plan apply (DONE on both sides, equal file sets, usable results
on both sides, passed stays passed, nothing disappears, new tests pass;
regressions and SVGD-touching both-sides failures rerun three times in
isolation on both builds). Because existing fits shift numerically
(section 4), a regression in a test with a non-round-tripping fixed value
is examined for whether its assertion was at the edge; it is reported
either way.

## 6. Risks

- Reading `SVGD.particles` (unconstrained) at fixed columns now gives the
  placeholder; constrained particles give the exact value.
- A value fixed at exactly 0 can make a model genuinely degenerate; that
  now surfaces as the model's own error instead of silently running at
  0.693.
- Numeric shifts in existing fits (section 4).

## 7. Decisions recorded

- Negative and zero fixed values are delivered exactly; only non-finite
  values are rejected (section 2 item 7).
- Placeholder for unconstrained fixed columns: 0.0 (safe for softplus,
  sigmoid, exp: zero gradient, no NaN; inf/NaN placeholders are not).

## 8. Review of v1 — dispositions

Blockers: the likelihood-ratio guard reading `fixed_values` (reviews 1, 2,
3) → section 2 item 6, test 17, section 4; three tests asserting the old
unconstrained value (review 3) → section 4; `theta_mean`/`theta_std`
exactness (reviews 1, 2, 3) → section 2 item 5, test 1 at 4/40/50
particles. Majors: dependency on bug 1 (reviews 1, 2) → prerequisite and
test 8; the no-transform contradiction (reviews 1, 2, 3) → section 2
items 1-2, test 3, `test_refine_respects_fixed.py`; missed paths —
exposure, `theta_init`, `plot_pairwise_hdr`, mask form with custom
transform, MCMC negative (review 1) → section 1 and tests 2, 4-7;
cross-build reference test (review 3) → tests 10, 18; binary-mask test
without power (review 3) → test 2; real-model tests (review 3) → tests 9,
11; tied slaves on every surface (reviews 1, 3) → section 2 item 5, test
12, section 4; notebook cell (review 3) → section 4. Minors: scalar prior
at the placeholder, custom transforms (coupling probe: raises), wrapper
input coercion, user preconditioner instance, FD wording,
`map_estimate_with_optimization`, numeric shifts → section 2 limits,
items 3 and 7, section 4. Negative-value policy (reviews 2, 3 asked for
an explicit choice) → section 7.

Re-review of v2: reviewers 1 and 3 SIGN-OFF (their notes adopted: epoch
half of test 6; user-fixed slots = fixed mask minus tied slaves when
assigning exports — implementation note; test 10 gains v = 0.3 and 5.0;
wording of the MAP limit; the affected-test list pasted into section 4;
test 9's joint variant specified). Reviewer 2 required two rewordings
(tests 4 and 6) and stated no further round is needed once applied as
written → applied in v3.

Implementation note (review 1): SVGD sees tied slaves as ordinary
`(slave, 0.0)` fixed entries, so "user-fixed" for the export assignment is
the fixed mask minus `model._tying_info['slave_to_master']`.
