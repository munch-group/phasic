# Bug 3: a BetaPrior's (0, 1) transform is not applied consistently — fix plan

**Status: v4, 2026-11-23 (reviewers 1 and 2 signed off — reviewer 2 conditional on test 10b, added as written; v4 adds reviewer 3's two changes: the clip in the custom-transform branch with an extended guard 14, and the slow tier run once in the gate). v1 was reviewed by three independent agents
(diagnosis; fix design, with a full prototype; tests/impact). The core
design survived; the diagnosis, three decisions, the function contract
and most tests changed (section 8). Branch
`fix/single-betaprior-transform`, worktree `/Users/kmt/phasic-bug3`. No
source changed yet. Backward compatibility is not required (user
decision).**

**Prerequisites (correctness): bug 1 (no shared compiled gradients) and
bug 2 (fixed values bypass the transform).** With this fix alone, a single
BetaPrior on a 2-parameter model with `fixed=[(0, 5.0)]` would deliver
sigmoid(inverse-softplus(5)) = 0.9933 to the model (measured on a
prototype); with bug 2's wrapper underneath it delivers exactly 5.0
(measured). The branch is rebased onto master after bugs 1 and 2 merge and
implemented then.

Evidence: `experiments/bug3_single_betaprior/` (`b1`; reviewers' scripts,
logs and the prototype in `review1/`, `review2/`, `review3/`), unmodified
code, fresh model object per fit.

## 1. The defect

`BetaPrior` declares `_natural_transform = sigmoid` (clipped to
[1e-9, 1-1e-9]; `src/phasic/svgd.py:1587`, `416`), and — like every
built-in prior — hard-codes the Jacobian and the inverse of that one
transform in its density and sampler (`svgd.py:1663-1676`, `1694`). Who
applies the transform to the MODEL:

- **SVGD**, list prior containing a prior with a natural transform:
  per-slot composite (`svgd.py:5937-5969`). Correct today.
- **SVGD**, single prior: softplus for the model (`svgd.py:5982`) while the
  prior is wired to its own sigmoid (`svgd.py:6019-6023`). Measured
  (`Graph.svgd`, single BetaPrior, 2 parameters): constrained particles up
  to 2.06 and 3.75. On a real discrete geometric model the constructor
  raises ("outgoing rate <= 1. Is 1.43") because the model is handed a
  probability above 1.
- **SVGD, epoch path**: a single prior is broadcast to a per-slot list
  first (`__init__.py:5566-5571`), so a single BetaPrior IS honoured there
  (measured: free slots sigmoid, particles <= 0.77). The two SVGD entry
  routes disagree.
- **SVGD, `positive_params=False`**: no transform at all; particles leave
  (0, 1) (4 of 20 below 0 in the single form; 9 below 0 and 5 above 1 in
  the list form).
- **MCMC**, `positive_params=True` (`mcmc.py:271-287`): softplus for the
  model AND handed to the BetaPrior as its phi → theta map, while the
  BetaPrior still uses its sigmoid Jacobian and its logit inverse for
  initial points — a three-way mix. Consequences, measured over 8 runs of
  4 chains: for theta > 1 the log-prior is NaN; a chain that STARTS there
  never moves (`log_u < nan` is always False) — 17 of 32 chains frozen,
  no warning; chains that move sample the wrong density (mean 0.37-0.41
  for a Beta with mean 0.5).
- **MCMC, `positive_params=False`, no transform**: CORRECT today — the
  untransformed BetaPrior returns -inf outside (0, 1) and proposals there
  are rejected (KS p 0.29-0.82 against the Beta over 5 seeds, single and
  list form).

Also measured:

- `plot_pairwise_hdr` inverts every axis with inverse softplus
  (`svgd.py:9345`): on a (0, 1) slot a hexagon labelled 0.5 is evaluated
  at 0.393, 0.9 at 0.593. Affects list forms today.
- The BetaPrior's log-density is computed from the CLIPPED theta, so it is
  flat beyond |phi| ≈ 20.7: the target in phi is improper. With alpha or
  beta < 1 (e.g. `ci=(0.01, 0.99)` → alpha 0.75) an MCMC chain escapes
  (Beta(0.1, 0.1): 100% of samples at the clip, |phi| up to 4.7e77).
- `prior._transform` is mutable state on an object the user may reuse: it
  is never reset to None, and a later MCMC silently re-wires a prior an
  earlier SVGD fit still uses.

## 2. Fix

**Principle: for each parameter slot, the transform is decided in one
place, and the model, the prior and the initial points all use that one
function. A prior that needs a bounded domain is honoured or refused —
never silently ignored.**

1. **One resolver** in `svgd.py`, used by `SVGD.__init__` and
   `MCMC.__init__`:
   `_resolve_param_transform(prior, prior_list, positive_params,
   param_transform, where)` → the model-facing transform (or `None`), and
   it assigns `_transform` on every prior object it is given —
   unconditionally, including `None`. `theta_dim` is not an argument (it
   is not known yet at either call site; the list length suffices). Call
   position in both constructors: where the wiring happens today — before
   any `prior.sample`, before fixed parsing, before bug 2's wrapper.
   Returned callable: in the per-slot composite case it carries
   `_per_dim_fns` (the preconditioner and bug 2's wrapper read that
   attribute); in the elementwise cases (all softplus; all natural) it is a
   plain elementwise function without it.
2. **Rules**
   - `positive_params=True`:
     - list prior: slot i uses its prior's natural transform if it has
       one, else softplus (today's SVGD behaviour; new for MCMC);
     - single prior with a natural transform: every slot uses it (new;
       consistent with how any single prior applies to all slots and with
       the epoch path's broadcast). Fixed slots are exempt through bug 2;
     - otherwise softplus (unchanged).
     Each prior's `_transform` is the function its slot uses.
   - `positive_params=False`, no custom transform:
     - **SVGD: a prior with a natural transform raises `ValueError`**
       (a gradient method cannot honour a bounded domain without a
       transform; the message names the fix: use `positive_params=True`).
       Same family as the existing rule for positive-support priors.
     - **MCMC: allowed, unchanged** — it is correct today (the prior's hard
       wall rejects proposals outside (0, 1)).
   - custom `param_transform`: every prior's `_transform` is the custom
     function (model and prior use the same map). No new refusal: a
     BetaPrior with a custom sigmoid is consistent today and stays so.
     That built-in priors hard-code their own Jacobian and inverse, so a
     custom transform other than the prior's own gives a wrong log-prior,
     is a separate, pre-existing defect covering EVERY built-in prior
     (`found-defects-2026-11-23.md`, item B) and is not changed here. The
     list form no longer wires a BetaPrior to its sigmoid while the model
     gets the custom transform (today's inconsistency).
3. **BetaPrior's log-density under its natural transform** becomes
   `alpha * log_sigmoid(phi) + beta * log_sigmoid(-phi)` (the Beta
   log-density plus log-Jacobian, with no clip): proper for all
   alpha, beta > 0 (measured: Beta(0.1, 0.1) KS D 0.064 instead of 0.45;
   Beta(0.5, 0.5) no chain enters the flat region). The model still
   receives the clipped theta. The no-transform branch (hard wall) is
   unchanged. Branch selection in `BetaPrior.__call__`, stated explicitly:
   `_transform is None` → hard wall; `_transform` is the class's natural
   sigmoid → the unclipped form (summed over elements); any other
   (custom) transform → the existing formula through that transform,
   with theta clipped to [1e-9, 1 - 1e-9] before the logarithms (an
   unclipped custom sigmoid such as the docstring's
   `lambda x: jax.nn.sigmoid(x)` is exactly 1.0 at phi = 37, which gives a
   -inf log-prior and a NaN gradient; today SVGD's single form avoids this
   only because it hands the prior its own clipped sigmoid).
4. **`plot_pairwise_hdr`** inverts each axis with its slot's inverse
   (inverse softplus or inverse sigmoid, taken from the resolver's
   per-slot information); with a custom transform and
   `unconstrained=False` it raises a clear error instead of inverting with
   softplus (the unconstrained view needs no inverse). Axis limits on a
   (0, 1) slot are clipped to (0, 1). The per-slot inverses live on the
   SVGD object (`self._slot_inverses`, set by the constructor from the
   resolver's per-slot decision), since the elementwise callable carries
   no per-slot information.
5. **Docs**: BetaPrior docstring and the comment at `svgd.py:1585`; Prior
   base class (`svgd.py:664-676`); `param_transform` docstring
   (`svgd.py:5246-5249`); `positive_params` statements (`svgd.py:5239`,
   `mcmc.py:119-121` — which also wrongly says `param_transform`
   "overrides" it — `__init__.py:6064, 7203`): a single BetaPrior confines
   EVERY parameter, including rates, to (0, 1); exposure on a (0, 1) slot
   scales after the transform. Generated `docs/api/*.qmd` are regenerated
   by `pixi run api` (noted for the user).

Not changed: softplus/sigmoid functions and their clips; how initial
points are drawn (single form: one draw of shape (n, dim); list form: a
split key per slot — the two forms legitimately start from different
particles); the preconditioner's moment-matching grid (item G of the
found-defects list); MCMC's silent freeze on a non-finite initial
log-probability (item A) — this fix removes one way of reaching it, and
its tests assert finite initial log-probabilities.

## 3. Tests (new file `tests/pytest/test_betaprior_transform_consistency.py`)

Spy models assert a minimum number of recorded calls. MCMC tests use
`MCMC(...)` directly with pure-JAX models (`Graph.mcmc` is not
seed-reproducible, item C).

Fail on master:
1. SVGD, single BetaPrior, dim 1 and 2: model inputs, initial particles
   after the transform, `get_results` particles, `theta_mean` and HPD
   bounds all in (0, 1); `param_transform(phi)` equals
   `prior._transform(phi)` slot by slot on a phi grid.
2. SVGD, single vs list form with the SAME explicit `theta_init`: dim 1
   (`BetaPrior` vs `[BetaPrior]`) and dim 2 (vs `[Beta, Beta]`, separate
   objects): particles array-equal (prototype: equal; master: differ by
   0.10 / 0.13).
3. Real model, SVGD: discrete geometric chain with success probability
   theta (4 lines with the Graph API), truth 0.3, n = 200, single
   `BetaPrior(ci=(0.5, 0.95))` (prior mass below 0.3 is 4e-4: the prior
   excludes the truth): the fit constructs (master raises), particles in
   (0, 1), and |SVGD mean - exact conjugate posterior mean| < 1 exact
   posterior sd (posterior Beta(a + n, b + sum(k - 1)); no
   truth-in-interval assertion because of the known under-dispersion).
4. Real model, MCMC: the same geometric model and prior; ESS-based check
   |mean - exact mean| < 4 sd / sqrt(ESS).
5. MCMC, flat likelihood, `positive_params=True`, priors: single
   Beta(1.35, 1.35); `[Beta(2, 5)]`; `[GaussPrior, BetaPrior]`; and one
   with alpha < 1 (`BetaPrior(ci=(0.01, 0.99))`). For each: every chain's
   initial log-probability finite and acceptance rate > 0; all Beta-slot
   samples in (0, 1); draws thinned by k = ceil(2N / ESS) then KS against
   the Beta — median p over >= 5 seeds > 0.1 (never a per-seed
   threshold); the Gauss slot against its truncated Gaussian the same way.
   2000 x 4 draws, burn-in 500. The multi-seed part is marked slow except
   for one prior form.
6. Fixed slots (needs bug 2), SVGD and MCMC: single BetaPrior, dim 2,
   `fixed=[(0, 5.0)]`; `[None, BetaPrior]` and `[BetaPrior, None]` with a
   fixed value outside (0, 1); binary-mask form with a single BetaPrior:
   the model input at the fixed slot is exactly the fixed value and the
   free slot is in (0, 1).
7. SVGD with `positive_params=False` and a BetaPrior (single, list)
   raises, and the message names `positive_params=True`.
8. `plot_pairwise_hdr` on a `[Gauss, Beta]` fit and on a single-BetaPrior
   2-parameter fit, called with explicit limits inside (0, 1) on the Beta
   axis (0.2, 0.8): the spy's Beta-slot values during the plot reach both
   ends (max >= 0.75, min <= 0.25; master tops out at
   sigmoid(inverse-softplus(0.8)) = 0.56); with a custom transform and
   `unconstrained=False` the plot raises.
9. `_transform` is reassigned on every construction: a prior object used
   with `param_transform=exp` and then in an MCMC without a transform has
   `_transform is None`; a prior used by an SVGD and then given to an MCMC
   is wired per the MCMC's own rule.
10. List prior with a custom transform: the BetaPrior's `_transform` is
    the custom function (today it is its sigmoid while the model gets the
    custom one).
10b. The log-density itself (pins fix item 3; deterministic): for a
    BetaPrior wired to its natural transform (alpha = 0.5, 0.75 and 2.0),
    at phi = ±25, ±40, ±50 and ±500, `prior(phi)` equals
    `alpha*log_sigmoid(phi) + beta*log_sigmoid(-phi)` (rtol 1e-12) and is
    strictly decreasing in |phi| (master is flat there: -15.5424 at
    phi = -21, -25, -30); for a 2-vector input the value is the sum over
    elements.

Guards (pass on master and after):
11. MCMC with `positive_params=False` and a BetaPrior, no transform:
    samples in (0, 1), thinned KS as in test 5 (the route that is correct
    today must stay correct).
12. SVGD list forms `[BetaPrior]`, `[Gauss, Beta]`: `param_transform`
    equals the per-slot functions on a phi grid (array-equal) and
    `_per_dim_fns` is present; model inputs in range. (Particles are NOT
    asserted bit-identical to master: item 3 of the fix changes the
    BetaPrior's log-density at the rounding level.)
13. Priors without a natural transform (Gauss, LogGauss, Gamma,
    HalfCauchy; single and list), `prior=None` and DataPrior paths: the
    transform is plain softplus without `_per_dim_fns`, on the model and
    on the prior, in SVGD and MCMC; fits bit-identical to a reference
    computed in the same test run through the unchanged list/softplus
    path where one exists.
14. BetaPrior + custom UNCLIPPED sigmoid `param_transform`
    (`lambda x: jax.nn.sigmoid(x)`, `positive_params=False`), SVGD single
    and list forms: model and prior agree on a phi grid that includes
    ±40; the prior's log-density and its gradient are finite there; does
    not raise. (The finite-at-±40 part fails on a build that wires the
    custom function without the clip.)
15. Single BetaPrior, dim 2, default preconditioner with a fixed slot:
    runs; free slot in (0, 1) (the elementwise transform works on the
    preconditioner's learnable subset).

## 4. Existing tests and documents

Inventory (reviews 1 and 3, both trees): no existing test or notebook uses
a BetaPrior as a single prior, with `positive_params=False`, with a custom
transform, or with MCMC.

- `tests/pytest/test_beta_prior.py` (12 tests; list forms and prior-level
  checks) and `tests/pytest/test_svgd_tied_export.py:117-122` (BetaPrior
  master in a list): transform behaviour unchanged. Their fits can shift
  numerically because the BetaPrior log-density changes at the rounding
  level (fix item 3); any assertion that fails for that reason is examined
  and reported, not loosened silently.
- `tests/pytest/test_gamma_prior.py:134-149`,
  `test_gauss_prior_truncation.py:140-201` pin the existing
  `positive_params=False` messages; the new SVGD refusal keeps the
  "positive_params=False" wording they match.
- Bug 2's `tests/pytest/test_fixed_values_exact.py` (on master by then):
  its rows combining BetaPrior with a custom transform are not refused by
  this fix (no new refusal under custom transforms); its MCMC
  `[BetaPrior, None]` rows now also get a confined Beta slot.
- Notebooks: `coalescent-derived.ipynb` cell 14 and
  `old-moran-coalescent-derived.ipynb` cell 25 use `[None, BetaPrior]` in
  SVGD with defaults — transform unchanged. No notebook cell needs
  editing for this fix.

## 5. Gate

Same tools and rules as bugs 1 and 2 (`gate2.sh`, `gate_compare.py
--new-file tests/pytest/test_betaprior_transform_consistency.py`,
`gate_rerun.sh`; DONE on both sides, equal file sets, usable results on
both sides, passed stays passed, nothing disappears, new tests pass;
isolated triple reruns), after rebasing onto master-with-bugs-1-and-2 and
`pixi run install-dev`; baseline = bug 2's gate run. Because the gate
runs `-m "not slow"`, the new file's slow tier (the multi-seed
distributional checks of test 5 for the remaining prior forms) is run
once explicitly, memory-capped, and its result reported with the gate.
Fast-tier cost of the new file: about 2-3 minutes.

## 6. Risks

- A single BetaPrior on a multi-parameter model now confines every
  parameter to (0, 1) (documented; test 1 at dim 2).
- BetaPrior fits shift numerically (unclipped log-density).
- Constructor edits in the same region as bug 2's: rebase first.

## 7. Decisions recorded (reported to the user)

- MCMC without a transform keeps accepting a BetaPrior (correct today);
  only SVGD refuses it.
- No new refusal under a custom transform; the hard-coded-Jacobian defect
  of all built-in priors is tracked separately.
- BetaPrior's log-density under its natural transform uses the unclipped
  form (proper for all shape parameters).
- `plot_pairwise_hdr` uses per-slot inverses and refuses custom
  transforms.

## 8. Review of v1 — dispositions

Review 1 (diagnosis): M1 MCMC mechanism (NaN, frozen chains, wrong
Jacobian) → section 1, test 5 assertions, defect list item A; M2 "never in
MCMC" false → section 1, rule for MCMC without transform, test 11; M3 root
cause (hard-coded Jacobians) → section 2 rules for custom transforms,
defect list item B, tests 10 and 14; M4 test 2 → explicit `theta_init`;
M5 plot and preconditioner grid → fix item 4 / defect list item G; M6
bug 2 prerequisite, resolver shape and call point, bug 2's test file →
header, fix item 1, section 4; m1 particle ranges, m2 epoch route, m3
unconditional `_transform`, m4 exposure sentence, m5 inventory → sections
1, 2, 4, test 9.
Review 2 (design, prototype): M1 prerequisite + fixed-slot test → header,
test 6; M2 test 2 → as above; M3 improper tails → fix item 3, test 5
(alpha < 1, thinning), `Graph.mcmc` not used in tests (defect list item
C); M4 plot → fix item 4, test 8; m1-m6 contract, stale `_transform`,
message, docs, stronger guard → fix items 1, 2, 5, tests 9, 12.
Review 3 (tests): B1 KS on correlated draws → thinned KS, medians, ESS
checks; B2 test 2 → explicit `theta_init`; M1 geometric model moved to
"fails on master", prior excluding the truth, conjugate-posterior
assertion, MCMC counterpart → tests 3, 4; M2 fixed-slot tests → test 6;
M3 plot → fix item 4, test 8; M4 refusal vs documented sigmoid route →
rules (no refusal under custom transform; MCMC without transform kept),
message, docstring, test 14; m1-m6 → section 4 inventory, guards 13, 15,
assertions in test 1, non-vacuous spies, section 6.
