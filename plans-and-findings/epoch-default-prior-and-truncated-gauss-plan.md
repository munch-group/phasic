# Default epoch prior + GaussPrior truncation: plan

**Status: v2.1, 2026-11-22. v1 went through two adversarial reviews (one
on the mathematics and statistics, one on code paths and regressions);
their confirmed findings are folded in (section 8 lists what changed).
All decisions D1-D8 are made (section 7). Plan approved; Batch 0 next.**

This plan covers the first half of `epoch-fit-prior-and-fd-floor-plan.md`
(its Batches 2 and 3), split out at the user's request. The positive-FD-step
fix and the slow-evaluation guard (that plan's Batches 1 and 4) go in a
second plan, written and reviewed after this one is implemented.

## 1. Goal and what "done" means

Two user-visible problems with `joint_prob_graph.svgd(..., epoch_starts=...)`:

1. **The hang (issue A).** With `prior=None` the fit never finishes.
   Cause: the default epoch prior comes from a reference fit that ignores
   the user's `fixed` and `exposure`. The overall rate scale is then not
   identified, and the resulting prior (about ±3.2e6 on the measured
   fixture) puts particles at rates where each evaluation takes minutes.
2. **Particles clipped onto 1e-6 (part of issue B).** `GaussPrior.sample`
   clips negative draws to 1e-6 under the positivity transform. Those
   particles barely move and sit 10x above the negative-FD-probe threshold.

Done means:

- `svgd(epoch_starts=..., prior=None, fixed=...)` builds a prior centred
  near the time-homogeneous estimate, on the same width convention as the
  non-epoch default, with exposure (uniform or varying) taken into account,
  and the fit completes.
- When the default prior cannot be built honestly (no rate fixed, differing
  per-epoch fixed values), it raises a `ValueError` naming the cause and the
  remedy. No silent fallback.
- `GaussPrior(..., lower=0.0)` gives an exact Gaussian truncated at zero:
  no particle is clipped at initialisation. Under positive_params,
  `lower=0.0` is required (D1); `lower=None` (the default) is only valid
  with positive_params=False.
- The SVGD gradient path is unchanged (3.3). Initial particles are
  bit-identical to today for every prior draw that was already positive.
- Every behaviour is covered by tests in the suite.

Out of scope: the FD step; the slow-band guard; LogGauss, HalfCauchy and
Beta priors; exposure on non-joint graphs; custom `param_transform`
(GaussPrior already assumes softplus there; see 6).

## 2. Evidence

### 2.1 Measured (investigation, 2026-11-22)

4-sample coalescent joint graph, 10,000 observations at coal=5, mu=3, mu
fixed (scripts in the session scratchpad
`/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/hang/`;
may have been cleared):

| call | result |
|---|---|
| non-epoch, `prior=None` | 3.1 s |
| `epoch_starts=[0, 0.15]`, `prior=None` | killed at 150 s, inside SVGD iterations |
| same, `theta_init` near the truth | 3.1 s |
| same, explicit `GaussPrior(ci=[-3.2e6, 3.2e6])` | hangs like the default |
| epoch-path `probability_matching` without `fixed` | theta (1.61, 0.95), std (643,681 ; 379,143) |

Measured by the code-path reviewer: the normalised `joint_prob_table()` of
`coalescent_indexed(4)`, reward_limit=3, at theta, 2*theta and 0.01*theta
agrees to 2.8e-17, for `discrete=True` as well as `discrete=False`.

### 2.2 Read from source (master `d875149d` plus this session's uncommitted edits)

- `src/phasic/__init__.py` 5551-5590: the epoch default prior calls
  `probability_matching(self, observed_indices, std_multiplier=sd,
  verbose=verbose)` (5556), with no `fixed` and no exposure. It then builds
  `GaussPrior(ci=[m - sd*s, m + sd*s])` per slot, broadcast over epochs.
  `else 1.0` (5581) is a silent fallback. The comment at 5575, "same
  convention DataPrior uses", is wrong.
- `pm_result.std` is the RAW delta-method std; `std_multiplier` is applied
  only in `pm_result.prior` (`probability_matching.py` 374-377). So the
  epoch prior has sigma = sd*s/1.96 (via `GaussPrior(ci)`, `svgd.py`
  771-777), while `DataPrior` has sigma = sd*s. The epoch default is 1.96x
  narrower.
- The builder receives the raw `user_fixed`, `exposure_arr` and
  `exposure_param_index` (signature 4415-4432). Both callers pass vertex
  indices already mapped (`Graph.svgd` 6651 then 6752; `epoch_model` 5689
  then 5710). Tied slaves are appended to `broadcast_fixed` as
  `(flat_idx, 0.0)` sentinels (4711-4717), so the reference fit must be
  built from `user_fixed`, never from `broadcast_fixed`.
- On the exposure model, observation i uses theta with the
  `exposure_param_index` slot multiplied by alpha_i in every epoch
  (5172-5179).
- `probability_matching.py` 346-350 and 362-364 hold two silent std
  fallbacks (0.5*|theta|). A rank-deficient Jacobian goes through `pinv`
  (344). `method_of_moments.py` 545-562 has the same two fallbacks.
- `svgd.py` 815-841: `GaussPrior.sample` draws `jax.random.normal(key,
  shape)*sigma + mu`, then under a transform `jnp.maximum(theta, 1e-6)`
  before inverse softplus. `__call__` (786-813) is the unnormalised
  Gaussian log density plus the softplus Jacobian. `_inverse_softplus`
  (383-405) also floors at 1e-6.
- SVGD (5665-5695) and MCMC (`mcmc.py` 280-287) set `prior._transform`
  but never reset it, so `_transform is None` does not reliably mean "no
  positivity transform" (a prior reused from an earlier fit keeps it).
- `joint_prob_graph` refuses constant base edges (10430). Event edges,
  trash edges and t-vertex exits are linear in theta (10848, 10857-10863,
  10910-10923). Start edges are constant but carry no time. `discrete=True`
  sets `was_dph`, whose `update_weights` divides each out-edge by the
  vertex's total out-weight (`phasic.c` 5800-5816).

### 2.3 What is and is not scale-identified (derived; confirmed by both reviews)

- **Every joint graph is scale-free unless a rate is fixed.** Multiplying
  every rate by c leaves the normalised outcome table unchanged: on a
  continuous graph every sojourn time scales by 1/c and the
  normalisation cancels it; on a discrete graph the per-vertex
  normalisation removes c. Exposure does not help: observation i's rates
  are c times (theta with slot k times alpha_i), a uniform rescaling of
  that observation's rates. So the test is exact and structural: **joint
  graph and no fixed rate means not identified.** No numerical threshold.
- **Epoch fits.** The final epoch runs to absorption, and its factor in
  the likelihood is an absorption table, invariant to scaling the
  final-epoch rates. Absolute epoch start times identify only the earlier
  epochs. So an epoch fit is identified only if some rate is fixed (a
  fixed entry fixes it in every epoch, 4610-4628) or some final-epoch rate
  is tied to an earlier epoch. The notebook case (mu fixed) is identified.
  An epoch fit with nothing fixed or tied is not: a prior silently sets
  the final epoch's scale. This is D7.
- The SVD-ratio check proposed in v1 would also fire on parameters that
  are merely weakly identified, at the 1e-10 bound, or at small scales.
  It is dropped as an error; at most a warning-level diagnostic (D5).

## 3. Design

### 3.1 Default epoch prior is built where it is used

The builder `_daisy_chain_svgd_model` is also used as a model factory, with
no prior and no fixed: `Graph.epoch_model` (whose `FreeEpochModel.fit` can
add the identifying `fixed` after construction, `epoch_model.py` 62-124)
and direct calls in tests. Building the default prior inside the builder
would make those raise once the guard exists.

- Move the default-prior construction into a helper,
  `_epoch_default_prior(observed_indices, user_fixed, exposure_arr,
  exposure_param_index, n_epochs, param_length, sd, verbose)`.
- The builder returns `prior=None` when `user_prior` is None; `Graph.svgd`
  calls the helper; `FreeEpochModel.fit` calls it at fit time with the base
  `fixed` and the fit's `fixed` combined (flat fixed converted back to
  local form; a fit-level fixed that varies across epochs follows D2).
- Batch 0 inventories every direct builder call that relies on the
  returned prior.
- Keep the `probability_matching` import inside the helper at call time
  (`test_prior_fallback_raises.py` monkeypatches the module attribute).

### 3.2 What the helper does

- **Fixed (D2, decided: raise on differing values).** Convert `user_fixed`
  (local index; scalar, or a per-epoch list) to the scalar `(index,
  value)` form. Scalars pass through; a list whose values are all equal
  passes as that value; differing values raise with the remedy (pass
  `prior=`).
- **No rate fixed → raise** (the guard, 3.4), naming the remedy.
- **Exposure (D3, decided: uniform and varying).** Extend
  `probability_matching` additively with `exposure=None,
  exposure_param_index=None`:
  - group observations by distinct alpha value; for group g, compare the
    group's empirical outcome frequencies with the model table at theta
    with slot k multiplied by alpha_g;
  - stack residuals weighted by sqrt(n_g/n); the delta-method covariance
    uses the block-diagonal multinomial covariance (each block for its
    group's n_g) with the same weights;
  - a fixed exposure parameter keeps its user value (the scaling is
    applied inside, per group), so no special casing of fixed v*alpha;
  - uniform exposure is the one-group case, and the result equals the
    rescaled fit exactly (a Batch 0 check);
  - cost: one table evaluation per distinct alpha per residual
    evaluation. D8 covers what happens when the number of distinct values
    is large (e.g. exposures drawn from a continuous distribution, as in
    `inference/test_svgd_exact_final_grad_kwarg.py`).
- **Width.** Per free slot, `GaussPrior(mean=m, std=sd*s, lower=...)`
  built in the helper (not copied from `pm_result.prior`, which carries
  `None` at fixed slots and would complicate masking). Remove the `else
  1.0` fallback; correct the comment.
- **Epoch differences.** The prior is centred on a time-homogeneous fit,
  and with much data sd*s is small, so a true per-epoch rate can sit many
  prior sds away. D7 decides whether to add a relative width floor
  (sigma >= f*|m|).
- **Tied parameters** need nothing; the reference fit is time-homogeneous.

### 3.3 GaussPrior truncation

- New keyword `lower`: `None` (no truncation) or `0.0`. Any other value
  raises `ValueError`: the softplus transform guarantees theta > 0 but
  cannot keep a particle above a positive bound, and a negative bound has
  no effect under it. The default is D1. `mean`/`std`/`ci` keep their
  meaning as the parent Gaussian, so `mu` and `sigma` are unchanged;
  documented that with `ci` near or below zero the truncated prior's mass
  inside `ci` is no longer the stated `prob`, and its mean moves up.
- **Log density: unchanged.** Under the transform theta =
  max(softplus(phi), 1e-9) > 0, so the SVGD and MCMC target already is the
  zero-truncated prior; truncation only adds a constant (log Phi(mu/sigma))
  that `__call__` already leaves out along with the Gaussian's own
  normaliser. Every value consumer found (SVGD log_prob and recorded
  traces, MCMC acceptance ratios) uses differences or a constant shared by
  all particles; `model_selection.py` does not use priors.
- **Sampling, key-free and exact.** Draw `z = normal(key)*sigma + mu` as
  today. For z <= 0 only, with a = -mu/sigma and w = (z - mu)/sigma:
  U = Phi(w)/Phi(a) is uniform on (0, 1) given z <= 0, and
  x = mu + sigma*Phi^-1(Phi(a) + U*(1 - Phi(a))) is an exact draw from the
  truncated distribution (computed in survival form,
  x = mu - sigma*Phi^-1((1 - U)*Phi(-a)), when Phi(a) is close to 1).
  Each particle depends only on its own z, so draws stay independent; no
  key reuse; positive draws are bit-identical to today.
- **Degenerate priors.** With `lower=0.0`, if less than 1e-6 of the parent
  mass is above zero (mu/sigma < -4.75), the constructor raises, saying to
  use `positive_params=False` with `lower=None` if a negative parameter is
  meant. Only user priors can hit it (fitted means are bounded below by
  1e-10).
- **D4 (decided: raise).** A prior with `lower` set, used with
  `positive_params=False`, raises. The check runs at SVGD and MCMC
  construction from `positive_params` itself (an `svgd_config` rule plus
  the MCMC equivalent), not from `_transform` (unreliable, 2.2) and not
  inside `__call__` (standalone evaluation of a prior stays possible;
  `test_gate_svgd_seams.py` 47-51 does that).
- **Internally built priors follow the fit.** `DataPrior`,
  `probability_matching`, `method_of_moments` and the epoch helper take a
  `lower` argument, and `Graph.svgd`/`mcmc` pass `lower=0.0` when
  `positive_params=True` and `lower=None` otherwise. So
  `positive_params=False` with `prior=None` keeps working (otherwise D4
  would break `inference/test_hpd_naming.py` 96 and
  `inference/test_mcmc_accuracy.py` 186-191).
- `plot()` draws the truncated density divided by Phi(mu/sigma). The SVGD
  prior printout (`svgd.py` 5774-5787) marks truncation, checked with
  `isinstance(.., GaussPrior)` (the current `hasattr(mu, sigma)` also
  matches LogGaussPrior).

### 3.4 The identifiability guard

- Structural and exact: a joint graph (`_joint_prob_base_graph_indexer` set)
  with no rate fixed cannot have its default prior built. Raise
  `ValueError`: "every rate can be scaled by the same factor without
  changing the joint probabilities, so the data cannot determine the
  overall scale; fix one rate with fixed=, or pass prior=". Scope is D5.
- Replace the two silent std fallbacks in `probability_matching` with an
  error (D5). The `method_of_moments` fallbacks are listed in 6, not
  changed.
- Epoch fits with nothing fixed or tied in the final epoch: D7.

## 4. Batches

Each batch: on a branch, de-risk first, then implement, add tests, gate.
Every run goes through the memory-capped runner
`.git/claude-scratch/runcap.py` (parent RSS watchdog; RLIMIT is not
enforced on macOS). Gates are "no new failures against the Batch 0
baseline", never "full suite green". `pixi run install-dev` after every
source change; never `git add -A`.

### Batch 0 — de-risk and baseline (no source changes)

The baseline needs a committed state (no stashing). First commit this
session's open work (the `event_rate_coefficient` rename and the plot fix,
with their tests), with the user's go-ahead.

a. **Scale invariance.** Confirm it numerically on one more model (a
   3-parameter joint model) for discrete and continuous, with and without
   uniform and varying exposure; confirm that fixing one rate breaks it.
   Confirm the epoch claim: at fixed earlier-epoch rates, scaling the
   final-epoch rates leaves the epoch likelihood unchanged.
b. **Exposure.** The grouped reference fit with one group equals the
   rescaled fit; a fixed exposure parameter is scaled per group as the
   daisy model does it (to 1e-12 on the tables).
c. **Exposure cost (feeds D8).** Time the grouped fit at 1, 10, 100, 1000
   distinct alpha values on a 1k- and a 20k-vertex joint graph.
d. **Sampler.** KS test of the key-free sampler against
   `scipy.stats.truncnorm` at a in {-3, 0, 2, 4.75}; bit-identity with
   today's sampler when all draws are positive; accuracy of the
   survival-form branch.
e. **Inventory.** List every call that will change behaviour:
   - `prior=None`, `DataPrior`, `probability_matching` or MCMC default on
     a joint graph with nothing fixed (guard, D5); known so far:
     `test_probmatch_input_check.py::test_mapped_indices_fit_ratio`
     (61-66), `inference/test_svgd_exact_grad_kwarg.py` (~143-200);
   - epoch fits with `prior=None` and nothing fixed: known so far:
     `test_epoch_normalization.py` 107, 154, 301;
     `inference/test_fd_gradient_mixed_scale.py` 197;
     `inference/test_svgd_exact_final_grad_kwarg.py` 193-293;
     `test_joint_degenerate_outcomes.py` 135; `test_joint_prob_tot_cap.py`
     121;
   - direct builder calls relying on the returned prior:
     `test_epoch_sojourn_finalread.py` 237-260,
     `inference/test_exact_grad_daisy_final.py` 69, 470;
   - D2 (differing per-epoch fixed with `prior=None`), D4
     (`lower` set + `positive_params=False`), and every seeded fit using a
     default prior (prior=None) that had negative draws (D1: only internal
     priors get `lower=0.0`);
   - epoch fits not identified per D7 (nothing fixed or tied).
   Each entry gets a planned disposition (add `fixed=` or an explicit
   prior, or expected new error) before Batch 1.
f. **Baseline.** Run the gate files on the committed state.

Gate files (all batches): the new test files, plus
`test_prior_fallback_raises.py`, `test_probmatch_input_check.py`,
`test_joint_outcome_count.py`, `test_svgd_assumptions.py`,
`test_method_of_moments.py`, `test_constant_parameterized_consistency.py`,
`test_epoch_normalization.py`, `test_epoch_sojourn_finalread.py`,
`test_exposure_daisy_chain.py`, `test_mcmc.py`,
`test_event_rate_coefficient.py`, `test_plot_epoch_true_theta.py`,
`test_joint_degenerate_outcomes.py`, `test_joint_prob_tot_cap.py`,
`test_daisy_chain_c_path.py`, `test_gate_svgd_seams.py`,
`test_beta_prior.py`, `test_bffg_accuracy.py`, `test_reward_validation.py`,
`test_epoch_layout_notice.py`, `test_svgd_effective_options_view.py`,
`test_joint_derived_graph_table.py`, `test_svgd_tied_export.py`,
`test_daisy_chain_svgd_kernel.py`, `test_weight_formula_daisy.py`,
`inference/test_svgd_api_parity.py`, `inference/test_hpd_naming.py`,
`inference/test_mcmc_accuracy.py`,
`inference/test_svgd_exact_final_grad_kwarg.py`,
`inference/test_svgd_exact_grad_kwarg.py`,
`inference/test_exact_grad_daisy_final.py`,
`inference/test_fd_gradient_mixed_scale.py`, `inference/test_epoch_model.py`,
`inference/test_svgd_exact_moment_grad_kwarg.py`,
`inference/test_log_prob_unified_bit_identity.py`,
`inference/test_model_selection.py`, `inference/test_lrt_at.py`, and
`test_notebook_model_recovery.py -m "not slow"`, `test_weight_formula_kwarg.py`,
`inference/test_log_likelihood_zero_inflated.py`.

### Batch 0 results (2026-11-22, master `54117196`)

Scripts: session scratchpad `.../scratchpad/b0/` (`a_scale.py`,
`c_cost.py`, `d_sampler.py`, `inv_plugin.py`, `run_gates.sh`); may be
cleared, the numbers below are the record.

- **0a scale invariance: confirmed.** Normalised joint table unchanged to
  1.1e-16 (continuous) and 5.6e-17 (discrete) under theta -> c*theta,
  c in {2, 0.01, 37}, on the 1-rate and a 2-rate coalescent (theta_dim 2,
  3); holding the event slot fixed while scaling the rest changes it by
  0.13-0.20. Single epoch with varying exposure: log-likelihood change
  exactly 0 under scaling. Two epochs [0, 0.3]: scaling only the final
  epoch's rates changes the log-likelihood by exactly 0; scaling the first
  epoch's by 0.747.
- **0b exposure: confirmed.** Daisy log-likelihood with uniform exposure
  3.5 equals the no-exposure model with the exposure slot multiplied by
  3.5 in every epoch, difference exactly 0, for 1 and 2 epochs.
- **0c cost (D8 cap = 100).** 680 vertices: one table 0.2 ms, 1000
  batched 8 ms. 113,734 vertices: one table 11.3 ms; batched 10 / 100 /
  1000 tables 45 / 301 / 2966 ms. A reference fit takes 37-39 table
  evaluations (0.1 s and 3.4 s). A 9-sample graph did not build within
  the 20-minute limit.
- **0d sampler: confirmed exact.** Key-free survival-form inverse CDF:
  over 40 keys the KS p-values against truncnorm are uniform (fraction
  below 0.05: 0.00-0.05) at a in {-3, 0, 2, 4.75}; finite and correct at
  a = 6 and 8; kept draws bit-identical; an all-positive prior gives
  identical samples.
- **0f baseline:** 574 passed, 4 skipped, 6 deselected (slow), 3 xfailed,
  0 failed, 31 min, peak 3.9 GB (`-m "not slow"` over all gate files).
  Two GaussPrior test files missing from the gate list were added and
  baselined: `test_weight_formula_kwarg.py`,
  `inference/test_log_likelihood_zero_inflated.py` (14 passed).
- **0e inventory** (record-only plugin over the gate run; same 574
  passed):
  - D1 (user GaussPrior under positive_params): sampled in 8 test files;
    all 24 test files with GaussPrior get `lower=0.0` where
    positive_params is on. Notebooks: 11 files, listed for the user, not
    changed.
  - D4: user GaussPriors with positive_params=False only in
    `inference/test_hpd_naming.py` (8) and `inference/test_mcmc_accuracy.py`
    (4); they keep `lower=None`, so nothing breaks.
  - Changed initial particles (default priors with negative draws):
    `inference/test_svgd_exact_final_grad_kwarg.py`,
    `inference/test_svgd_exact_grad_kwarg.py`,
    `inference/test_svgd_exact_moment_grad_kwarg.py`,
    `test_constant_parameterized_consistency.py`,
    `test_epoch_layout_notice.py`, `inference/test_hpd_naming.py`; user
    prior: `inference/test_log_prob_unified_bit_identity.py`. Their
    assertions compare paths under the same prior, so they should hold;
    confirmed at the gate.
  - Direct builder calls (`test_epoch_normalization.py` 83,
    `inference/test_exact_grad_daisy_final.py` 69/470,
    `test_daisy_chain_c_path.py` 742/1037, `test_epoch_sojourn_finalread.py`
    237-260, `test_exposure_daisy_chain.py` 617-979) all discard the
    returned prior: unaffected by 3.1. `epoch_model` callers never fit
    with the default prior: unaffected.
  - Will raise under D5 (default prior, joint graph, nothing fixed),
    disposition = pass an explicit `GaussPrior(..., lower=0.0)` (keeps the
    tests' free parameters and intent): `inference/test_svgd_exact_final_grad_kwarg.py`
    (13 svgd epoch calls + `test_none_never_raises[joint_prob]`),
    `inference/test_svgd_exact_grad_kwarg.py` (7 tests via DataPrior).
    `test_probmatch_input_check.py::test_mapped_indices_fit_ratio` fits the
    unidentified case on purpose: change to fix one rate and check the
    estimate, plus a new assertion that the unfixed case raises. Its input
    validation tests must keep their own errors: the guard runs AFTER input
    validation.
  - `test_prior_fallback_raises.py::test_epoch_path_raises_when_probability_matching_fails`
    expects `epoch_model(obs, ...)` to raise at construction; under 3.1 the
    default prior is built at fit time: change it to `.fit(fixed=...)` and
    add the same check through `svgd(epoch_starts=..., fixed=...)`.
  - D2: no test uses differing per-epoch fixed values with prior=None.
  - D7 notice: epoch fits with the final epoch unidentified occur in 12
    gate files; the notice is additive (no test asserts the full notice
    list for these fits; confirmed at the gate).
  - D8: existing tests have at most 30 distinct exposure values, so the
    binning path (>100) is covered only by the new tests.

### Batch 1 — GaussPrior `lower`

- Add `lower=0.0` to every GaussPrior in tests, experiments and
  docstrings (D1). **Docs notebooks are NOT changed (user, 2026-11-22).**
  Instead, Batch 1 reports a list of every notebook cell with a GaussPrior
  that needs `lower=0.0`; those cells raise under the new rule until the
  user updates them (this includes the notebook-to-script step of
  `pixi run test`).
- Implement 3.3 (`svgd.py`, `svgd_config.py`, `mcmc.py`, and the `lower`
  pass-through in `DataPrior`, `probability_matching`, `method_of_moments`,
  `Graph.svgd`, `Graph.mcmc`).
- New `tests/pytest/test_gauss_prior_truncation.py`:
  - with `lower=0.0` under the transform: no initial theta <= 0; KS test
    against truncnorm; a prior with no mass below zero gives bit-identical
    samples for `lower=0.0` and `lower=None`; `lower=None` reproduces
    today's samples bit-for-bit (clip included);
  - `__call__` value and gradient identical for `lower=0.0` and `None`;
  - `lower` not in {None, 0.0} raises; mu/sigma < -4.75 raises;
  - `lower` set with `positive_params=False` raises in SVGD and MCMC;
    `lower=None` with `positive_params=True` raises in SVGD and MCMC (D1);
    standalone `prior(phi)` still evaluates;
  - `positive_params=False` with `prior=None` still works (internal
    priors get `lower=None`);
  - the plotted density integrates to 1 over theta > 0;
  - assertions on the parent `mu`/`sigma`, never on sample means;
  - a small-scale parameter (mu ~ 1e-4, sigma ~ 5e-5) documents how often
    the `_inverse_softplus` 1e-6 floor still catches a draw.
- Gate: no new failures; changes limited to the Batch 0e inventory.

### Batch 1 results (2026-11-22, branch `fix/gauss-prior-lower`, worktree `../phasic-gausslower`)

- Implemented as designed, with one placement change: the
  lower/positive_params check runs in `SVGD.__init__` and `MCMC.__init__`
  (`svgd._check_gauss_prior_lower`, right after the transforms are
  attached), not as an `svgd_config` rule; it covers `Graph.svgd`, direct
  `SVGD`, `FreeEpochModel.fit` and MCMC alike. `DataPrior`,
  `probability_matching`, `method_of_moments` (and their `Graph`
  wrappers) take `lower` (default 0.0); `Graph.svgd`/`Graph.mcmc` pass
  `0.0 if positive_params else None`; the epoch builder takes
  `prior_lower` (moved to fit time in Batch 2). No `__repr__` was added:
  the effective-options view shows default-repr objects as `svgd.GaussPrior`
  and its tests pin that.
- `tests/pytest/test_gauss_prior_truncation.py`: 27 tests, all pass.
- Gate: 614 of 615 pass (574 baseline + 14 added files + 27 new). The one
  failure, `test_reward_validation.py::test_svgd_runs_silently_on_partial_coverage_with_introspection`
  (11.98 vs true 10, tolerance 1.5), is PRE-EXISTING flakiness: the test
  samples without a seed. On six seeded datasets master and the branch
  give bit-identical estimates (10.50, 10.66, 10.24, 10.47, 11.33, 10.47).
  All six are above 10: a ~5% upward bias at this budget, not
  investigated (out of scope). Test left unchanged.
- Notebooks not changed (user); the cells needing `lower=0.0` are listed
  in the Batch 0 report.

### Batch 2 — default epoch prior and the guard

(The guard and the new helper ship together: the helper needs the guard to
raise on "no rate fixed", and the guard's epoch-path reach depends on the
helper.)

- Implement 3.1, 3.2 and 3.4 (`__init__.py`, `epoch_model.py`,
  `probability_matching.py`).
- New `tests/pytest/test_epoch_default_prior.py`, using a fixture whose
  per-epoch true rates DIFFER:
  - prior=None with scalar fixed: each free slot's parent mean equals an
    independent estimate (a scipy fit of the time-homogeneous model on the
    same data, same fixed) within tolerance; sigma finite; sigma equals
    DataPrior's for the same data and `sd` (before any D7 floor); each
    free slot's prior 95% interval covers the truth (D6);
  - a 3-iteration epoch fit with prior=None and fixed completes (the hang
    regression test, asserted through the prior's centre and width, not
    a timeout: pytest-timeout cannot interrupt an FFI call cleanly);
  - per-epoch fixed lists: equal values behave as the scalar; differing
    values raise;
  - exposure: the notebook case (mu fixed at 1.2e-4, exposure 10000,
    uniform); varying exposure with a few distinct values; a free exposure
    parameter on a model with another rate fixed; the D8 behaviour at many
    distinct values;
  - no rate fixed: `probability_matching`, `DataPrior`, `svgd(prior=None)`,
    MCMC `prior=None` and the epoch default raise with the remedy (per D5's
    scope); one rate fixed: no raise; discrete graphs raise the same;
  - `Graph.epoch_model(prior=None)` builds without error with nothing
    fixed, and `fit(fixed=...)` builds the default prior then;
  - the former silent std fallbacks raise (per D5);
  - D7's behaviour for an epoch fit with nothing fixed or tied.
- Update the Batch 0e tests per their planned disposition, each reason in
  the commit message.
- Manual acceptance (not in the suite): notebook cell 49
  (`docs/pages/tutorial/svgd-joint-prob.ipynb`) with `prior=None`,
  memory-capped, completes.
- Gate: no new failures beyond the inventoried, explained ones.

### Batch 2 results (2026-11-23, branch `fix/epoch-default-prior`, worktree `../phasic-epochprior`)

- Implemented as designed: `Graph._epoch_default_prior` (reference fit with
  the fit's fixed rates and exposure; sigma = max(sd*se, |m|); lower from
  positive_params) called from `Graph.svgd` and `FreeEpochModel.fit`
  (factory set by `Graph.epoch_model(prior=None)`); the builder returns
  prior None. `probability_matching` gained `exposure`,
  `exposure_param_index`, `max_exposure_groups=100` (quantile bins above;
  `ProbMatchResult.exposure_groups/exposure_binned`), the structural
  "no rate fixed" error (after input validation), and errors in place of
  the two silent std fallbacks. Notices: binned exposure, and an epoch fit
  whose last-epoch scale nothing pins (`_epoch_final_scale_unidentified`;
  FreeEpochModel.fit checks its merged fixed). The builder's `prior_lower`
  (Batch 1) is gone.
- Width floor raised to |m| (user, 2026-11-23), see D7.
- MCMC on a joint graph cannot take row observations at all, so its
  default-prior path is not reachable there (covered via DataPrior).
- Test changes per the Batch 0 dispositions: explicit prior in the
  `_svgd` helpers of `inference/test_svgd_exact_final_grad_kwarg.py` and
  `inference/test_svgd_exact_grad_kwarg.py` (joint graphs only);
  `test_probmatch_input_check.py` fits with mu fixed and asserts the
  unfixed case raises; `test_prior_fallback_raises.py` checks the epoch
  default through svgd and `FreeEpochModel.fit`.
- New `tests/pytest/test_epoch_default_prior.py` (16 tests, closed-form
  n=2 data with per-epoch truths 2 and 6).
- Gate: 661 passed, 0 failed.
- Manual acceptance: the investigation fixture (4 samples, epochs
  [0, 0.15], 10,000 windows, mu fixed, exposure 1000, prior=None) that
  hung before finishes 300 iterations in 16.7 s; default prior
  N(4.86, 4.86^2) truncated at 0 (truth 5).
- Found, not fixed (pre-existing, also on master): `theta_mean` of an
  epoch fit reports FIXED slots on the unconstrained scale (-9.028 =
  log(expm1(1.2e-4)) for mu = 1.2e-4).

After Batch 2: a short CLAUDE.md entry, then the second plan (positive FD
step + slow-evaluation guard) and its adversarial review.

## 5. Risks

- **Behaviour changes are wider than "epoch fits".** The guard (if D5 is
  "all callers") reaches the default discrete `prior=None` path and MCMC.
  Every docs notebook joint-graph inference cell passes `fixed=` (grep),
  so the reach is mostly tests; Batch 0e lists them.
- **Exposure grouping cost** with many distinct values (D8, measured in
  Batch 0c).
- **Changed initial particles** for seeded fits with a default prior
  (prior=None) that had negative draws (D1). Bounded to those draws by
  construction; inventoried.
- **The model-factory path** (3.1) changes the builder's return contract
  (a private function); Batch 0e inventories its direct callers.

## 6. Known limits, flagged not fixed

- GaussPrior's Jacobian term and inverse transform assume softplus even
  with a custom `param_transform` (pre-existing).
- `_inverse_softplus` floors at 1e-6, so a truncated draw below 1e-6 is
  still floored. Per draw the probability is about
  phi(mu/sigma)/(sigma*Phi(mu/sigma)) * 1e-6, not small when sigma is
  small (e.g. a rate near 1.2e-4); measured in Batch 1, fixed in the
  second plan if it matters.
- `_transform` is never reset between fits (pre-existing).
- `method_of_moments` has the same two silent std fallbacks (545-562).
- The default priors ignore exposure on non-joint graphs (pre-existing;
  joint-graph exposure is epoch-only, rule R9).
- The default prior uses the data twice (empirical Bayes), as the
  non-epoch default already does.

## 7. Decisions

Decided (user, 2026-11-22):

- **D2 — differing per-epoch fixed values with prior=None:** raise.
- **D3 — exposure in the default prior:** uniform and varying exposure.
- **D4 — a truncated prior with positive_params=False:** raise.
- **D1 — keyword `lower`, default `None`, and it must be stated under
  positive_params.** A GaussPrior used under `positive_params=True` must
  have `lower=0.0`; one with `lower=None` raises at SVGD/MCMC construction
  ("under positive_params the prior must be truncated at zero; pass
  GaussPrior(..., lower=0.0)"). With `positive_params=False`, `lower` must
  be None (D4). The priors phasic builds itself follow the same rule
  (`lower=0.0` under positive_params). Every GaussPrior in the repo (about
  55 calls in 24 test files, 45 calls in 11 docs notebooks, 2 experiment
  scripts, docstring examples) gets `lower=0.0` in Batch 1, except the
  docs notebooks, which are not changed (the user updates them from a
  list Batch 1 provides). User scripts
  outside the repo break with a clear message (accepted by the user).
  Since every prior under positive_params is now truncated, no particle is
  clipped at initialisation.
- **D5 — the "no rate fixed" error lives in `probability_matching`**, so
  every default prior on a joint graph raises (non-epoch svgd, MCMC,
  DataPrior, Graph.probability_matching, the epoch default), and its two
  silent std fallbacks become errors. No SVD-ratio warning in this plan.
- **D6 — the default-prior test may compare with the true values.** The
  "priors exclude the truth" rule applies to user-specified priors in
  recovery tests, not to tests of how the default prior is built. The
  test asserts both the independent-scipy-estimate match and that each
  free slot's prior 95% interval covers the true value.

- **D7 — (a):** an epoch fit whose final-epoch scale is not identified
  (nothing fixed, no final-epoch rate tied to an earlier epoch) proceeds
  with a visible notice in the fit's notices. The epoch default prior
  gets a relative width floor, sigma = max(sd*s, |m|) (raised from
  0.5*|m| on 2026-11-23: with 0.5 the 95% interval reached only ~2x the
  time-homogeneous estimate and missed a per-epoch rate 2.6x away in the
  Batch 2 test; with |m| it reaches ~3x).
- **D8 — (b):** the grouped reference fit evaluates every distinct
  exposure value exactly up to a cap of 100 distinct values (Batch 0c:
  at 114k vertices one table is 11 ms, 100 values batched 0.3 s, 1000
  values 3.0 s, and a reference fit needs ~39 table evaluations, so the
  cap adds ~12 s there); above it, exposures are grouped into 100
  quantile bins, each
  treated at its mean exposure, with a visible notice that the default
  prior used binned exposure. Only the default prior's centre and width
  are approximated; the fit's likelihood uses each observation's exact
  exposure.
- **Order:** commit this session's open work first, then Batch 0.

## 8. Changes from v1 (review findings)

- "A discrete graph or non-uniform exposure identifies the scale" was
  false (both reviews; measured 2.8e-17). The guard is now structural and
  exact; the SVD-ratio threshold is dropped as an error.
- Epoch fits are not scale-identified in the final epoch unless a rate is
  fixed or tied (new D7).
- The default epoch prior moves out of the builder, so model-factory
  callers (`epoch_model`, direct builder calls) do not raise (3.1).
- The sampler is key-free (v1's split key reused the original key).
- `lower` is threaded into internally built priors and the D4 check runs
  at construction from `positive_params`, so `positive_params=False` with
  `prior=None` keeps working.
- The free-exposure prior is rebuilt per slot, not copied from
  `pm_result.prior`; reference fit built from `user_fixed`, never
  `broadcast_fixed`.
- Batches 2 and 3 merged; inventory, gate list, and test design expanded
  (differing per-epoch truths, no timeout-based test, parent-mu
  assertions, 1e-6-floor measurement); baseline on a committed state;
  memory-cap runner named; exposure in cell 49 is 10000, not 1000.

## 9. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/epoch-default-prior-and-truncated-gauss-plan.md`
> (v2.1) and CLAUDE.md. All decisions are in section 7. Commit the session's open work first if I
> say so, then do Batch 0 on a branch (no source changes) and report the
> measurements and the inventory with planned dispositions before Batch 1.
> Then one batch at a time: implement, add the tests, run the gates through
> `.git/claude-scratch/runcap.py`, and report before the next batch.

State at writing: master at `d875149d` plus uncommitted work from this
session (`event_rate_coefficient` rename + plot_* epoch true_theta fix,
with tests `test_event_rate_coefficient.py`, `test_plot_epoch_true_theta.py`),
and the earlier combined plan `epoch-fit-prior-and-fd-floor-plan.md`.
