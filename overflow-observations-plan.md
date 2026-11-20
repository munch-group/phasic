# Overflow observations for joint-probability inference — plan

Status: DEFERRED 2026-11-19 by user decision (normalise everywhere first:
see epoch-normalization-plan.md). DRAFT v3 (after review rounds 1-2). Author:
Claude, 2026-11-19. NOT signed off. Decisions D1-D5 in §7 are open.

## 0. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/overflow-observations-plan.md` in full, including
> the review log in §11, then implement it batch by batch on branch
> `feature/overflow-observations` in a worktree. Do not start a batch until
> the previous batch's gate has passed. Do not change code outside the
> files named in a batch without asking. Memory-cap every run (RLIMIT plus
> RSS watchdog; one heavy run at a time). Gates are "no new failures
> against `b3-test-baseline.md`" plus the batch's own tests, never "full
> suite green". Existing tests may be edited ONLY where §8 lists them.

State snapshot: branch `master`, HEAD `57596461`, 22+ unpushed commits;
user edits in three notebooks; untracked `docs/pages/tutorial/test.py`
(the reproducer: tied two-epoch fit returns 6.44 for true 5.0),
`docs/pages/tutorial/svgd-priors.ipynb`, `half-way-feedback.md`.

## 1. Problem, with evidence

`Graph.joint_prob_graph(..., reward_limit=k, tot_reward_limit=T)` caps
each outcome feature at k and the total at T. Mass that would exceed a cap
goes into one "trash" trap (`src/phasic/__init__.py` ~10413-10423), so the
outcome table holds mass Z(theta) < 1 and P(overflow) = 1 - Z(theta).

* Users cannot say that an observation overflowed: an over-cap row raises
  a bare `KeyError` (`__init__.py:4320`).
* Single-epoch path (`pmf_from_graph_joint_index`) divides by Z: the
  correct likelihood for data whose overflow rows were removed.
* Epoch path (`_daisy_chain_svgd_model`, `__init__.py:5003-5013`, exposure
  branch ~5207-5240) returns raw probabilities. Raw prod p_i is the correct
  likelihood only for COMPLETE data that happen to contain no overflow
  row. Applied to data whose overflow rows were removed — which is every
  dataset today, since over-cap rows cannot be passed — it biases the
  estimate toward parameters that minimise overflow.

Measured (truth coal=5, mu=3, reward_limit=3, ~11% overflow):

| estimator | mean | sd |
|---|---|---|
| single epoch, raw, overflow rows removed | 6.387 | 0.056 |
| single epoch, normalised, overflow rows removed | 5.003 | 0.067 |
| single epoch, censored, complete data | 4.997 | 0.047 |
| tied 2-epoch, raw (CURRENT epoch behaviour), overflow rows removed | 6.370 | 0.062 |
| tied 2-epoch, normalised | 4.984 | 0.074 |
| tied 2-epoch, censored | 4.990 | 0.036 |

Closed-form check (n=2 coalescent, K ~ Geometric, P(k) = (1-q) q^k,
q = 2 mu/(c + 2 mu)): the discrete table and the epoch model with
`epoch_starts=[0.0]` match it exactly for both `reward_limit=2` and
`tot_reward_limit=2`.

## 2. Goal and definition of done

Every joint-graph inference path uses the correct likelihood for the data
the user has, and users can supply overflow rows, with or without
exposure.

* `overflow='condition'` — overflow rows were removed (or never recorded)
  by exactly this graph's caps. Likelihood prod_i p_i / Z(theta, alpha_i).
  OVERFLOW rows present -> error.
* `overflow='censor'` — data are complete; overflow rows are present and
  marked. Likelihood prod_in p_i(theta, alpha_i) * prod_over (1 - Z(theta, alpha_j)).
  Requires `epoch_starts` (use `[0.0]` for a time-homogeneous model) —
  the same route rule R9 already prescribes for exposure.

Scope note: the single-epoch (no epoch_starts) path on a CONTINUOUS graph
with a finite `tot_reward_limit` is wrong today (§9 item 1) and is
REJECTED by new rule R37 rather than fixed here.

Done when:
1. Both modes work through `Graph.svgd` for: epoch_starts=[0.0]; two
   epochs untied; two epochs tied; each with and without exposure; each
   with `exact_final_grad` False and True. Through `Graph.epoch_model`
   (which supports neither tying nor `exact_final_grad`): epoch_starts=[0.0]
   and two epochs untied, with and without exposure. 'condition' also
   works on the single-epoch (no epoch_starts) path, which stays
   byte-identical.
2. Per-observation outputs match INDEPENDENT references (§6): to <= 1e-12
   relative for epoch_starts=[0.0] (no intermediate stop-probability
   step); for two or more epochs to a tolerance set by a granularity
   convergence check (earlier epochs use `stop_probability` with a
   granularity, first-order error ~1e-4..1e-3 at default granularity —
   measured in round-2 review), run at high granularity. References: the
   n=2 closed form (reward_limit and tot_reward_limit, several exposure
   values), and a forward simulator with its own capping function
   (chi-square incl. the overflow cell).
3. Gradients (FD and exact) match finite differences of the reference
   likelihood, including under `vmap(grad)`.
4. Primary correctness gate (not confounded by SVGD's known
   under-dispersion): over >= 200 replicate datasets, the mean score at the
   truth is ~0 and the replicate MLE is unbiased within Monte-Carlo error,
   for both modes, incl. a case where overflow correlates with exposure
   and rows are dropped (catches a single shared Z). The current raw
   estimator must FAIL the same test. Run on epoch_starts=[0.0] (exact);
   two-epoch cases only at a granularity where the convergence check
   shows the discretisation bias is far below the Monte-Carlo error.
5. Secondary: SVGD recovery tests (prior 95% interval EXCLUDES the truth,
   manually set step-size schedule) for tied epochs, with and without
   exposure, at >= 10% overflow; posterior width bounded.
6. Scale: censor-mode likelihood + gradient evaluated on one model of
   20k-60k vertices within a memory cap (finite, timed, recorded).
7. No new failures against `b3-test-baseline.md`, apart from the tests
   listed in §8, re-baselined with sign-off.

## 3. Non-goals

* Partial censoring ("singletons >= 4, others exact"): the graph merges all
  overflow into one trap.
* Censor mode on the single-epoch `pmf_from_graph_joint_index` model:
  that model reads expected sojourn times, which on a CONTINUOUS joint
  graph equal probability / mu (each outcome vertex exits at rate mu) and
  so are not probabilities (measured: table sums to 1.145 for n=2,
  reward_limit=2). Normalising cancels that factor, censoring would not.
  Censor goes through the epoch path, which reads probabilities exactly.
* MCMC. `bffg_log_prob` accepts joint graphs only with caller-supplied
  vertex indices, so OVERFLOW cannot reach it; unchanged.
* No raw (unnormalised, no overflow term) mode is offered.
* The pre-existing defects in §9 are separate tickets.

## 4. User-facing API

```python
from phasic import OVERFLOW

jpg = graph.joint_prob_graph(indexer, reward_limit=3, discrete=False)

obs = jpg.cap_observations(raw_counts)          # real data -> rows or OVERFLOW
obs = jpg_discrete.sample_joint_observations(   # simulation incl. overflow cell
          10_000, overflow=True, seed=1)

svgd = jpg.svgd(obs, overflow='censor', epoch_starts=[0.0],
                exposure=alpha, exposure_param_index=1, fixed=..., prior=...)
svgd.summary()   # "likelihood: censored; 8,903 in-table, 1,097 overflow rows"
```

### 4.1 `phasic.OVERFLOW`
Module-level singleton of class `_Overflow`: `__repr__` 'OVERFLOW';
`__reduce__` returns the module attribute (identity survives pickle,
SLURM, multiprocessing); `__eq__` by type and a matching `__hash__`.
Detection is `obs is OVERFLOW` (never `==` on rows: ndarray rows compare
elementwise). It replaces a whole row. Accepted inside lists and object
arrays; numeric arrays cannot hold it and never imply overflow.
A pandas DataFrame passed as observations (today silently iterates
column names, `__init__.py:4319`) -> `TypeError` with a fix hint.

### 4.2 `overflow=` on `Graph.svgd` and `Graph.epoch_model`
`None` (default) | `'condition'` | `'censor'` — see D1. Stored as
`svgd.overflow` (resolved value) and shown in `summary()`; recorded in the
assumption ledger when inferred.

### 4.3 `Graph.cap_observations(raw)`
Joint-prob graph only (rejects joint-stop-prob / joint-sojourn graphs and
graphs whose cap rule is not recorded). Per raw row (columns as in
`joint_prob_table()`):
* any entry > reward_limit or sum > tot_reward_limit -> `OVERFLOW`;
* else row must be a reachable outcome -> returned as a tuple;
* else `ValueError` naming the row (within caps but impossible, e.g. a
  non-zero `descendants_n`).
Negative / non-integer entries -> `ValueError`. `OVERFLOW` passes through.
Membership set built from t-vertex STATES only (no sojourn elimination,
weight-independent) so it is cheap at 200k-600k vertices.
Requires new attributes `_joint_reward_limit`, `_joint_tot_reward_limit`,
`_joint_cap_rule_valid` set by `joint_prob_graph` (False when a custom
`reward_rates_callback` was used, or dict `reward_limit`, a dead branch).
Propagated wherever joint metadata is copied by hand (`joint_stop_prob_graph`
~10953, `joint_sojourn_graph` ~11119) and by `clone()` if it copies joint
metadata (Batch 0 checks; `clone()` ~9608 currently drops it).

### 4.4 `Graph.sample_joint_observations(n, overflow, seed=None)`
DISCRETE joint-prob graph only (the continuous table is probability/mu,
§3). Fitting needs a CONTINUOUS graph (the epoch builder rejects
discrete ones, `__init__.py:4449`), so simulation studies build both
graphs from the same base graph and caps, as `test.py` already does;
the docs say so and Batch 0 confirms the two outcome row sets are
identical. The sampler has no exposure or epochs: it simulates the
time-homogeneous, exposure-free case only (documented). Exposure and
epoch data in tests come from the independent forward simulator (§6). Categories = table rows (+ overflow cell 1 - Z if `overflow=True`).
`overflow` has no default. `seed=None` uses global `np.random`, matching
`_map_joint_observations_to_indices(seed=None)`. (Note: `Graph.sample()`
uses stdlib `random`; this is a different stream.) Returns a list of tuples
and `OVERFLOW`.
`joint_prob_table(overflow=True)` is DROPPED (it would make every outcome
column nullable and break `.to_numpy().tolist()` consumers).

### 4.5 Error messages
Over-cap raw row passed to svgd/epoch_model -> `ValueError` pointing to
`cap_observations` (replaces the bare `KeyError`). OVERFLOW rows with
`overflow` None or 'condition' -> error giving the count and both
remedies.

## 5. Internal design

### 5.1 Mapping
`_map_joint_observations_to_indices(observed_data, *, seed=None,
allow_overflow: bool)` — `allow_overflow` REQUIRED (no default; the two
existing callers pass it). OVERFLOW -> internal index `-1` only when
`allow_overflow=True`; otherwise raise. Overflow-free input is
bit-identical (same RNG consumption for the existing probability-weighted
tie-break of degenerate outcomes, §9 item 3). Batch 1 lands this with
`allow_overflow=False` everywhere, so no `-1` can reach any model before
Batch 3.

### 5.2 Validation (config rules R33-R36, `svgd_config.py`)
* R33: `overflow` not None on a non-joint graph, or on joint-stop-prob /
  joint-sojourn graphs -> error.
* R34: OVERFLOW rows with `overflow` None/'condition' -> error (count +
  remedies).
* R35: `overflow='censor'` without `epoch_starts` -> error: "use
  epoch_starts=[0.0]" (mirrors R9).
* R36: `overflow='censor'` with `final_read='stopprob'` -> error (1 - sum
  then includes mass unabsorbed at t_eval).
* R37: a CONTINUOUS joint graph with finite `tot_reward_limit` and no
  `epoch_starts` -> error ("wrong probabilities on this path, §9 item 1;
  use epoch_starts=[0.0] or a discrete graph"). Changes behaviour of a
  currently silently-wrong path; needs sign-off (D5).
* censor with zero OVERFLOW rows: allowed, announced via `_svgd_assume`
  ("censored likelihood assumes no rows were removed").
Because `Graph.epoch_model` / `FreeEpochModel.fit` bypass the svgd
validator (`epoch_model.py` ~93-128), R34-R36 are ALSO enforced inside
`_daisy_chain_svgd_model`, `FreeEpochModel` stores and sets
`svgd.overflow`, and `epoch_model` emits the zero-overflow-rows notice
itself (`warnings.warn`, since that route has no assumption ledger).

### 5.3 Epoch model (`_daisy_chain_svgd_model`, ~4333-5420)
New keyword `overflow_mode` (D3 decides its default at this private
level). One shared helper `_overflow_transform(rows, mode)` applied to a
(…, n_t) array: condition -> rows / rows.sum(-1); censor -> concat(rows,
r) with r = 1 - rows.sum(-1). The helper runs inside jit/vmap-traced
code, so it cannot raise on values: if r < -1e-9 (mass conservation
violated) it returns NaN for that row (loud, not clipped away); else
`jnp.maximum(r, 0)`. Near Z -> 1 the log of the overflow term is kept
finite by the existing `_PMF_LOG_OFFSET` (1e-10) inside SVGD's log, and
`jnp.maximum` has a finite subgradient; Batch 3 tests the gradient at
very small mu with overflow rows present (FD relative error there is
recorded, not assumed). Conservation is asserted directly in tests on
the forward. Gather positions are checked against the transformed width
at BUILD time (concrete arrays; an out-of-range JAX gather clamps
silently).
* `observed_pos` (~4480-4495): `-1` -> `n_t` (censor only).
* No-exposure branch: transform applied in `model()` after `_autodiff`
  (outside the custom_vjp): JAX differentiates it; FD and
  `exact_final_grad` overrides unchanged.
* Exposure branch: the gather is inside custom_vjp/custom_vmap'd
  `_per_obs_core`. Apply the helper identically in `_per_obs_core`
  (~5207-5215) and its vmap rule (~5218-5233). FD backward re-calls the
  core: unchanged. `exact_final_grad` exposure override (~5271-5305):
  transform the per-unique block — censor: append `-sum_t J`; condition:
  `(J_i - (p_i / Z) * sum_t J) / Z` with raw p_i, Z per unique alpha. The
  forward rows needed by the quotient rule come from the exact-block
  callback itself (it already evaluates the sojourn vector; return it
  alongside J) — NO extra `_dc_call` in the backward (a bare `_dc_call`
  under SVGD's `vmap(grad)` gets default batching and a 3-D theta buffer
  the C handler rejects, cf. comment ~5194-5197).
* Daisy prior: when `user_prior is None` the builder calls
  `probability_matching(self, observed_indices, ...)` (~5397) inside a
  bare `except` that silently falls back to `prior=None`. Strip `-1`
  before that call (in-table rows only; it is a prior/initialisation).

### 5.4 Single-epoch path
`pmf_from_graph_joint_index` is NOT modified. `overflow` None/'condition'
keeps today's normalised behaviour byte-identically; R35 routes censor
to the epoch path.

### 5.5 Model comparison (`model_selection.py`)
LRT (`likelihood_ratio_test`, `likelihood_ratio_test_at`) and AIC/BIC
comparison require identical resolved `overflow`, `reward_limit`,
`tot_reward_limit` across fits (error otherwise). `n_observations` counts
overflow rows.

## 6. Independent test references (new `tests/pytest/test_overflow_*.py`)

* Closed form n=2 (as §1), with exposure alpha scaling mu: one epoch, and
  two epochs (piecewise-constant c: P(k) by conditioning on the
  coalescence epoch, written out in the test).
* Forward simulator: coalescent + Poisson mutations per branch class,
  piecewise-constant epochs, its OWN capping function (not
  `cap_observations`); chi-square of frequencies against model outputs
  including the overflow cell.
* Large-cap consistency: build at a large cap (Z ~ 1), sample, cap at the
  small cap with `cap_observations`, compare to the small-cap model incl.
  1 - Z (tests the cap mirror end to end).
* Enumeration: every table row classifies in-table; each row with one
  entry pushed past a cap classifies OVERFLOW; on scalar, tot-only and
  `reward_only` graphs.
* Score / replicate-MLE (§2 item 4), incl. exposure-correlated dropout.
* `vmap(grad)` through every transformed gather site (no-exposure
  `model()`, `_per_obs_core`, its vmap rule, exact-final exposure block).
* Byte-identity of the single-epoch default path and of overflow-free
  mapping, before vs after each batch.
* Pickle round-trip of OVERFLOW and of observation lists containing it
  (FreeEpochModel itself holds JAX closures and was never picklable).

## 7. Decisions for the user

* D1 — default of `overflow`. Recommended: `None`, resolved on joint
  graphs to 'condition' and recorded in the ledger/summary; any OVERFLOW
  row with None raises. (Follows the existing `joint_index=None` pattern,
  `__init__.py:6424`.) Effect: single-epoch results unchanged; epoch
  results change from raw to normalised (a bias fix, announced once).
  Alternative: required on epoch paths (asymmetric; edits every epoch
  call site in tests/notebooks).
* D2 — sign-off to MODIFY existing code (repo prefers additive):
  `_map_joint_observations_to_indices`, `_daisy_chain_svgd_model` (both
  branches + exact-final exposure override + prior stripping),
  `Graph.svgd` / `epoch_model` / `FreeEpochModel` plumbing, `joint_prob_graph`
  (store limits), `joint_stop_prob_graph` / `joint_sojourn_graph`
  (propagate), `model_selection.py`, `svgd_config.py` (new rules), `summary()`.
* D3 — default of the private builder's `overflow_mode`: 'condition'
  (24 direct test calls keep running; their pinned raw values are
  re-baselined, §8) vs required (24 call sites edited). Recommended:
  'condition'.
* D4 — order: Batch 2 (epoch normalisation, a live bias fix) ships before
  the censor feature. Recommended: yes.
* D5 — R37: reject the continuous + finite `tot_reward_limit` +
  no-epochs path (silently wrong today) until §9 item 1 is fixed.
  Recommended: yes. Also part of D2: a private test seam on the daisy
  model exposing the pre-transform forward (`model._raw_forward`), so
  existing raw-FFI regression tests keep guarding the chain (§8).

## 8. Existing tests expected to change (need sign-off; final list from Batch 0)

Daisy-model outputs move from raw to normalised. Re-baseline against the
normalised value, and keep the raw FFI chain guarded by asserting the
pre-transform forward (via the new `model._raw_forward` seam, D5)
against the old reference:
`test_exposure_daisy_chain.py` (~170, 549, 1002-1069),
`test_epoch_sojourn_finalread.py` (~169, 194; the stopprob ≈ sojourn
pins at 243/265 should still hold after both are normalised),
`inference/test_exact_grad_daisy_final.py`,
`inference/test_svgd_exact_final_grad_kwarg.py`. Recovery-value
assertions in epoch fits may shift (expected toward the truth).

## 9. Pre-existing defects found during review (separate tickets, not fixed here)

1. `tot_reward_limit` puts a trash edge on OUTCOME vertices
   (`fast_reward` runs on absorbing base states). On a CONTINUOUS joint
   graph the single-epoch path then returns wrong probabilities: n=2,
   tot_reward_limit=2, c=1, mu=0.7 gives [0.570, 0.333, 0.097] vs closed
   form [0.520, 0.303, 0.177]. Discrete graphs and the epoch model are
   unaffected (match the closed form). Likely fix: no trash edges at
   absorbing base states — a graph-structure change, needs its own plan.
2. `joint_prob_table()` on a CONTINUOUS joint graph returns
   probability / mu, not probability (sums to 1.145 in the n=2 example).
3. Outcomes that map to several t-vertices are resolved by a weighted
   random pick at the weights loaded at mapping time
   (`__init__.py:4319-4326`); the correct likelihood sums over the group.
   Batch 0 checks whether any shipped model is affected.
4. Silent prior fallbacks: `Graph.svgd` DataPrior (~6393, quiet-able
   notice) and the daisy builder's `probability_matching` (~5401, no
   notice at all) fall back to a standard-normal prior on any exception.

## 10. Batches and gates

* Batch 0 — de-risk, no production code. Scripts under
  `experiments/dr_overflow_*.py`, findings in
  `overflow-observations-findings.md`:
  (i) n=2 closed form vs epoch [0.0] and two epochs, with exposure;
  (ii) COST and memory of `epoch_starts=[0.0]` vs single-epoch at 20k-60k
  vertices on coalescent and two-locus joint graphs (memory-capped) —
  censor depends on this route;
  (iii) degenerate-outcome scan of the svgd-* notebook models;
  (iv) confirm the exact-final block can return its forward rows;
  (v) final list of tests pinning raw daisy outputs (§8);
  (vi) `clone()` and joint-metadata propagation;
  (vii) `reward_only` graphs: can overflow on a hidden (unrewarded-in-
  output) property be seen from the visible columns? If not,
  `cap_observations` rejects `reward_only` graphs;
  (viii) discrete and continuous graphs from the same base + caps have
  identical outcome row sets;
  (ix) granularity convergence of two-epoch outputs vs the two-epoch
  closed form, to set §2 item 2 tolerances.
  Gate: findings written; plan revised and re-approved if anything moves.
* Batch 1 — data plumbing: OVERFLOW, stored/propagated limits,
  `cap_observations`, `sample_joint_observations`, mapping with required
  `allow_overflow` (False everywhere), error messages.
  Gate: enumeration + large-cap tests, pickle, bit-identical mapping.
* Batch 2 — epoch 'condition' normalisation (both branches, exact-final
  exposure quotient rule), `overflow` kwarg (None -> condition), R33-R36,
  builder-side enforcement, summary/ledger, LRT guard, epoch_model
  plumbing, §8 re-baselines.
  Gate: §6 references for condition; score/MLE gate; tied recovery on
  the test.py model recovers 5; vmap(grad) sites; byte-identical
  single-epoch path.
* Batch 3 — censor mode on the epoch path (helper, observed_pos, both
  branches, exact-final), `allow_overflow=True` when censor, daisy-prior
  stripping.
  Gate: full §2 matrix against §6 references; score/MLE incl.
  exposure-correlated dropout; SVGD recovery; scale check (§2 item 6).
* Batch 4 — docs: when to use condition vs censor (condition is valid
  only if rows were removed by exactly this graph's caps; with real data,
  run `cap_observations` and report the overflow count); identifiability
  (single epoch identifies only mu/c; condition discards the overflow
  count's information); exposure needs epoch_starts. Tutorial text is
  proposed to the user, not written into their notebooks.

## 11. Review log

Round 1 (2026-11-19): three independent reviewers (statistics, code
paths, API/scope), read-only. Changes made in v2 in response:
* Single-epoch + exposure is unreachable via svgd (R9) -> scope rewritten.
* Censor on the single-epoch model dropped: continuous-graph sojourn =
  probability/mu (verified by me: ratio 1/0.7 on every row) -> censor
  routed through epoch_starts (R35).
* tot_reward_limit trash edges at outcome vertices (reviewer hypothesis;
  verified by me against the closed form) -> §9 item 1; epoch route shown
  unaffected.
* Daisy builder's own prior call would swallow -1 -> stripping (5.3).
* Extra backward `_dc_call` would break under vmap -> rows from the
  exact-block callback instead.
* Out-of-range JAX gathers clamp silently -> shared helper with width
  assertion + per-site vmap(grad) tests.
* epoch_model / FreeEpochModel bypass the validator -> rules duplicated in
  the builder.
* Mapping would leak -1 before models handle it -> required
  `allow_overflow`, False until Batch 3.
* stopprob + censor rejected (R36); joint-stop-prob/sojourn graphs
  rejected (R33).
* "No churn" claim was false -> D3 and §8.
* Tests were self-referential -> independent references (§6) and a
  score/MLE gate not confounded by SVGD under-dispersion.
* LRT guard extended to caps; 1 - Z clipped with conservation check;
  OVERFLOW gets `__hash__`, identity checks; DataFrame input rejected;
  `joint_prob_table(overflow=True)` dropped; sampler restricted to
  discrete graphs.
* Premise reworded: raw is correct for complete data with no overflow
  rows; the bias comes from dropped rows.
Not adopted: computing trash mass directly instead of 1 - Z (requires a
graph-structure change; the epoch read is exact to ~1e-15 on the
closed-form checks; revisit if Batch 0 scale runs show drift).

Round 2 (2026-11-19): one reviewer on v2. Confirmed sound: transform
placement in both branches, mixing exact final-epoch slots with FD
earlier slots, the quotient rule and `-sum J`, tie/fixed handling,
single-epoch byte-identity, no rule blocks epoch_starts=[0.0], prior
stripping needed. Changes in v3:
* 1e-12 tolerance only for [0.0]; two-epoch outputs carry first-order
  granularity error -> convergence-based tolerance (§2 items 2, 4; Batch
  0 (ix)).
* `epoch_model` supports neither tying nor exact_final_grad -> matrix
  narrowed (§2 item 1).
* Value-dependent raise impossible in traced code -> NaN on conservation
  violation + direct test assertions (§5.3).
* FreeEpochModel pickle test dropped (never picklable).
* Small-mu / Z -> 1 gradient behaviour specified and tested (§5.3).
* Two graphs needed for simulate-then-fit; sampler limits documented
  (§4.4); row-set identity checked (Batch 0 (viii)).
* Goal over-claimed for continuous + tot_reward_limit single-epoch ->
  R37 (D5).
* `reward_only` hidden-column overflow -> Batch 0 (vii).
* epoch_model zero-overflow notice; `_raw_forward` test seam (D5).
