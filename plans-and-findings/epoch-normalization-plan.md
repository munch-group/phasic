# Normalise the epoch (daisy-chain) joint-probability likelihood — plan

Status: DRAFT v2 (after adversarial review round 1). Author: Claude,
2026-11-19. NOT signed off. Decisions D1-D4 in §7 are open.
Supersedes the censoring part of `overflow-observations-plan.md` (v3,
reviewed, DEFERRED by user decision 2026-11-19: normalise everywhere
first; censoring with overflow rows is a possible later efficiency gain).

## 0. Copy-paste prompt for a fresh session

> Read `/Users/kmt/phasic/epoch-normalization-plan.md` in full, including
> the review log (§11), then implement it batch by batch on branch
> `fix/epoch-normalization` in a worktree. Do not start a batch until the
> previous batch's gate has passed. Change only the code named in §5;
> edit existing tests only where §8 lists them. Memory-cap every run
> (RLIMIT plus RSS watchdog; one heavy run at a time). Gates are "no new
> failures against `b3-test-baseline.md`" plus the batch's own tests.

State snapshot: branch `master`, HEAD `57596461`, 22+ unpushed commits;
user edits in three notebooks; untracked `docs/pages/tutorial/test.py`
(reproducer: tied two-epoch fit gives 6.44 for true 5.0). Verified
helper scripts from the planning session (scratchpad, may be gone —
recreate under `experiments/dr_epochnorm_*.py` in Batch 0): two-epoch
closed form + brute-force check; benchmark of single-epoch vs
`epoch_starts=[0.0]`.

## 1. Problem and evidence

Joint-probability graphs cap outcome features (`reward_limit`,
`tot_reward_limit`); overflow mass goes to a trap, so outcome
probabilities sum to Z(theta) < 1. Over-cap observations cannot be
passed to inference (`KeyError`, `src/phasic/__init__.py:4320`), so all
data reaching a likelihood had its overflow rows removed. Because counts
only increase, "overflowed in some epoch" = "final outcome over the cap";
removal is a deterministic function of the outcome, so for independent
observations the correct likelihood is prod_i p_i(theta, alpha_i) /
Z(theta, alpha_i), per observation, across epochs and under tying
(reviewer-verified).

Who normalises (verified against source by two reviewers):

| path | normalised? |
|---|---|
| single-epoch `pmf_from_graph_joint_index` (`__init__.py` 9040, 9069, 9133, 9157) | yes |
| `probability_matching` (`probability_matching.py:232-247`) | yes |
| `bffg_log_prob` (`bffg.py`, "normalized_probs") | yes |
| epoch `_daisy_chain_svgd_model`, no exposure (`model()` 5003-5013) | NO |
| epoch, exposure (`_per_obs_core` 5207-5215 + vmap rule) | NO |

`sum` over t-vertices = table mass Z (trash vertices are not t-vertices;
earlier-epoch outcome mass is handed back through the aux partners,
~4760-4763; r_v is computed identically in C++ and in the exact block).

Measured on the `test.py` model (coal 5, mu 3 fixed, reward_limit 3,
11% overflow), 10 datasets x 10,000: tied 2-epoch raw 6.370 (sd 0.062),
normalised 4.984 (sd 0.074). Normalised epoch output equals the
normalised discrete table to <= 7e-16 (tied). With `epoch_starts=[0.0]`
the epoch model matches the n=2 closed form exactly (reward_limit and
tot_reward_limit). Two epochs (c 2->7, mu 0.7->1.5, t1 0.5): relative
error vs the two-epoch closed form 2.8e-4 at default granularity,
2.9e-5 at granularity 10000 (first order; normalising does not change
it). The closed form itself matches brute-force integration to 1e-14.

## 2. Goal and definition of done

Every `Graph.svgd(..., epoch_starts=...)` and `Graph.epoch_model(...)`
likelihood is the conditional likelihood prod_i p_i / Z(theta, alpha_i)
for `final_read='sojourn'` (the default). `final_read='stopprob'`: D2.

Out of scope, ticketed: `joint_stop_prob`-kind graphs passed to svgd
(their terminal set is not the outcome set, `svgd_config.py` 337/1206).

Done when:
1. Both branches (no exposure / exposure) normalise; tied and untied;
   `exact_final_grad` False and True; through `Graph.svgd`, and through
   `Graph.epoch_model` for what it supports (no tying, no
   exact_final_grad).
2. Outputs match INDEPENDENT closed forms (§6): n=2 (one feature) and
   n=3 (two features, separating per-feature from total caps), one epoch
   to <= 1e-12 relative, two epochs (c1 != c2) to a tolerance set by the
   granularity-convergence measurement in Batch 0, several alpha.
3. Exposure: per-alpha normalisation verified against the closed form
   with alpha in {0.5, 1, 3} (a shared-Z bug must fail it — shown in
   Batch 0).
4. Gradients of the summed log-likelihood (FREE slots only; fixed slots
   are 0 by contract) match central differences of an independent numpy
   implementation, FD and exact_final_grad, tied + fixed, also under
   `vmap(grad)`; tolerance accounts for the FD step 1e-7.
5. PRIMARY gate, deterministic (no Monte-Carlo error): with one rate
   fixed (single epoch and tied epochs identify only rate ratios),
   maximise the expected log-likelihood sum_k p_true(k) log p_impl(k|theta)
   with p_true from the closed form; the maximiser equals the truth
   within the granularity tolerance; raw and shared-Z variants miss by
   >= 10x that tolerance (Batch 0 proves detectability at the chosen
   alpha spread and overflow rate). Optimiser tolerance < 0.1 x that
   tolerance.
6. End-to-end gate: ~50 replicate MLEs on data from an INDEPENDENT
   simulator (exponential coalescence, Poisson mutations, own cap
   function, one alpha per row, numpy AND stdlib `random` seeded),
   |z| < 3.5 with Bonferroni across configurations.
7. SVGD check on the `test.py` tied model: posterior mean within 2
   posterior sd of the independent MLE (not HPD coverage — SVGD's known
   under-dispersion would confound it); prior 95% interval excludes the
   truth, manual step-size schedule.
8. Existing tests: no new failures vs `b3-test-baseline.md` except the
   §8 list, changed with sign-off. Single-epoch / probability_matching /
   bffg byte-identical.
9. Cost: median of repeated vmap(grad) x32 at ~45k vertices within noise
   of before (one reduction over n_t per call; no extra FFI call).

## 3. Non-goals

* Overflow rows / censored likelihood (deferred; CLAUDE.md note, Batch 3).
* Public readers (`daisy_chain_joint_probs`, `EpochContext`,
  `joint_prob_table`) stay raw — not likelihoods.
* Degenerate outcomes (several t-vertices per outcome; §9 item 3) — the
  conditional formula's numerator is then one vertex's mass. Not fixed
  here; every test model asserts it has none, and Batch 0 scans the
  svgd-* notebook models.
* §9 defects (separate tickets).

## 4. User-visible change

No API change. Epoch fits change likelihood from raw to normalised:
estimates move toward the truth when Z depends on theta; log-likelihoods
and LRT statistics change. Docstrings and changelog say: the likelihood
conditions on the observation lying within the graph's caps, so data
must be truncated at EXACTLY the graph's caps; when caps exceed every
observed count, normalising discards a little information the raw form
used (efficiency, not bias); single epoch / tied epochs identify only
rate ratios — fix one rate. No runtime warning (D3).

## 5. Design (all in `Graph._daisy_chain_svgd_model`, `__init__.py` ~4333-5420)

### 5.1 No-exposure branch
In `model()` (~5003-5013) replace `joint_probs[observed_pos_jnp]` with
`joint_probs[observed_pos_jnp] / jnp.sum(joint_probs)`. Outside the
custom_vjp: JAX differentiates the division; the FD backward (~4943-4956)
and the exact override (`cotangent @ J_final` over all n_t, ~4958-4994)
are unchanged (reviewer-verified). Applies after `_apply_tying`.

### 5.2 Exposure branch
* One helper `_normalise_rows(joint)` = `joint / joint.sum(-1,
  keepdims=True)`, used in `_per_obs_core` (~5211, on `(n_unique, n_t)`)
  and its vmap rule (~5229, on `(P, n_unique, n_t)`) before the gather.
  FD backward re-calls the core: unchanged.
* `exact_final_grad` exposure override (~5271-5305): `_efg_block_exp_np`
  returns `(raw_rows, Jb)`; `raw_rows[u] = r_v * soj * mass` at
  `th * scale[u]` (the block has r_v, soj and mass, ~4801-4804; add the
  product — it does not compute it today). Transform
  `Jn[u] = (Jb[u] - q[u] * S[u]) / Z[u]`, `Z = raw.sum(-1)`, `q = raw/Z`,
  `S = Jb.sum(axis=t)`; then gather as today. When the block's
  `mass == 0` early return fires (~4772-4782) it returns zero rows and
  zero J: define `Jn[u] = 0` where `Z[u] == 0` (preserves today's exact
  zero; reviewer-found 0/0). Tuple result works with
  `vmap_method='sequential'` (reviewer-verified). No extra `_dc_call` in
  the backward.

### 5.3 final_read='stopprob' — see D2
Normalising the stopprob read by its own sum also removes the penalty
for mass not yet absorbed at `t_eval`; as coalescence rates go to 0 the
conditional likelihood plateaus instead of vanishing (improper tail for
drifting particles). The `t_eval` probe checks residual mass only at
`probe_theta = ones`, absolutely, not at the fitted theta.

### 5.4 Numerics
NaN/inf at ANY t-vertex now propagates to every observation (it is in
the sum); `_log_lik_from_pmf` raises on NaN (`svgd.py` ~6248-6260), so
loud, not silent — documented. FD noise in dZ when Z ~ 1 is measured in
Batch 0 at n_obs = 1e4 and ~45k vertices. The probability-Jacobian
preconditioner (`svgd.py` 3739-3791) gets a ~zero column for any
direction that only scales Z (unidentified under the conditional
likelihood) — floored at 1e-8, acceptable; documented.

### 5.5 Test seam
`model._raw_forward` (no exposure: `theta -> joint_probs`; exposure:
`theta -> (n_unique, n_t)` raw rows) for tests. Not vmappable in the
exposure branch (a bare `_dc_call` under vmap is rejected by the C
handler) — docstring says so.

### 5.6 Untouched
`pmf_from_graph_joint_index`, `probability_matching`, `bffg`, C/C++,
public readers, svgd_config (unless D2 = reject), LRT code (consumes
`svgd.log_likelihood`).

## 6. Independent references (new `tests/pytest/test_epoch_normalization.py`)

* n=2, one epoch: P(k) = (1-q) q^k, q = 2 alpha mu / (c + 2 alpha mu),
  truncated at L and normalised (Z = 1 - q^(L+1)).
* n=2, two epochs (c1 on [0, t1), c2 after; m_e = 2 alpha mu_e;
  q_e = m_e / (c_e + m_e)):
  P(k) = (1-q1) q1^k * gammainc(k+1, (c1+m1) t1)
       + exp(-c1 t1) * sum_{j=0..k} Pois(k-j; m1 t1) (1-q2) q2^j,
  normalised over k <= L. (Verified vs brute-force integration to 1e-14;
  reduces to the one-epoch form when c1=c2, m1=m2.)
* n=3, one epoch, two features (singleton count k1, doubleton count k2).
  Phase 1 (3 lineages): coalescence rate 3c, singleton mutations 3 alpha mu,
  so the number n of phase-1 mutations is Geometric with
  a = alpha mu / (c + alpha mu). Phase 2 (one singleton + one doubleton
  lineage): coalescence rate c, mutations 2 alpha mu, total s Geometric
  with q = 2 alpha mu / (c + 2 alpha mu), each split singleton/doubleton
  with probability 1/2:
  P(k1, k2) = sum_{n<=k1} (1-a) a^n (1-q) q^s C(s, k2) 2^(-s),
  s = k1 - n + k2. Brute-force verified in Batch 0 before use (phase
  rates depend on how the model scales its coalescence rate — check
  against the builder); used to separate per-feature from total caps.
* Deterministic expected-log-likelihood maximiser gate (§2 item 5).
* Independent simulator + replicate MLE (§2 item 6).
* Gradient checks (§2 item 4). Zero-mass seam test with exposure
  (mirror of `test_exact_grad_daisy_final.py:300-343`).
* Assertion in every test model: no outcome maps to several t-vertices.
* Timing (§2 item 9), slow-marked.

## 7. Decisions for the user

* D1 — sign-off to modify `_daisy_chain_svgd_model` (§5.1, §5.2 incl.
  `_efg_block_exp_np` return value, `_raw_forward`) and change the §8
  tests.
* D2 — `final_read='stopprob'` (non-default; `'sojourn'` is exact and
  the default):
  (a) reject it for `svgd` / `epoch_model` fits with a message pointing
      to 'sojourn'; the private builder still offers it (tests use it as
      an independent reference). Affects
      `test_svgd_exact_final_grad_kwarg.py:301,310` (they expect a
      different error today — re-check). RECOMMENDED.
  (b) normalise and document as an approximation whose error grows for
      slow rates; add a slow-rate likelihood-profile test.
  (c) normalise by Z' = mass absorbed into outcomes + unabsorbed mass
      (keeps the unabsorbed penalty) — needs the FFI to return absorbing
      mass: a C/C++ change, own plan.
* D3 — no runtime warning; docstrings + changelog (recommended).
* D4 — §9 item 1 guard (reject continuous graph + finite
  `tot_reward_limit` + no epochs, silently wrong today): own ticket,
  next (recommended).

## 8. Existing tests that change (reviewer-derived; Batch 0 confirms by running)

WILL change:
* `inference/test_exact_grad_daisy_final.py:220` — rtol 1e-12 "plain
  FD" identity: backward becomes an analytic chain rule over FD of raw
  outputs vs FD of the normalised output; agree only to ~1e-9. Tolerance
  / semantics must change.
* `inference/test_exact_grad_daisy_final.py:178-185, 238-251, 270-283` —
  oracle `_oracle_per_obs` returns raw rows at observed positions only;
  must return the full t-vector and normalise.
MAY change (measure in Batch 0):
* `inference/test_svgd_exact_final_grad_kwarg.py:153, 263` (1e-4
  relative; normalised score has cancellation).
* `inference/test_fd_gradient_mixed_scale.py:225` (rtol 1e-3; its xfail
  at 240 stays xfail).
* `test_exposure_daisy_chain.py:822` (monotone in alpha; likely holds).
* D2(a): `test_svgd_exact_final_grad_kwarg.py:301, 310`.
Unaffected (call FFI/public readers directly, or compare two daisy
models / structure): `test_exposure_daisy_chain.py:170, 260, 549,
644-658, 1015-1069`; `test_epoch_sojourn_finalread.py:169, 194, 243,
265`; `test_lrt_at.py:68-70, 165-167`; `test_epoch_model.py:94, 126,
175`; `test_model_selection.py` epoch tests; `test_weight_formula_daisy.py:110,
121, 255`; `test_svgd_tied_export.py`; `test_daisy_chain_c_path.py:507+`;
`test_daisy_chain_svgd_kernel.py`; `test_exact_grad_daisy_final.py:189,
342, 470`.
Notebooks with epoch fits (`svgd-joint-prob`, `model-selection`,
`time-inhomogeneous`, `popgen/ghost`) show changed numbers when re-run;
the user re-runs them. Note `test.py`'s prior `ci=[5, 10]` contains the
truth — not usable as a recovery test as-is.

## 9. Pre-existing defects (separate tickets, not fixed here)

Status 2026-11-20: item 1 FIXED (merge `9cedb4f3`, commit `86e1f8ac`);
items 2-7 are tickets T3, T4, T5, T6, T7, T8 in
`joint-prob-followups-plan.md`, which tracks them from here. The exit
segfault seen in this plan's test runs is FIXED (merge `038dcc63`).

1. Continuous joint graph + finite `tot_reward_limit` + no epochs:
   single-epoch path returns wrong probabilities (trash edges on outcome
   vertices; n=2: 0.097 vs 0.177 for the capped row).
2. `joint_prob_table()` on a continuous graph returns probability / mu.
3. Degenerate outcomes mapped by weighted random pick, not group sum.
4. Silent standard-normal prior fallbacks (svgd DataPrior ~6393; daisy
   builder `probability_matching` ~5401, no notice).
5. `epoch_starts=[0.0]` build 43 s vs 1.2 s single-epoch at 45,704
   vertices (per-iteration cost within 6-12%) — profile.
6. `joint_stop_prob`-kind graphs through svgd: terminal set != outcome
   set.
7. No construction check that a joint table has >= 2 outcomes (a
   1-outcome table carries no information under normalisation).

## 10. Batches and gates

* Batch 0 — de-risk on the branch (prototype only, not committed as
  production): implement §5.1/§5.2 locally; recreate and verify the
  closed forms (n=2 one/two epoch, n=3 incl. brute force); granularity
  convergence -> tolerances; detectability of raw and shared-Z variants
  in the deterministic gate; FD-noise of dZ at scale; run the existing
  suite to fix the §8 list; median timing at ~45k vertices; degenerate-
  outcome scan. Findings: `epoch-normalization-findings.md`; plan
  revised and re-approved if anything moves.
* Batch 1 — no-exposure branch, `_raw_forward`, D2 handling. Gate: §6
  (no exposure), gradients, deterministic + replicate gates, §8
  no-exposure tests.
* Batch 2 — exposure branch incl. exact-final override and zero-mass
  rule. Gate: per-alpha closed form, vmap(grad) at both sites,
  exposure-correlated-dropout cases, zero-mass seam with exposure,
  remaining §8 tests, timing.
* Batch 3 — docs (docstrings, changelog, §4 text) and the CLAUDE.md note
  below. Gate: user review of the text.

CLAUDE.md note (to add in Batch 3, user-requested):

> **Joint-probability likelihood is conditional (normalised) on every
> path.** Observations beyond `reward_limit`/`tot_reward_limit` cannot be
> passed, so all paths use prod p_i / Z(theta, alpha_i); data must be
> truncated at exactly the graph's caps. Potential efficiency gain, not
> implemented: if over-cap observations are kept as an explicit overflow
> category, the full likelihood prod_in p_i x prod_over (1 - Z) is
> unbiased and about twice as informative (tied two-epoch test model: sd
> 0.036 vs 0.074). A reviewed design exists in
> `overflow-observations-plan.md` (v3, deferred). Constraints found: the
> single-epoch joint-index model reads sojourn = probability / mu on
> continuous graphs, so censoring must use the epoch path
> (`epoch_starts=[0.0]`), whose build is ~36x slower at 45k vertices
> though per-iteration cost is within 6-12%.

## 11. Review log

Round 1 (2026-11-19): two independent reviewers (code paths; statistics
and tests), read-only. Confirmed: §1 table; sum over t = Z; §5.1
placement and unchanged backward; §5.2 placement, transform
(J - qS)/Z, commutation with exposure scaling, tuple callback output;
weight-formula models use the same `model()`, callback rejected on the
epoch path; no other likelihood path bypasses `model()`; conditional
likelihood correct per observation, across epochs, under tying; n=2
closed form. Changes in v2:
* D2 rewritten: normalising stopprob removes the unabsorbed-mass penalty
  (plateau at slow rates) -> recommend rejecting stopprob for fits.
* §8 rewritten from the reviewer's per-test classification (v1 list was
  largely wrong; the real breakers are in test_exact_grad_daisy_final).
* Zero-mass exact-block 0/0 -> Jn = 0 rule + exposure seam test.
* Plan text fixed: the block has r_v, soj, mass but not their product.
* Tests: deterministic expected-log-likelihood gate replaces the
  Monte-Carlo-only gate (granularity bias ~ MC error at 200 reps);
  independent simulator (table-drawn data cannot catch a
  self-consistent wrong model); one rate fixed for identifiability; n=3
  closed form separates per-feature and total caps; two-epoch closed
  form added and verified (by me: 1e-14 vs brute force; phasic converges
  first order, 2.8e-4 -> 2.9e-5); gradient checks on free slots only;
  SVGD check vs MLE not HPD; timing by median.
* Documented: NaN propagation through the sum; FD dZ noise; preconditioner
  zero column; `_raw_forward` not vmappable; joint_stop_prob scoped out;
  degenerate outcomes asserted absent; data must be truncated at exactly
  the graph's caps; identifiability.
