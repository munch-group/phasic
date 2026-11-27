# Plan A (domain-aware finite-difference steps), Batch 0 findings

**Date:** 2026-11-27. **Branch:** `fix/fd-domain-steps` (worktree `../phasic-fdsteps`),
based on master `ff049679`. **No source change**: this batch only reads code
and measures. Plan: [fd-domain-steps-plan.md](fd-domain-steps-plan.md) v2.9,
§6 Batch 0.

All probe scripts, raw outputs and logs are in
`experiments/fd_domain_steps/batch0/`. Every run was memory-capped (runcap,
RLIMIT plus an RSS watchdog), one heavy run at a time, and none ran beside
the suite baseline.

## 0. Summary

1. **Fresh master suite baseline** taken with `gate2.sh` (tag
   `master_fdsteps`, in the worktree; 204 test files, all usable). 3032 passed,
   113 skipped, 5 failed. Four failures are the known always-failing plot
   tests (OI-060). The fifth, a last-bit difference in
   `test_log_prob_unified_bit_identity_zi`, passed 3 of 3 isolated reruns: a
   flaky test, now OI-121. Batch 1 and Batch 2 compare against this baseline.
2. **Census.** There are 12 `custom_vjp` definitions (11 in `__init__.py`, 1 in
   `ffi_wrappers.py`) and every one has a finite-difference (FD) backward. With
   the numpy loop in the joint-index host fallback, that makes 13 FD sites and 15
   loops, all matching the plan's §5 table and line numbers on current master. Two
   corrections to the plan's text (section 2).
3. **Refusal grid** (141 cells). It confirms the §4.4 model property with one
   real correction and two harmless over-predictions:
   - **`moments_from_graph` always refuses a negative probe**, even on a
     linear continuous graph. It builds through generated C++ whose add-edge
     check refuses weights of 0 or below, and today's gradient crashes there at a
     slot of 5e-8. §4.4 must list it beside `pmf_from_cpp`.
   - `reward_visit_probability` on discrete graphs, and the joint-index
     forward on a discrete joint graph, ACCEPT negative probes. §4.4 predicts
     refusal. Applying the fallback there is safe; it only changes numbers in the
     §2 band.
   - **New defect OI-119:** with rewards, a refused weight is reported one call
     late. The bad call silently returns a wrong value, and the NEXT call, at a
     valid theta, raises.
4. **Step sizes per forward family** (section 4). Applying the plan's
   acceptance criterion literally, s = 1e-4 passes in only one family and
   regime. The picks are 1e-5 (continuous Euler pdf), 3e-5 (multi-epoch,
   native discrete including the unit rule), 3e-6 (discretized, elimination)
   and 1e-5 (daisy chain with explicit granularity). At s = 1e-4 every family is
   still at or below about 2e-8 median relative error, against today's 4e-7 to
   5e-6 at population-genetics values. **Decided (user, 2026-11-27): s = 3e-5
   for every family** (section 4.3).
5. **The discretize slot has no jumps** (scale invariance and smoothness
   verified). It takes the capped rule with the discretized family's s.
6. **Degenerate values (§4.5).** These are confirmed:
   - the zero-difference band starts at theta ≈ 1.78e9 (numpy) / 1.80e9 (jit);
   - every forward accepts a slot of exactly 0 except log mode, so the one-sided
     [0, 1e-7] step is safe as planned.

   One wording correction: the capped up-probe can exceed theta + 1e-7 by up to
   2 ulp(theta) once theta ≳ 1e6.
7. **§7 additions:** `test_discrete_row_sum.py::test_gradient_at_the_lower_bound`
   pins today's crash and will flip in Batch 1. `test_likelihood_correctness.py`
   is confirmed NOT affected.

## 1. Baseline (plan §6 Batch 0, §8)

- Command: `gate2.sh /Users/kmt/phasic-fdsteps master_fdsteps` (one memory-capped
  pytest process per file, `-m "not slow"`). Output in the worktree:
  `.gate_master_fdsteps/` (status.tsv, per-file junit XML and logs, DONE).
- 204 files, every one usable (exit 0/1, XML present). 3032 passed, 113
  skipped, 5 failed.
- Failures:
  - `inference/test_plot_ci.py` ×4 (`test_ci_methods[hpd-HPD]`, `[kde-HDR]`,
    `[percentile-CI]`, `test_map_line_ends_at_map_estimate`): OI-060, as
    expected.
  - `inference/test_log_prob_unified_bit_identity.py::test_log_prob_unified_bit_identity_zi[theta0]`:
    `actual -175.787865334752 != expected -175.78786533475204` (diff 2.8e-14).
    Passed 3 of 3 isolated reruns (and in the 2026-11-23 baseline). This is a
    last-bit non-reproducibility on the zero-inflated (rewarded) path, most
    likely the same cause as OI-022 / OI-015. Registered as flaky, OI-121.
- For Batch 1/2 gates: compare with `gate_compare.py .gate_master_fdsteps
  <candidate> --new-file tests/pytest/test_fd_domain_steps.py`. Before calling
  a failure of `test_log_prob_unified_bit_identity_zi` a regression, rerun it in
  isolation.

## 2. Census of FD sites (plan §5, §6 enumeration)

`grep custom_vjp|defvjp` finds 37 lines in `__init__.py`, as review 5 counted:
11 definitions, 11 `defvjp` calls and 15 comment or docstring lines. It finds 2
lines in `ffi_wrappers.py`. The four `custom_vmap` rules (`:5272`, `:5583`,
`:9769`, `:9789`) and the `profiling.py` string list are resolved-not-FD. All
12 `custom_vjp` backwards are FD, some mixed with an exact path.

| # | Site (definition → loop) | Entry point | Slots probed | Step / divisor | Clamp / guard | Forward family |
|---|---|---|---|---|---|---|
| 1 | `__init__.py:4078` → loop `:4091` | `pmf_from_graph`, `weight_mode='callback'` branch | all (no fixed-slot info) | ±1e-7 / literal 2·eps | none | Euler pdf or discrete pmf (GraphBuilder on the callback-materialised graph) |
| 2 | `:4198` → `:4215` | `pmf_from_graph`, parameterized FFI branch (linear, log, formula) | all | ±1e-7 / 2·eps | none | Euler / discrete (`compute_pmf_ffi`, granularity 0) |
| 3 | `:4588` → `:4604` | `pmf_from_cpp` | all | ±1e-7 / 2·eps | none | Euler / discrete (JIT library) |
| 4 | `:5287` → `:5297`; Batch H override → `:5331` (earlier-epoch slots only; final-epoch slots exact) | `_daisy_chain_svgd_model`, no exposure | all but `fd_skip_indices` (truly fixed; tied slaves probed) | ±1e-7 / 2·eps | none | multi-epoch |
| 5 | `:5619` → `:5638`; override → `:5685` | `_daisy_chain_svgd_model`, exposure | as 4 (probe is on unscaled theta; the forward multiplies the exposure column by alpha) | ±1e-7 / 2·eps | none | multi-epoch |
| 6 | `:7994` → `:8010` | `moments_from_graph` | all | ±1e-7 / 2·eps | none | elimination via generated C++ (**refuses**, section 3) |
| 7 | `:8650` → `:8664` | `pmf_and_moments_from_graph`, callback-mode cdf at zero | all but `_fixed_dims` | +1e-7 / minus probe `max(theta-1e-7, 1e-9)`, divisor = actual difference | clamp 1e-9; zero-difference guard | Euler + reward transform |
| 8 | `:8932` → `:8946` | `pmf_and_moments_from_graph`, cdf at zero (pybind) | all but `_fixed_dims` | as 7 | as 7 | as 7 |
| 9 | `:8968` → `:9069` | `pmf_and_moments_from_graph`, main | all but `_fixed_dims` | as 7; moments FD term replaced by the exact Jacobian where `_exact_ok` | as 7; `nansum` on the pdf | Euler + elimination |
| 10 | `:9852` → `:9896` (`_fd_theta_bar`) | `pmf_from_graph_joint_index` (used on static exclusion or probe failure; otherwise exact) | all but `fixed_mask` | ±1e-7 / 2·eps | none | elimination |
| 11 | numpy loop `:9551` (inside the exact-Jacobian host callback) | joint-index host fallback, `exact_grad_decline='fd'` | all (no fixed skip) | ±max(\|theta\|·1e-6, 1e-10) / 2h | none | elimination (pybind `update_weights` + `expected_sojourn_time`) |
| 12 | `:12029` → `:12039` | `daisy_chain_joint_probs` | all but `fixed_indices` | ±1e-7 / 2·eps | none | multi-epoch |
| 13 | `ffi_wrappers.py:1278` → `:1299` | `reward_visit_probability` backward (`Graph.reward_visit_probability`; zero inflation's FFI branch) | all | ±1e-7 / 2·eps; output is a scalar, so the unsummed `cotangent * (fp - fm)` is correct | none | elimination (GraphBuilder.build + backward probabilities) |

Fixed-slot information: sites 4, 5, 7, 8, 9, 10 and 12 have it; sites 1, 2, 3, 6
and 13 do not (plan §4.6, confirmed). Site 11 probes all slots, as the plan
states.

Two corrections to the plan's text:

- **§1 and §5: the 1e-9 minus-probe clamp is unconditional and sits at three
  sites, not two.** It is at sites 7, 8 and 9 (`:8664-8679`, `:8946-8962`,
  `:9069-9076`). It fires whenever theta < 1e-7 + 1e-9, whatever
  `positive_params` is; `positive_params=False` only makes negative theta
  possible. Because of it, the `pmf_and_moments_from_graph` sites never probe
  below zero for theta >= 0 (grid, section 3: every one survives a slot of 5e-8
  in every weight mode and discreteness). §4.4's fallback therefore replaces the
  clamp secant at all three sites on refusing models, as §2 goal 2 already
  allows.
- **§4.4's `_rewards_provided` line** is `__init__.py:9021` on master
  `ff049679` (unchanged from `003559ed`). An empty array still means no
  rewards.

Other plan citations re-checked on master: the daisy negative-rate check is
`graph_builder_ffi.cpp:1620` (`kMinEdgeWeight = -1e-12`), tested at `:1919`,
`:2157` and `:2204`, with messages at `:2000` and `:2242`. Note the test is
`< -1e-12`, so a weight in [-1e-12, 0) is accepted. The log-mode refusal is
`graph_builder.cpp:620` / `phasic.c:5748`; the formula refusal
`phasic.c:5706-5718`; the discrete row-sum wall `phasic.c:6771`. The add-edge
check moved from `phasic.c:4938` to **`:4957`**.

## 3. Refusal grid (plan §4.4, §6)

Scripts `g1_refusal_grid.py` (single-graph sites), `g2_refusal_joint.py`
(joint and epoch sites) and `g3_summarise.py`. Raw output is in
`g1_refusal_grid.jsonl` and `g2_refusal_joint.jsonl`; the full table is in
`g3_summary.txt`.

For each cell three things are recorded:

- P: the forward at slot value v in {1, 5e-8, 0, 1-5e-8};
- N: the forward at a negative probe in {-1e-12, -1e-9, -5e-8, -1e-7}, which
  is the model property (the refactor plan's phase-1 matrix uses these
  probes, so its overlapping cells can be checked against this table);
- G: today's `jax.grad` at v.

Axes:

- builder: graph-based, `pmf_from_cpp`;
- rewards: none, 1-D (`pmf_and_moments_from_graph`, its cdf at zero), 2-D
  (multivariate);
- weight mode: linear, log, formula (`c0*t0 + c1*t1`, division, sqrt),
  callback (linear, sqrt);
- discreteness: continuous, native DPH, was_dph (`discretize()`, slot 2 = the
  discretize slot);
- joint and epoch sites: continuous and discrete joint graph.

The model is a 4-vertex chain whose test slot sits alone on one edge (log
mode: every edge carries both slots).

**Result: 141 cells, of which 131 agree with §4.4.** The 10 that do not:

| Cells | Measured | §4.4 predicts | Meaning |
|---|---|---|---|
| `moments_from_graph`, linear continuous | refuses every negative probe; today's gradient crashes at 5e-8 ("Edge weight evaluates to non-positive value") | accepts | **Plan correction:** list `moments_from_graph` as refusing by construction, like `pmf_from_cpp` (generated C++, same add-edge check). It accepts a slot of exactly 0 (`a4_theta_zero.log`), so the one-sided step works there. It is linear-only: other modes raise at build ("Use Graph.pmf_and_moments_from_graph()"). |
| `reward_visit_probability` on native DPH and was_dph graphs (3 cells) | accepts | refuses (discrete) | Over-prediction. The forward builds the graph without discrete row validation. Applying the fallback is safe but changes the band numbers, so bit-identity tests must avoid the band there too. |
| joint-index FD loop and host forward on a DISCRETE joint graph (2 cells) | accepts | refuses (was_dph) | Same as above. |
| linear + 1-D rewards (`pam_pybind_rw1d`, cdf at zero; continuous, DPH, was_dph; 4 cells) | "partial": refuses -1e-9 and -1e-7, accepts -1e-12 and -5e-8 | refuses | Not a threshold: the refusal is DEFERRED by one call (OI-119, below). The model does refuse, so §4.4 is right. |

Everything else agrees, including:

- log, formula and callback modes refuse, through several different messages
  (log products, formula "finite and non-negative", add-edge check);
- native DPH and was_dph refuse;
- `pmf_from_cpp` refuses, continuous and discrete;
- every daisy site refuses: no exposure, exposure, `exact_final_grad` (its
  earlier-epoch slot crashes; its final-epoch slot is exact and does not),
  explicit granularity, and `daisy_chain_joint_probs`;
- linear continuous `pmf_from_graph`, `pmf_and_moments_from_graph` (pybind
  and FFI) and the single-epoch continuous joint index accept negative probes;
- `reward_visit_probability` on linear continuous graphs accepts.

Today's gradient at a slot of 5e-8 (G column) crashes on every daisy site,
every non-linear `pmf_from_graph` and `reward_visit_probability`, native and
was_dph `pmf_from_graph`, `pmf_from_cpp` and `moments_from_graph`. It survives on
every `pmf_and_moments_from_graph` site, through the clamp.

**New defect OI-119, deferred refusal with rewards** (`a1_anomalies.log`,
`a2_state_leak.log`, `a3_rewards_band_grad.log`):

- On `pmf_and_moments_from_graph` with 1-D rewards, pybind and FFI paths alike,
  a call at an invalid theta (a negative edge weight) is NOT refused. It
  silently returns a wrong answer: an Exp(1) density with moments [1, 2] for
  the test chain.
- The refusal is then raised by the NEXT call, at a perfectly valid theta, with
  the old message ("non-positive value '-0.000000'").
- Repeated calls at valid theta are bitwise consistent, with or without
  rewards, so normal use is unaffected.
- Today's FD backward never probes below zero there (clamp), so today's
  gradients are right: 6.0e7 against a positive-only central difference
  converging to 6.02e7.
- The hazard is a direct forward at an invalid theta, or any future rule that
  probes below zero with rewards.
- The same class as the fixed add-edge `ptd_err` leak (`test_ptd_err_no_leak.py`),
  on the reward-transform path.

**For Batch 1:** the band fallback keeps rewarded probes non-negative, so FD
never triggers OI-119. But a test that deliberately calls an invalid theta
with rewards will see the error on its following call. Use a fresh model per
assertion.

**Other observations:**

- `Graph.reward_visit_probability` does not work on callback-mode graphs at
  all ("Unknown weight_mode: callback" from the GraphBuilder parser). Zero
  inflation avoids it in callback mode (it uses `_cdf_zero_fn`), so the rvp FD
  site never sees a callback model. Registered as OI-120.
- `pmf_from_cpp` and `moments_from_graph` do not build on this non-editable
  install without `PHASIC_SOURCE_DIR` (already OI-058 / OI-089). Those cells
  were measured with `PHASIC_SOURCE_DIR` set to the worktree.
- Memory: the multivariate model on the division-formula chain peaked at 5.3 GB
  (4 vertices). The reward-transformed coalescent at theta = 1e-5 with times up
  to ~9e5 exceeded 6 GB and was killed. Both are long Euler runs (about 1e8 and
  9e8 steps), so the Euler forward's memory appears to grow with step count.
  This is not investigated; registered as OI-122. It matters for the
  "slow band" gap recorded in CLAUDE.md.

## 4. Step size per forward family (plan §4.1-4.3, §6)

Script `s1_step_sizes.py` (one model per process), summary `s2_summarise.py`
→ `s2_summary.txt`. Rules compared on the log-likelihood-type scalar
L = sum(F/F(theta0)):

- today's ±1e-7;
- the full positive rule theta·e^(±s);
- the capped rule u = min(log1p(1e-7/theta), s);
- for the native DPH probability slots, also the unit (logit) rule, full and
  capped.

The scan grid is s ∈ {1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6}.

References:

- **exact** where phasic has one: the moments Jacobian, the joint-index
  sojourn adjoint, and the epoch final-epoch block;
- otherwise **Richardson** extrapolation of log-coordinate central
  differences, at relative steps 1e-2, 5e-3 and 2.5e-3.

Where both references exist they agree to 4e-14 to 1e-8 (median ≤ 1e-9). The
largest Richardson spread is 8e-9.

Models and points (times in natural units, rate·t from 0.05 to 3):

| Family | Models | Ordinary | Population genetics |
|---|---|---|---|
| Continuous Euler pdf (capped) | coalescent (5 samples), two-island (3), island with split populations (3 slots), reward-transformed coalescent | theta ~ 0.05-7 | 1e-5 to 1e-4 |
| Elimination (full) | moments of coalescent (10), two-island (4), split islands; joint-index sojourn (4 samples) | as above | as above |
| Multi-epoch, granularity 0 (capped) | daisy chain, 2 epochs × 2 slots | [1, .5, 1.5, .3], [.3, 2, .6, 1] | [1e-4]*4 |
| Multi-epoch, explicit granularity 200 (full) | same | as above | not run |
| Native discrete (capped; plus unit rule) | two-island read as per-generation probabilities | probabilities .02-.1 | 1e-5 to 1e-4 |
| Discretized, was_dph (capped) | coalescent(5).discretize(1.0); slot 1 = discretize slot | [1, .5], [.3, 2] | [1e-4, 1e-4], [1e-5, 3e-5] |

Incomplete points, stated:

- `euler_reward` at theta = 1e-5 was killed at the 6 GB cap (OI-122), so
  Euler/popgen uses 10 slot-points, not 11.
- The epoch model's second population-genetics point ([1e-5, 1e-4, 3e-5, 1e-4])
  hit the 1-hour timeout, so epoch/popgen has 4 slot-points.

One fixture was corrected and rerun: the discretized model's first jump grid
included structurally-zero pmf entries holding rounding noise (~1e-17),
amplified ~1e16 by 1/F. See `a5_wasdph_scan.log`. That was a fixture problem,
not a phasic one.

### 4.1 Results (median relative error; full tables with maxima in `s2_summary.txt`)

| Family / regime | today | s=1e-3 | 3e-4 | **1e-4** | 3e-5 | 1e-5 | 3e-6 | 1e-6 | criterion pick |
|---|---|---|---|---|---|---|---|---|---|
| Euler, ordinary (cap) | 4.2e-9 | 8.6e-9 (flat: the cap binds) | | | | | | | any |
| Euler, popgen (cap) | 6.3e-7 | 2.4e-7 | 2.2e-8 | **6.6e-9** | 6.0e-9 | 7.3e-9 | 3.0e-8 | 8.3e-8 | 1e-5 (1e-4 also passes) |
| Elimination, ordinary (full) | 9.0e-10 | 2.0e-7 | 1.8e-8 | **2.7e-9** | 2.7e-10 | 3.8e-11 | 5.0e-11 | 8.8e-11 | 3e-6 (best 1e-5) |
| Elimination, popgen (full) | 1.7e-6 | 2.5e-7 | 2.3e-8 | **2.5e-9** | 2.6e-10 | 3.0e-11 | 5.9e-11 | 1.9e-10 | 3e-6 (best 1e-5) |
| Multi-epoch, ordinary (cap) | 3.3e-9 | 2.4e-9 (flat) | | | | | | | any |
| Multi-epoch, popgen (cap) | 1.5e-8 | 2.3e-7 | 2.1e-8 | **1.9e-9** | 2.9e-10 | 4.0e-9 | 4.7e-9 | 2.1e-8 | 3e-5 |
| Explicit granularity, ordinary (full) | 1.3e-9 | 1.2e-7 | 1.1e-8 | **1.2e-9** | 1.2e-10 | 3.4e-11 | 7.4e-11 | 6.2e-10 | 1e-5 |
| Native discrete, ordinary (cap) | 8.0e-10 | 3.6e-10 (flat to 3e-6) | | | | | | 7.7e-10 | ≥3e-6 |
| Native discrete, popgen (cap) | 4.7e-6 | 2.0e-7 | 1.8e-8 | **1.2e-9** | 2.6e-10 | 2.4e-9 | 1.7e-8 | 5.8e-9 | 3e-5 |
| Native discrete, ordinary (unit cap) | — | 8.1e-10 | 6.7e-10 | **5.3e-10** | 2.2e-10 | 5.0e-10 | 1.9e-10 | 4.2e-10 | 3e-5 or 3e-6 |
| Native discrete, popgen (unit cap) | — | 2.0e-7 | 1.8e-8 | **2.0e-9** | 2.1e-10 | 1.5e-9 | 3.9e-9 | 1.1e-8 | 3e-5 |
| Discretized, ordinary (cap) | 5.2e-10 | 2.0e-10 (flat) | | | | | | | any |
| Discretized, popgen (cap) | 4.4e-6 | 9.0e-7 | 8.1e-8 | **9.0e-9** | 8.2e-10 | 8.6e-11 | 3.6e-11 | 2.2e-10 | 3e-6 |

"Flat" means the capped rule's u = 1e-7/theta is below every s on the grid at
these theta values, so s does not matter there. The population-genetics regime
decides the capped families.

### 4.2 Reading

- **Every family is smooth in theta at these points.** There are no jumps: Euler
  max exit rates stay below 512, and was_dph scale invariance holds to the last
  printed digit over seven decades (`a5_wasdph_scan.log`). The error follows
  the s² truncation curve down to s ≈ 1e-5. Below that the forward's rounding
  noise (≈ eps_fwd/s) takes over, earlier on the long Euler and epoch runs at
  population-genetics times.
- **The new rules beat today's at population-genetics values by 2-3 orders of
  magnitude at any s from 1e-4 down.** Today's ±1e-7 is a relative step of
  1e-3 to 1e-2 there; it gives 6e-7 (Euler), 1.7e-6 (elimination) and 4-5e-6
  (discrete). At ordinary values today's rule is already good (1e-9), and
  the capped rule matches it, as §4.3 predicts (~1e-8).
- **The plan's starting value s = 1e-4 fails the literal criterion** ("the
  smallest s within 2x of the best on the grid") in every family except
  Euler/popgen. The gap is 4.6x (native discrete) to 250x (discretized), but
  in absolute terms it is ≤ 9e-9 median.
- **Tails:** smaller s has worse worst cases where forwards are long. For
  Euler/popgen the max error is 1.5e-7 at 1e-4, 7.9e-8 at 3e-5 and 9.3e-7 at
  1e-5. For elimination/popgen it is 4.4e-9 at 1e-4, 7.6e-9 at 1e-5 and
  9.1e-8 at 1e-6.
- **The discretize slot** behaves like the other was_dph slot (no jumps, the
  same errors). It takes the discretized family's capped rule and s. Its
  negative probe is refused (grid), so the capped rule is the right one.
- The **unit (logit) rule** on probability slots behaves like the positive
  rule, with the same pick (3e-5 at population-genetics values).

### 4.3 Step size: the criterion's picks, and the choice

Literal application of the plan's criterion (both regimes must pass; a flat
regime does not constrain):

| Family | Rule (§4.1) | s by the criterion | median at that s, popgen |
|---|---|---|---|
| Continuous Euler pdf | capped | **1e-5** (1e-4 and 3e-5 also pass) | 7.3e-9 |
| Multi-epoch, granularity 0 | capped | **3e-5** | 2.9e-10 |
| Native discrete (positive and unit) | capped | **3e-5** | 2.6e-10 |
| Discretized (was_dph), incl. the discretize slot | capped | **3e-6** | 3.6e-11 |
| Elimination | full | **3e-6** (1e-5 is the best; 3e-6 is within 2x, at 1.97x) | 5.9e-11 |
| Multi-epoch, explicit granularity | full | **1e-5** (ordinary only measured) | — |

Two defensible options for Batch 1, differing by at most ~1e-8 median error:

- **(a) Per-family values as above** (the criterion as written).
- **(b) One value, s = 3e-5, for every family.** It is within 2x of the best
  for Euler, epoch and native discrete. It costs 7x-23x the best median for
  elimination and discretized (2.6e-10 / 8.2e-10), with better tails than the
  smaller picks. It is simpler to implement and test.

**Decision (user, 2026-11-27): option (b), s = 3e-5 for every family**, for both
the full and the capped rule (positive and unit domains), including the
discretize slot. The reasons:

- Every candidate is at or below 1e-8 median error, so accuracy does not
  separate them.
- The finite-difference error is truncation (≈ s², nearly size-independent)
  plus forward rounding noise divided by s, which grows with model size.
- These measurements were on models of at most ~1,000 vertices, while
  production models are 20k-600k, so the best step at scale is likely larger
  than measured. The small per-family picks (3e-6) are the most exposed.
- 3e-5 is the best or within 2x of it where forwards are long (continuous pdf,
  multi-epoch, native discrete), costs at most ~1e-9 elsewhere, and has better
  worst cases than smaller steps (continuous pdf at population-genetics values:
  max 7.9e-8 at 3e-5 against 9.3e-7 at 1e-5).
- One constant is simpler to implement and test.

This knowingly departs from the literal "smallest s within 2x of the best"
criterion, which ignores tails and model size.

## 5. Degenerate values (plan §4.5)

`d1_degenerate_arith.py` (probe arithmetic) and `a4_theta_zero.py` (forwards at
0).

- **Zero-difference band of the capped positive rule (s = 1e-4):**
  - the up-probe equals theta from 9.0e8;
  - both probes equal theta, so the difference is 0, from **1.78e9 (numpy)**
    and **1.80e9 (jnp under jit)**;
  - between 1e9 and 2e9 the difference is zero at 15-17% of grid points, and
    above 2e9 at all of them;
  - u at 2e9 is 5.0e-17.

  This matches the plan ("above ~2e9"; "at 1e9 the down-probe still moves").
  The full rule never collapses (tested to 1e300).
- **Up-probe overshoot (§4.2 wording):** the plan says the capped up-probe is
  "never more than 1e-7 above theta". In floating point it is at most
  1e-7 + 2 ulp(theta). It exceeds 1e-7 by ~1e-10 from theta ≈ 1e6, and by up
  to 1.4e-7 near theta ≈ 8e8, where the maximum overshoot is 2.38e-7. This is
  harmless: the row-sum wall concerns theta ≤ 1, and the Euler granularity jumps
  are integer crossings of 2·max rate. Suggested wording: "at most 1e-7 above
  theta, up to rounding (2 ulp)".
- **Near-zero and near-one probes:**
  - the capped positive rule's probes stay positive down to theta = 1e-300;
  - the capped unit rule's probes stay inside (0, 1) from 1e-12 to 1 - 1e-15,
    with a nonzero difference;
  - the unit up-probe overshoot is at most 9.9995e-8.
- **theta == 0 (one-sided [0, 1e-7] step):** with the slot as an exit beside
  another (so 0 is a valid model), the forward accepts exactly 0 in linear,
  formula and callback modes, continuous and discrete. That holds for
  `pmf_from_graph`, `pmf_and_moments_from_graph` with and without rewards,
  `moments_from_graph` and `pmf_from_cpp`. **Only log mode refuses 0**, as §4.5
  states.

  Today's gradient at 0 crashes in `pmf_from_graph` (discrete, formula,
  callback), `moments_from_graph` and `pmf_from_cpp`. The
  `pmf_and_moments_from_graph` sites survive through the clamp. Batch 1
  should test a slot at 0 on a model where 0 is valid: in the grid's own
  chain, a slot at 0 is a vertex's only exit, so the forward itself fails
  there ("Computation produced NaN ... numerical catastrophe").

## 6. Tests that replicate or pin today's rule (plan §7)

- **Will change in Batch 1 (not in the plan's §7):**
  `tests/pytest/test_discrete_row_sum.py::test_gradient_at_the_lower_bound`
  (`:369-380`). It asserts that `jax.grad` at theta = 5e-8 on a native discrete
  `pmf_from_graph` model RAISES "negative weight". Native DPH is refusing, so
  the §4.4 fallback removes that crash and the test must be rewritten to assert
  a finite gradient. Listed for your approval with Batch 1. The neighbouring
  `test_gradient_at_the_bound` (upper row-sum wall at 1 - 5e-8) stays, as §4.1
  says.
- **Must stay bit-exact (confirmed, no edit):**
  - `test_daisy_chain_c_path.py:340`, `:797`;
  - `inference/test_exact_grad_daisy_final.py:224`;
  - `inference/test_exact_grad_rewards.py:106-111`.

  These are undeclared direct builders at ordinary theta.
  `test_exposure_daisy_chain.py:1048` only mentions eps in a docstring.
- **`test_likelihood_correctness.py`: NOT affected.** `_shipped_and_true_grad`
  and `test_loglik_gradient_degrades_only_below_1e8` use
  `pmf_and_moments_from_graph` on a linear continuous graph without rewards.
  That is a non-refusing undeclared model, so Batch 1 keeps today's code there
  bit for bit, and the plan's conditional item for this file is dropped.
- `test_fixed_values_exact.py:363`, `:418`: comments that name the 1e-7 step;
  no assertion depends on it.
- `inference/test_fd_gradient_mixed_scale.py`: `:229` flips in Batch 1, as
  planned; `:112`, `:132` unchanged; `:141-159` is a test oracle (out of
  scope).
- Forward-level refusal tests (`test_weight_formula_kwarg.py:140`,
  `test_omp_exception_safety.py:149`) use theta that is invalid at every scale;
  they are unaffected.

## 7. What Batch 1 needs from this (proposed plan amendments)

1. §4.4: add `moments_from_graph` to the always-refusing class beside
   `pmf_from_cpp`. **APPROVED (user, 2026-11-27); folded into the plan as
   v2.10.** Note that rvp on discrete graphs and the joint index on
   discrete joint graphs over-predict, harmlessly.
2. §1, §5: the clamp is unconditional, at three sites (section 2).
3. §4.2: the overshoot wording (section 5).
4. §4.1 / §6: **s = 3e-5 for every family** (user decision 2026-11-27,
   section 4.3), replacing the starting value 1e-4.
5. §7: add `test_discrete_row_sum.py::test_gradient_at_the_lower_bound`; drop
   the conditional `test_likelihood_correctness.py` item.
6. Batch 1 tests: a fresh model per rewarded invalid-theta assertion (OI-119);
   theta = 0 tests on models where 0 is valid.

**All items above APPROVED by the user (2026-11-27) and folded into the plan as v2.10.**

## 8. Register changes (`OPEN-ITEMS.md`)

- **OI-119** (new defect): deferred refusal with rewards.
- **OI-120** (new defect): `reward_visit_probability` fails on callback-mode
  graphs.
- **OI-121** (new, tests): `test_log_prob_unified_bit_identity_zi` is flaky in
  the last bit.
- **OI-122** (new follow-up): Euler forward memory grows with step count.
- The Priorities section and the refactor plan's §5 row: Plan A Batch 0 done.

## 10. Batch 1 (implementation, 2026-11-27)

Commit `a7fa7839` on `fix/fd-domain-steps`. Plan v2.10 §6 Batch 1.

### 10.1 What was built

- **`src/phasic/_fd_steps.py`**, a leaf module. It holds:
  - the per-slot rules (`slot_rule`: today, band, plain, unc_band, positive,
    unit, skip) and the build-time refusal (`build_info`, from the serialized
    graph's `weight_mode` / `is_discrete` / `was_dph`, or forced on);
  - `model_refuses(info, has_rewards)`, the ONE refusal query the refactor's
    phase 5 replaces;
  - `probes()`, which returns probe vectors selected with `jnp.where`;
  - `alt_ratio()` (the alternative quotient over the actual difference, with
    the zero guard);
  - `flags()` / `report()`: ties and zero differences, counted per model in
    `model._fd_stats` and logged once per model, at WARNING, through one
    `io_callback(ordered=False)` per backward (a single rolled callback under
    vmap; the first version used `jax.debug.callback`, which vmap unrolls
    into one callback per particle, review m3);
  - a test-only probe hook (`set_probe_hook`), and the numpy variant
    `host_probes_np` for the joint-index host fallback.
- **Every FD loop** keeps today's probe and quotient lines verbatim:
  - where another rule can apply, the probes are selected first, both
    quotients are computed, and the QUOTIENT is selected (§4.7);
  - on non-refusing models with undeclared slots, `probes()` returns the
    site's own arrays, so the traced program is today's.
- **`param_domains`** is accepted, and defaulted nowhere, on all eight
  builders of §5 and on the multivariate wrapper (passthrough plus a copied
  `_param_domains`). On the rvp backward it is a second nondiff argument
  (domains, BuildInfo), and zero inflation's FFI branch passes the model's
  domains. Builders attach `_param_domains`, `_fd_build_info` and
  `_fd_stats`. The joint-index builder merges `'fixed'` slots into its
  existing fixed set, so both its FD loop and its exact path zero them.
- **Refusal per site:**
  - from the graph: `pmf_from_graph`, `pmf_and_moments_from_graph` (plus
    rewards at trace time), the joint index, rvp;
  - always: `pmf_from_cpp`, `moments_from_graph` (v2.10), the two
    cdf-at-zero closures, the epoch model, `daisy_chain_joint_probs`.
- **Family per site:** capped for `pmf_from_graph`, `pmf_from_cpp`,
  `pmf_and_moments_from_graph` and its cdf at zero, and the multi-epoch
  sites with automatic granularity. Full for `moments_from_graph`, the
  joint index (FD loop and host), rvp, and the multi-epoch sites with an
  explicit granularity.

### 10.2 Evidence

- **Bit identity (`tests/pytest/test_fd_domain_steps.py` §1):** 74 golden
  cells, 25 site/theta cells × eager / jit / vmap(grad), captured on master
  `ff049679` before any source change (`experiments/fd_domain_steps/batch1/
  capture_golden.sh`, one process per cell). They include refusing models
  just above the band (2e-7) and a non-refusing clamp cell at 5e-8. All are
  equal bit for bit on the changed code, except the four reward-transform
  cells:
  - master is not reproducible run to run on those (two fresh captures
    differed by up to 9.6e-8 relative in the FD gradient, OI-022: last-bit
    forward noise times 1/(2e-7));
  - they compare at rtol 1e-6. A wrong rule selection would move them by
    orders of magnitude more.
  - The goldens are platform-tagged (`darwin-arm64`); other platforms skip
    the cells.
- **Crash fallback:**
  - **every refusing site:** {`pmf_from_graph`, `pmf_and_moments_from_graph`
    ± rewards, rvp} × weight mode × {continuous, native DPH}, plus
    `moments_from_graph`, `pmf_from_cpp` and the discretize slot, at slot
    values 5e-8 and 0. No crash; values match a reference whose probes stay
    >= 0 (rel 1e-3). Log mode at exactly 0 keeps today's primal error.
  - **the epoch model, on every route** (`Graph.svgd` with particles around
    1e-8, `epoch_model(...).fit`, direct `SVGD(model=...)`, direct
    `jax.grad`), at the §1 thetas, with and without exposure, with
    `exact_final_grad`, and through `daisy_chain_joint_probs`.
- **Measured accuracy in the band (`batch1/m1_epoch_band_errors.log`):**
  - epoch-model slots at 1e-8 and 1e-9 agree with a Richardson reference to
    ~1e-10 relative;
  - a coalescence rate at 1e-9 (an effect linear near 0) to 2-6e-3, the §4.3
    limit;
  - on `daisy_chain_joint_probs`, whose loss barely depends on the slot, the
    relative error is noise-dominated (up to 4x at 1e-9) but below 1e-9 in
    loss units in the domain coordinate (theta·d/dtheta). The test asserts
    that coordinate, as §6 prescribes.
- **Declared-domain accuracy:**
  - `moments_from_graph` (full rule) against closed-form moments at theta
    scale {1e-6, 1, 1e3}: rel 1e-6 in theta·d/dtheta. Today's rule at 1e-6 is
    more than 100x worse.
  - `pmf_and_moments_from_graph` moments (capped) at {1e-6, 1}.
  - `pmf_from_graph` (capped): against Richardson at scale 1; at scale 1e3
    against the same forward at fixed granularity (see 10.3).
- **Domains and slots:**
  - fixed / tied: the probe spy (exposure model, 2 epochs × [unit, positive
    exposure slot, unit tied slave, fixed]) shows only slot i changes, every
    probe is inside its domain, the fixed slot is never probed, and the tied
    slave is;
  - the tied-master gradient is the sum of the free model's partials
    (rel 1e-12);
  - `'fixed'` skips probing on `pmf_from_graph`, `moments_from_graph` and
    rvp.
- **Degenerate values (§4.5):**
  - a zero difference at theta = 3e9 is counted, gives a zero gradient and is
    logged once;
  - a forward tie (an offset slot at 1e-13 on a formula model) is counted
    and logged once;
  - 'unit' outside (0, 1) and 'positive' below 0 are bit-identical to
    undeclared;
  - 'positive' at 0 takes the one-sided step.
- **Other:**
  - signed parameter: 'unconstrained' gives d/dt1 == d/dt0 = -0.50049 at
    [6, -5], where today's clamp gives -0.49382;
  - row-sum wall: undeclared at 1 - 5e-8 keeps today's error, while a 'unit'
    slot at 1 - 1e-9 stays inside.
- **Existing tests:**
  - the strict xfail `test_daisy_fd_correct_at_mixed_scale` is removed and
    the test passes;
  - `test_discrete_row_sum.py::test_gradient_at_the_lower_bound` now asserts
    the geometric closed form at 5e-8 and 2e-7 (rel 1e-6).

### 10.3 Observations made while testing (not changes)

- **Euler forward at large rates:** above a max exit rate of 512 the
  automatic granularity is int(2 × max rate). That has two consequences:
  - (a) the forward jumps every ~2.5e-4 relative change of the max rate.
    Any 1e-7 probe, today's or capped, straddles a jump when 2 × max rate
    sits on an integer (theta = [1000, 500, 2000] gave -3.9e9 for -35.7);
  - (b) at ~2 Euler steps per mean holding time the forward is far from the
    closed form (about 2x in the log-density slope at scale 1e3).

  Both are Plan B's territory (true uniformization, §9 "the cap keeps
  today's jump behaviour"). The capped rule's job is to stay on one
  granularity plateau, which it does.
- **Capped moments at large theta:** with `exact_moment_grad=False` the
  `pmf_and_moments_from_graph` moments inherit the pdf's capped step (§4.1).
  At theta ~ 1e3 the capped step is today's 1e-7 absolute (relative 1e-10),
  with today's noise (5.6e-4 measured). That is not a regression, and the
  exact moments gradient (the default) is unaffected.
- **New OI-123:** `pmf_from_cpp` models cannot be vmapped. Their
  `pure_callback` returns one row for a batch of particles ("Expected (2, 3),
  Actual (1, 3)"), so `vmap(grad)` over SVGD particles fails on master.

### 10.4 Implementation review (Opus, one reviewer) and fixes

Report: `experiments/fd_domain_steps/batch1/review/REPORT.md` (probes r1-r9 in
the same directory). **Verdict: SIGN-OFF**, no blocker or major, six minors.

- The reviewer verified bit identity independently against an emulated
  master, including SVGD's jit(vmap(grad)) mode and mixed in-band /
  out-of-band particle batches: 0 mismatches. It found no missed FD site.
- **m2 fixed:** a `'fixed'`-domain slot on the exact-final-epoch path got a
  nonzero exact gradient. `'fixed'` slots now join the epoch model's skip set,
  so they are zero there too. Test:
  `test_fixed_domain_is_zero_on_the_exact_final_epoch_path`.
- **m3 fixed:** `jax.debug.callback` was unrolled to one callback per
  particle under vmap; `io_callback(ordered=False)` makes it one rolled
  callback. Measured with the reviewer's probe (`r2_cost_after_fix.log`):
  1 callback in the compiled program at 8, 64 and 256 particles, against
  8 / 64 / 256 before; steady-state time within noise of master; gradients
  bit-identical.
- **m4 fixed:** the rvp counter is now per caller. Zero inflation passes the
  model's counter; a direct call gets its own. Test:
  `test_rvp_counts_ties_per_model`.
- **m5 fixed** (tests that could not fail):
  - the rvp cells reward a vertex whose visit probability depends on theta,
    and assert a nonzero reference;
  - the pmf_and_moments crash cells use `exact_moment_grad=False`;
  - the epoch band test asserts rel 1e-6 on the mutation slots (2e-2 on a
    coalescence rate at 1e-9, the §4.3 limit) and marks
    daisy_chain_joint_probs finite-only with the reason;
  - the reason for the golden rtol is corrected;
  - NEW: a platform-independent emulated-master bit-identity test (all 25
    cells, eager and jit(vmap(grad))), and a test of the joint-index host
    fallback at the site itself (a forced decline).
- **m6 fixed:** the zero-difference message is domain-neutral.
- **m1 DEFERRED to Batch 2 (user decision 2026-11-27), OI-124:** a slot
  declared 'unit' at exactly 0 falls back to today's probe and so crashes on a
  refusing model, where an undeclared slot does not. This follows §4.5's
  text ("unit outside (0, 1): today's probe"), so changing it is a plan
  amendment.

### 10.5 Gate (plan §8)

Compared with `gate_compare.py` against Batch 0's master baseline
(`.gate_master_fdsteps`, with `--new-file tests/pytest/test_fd_domain_steps.py`):

- **First run (`.gate_batch1`, commit a7fa7839):** one real regression,
  `test_graph_cache_key::test_14_import_order` (3/3 in isolation).
  `_fd_steps` created its logger at import time, and `get_logger()` imports
  `phasic_pybind`, so the native module loaded before `OMP_NUM_THREADS` was
  set. Fixed in dd972b31 (lazy logger); reproducer
  `batch1/p_import_order.py`.
- **Final run (`.gate_batch1_final`):** the full run `.gate_batch1b` on commit
  59b9aac7, with `test_fd_domain_steps.py` replaced by a rerun after a
  test-only change (the cpp × jit(vmap) cell is no longer generated instead
  of skipped); `COMPOSED.txt` records this.
  - 3268 passed. The same 4 plot tests fail on both sides (OI-060).
  - 235 new tests, all passing.
  - `test_daisy_fd_correct_at_mixed_scale` went from skipped (strict xfail)
    to passed, as planned.
  - **One flagged regression, accepted under the gate rule:**
    `inference/test_svgd_correctness::test_positive_constraint` failed once
    and passed 3/3 in isolation (min phi 3.18, 4.61, 3.21). Its data come
    from `Graph.sample` under `np.random.seed` only, so they differ every
    run, and its model (linear, continuous, no rewards, undeclared) runs
    today's FD code bit for bit. Added to OI-061.
- Suite time: 70.9 min against the baseline's 60.3; the new file accounts
  for ~2 min.

## 9. Reproduce

From the worktree:

```
cd experiments/fd_domain_steps/batch0
./run_g1.sh && ./run_g2.sh   # refusal grid; pmf_from_cpp / moments_from_graph cells need PHASIC_SOURCE_DIR
pixi run python g3_summarise.py
./run_s1.sh                  # step sizes (~3 h, sequential)
pixi run python s2_summarise.py
```

The other scripts (`a1`-`a5`, `d1`) each run standalone under runcap.

## 11. Batch 2 (fitters declare domains, 2026-11-27)

Plan v2.10 §6 Batch 2. Tests: `tests/pytest/test_fd_domain_steps_batch2.py`
(54 tests). Measurements: `experiments/fd_domain_steps/batch2/`.

### 11.1 What was built

- **The resolver** (`svgd._resolve_param_domains`). It is side-effect free.
  `_resolve_param_transform` was split into a pure decision,
  `_param_transform_plan`, plus the wiring of `prior._transform`. SVGD and the
  resolver both call the pure part, so the declared domains and the fitted
  transforms cannot drift. The split is behaviour-identical: same checks, same
  errors, same wiring order.
  - Mapping: softplus → 'positive'; sigmoid → 'unit'; no transform
    (`positive_params=False`) → 'unconstrained'; custom transform → None;
    fixed → 'fixed'.
  - The exposure slot is widened from 'unit' to 'positive'.
  - A prior list of the wrong length gives None, so the caller raises its
    own length error.
- **`Graph.svgd` declares domains before the model is built.**
  - Single-epoch (`_svgd_fd_domains`): after the default DataPrior is built.
    The domains are passed to all five builder calls: joint index,
    multivariate, 1-D rewards, no rewards, and the epoch route.
  - Epoch route (`_epoch_fd_domains`): the prior is broadcast exactly as the
    daisy builder broadcasts it. Local fixings are 'fixed' in every epoch, a
    tied slave takes its master's domain, and the exposure slot is widened in
    every epoch.
  - The truly fixed set comes from the user's `fixed`, not from the
    builder's returned list, which also contains the tied slaves.
- **`Graph.epoch_model(param_domains='positive')`**:
  - accepts one string for every slot, a `param_length` list repeated per
    epoch, a flat list, or None (undeclared);
  - is passed to the builder and stored on `FreeEpochModel._param_domains`.
- **Containment (`svgd._check_param_domains`):**
  - In `SVGD.__init__`: read from the model argument before the exposure
    wrapper. Fit-side fixed slots, which include tied slaves, are 'fixed' and
    fit any model domain.
  - In `FreeEpochModel.fit`: read from the object, not from `.model`, and
    checked before the default prior's reference fit. The error message asks
    to rebuild with `param_domains`.
  - The fit side is NOT widened at the exposure slot. A sigmoid fit there is
    'unit', which fits a 'positive' model; a direct caller with a 'unit'
    model and a sigmoid fit also passes.
- **D-NZ documentation** in four places: the `positive_params` docstrings of
  `Graph.svgd` and `SVGD`, `Graph.epoch_model`'s `param_domains`, and
  `FreeEpochModel.fit`. A test asserts it is present.

### 11.2 Evidence

- **Declared vs today at fixed particles**
  (`batch2/m1_measure_part1_loglik.log`). The loss is SVGD's likelihood term
  (sum of log pdf); five Graph.svgd leaves; theta scale 1, 1e-3, 1e-4, 1e-5.
  - theta >= 1e-3: at most 1.7e-7 relative (plan: within 1e-6). ✓
  - 1e-4 and 1e-5: up to 1.65e-4 (joint index) and 9.8e-5 (epoch). That is
    above the plan's expected "1e-7 to 1e-5", but all of it is TODAY's error
    there:
    - today is off the Richardson reference by the same amount;
    - the declared rule is within 7e-7 of it everywhere (joint index 3e-10).
    - On the Euler leaves today's large relative step is slightly better
      (2e-9 against 3e-7), because that pdf is nearly linear in theta at
      these values. Both are far inside 1e-6.
  - The test asserts: declared within 1e-6 of the reference, and the move
    from today within today's own error plus 1e-6.
  - The first measurement (`m1_measure.log`, part 1) used a loss dominated
    by the exact moments term and showed no difference on the
    `pmf_and_moments` leaves; it was replaced by the log-likelihood run.
- **Fit level** (same seed, declared vs undeclared, 8 particles × 40
  iterations):
  - drift 5.9e-5 single-epoch, 1.5e-4 epoch (bound 1e-3);
  - Graph.svgd's epoch route and epoch_model drift identically, so the two
    routes are still in parity (`test_epoch_model.py:83`, `:161` in the gate).
- **Band trade-off and D-NZ pins** (`batch2/m1_measure.log`):
  - log-mode pole under the band rule: exact to 2e-11 at 5e-8, 1e-9 and
    1e-13;
  - formula-mode offset: 1.5e-5 relative at 5e-8 and 8.0e-4 at 1e-9; at
    1e-13 a counted, logged tie;
  - linear (non-refusing) offset under the plain step: 1e-10 (today 1e-7);
  - D-NZ pin: the plain step gives +1.333e14 against the closed form
    -4.0e14 (wrong sign, as documented); today's clamp gives -6.67e15;
  - the joint-index host site with 'unconstrained' at 5e-8 and 1e-9 is
    bit-identical to undeclared, with the correct sign.
  - Deviation from the plan's wording: log mode cannot express an additive
    offset (a weight is a product of every slot), so the offset pin uses
    formula mode, also a refusing model.
- **positive_params=False × {log, formula, callback, discretized, linear
  with rewards} at 5e-8:** no crash, and the value matches a positive-only
  central difference (rel 1e-3); the discretized case also checks its
  discretize slot.
- **Spy through Graph.svgd** (particles near 1e-4, where the declared probe
  ratio of 6e-5 is distinguishable from today's ~2e-3): the multivariate
  leaf, a single-epoch exposure fit and an epoch exposure fit all probe with
  the declared rule at every free slot.
- **Mutation check:**
  - Graph.svgd passing no domains, together with a disabled containment
    check, fails 11 of the 54 tests: every resolver, declaration,
    containment, spy, near-floor and fit-level test that depends on them.
  - The epoch fit-level test did not fail at first (it checked only the
    undeclared side); it now asserts the declared domains too.

### 11.3 Behaviour changes (plan §7), as listed

- epoch_model gradients change for every user (declared 'positive'):
  ~1e-8 relative at theta >= 1e-3, up to ~1e-4 at 1e-5, where today's step
  is the less accurate one.
- `epoch_model(...).fit(positive_params=False)` on a default model raises
  with the rebuild message.
- With `positive_params=False` the `pmf_and_moments_from_graph` clamp sites
  stop clamping (D-NZ).
- Graph.svgd single-epoch fits declare 'positive' and shift at rounding
  level (fit drift above).
- New, small: a BetaPrior with `positive_params=False` in Graph.svgd now
  raises before the model is built rather than inside SVGD (same message).

- Added after the review (m1):
  - Reusing a Graph.svgd model (`svgd.model`) in a direct
    `SVGD(model=...)` call with a WIDER fit now raises:
    - `positive_params=False`, or a softplus prior in place of a BetaPrior;
    - a nested model reused with its fixed slot freed. Before, that slot's
      gradient was silently zero.
  - The error message now explains that such a model carries the domains of
    the fit it was built for.
- Added after the review (m2): with a BetaPrior slot,
  `Graph.svgd(epoch_starts=...)` (declares 'unit') and the default
  `epoch_model` ('positive') are no longer bit-identical. Learnable-slot
  gradients differ by ~1e-8 relative; this is documented.

### 11.4 OI-124

Drafted as plan §12, amendment A-124, awaiting approval; not implemented.
It also corrects the Batch 1 review's reachability claim: SVGD's sigmoid is
clipped to [1e-9, 1 - 1e-9], so only an explicit 'unit' declaration at
exactly 0 reaches it.

### 11.5 Tooling note

The worktree-isolated session may not run commands that name `.git/`
paths, so `gate2.sh`, `gate_compare.py` and `runcap.py` were run from
unchanged copies in the job's tmp directory. `gate2.sh` itself still calls
the original `runcap.py` (OI-067 would remove this indirection).

### 11.6 Gate (plan §8)

`gate2.sh` on the Batch 2 working tree (`.gate_batch2`), compared with
`gate_compare.py` against `.gate_master_fdsteps`, with
`--new-file test_fd_domain_steps.py` and `--new-file test_fd_domain_steps_batch2.py`:

- 3321 passed, 6 failed, 112 skipped or xfailed; 289 new tests, all passing.
  The same 4 plot tests (OI-060) fail on both sides. Suite time 68.2 min
  (baseline 60.3).
- `test_log_prob_unified_bit_identity_zi` passed (OI-121, flaky).
  `test_daisy_fd_correct_at_mixed_scale` passed (strict xfail removed in
  Batch 1).
- **Two flagged regressions**, each rerun three times in isolation:
  - `inference/test_mcmc_accuracy::TestErlang2Posterior::test_two_param_recovery`:
    passed 3 of 3. This is the OI-061 flaky test (an R-hat threshold), and
    MCMC is gradient-free, so Batch 2 cannot affect it.
  - `inference/test_svgd_api_parity::test_graph_svgd_and_direct_svgd_agree`:
    failed 3 of 3 (std 0.08 vs 0.18, 0.28 vs 0.84, 0.17 vs 0.48). **Cause:**
    Graph.svgd now declares 'positive', while the test's direct SVGD built
    an undeclared model. The per-particle gradients differ by at most
    4e-7 (`batch2/p2_parity_grads.log`) and the particles by 1e-9 after 5
    iterations (`p3_parity_steps.log`). This fit, however, is chaotic: on
    the UNDECLARED (master) model, 1e-9 noise added to the initial particles
    alone moves the posterior std between 0.11 and 0.60
    (`p4_parity_chaos.log`). The test passed only because both surfaces
    ran bit-identical numerics. (Its data come from `Graph.sample` under
    `np.random.seed` only, so they differ every run; that is the OI-061
    class.)
    - **Test edit (a §7 addition, for your review):** the direct route now
      builds its model with `param_domains=['positive']`, as Graph.svgd
      declares for that fit, which restores identical numerics. It passed
      3 of 3 runs of the whole file after the edit. No other test file
      changed, so the full gate was not rerun; the rerun is the composed
      result.
- **Fit-level strict xfails (plan §7):** `test_notebook_model_recovery.py`
  `:484` and `:699` (not slow) were unchanged in the gate. `:553` and
  `:659` (slow) were run explicitly and remain XFAIL. No flip to report.

### 11.7 Implementation review (Opus, one reviewer) and fixes

Report: `experiments/fd_domain_steps/batch2/review/REPORT.md` (probes r1-r4
in the same directory). **Verdict: SIGN-OFF**, no blocker or major, six
minors, all folded in:

- **m1:** reusing a Graph.svgd model with a wider fit now raises. Listed in
  §11.3 and plan §7; the SVGD error message now explains it.
- **m2:** epoch_model and Graph.svgd parity with a BetaPrior needs a
  matching `param_domains`. Documented in the epoch_model and fit
  docstrings and in plan §3.
- **m3:** the redundant assertion is replaced by "declared no worse than
  today + 1e-6"; the epoch margin (7.1e-7 against 1e-6) is stated.
- **m4:** three new real-path tests:
  - a BetaPrior exposure slot reaches the builder as 'positive' through
    Graph.svgd (spied builder);
  - a widening with exposure raises, so the check runs before the
    wrapper;
  - fit() raises before the default prior factory runs.
- **m5:** the parity test also calls `random.seed(0)`.
- **m6:** the domain and D-NZ text in FreeEpochModel.fit moved to a Notes
  section; the doc test asserts "forward tie".

Rerun after the fixes:

| File | Result |
|---|---|
| `test_fd_domain_steps_batch2.py` | 57 passed |
| `test_fd_domain_steps.py` | 235 passed |
| `inference/test_svgd_api_parity.py` | 3 passed |
| `inference/test_epoch_model.py` | 7 passed (route parity) |

The fixes are docstrings, one error message and tests, so the full gate was
not repeated.
