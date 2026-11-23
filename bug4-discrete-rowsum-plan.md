# Bug 4: discrete models accept row sums above 1 and return negative probabilities — fix plan

**Status: v1, 2026-11-23. Branch `fix/discrete-negative-pmf`, worktree
`/Users/kmt/phasic-bug4` (own pixi env, built from master `ca09e2e1`). No
source changed yet. Backward compatibility is not required (user
decision). Native (C) change; independent of bugs 1-3 in code, but gated
after them (rebased onto master first).**

Evidence: `experiments/bug4_discrete_rowsum/` (`b1_scope.py`, `b1.log`,
`b2_legit_excess.py`, `b2.log`), unmodified code.

## 1. The defect (read from source + measured)

In a discrete phase-type model each vertex's outgoing edge weights are
per-step transition probabilities; the remainder 1 - sum stays put. The C
code checks the sum at six places with the tolerance `rate > 1.0001`
(`src/c/phasic.c:6715, 6824` in `ptd_graph_dph_reward_transform`; `12543`
`ptd_dph_random_sample`; `12615` `ptd_mdph_random_sample`; `12733`
`ptd_dph_random_sample_stop_vertex`; `12819`
`_ptd_dph_probability_distribution_context_create`), and the step function
(`phasic.c:12943-12990`) moves `mass x weight` along every edge and
subtracts it from the source, so a sum above 1 leaves NEGATIVE mass behind.

Measured on a geometric chain with success probability p (b1.log):

| p - 1 | pmf(1..4) | cdf(1) | expectation / variance / moments | sample |
|---|---|---|---|---|
| 0 | 1, 0, 0, 0 | 1 | 1 / 0 | ok |
| 1e-12 | 1, **-1.0e-12**, 1.0e-24, -1.0e-36 | 1 | 0.999999999999 / **-1.0e-12** | ok |
| 1e-9 | 1, **-1.0e-9**, ... | 1 | / **-1.0e-9** | ok |
| 1e-5 | **1.00001, -1.00001e-5**, ... | **1.00001** | / **-1.0e-5** | ok |
| 1e-4 | **1.0001, -1.0001e-4**, ... | **1.0001** | / **-1.0e-4** | ok |
| 2e-4 | raises ("outgoing rate <= 1") | raises | 0.9998 / **-2.0e-4**, no error | raises |
| 0.5 | raises | raises | **0.667 / -0.222, no error** | raises |

So: (a) between 1 and 1.0001 the pmf is negative and exceeds 1, the cdf
exceeds 1, and the reward-transformed chain inherits it; (b) the
elimination-based read-outs (`expectation`, `variance`, `moments`) have NO
check at any excess and return a negative variance silently.

Legitimate graphs do exceed 1 — by rounding only (b2.log): `discretize()`
graphs and `joint_prob_graph(discrete=True)` at ordinary parameters have
`max(row sum) - 1 = 2.2e-16` (one unit in the last place), in up to half of
their rows (375 of 678; 1,344 of 2,806). Nothing legitimate is anywhere
near 1e-4.

The existing behaviour beyond the tolerance is a deliberate hard error — a
row sum above 1 is a modelling error, not a diverging-rate case
(`tests/pytest/test_svgd_divergence_robustness.py:52-53` asserts it is NOT
treated as fail-soft).

## 2. Fix

**Principle: a discrete model is valid only if every row sums to at most 1
up to rounding; a valid model never yields negative mass; an invalid one
raises on every read-out.**

1. One constant, `PTD_DPH_ROW_SUM_TOL = 1e-9` (rounding in the measured
   graphs is 2.2e-16; 1e-9 leaves six orders of margin for long rows and
   is five orders tighter than today), and one C helper
   `ptd_dph_validate_row_sums(graph, &scale)` that, for every non-starting
   vertex with edges:
   - raises the existing error message (same wording, so callers and tests
     that match it keep working) if `sum > 1 + tol`;
   - otherwise records a per-vertex scale `1/sum` when `1 < sum <= 1 + tol`
     and `1` when `sum <= 1`.
2. The six existing checks call the helper. The discrete step applies the
   per-vertex scale to the outgoing mass (a rounding-level correction that
   makes the leaving mass at most the mass present, so no negative mass can
   arise); the samplers and the reward transform normalise their
   probabilities the same way.
3. The elimination-based discrete read-outs (expectation, variance,
   moments, expected visits; and the parameterized/FFI moment paths for
   discrete graphs) call the same validation before computing, so an
   invalid model raises there too instead of returning a negative
   variance.
4. Graphs converted from continuous ones for Euler stepping (`dont_worry`
   contexts, `rate / granularity` check at `phasic.c:13055`) are a
   different code path and are not changed.

Consequence stated up front: a discrete fit whose particle — or whose
finite-difference probe (theta + 1e-7) — crosses a row-sum bound by more
than 1e-9 now raises, where it used to return slightly wrong numbers
(a reviewer of another plan measured 6.7e-2 relative gradient error at a
gap of 5e-8). That is the intended "invalid model raises" rule; keeping
particles inside a model's own bounds belongs to the gradient-step plan.

## 3. Tests (new file `tests/pytest/test_discrete_row_sum.py`)

Fail on master:
1. Geometric chain, p - 1 in {1e-6, 1e-5, 1e-4}: `pdf`, `cdf`, `sample`,
   `reward_transform_discrete(...).pdf`, `expectation`, `variance`,
   `moments` all raise with the row-sum message (today only none of them
   do, below 1.0001).
2. p - 1 in {2e-4, 0.5}: `expectation`, `variance`, `moments` raise
   (today they return a negative variance).
3. p - 1 in {1e-12, 1e-15, 2.2e-16}: accepted; pmf(k) >= 0 for all k, pmf
   sums to 1 within 1e-12, cdf <= 1, variance >= 0 (today pmf(2) and the
   variance are negative).
4. A legitimate graph with rounding-level excess (`joint_prob_graph`
   discrete, n=5, theta = [2.0, 0.05]: 375 rows one ulp above 1): every
   pmf value >= 0 and the table sums to at most 1.
5. The parameterized paths (`Graph.pmf_from_graph(..., discrete=True)`,
   `pmf_and_moments_from_graph(..., discrete=True)`, pybind and FFI): at
   c*theta = 1 + 1e-5 they raise; at 1 + 1e-12 they return non-negative
   values.

Guards (pass on master and after):
6. p in {0.1, 0.5, 1.0}: values unchanged (closed-form geometric pmf, cdf,
   mean, variance) to 1e-14.
7. `discretize()`-built graphs keep their existing behaviour
   (`test_parameterized_discretized_no_rate_overflow` and the rest of
   `test_discrete_moments_and_reward.py` unchanged).
8. Continuous graphs: `pdf`/`cdf`/`stop_probability` bit-identical to
   master for a small set of models (the shared step function is edited).
9. The error is not fail-soft:
   `phasic._is_rate_blowup` still returns False for it.

## 4. Gate

Same tools and rules as the other fixes (`gate2.sh`, `gate_compare.py
--new-file tests/pytest/test_discrete_row_sum.py`, `gate_rerun.sh`), after
rebase and `pixi run install-dev` (native rebuild); baseline = the
previous fix's gate run. `tests/pytest/test_omp_exception_safety.py` and
`test_discrete_moments_and_reward.py` mention the 1.0001 tolerance in
comments; they are checked and their comments updated if the tests still
hold.

## 5. Risks

- Discrete fits that silently ran with particles or probes slightly over a
  row-sum bound now raise.
- The step function is shared with continuous Euler stepping; guard 8
  pins bit-identity there.
- A user graph with long rows whose rounding excess exceeds 1e-9 would be
  refused (not observed: measured excess is 2.2e-16).
