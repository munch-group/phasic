# Bug 4: discrete models accept row sums above 1 and return negative probabilities — fix plan

**Status: v2, 2026-11-23 (v1 rejected by all three reviewers; this version
replaces its mechanism and scope). Branch `fix/discrete-negative-pmf`,
worktree `/Users/kmt/phasic-bug4` (own pixi env). No source changed yet.
Backward compatibility is not required (user decision). Native (C/C++)
change; gated after bugs 1-3 (rebased onto master first).**

Evidence, all on unmodified code: `experiments/bug4_discrete_rowsum/`
(`b1`, `b2`: mine; `review1/`, `review2/`, `review3/`: the reviewers').

## 0. What changed from v1, and why

| v1 | finding | v2 |
|---|---|---|
| scale each row by 1/sum to stop negative mass | wrong: negative mass also arises on rows summing to exactly 1 or below (sequential subtraction rounds; 8,013 of 20,000 random rows), and scaling leaves it (review1/r4.log, review2/r1.log, review3/t3.log) | the discrete step caps each outflow at the mass still present at the source (reviewer 2's variant C: zero negative mass, zero negative pmf on every graph tried) |
| "every read-out raises" incl. elimination read-outs | would break `joint_prob_table()` on a freshly built discrete joint graph, which holds raw rates (rows 3-15) and is correct today to 3e-16 because elimination is invariant to row scale on such graphs (review3 B1); C has no discreteness flag | validation in elimination read-outs only for NATIVE discrete graphs (`was_dph == false`), at named sites (§2.3); `expected_sojourn_time`, `expected_waiting_time`, `joint_prob_table`, `probability_matching`, `sample_path` are not validated |
| helper skips the starting vertex | today's context check includes it; skipping loses an error (review3 M3) | all vertices validated, starting vertex included |
| "same wording" error | sum printed with `%f`: 1+1e-7 prints as 1.000000 | `%.17g`, existing matched substrings kept |
| crashes not mentioned | two NULL dereferences when the check fires | fixed (§2.5) |
| bit-identity as a suite test | suite cannot see master; pins not portable | one-off gate diff + structural guarantee (§4) |
| pinned discretize values "unchanged" | two bit-exact pin records change | regenerated, listed, bounded (§2.6) |

## 1. The defect (read from source + measured)

In a discrete phase-type model each vertex's outgoing edge weights are
per-step transition probabilities; the remainder stays put. Two separate
faults:

**(a) The tolerance.** Six C sites accept a row sum up to 1.0001
(`src/c/phasic.c:6715, 6824` reward transform; `12543`, `12615`, `12733`
samplers (12733 is dead code); `12819` step-context creation). Between 1
and 1.0001 the geometric chain with success probability p returns
pmf(1) = p > 1, pmf(2) = -(p-1)p < 0, cdf > 1 (b1.log). The
elimination read-outs have no check at any excess (p = 1.5: expectation
0.667, variance -0.222, no error).

**(b) The step function** (`phasic.c:12943-12990`) computes every edge's
move as `old_mass * weight` and subtracts each from the source. On rows
that sum to exactly 1, or to 1 plus one unit in the last place — which is
what legitimate `discretize()` and joint-probability graphs have, in up to
half their rows (b2.log) — the subtraction leaves a residual of about
-1e-16 x mass. On the n=5 discrete joint graph: 7,253 negative vertex
masses and 9 negative pmf values over 400 steps (review2/r1.log);
`stop_probability` returns 334 negative entries.

A row sum above the tolerance is a deliberate hard error, not a
diverging-rate case (`tests/pytest/test_svgd_divergence_robustness.py:52-53`).

## 2. Fix

**Principle. (i) A discrete model is valid only if every row, the starting
vertex's included, sums to at most 1 + 1e-9. (ii) Stepping a valid model
never produces negative mass or a negative pmf, exactly, not within a
tolerance. (iii) Step-based read-outs of an invalid model raise; so do the
elimination read-outs listed in §2.3. The others are stated as out of
scope, not silently skipped.**

### 2.1 One tolerance, one helper

`#define PTD_DPH_ROW_SUM_TOL 1e-9` and
`int ptd_dph_validate_row_sums(struct ptd_graph *graph)` in
`src/c/phasic.c` (declared in `api/c/phasic.h`): loops over ALL vertices,
sums edge weights in `double`, and on the first vertex with
`sum > 1 + PTD_DPH_ROW_SUM_TOL` sets `ptd_err` and returns nonzero. The
message keeps the substrings existing code and tests match ("to have
outgoing rate <= 1", "Are you sure this is a discrete phase-type
distribution?", the vertex index, "(starting vertex)") and prints the sum
as `%.17g`.

Why 1e-9: measured rounding excess on legitimate graphs is 2.2e-16 on
ordinary rows, 1.3e-15 after reward transforms at 2,004 vertices, and
1.9e-12 (below 1, same magnitude expected above) on a 100,000-edge row
(review3/t4.log). 1e-9 leaves three orders of margin over the worst of
these and is five orders tighter than today.

The six existing checks are replaced by calls to the helper (the reward
transform and samplers validate only; they do not rescale — a sampler
walking cumulative weights is unaffected by an excess of 1e-9).

### 2.2 The step: cap outflow at the mass present (discrete contexts only)

The context gets an explicit `bool cap_outflow`, set true by
`_ptd_dph_probability_distribution_context_create(graph, dont_worry=false)`
(real discrete graphs) and false for the `dont_worry=true` contexts used
for continuous Euler stepping (`phasic.c:13087, 13098`).

When `cap_outflow` is false the step executes exactly the statements it
executes today (the existing loop is kept verbatim in the else-branch), so
continuous results cannot change.

When true, the increment loop becomes (increments are already grouped by
source vertex, in edge order):

    remaining = old[from]            (reset when `from` changes)
    add = old[from] * weight
    if (add > remaining) add = remaining
    probability_at[to]   += add
    remaining            -= add
    ... after the vertex's last edge: probability_at[from] = (mass that
        arrived at `from` during this step) + remaining

Implementation detail that matters: today `probability_at[from]` is both
decremented by its own outflow and incremented by inflow from other
vertices in the same pass. The capped version accumulates inflow into
`probability_at` (zeroed for every vertex at the start of the step) and
adds each vertex's `remaining`; this is the same arithmetic when no cap
fires, up to the order of additions.

Invariant (provable, and asserted in tests): if all masses are >= 0 and
all weights are >= 0 before a step, then every `add` is in
[0, remaining], `remaining` stays >= 0, and every mass after the step is a
sum of non-negative terms. Hence pmf >= 0 exactly. Total mass is
conserved up to rounding of the additions, so cdf <= 1 holds to a few
units in the last place per step, not exactly; tests use 1 + 1e-12.

Negative edge weights are not a new case: they are rejected at
construction/update today and stay out of scope.

For a starting-vertex row in (1, 1 + 1e-9]: the first step is an ordinary
capped step from vertex 0 holding mass 1, so the initial distribution is
capped to total 1 as well.

### 2.3 Which read-outs validate

Step-based (all discrete graphs, native or converted; these go through
context creation or the samplers and so call the helper):
`pdf`/`pmf`, `cdf`, `stop_probability`, `accumulated_visits`,
`distribution_context().step()`, `sample`, `sample_multivariate`,
`reward_transform_discrete`, and every parameterized/JAX path that
evaluates a discrete pmf (`pmf_from_graph(discrete=True)`,
`pmf_and_moments_from_graph(discrete=True)`: the pmf half runs first in
the same native call, `graph_builder.cpp` discrete branches at 759, 1150,
1177, 1199, 1364).

Elimination-based, NATIVE discrete graphs only (`graph->was_dph == false`):
the three pybind entry points that are discrete by name —
`_expectation_discrete` (`src/cpp/phasic_pybind.cpp:336`),
`_variance_discrete` (`:342`), `_covariance_discrete` (`:385`) — call the
helper first and throw `std::runtime_error(ptd_err)`. `Graph.expectation`,
`Graph.variance`, `Graph.covariance` reach them through the existing
`is_discrete` dispatch (`src/phasic/__init__.py:2408-2411, 2468-2471`), and
`Graph.moments(power, discrete=True)` (`:2343-2350`) calls a new pybind
method `_validate_dph_row_sums()` that applies the same rule.

Why native only: a graph produced by `discretize()` or
`joint_prob_graph(discrete=True)` (`was_dph == true`) may legitimately hold
un-normalised rows until `update_weights` runs, and the elimination result
does not depend on row scale there (review3/t9.log, t10.log: tables agree
to 3.3e-16). On a native graph the row sum IS the model (the remainder is
the stay probability), so a sum above 1 is an error.

At a rounding-level excess (row sum in (1, 1 + 1e-9]) on a native graph
the elimination read-outs are accepted and return the formula's value,
which is within the same 1e-9 of the row-sum-1 value (variance of the
geometric chain: -(p-1)/p^2). They are not rescaled: rescaling inside
elimination would shift pinned values for no gain.

Out of scope, stated: `expected_waiting_time`, `expected_sojourn_time`,
`joint_prob_table`, `probability_matching`, `sample_path`,
`random_sample_discrete_stop_vertex`, `Graph.moments(power)` WITHOUT
`discrete=True` on a discrete graph (returns the continuous-type moment
E[N(N+1)], a separate defect to log), `moments_from_graph` on a discrete
graph. These are added to `found-defects-2026-11-23.md`, not fixed here.

### 2.4 Stale cached context

`dph_accumulated_visits` (`api/cpp/phasiccpp.h:1773`) reuses a cached
context without the weight-version check its siblings have; it gets the
same check, so validation re-runs after `update_weights`.

### 2.5 Crashes when the check fires

- `DPHProbabilityDistributionContext` (`api/cpp/phasiccpp.h:2346`) stores a
  NULL context when creation fails; `distribution_context()` on an invalid
  discrete graph then dereferences it. The constructor throws
  `std::runtime_error(ptd_err)` instead.
- `sample_multivariate_discrete` (`src/cpp/phasic_pybind.cpp:2511`) uses the
  NULL returned by `ptd_mdph_random_sample`. It checks and throws.

### 2.6 Pinned values that change

`tests/pytest/test_gate_discretize_pins.py` compares float hex. Cases
`param_callback` and `constant_callback` each have one row at 1 + 2.2e-16;
their pinned `pdf(1..5)` may change in the last bits. Procedure: after the
implementation, run the pin test; the only permitted failures are pdf/cdf
entries of those two cases, each within 1e-14 relative of the old value;
then regenerate exactly those records with the file's own generator and
record old/new hex in the plan. Any other pin change stops the work.
`expectation_discrete` pins must not change (elimination is not edited).

## 3. Tests (new file `tests/pytest/test_discrete_row_sum.py`)

Geometric chain = start -> v (weight 1), v -> absorbing (weight p).

Fail on master:
1. p - 1 in {1e-6, 1e-5}: each of `pdf`, `cdf`, `stop_probability`,
   `accumulated_visits`, `sample`, `reward_transform_discrete(...).pdf`,
   `expectation`, `variance`, `covariance`, `moments(2, discrete=True)`
   raises with "outgoing rate <= 1" (one assertion per read-out; today
   none raises). The message contains the vertex index and a sum that
   parses back to p within 1e-15.
2. p - 1 in {2e-4, 0.5}: `expectation`, `variance`, `covariance`,
   `moments(2, discrete=True)` raise (today: negative variance, no error).
3. p - 1 in {2.2e-16, 1e-15, 1e-12}: accepted. `pdf(k) >= 0` exactly for
   k = 1..20, `pdf(1) == 1`, `cdf <= 1`, every `stop_probability` entry
   >= 0 (today pdf(2) is negative). `abs(variance) <= 2e-12` and
   `abs(expectation - 1) <= 2e-12` (elimination not rescaled, §2.3).
4. Legitimate graphs, through the STEP function: discrete joint graph n=5
   after `update_weights([2.0, 0.05])` built without `mutation_rate`
   (precondition asserted: at least one row above 1), coalescent n=10
   `discretize`, two-island n=8 `discretize`. Over 400 jumps: every `pdf`
   value >= 0 exactly, every `stop_probability` entry >= 0 exactly, cdf
   non-decreasing exactly and <= 1 + 1e-12 (today: 9 negative pmf values
   and 334 negative stop-probability entries on the first graph).
5. Starting vertex: initial-probability sum 1.5 raises (as today); sum
   1 + 1e-5 raises (today cdf reaches 1.00001); sum 1 + 1e-12 accepted
   with cdf <= 1 + 1e-12.
6. Parameterized native discrete graph at c*theta = 1 + 1e-5:
   `Graph.pmf_from_graph(g, discrete=True)` and
   `pmf_and_moments_from_graph(g, discrete=True, use_ffi=...)` for both
   values raise; matched on the message text, not the exception type
   (types differ by path). At 1 + 1e-12 values are returned and >= 0.
7. Crashes, each in a subprocess (pattern of `test_omp_exception_safety.py`):
   `distribution_context()` and `sample_multivariate_discrete` on p = 1.5
   exit with a Python exception, not a signal.
8. Batched FFI: `vmap` of `pmf_and_moments_from_graph(discrete=True,
   use_ffi=True)` at 1 + 1e-5, in a subprocess: catchable error, no abort.
9. `accumulated_visits` after `update_weights` to p = 1.5 on a graph that
   was valid when first read: raises (stale context).

Guards (pass on master and after):
10. p in {0.1, 0.3, 0.5, 1.0}: geometric closed forms — pmf and cdf to
    absolute 1e-15, mean and variance to relative 1e-14.
11. Fresh discrete joint graph (no `update_weights`): `joint_prob_table()`
    does not raise and equals the table after `update_weights([1, 1])` to
    1e-14; `expectation()` on it does not raise.
12. Validation re-runs on a cached pdf context: valid, `pdf`, update to
    1.5, `pdf` raises.
13. Gradient near the bound, native discrete, pinned so the later
    gradient-step plan has a target: value calls work at a gap of 5e-8
    below 1; `jax.grad` raises there (the +1e-7 probe crosses) and works
    at a gap of 2e-7.
14. Not fail-soft: `phasic._is_rate_blowup` is False for the real
    exception raised by a 1 + 1e-5 graph.
15. Continuous closed form (exponential and Erlang pdf/cdf) within the
    documented 2.5e-3 floor — the suite-side check that Euler stepping
    still works; bit-identity is established in the gate (§4).

## 4. Gate

1. Rebase on master, `pixi run install-dev` (native rebuild).
2. Continuous bit-identity, one-off: a script in
   `experiments/bug4_discrete_rowsum/` dumps float hex of `pdf`, `cdf`,
   `stop_probability`, `accumulated_visits` for a model set (coalescent,
   two-island, two-locus, reward-transformed, parameterized through
   `pmf_from_graph`) from the master build (`/Users/kmt/phasic`) and the
   fix build on this machine; the dumps must be identical.
3. Pin procedure of §2.6.
4. `gate2.sh` + `gate_compare.py --new-file
   tests/pytest/test_discrete_row_sum.py`; baseline = the previous fix's
   gate run; regressions rerun three times in isolation.
5. New tests recorded on the master build (which fail, which pass).
6. Comments mentioning 1.0001 updated
   (`tests/pytest/test_omp_exception_safety.py:8,117`,
   `src/cpp/parameterized/graph_builder_ffi.cpp:51`).

Known from review 3 (tight raise rule simulated by binary patch): 36
discrete-related test files pass unchanged, including the notebook-model
recovery tests; no existing test relies on the window between 1 + 1e-9 and
1.0001; no notebook builds a native discrete graph.

## 5. Risks

- A discrete fit whose particle or finite-difference probe (theta + 1e-7)
  crosses a row-sum bound by more than 1e-9 now raises where it returned
  slightly wrong numbers (guard 13 pins the boundary). Joint-probability
  and `discretize()` fits cannot cross: `update_weights` renormalises
  their rows (review3/t2b.log).
- Discrete pdf/cdf values change at the 1e-15 level wherever a cap fires
  or the order of additions differs (§2.6 bounds this on the pins).
- The capped step does more work per edge (one compare, one subtract);
  measured before merge on the n=5 and a ~5,000-vertex joint graph and
  recorded; more than 10% slower stops the work for a rethink.
- A user graph whose rounding excess exceeds 1e-9 would be refused (not
  observed; worst measured 1.9e-12 at 100,000 edges in a row).
