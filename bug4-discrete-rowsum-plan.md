# Bug 4: discrete models accept row sums above 1 and return negative probabilities — fix plan

**Status: v3.1, 2026-11-23 (v1 and v2 each returned CHANGES REQUIRED by all
three reviewers; §0b lists the v2 -> v3 changes). Branch `fix/discrete-negative-pmf`,
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

## 0b. What changed from v2 (second review round)

- Step: the cap is inserted into today's statements in place (two
  statements) instead of restructuring the loop; bit-identical to today
  whenever no cap fires (review2/r4.log).
- Validation is per vertex (static function); the samplers keep checking
  only the vertex they are at, as today. A whole-graph pass per draw would
  have slowed sampling 16-300x (review2/r5.log, review3/t16.log).
- The per-vertex check also rejects negative and non-finite weights:
  linear `update_weights` accepts negative weights today
  (review1/r6.log), which would defeat the non-negativity invariant.
- A context held by the user is re-validated in the step when the graph's
  weights have changed since it was created.
- Pins: every `pdf` pin record changes (20 of 21 measured), not two; new
  acceptance rule (§2.6).
- `cap_outflow` initialisation, explicit validation in the
  pmf-and-moments branches, corrected test classification and names,
  completed out-of-scope list, concrete timing protocol.

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
masses over 300 steps (7,553 over 400) and 9 negative pmf values
(review2/r1.log, r3.log); the coalescent n=10 `discretize` graph returns
104 negative pmf values out of 400;
`stop_probability` returns 334 negative entries.

A row sum above the tolerance is a deliberate hard error, not a
diverging-rate case (`tests/pytest/test_svgd_divergence_robustness.py:52-53`).

## 2. Fix

**Principle. (i) A discrete model is valid only if every edge weight is
finite and non-negative and every row, the starting vertex's included,
sums to at most 1 + 1e-9. (ii) Stepping a valid model
never produces negative mass or a negative pmf, exactly, not within a
tolerance. (iii) Step-based read-outs of an invalid model raise; so do the
elimination read-outs listed in §2.3. The others are stated as out of
scope, not silently skipped.**

### 2.1 One tolerance, one per-vertex check

In `src/c/phasic.c` (whole-graph function declared in `api/c/phasic.h`):

- `#define PTD_DPH_ROW_SUM_TOL 1e-9`
- `static int ptd_dph_check_vertex(struct ptd_graph *, struct ptd_vertex *)`:
  sums the vertex's edge weights in `double`; fails if any weight is
  negative or non-finite, or if `!(sum <= 1 + PTD_DPH_ROW_SUM_TOL)` (so a
  NaN sum fails). On failure it sets `ptd_err` and returns nonzero. The
  row-sum message keeps the substrings existing code and tests match ("to
  have outgoing rate <= 1", "Are you sure this is a discrete phase-type
  distribution?", the vertex index, "(starting vertex)") and prints the
  sum as `%.17g`; the weight message names the vertex, the edge target
  and the value.
- `int ptd_dph_validate_row_sums(struct ptd_graph *)`: loops the
  per-vertex check over ALL vertices.

Use at the six existing sites:
- step-context creation (`12819`) and the two reward-transform sites
  (`6715`, `6824`, already per vertex inside their loops): whole-graph
  function, respectively the per-vertex check where the loop already
  visits each vertex once;
- the three samplers (`12543`, `12615`, `12733`): the per-vertex check
  for the vertex the walk is at — the same cost as today's inline sum.
  Never the whole-graph function per draw.

Nothing is rescaled anywhere. A sampler walking cumulative weights is
unaffected by an excess of 1e-9.

Why 1e-9: measured rounding excess on legitimate float64 graphs is
2.2e-16 on ordinary rows, 1.3e-15 after reward transforms at 2,004
vertices, 4.4e-13 at 600,000 edges in a row (review1), 1.9e-12 below 1 on
a 100,000-edge row (review3/t4.log). 1e-9 leaves three orders of margin
and is five orders tighter than today.

### 2.2 The step: cap each outflow at the mass still present (discrete contexts only)

The context struct gets `bool cap_outflow` and
`uint64_t validated_weight_version`.
`_ptd_dph_probability_distribution_context_create` assigns
`cap_outflow` in BOTH branches (`true` when `dont_worry` is false, `false`
otherwise), and `validated_weight_version = graph->weight_version` and
the existing `weight_version_at_creation` (today assigned only at
`12922`, after the internal step), all immediately after the `malloc` at
`phasic.c:12848` — before the initialising step the creation function
runs internally at `12917`. (The struct is
`malloc`'d; an unassigned flag would be the known
uninitialised-field-on-Linux fault.)

The increment loop (`phasic.c:12968-12973`) becomes:

    size_t current_from = (size_t) -1;     /* no vertex has this index */
    long double remaining = 0;             /* same type as the masses */
    for each increment inc (already grouped by source, in edge order):
        add = old_probability_at[inc.from] * (*inc.weight) * context->priv3;
        if (context->cap_outflow) {
            if (inc.from != current_from) {          /* new source vertex */
                current_from = inc.from;
                remaining = old_probability_at[inc.from];
            }
            if (add > remaining) add = remaining;
            remaining -= add;
        }
        context->probability_at[inc.to]   += add;
        context->probability_at[inc.from] -= add;

Everything else in the step is unchanged. With `cap_outflow` false the
executed arithmetic is exactly today's (same expressions, same order), so
continuous Euler stepping (`13087`, `13098`, which set `priv3` after
creation) cannot change. With it true and no cap firing, the arithmetic
is also exactly today's: native discrete models whose rows are safely
below 1 keep bit-identical results (0 of 200 values changed in the
replica, review2/r4.log).

Invariant (reviewer 2's proof, by monotonicity of rounding). Assume all
masses and weights are >= 0 before the step (weights by §2.1). For source
vertex v let r_k be `remaining` and x_k the in-place value
`probability_at[v]` after its k-th edge. x_0 = old + (inflow already
received this step) >= old = r_0, and each edge subtracts the same `add`
<= r_{k-1} from both, so x_k >= r_k >= 0 throughout; inflow arriving
later only adds non-negative terms. Every `add` is >= 0. Hence every
mass and every pmf value is >= 0 exactly. Total mass is conserved up to
rounding, so cdf <= 1 is asserted with a tolerance (1 + 1e-12; measured
worst 1 + 2.2e-16 over 400-2,000 steps).

Self-loop edges cannot exist (`add_edge` refuses them,
`phasic.c:4846-4865`).

Starting vertex: the first step is an ordinary capped step from vertex 0
holding mass 1, so an initial distribution summing to (1, 1 + 1e-9] is
capped to total 1.

**Contexts held by the user.** The step reads weights through pointers,
so a context created before `update_weights` steps with the new weights
unvalidated (measured: pmf [0.75, -0.375, 0.1875] after updating 0.5 ->
1.5, review1/r6.log). In a `cap_outflow` context the step, as its very
first statements — before `context->jumps++`, `context->pmf = 0` and the
allocation of the old-mass array (`12946-12951`), so a failure leaves the
context untouched and leaks nothing — compares `graph->weight_version`
with `validated_weight_version` (one integer compare per step). When they
differ it runs the whole-graph validation, returns nonzero with `ptd_err`
set if it fails, and otherwise stores the new version in
`validated_weight_version`. `weight_version_at_creation` keeps its
meaning (the C++ caches rebuild on it, `phasiccpp.h:1627, 1747`). The
four C++ call sites that ignore the step's return value today
(`phasiccpp.h:1644, 1759, 1787, 2356`) check it and throw via
`PTD_THROW_AND_CLEAR()`.

Not covered (existing behaviour, out of scope): changes made after a
context was created that do not bump the weight version — structural
edits (`add_edge`, auxiliary vertices; the context's arrays are sized at
creation) and the in-place division in `ptd_dph_normalize_graph`
(`phasic.c:2684`).

### 2.3 Which read-outs validate

Step-based (all discrete graphs, native or converted; these go through
context creation or the samplers and so call the helper):
`pdf`/`pmf`, `cdf`, `stop_probability`, `accumulated_visits`,
`distribution_context().step()`, `sample`, `sample_multivariate_discrete`,
`reward_transform_discrete`, and every parameterized/JAX path that
evaluates a discrete pmf (`pmf_from_graph(discrete=True)`,
`pmf_and_moments_from_graph(discrete=True)`); all of them reach
`Graph::dph_pmf` (`graph_builder.cpp` discrete branches at 759, 956, 994,
1020, 1150, 1177, 1199, 1364; FFI handlers `graph_builder_ffi.cpp:436,
465, 743, 755, 808, 820, 1020, 1071`). Because the pmf loop is skipped
for empty or all-NaN `times` (`graph_builder.cpp:990-1035`; measured:
moments [0.667, 0.222] returned at p = 1.5), the `is_disc` branches of
`compute_pmf_and_moments` additionally call the whole-graph validation
once, before the moments, for native graphs (`was_dph == false`).

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
graph, `normalize()` on a discrete graph (`ptd_dph_normalize_graph` is a
silent no-op above 1), the joint-index / sojourn FFI path
(`ComputeSojournTimesFfiImpl`, `pmf_from_graph_joint_index`), the
moments-only FFI handler (`ComputeMomentsFfiImpl`,
`graph_builder_ffi.cpp:546-575`), and the trace paths (`is_discrete`
hardcoded false, `trace_elimination.py:939`, `phasic.c:13920`). These are added to `found-defects-2026-11-23.md`, not fixed here.

Two existing behaviours that bear on the `was_dph` rule, unchanged: a
native graph deserialized from a record WITHOUT the `was_dph` key is
latched `was_dph = true` (`graph_builder.cpp:148-152`,
`_graph_serialize.py:709`) and so is exempt and renormalised; and
`Graph(g)` re-wrapping drops the Python `is_discrete` attribute.
`was_dph` itself survives `clone`, `copy`, serialize round trip and both
reward transforms (review1/r6.log section A).

### 2.4 Graph-owned cached context (defensive)

`dph_accumulated_visits` (`api/cpp/phasiccpp.h:1773`) reuses a cached
context without the weight-version check its siblings have; it gets the
same check. This is defensive: `update_weights` already destroys the
graph-owned contexts (`notify_change`, `phasiccpp.h:1526-1535`), so no
failing case is known and the test for it is a guard.

### 2.5 Crashes when the check fires

- `DPHProbabilityDistributionContext` (`api/cpp/phasiccpp.h:2346`) stores a
  NULL context when creation fails; `distribution_context()` on an invalid
  discrete graph then dereferences it on the first `.step()`/`.pmf()`. The
  constructor throws via the existing `PTD_THROW_AND_CLEAR()`
  (`phasiccpp.h:46`) instead.
- `sample_multivariate_discrete` (`src/cpp/phasic_pybind.cpp:2511`) uses the
  NULL returned by `ptd_mdph_random_sample`. It checks, frees `vrewards`, and throws via
  `PTD_THROW_AND_CLEAR()`.

### 2.6 Pinned values that change

`tests/pytest/test_gate_discretize_pins.py` compares float hex of
`pdf(1..5)` and other fields for 21 `discretize()` cases. Caps fire on
rows that sum to exactly 1 or just below, so the `pdf` record changes in
most cases (17 of 21 measured for the in-place cap, review2/r4.log;
several pinned pdf(4) values are rounding noise around a true 0, two of
them negative today). Procedure after the implementation:

1. Run the pin test and collect every differing field.
2. Permitted: `pdf` entries only, each with |new - old| <= 5e-16, and no
   new value negative. Every other pinned field (`expected_aux_visits`,
   `expectation_discrete`, edges, structure) must be bit-identical.
3. If and only if 2 holds, regenerate all `pdf` records with the file's
   own generator and record old/new hex for every changed entry in
   `experiments/bug4_discrete_rowsum/pins_diff.log`. Anything else stops
   the work.

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
   exit with a Python exception, not a signal (the first by calling
   `.step()` on the context: construction alone does not crash today).
8. Batched FFI: `vmap` of `pmf_and_moments_from_graph(discrete=True,
   use_ffi=True)` at 1 + 1e-5, in a subprocess: catchable error, no abort.
9. Context held by the user: created at p = 0.5, `update_weights` to 1.5,
   `.step()` raises; updated to a valid 0.25 instead, stepping continues
   and matches a fresh context's values from that state (today: pmf
   [0.75, -0.375, 0.1875], no error).
9b. Negative weight: native parameterized graph, `update_weights` giving
   weights [-0.5, 0.9] (row sum 0.4): `pdf`, `stop_probability`,
   `sample`, `expectation`, `variance` raise naming the weight (today:
   stop_probability [0, 0.6, -0.5, 0], variance -13.5). A NaN weight
   raises too.
9c. `pmf_and_moments_from_graph(discrete=True)` with an empty `times`
   array at 1 + 1e-5 and at 1.5, both `use_ffi` values: raises (today
   moments [0.667, 0.222] at 1.5).
9d. Gradient at the bound, native discrete: `jax.grad` at a gap of 5e-8
   below 1 raises (the finite-difference probe crosses; today it returns
   a value); it works at a gap of 2e-7 and value calls work at 5e-8.
   Written against the post-rebase gradient step (bugs 1-3 do not change
   it; re-measured after the rebase before the numbers are fixed).
9e. Not fail-soft: `phasic._is_rate_blowup` is False for the real
   exception raised by a 1 + 1e-5 graph.

Guards (pass on master and after):
10. p in {0.1, 0.3, 0.5, 1.0}: geometric closed forms — pmf and cdf to
    absolute 1e-15, mean and variance to relative 1e-14.
11. Fresh discrete joint graph (no `update_weights`): `joint_prob_table()`
    does not raise and equals the table after `update_weights([1, 1])` to
    1e-14. (`expectation()` on such a graph returns `inf` today and is
    not asserted.)
12. Validation re-runs on a cached pdf context: valid, `pdf`, update to
    1.5, `pdf` raises.
13. `accumulated_visits` after `update_weights` to p = 1.5 on a graph
    that was valid when first read raises (already true; §2.4).
14. Native discrete graph with rows at most 0.6: `pdf(1..200)` unchanged
    — checked against the closed form here and bit-for-bit in the gate.
15. Continuous closed form (exponential and Erlang pdf/cdf) within the
    documented 2.5e-3 floor — the suite-side check that Euler stepping
    still works; bit-identity is established in the gate (§4).

## 4. Gate

1. Rebase on master, `pixi run install-dev` (native rebuild).
2. Bit-identity, one-off: a script in
   `experiments/bug4_discrete_rowsum/` dumps float hex of `pdf`, `cdf`,
   `stop_probability`, `accumulated_visiting_time` for continuous models
   (coalescent, two-island, two-locus, reward-transformed, parameterized
   through `pmf_from_graph`) and of `pdf(1..200)` for a native discrete
   graph with rows at most 0.6, from the master build (`/Users/kmt/phasic`)
   and the fix build on this machine; the dumps must be identical. The
   fix-build dump is repeated with `MallocScribble=1` (catches an
   unassigned `cap_outflow`).
3. Pin procedure of §2.6.
3b. Timing, on a ~5,000-vertex discrete joint graph and the 2,004-vertex
   discretized coalescent, master build vs fix build, median of 5
   repeats: `pdf` over 1,000 jumps on one context; `sample` of 10,000
   draws. Recorded in the plan. More than 10% slower on either stops the
   work for a rethink.
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
- Discrete pdf/cdf values change at the 1e-16 level wherever a cap fires
  (rows summing to 1 or within rounding of it); nowhere else (§2.6, gate
  step 2).
- The capped step does more work per edge (a compare and a subtract);
  gate step 3b bounds it, and the samplers' cost is unchanged by design.
- Graphs refused that were accepted: rows whose excess is above 1e-9.
  Known cases (review1/r3.log): probabilities normalised in float32
  (excess up to 1.7e-7) and probabilities rounded to 8 decimals (up to
  4e-8). The error message now prints the sum in full, so the cause is
  visible; the remedy is to normalise in float64.
- A discrete graph updated to a negative weight now raises on step-based
  and the §2.3 elimination read-outs, where it returned negative
  probabilities.
