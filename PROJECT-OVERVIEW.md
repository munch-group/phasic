# phasic — project overview

Written 2026-10-10 as the single entry point to the plan, findings and
review files (since 2026-11-26 all in `plans-and-findings/`); updated 2026-10-12 with the October
model-building consistency work (§1d) and the Stage-3 status correction;
updated 2026-10-15 after the C++ `discretize()` merge (§1f); updated
2026-10-27 with the parallel-elimination weight-mode fix (§1c item 8);
updated 2026-11-20 with the November joint-probability likelihood work
(§1g) and its follow-up plan (§1c item 12). Everything here is a summary of those
files (named in each entry) plus git history; nothing new was measured to
write it. When a file and this overview disagree, the file is the record —
and this overview should be corrected.

**Open items are tracked in `OPEN-ITEMS.md`, not here** (since
2026-11-25). Section 1c below is kept as the August-November history of
the open list; for current status and priorities read `OPEN-ITEMS.md`.
Updated 2026-11-26: files moved to `plans-and-findings/`; the November
bug-fix series (bugs 1-5) is recorded in
`plans-and-findings/bugfix-handoff-2026-11-24.md` and the bug plans.

Convention from here on: every ID (batch letter, deferred number, tier,
gate, rule) is written together with a few words saying what it is.

---

## 1. Where the project stands (November 2026)

**November (§1g):** master `038dcc63` carries three merges from
2026-11-20, not yet pushed: the epoch joint-probability likelihood is
now normalised, `tot_reward_limit` no longer corrupts the single-epoch
likelihood, and a C++ segfault at thread exit is fixed. Each was gated
on the 35 test files that build joint-probability graphs, not on the
full suite. Follow-up tickets are tracked in
`plans-and-findings/joint-prob-followups-plan.md`; twelve of them are merged too (master
`7b0b7a7c`, four more merges, not pushed): see §1g.

**October:** master carries both October work streams, merged and pushed: the
model-building consistency work (§1d, merged 2026-10-13) and the C++-owned
`discretize()` (§1f, merged 2026-10-15 at `61b1ab93`). No branch is in
flight. Last full-suite result (on the discretize branch, which contains
the consistency work): 2347 passed, 0 failed, 84 skipped, 26 expected
failures, plus the two files run separately because they exceed the
memory and time caps, both passing. The full suite has not been run on
the merged master, which adds your progress-bar and plot commits
(`aed8cc1c`, `9b82a427`); they touch no file the branch changed.

### 1a. The question the project is currently about: does inference recover the truth?

Work in mid-August turned from "make gradients exact" to "does SVGD fitting
actually recover known parameter values on the tutorial models". The test
file `tests/pytest/test_notebook_model_recovery.py` fits five models derived
from the `docs/pages/tutorial/svgd-*` notebooks to data simulated at known
parameters, from priors that exclude the true value. Four cases fail and are
recorded as expected failures ("pins"). Whether each failure is a real
defect or just too few iterations was settled in
`plans-and-findings/b3-budget-vs-defect-findings.md`:

| Model | What goes wrong | Verdict |
|---|---|---|
| Coalescent, one rate (43 vertices) | Point estimate fine, but the posterior is too narrow and misses the truth; more iterations make it narrower, not better | Defect |
| Joint-probability coalescent (168 vertices) | Stuck ~54% below the true rate; particles pile up at the 1e-9 lower bound | Defect |
| Two-locus ARG, coalescence + recombination | Coalescence recovered; recombination ~86% off, and more compute shrinks the interval around the wrong value | Defect |
| Two-island (186 vertices) | Off at the suite's budget, steadily improving with more iterations | Budget-limited, not a defect |

The mechanism behind the three defects is not known. Candidates named in the
records: kernel bandwidth, preconditioner, step-size schedule.

Two further facts limit how precise any fit can be:

- **The default likelihood is only ~2.5e-3 accurate** (relative), and that
  error biases the estimate by the same amount regardless of sample size.
  Cause: the default discretisation granularity (1024 steps) in the
  continuous PDF. Record: `plans-and-findings/b3-mixed-scale-defect-settled.md`; guarded by
  `tests/pytest/test_likelihood_correctness.py`.
- **Recombination is only identifiable at larger rates**: at 0.05 the
  likelihood is flat over a 25-fold range, so the test uses 0.5.

Your notebook (`tests/pytest/notebook_model_recovery.ipynb`, untracked) is a
first draft of a new test design with MAP-accuracy and trajectory-smoothness
checks. It was copied from an older version of the test file, so it carries
outdated wording for the pins.

### 1b. The exact-gradient programme ("B3") — finished, but aimed at the wrong target

From late July to mid-August most effort went into replacing
finite-difference gradients (estimating a derivative from nearby
evaluations, which fails when parameters differ hugely in scale, e.g. 1e-8
next to 1) with exact analytic gradients. Every planned unit of that work
shipped and passed review. A stock-take on 16 August
(`plans-and-findings/b3-stocktake-2026-08-16.md`) then found two problems:

1. **Scale.** Production models are 20k–60k vertices, and the library must
   handle 200k–600k. The exact gradients were only ever tested up to ~8k
   vertices. At 22.6k vertices no gradient call finished within 15 minutes;
   on two-locus models a safety check makes the exact path quietly hand
   back finite differences anyway, already at ~1k vertices.
   Record: `plans-and-findings/b3-scale-reality-findings.md`.
2. **Wrong quantity.** The programme made the gradient of *moments* exact,
   but the default SVGD likelihood uses the *PDF at the observed times*,
   whose gradient is still finite differences everywhere in the library.
   Record: `plans-and-findings/b3-likelihood-path-map.md`.

### 1c. Open items, in plain language (superseded by `OPEN-ITEMS.md`; kept as history)

1. Explain the three inference defects in 1a. Not started.
2. **Exact gradients for big models, by splitting them into components**
   ("Deferred 1", plan `plans-and-findings/b3-d1-implementation-plan.md`). Large graphs are
   already processed as strongly connected components (clusters of states
   that can reach each other), which are then stitched together. This plan
   would make the gradient follow the same split. It is the only designed
   route to exact gradients above ~10k vertices. Exploratory experiments
   all came back positive. The plan is written and reviewed but **not
   signed off by you**. Its header says v3; the stock-take calls it v4.
3. **Two cheap fixes, not built:**
   - Let the moments gradient skip its numerical-conditioning safety check.
     That check is what makes it fall back to finite differences on
     two-locus models, and the sojourn-time gradient already has such an
     opt-out.
   - Cache the moments recording (tape) between calls. Measured cost
     difference: 3,062x.
4. **Recombination recovery from richer data** (several summary statistics,
   or joint probabilities, instead of only absorption times). Not yet made
   to converge.
5. **Exact gradient of the PDF itself** ("Deferred 3",
   `plans-and-findings/deferred-3-pdf-gradient-revival-plan.md`, `plans-and-findings/b3-d3-derisk-findings.md`).
   This is the only item that would make the *default* fit exact. The
   method is chosen (Poisson-mixture route) and derived on paper. No
   implementation plan yet.
6. **Exact gradient for the earlier time epochs** of time-inhomogeneous
   models ("Deferred 2", `plans-and-findings/deferred-2-daisy-intermediate-epoch-plan.md`,
   `plans-and-findings/b3-d2-derisk-findings.md`). Exploratory work is done. The main reason
   to build it: the current backward pass crashes for parameters below
   ~1e-7. No implementation plan yet.
7. **High-precision (MPFR) gradients** ("Deferred 4"). Parked by your
   decision. Its cheap end is item 3a above.
8. **Loose ends** (the master plan's §16b list). The most important one,
   parallel elimination silently treating every weight as linear (§16b
   item 6), is **fixed** (2026-10-27, `ccc54e33`): the parallel route now
   composes at the graph's current edge weights, so `expectation()`,
   `moments()`, `variance()`, `covariance()` and the JAX moments path give
   the same answer with or without `parallel_elimination` for all four
   weight modes (linear, log, callback, formula). It was wider than
   recorded: callback and formula were wrong too, and a formula graph
   whose coefficient count differs from θ's length left a stale error that
   made the next `update_weights()` fail. Guarded by
   `tests/pytest/test_parallel_elimination_weight_modes.py`. Still open:
   documentation overstating parallel-elimination benefits, a
   forward/backward mismatch when a rate blows up, and failures in the
   multi-epoch FFI that become unlogged NaNs. Full list in
   `plans-and-findings/b3-stocktake-2026-08-16.md` §4.
9. **Leftovers outside the gradient programme** (end of
   `plans-and-findings/b3-execution-tracker.md`):
   - `model-selection.ipynb` uses an outdated, broken pattern.
   - A random tie-break in the epoch model is untested.
   - The old cache format needs hardening.

10. **Notebook edits for you** (§1d): five notebooks used patterns master
    now refuses or changes. As of 2026-10-15, `parametrization.ipynb` and
    `joint-probability.ipynb` look edited; `time-inhomogeneous.ipynb`
    (coalescent cell, rates still plain numbers) and the two `popgen/`
    notebooks (`joint_prob_graph` on a discretized graph) look unchanged.
    None of them re-run.
11. **Follow-ups recorded by the discretize work** (§1f): the θ layout of
    SVGD multi-epoch models, a C++ `StateIndexer` and `joint_prob_graph`,
    `is_aux` in the C++ serialized graph and the SLURM per-component path,
    constant edges in discrete graphs, and a stale Stage-3 plan record.
12. **Joint-probability follow-ups** (§1g), tracked ticket by ticket in
    `plans-and-findings/joint-prob-followups-plan.md`: stale cached graphs in the C++ FFI
    (T1), the broken `reward_rates_callback` route (T2), the continuous
    `joint_prob_table()` returning probability / mutation rate (T3),
    degenerate outcomes picked at random instead of summed (T4), silent
    prior fallbacks (T5), the slow `epoch_starts=[0.0]` build (T6),
    joint-stop-probability graphs mis-routed in `svgd` (T7), no check
    for tables with fewer than two outcomes (T8), multi-property reward
    counting (T9), and a shutdown-order nit in the pybind layer (T10).
    Decided 2026-11-20: T2 implement, T4 merge at construction, T5
    raise, T9 require an explicit `reward_only`. MERGED 2026-11-20:
    T1, T2, T3, T4, T5, T7, T8, T9, T10 and T12 (the `mcmc` default
    prior now gets `rewards=`). MERGED 2026-11-21: T6 (`epoch_starts=[0.0]`
    build 6x faster) and T11 (cap and clear the per-model memory of the
    C++ caches). T13 (suspected poor fit of the default data-informed
    prior) CLOSED 2026-11-21 as not a defect: the check was invalid.

### 1d. Model-building consistency (10–12 October)

Started from your question why a coalescent built with float rates came
back parameterized. Plan `plans-and-findings/constant-parameterized-consistency-plan.md`
(record in its §13), with Part 1 detailed in
`plans-and-findings/graph-cache-constant-roundtrip-plan.md`. Branch
`fix/constant-parameterized-consistency`, merged to master 2026-10-13, one
commit per part:

| Part | What it does | Commit |
|---|---|---|
| Plans | the two plans + this overview | `50f42033` |
| 1 — cache round trip | constant graphs no longer come back parameterized from `serialize()`/the graph cache (they were silently θ-scaled); cache version 1.0 → 1.1, old cache deleted | `553dc412` |
| 2 — one guard | every entry point that takes θ (svgd, mcmc, method_of_moments, joint_* builders, add_epoch, …) refuses a constant model with one message | `e7b7679a` |
| 3a — edge arrays | memory-safety fix: four rate edges + `add_edge_constant` + one more edge on a vertex crashed Python | `54533414` |
| 3 — lossless widening | layout widening no longer drops edges, merges aux vertices, truncates coefficients or drops metadata; refuses where undefined (non-linear weight modes, ragged coefficients, constant edges in discretized/joint graphs, epochs on discrete graphs) | `2181b9ff` |
| 4 — discretize callback | `discretize(rate=callable)` returns one number for any graph; on a parameterized graph it goes into a new θ slot | `f67e175d` |

Every part passed its own new tests and the existing tests touching it; an
adversarial diff review ran for each (for Parts 2 and 3a only after
committing — recorded). The full-suite run was done on the discretize
branch built on top of it (§1f): 0 failures.

**Notebooks to edit (yours):** `parametrization.ipynb` cell 23 (now
raises: drop the reserved `[rate, 0]` slot, callback returns
`nr_lineages`); `joint-probability.ipynb` cell 3 and
`time-inhomogeneous.ipynb` cell 32 (rates as lists); `popgen/coalescent-derived.ipynb`
cells 8→26 and `popgen/old-moran-coalescent-derived.ipynb` cells 19→37
(`joint_prob_graph` on a discretized graph; never run, already broken
before the branch).

**Left open (recorded in the plan's §13):** constant edges in discrete and
joint graphs are refused rather than supported (needs a C change);
`joint_prob_graph(reward_rates_callback=...)` has been broken since
21 May 2026 (calls a commented-out method; now ticket T2 in
`plans-and-findings/joint-prob-followups-plan.md`); per-state rate APIs not yet on
the new rules (`joint_prob_graph`'s `mutation_rate`; the legacy
`_discretize_inplace` and the C++ `discretize` are resolved, §1f);
`expectation()` on a discretized
graph counts all steps, the mutation count needs `graph.rewards` passed
explicitly.

### 1e. The Stage-3 refactor is only partly done

The July refactor plan (`plans-and-findings/stage3-refactor-plan.md`) is the larger plan in
force: one implementation per capability, with the C++ API as the single
source of truth (its decision Q4) and Python delegating. Executed so far:
WS-A (safety-net test gates), WS-B (dead-code removal) and part of WS-C
(splitting the `Graph` class), merged at `9de5d61a`. Not started: WS-D
(thinning the pybind layer), WS-E (the C++ single-source promotions), WS-F
(splitting the C core), WS-G (doctrine document and CI guards), except
that WS-E's step 11 (discretize to C++) is done (§1f), its first
promotion.

### 1f. One `discretize()`, owned by C++ (13–15 October)

Plan `plans-and-findings/discretize-cpp-ssot-plan.md` (v2; its §13 is the implementation
record), exploratory measurements in `plans-and-findings/discretize-cpp-derisk-findings.md`.
Branch `refactor/discretize-cpp-ssot`, merged to master 2026-10-15
(`61b1ab93`, pushed).

| Step | What it does | Commit |
|---|---|---|
| Exploration | Where `discretize()` spent its time (the Python widening loop, 70–95%); found `clone()` quadratic | `2de18c39` |
| Phase 0 — pins | Records of today's `discretize()` behaviour, compared bit for bit by `test_gate_discretize_pins.py` | `fa67c9e5`, `14a8ce13` |
| Phase 1 — widening | One C++ widening primitive (`widen_layout`) replaces the Python helper; 14–35× faster | `c309719c` (+ crash fix `3ada4c08`) |
| Phase 2 — θ layout | `Graph.theta_layout` / `set_parameter_name`: which θ slot is which, carried through copies and widening | `54dcf36e` |
| Phase 3 — C++ discretize | The C++ `discretize` reproduces the Python one exactly | `fcd70289` |
| Phase 4 — thin wrapper | Python `discretize()` calls C++; the legacy `_discretize_inplace` deleted; a discretized constant graph keeps its callback and indexer | `560167a5` |
| Phase 5 — serialize | `is_aux` (which vertices are discretization steps) survives `serialize()` | `dad3c5e5` |
| `clone()` linear | C `ptd_clone_graph` no longer scans all vertices per edge: 110 s → 0.55 s at 204k vertices | `bbb016bc` |
| Refusals | `discretize()` refuses a native discrete graph (it returned a wrong one); `skip_existing` keeps weights | `644132a0` |
| C fixes | `normalize()` no longer produces NaN on all-zero rows; `joint_stop_prob_graph` / `joint_sojourn_graph` refuse discrete sources | `33cd4fa5` |
| Removal | The unused C function `ptd_graph_pdf_with_gradient` and math-reference chapter 16 | `9ae753c4` |

Result at 204k vertices (coalescent): `discretize()` 17.7 s → 0.8 s with a
fixed rate, 18.3 s → 1.7 s with a rate callback. Follow-ups are item 11 in
§1c; full list in the plan's §13.

### 1g. Joint-probability likelihood correctness (19–20 November)

Started from your reproducer `docs/pages/tutorial/test.py`: a two-epoch
model with both epochs' rates tied estimated a coalescence rate of 6.44
where the truth was 5.0, while the same data fitted without epochs
recovered 5.0. Three fixes, each on its own branch, merged into master
2026-11-20 with `--no-ff` (not pushed):

| Merge | Fix | Record |
|---|---|---|
| `e7282362` (commits `8feea1d1`, `656beb41`) | **Epoch likelihood normalised.** Observations beyond `reward_limit`/`tot_reward_limit` cannot be passed to inference, so the correct likelihood conditions on not overflowing: each observation's probability divided by its own table mass Z(θ, exposure). The single-epoch path did this; the epoch (daisy-chain) path used raw probabilities, which pulls estimates toward rates that reduce overflow (the 6.44). Now normalised on both the no-exposure and the exposure branch, including the exact final-epoch gradient. `final_read='stopprob'` is rejected for fits (new rule R33), because normalising it would cancel the penalty for mass not yet absorbed. The reproducer's likelihood now peaks at 5.000000. No measurable slowdown at 45,704 vertices | `plans-and-findings/epoch-normalization-plan.md` (v2, two review rounds), `plans-and-findings/epoch-normalization-findings.md`, `tests/pytest/test_epoch_normalization.py` (17 tests) |
| `9cedb4f3` (commit `86e1f8ac`) | **Total-cap trash edge.** `joint_prob_graph` gave an outcome vertex sitting at `tot_reward_limit` an edge to the trash vertex. On a continuous graph that made the single-epoch likelihood 43–49% wrong for every capped outcome; with `reward_only` discrete graphs were wrong too. The epoch path was never affected | `tests/pytest/test_joint_prob_tot_cap.py` (28 tests, against closed forms) |
| `038dcc63` (commit `2dc4f60f`) | **Segfault at interpreter exit** after long test runs. A C++ destructor (`~GraphBuilder`) erased from a per-thread map that had already been destroyed when JAX's worker threads shut down. Located with lldb and disassembly, confirmed by thread tracing | `tests/pytest/test_thread_local_teardown.py` (macOS-reliable reproducer) |

Every fix was checked against independent closed forms or a
deterministic reproducer, passed the 35 test files that build
joint-probability graphs, and was read by an independent adversarial
reviewer. The full test suite was not run.

**Deferred by your decision:** keeping over-cap observations as an
explicit overflow category would make the likelihood about twice as
informative. A reviewed design is in `plans-and-findings/overflow-observations-plan.md`
(v3); the note is also in `CLAUDE.md`.

**Your notebooks:** the epoch-fit notebooks (`svgd-joint-prob`,
`model-selection`, `time-inhomogeneous`, `popgen/ghost`) and any
notebook using `tot_reward_limit` on a continuous joint graph will show
different (corrected) numbers when re-run.

**Follow-ups merged the same day** (not pushed), each ticket with its
own tests and both branches reviewed independently:

| Merge | Tickets | Record |
|---|---|---|
| `31e799b3` (branch `fix/joint-prob-callback`) | T7 `joint_prob_table()` refuses graphs derived from a joint graph; T8 joint-probability inference needs at least two outcomes; T3 the continuous `joint_prob_table()` reports probabilities (was probability / mutation rate); T4 outcomes that differ only in their absorbing base state are merged at construction (two-island: 54 → 27 outcome vertices, results equal the old group sums); T5 no silent fallback to a standard-normal prior in `svgd`, the epoch path or `mcmc`; T2 `reward_rates_callback` works again, with the caps enforced by the builder; T9 an indexer with several properties requires an explicit `reward_only`; T12 `mcmc`'s default prior receives `rewards=`. Review fixes: `reward_only` names are validated, the subset merge no longer depends on property order, a dict `reward_limit` (never worked) is refused | `plans-and-findings/joint-prob-followups-plan.md` §6; gate 742 + 146 + 112 passed, 0 failed |
| `928bfb6c` (branch `fix/ffi-stale-graph-cache`) | T1 the C++ FFI per-thread graph caches are keyed by a process-unique model id, so a new model can no longer run on a destroyed model's cached graph (silent NaN, 13–50 per 3,000 threads before); T10 a cached Python object is never released after interpreter shutdown | `tests/pytest/test_ffi_stale_graph_cache.py`; gate 592 passed, 0 failed |

**Your notebooks after T9:** notebooks that build a joint graph from an
indexer with several properties must now pass `reward_only`: two-locus
models `['loc1', 'loc2']`, models with a label property (`in_pop`,
`is_derived`) `['descendants']`. Before T9 a label property was counted
as an extra independent mutation process, so those notebooks' earlier
numbers were probably wrong (inferred from the code, not re-run):
`svgd-joint-prob`, `model-selection`, `smc`, `method-of-moments`,
`coalescent-derived`, `old-moran-coalescent-derived`, possibly
`time-inhomogeneous`.

**Merged 2026-11-21** (not pushed):

| Merge | Ticket | Record |
|---|---|---|
| `36edaaa8` (branch `fix/epoch-build-cost`) | T6: the `epoch_starts=[0.0]` build took 47.7 s at 45,704 vertices, now 7.5 s (4.9 s with an explicit prior), outputs bit-identical. Causes: `joint_stop_prob_graph` scanned a list of ~16k numpy ints once per vertex, and the default prior re-encoded the 6.3 MB graph JSON on each of 158 calls | `plans-and-findings/joint-prob-followups-plan.md` §6; gate 629 passed, 0 failed |
| `7b0b7a7c` (branch `fix/ffi-cache-eviction`) | T11: the C++ caches kept every model's structure plus one graph per worker thread until exit (~290 MB per 45,704-vertex model). Now at most `model_memory_cache_size` models (default 4, `phasic.configure`) stay cached process-wide; `phasic.clear_model_memory()` frees them and `phasic.model_memory_info()` reports usage. The review caught a use-after-free in the first version (fixed before merge) | `tests/pytest/test_model_memory.py`; gate 602 passed, 0 failed |

**Closed, not a defect:** T13, the suspected poor fit of the default
data-informed prior (`probability_matching`). The check that found it
called the internal module function with observation tuples instead of
vertex indices. Through `Graph.probability_matching`, which every
caller uses, the fit matches the truth (coalescent n=4/6: estimated
mu/c 0.589 vs 0.6, log-likelihood within 0.6 of the truth's);
`plans-and-findings/joint-prob-followups-plan.md` (§1c item 12).

**SVGD notices (21 Nov):**
- `probability_matching` now rejects input that isn't vertex indices.
- An explicit `preconditioner='jacobian'` no longer gets a notice.
- The `SvgdAssumptionWarning` notices from `Graph.svgd` are one short
  sentence each. Each ends with `.effective_options('<option>', ...)`, a
  new single-option view that prints the full explanation and the
  arguments that state the choice explicitly. The epoch notice appears
  only when prior, fixed or theta_dim were actually expanded. The
  `effective_options()` table shows readable values (`svgd.GaussPrior`,
  `'auto' -> 'jacobian'`).
- Record: `plans-and-findings/svgd-notices-plan.md`.

---

## 2. How the work unfolded (timeline)

| When | Work stream | What it was | Status |
|---|---|---|---|
| early Jul | Hardening fixes (H1–H10, C1–C3) | Ten bug fixes and three cache-security fixes (untrusted pickles, tar members) | Done |
| 5–9 Jul | Refactor, Stages 1–3 | Remove dead code; build a test safety net; split the 12k-line `Graph` class into modules | Stages 1–2 done; Stage 3 partly (WS-A, WS-B, part of WS-C; merged `9de5d61a`); WS-D–G not started (§1e) |
| 15–19 Jul | Differential audit | Compare post-refactor behaviour with the pre-refactor commit `3082ebc6`; log every divergence | Done; findings F-001… in `plans-and-findings/audit-differential-findings.md` |
| 17 Jul | Category-A fixes (A1–A3) | The three audit bugs reachable by users | Done |
| 27–30 Jul | SVGD robustness + likelihood-ratio tests | Stop runaway particles from crashing a fit; make nested LRTs work for tied-vs-free epoch models; public `Graph.epoch_model` | Done |
| 30 Jul–4 Aug | B3 exploration | Prototype exact gradients; first exact moments gradient (continuous, then discrete); log weight mode; joint-index (sojourn) gradient | Done |
| 4 Aug | Exact/FD atlas | Map of which code paths have exact gradients and which `Graph.svgd()` can actually reach | Reference |
| 11–15 Aug | B3 master plan, Batches 0, A–H | Systematic rollout of exact gradients across weight modes, rewards and SVGD routes | All shipped |
| 15–16 Aug | Deferred 1–4 exploration | Experiments for the four big postponed items | Done; decisions pending |
| 16 Aug | Stock-take + scale check | Discovery that B3 doesn't reach production size or the default likelihood | See 1b |
| 16–17 Aug | Inference correctness | Recovery tests on the tutorial models; budget-vs-defect verdicts | See 1a |
| 10–12 Oct | Model-building consistency | Constant vs parameterized models, lossless widening, one-number discretize callback, edge-array memory fix | Merged 13 Oct (§1d) |
| 12–15 Oct | discretize → C++ | One C++ `discretize()` + widening and layout standards; linear `clone()`; C fixes | Merged 15 Oct (§1f) |
| 19–20 Nov | Joint-probability likelihood | Epoch likelihood normalised; total-cap trash edge; thread-exit segfault; overflow design deferred | Merged 20 Nov, not pushed (§1g); ten follow-up tickets merged the same day (`928bfb6c`), T6 and T11 on 21 Nov (`7b0b7a7c`); T13 closed as not a defect |

---

## 3. Label decoder

Several schemes reuse the same letters. **Collisions are marked ⚠.**

| Label | Meaning | Defined in |
|---|---|---|
| **B3** | The whole exact-gradient programme. Originally step 3 of a five-step plan B0–B4 for fixing finite-difference gradients (B0 pin the bug with a test, B1 guard against negative rates, B2 build a verification harness, B3 replace the FD backward passes, B4 adversarial verification) | `plans-and-findings/b3-derisking-strategy.md`; decoder in `tmp.md` |
| **Batch 0** | Merge three near-identical C gradient functions into one shared core | `plans-and-findings/b3-batch0-plan.md` |
| **Batch A** ⚠ | (Aug) Exact moments gradient with `rewards`. ⚠ In July "Batch A / A1–A4" was an unrelated audit-fix batch, and "A1–A3" are also the Category-A bug fixes | `plans-and-findings/b3-batchA-plan.md` |
| **Batch B / C** | Exact gradient for `weight_mode='formula'` / `'callback'` | `plans-and-findings/b3-batchB-plan.md`, `plans-and-findings/b3-batchC-plan.md` |
| **Batch D (D.1–D.4)** ⚠ | D.1 fix a crash of `moments_from_graph` under `jax.vmap`; D.2 add safety checks to a debug-only C function; D.3 pass an option through to joint-probability SVGD (cancelled, folded into G.1); D.4 pass `exact_moment_grad` through `Graph.svgd()`. ⚠ Not the same as D1 = Deferred 1 | `plans-and-findings/b3-batchD-tier1-plan.md`, `plans-and-findings/b3-batchD3-plan.md` |
| **Batch E** | Exact gradient for joint-probability models with de-duplicated observations ("baked" mode) | `plans-and-findings/b3-batchE-plan.md` |
| **Batch F** / **D6** | Remove a JAX construct (`lax.cond`) that, under `vmap`, computed both the exact and the FD gradient. D6 was its name inside the joint-index plan | `plans-and-findings/b3-batchF-plan.md` |
| **Batch G (G.1, G.2)** ⚠ | Expose exact-gradient options through `Graph.svgd()` (G.1: multi-epoch final epoch; G.2: multi-feature rewards). ⚠ Not gates G0–G5, nor audit findings G1/G2 | `plans-and-findings/b3-batchG1-plan.md`, `plans-and-findings/b3-batchG2-plan.md` |
| **Batch H** ⚠ | Exact gradient for the final epoch of multi-epoch (daisy-chain) models. ⚠ Not hardening fixes H1–H10 | `plans-and-findings/b3-batchH-plan.md` |
| **Batch 1–3** (Jul) ⚠ | Early B3 steps: tape adjoint, first exact gradient, higher moments. ⚠ "Batch 1/2" also name the LRT fix and the `epoch_model` builder | `plans-and-findings/b3-batch{1,2,3}-*-findings.md`, `plans-and-findings/batch2-free-epoch-model-plan.md` |
| **Deferred 1 (D1, Def-1)** | Exact gradients for big models split into strongly connected components | `plans-and-findings/deferred-1-…-plan.md`, `plans-and-findings/b3-d1-*.md` |
| **D1-E2 guard** | Small shipped fix from Deferred 1 experiment E2: exact-gradient code now refuses placeholder "synthetic" component graphs, where it used to return plausible but wrong numbers | `plans-and-findings/b3-d1-e2-guard-plan.md` |
| **Deferred 2 (D2)** | Exact gradient for the earlier epochs of multi-epoch models | `plans-and-findings/deferred-2-…-plan.md`, `plans-and-findings/b3-d2-derisk-findings.md` |
| **Deferred 3 (D3)** | Exact gradient of the PDF/PMF at observed times | `plans-and-findings/deferred-3-…-plan.md`, `plans-and-findings/b3-d3-*.md` |
| **Deferred 4 (D4)** | High-precision (MPFR) gradients for badly conditioned cases; parked | `plans-and-findings/deferred-4-…-plan.md`, `plans-and-findings/b3-d4-sweep-findings.md` |
| **E0–E5** | Numbered experiments *inside* each deferred plan; the numbers restart in every plan | the deferred plans |
| **CC-1, CC-2** | "Cheap checks" in the master plan. CC-2 = the conditioning sweep that led to parking Deferred 4 | `plans-and-findings/b3-exact-gradient-master-plan.md` §15 |
| **Phase 1–4** ⚠ | (B3) Ordering groups of batches in the master plan; all complete. ⚠ In July: the four phases of the numerical audit (forward values, gradients, invariants, weak tests) | master plan §15 / `tmp.md` |
| **Tier 1–3** ⚠ | (B3) Three candidate strategies: exact differentiation via the trace / smarter FD step (abandoned) / C adjoint (chosen). ⚠ In the recovery tests, Tiers 0–2 mean sampler check / likelihood-peak check / full SVGD fit. ⚠ "Batch D Tier-1" and "SVGD plumbing Tier 3" mean yet another thing | `plans-and-findings/b3-derisking-strategy.md`; test file docstring |
| **G0–G5** | Checklist every batch passed: baseline current, own tests, targeted tests, full suite vs ledger, adversarial diff review, merge record | `plans-and-findings/b3-execution-process.md` §4 |
| **§16b items 1–14** | The master plan's list of unscheduled loose ends | master plan §16b |
| **R9, R29–R33** | Numbered argument-validation rules in `svgd_config.py` that reject an option where it would have no effect (R33: `final_read='stopprob'` refused for epoch fits) | `src/phasic/svgd_config.py` |
| **T1–T10** | Joint-probability follow-up tickets (November) | `plans-and-findings/joint-prob-followups-plan.md` |
| **D1–D4 (epoch plan)** ⚠ | Decisions in the epoch-normalisation plan (D2a = refuse `stopprob` for fits). ⚠ Not Deferred 1–4 | `plans-and-findings/epoch-normalization-plan.md` §7 |
| **Leaf 1–5** | Branches of `Graph.svgd()`'s internal routing: 1 multi-epoch, 2 joint-probability, 3 multi-feature rewards, 4 single rewards, 5 plain moments | `plans-and-findings/exact-fd-atlas-svgd-reachability.md` |
| **DR-A, DR-D** | Early "de-risk" experiments: DR-A an analytic gradient oracle, DR-D a scale-aware FD step (dead end) | `plans-and-findings/b3-experiment-findings.md` |
| **F-001…** ⚠ | Audit findings. ⚠ Also "F1–F7" (audit phase-1 findings), "F-P1…" (reviewer findings inside plans) | `plans-and-findings/audit-differential-findings.md` |
| **N1–N12, G1–G2, P2-x** | Defects found during the July audit (N), its headline findings (G), its early claims (P2) | `tmp.md` |
| **WS-A … WS-G** | Workstreams of the Stage-3 refactor: A safety-net test gates, B dead-code removal, C splitting Python god-objects (the Graph class, svgd.py), D thinning the pybind layer, E C++ single-source promotions, F splitting the C core, G doctrine document + CI guards | `plans-and-findings/stage3-refactor-plan.md` §7 |
| **Stage-3 steps 0–20** | The refactor's ordered commit sequence; step 11 = discretize to C++ | `plans-and-findings/stage3-refactor-plan.md` §8 |
| **Q1–Q13, Q-CODEGEN** | Decisions recorded in the refactor plan; Q4 = "C++ API is a product, its ports become the single source of truth"; Q13 = freeze the public C symbols | `plans-and-findings/stage3-refactor-plan.md` §9 |
| **Part 1, 2, 3a, 3, 4** | The model-building consistency work: cache round trip / one guard for constant models / edge-array memory fix / lossless widening / one-number discretize callback | `plans-and-findings/constant-parameterized-consistency-plan.md` |
| **P1–P4, O1–O8** | Widening standard: P = what the shared widening primitive guarantees; O = what each operation (discretize, add_epoch, joint_prob_graph) must check, change and record | `plans-and-findings/discretize-cpp-ssot-plan.md` §3 |
| **Fix A / C / D** | Fixes for the nested likelihood-ratio test on epoch models | `plans-and-findings/svgd-lrt-fix-a-plan.md` |
| **Pins 1–4** | The four expected-failure recovery tests in 1a | `plans-and-findings/b3-budget-vs-defect-findings.md` |
| **Ledger stamps** (e.g. "11th stamp 2012/0/84/24") | Recorded full-suite results: passed / failed / skipped / expected-fail | `plans-and-findings/b3-test-baseline.md` |
| **SOUND-WITH-CORRECTIONS** etc. | Verdicts from adversarial review agents | — |

---

## 4. File index

**Current = read for present work. History = record of finished work; keep, but no need to read.**

### Start here
| File | What it is | |
|---|---|---|
| `OPEN-ITEMS.md` | The register of everything still open, with priorities | Current |
| `PROJECT-OVERVIEW.md` | This file | Current |
| `CLAUDE.md` | Build, test, architecture notes and the long list of known gaps | Current |
| `plans-and-findings/b3-stocktake-2026-08-16.md` | Is the gradient work aligned with what the library is for? (No: scale and wrong quantity) | Current |
| `plans-and-findings/b3-budget-vs-defect-findings.md` | Verdicts on the four failing recovery tests | Current |
| `plans-and-findings/b3-likelihood-path-map.md` | What each inference method actually computes, traced through the source | Current |

### Inference correctness (Aug 16–17)
| File | What it is | |
|---|---|---|
| `plans-and-findings/b3-mixed-scale-defect-settled.md` | FD gradient error at mixed scales is smaller than thought; the 2.5e-3 likelihood error is the bigger problem | Current |
| `plans-and-findings/b3-scale-reality-findings.md` | Measurements of the exact gradients at 1k–22k vertices; tape growth per model type | Current |
| `plans-and-findings/b3-inference-accuracy-table.md` | Generated table: true vs inferred parameters with seven convergence checks | Current |

### Exact-gradient programme (B3): overview and process
| File | What it is | |
|---|---|---|
| `plans-and-findings/b3-execution-tracker.md` | Status table for every batch (dense; see decoder above) | History, partly current |
| `plans-and-findings/b3-exact-gradient-master-plan.md` | 1,482-line plan sequencing Batches 0, A–H and Deferred 1–4 | History |
| `plans-and-findings/b3-execution-process.md` | Branching, review and test-gate rules used for every batch | Reference |
| `plans-and-findings/b3-test-baseline.md` | Full-suite results ledger | Reference |
| `plans-and-findings/exact-fd-atlas-SUMMARY.md` + `plans-and-findings/exact-fd-atlas-*.md`, `plans-and-findings/plan-feasibility-*.md` (14 files, moved from `atlas/` 2026-11-26) | August map of exact vs FD gradient coverage (7 atlas files) and the six feasibility studies behind Batches A-H and Deferred 1-4. Coverage claims are superseded by those batches; still-open findings are in `OPEN-ITEMS.md` (reviewed 2026-11-26) | History |
| `plans-and-findings/b3-derisking-strategy.md` | The original exploration plan for replacing FD gradients | History |

### B3: per-batch plans and findings (all shipped)
`plans-and-findings/b3-batch0-plan.md`, `plans-and-findings/b3-batchA-*`, `plans-and-findings/b3-batchB-*`, `plans-and-findings/b3-batchC-*`,
`plans-and-findings/b3-batchD-tier1-plan.md`, `plans-and-findings/b3-batchD3-plan.md`, `plans-and-findings/b3-batchE-*`,
`plans-and-findings/b3-batchF-plan.md`, `plans-and-findings/b3-batchG1-*`, `plans-and-findings/b3-batchG2-*`, `plans-and-findings/b3-batchH-*` —
meanings in the decoder. History.

### B3: early work, late July – early August (shipped)
`plans-and-findings/b3-experiment-findings.md`, `plans-and-findings/b3-prototype-findings.md`,
`plans-and-findings/b3-c-theta-adjoint-plan.md`, `plans-and-findings/b3-batch0-realtape-findings.md`,
`plans-and-findings/b3-batch1-reverse-adjoint-findings.md`, `plans-and-findings/b3-batch2-exact-grad-findings.md`,
`plans-and-findings/b3-batch3-higher-moments-findings.md`, `plans-and-findings/b3-batch3-mpfr-and-discrete-derisk.md`,
`plans-and-findings/B3-MERGE-REVIEW.md` (first exact moments gradient),
`plans-and-findings/b3-discrete-theta-adjoint-plan.md` + `plans-and-findings/B3-DISCRETE-MERGE-REVIEW.md`
(discrete models), `plans-and-findings/b3-log-weight-mode-plan.md`, `plans-and-findings/b3-joint-index-plan.md`
(sojourn-time gradient). History.

### B3: deferred items (open decisions)
| File | What it is | |
|---|---|---|
| `plans-and-findings/deferred-1-hierarchical-scc-adjoint-plan.md`, `plans-and-findings/b3-d1-derisk-findings.md`, `plans-and-findings/b3-d1-implementation-plan.md` | Component-split gradients for big models: exploration plan, results, implementation plan awaiting sign-off | Current |
| `plans-and-findings/b3-d1-e2-guard-plan.md` | The shipped synthetic-graph guard | History |
| `plans-and-findings/deferred-2-daisy-intermediate-epoch-plan.md`, `plans-and-findings/b3-d2-derisk-findings.md` | Earlier-epoch gradients | Current |
| `plans-and-findings/deferred-3-pdf-gradient-revival-plan.md`, `plans-and-findings/b3-d3-derisk-findings.md`, `plans-and-findings/b3-d3-e5-derivation-dossier.md` | PDF gradient: plan, results, derivation | Current |
| `plans-and-findings/deferred-4-mpfr-conditioning-floor-plan.md`, `plans-and-findings/b3-d4-sweep-findings.md` | High-precision gradients (parked) | History |

### Model-building consistency and the C++ discretize (October)
| File | What it is | |
|---|---|---|
| `plans-and-findings/constant-parameterized-consistency-plan.md` | Parts 1–4: constant vs parameterized models, lossless widening, discretize callback; §13 is the implementation record | Current |
| `plans-and-findings/graph-cache-constant-roundtrip-plan.md` | Part 1 in detail: constant graphs coming back parameterized from the graph cache | Current |
| `plans-and-findings/discretize-cpp-ssot-plan.md` | One `discretize()` owned by C++, with the widening and layout standards; implemented and merged, §13 is the record and lists the follow-ups | Current |
| `plans-and-findings/discretize-cpp-derisk-findings.md` | Measurements behind that plan (where the time went, quadratic `clone()`, C bugs found on the way) | History |
| `plans-and-findings/test-suite-timings-2026-10-12.md` | Time per test file and the slowest tests | Reference |
| `plans-and-findings/stage3-refactor-plan.md` | The larger refactoring plan still in force (doctrine, workstreams, commit sequence, decisions) | Current |

### Joint-probability likelihood (November)
| File | What it is | |
|---|---|---|
| `plans-and-findings/joint-prob-followups-plan.md` | Plan and tracker for the follow-up tickets T1–T10, with decisions and a log | Current |
| `plans-and-findings/epoch-normalization-plan.md` | Plan (v2) for normalising the epoch likelihood; §9 lists the defects found on the way | History |
| `plans-and-findings/epoch-normalization-findings.md` | Closed-form checks, test gate and timing for that fix | History |
| `plans-and-findings/overflow-observations-plan.md` | Reviewed design (v3) for keeping over-cap observations as an overflow category; deferred | Current (deferred) |
| `experiments/dr_epochnorm_checks.py`, `experiments/dr_tot_reward_trash.py` | Scripts behind those checks | History |

### Priors, notices and the November bug-fix series (21-26 November)
| File | What it is | |
|---|---|---|
| `plans-and-findings/bugfix-handoff-2026-11-24.md` | Handoff and state of bugs 1-5 (all merged) | History |
| `plans-and-findings/bug1-compiled-cache-plan.md` … `bug5-graph-cache-plan.md` | The five bug plans (compiled-gradient cache, fixed values, BetaPrior transform, discrete row sums, stale graph cache), each with its review record | History |
| `plans-and-findings/found-defects-2026-11-23.md` | Defects found during those reviews, not approved for fixing (now rows OI-010 to OI-024 in `OPEN-ITEMS.md`) | Current |
| `plans-and-findings/fd-step-and-slow-band-guard-plan.md` | Scale-free likelihood and gradients (v5, parked, not reviewed) | Current (parked) |
| `plans-and-findings/epoch-fit-prior-and-fd-floor-plan.md` | Investigation of the epoch-fit hang and negative-rate FD error; split into the next two | History |
| `plans-and-findings/epoch-default-prior-and-truncated-gauss-plan.md` | Default epoch prior and GaussPrior `lower=` (merged) | History |
| `plans-and-findings/svgd-notices-plan.md` | Short SVGD notices and `effective_options('<option>')` (merged) | History |
| `plans-and-findings/hpd-hdr-naming-plan.md` | HPD/HDR method renames with deprecation aliases (October) | History |

### Refactor and audit (July)
`plans-and-findings/stage2-coverage-safety-net-handoff.md`,
`plans-and-findings/stage3-execution-handoff.md` (what of Stage 3 was executed), `plans-and-findings/deferred-notebook-triage.md`,
`plans-and-findings/audit-situation-map.md` (also the map of all compute paths — still useful),
`plans-and-findings/audit-differential-findings.md`, `plans-and-findings/category-A-fix-plans.md`. History.

### SVGD robustness and model selection (late July)
`plans-and-findings/deferred-svgd-divergece-fix.md`, `plans-and-findings/svgd-divergence-robustness-plan.md`,
`plans-and-findings/deferred-svgd-lr-bug.md`, `plans-and-findings/svgd-lrt-model-reuse-plan.md`,
`plans-and-findings/svgd-lrt-fix-a-plan.md`, `plans-and-findings/batch2-free-epoch-model-plan.md`. History.

### Other
| File | What it is |
|---|---|
| `plans-and-findings/sojourn-fix.md` | Cut memory of sojourn-time computation from O(n·k) to O(n) (shipped) |
| `plans-and-findings/tree_toplogy_encoding.md` | Encoding tree topologies as states |
| `plans-and-findings/MCMC_TODO.md` | April 2026 to-do list for the MCMC class (moved from `docs/`). Since done: plots (posterior, chains, autocorrelation), vmap chains, thinning. Still open: warm restart (OI-084); its claim of 27 passing tests in `test_mcmc.py` is outdated (all commented out, OI-066) |
| `tmp.md` (git-ignored) | An earlier decoder of the July audit labels |
| `experiments/` (58 scripts) | Scripts behind the findings files (`dr_*.py`) |

---

## 5. Suggested tidy-up (partly done 2026-11-26: plans and findings moved to `plans-and-findings/`)

- Move every "History" file into `docs/history/` (or `notes/archive/`), so
  that the repo root holds only this overview, `CLAUDE.md`, `README.md` and
  the "Current" files.
- Give new plans descriptive names (for example `pdf-gradient-plan.md`)
  instead of letter-number IDs.
- Trim the B3 sections of `CLAUDE.md`, which are now mostly history, to a
  pointer to this file.
