# Joint-probability follow-ups — plan and tracker

Written 2026-11-20. The overall plan for the defects found during the
November joint-probability work (`epoch-normalization-plan.md` §9, the
two adversarial reviews of the tot-cap and thread-exit fixes, and the
user's list of 2026-11-20). One ticket at a time, each on its own
worktree branch from master, merged with `--no-ff` after its gate.
This file is updated after every ticket (status table + §6 log).

## 0. Copy-paste prompt for a fresh session

> Read `joint-prob-followups-plan.md` (this file) and `PROJECT-OVERVIEW.md`
> §1g first. Work the tickets in §3 in the order of §4, one worktree
> branch per ticket from master (`git worktree add -b fix/<name>
> ../phasic-<name> master`, then `pixi install && pixi run install-dev`
> in it). For each ticket: reproduce the defect against an independent
> reference first, then fix, then the ticket's gate (§5), an independent
> adversarial review of the diff, commit, update this file's §2 table and
> §6 log, and ask before merging. Every run memory-capped (scratchpad
> `capped.py` / `capped_pytest.py` pattern: RSS watchdog); one heavy run
> at a time. Never change behaviour behind a decision marked OPEN in §2
> without the user's answer.

## 1. State snapshot (2026-11-20)

Master `038dcc63` carries three merged fixes (none pushed):

| Merge | What | Record |
|---|---|---|
| `e7282362` | Epoch (daisy-chain) joint-prob likelihood normalised per observation; `final_read='stopprob'` rejected for fits (rule R33) | `epoch-normalization-plan.md`, `-findings.md` |
| `9cedb4f3` | `joint_prob_graph`: t-vertices get no trash edge (continuous single-epoch likelihood was 43-49% wrong at a binding `tot_reward_limit`; discrete with `reward_only` too) | commit `86e1f8ac`, `tests/pytest/test_joint_prob_tot_cap.py` |
| `038dcc63` | C++ segfault at thread exit: `~GraphBuilder` erased from an already-destroyed thread_local map | commit `2dc4f60f`, `tests/pytest/test_thread_local_teardown.py` |

Gate subset: until 2026-11-20 the 35 files that build a
joint-probability graph (`grep -rl "joint_prob_graph\|tot_reward_limit"
tests/pytest`). WIDENED 2026-11-20 after the T8 review: that list missed
17 tests in 5 files that use `pmf_from_graph_joint_index` on plain
graphs. Now every file matching
`joint_prob_graph|tot_reward_limit|pmf_from_graph_joint_index|joint_prob_table|probability_matching|epoch_model|_map_joint_observations`
plus `test_thread_local_teardown.py` (44 files on
fix/joint-prob-tickets-1). C++ tickets additionally run every file that
reaches the FFI. Tot-cap fix: 618 passed, 6
skipped, 5 xfailed, 0 failed.

## 2. Ticket status

| # | Ticket | Kind | Decision | Status |
|---|---|---|---|---|
| T1 | `per_thread_graph_cache` stale address key (C++) | correctness | — | committed `5877fb9f`, `74d15736` on fix/ffi-stale-graph-cache (worktree ../phasic-tlfix); review clean; gate pending |
| T2 | `joint_prob_graph(reward_rates_callback=...)` calls a commented-out method | broken API | DECIDED: implement in the fast builder; review done (oversight); the BUILDER enforces the caps (2026-11-20) | committed `9ddb0d96` on fix/joint-prob-callback (stacked on fix/joint-prob-tickets-1) |
| T3 | `joint_prob_table()` on a continuous graph returns probability / mu | correctness | — | committed `ad76a044` on fix/joint-prob-tickets-1 |
| T4 | Degenerate outcomes (several t-vertices, same reward vector) resolved by a random pick | correctness | DECIDED 2026-11-20: merge at construction | committed `242cabd0` on fix/joint-prob-tickets-1 |
| T5 | Silent / warned standard-normal prior fallbacks | API contract | DECIDED 2026-11-20: raise | committed `44f29924` on fix/joint-prob-tickets-1 (svgd, epoch path, and mcmc) |
| T6 | `epoch_starts=[0.0]` build cost (43 s vs 1.2 s single-epoch at 45,704 vertices) | performance | — | MERGED `36edaaa8` (2026-11-21): 47.7 s → 7.5 s (default prior), 22.8 s → 4.9 s (explicit prior); outputs bit-identical |
| T7 | `joint_stop_prob` graphs reach the joint-index leaf in `svgd` with the wrong terminal set | correctness | — | committed `ea467324` on fix/joint-prob-tickets-1 |
| T8 | No check that a joint table has >= 2 outcomes; zero outcomes gives an `IndexError` | robustness | — | committed `1cb3b9fb` on fix/joint-prob-tickets-1 |
| T9 | Multi-property indexers: one independent mutation process per counted property | semantics | DECIDED 2026-11-20: require an explicit reward_only when the indexer has several properties (option 1) | committed `33eb75ae` on fix/joint-prob-callback |
| T11 | FFI caches retain memory: per-thread builder_cache keeps every distinct model's builder for the process lifetime; worker caches keep dead builders' Graphs until address reuse | memory | LRU cap + explicit clear (user, 2026-11-20) | MERGED `7b0b7a7c` (2026-11-21): `configure(model_memory_cache_size=K)`, `clear_model_memory()`, `model_memory_info()` |
| T12 | `Graph.mcmc` builds its default DataPrior without `rewards=` (svgd passes it), so with rewards the prior is fitted to the unrewarded model | correctness | — | committed `7fc6caee` on fix/joint-prob-callback |
| T13 | Default data-informed prior (`probability_matching`) fits joint-probability data badly: on coalescent n=4/6 its estimate has log-likelihood 1,800-4,000 below truth; mu stays at its grid initial guess | correctness (prior quality) | — | found 2026-11-20, not investigated |
| T10 | `static py::object np_number` (`phasic_pybind.cpp:682`) DECREFs after interpreter shutdown without the GIL | nit | — | committed `607fdbfc` on fix/ffi-stale-graph-cache (with T1) |
| — | Exit segfault | crash | — | DONE `038dcc63` |
| — | `tot_reward_limit` trash edge on t-vertices | correctness | — | DONE `9cedb4f3` |

Not tickets, recorded: CI's pytest job runs only `-m equivalence`
(`.github/workflows/native-tests.yml:79`), so none of the new tests run
in CI; the thread-exit regression test is a reliable guard on macOS only
(see its docstring).

## 3. Tickets

### T1 — `per_thread_graph_cache` stale address key

Evidence (read from source): `graph_builder_ffi.cpp:133`
`thread_local unordered_map<const GraphBuilder*, unique_ptr<Graph>>`,
filled at lines 1071, 1141, 1743, 2033, 2038 inside OpenMP regions,
never erased. Builders are owned by the CALLING thread's `builder_cache`
and freed when that thread exits; OpenMP workers are long-lived. A new
builder allocated at a freed builder's address hits the old entry and
reuses a Graph of the wrong structure. The same hazard exists across
threads for the persistent-graph map in `graph_builder.cpp` (an entry
on thread B for a builder destroyed on thread A; recorded in the
comment at `graph_builder.cpp:380-389`).

Approach: give every GraphBuilder a process-unique 64-bit id (atomic
counter at construction); store it next to each cached Graph and treat
an entry whose id differs from the builder's as a miss (replace it).
Keys stay pointers, so memory behaviour is unchanged.

Done when: a reproducer that forces address reuse returns wrong values
(or a structure error) on master and correct values on the fix; gate
subset + thread-exit test pass.

### T2 — `reward_rates_callback`

Evidence: `joint_prob_graph` (`__init__.py` ~10271) forwards to
`self._joint_prob_graph`, which exists only as commented-out code →
`AttributeError` since 2026-05-21 (recorded in
`constant-parameterized-consistency-plan.md` §13). The old reference
called `callback(current_state, base_graph_indexer, reward_indexer,
current_rewards, mutation_rate=, reward_limit=, tot_reward_limit=) ->
(rates, trash_rate)`.

History (git, 2026-11-20): before commit `89661051` (2026-05-21,
"Reimplemented joint_prob_graph method") `joint_prob_graph` itself
called the callback (default `self._joint_prob_reward`). That commit
added the fast builder, renamed the old body `_joint_prob_graph`,
labelled it "DEPRECATED reference ... kept for A/B comparison", and
commented it out — while the new code forwards custom callbacks to it.
So the route never worked in its current form; no live
`_joint_prob_graph` ever existed on any branch (`git log --all -G`).
Adversarial review 2026-11-20: no recorded reason for disabling it — an
oversight (the same commit's docstring says behaviour is "unchanged for
custom callbacks"). Correction: `weight_formula` did not cause it, but
commit `4315450d` (2026-06-08) swept `base_coeff_length` into the
commented reference, which is now internally inconsistent (would raise
NameError; truncates coefficients via `edge_state(param_length)`; lacks
the t-vertex rule) — do NOT revive the old body.

Requirements from the review: route the callback through the fast
builder's own coefficient handling (`base_coeff_length`, mutation slot
last); never call it at t-vertices (builder rule from `86e1f8ac`) or the
start vertex; validate its output like the `discretize` rate callback
(Part 4 contract: rates shape `(reward_length,)`, finite, >= 0; trash
rate finite scalar >= 0; TypeError / ValueError naming the state);
document that on epoch graphs `current_state` includes the epoch slot
and that callback rates are constants scaled by the mutation slot
`theta[-1]`; rewrite the docstring (it names the non-existent
`_joint_prob_graph`) and the mathref table; if the overflow plan lands,
set its cap-rule flag on callback-built graphs. Decided 2026-11-20: the
BUILDER enforces `reward_limit`/`tot_reward_limit` on the callback's
rates (any increment past a cap goes to trash, as for the default); the
callback only chooses which counter increments at what rate, plus any
extra trash rate. Termination is guaranteed and the caps mean the same
for every graph. No caller in
the repo; mentioned in `docs/api/Graph.qmd` and the symbol table of
`docs/mathref/22_probability_matching.qmd`.

Options: (a) implement in the fast builder: call the callback where
`fast_reward` is called, same signature, with the t-vertex rule from
`9cedb4f3` applied by the builder (not the callback); (b) remove the
kwarg (clear error). Recommendation (a): it is a documented parameter
and the old contract is known.

Done when (a): a callback reproducing the default rates gives a
bit-identical graph to the default path, and a non-default callback
matches a closed form.

### T3 — continuous `joint_prob_table()`

Evidence: `_get_joint_probs` (`__init__.py` ~11818) returns
`expected_sojourn_time` at each t-vertex. On a continuous graph that is
probability / (rate of the t-vertex's edges into absorbing vertices),
i.e. probability / mu. Approach: on a continuous graph multiply each
sojourn by that rate r_v (the epoch path's final read already does
this). Users of the table on continuous graphs (notebooks, data
simulation from the table) change numbers — to the correct ones.

Done when: continuous and discrete tables agree with the closed forms
(n=2, n=3, all cap combinations).

### T4 — degenerate outcomes

Evidence (measured 2026-11-20, scratchpad `fu/degen.py`): an indexed
two-island coalescent (n=2, two absorbing base states: ancestor in
either population), `reward_limit=2`, no `reward_only`: 54 t-vertices,
27 distinct outcomes, every outcome degenerate. With
`reward_only=['descendants']` the existing merge removes them.
`_map_joint_observations_to_indices` maps a degenerate observation to
ONE vertex, drawn with probability proportional to the table at the
graph's current weights; the likelihood then uses that vertex's
probability instead of the group sum.

Options: (a) merge t-vertices with identical reward vectors into one
outcome vertex inside `joint_prob_graph`, generalising the existing
`reward_only` merge (every consumer becomes correct; graph structure
changes for multi-absorbing models); (b) carry groups through every
likelihood path and sum. Recommendation (a).

Done when: on the two-island model every outcome is unique and
probabilities equal the group sums of the unmerged graph (discrete and
continuous, single-epoch and epoch path).

### T5 — prior fallbacks

Evidence: epoch path `__init__.py:5471-5480`: a failing
`probability_matching` silently returns no prior (SVGD then uses a
standard normal on the unconstrained scale) — no log line. `svgd`
`__init__.py:6503`: same fallback with an `SvgdAssumptionWarning`.
Options: raise with a message naming the failure and telling the user
to pass `prior=` explicitly (the repo's no-silent-fallback rule), or
warn on both paths. Raising changes behaviour for fits that currently
proceed (one existing test, `test_weight_formula_kwarg.py:141`, hits
the warned fallback).

### T6 — `epoch_starts=[0.0]` build cost

Profile the build at ~45k vertices (memory-capped), report where the
time goes, propose a fix if one is local. Measurement-only until then.

### T7 — `joint_stop_prob` graphs through `svgd`

Evidence: `svgd_config.py:337` classifies `_joint_stop_prob_graph` as
kind `'joint_stop_prob'`, routed to the joint-index leaf
(`svgd_config.py:1206-1208`). In a JSP graph t-vertices end in a trap
loop, not an edge to absorbing, so the joint-index terminal set is not
the outcome set. Approach: reproduce, then reject with a rule message
pointing to the source joint-prob graph.

### T8 — outcome-count check

A table with one outcome carries no information under normalisation
(likelihood constant); zero outcomes currently fail with an
`IndexError` in `_get_joint_probs`. Approach: raise `ValueError` with a
clear message where the joint-index / epoch likelihood is built, and in
`_get_joint_probs` for zero outcomes.

### T9 — multi-property reward counting

Evidence (adversarial review, read from `fast_reward`): with an
indexer of several properties (e.g. descendants and population), each
base-state index maps to one reward counter PER PROPERTY, so one
mutation increments several counters and the total cap counts it
several times. Investigate on the two-island model, then report the
semantics and options to the user before any change.

### T10 — `np_number` static

Leak the object (`new py::object`) or use
`py::gil_safe_call_once_and_store`. Low priority.

## 4. Order

T7, T8, T3 (small, decision-free) → T1 (C++) → T4, T2, T5 (after
decisions) → T9 (investigation) → T6 (profiling) → T10.

## 5. Gate per ticket

1. Reproducer against an independent reference fails on master.
2. Fix; reproducer passes; new pytest file for the ticket.
3. The 35-file gate subset (§1) on the branch: no new failures vs the
   last recorded result; process exits cleanly.
4. Independent adversarial review of the diff (code given, not my
   conclusion); act on its findings.
5. Commit on the branch; update §2 and §6; ask the user before merging.

## 6. Log

- 2026-11-20: plan written; T4 degenerate outcomes confirmed on an
  indexed two-island model (27 of 27 outcomes degenerate).
- 2026-11-20: decisions — T2 implement (after an adversarial review of
  why it was disabled), T4 merge at construction, T5 raise.
- 2026-11-20: thread-exit fix gate (branch fix/tl-destructor-order,
  same code as master): 591 passed, 6 skipped, 5 xfailed, 0 failed, and
  the 56-minute 35-file process exited cleanly (exit 0; every unfixed
  run of this size had crashed at exit).
- 2026-11-20: T7 committed (`ea467324`, branch fix/joint-prob-tickets-1):
  `joint_prob_table()` refuses stop-probability and sojourn graphs (it
  listed 6 wrong outcomes with NaN for a 9-outcome model; svgd failed
  with KeyError / NaN). Small Python tickets T7, T8, T3 are stacked on
  this branch, one commit each, with one gate for the branch.
- 2026-11-20: T2 adversarial review: oversight, no recorded reason;
  requirements and the open cap-enforcement choice recorded under T2.
- 2026-11-20: T8 committed (`1cb3b9fb`): fewer than two outcomes refused
  in `pmf_from_graph_joint_index` and the shared observation mapping;
  zero outcomes give a clear error. `Graph.probability_matching` now uses
  the shared mapping (its private copy duplicated the random tie-break
  that T4 removes). 7 tests, 5 fail on master.
- 2026-11-20: T3 committed (`ad76a044`): continuous table multiplied by
  r_v. 23 tests against closed forms; the 13 continuous ones fail on
  master. Branch gate (all joint-prob test files) and an adversarial
  review of the three commits started.
- 2026-11-20: review of T7/T8/T3: BLOCKER in T8 — the single-terminal
  guard in `pmf_from_graph_joint_index` applied to every graph, but the
  API is generic (a non-terminal vertex read against one terminal
  depends on theta) and 17 existing tests in 5 files failed. Fixed in
  `9f9e4707` (guard only on joint-prob graphs; regression test against
  the closed form (t0+2t1)/(2t0+3t1)); those 5 files pass (45 passed,
  1 skipped). T3 and T7 judged correct. Should-fix S1 (the T7/T8 error
  reaches svgd users only after a misleading "DataPrior construction
  failed ... falling back" warning) is resolved by T5 (prior failures
  raise). Gate restarted on the widened list.
- 2026-11-20: T1 reproduced and fixed (`5877fb9f`, branch
  fix/ffi-stale-graph-cache in ../phasic-tlfix): thread tracing showed a
  builder address reused and OpenMP workers serving the old model's
  graph ("Invalid vertex index 32 (graph has 25 vertices)", NaN, no
  exception). Pre-compiled reproducer: 6-8 wrong results per 2000
  short-lived threads on master, 0 of 6000 with the fix; pytest
  `test_ffi_stale_graph_cache.py` (3000 threads, ~7 s) fails on master
  (15) and passes on the fix. Fix: builder instance ids stored in both
  per-thread caches.
- 2026-11-20: T1 review: correct and complete (13 wrong per 3000 on
  master, 0 with the fix; all five sites converted; no other
  address-keyed cache). Test now pins OMP_NUM_THREADS=4 (with one thread
  the hazard cannot occur). New ticket T11 (stale Graphs retained in
  worker caches until address reuse).
- 2026-11-20: T4 design changed from the plan's "generalise the
  reward_only merge": that merge sends t-vertices through an extra
  vertex on a mutation-rate edge, which delays mass at finite times
  (EpochContext). Instead `joint_prob_graph` substitutes ONE canonical
  absorbing base state (first by vertex index) whenever it creates a
  joint vertex, so equal reward vectors meet in one t-vertex; a no-op
  with a single absorbing state. De-risked on the two-island model: 54
  -> 27 t-vertices; discrete and continuous tables, single-epoch and
  epoch likelihoods equal master's group sums to <= 1.1e-15;
  EpochContext mass at t=50 equals the table mass. The random
  tie-break in the observation mapping is replaced by a refusal (now
  unreachable for joint_prob_graph output); `seed` kept, unused.
- 2026-11-20: T4 committed (`242cabd0`). T5 committed (`44f29924`):
  svgd, the epoch path and Graph.mcmc raise a ValueError chaining the
  cause. The mcmc site was not on the original list; it is the same
  silent fallback and follows the same rule for svgd/mcmc consistency.
  The first branch gate was stopped (it predated T4/T5); a single gate
  of the branch head (T7, T8, T3, T4, T5) on the widened list, plus an
  adversarial review of T4 and T5, are running.
- 2026-11-20: T2 committed (`9ddb0d96`) on its own branch
  fix/joint-prob-callback, stacked on fix/joint-prob-tickets-1 so that
  branch stays exactly what its gate tests. Callback runs in the fast
  builder with validation and builder-enforced caps; commented-out
  reference deleted; math reference updated (docs/api/Graph.qmd
  regenerates with `pixi run api`). 16 tests, including byte-identical
  serialization against the default build and an add_epoch graph.
- 2026-11-20: T9 investigated. With several rewarded properties
  (e.g. descendants and pop), `base_idx_to_rewards` maps each base index
  to one counter PER PROPERTY and the builder adds the full mutation
  rate to each as a SEPARATE transition: one independent mutation
  process per property, each incrementing only its own counter. Each
  property block's marginal is right, but the blocks are not counts of
  the same mutations: on the n=2 two-island model every mutation is on a
  singleton, so joint counting would force descendants_1 == pop_0 + pop_1,
  yet outcome (1,0,0,0) has probability 0.083. tot_reward_limit caps the
  sum over all blocks. Options put to the user.
- 2026-11-20: branch gate of fix/joint-prob-tickets-1 head (T7, T8, T3,
  T4, T5) on 49 files: 738 passed, 10 failed. 8 failures = T5 broke the
  fixed/tied plumbing tests in test_daisy_chain_c_path.py (no
  observations, relied on the silent prior fallback); 1 =
  svgd(theta_init=..., no theta_dim) on a callback graph, whose default
  prior was built with theta_dim=None (a real bug the fallback hid);
  1 = inference/_test_mcmc.py (disabled file, picked up by the widened
  grep; excluded from now on). Review of T4/T5 found the same 8 plus:
  data errors reported as prior failures (svgd built the prior before
  mapping observations), and an untested parallel-edge case in T4.
  All fixed in `21853f77` (fix/joint-prob-callback): observations mapped
  before the prior, DataPrior gets theta_init's dimension, explicit
  prior in the 8 plumbing tests, closed-form test for a parent with two
  absorbing children. Remaining review notes (not changed): Graph.mcmc
  builds DataPrior without rewards= (pre-existing; ticket T12); the
  epoch prior's probability_matching ignores fixed/exposure (prior
  centring only).
- 2026-11-20: T9 decision history: "single property" was chosen first,
  then found to break the two-locus models (loc1/loc2, where independent
  per-property processes are CORRECT); options re-presented; final
  choice option 1: several properties require an explicit reward_only
  (several allowed). Committed `33eb75ae`. Notebooks that build joint
  graphs from multi-property indexers without reward_only and should be
  checked by the user (from a source scan; per-call indexers not
  resolved): svgd-joint-prob (5 of 6 calls), model-selection (3), smc
  (1), method-of-moments (1), coalescent-derived (3),
  old-moran-coalescent-derived (3), time-inhomogeneous (4; may use a
  single-property indexer), _maybe_trash_method-of-moments (1). Already
  using reward_only: svgd-multi-param, ghost, two-locus-island (4 of 5).
- 2026-11-20: branches now: fix/joint-prob-tickets-1 (T7, T8, T3, T4,
  T5) is an ancestor of fix/joint-prob-callback (+ T2, review fixes,
  T9); only the latter will be merged. Gate of its head running.
- 2026-11-20: T11 analysed, not built. Stale worker entries need a
  calling thread to EXIT (that is what frees a builder); in normal use
  the FFI runs on the caller's (usually main) thread, which lives for the
  process, so they arise only when models are evaluated from many
  short-lived threads. The larger retention is that builder_cache
  (keyed by structure JSON, never shrunk) keeps every distinct model's
  builder - and each OpenMP worker keeps its Graph - for the process
  lifetime: at 200k-600k vertices, many distinct models in one session
  (e.g. model selection over many graphs) accumulate large graphs. A fix
  needs an eviction policy (LRU size cap, explicit clear API, or
  freeing with the Python model object) - a design choice for the user.
  A narrow fix for the stale worker entries alone (process-wide set of
  live builder ids; prune a thread's dead entries on cache miss) was
  designed but not built.
- 2026-11-20: T12 committed (`7fc6caee`, mcmc default prior gets
  rewards=; spy test). T10 committed (`607fdbfc`, leaked cached
  numpy.number) on the C++ branch, to be gated with T1.
- 2026-11-20: branch gate of fix/joint-prob-callback at `33eb75ae`
  (T7, T8, T3, T4, T5, T2, T9 and the review fixes): 50 files that
  reach joint graphs, the observation mapping, DataPrior or mcmc:
  742 passed, 0 failed, 7 skipped, 5 xfailed (57 min). The 10 earlier
  failures (daisy-chain prior fixtures, theta_init-only prior) are gone.
  No suite test needed a reward_only for T9. T12 (`7fc6caee`) rerun
  separately on the mcmc/DataPrior files; independent review of T2 and
  T9 started.
- 2026-11-20: T12 rerun at `7fc6caee` (reinstalled): test_prior_fallback_raises,
  test_mcmc, inference/test_svgd_api_parity, test_svgd_assumptions,
  test_method_of_moments, test_constant_parameterized_consistency:
  146 passed, 0 failed. C++ branch (T1 + T10) gate started: the 35-file
  FFI list of the earlier T1 gate (591 passed) plus the T1 test.
- 2026-11-20: independent review of T2 and T9 found no defect in those
  commits; it found four next to them, fixed in `cf15da65`: the
  reward_only merge paired names and value ranges in different orders
  (crash for a subset in non-alphabetical property order; now reached
  by every multi-property user because of T9); unknown reward_only
  names were dropped silently; a dict reward_limit on the default path
  sent every mutation to trash (now refused on every path); the
  callback got the builder's live arrays (now copies; non-numeric
  output refused). 7 joint-graph test files: 112 passed. Review also
  notes that notebooks counting an attribute property (model-selection
  in_pop, coalescent-derived is_derived) were counting extra
  independent mutation blocks before T9, so their earlier outputs were
  probably wrong, not only about to break (inferred, not run).
- 2026-11-20: C++ branch fix/ffi-stale-graph-cache (T1 + T10, head
  `607fdbfc`) gate: 35 FFI-reaching files plus the T1 test: 592 passed,
  0 failed, 6 skipped, 5 xfailed (56 min). Both branches ready to merge.
- 2026-11-20: T6 profiled (coalescent n=8, reward_limit 3, 45,704
  vertices, 2,000 observations). `epoch_model(obs, [0.0])` took 47.7 s:
  ~25 s building the default data-informed prior and ~23 s building the
  model. Causes, found with a native stack sampler (macOS `sample`;
  Python's own sampler blamed the wrong line): (1) `joint_stop_prob_graph`
  tested `v.index() in t_vertex_old_indices`, a list of ~16k numpy ints,
  once per vertex - O(vertices x t-vertices), 18.9 s of the model build;
  (2) `probability_matching` passed the graph as a dict, which
  `compute_sojourn_times_ffi` re-encodes to 6.3 MB of JSON on each of
  158 calls (~20 s). Same list-membership pattern fixed in
  `joint_prob_graph`'s reward_only merge loop. After: 7.5 s (default
  prior), 4.9 s (explicit prior), `joint_stop_prob_graph` 1.0 s; the
  rest is spread out (derived-graph copies ~1 s each, observation
  mapping, JAX compile) with no single hot spot. Identity check against
  master: derived-graph serializations (n=4, 6), probability_matching
  estimates and reward_only merged graphs bit-identical.
- 2026-11-20: NEW ticket T13 (found during the T6 identity check, also
  on master): the default data-informed prior (`probability_matching`)
  fits joint-probability data badly. Coalescent n=4 / n=6, reward_limit
  3, 5,000 observations from the continuous table at (c, mu) = (5, 3):
  estimates (0.34, 0.85) / (0.26, 0.85). Only mu/c is identified here,
  but the fit is not on the right ratio: log-likelihood at truth
  -13,961 / -19,667, at the prior's fit -15,795 / -23,711, and at the
  fit's mu with the true ratio -13,961 / -19,667 (equal to truth, as
  expected). mu never moves from its grid initial guess (0.853). Svgd
  and the epoch path centre their default prior here (T5 made failures
  raise, but a bad fit is silent). Not investigated further.
- 2026-11-20: T6 gate: 42 files touching joint_stop_prob / joint_sojourn
  graphs, epoch_model, probability_matching, DataPrior, reward_only,
  daisy-chain and exposure: 629 passed, 0 failed, 7 skipped, 5 xfailed.
- 2026-11-20: T11 measured on master: RSS retained after each distinct
  model's Python objects are deleted and jax.clear_caches() runs:
  ~68 MB per 11,320-vertex model (6 models), ~290 MB per 45,704-vertex
  model (4 models), linear (~6 KB/vertex; ~1.3-3.8 GB per model at
  200k-600k). User decision: LRU cap + explicit clear.
  Design (branch fix/ffi-cache-eviction):
  1. Live-builder registry: the builder's id moves into a member token
     that registers on construction (a copy gets a fresh id) and
     unregisters on destruction; a process-wide generation counter is
     bumped whenever builders are evicted or cleared.
  2. One lookup helper replaces the 11 find-or-create sites of
     builder_cache: entries carry a last-used stamp; on insert beyond
     the capacity (env PHASIC_MODEL_MEMORY_CACHE_SIZE, default 4, set
     via phasic.configure(model_memory_cache_size=K)) the least recently
     used entry is dropped together with its daisy-chain metadata.
  3. Each thread sweeps its per-thread graph caches (FFI worker slots
     and GraphBuilder persistent graphs) of entries whose builder is no
     longer live, when the generation counter has changed - checked on
     every access, so idle workers release on their next use.
  4. phasic.clear_model_memory(): clears the calling thread's caches now
     and bumps a clear counter so every other thread drops its
     builder_cache and graphs on its next FFI call;
     phasic.model_memory_info() reports capacity and counts.
  Gate: correctness (cycling more models than the capacity gives the
  same values as fresh), memory bounded in a subprocess, the T1 stale
  cache test, and the FFI-reaching test files.
- 2026-11-20: T11 independent review (commit 987c8d77). Thread-exit and
  shutdown ordering judged sound (leaked registries, trivially
  destructible atomics, no sweep from a destructor). Fixed:
  (1) use-after-free: DaisyChainSojournFfiImpl holds the JSP graph
  pointer while fetching the sojourn graph, and a concurrent
  clear_model_memory() made that second lookup clear ALL slots, freeing
  the held graph. Sweeps now only drop graphs of destroyed builders,
  never a live builder's (a clear destroys the cached builders, so their
  graphs still go). (2) clear left most memory in idle OpenMP workers
  until their next batched call (probe: 12 graphs -> 9 left); clear now
  runs an OpenMP region in which every pool thread drops its builders,
  then (after a barrier) sweeps. (3) Python and the native reader parsed
  PHASIC_MODEL_MEMORY_CACHE_SIZE differently ('1_0', '5 '); same rule
  now, and the native error no longer reads as a JSON problem.
  (4) builder assignment deleted (keeping either id could serve a graph
  of the wrong structure; nothing assigns builders today). numpy ints
  accepted for model_memory_cache_size. Not changed: capacity counts
  structure JSONs, so alternating between two epoch models needs 4 (the
  default) and three need 6 (documented); a rejected configure() value
  stays in the config object (pre-existing pattern for every field).
- 2026-11-21: T11 gate at `0e834208` (FFI-reaching files + T1 test +
  test_model_memory): 602 passed, 0 failed, 6 skipped, 5 xfailed. (At
  987c8d77, before the review fixes: 598 passed, 0 failed.)
- 2026-11-21: T6 and T11 merged into master (`36edaaa8`, `7b0b7a7c`, --no-ff, not pushed). Open: T13 (default prior fit quality).
