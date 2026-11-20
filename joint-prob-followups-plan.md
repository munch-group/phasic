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

Gate subset used throughout (35 files): every test file that builds a
joint-probability graph (`grep -rl "joint_prob_graph\|tot_reward_limit"
tests/pytest`) plus the new test files. Tot-cap fix: 618 passed, 6
skipped, 5 xfailed, 0 failed.

## 2. Ticket status

| # | Ticket | Kind | Decision | Status |
|---|---|---|---|---|
| T1 | `per_thread_graph_cache` stale address key (C++) | correctness | — | open |
| T2 | `joint_prob_graph(reward_rates_callback=...)` calls a commented-out method | broken API | DECIDED 2026-11-20: implement in the fast builder, after an adversarial review of why it was disabled | open |
| T3 | `joint_prob_table()` on a continuous graph returns probability / mu | correctness | — | open |
| T4 | Degenerate outcomes (several t-vertices, same reward vector) resolved by a random pick | correctness | DECIDED 2026-11-20: merge at construction | open |
| T5 | Silent / warned standard-normal prior fallbacks | API contract | DECIDED 2026-11-20: raise | open |
| T6 | `epoch_starts=[0.0]` build cost (43 s vs 1.2 s single-epoch at 45,704 vertices) | performance | — | open |
| T7 | `joint_stop_prob` graphs reach the joint-index leaf in `svgd` with the wrong terminal set | correctness | — | open |
| T8 | No check that a joint table has >= 2 outcomes; zero outcomes gives an `IndexError` | robustness | — | open |
| T9 | Multi-property indexers: one mutation increments one counter per property, so `tot_reward_limit` counts it several times | semantics | OPEN: investigate first | open |
| T10 | `static py::object np_number` (`phasic_pybind.cpp:682`) DECREFs after interpreter shutdown without the GIL | nit | — | open, low priority |
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
So the route never worked in its current form. `weight_formula`
(`2d7c1d81`, 2026-06-08) came later and is unrelated. No caller in
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
