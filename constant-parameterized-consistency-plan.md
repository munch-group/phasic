# Plan — consistent handling of constant vs parameterized models

Status: v3, 2026-10-11. Both adversarial reviewers returned
SOUND-WITH-CORRECTIONS: reviewer A (correctness) and reviewer B (blast
radius, re-run read-only after the crash). Their corrections are folded in
(record in §12). The decisions in §9 are made. No code changes are
authorized until Kasper signs off the plan as a whole. Everything here
modifies existing code, including C/C++ (Part 3a, and the C++ `add_epoch`
port in §6.10).

This plan extends `graph-cache-constant-roundtrip-plan.md` (v2, reviewed;
the fix for constant graphs coming back parameterized from the graph
cache). That plan becomes Part 1 here, unchanged except where §3 says so.

## 1. Goal (Kasper, 2026-10-11)

After the change, the user-facing logic is consistent across the code
base in three respects:

1. **Constant models raise an error when treated as parameterized.** A
   constant model is one built with scalar rates (`add_edge(v, 3.0)`, or a
   callback that returns floats), or a derived graph with baked-in weights
   (e.g. `reward_transform()` output). Any entry point that needs θ must
   refuse it, with the same clear message everywhere.
2. **Widening the coefficient layout is automatic and transparent.**
   Operations that add parameter slots (`discretize()` on a parameterized
   graph, `add_epoch()`, `joint_prob_graph()`) do it without the user
   managing coefficient vectors. Nothing else about the model changes:
   no edge, vertex, coefficient or metadata is lost. Where widening has
   no defined meaning, the operation refuses loudly.
3. **A `discretize()` callback returns one number.** The user does not need
   to know the coefficient layout. On a parameterized graph, the returned
   value goes into the widened slot, exactly as a fixed rate does today.

## 2. What happens today (measured on the installed master build)

The author's probes (`probe_ui.py`, `probe_widen.py`, `probe_modes.py`)
and reviewer A's (`review2_A/`) were in the session scratchpad and were
lost when the machine crashed and rebooted; the results below are as
recorded before the crash. The graph cache was cleared before every build.
Without that, every graph after the first is a cache hit and comes back
parameterized, which is the Part 1 bug. The installed package matched
`src/` (reviewer A, `diff -rq`).

### 2a. Constant models passed to θ-taking entry points

| Entry point | Constant graph today | Verdict |
|---|---|---|
| `update_weights` | raises "Cannot call update_weights() on constant graph" | already consistent |
| `pmf_and_moments_from_graph` | **silently θ-scaled**: mean 0.75 at θ=2, true mean 1.5 | Part 1 makes it raise |
| `moments_from_graph` | not runnable on this install (JIT needs the C/C++ source tree); in source it builds a fake 1-parameter model | Part 1 makes it raise |
| `pmf_from_graph` | by design returns a θ-free function `f(times)` (`__init__.py:3833-3842`); not runnable on this install (same JIT issue) | consistent: no θ involved |
| `svgd` | **runs** and fits a θ multiplier | needs the guard |
| `mcmc` | **runs** | needs the guard |
| `method_of_moments` | **runs** | needs the guard |
| `probability_matching` | raises "requires a joint probability graph" (`probability_matching.py:197` would default θ to `param_length()`) | needs the guard, for the message |
| `reward_visit_probability(theta=...)` | correct, because a common scaling leaves hitting probabilities unchanged | after Part 1: XLA "theta length mismatch"; needs the guard |
| `joint_prob_graph`, `joint_stop_prob_graph`, `joint_sojourn_graph` | guard `param_length() == 0` (`__init__.py:9985`, `:10680`, `:10864`) never fires, because a constant graph has `param_length() == 1`. Measured (reviewer A): `joint_prob_graph` of a constant base gives P=2, parameterized=True, i.e. the constant model silently becomes a 2-parameter joint model | guard must use `parameterized()` |
| `add_epoch` | cryptic C error "Edge has too few coefficients: graph expects at least 4, got 1" | needs the guard |
| `epoch_model`, `daisy_chain_joint_probs`, `pmf_and_moments_from_graph_multivariate`, `pmf_from_graph_joint_index`, `profile(theta)` | not probed; source shows they default θ to `param_length()` or serialize with it | needs the guard (to be verified per site) |
| `discretize`, `reward_transform`, `laplace_transform` | stay constant (measured) | consistent: no θ involved |

The predicate `parameterized()` was checked by reviewer A against every
edge case found, with no wrongful rejection: `set_param_length(1)` with
IPV edges only → False (C sets `parameterized = (P>1)`, `phasic.c:5073`);
the same plus one `[1.0]` edge → True; joint graphs, discretized
parameterized graphs, `laplace_transform`, `clone` and `from_serialized`
of parameterized graphs → True; `reward_transform` of a parameterized
graph → False, correctly, because θ is baked in (E = 1.833 at θ=1, and its
own `update_weights` says "empty graph").

### 2b. The widening helper `_rebuild_with_wider_layout` (`__init__.py:9504`)

Used by `discretize()` (fixed rate on a parameterized graph) and
`add_epoch()`. It has four silent defects, all measured, and all
reproduced independently by reviewer A:

1. **Drops every coefficient-less edge.** It copies
   `vertex.parameterized_edges()` (`__init__.py:9555-9565`), which skips
   edges with no coefficients (`phasiccpp.cpp:1764`). These are the
   aux→vertex back-edges from `discretize()`, `add_edge_constant` edges,
   and the trapping loops from `joint_stop_prob_graph`. Measured: a graph
   with one `add_edge_constant` edge loses it on `discretize(0.1)`.
2. **Merges vertices that share a state.** It uses `find_or_create_vertex`
   (`:9551`), so the all-zero-state aux vertices from an earlier
   `discretize()` collapse into one. That one vertex has no out-edges, so
   it becomes absorbing. Measured: `discretize(0.1)` followed by
   `discretize(0.1, skip_existing=True)` goes from 10 vertices to 11
   instead of staying at 10, and gains a parameter slot it should not
   have. Widening a `joint_prob_graph` output crashes today with "Caught an
   unknown exception!": its four zero-state duplicates merge into a
   self-loop, and `add_edge` does `throw new` (`phasiccpp.cpp:1467`)
   (reviewer A, measured).
3. **Truncates coefficient vectors longer than `param_length()`.** It reads
   `edge_state(self.param_length())` (`:9561`), so extra auxiliary data is
   lost. Measured: `[1.0, 7.0]` becomes `[1.0, 0.0]`. Such vectors occur
   in linear mode too: the fixture at `test_weight_formula_daisy.py:195`
   is linear with `theta_dim=1` and edges `[pair, 1.0]`.
4. **Drops metadata.** It copies the callback, the weight state and
   `is_discrete` only. Measured: `add_epoch` followed by `discretize` loses
   the epoch metadata (`_n_epochs`, `_epoch_state_index`,
   `_base_param_length`) and the epoch indexer. `was_dph`, `rewards`,
   per-vertex `is_aux`, the C `synthetic` marker and `_base_callback` /
   `_base_callback_kwargs` (set at `__init__.py:3105-3106`) are also lost
   (source).

`joint_prob_graph` shares defect 1: it copies only `parameterized_edges()`
(`__init__.py:10187`, `:10216`). It also does not propagate the weight
state, so a joint graph built from a formula base comes back linear (no
`_propagate_weight_state` between `:10150` and `:10330`, source;
reviewer A).

### 2c. Widening under non-linear weight modes (measured)

- **Formula mode, `discretize`:** `discretize(0.5)` produces discretization
  edges of weight 0 (at θ=[2,1] and θ=[2,7]). The formula (`t0*c0`) never
  reads the new slot, so the rate is silently ignored.
- **Formula mode, `add_epoch`:** silently wrong. The formula `c0*t0` never
  reads the epoch slots; its pdf differs from the linear twin by up to
  0.376. Setting a formula that covers all four slots afterwards gives an
  exact match (reviewer A).
- **Log mode:** `discretize(0.5)` succeeds, then every
  `update_weights(..., log=True)` fails with "requires all (coefficient *
  parameter) products to be positive". A plain `update_weights([2,1])`
  without `log=True` succeeds. In log mode each weight is a product over
  all slots, so padding with 0 zeroes every edge, and no neutral pad value
  exists.

### 2d. The `discretize()` callback

- On a constant graph, a callback returning a number works (measured).
- On a parameterized graph, the same callback raises "add_aux_vertex()
  requires array of coefficients" (measured). The callback must return a
  full coefficient vector of length `param_length()`, and that vector does
  NOT get the widened slot (`__init__.py:2807-2810`).
- Kasper's own `docs/pages/tutorial/discrete.ipynb` cell 17 does exactly
  this (`nr_lineages * mutation_rate`) and its saved output is this error.
- Two tests use the vector form: `tests/pytest/inference/test_exact_grad_log_weight_mode.py:272`
  and `tests/pytest/inference/test_exact_grad_joint_index.py:371`.
- **A per-state rate of 0 is legitimate and works today**, e.g.
  singleton-only mutation `rate = 0.3*s[0]`: E = 0.6000 against an oracle
  of 0.6000, on constant graphs and on parameterized graphs via the vector
  form (reviewer A). Only the fixed-scalar path rejects 0;
  `ptd_graph_add_edge` rejects only negative weights (`phasic.c` ~:4925).

### 2e. Edge-array capacity bug (pre-existing memory-safety defect)

Found by reviewer A; mechanism confirmed in source by the author.
- `ptd_directed_graph_add_edge` (`src/c/phasic.c:2938-2955`) grows a
  vertex's edge array only when the edge count is a power of two, so it
  assumes the capacity is always the next power of two.
- These all `realloc` to exactly `len+1`:
  - `Vertex::add_edge_constant` (`src/cpp/phasiccpp.cpp:1700-1730`);
  - `add_aux_vertex`, both overloads (`:1545`, `:1589`). They append the
    back-edge to a fresh aux vertex, which is safe on its own;
  - `add_aux_vertex_constant` (`:1633`), which appends to an existing
    vertex;
  - `ptd_clone_graph`, in its constant-edge branch AND in its
    starting-vertex loop, for every IPV edge (`src/c/phasic.c:1263-1278`;
    reviewer B, confirmed by the author);
  - `GraphBuilder::build`, for constant edges
    (`src/cpp/parameterized/graph_builder.cpp:326-338`; reviewer B);
  - `add_edge_raw` in `src/c/scc_synthetic.c:396-406`, so every edge of a
    synthetic graph has exact capacity (reviewer B).
- After such an append, the next normal `add_edge` on that vertex can
  write past the end of the array. Reviewer A measured a segfault under a
  guard allocator with 5 coefficient edges, then a constant edge, then 2
  more coefficient edges. Two controls passed: no constant edge, and the
  constant edge added last.
- Part 3 would hit this in four places: copying constant edges in place;
  `add_epoch` adding edges afterwards; `discretize` adding aux edges
  afterwards; `joint_prob_graph` adding mutation and trash edges after the
  constant copies.
- **Exposed today** (reviewer B, inferred from source; not measured):
  - `clone()` / `copy()` of any graph with 3, 5, 6, 7, 9 or more
    starting-vertex edges, followed by `starting_vertex().add_edge(...)`;
  - `laplace_transform` (C clone at `phasic.c:14949`, then
    `ptd_graph_add_edge` at `:14981`) and `discretize()`'s clone branch
    (`__init__.py:2809-2812`, then `add_aux_vertex`), on any vertex whose
    last cloned edge is coefficient-less and whose edge count is not a
    power of two. That means user graphs that used `add_edge_constant` or
    `add_aux_vertex_constant` on a vertex with 2 or more edges.
- **Not exposed today** (reviewer B, inferred):
  - library-built graphs put constant edges only on aux vertices and JSP
    t-vertices, which hold one edge each;
  - `from_serialized` appends constant edges last
    (`_graph_serialize.py:640-653`);
  - `reward_transform` builds a fresh graph;
  - no edge-removal path reallocates the array (`phasic.c:6868` only zeroes
    the length).

## 3. Part 1 — cache round trip (from the v2 plan)

Unchanged: the one-line `serialize()` fix, stale-entry invalidation, the
call-site table, tests T1–T9, and gates. See
`graph-cache-constant-roundtrip-plan.md` §4–§8.

Two changes. First, stale entries are handled by deleting the cache
and bumping the callback-hash version (§9.1), not by a format marker.
Second, the v2 plan's open question on `reward_visit_probability`
(keep it working on constant graphs, or accept a loud error) is settled by
Goal 1: it raises. The error comes from the Part 2 guard with the standard
message, not from the XLA theta-length check. T9 asserts that message.

## 4. Part 2 — one guard for constant models

**Helper.** Add `_require_parameterized(graph, entry_point)` in
`src/phasic/__init__.py`, raising `ValueError`:

> `<entry_point>` needs a parameterized model, but this graph has no
> parameterized edges (it must have parameterized edges to take θ). This
> happens when rates are given as scalars, such as `add_edge(v, 3.0)` or a
> callback returning floats, and for derived graphs whose weights are
> baked in, such as `reward_transform()` output. Give rates as coefficient
> lists, such as `[3.0]`, to make them parameters.

The message keeps the substring "must have parameterized edges" of the
current messages (`__init__.py:7099`, `:7408`, `:8561`), for continuity.
No existing test matches on it (reviewer B).

**Test.** `graph.parameterized()`, never `param_length()`.

**Sites.** The guard is the first statement of each entry point, before
any expensive work:
- existing guards, switched to the helper: `moments_from_graph`,
  `pmf_and_moments_from_graph`, `pmf_from_graph_joint_index`;
- broken guards (`param_length() == 0`), replaced: `joint_prob_graph`,
  `joint_stop_prob_graph`, `joint_sojourn_graph`;
- new: `svgd`, `mcmc`, `method_of_moments`, `probability_matching`
  (`probability_matching.py` too, if it can be called without the Graph
  method), `epoch_model`, `daisy_chain_joint_probs`,
  `pmf_and_moments_from_graph_multivariate`, `add_epoch`,
  `reward_visit_probability` (when `theta` is given), `profile` (when
  `theta` is given);
- added from reviewer B:
  - the module-level `method_of_moments.method_of_moments`
    (`method_of_moments.py:205`);
  - `phasic.profile_graph` (exported at `__init__.py:374`;
    `profile.py:182`), when `theta` is given. Also, `_max_rate`
    (`profile.py:131-140`) calls `update_weights` whenever
    `param_length() > 0`, so `profile()` on a constant graph probably
    raises today even without θ (inferred). Switch that test to
    `graph.parameterized()`;
  - `bffg.path_exit_rates_by_param` and `bffg.bffg_log_prob` (exported;
    `bffg.py:121`, `:253`). On a constant graph they silently treat rates
    as θ0 coefficients (`bffg.py` ~155-172).

**Internal call chains** (reviewer B, read from source). No library path
passes a legitimately constant graph to a guarded site:
- svgd, mcmc and method_of_moments build their models from `self`;
- the daisy-chain and epoch internals call `joint_stop_prob_graph` /
  `joint_sojourn_graph` on joint graphs, whose interior edges are lists;
- synthetic SCC graphs are always marked parameterized.

`SVGD`, `MCMC` and `model_selection` take callables or SVGD objects, not
graphs, and need no guard. `laplace_transform(theta)` takes a Laplace
variable, not model θ. In `joint_stop_prob_graph` / `joint_sojourn_graph`
the guard can move ahead of the "produced by joint_prob_graph" check,
because the existing rejection tests use parameterized graphs
(`test_daisy_chain_c_path.py:176-182`, `test_epoch_sojourn_finalread.py:136-142`).

**Not guarded, deliberately:**
- `update_weights` already raises in C++.
- `pmf_from_graph` returns a θ-free function for constant graphs by design.
- `discretize`, `reward_transform` and `laplace_transform` take no θ and
  keep constant graphs constant.

## 5. Part 3a — fix the edge-array capacity bug (prerequisite for Part 3)

C/C++ change; needs sign-off as part of this plan.
- Route every exact-`realloc` edge appender through one helper that keeps
  the power-of-two capacity invariant of `ptd_directed_graph_add_edge`:
  `Vertex::add_edge_constant`, both `add_aux_vertex` overloads,
  `add_aux_vertex_constant`, `ptd_clone_graph` (constant-edge branch and
  starting-vertex loop), `GraphBuilder::build`'s constant edges, and
  `scc_synthetic.c`'s `add_edge_raw`. The helper is either
  `ptd_directed_graph_add_edge` itself or a small shared function with the
  same growth rule. The implementation greps for any other
  `realloc(... edges ...)` before closing this part.
- Add an additive pybind getter `_is_synthetic` next to `_set_synthetic`
  (`phasic_pybind.cpp:1389`), which Part 3 needs (§6.7).
- "Copy constant edges last" is not a fix, because edges are added after
  any widening anyway.
- **Regression test.** A pytest that builds the measured pattern (5
  coefficient edges, a constant edge, 2 more coefficient edges, on one
  vertex; repeated for each appender and for clone-then-add), plus "clone
  of a graph with 3 starting-vertex edges, then `start.add_edge`" and
  "`laplace_transform` of the measured pattern". Each checks edges,
  weights and pdf against a hand-built twin. Without a guard
  allocator, an overflow may not crash, so in addition the pattern is run
  ONCE under a guard allocator or AddressSanitizer build, before and after
  the fix, memory-capped and only with Kasper's go-ahead (guard allocators
  multiply memory use; see §10).

## 6. Part 3 — make the widening helper lossless

Rewrite `_rebuild_with_wider_layout` so that, apart from the added slots,
the new graph is the old one:

1. **Vertices.** Recreate every vertex in index order. The starting vertex
   maps to the new graph's starting vertex. For every other vertex, call
   `find_or_create_vertex(state)` only when `self.vertex_exists(state)` and
   `self.find_vertex(state)` returns that same vertex (same index);
   otherwise call `create_vertex`. (`find_vertex` throws when the state is
   absent, `phasiccpp.cpp:103-124`, hence `vertex_exists` first, pybind
   `:920`.) This follows the original's lookup tree, which differs between
   graph origins: a freshly built graph does not register its starting
   vertex; a clone does (`phasic.c:1373-1377`); and `from_serialized`
   registers the first occurrence of every state, including the zero
   state (`_graph_serialize.py:509-531`). One residual: a clone maps the
   zero state to the starting vertex, which a fresh graph cannot
   reproduce (minor, recorded). Both calls append, so assert that every
   new index equals the old one.
2. **Coefficient lengths.** `discretize()` and `add_epoch()` refuse, with a
   clear `ValueError`, any graph that has a non-starting edge whose
   `coefficients_length` is neither 0 nor `param_length()`. The new slot
   would otherwise land on auxiliary data (discretize puts the rate at
   `P_new-1`, and `add_epoch`'s interleaved layout at
   `__init__.py:2943-2950` assumes length B). The `base_coeff_length`
   logic stays only in `joint_prob_graph`.
3. **Coefficient edges.** Copy at `param_length()` and append the pad, as
   today. For ordinary linear graphs this is bit-identical to today
   (`edge_state` returns `min(len, P)`, `phasiccpp.h:2062`).
4. **Coefficient-less edges.** Copy with `add_edge_constant` (safe after
   Part 3a), unpadded, so they stay constant.
5. **Edge order.** Keep each vertex's original edge order. Constant edges
   are not moved to the end.
6. **Starting-vertex edges.** Copy as today. In callback graphs they carry
   coefficients (`[1.0]`, measured), and `update_weights` skips the
   starting vertex (`phasic.c:5654`).
7. **Metadata**, as an explicit list:
   - everything `clone()` carries (`__init__.py:9490-9502`, plus
     `_propagate_weight_state`);
   - `was_dph`; per-vertex `is_aux`; `_base_callback`,
     `_base_callback_kwargs`; `_indexer`, `_base_indexer`; `_n_epochs`,
     `_epoch_state_index`, `_base_param_length`;
   - `rewards`, with an assertion that `len(rewards) == vertices_length()`;
     this is correct because vertex order is preserved;
   - the C `synthetic` marker: widening refuses a synthetic graph, using
     the new `_is_synthetic` getter (Part 3a). Python cannot read the
     marker today (pybind has only `_set_synthetic`, reviewer B).
     Dropping it would bypass the synthetic-SCC Jacobian guard documented
     in CLAUDE.md.
   Callers that set these afterwards (`add_epoch`, `discretize`) keep
   overriding them. The only reader of the epoch metadata is `add_epoch`
   itself (grep, reviewer A).
8. **`joint_prob_graph`** also copies coefficient-less edges (via
   `add_edge_constant`), instead of dropping them. The copy mirrors the
   full per-edge handling of the existing loop (self-loop skip
   `__init__.py:10221`, child creation, t-vertex detection `:10245`), not
   just insertion. No current base has such edges, so the test needs a
   built fixture.
9. **`joint_prob_graph` refuses a discretized graph** (`was_dph`, or any
   aux vertex). Carrying `_indexer` through `discretize` would otherwise
   let it past the `TypeError` at `:9990-9994`, into a state-based base
   lookup (`self.find_vertex(base_state)`, `:10207`) that is undefined
   with duplicate zero states (reviewer B).
10. **The C++ parity port** `phasic::Graph::add_epoch` uses
    `epoch_rebuild_wider` (`phasiccpp.cpp:1285-1335`), a "faithful port"
    with the same defects: it copies `parameterized_edges` only and uses
    `find_or_create_vertex`. It gets the same vertex, edge and edge-order
    rules (reviewer B).

**Weight-mode policy for widening operations** (decision in §9):
- `discretize()`, `add_epoch()` and `joint_prob_graph()` refuse a graph in
  formula, callback or log mode with a clear `ValueError`. The new slots
  have no defined meaning there, and today the results are silently wrong
  (formula), lost (`joint_prob_graph` reverts to linear) or broken later
  (log).
- The refusal lives in those three public methods, NOT in
  `_rebuild_with_wider_layout`: `test_modeling_compose.py:307-316` calls
  the helper on a log graph and asserts the mode is carried.
- Unaffected, because they set the mode after `joint_prob_graph` on a
  linear base (reviewer B): `test_weight_formula_daisy.py:75-78`,
  `:128-131`, `:195`, `:214`, `:239`; `test_weight_formula_kwarg.py:218-224`;
  `test_joint_index_callback.py:91-92`; `test_callback_svgd_kwarg.py:267`;
  `inference/test_exact_grad_joint_index_baked.py:233`.

**Also loud:** `add_epoch()` on a discrete graph (`is_discrete` or
`was_dph`) raises a clear `NotImplementedError`, instead of the pybind
TypeError from `stop_probability_discrete` (measured). This also makes
carrying `rewards` safe: nothing adds vertices to a discrete graph after
it is discretized.

**Layout position (unchanged, now documented).** `discretize()` appends
one slot at the end, also on an epoch graph; it is not repeated per epoch.
That matches `joint_prob_graph`'s mutation slot. The `discretize()` and
`add_epoch()` docstrings say where new slots land.

## 7. Part 4 — `discretize()` callback returns one number

- Contract: `rate(state, **kwargs)` returns one finite number ≥ 0, for
  constant and parameterized graphs alike. Python numbers, numpy scalars
  and 0-d arrays are accepted (the notebook callback returns `np.float64`).
- **Constant graph:** the number is the aux edge's constant rate
  (unchanged).
- **Parameterized graph:** layout widened by one slot (Part 3). The number
  is the aux edge's coefficient in that slot, so the effective rate at a
  state is `rate(state) * θ[-1]`. This is the same code path the fixed
  rate uses today (`__init__.py:2837-2840`); reviewer A confirmed the
  semantics.
- A non-scalar return raises `TypeError` explaining the contract
  (decision in §9). A negative or non-finite value raises `ValueError`
  naming the state. A 0 is accepted.
- The fixed scalar rate keeps its existing "> 0" check.
- **Callers of the vector form**, all rewritten to the scalar form (full
  list from reviewer B):
  - `tests/pytest/inference/test_exact_grad_log_weight_mode.py:272`,
    `tests/pytest/inference/test_exact_grad_joint_index.py:371`;
  - `tests/pytest/test_graph_discretize.py:36-54` (in the §10 gate list);
  - `tests/pytest/test_reward_transform_discrete_flag.py:34-53`;
  - `experiments/dr_log_mode_moments_jac_gate.py:160` and
    `experiments/dr_sojourn_grad_theta_gate.py:164`, which CLAUDE.md names
    as verification gates;
  - `docs/pages/tutorial/parametrization.ipynb` (~line 2205-2215 of the
    JSON). Its base reserves a slot (`[rate, 0]`) and the callback fills it
    (`[0, n]`). The equivalent scalar form gives the same model: base
    `[rate]`, callback returning `n`, so discretize adds the slot and θ is
    still `[coalescent, mutation]` (§12);
  - `docs/pages/popgen/two-locus-island.ipynb` has the same pattern, but
    commented out (left alone).
- The log-mode test checks that the exact-gradient path declines a
  `was_dph` log graph. Today it gets one from a linear `discretize` with a
  vector callback, then sets log mode. With the scalar form, P widens to 3
  and the log product with a 0 pad fails, so the test builds the graph
  with `set_was_dph(True)` instead and keeps its assertion.
- Update stale texts: the `discretize()` docstring (`__init__.py:2785-2788`),
  the C comment at `src/c/phasic.c:11341-11344`, and the CLAUDE.md
  sentence quoting it.
- The C++ API `phasic::Graph::discretize` keeps refusing parameterized
  graphs (`phasiccpp.cpp:443-449`). That is loud and documented, so it is
  out of scope.

## 8. Tests (new file `tests/pytest/test_constant_parameterized_consistency.py`, plus Part 1's file)

All tests use a temporary cache directory and small graphs (tens of
vertices). Values are compared with tolerances. Tests avoid
`Graph(cb, theta_dim=...)` together with `add_epoch` (pre-existing
TypeError, §11).

- **Guard.** A parametrized test over every §4 site: a constant graph
  raises `ValueError` matching "must have parameterized edges", and the
  parameterized twin does not raise at the guard. Also a
  `reward_transform` output. It fails today for svgd, mcmc,
  method_of_moments, reward_visit_probability, the joint_* builders,
  add_epoch, `profile_graph` and the two bffg functions. Also: `profile()`
  on a constant graph WITHOUT θ works (after the `_max_rate` fix).
- **Edge-array capacity** (Part 3a), as described in §5.
- **Lossless widening.** For each chain below, the widened graph matches
  an independent oracle (distributions of the un-widened graph with the
  new slots neutral, or a hand build):
  - a graph with an `add_edge_constant` edge in the middle of a vertex's
    edge list → `discretize(0.1)`;
  - `add_epoch` → `discretize`, which keeps the epoch metadata and indexer;
  - `discretize` → `discretize(skip_existing=True)`, which adds no vertex
    and no slot;
  - widening a `joint_prob_graph` output (crashes today);
  - IPV edges and per-vertex `is_aux` preserved.
- **Bit-identity control.** For an ordinary linear coalescent, `add_epoch`
  and `discretize(0.1)` give exactly today's coefficient layout, vertex
  order and edge order (passes before and after).
- **Coefficient length.** A non-starting edge with length ≠ P makes
  `discretize` and `add_epoch` raise `ValueError`.
- **Weight modes.** `discretize`, `add_epoch` and `joint_prob_graph` on
  formula, callback and log graphs raise. The existing formula
  `joint_prob_graph` tests are unchanged.
- **Discrete add_epoch.** Raises `NotImplementedError`.
- **joint_prob_graph on a discretized graph.** Raises.
- **joint_prob_graph with a coefficient-less base edge.** Kept, with the
  same self-loop and t-vertex handling as coefficient edges (built
  fixture).
- **C++ `add_epoch`.** Same lossless chains as the Python one, through
  the C++ API (existing C++ test harness in `tests/cpp/`).
- **Callback.**
  - A scalar callback on a parameterized graph equals the fixed-rate path
    when the callback is constant.
  - A state-dependent callback (`nr_lineages * mu`) matches a hand-built
    graph.
  - A callback that returns 0 for some states (`0.3*s[0]`) gives E = 0.6
    on both graph kinds.
  - `np.float64` and 0-d array returns are accepted.
  - A vector return raises `TypeError`; a negative value or NaN raises
    `ValueError`.
  - A constant-graph callback is unchanged.

Every new test must fail on master for the stated reason, except the
controls, which pass before and after.

## 8b. Notebook updates

Part 2 and Part 4 change tutorial behaviour. Each notebook is changed to
the equivalent parameterized form; the model and θ meaning stay the same:
- `docs/pages/tutorial/joint-probability.ipynb`: the coalescent uses
  scalar rates (~JSON lines 197, 336), then `joint_prob_graph` (:366,
  :3388). Today that silently becomes a 2-parameter joint model. Fix: give
  the rates as lists (`[rate]`).
- `docs/pages/tutorial/time-inhomogeneous.ipynb`: same pattern (~:10853,
  :10867, then `joint_prob_graph` + `update_weights([7, 1])` at :11519,
  :13366, :13543). **This notebook has uncommitted edits by Kasper**, so
  it is not touched without his say-so.
- `docs/pages/tutorial/parametrization.ipynb`: the reserved-slot vector
  callback becomes the scalar form (§7).

The source cells are edited only. Outputs are regenerated by running the
notebooks only with Kasper's go-ahead, one at a time, under the §10
memory wrapper.

## 9. Decisions (made by Kasper, 2026-10-11)

1. **Stale graph-cache entries: delete the cache and bump the version.**
   `PHASIC_CALLBACK_VERSION` goes from "1.0" to "1.1" (`callback_hash.py:28`),
   so every old entry misses and is rebuilt. The existing cache under
   `~/.phasic_cache/graphs/` is deleted, which removes the files the bump
   would otherwise orphan. The format-marker option is dropped. Part 1's
   test T6 becomes: an entry written under version "1.0" is not loaded.
2. **`discretize()` callback returning a vector: raise `TypeError`.** One
   contract, no branching on the return type. Checked after the decision:
   the one notebook that fills a pre-reserved slot (`parametrization.ipynb`)
   has an equivalent scalar form (§7), so nothing becomes inexpressible.
3. **Widening under formula, callback and log weight modes:**
   `discretize()`, `add_epoch()` and `joint_prob_graph()` refuse all three
   modes (§6).
4. **`add_epoch()` and the joint_* builders on a constant graph: raise**
   (Goal 1). Consequence found by reviewer B after the decision: two
   tutorials rely on today's behaviour and need the list-form rates of
   §8b.
5. **One guard-allocator check in Part 3a: allowed.** One memory-capped
   run before the fix and one after, under macOS's guard allocator or an
   AddressSanitizer build.

## 10. Batches, gates and resource limits

Branch `fix/constant-parameterized-consistency`, one commit per part, in
order: Part 1, Part 2, Part 3a, Part 3, Part 4. Each part must pass these
gates before the next one starts:
1. Its new tests fail before and pass after (controls pass both ways).
2. Targeted files: the Part 1 list; `test_modeling_compose.py`,
   `test_modeling_epoch.py`, `test_modeling_joint_prob.py`,
   `test_graph_discretize.py`, `test_discrete_moments_and_reward.py`,
   `test_weight_formula_*.py`, `test_reward_transform_*.py`,
   `test_aux_vertex_constant.py`, and `inference/test_exact_grad_*.py`.
3. Full suite: no new failures relative to `b3-test-baseline.md`.
4. Adversarial review of the part's diff before it is committed.

**Resource limits (after the 2026-10-11 crash at ~38 GB).** Every probe
and test run goes through a wrapper with a hard memory limit in the child
process plus a parent RSS watchdog that kills it at a fixed budget. Runs
are sequential, never parallel. Review agents only read code unless
Kasper clears a run. The full-suite gate is run only with Kasper's
go-ahead, under the same wrapper.

`pixi run install-dev` runs after every source edit. Kasper merges and
pushes.

## 11. Out of scope (recorded)

- `joint_prob_graph(reward_rates_callback=...)` calls
  `self._joint_prob_graph`, which is commented out (`__init__.py:10350`).
  Any custom callback therefore raises `AttributeError` (source; not run).
  Confirmed by reviewer B (`__init__.py:10026-10035`); no test, notebook
  or experiment uses it, and the docstring (`:9977-9981`) claims it works.
  It is a candidate for a clear `NotImplementedError` (and a docstring
  correction), or for revival under the Part 4 contract.
- `pmf_from_graph` and `moments_from_graph` on this install fail with "JIT
  compilation requires the phasic C/C++ source tree on disk" (measured).
  This is an install-mode issue, not a θ issue.
- `add_epoch` on a graph built with `Graph(cb, theta_dim=1)` raises a
  TypeError: `_callback_kwargs` keeps `theta_dim` (`__init__.py:1567`),
  whereas `:1546` pops it for the C++ callback (reviewer A, measured).
- `from_serialized` does not restore the current θ (as in Part 1 §6).

## 12. Review record

**Reviewer A — correctness: SOUND-WITH-CORRECTIONS.** All §2 findings
reproduced. Corrections folded in:
- Blocker: the edge-array capacity bug (§2e), which Part 3 would trigger.
  It became Part 3a (§5).
- `add_epoch` on a formula graph is silently wrong, and `joint_prob_graph`
  drops the weight state. Both now refuse non-linear modes (§2c, §6).
- Coefficient length ≠ P is independent of the weight mode. `discretize`
  and `add_epoch` now refuse it instead of padding (§6.2).
- A per-state rate of 0 is legitimate, so the rule is ≥ 0; numpy scalars
  are accepted (§2d, §7).
- The vertex lookup rule was corrected (§6.1).
- The metadata list was completed: `synthetic`, `is_aux`,
  `_base_callback` (§6.7).
- Notes folded in: the joint-graph widening crash (§2b), the measured
  constant-base `joint_prob_graph` (§2a), the log-mode wording (§2c), the
  guard message (§4), bit-identity and edge order (§6.3, §6.5, §8), and the
  `theta_dim` TypeError (§11).

**Reviewer B — blast radius: SOUND-WITH-CORRECTIONS** (first run lost
in the crash; re-run read-only on v2, nothing executed). Corrections
folded in:
- Blocker: three more exact-size edge appenders (the clone's
  starting-vertex loop, `GraphBuilder::build`, `scc_synthetic.c`), plus
  the paths exposed today (§2e, §5). The clone claim was confirmed in
  source by the author.
- Six more callers of the vector-returning discretize callback, including
  a tutorial (§7).
- Two tutorials that rely on constant graphs going into `joint_prob_graph`
  (§8b).
- `synthetic` cannot be read from Python, so a getter was added (§5, §6.7).
- Missed entry points: `profile_graph` and `_max_rate`, module-level
  `method_of_moments`, two bffg functions (§4).
- The C++ `add_epoch` port has the same defects (§6.10).
- `find_vertex` throws, and the lookup-tree justification was corrected
  (§6.1).
- The weight-mode refusal lives in the public methods, not the helper
  (§6). The log-mode test's reason was corrected (§7).
- `joint_prob_graph` refuses discretized graphs; the edge copy mirrors the
  full loop (§6.8, §6.9).
- Confirmed: no internal chain passes a constant graph to a guarded site;
  the `reward_rates_callback` breakage; nothing relies on today's lossy
  widening (§4, §11).

## 13. Implementation record

- **Part 1** (commit `553dc412`): see `graph-cache-constant-roundtrip-plan.md` §13.
- **Part 2** (shared guard): `_require_parameterized` in `src/phasic/__init__.py`,
  called first in 15 Graph entry points and 5 module-level functions
  (`method_of_moments`, `probability_matching`, `profile_graph` when θ is
  given, `bffg.path_exit_rates_by_param`, `bffg.bffg_log_prob`).
  `profile._max_rate` now tests `parameterized()`: measured before the fix,
  `profile()` on a constant graph raised "Cannot call update_weights() on
  constant graph" even with no θ. The old later checks
  (`param_length() == 0`, serialized `param_length == 0`) are left in place.
  Tests: `tests/pytest/test_constant_parameterized_consistency.py`, 46 tests;
  24 fail before, all pass after. The 82 existing test files that call any
  guarded entry point: 937 passed, 71 skipped, 3 xfailed, 1 failed. The
  failure, `test_reward_validation.py::test_svgd_runs_silently_on_partial_coverage_with_introspection`,
  is an unseeded SVGD accuracy test (θ 11.59 vs 10 ± 1.5); it passed in the
  Part 1 full run and on 3 of 3 re-runs, so it is pre-existing flakiness.
- **Part 3a** (edge-array capacity): new `ptd_vertex_append_edge`
  (`src/c/phasic.c`, declared in `api/c/phasic.h`) appends with the same
  power-of-two growth rule as `ptd_directed_graph_add_edge` and leaves the
  vertex unchanged on allocation failure. All seven exact-size appenders
  now use it; a grep confirms only these two functions append to
  `vertex->edges`. Additive pybind getter `_is_synthetic`. Measured before
  the fix: `test_edge_array_capacity.py` crashed Python with a segmentation
  fault at 4 rate edges + `add_edge_constant` + 1 rate edge on one vertex,
  without any guard allocator. After: 40/40 pass, also under the macOS
  guard allocator (16-byte alignment, so only overruns at even pointer
  counts are detectable; byte-exact sizing stops Python from starting).
  Focused run of 24 files covering clone, discretize, aux vertices,
  serialization, FFI/GraphBuilder, SCC and synthetic graphs, daisy chain,
  epoch and joint graphs: 661 passed, 4 skipped, 2 xfailed, 0 failed. A
  broader 124-file run was stopped after 18 files (all passing) because
  it would take hours; the full suite is deferred to one run after Part 4.
- **Process slip:** Parts 2 and 3a were committed before their diff review
  (gate 4). A read-only review of Parts 2, 3a and 3 together followed:
  Parts 2 and 3a APPROVED (no defects); Part 3 APPROVE-WITH-CHANGES.
- **Part 3** (lossless widening). `_rebuild_with_wider_layout` rewritten as
  §6 (vertex registration rule, edge order, coefficient-less edges copied
  with `add_edge_constant`, explicit metadata list, synthetic graphs
  refused). Refusals in `discretize`/`add_epoch`/`joint_prob_graph` for
  non-linear weight modes and ragged coefficient vectors; `add_epoch`
  refuses discrete graphs after the more specific joint-graph check; and
  `joint_prob_graph` refuses discretized graphs. `discretize(skip_existing=True)`
  adds no slot when no vertex needs an aux. `add_epoch` drops a carried
  `rewards` (it adds vertices). C++ `add_epoch`: same lossless rules, plus
  refusals of non-parameterized and `was_dph` graphs (its non-parameterized
  branch could never succeed).
- **Deviation from §6.8 (decided during review, reversible).** Measured in
  source (`src/c/phasic.c:5796-5811`): a `was_dph` graph's `update_weights`
  divides EVERY edge by the vertex total, including coefficient-less edges
  it never recomputes, so a constant edge next to rate edges drifts between
  calls. And `joint_stop_prob_graph`/`joint_sojourn_graph` copy only
  parameterized edges. So instead of copying coefficient-less edges,
  `discretize()` on a parameterized graph and `joint_prob_graph()` REFUSE a
  non-aux vertex with one (clear `ValueError`). Previously such edges were
  silently dropped. `add_epoch` (continuous, no renormalisation) still
  carries them losslessly. Supporting them in discrete/joint graphs needs
  an idempotent renormalisation in C plus JSP/sojourn changes: a possible
  follow-up.
- **Notebooks affected by Part 3** (for Kasper):
  `docs/pages/popgen/coalescent-derived.ipynb` (cells 8 then 26) and
  `docs/pages/popgen/old-moran-coalescent-derived.ipynb` (cells 19 then 37)
  call `joint_prob_graph` on a `discretize()`d graph. Those cells have no
  saved output; by source reading they already failed before (discretize
  dropped `_indexer`, giving a TypeError). They now fail with the clear
  "discretized graph" error.
- Tests: 63 + 5 C++ in `test_constant_parameterized_consistency.py` and
  `test_cpp_add_epoch_lossless.py`; the original 14 Part 3 tests fail
  before. Focused run, 45 files: 950 passed, 5 skipped, 2 xfailed, 0 failed.
