# Plan — one discretize(), owned by C++, and standards for layout widening

Status: v2, 2026-10-12. Two read-only adversarial reviews, both
SOUND-WITH-CORRECTIONS; their corrections are folded in (record in §11).
All nine decisions in §10 made by Kasper on 2026-10-12 (the recommended
option in each). Implementation starts after the prerequisite branch below
is merged; each phase still passes its gates and diff review. Builds on
branch `fix/constant-parameterized-consistency` (the constant-vs-parameterized
consistency work: Part 1 cache round trip, Part 2 one guard for constant
models, Part 3a edge-array capacity, Part 3 lossless widening, Part 4
one-number discretize callback; record in
`constant-parameterized-consistency-plan.md` §13), which must be merged
first.

## 1. Goal and scope

Kasper's grand plan: every capability is implemented in exactly one place,
and is reachable from both the Python and the C++ API.

This plan does that for **discretize()** and fixes, on the way, the standards
every layout-changing operation (`discretize`, `add_epoch`,
`joint_prob_graph`) must follow:
- the **widening standard** (§3): what the shared widening primitive
  guarantees, and what each operation must check, change and record;
- the **layout-description standard** (§4): how a graph describes its state
  vector (the StateIndexer held in `graph._indexer`), its vertex roles, and
  its parameter vector θ, including where a mutation or discretization rate
  goes.

Both standards are written so a later C++ `joint_prob_graph()` can follow
them (§8).

**In scope:** a de-risk phase; a pinning gate; one C++ widening primitive;
C++ `discretize()` at parity with today's Python; Python `discretize()` as a
thin wrapper; a graph-level θ-layout descriptor (not persisted yet).
**Defined but not built:** `add_epoch` delegation beyond the shared
primitive, a C++ StateIndexer, C++ `joint_prob_graph()`, the model-level θ
layout used by SVGD epoch routes.

## 2. Today (read from source on the branch above; confirmed by reviewer A)

| Capability | Python | C++ API |
|---|---|---|
| Widening | `Graph._rebuild_with_wider_layout` (`src/phasic/__init__.py`), a Python loop over all vertices and edges; lossless since Part 3; refuses only synthetic graphs; used by `discretize` and `add_epoch` and called directly by tests (`test_modeling_compose.py:266-313`) | `epoch_rebuild_wider` (`src/cpp/phasiccpp.cpp`), file-local, used only by C++ `add_epoch` (`:1399`); a separate port of the same logic, with its own ragged-coefficient refusal (`:1305-1316`) |
| `discretize` | constant and parameterized graphs, fixed rate or one-number callback (receives a numpy int array, `phasic_pybind.cpp:4359-4365`), widening, refusals, `was_dph`, `is_aux`, `rewards`. Constant path uses `clone()` and DROPS `_indexer`, `_callback`, `_callback_kwargs`, epoch attributes (`__init__.py:2974`); parameterized path CARRIES them | `phasic::Graph::discretize` (`phasiccpp.cpp:442`, `:504`): refuses parameterized graphs (`:444-449`); never sets `was_dph`; fixed rate must be in (0, 1) (`:506`) while Python allows > 0; callback values unvalidated (`:486`). No binding, no caller, no test, no tutorial use |
| Legacy copy | `_discretize_inplace` (`__init__.py:2794`): old vector semantics, no validation; used only by `test_modeling_compose.py:488`, `:527` | — |
| Refusals from Parts 2-4 | all in Python; exception types: guard `ValueError`, discrete `add_epoch` `NotImplementedError`, callback `TypeError`/`ValueError` | C++ `add_epoch` only: constant (`:1369`), `was_dph` (`:1377`), ragged; all `std::invalid_argument` (→ `ValueError`) |
| Weight mode | `_weight_mode` written at 13 places (`__init__.py:669, 797, 1621, 1694, 1924, 1942, 1944, 1983, 1989, 2012, 6825`; `_graph_serialize.py:712`; `distributed_scc.py:55`); `Graph(_Graph)` resets it to linear (`:1694`) | partial: `_weight_formula_src` + a `weight_formula` setter (`api/cpp/phasiccpp.h:253, 1105`); formula detectable in C via `weight_tape != NULL`; log/callback are per-call arguments (`update_weights(..., use_log)`, `phasiccpp.h:872`) |
| `is_discrete` | Python attribute | none (C has `was_dph` only) |
| `is_aux` | set by discretize | C flag; **not serialized** (`_graph_serialize.py` has no `set_aux`), so lost on `from_serialized` and graph-cache hits |
| State layout | `StateIndexer` in `graph._indexer`; read only by `add_epoch` (`:3152-3168`) and `joint_prob_graph` (`:10257-10611`) | none |
| θ layout | nowhere (the `add_epoch` docstring describes its interleaved layout) | none |
| C++ `clone()` | — | `phasiccpp.h:1459-1468` builds a fresh `Graph` from the C clone: drops every C++-only member (`_weight_formula_src`, epoch metadata, `_callback`); only copy/move/assign carry them (`:248-273`) |

## 3. The widening standard

Reviewer B showed the v1 rules mixed two things: what the shared primitive
guarantees, and what an operation does around it. They are separated here.

### 3a. Primitive contract — `widen_layout` (mechanics only)

P1. **Appends** `extra_state` entries to every state and `extra_coeff`
    slots to every coefficient-carrying edge; existing θ slots keep their
    index (a θ-index rule; coefficient index equals θ index only under O3).
P2. **Lossless otherwise:** vertex indices, duplicate states (created
    unregistered, never merged), edge order, coefficient-less edges
    (copied unpadded), `is_aux`, `was_dph`.
P3. **Refuses only synthetic SCC graphs** (placeholder coefficients).
P4. **No policy checks.** Constant, discretized and joint graphs are widened
    if asked (existing tests do this).

### 3b. Operation contract — `discretize`, `add_epoch`, (later) `joint_prob_graph`

O1. **Constant models:** `discretize` keeps them constant; `add_epoch`
    and `joint_prob_graph` refuse (Part 2 guard).
O2. **Linear weight mode only** for operations that add θ slots.
O3. **Uniform coefficient length** (0 or `param_length()` on every
    non-starting edge), else refuse.
O4. **Refuse what cannot be represented:** coefficient-less edges on
    transient vertices where the result renormalises per θ (`discretize`,
    discrete joint graphs) or is read by builders that ignore them.
O5. **Each operation lists what it changes besides widening** (and that
    list is documented and pinned):
    - `discretize`: adds one aux vertex per transient vertex, `normalize()`
      rewrites every weight, sets `was_dph`, `is_discrete`, `rewards`;
    - `add_epoch`: adds vertices and edges for the new epoch, drops
      `rewards`, adds the `'epoch'` state slot;
    - `joint_prob_graph`: a **product construction**, not a widening. It
      rebuilds by traversal (no input vertex index survives), adds a trash
      pair and t-vertices, and only reuses the slot rules (one appended
      mutation slot).
O6. **Records every added θ slot** in the θ layout (§4c).
O7. **One implementation of every numeric and structural check**, in C++
    (`std::invalid_argument` → Python `ValueError`, unless the frozen API
    says otherwise — see the error table §5e). Python keeps only argument
    marshalling and checks of Python-only state (O2 for log/callback mode,
    §5b).
O8. **Changing the weight mode replaces the θ layout.** Setting
    `weight_formula`, a weight callback or log mode on a graph turns its
    layout into "opaque" (formula/callback) or plain `base` slots (log),
    because θ is then resolved at inference time (`_resolve_inference_theta_dim`,
    `__init__.py:5665-5716`).

## 4. The layout-description standard

### 4a. State layout — the StateIndexer (`graph._indexer`)

Rule: after any operation, `graph._indexer` is `None` or has
`state_length == graph.state_length()`; an operation that widens the state
records the input's indexer as the base.

| Operation | State change | Indexer change (today, kept) |
|---|---|---|
| `add_epoch` | one entry appended | `'epoch'` slot appended; base in `_base_indexer` |
| `joint_prob_graph` | one count per rewarded property | base `+` reward indexer (property sets `{pset}_{prop}`); base in `_joint_prob_base_graph_indexer` |
| `discretize` | none: aux vertices get the all-zero state | none: aux vertices lie outside the indexer (decision 1) |

**Where the mutation rate goes.** A mutation or discretization *rate* is a θ
slot, described by the θ layout (§4c). The StateIndexer only gains mutation
*counts*, and only in `joint_prob_graph` (state), as today. In `discretize`
mutations are aux-vertex visits, counted by `rewards`. No consumer of
`_indexer` is affected by aux vertices outside it (reviewer A: the only
readers are `add_epoch` and `joint_prob_graph`, which refuses discretized
graphs; joint graphs already contain unregistered all-zero vertices).

Small additions: a public read-only `graph.indexer` property.

### 4b. Vertex roles

Roles are explicit flags, not inferred from states:
- **aux** (exists): `is_aux`. Made persistent by adding it to `serialize()`
  as an additive key (decision 1b). `discretize(skip_existing=True)` keeps
  today's "state sums to 0" rule until then, and phase 0 pins both.
- **trash** and **t-vertex** (joint graphs): today inferred (`_is_trash`,
  `__init__.py:10955`: state sum 0 plus a 2-cycle; t-vertices during the
  build). Defined here as future explicit flags, set by `joint_prob_graph`;
  required before a C++ `joint_prob_graph` (§8).

### 4c. θ layout — `ParameterLayout`

The layout of the **graph's** θ (the vector `update_weights` takes). It is
not the θ SVGD optimizes on epoch routes: the daisy-chain / `epoch_model`
routes replicate a non-epoch graph per epoch (`n_epochs * param_length`,
`__init__.py:4518`) and then apply tied/fixed/exposure. That **model θ
layout** is derived from the graph layout by the SVGD configuration code,
later (§8).

- **Descriptor** per slot: `index`, `name`, `kind`, optional `epoch`.
  Kinds: `base`, `add_epoch_dynamics`, `add_epoch_transition`,
  `add_epoch_dummy`, `discretization`, `mutation`.
- **Linear mode only.** Formula/callback graphs report an opaque layout of
  unknown meaning (O8).
- **Stored as annotations, built on demand.** The graph stores the blocks
  that operations appended (name, kind, count); `theta_layout` builds the
  full list when asked, with `base` slots named `theta0..` for the rest.
  Every reader checks `len == param_length()` and raises on a mismatch (it
  can go stale if `set_param_length` or new edges change the length).
- **Descriptors are immutable** except an epoch annotation: after the first
  `add_epoch`, slots 0..B-1 keep index and name and gain `epoch=0`.
- **Owner:** a C++ member of `phasic::Graph`, which means C++ `clone()`,
  `clone_p()`, the copy/move/assign operators and the Python
  `Graph(_Graph)` wrap must all carry it (reviewer A; today `clone()`
  drops C++-only members).
- **Not persisted yet** (decision 2b): a graph reloaded via
  `from_serialized` or the graph cache gets the default layout. Persisting
  waits for Stage-3 step 9, which makes one serializer the owner. Names are
  never part of cache identity (the graph hash is structural).
- **Naming:** `graph.set_parameter_name(index, name)`. A `rate_name=`
  argument on `discretize` would collide with `**kwargs`, which go to the
  rate callback (decision 9). Default name of the discretize slot:
  `discretization`.

## 5. Design of the C++ single source of truth

5a. **Widening primitive** `phasic::Graph::widen_layout(size_t extra_state, size_t extra_coeff)`
(non-const: `vertices()`, `vertex_exists`, `find_vertex` are non-const),
implementing §3a only. Replaces `epoch_rebuild_wider`. Python
`_rebuild_with_wider_layout` calls it and copies only Python-side attributes
(`_indexer`, epoch metadata, callbacks, weight-formula state, `rewards` with
its length check, `_last_callback_vertices_length`, `_callback_kwargs`).
Rewiring `add_epoch` (Python and C++) onto it is decision 7.

5b. **Weight mode: hybrid** (decision 3). C++ detects formula mode itself
(`weight_tape != NULL`). Log and callback mode are Python-side state, so
the wrapper passes the mode in as an argument (`weight_mode`) and C++
raises the refusal at its original position in the check order (as built:
an argument rather than a separate Python check, so the check stays in
one place). Likewise `is_discrete` is passed in (`input_is_discrete`, which
selects the DPH normalisation). No new C++ `weight_mode`/`is_discrete`
fields; those are defined together with Stage-3 step 10 (weight-formula
promotion), which owns C++ weight state.

5c. **C++ discretize at parity.** Fixed rate: finite and > 0 (Python's
rule). Callback: `std::function<double(const std::vector<int>&)>`, value
finite and >= 0. Constant graphs stay constant; parameterized graphs widen
by one slot via 5a (none when `skip_existing` finds nothing new) with the
rate in the new slot; checks O2 (formula), O3, O4; sets `is_aux` and
`was_dph` (also on constant graphs: today's Python does); returns
`DiscretizeResult{graph, rewards}`; records the `discretization` block.

5d. **Python wrapper** `Graph.discretize`, same signature. It:
- checks the Python-only state (log/callback mode, `is_discrete`);
- wraps a callable as `lambda s: _marshal(rate(np.asarray(s, dtype=<the Vertex.state dtype>), **kwargs))`,
  so callbacks keep receiving a numpy array; `_marshal` turns any real
  scalar into `float` and raises today's `TypeError` otherwise;
- calls the bound C++ `discretize`;
- sets the Python attributes (`is_discrete`, `rewards`, and the carried
  attributes per decision 8).
Callback exceptions propagate with their original type (pybind's
`functional.h` takes the GIL and re-raises). The wrapper stays in `Graph`
(it uses `super()`); it moves with the Stage-3 Graph-class split later.

5e. **Error table** (pinned in phase 0; types are frozen API):

| Rule | Python type today | C++ throws | Arrives in Python as |
|---|---|---|---|
| constant model (Part 2 guard) | ValueError | `std::invalid_argument` | ValueError |
| non-linear weight mode | ValueError | formula: `std::invalid_argument`; log/callback: Python | ValueError |
| ragged coefficients | ValueError | `std::invalid_argument` | ValueError |
| constant edges (O4) | ValueError | `std::invalid_argument` | ValueError |
| fixed rate not a number / bool | TypeError | (Python) | TypeError |
| fixed rate <= 0 or non-finite | ValueError | `std::invalid_argument` | ValueError |
| callback returns non-scalar | TypeError | (Python `_marshal`) | TypeError |
| callback returns < 0 or non-finite | ValueError | `std::invalid_argument` | ValueError |
| kwargs with fixed rate | TypeError | (Python) | TypeError |
| `add_epoch` on a discrete graph | NotImplementedError | `std::invalid_argument` today | must stay NotImplementedError: map explicitly |

Messages: phase 0 pins the type, the word "discretize" (or the operation
name) and the key phrase of each message, not the full text (state
formatting differs between numpy and C++).

5f. **Remove the duplicates** after delegation: the Python discretize loop,
the Python widening loop, `epoch_rebuild_wider` (if decision 7), and
`_discretize_inplace` (decision 5; its two equivalence tests become recorded
outputs in the phase-0 gate).

## 6. Frozen-API changes this plan makes (for sign-off)

**As built (phases 1-4), in addition to the list below:**
- `discretize(0)` / `discretize(-1)` (integer rates) now say "got 0.0" /
  "got -1.0" (the value is formatted after conversion to float), and a
  `np.float32` rate prints its float64 value.
- The callback `TypeError` prints the state as `[3, 0]` instead of
  `[np.int32(3), ...]`.
- Private bindings added: `_widen_layout`, `_discretize` (two overloads),
  `_discretize_check_rate`, `_theta_layout`, `_set_parameter_name`,
  `_append_theta_block`.
- C++ API: `discretize(rate | fn, skip_existing, input_is_discrete,
  weight_mode)`; refusals are `std::invalid_argument` (the fixed-rate check
  threw `std::runtime_error` before, and required rate < 1); new
  `discretize_check_rate`, `copy_construction_state_from`, `widen_layout`,
  `theta_layout`, `set_parameter_name`, `append_theta_block`. C++
  copy-assignment fixed (it crashed).
- Kept unchanged on purpose: the `extend()` resume point
  (`_last_callback_vertices_length`, pinned) and the rewards-length check on
  a parameterized input.


Python: new read-only properties `graph.indexer` and `graph.theta_layout`;
new `graph.set_parameter_name()`; `_discretize_inplace` deleted; `is_aux`
added to `serialize()` output (decision 1b); constant-graph `discretize`
now carries `_indexer`, callbacks and epoch attributes like the
parameterized path (decision 8). Nothing else changes: phase 0 pins it.
C++: `discretize` accepts parameterized graphs, accepts fixed rates >= 1,
validates callback values, sets `was_dph`; new `widen_layout`,
`theta_layout` accessors. No existing C++ caller exists (reviewer B).

## 7. Phases and gates

Branch `refactor/discretize-cpp-ssot`. Each phase: tests that fail before
and pass after; a read-only adversarial review of the diff before commit;
targeted test files; the full suite once per phase only with Kasper's
go-ahead. Every run memory-capped and sequential.

-1. **De-risk (branch `derisk/discretize-cpp`, measurements only, memory
    capped, per model family, at 20k-60k and 200k-600k vertices):**
    (a) cost of today's Python widening loop vs the aux-vertex loop vs the
    per-vertex Python callback, to confirm the C++ move pays and where;
    (b) cost and GIL behaviour of a pybind `std::function` callback at
    600k vertices (a small prototype binding on the de-risk branch);
    (c) bit-identity of `normalize()` and of the aux-edge construction
    between today's Python path and the C++ path on the same graph.
    Re-evaluate the remaining phases from the findings.
0. **Pin today's behaviour (gate).** Records (not functions) Python
   `discretize` outputs: states, `is_aux`, edges with weights and
   coefficients, `rewards`, `param_length`, `was_dph`, `is_discrete`,
   carried attributes per path, distribution values. Fixtures: constant and
   parameterized coalescents; fixed rate; numpy-state callback
   (`state.sum()`, slicing); callback with zeros; `skip_existing`; epoch
   graph; indexer graph discretized constant and parameterized; serialize
   round trip of a discretized graph; the error table §5e; the three
   existing direct widening tests; the two `_discretize_inplace`
   equivalence tests' outputs. Green on the starting commit.
1. **C++ widening primitive** (5a) + binding; Python helper delegates.
   Gates: identical to today's helper on all Part 3 fixtures; if decision 7
   rewires `add_epoch`: `test_gate_daisy_chain_joint_probs.py`,
   `test_cpp_add_epoch_lossless.py`, the epoch tests, `test_modeling_compose.py:266-313`,
   and C++ `add_epoch` still refuses ragged and `was_dph` graphs with its
   current messages (the checks move to an operation-level guard).
2. **ParameterLayout** (§4c): C++ member carried by `clone()`/`clone_p()`,
   copy/move/assign and the Python wrap; on-demand build; naming; not
   persisted. Gates: clone, `copy()`, `from_serialized` and a graph-cache
   hit give the stored or the documented default layout; O8 (setting a
   formula makes it opaque).
3. **C++ discretize at parity** (5c), formula refusal via the tape. Gate:
   phase-0 gate passes when driven through C++ directly (cppimport, as
   `test_cpp_add_epoch_lossless.py` does); C++ `add_epoch` now refuses a
   C++-discretized graph (it carries `was_dph`).
4. **Python delegates** (5d, 5f). Gates: phase-0 gate unchanged; all
   discretize-related test files; the two experiment gates
   (`dr_log_mode_moments_jac_gate.py`, `dr_sojourn_grad_theta_gate.py`).
5. **`is_aux` in serialize** (decision 1b): additive key; `skip_existing`
   keeps its state-sum rule (switching would change pinned behaviour).
   Gate: round trip and the graph-cache JSON path keep `is_aux`; the
   phase-0 round-trip pin records the intended change (flags now kept).
   Behaviour change, pinned: a reloaded parameterized discretized graph can
   be re-discretized with `skip_existing` (its aux back-edges are now
   recognised as aux, not refused as constant edges); data without the key
   (older caches) behaves as before.

## 8. Later work this plan prepares (defined, not built)

- **`add_epoch` delegation** (beyond the primitive): needs a C++ indexer for
  the `'epoch'` slot, `add_epoch_*` blocks in the θ layout, parity checks,
  and the `NotImplementedError` mapping.
- **Model θ layout** for SVGD epoch routes (daisy chain, `epoch_model`):
  derived from the graph layout (replicate per epoch, then tied/fixed);
  `exposure_param_index` and plot `param_names` (`plot.py:131`) can then
  resolve by name.
- **C++ StateIndexer (index codec):** must reproduce `entity_order`
  (`state_indexing.py:804-885`), property sets, `+`, `props_to_index`. A
  design task of its own; matches the Stage-3 "IndexCodec" idea.
- **C++ `joint_prob_graph`:** O1-O8 with O5's "product construction"
  entry; state = base `+` reward indexer; θ = one appended `mutation`
  block; explicit trash and t-vertex flags (§4b); adds an O3 refusal that
  today's Python lacks; must NOT copy the dead formula-era
  `base_coeff_length > param_length` branch (`__init__.py:10430-10450`,
  unreachable since the linear-mode guard); then `joint_stop_prob_graph`,
  `joint_sojourn_graph`.
- **Constant edges in discrete graphs:** idempotent per-θ renormalisation in
  C plus the joint_stop / joint_sojourn copy loops; would lift O4.
- **Persisting the θ layout:** with Stage-3 step 9.

## 9. How this fits the larger refactoring plan (`stage3-refactor-plan.md`)

The larger plan (July; the Stage-3 refactor) sets a delegation doctrine (C
core owns numeric kernels; C++ is the typed facade; pybind only marshals;
Python never re-implements native logic) and records decision Q4, "C++ is a
product, so the C++ ports become the single source of truth", done by
complete-then-promote (reach parity, bind and gate, delegate and delete).
Executed so far: WS-A (safety-net test gates, `3298793f`), WS-B (dead-code
removal) and part of WS-C (splitting the Graph class), merged at
`9de5d61a`; no WS-E promotion (the C++ single-source ports) has started
(both reviewers confirmed from git log and `stage3-execution-handoff.md`).

**Placement.** This plan is Stage-3 step 11 ("WS-E: discretize — extend the
native port to parameterized graphs and DPH flagging; bind; Python
delegates"), expanded with the standards. Recommended: **the first WS-E
promotion**, ahead of step 9 (serialize single owner) and step 10
(weight-formula promotion). Reviewer B: defensible, with no hard dependency
on step 3 (splitting the pybind module into `register_*()` functions), on
moving the discretize cluster out of `Graph`, or on the #9 (SCC ordering)
and #12 (C elimination variants) gates; the GraphBuilder/FFI path needs
only `was_dph`, which serialize already carries. Two conditions, both
built into v2:
1. the θ layout is **not persisted** until step 9 (else do step 9 first);
2. no C++ `weight_mode`/`is_discrete` field is created before step 10
   defines it (the hybrid of 5b).

**Records in the larger plan to update** (no code):
- §3a's note on C++ `discretize` predates Parts 3-4; parity now means
  today's Python, pinned by phase 0.
- §3b row #3 ("Python `compile_formula` is the sole compiler") contradicts
  §9 Q3 ("promote the C++ `wf_*` compiler; Python delegates"); record it as
  resolved in Q3's favour.
- The C API freeze (decision Q13, "freeze the public C symbols") vs Part 3a's
  additive `ptd_vertex_append_edge`: confirm that the freeze means "no
  removals or signature changes".
- Line numbers date from July; `PROJECT-OVERVIEW.md` lists "Stages 1-3" as
  Done although only WS-A, WS-B and part of WS-C were executed.
- Move the StateIndexer/index-codec item (step 19) ahead of any C++
  `joint_prob_graph` work.

## 10. Decisions (made by Kasper, 2026-10-12)

1. **Aux vertices stay outside the StateIndexer** (all-zero state,
   `is_aux=True`), as now. **1b:** `is_aux` is added to `serialize()`
   output (phase 5).
2. **θ layout on the graph, owned by C++**, separate from the
   StateIndexer. **2b:** not persisted until Stage-3 step 9 (serialize
   single owner); reloaded graphs get the default layout.
3. **Weight mode / `is_discrete`: hybrid.** C++ checks the formula tape and
   `was_dph`; Python checks log/callback mode and `is_discrete` (the one
   recorded exception to O7). The C++ fields are defined with Stage-3
   step 10.
4. **Widening primitive: a C++ method** (`phasic::Graph::widen_layout`), not
   a new C function.
5. **`_discretize_inplace` is deleted** after its two equivalence tests'
   outputs are recorded in the phase-0 gate.
6. **Placement: first WS-E promotion** (first C++ single-source port), under
   the two conditions of §9.
7. **`add_epoch` (Python and C++) is rewired onto the primitive in phase 1**,
   removing `epoch_rebuild_wider`; gated by the daisy-chain and epoch tests.
8. **Constant-graph `discretize` carries the same attributes as the
   parameterized path** (`_indexer`, callbacks, epoch attributes). A
   behaviour change, pinned in phase 0 as intended.
9. **θ slots are named with `graph.set_parameter_name(index, name)`**; no
   `rate_name=` argument on `discretize`.

## 11. Review record

**Reviewer A — design vs code: SOUND-WITH-CORRECTIONS.** §2 confirmed line
by line. Folded in: the primitive must not enforce policy (blocker → §3a
P4); C++ `clone()` drops C++-only members (blocker → §4c owner, phase 2
gates); weight mode is not C++ state, a C++ copy would be a third source
(→ 5b hybrid, phase 3 of v1 dropped); callback state type, error types and
messages (→ 5d, 5e); constant vs parameterized attribute asymmetry (→
decision 8); θ length is not `param_length` outside linear mode (→ §4c
linear-only, on-demand, length check); `is_aux` not serialized (→ §4b,
decision 1b, phase 5); missing fixtures and gates (→ phases 0, 1);
`widen_layout` cannot be `const`; performance claim unmeasured (→ phase -1).

**Reviewer B — fit and API impact: SOUND-WITH-CORRECTIONS.** Stage-3 status
claims confirmed. Folded in: "lossless" is false for the operations (→ §3
split, O5, joint_prob_graph as construction); the θ layout cannot describe
the SVGD model θ (→ §4c graph-only, model layout in §8, `add_epoch_*`
kinds); stale descriptors on mode change and re-kinding (→ O8, immutable
descriptors); serialize key before step 9 (→ not persisted, decision 2b);
weight-mode field vs step 10, and §2's "not stored" was wrong (→ §2, 5b);
`is_discrete` (→ 5b, decision 3); exception types are frozen API (→ 5e);
phase 1 also changes `add_epoch` (→ decision 7, gates); missing de-risk
(→ phase -1); what C++ `joint_prob_graph` needs (→ §4b roles, §8);
the dead formula branch in `joint_prob_graph` (→ §8); `rate_name` collides
with `**kwargs` (→ decision 9); frozen-API change list (→ §6); wrapper
location (→ 5d); bare labels qualified.

## 12. Implementation record (2026-10-13)

Worktree `/Users/kmt/phasic-discretize`, branch `refactor/discretize-cpp-ssot`
(from `derisk/discretize-cpp`), own pixi environment; every run memory-capped
and sequential. Each phase: new tests failing before / passing after,
read-only adversarial diff review (APPROVE-WITH-CHANGES each time, changes
applied), the related test files (52 files, 0 failures at each phase).

- Phase -1 (`2de18c39`): `discretize-cpp-derisk-findings.md`. Python
  widening was 70-95% of discretize time; existing C++ discretize
  bit-identical to Python on constant graphs; C `ptd_clone_graph` is
  O(V*E) (110 s at 204k vertices; not fixed, decision pending).
- Phase 0 (`fa67c9e5`, strengthened `14a8ce13`): pinned records of
  Python discretize, generated with the master build as reference.
- Phase 1 (`c309719c`): `phasic::Graph::widen_layout`, 14-35x faster,
  linear (`experiments/dr_widen_layout_perf_gate.py`).
- C++ copy-assignment crash fixed (`3ada4c08`).
- Phase 2 (`54dcf36e`): `theta_layout`, `set_parameter_name`.
- Phase 3 (`fcd70289`): C++ discretize at parity; every pinned case and
  error reproduced through C++.
- Phase 4: Python `discretize` delegates; Python loops,
  `_discretize_inplace` and the old C++ widening helper deleted. discretize
  at 204k vertices: 17.7 s -> 0.8 s (fixed rate), 18.3 s -> 1.7 s
  (callback). Intended changes pinned: decision 8 (constant path keeps
  callback and indexer).
- Existing behaviour found and pinned, not changed: discretize of a native
  DPH gives a rewards vector one short; a second discretize with
  skip_existing leaves weights un-normalised until update_weights.
- Phase 5: `is_aux` serialized (additive key, strict parsing); not yet in
  the C++ `SerializedGraph` port or the SLURM `deserialize_scc_synth` path
  (documented residuals).

## 13. Status and hand-off (2026-10-14)

**Where the work is.** Branch `refactor/discretize-cpp-ssot` in the git
worktree `/Users/kmt/phasic-discretize` (own pixi env; created by Claude,
allowed by Kasper). Not merged, not pushed. All five plan phases plus the
follow-up fixes are committed (15 commits on top of master `a39afc17`, see
`git log master..HEAD`).

**Follow-up fixes after the plan (Kasper's requests):**
- `bbb016bc` C `ptd_clone_graph` linear (110 s -> 0.55 s at 204k vertices).
- `644132a0` discretize refuses native DPH input; skip_existing keeps weights.
- `33cd4fa5` PDF-gradient lookup by index, NaN-free `normalize`, joint_stop /
  joint_sojourn refuse discrete sources.
- `9ae753c4` removal of the unused C function `ptd_graph_pdf_with_gradient`
  (+ `compute_pmf_with_gradient`, `alloc_2d`, `free_2d`, dead `kahan_init`,
  header declaration, its test pins and fixture) and its stale docs (mathref
  chapter 16 deleted; references in chapters 03, 14, 25, mathref index and
  _quarto.yml; docs/cpp_api/c_distributions.qmd; note in mathref
  PRODUCTION_PLAN.md). The two file deletions landed one commit early, in
  `43495114`. Related files after the removal: 65 files, 1300 passed,
  0 failed.
  Origin of the function (git history): added 2025-10-17 (`3d1080c2`) with a
  wrapper `ptd_graph_pdf_parameterized`, used by the generated C++ of the
  Channel-3 path `trace_to_log_likelihood(use_cpp=True)` (also exported to
  R); wrapper commented out 2025-11 (`a7134bc9`), path removed 2026-07-05
  (`4ee95276`), leaving this function with no callers.

**Test state.** Full suite on `33cd4fa5` (174 files, one at a time, 6 GB
cap): 2347 passed, 0 failed, 84 skipped, 26 xfailed. As on master, two files
exceed the caps and are run separately: `test_notebook_model_recovery.py`
(~26 min) and `test_svgd_divergence_robustness.py::test_grad_at_diverged_theta_is_finite_not_nan`
(7.5 GB). Separate run on `33cd4fa5`: recovery 19 passed, 4 xfailed; divergence
file 8 passed (peak 6.2 GB).

**Next steps, in order:**
1. Kasper merges: in `/Users/kmt/phasic`: `git switch master`,
   `git merge --ff-only refactor/discretize-cpp-ssot`, `git push origin
   master`; then `git worktree remove ../phasic-discretize` and, in the
   main checkout, `pixi run install-dev` (the main env still has master's
   build).
2. Update PROJECT-OVERVIEW.md on master at merge (the branch copy already
   carries the status, §1).

**Recorded, not done:** model θ layout for SVGD epoch routes; C++
StateIndexer and C++ `joint_prob_graph` (§8); `is_aux` in the C++
SerializedGraph and the SLURM deserialize path; constant edges in discrete
graphs; `joint_prob_graph(reward_rates_callback=...)` broken since 2026-05;
Stage-3 records to update (§9).
