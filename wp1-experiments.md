# Pre-WP-1 experiments — results

Three experiments to retire unknowns before drafting the WP-1 plan.
Results below; all run on commit ``41e37bd`` of branch
``hierar-elimin-cache``.

## Experiment 1 — SCC content-hash invariance

**Question:** does `ptd_graph_content_hash` produce the same hash
for a structurally-identical SCC extracted from two different parents?

**Result: YES, even stronger than expected.**

Tested on three parent variants (`Toy-C P`, `Toy-C P'`, plus a third
hand-built variant) with wildly different external coefficients into
and out of SCC₂ but identical SCC₂ internal coefficients. SCC₂'s
hash agreed across all three: `262f3cefa1f8d320b2f7b96e...`.

Additionally tested vertex-creation-order independence: a parent
that creates `D` before `C` (so parent vertex indices differ:
C=4, D=3) still yields the same SCC₂ hash. Reason: `as_graph()`
uses `find_or_create_vertex(state)` which routes through the AVL
tree and re-orders vertices by state vector during the SCC
extraction. The state-vector-driven ordering is deterministic.

**Implication for WP-1/WP-2:** the canonical-hashing problem (§6.2
of the reference doc) is *already solved* for graphs with unique
state vectors. The WP-2 work shrinks to: keeping that property
true under the synthetic-graph-with-source-and-absorbing wrapping,
plus handling the duplicate-state case (Toy-D — see Experiment 1b
below).

**Caveat (Experiment 1b — Toy-D):** SCC₂ with an aux vertex (all-zero
state, colliding with the starting vertex) **crashes `as_graph()`**:

```
RuntimeError: Cannot mix constant and parameterized edges. Graph
mode is CONSTANT (locked by first non-IPV edge using scalar syntax).
```

The crash root-causes to the aux→parent edge, which
`add_aux_vertex` creates with `coefficients_length = 0` (pure
constant) regardless of the parent graph's parameterised mode. When
`as_graph()` copies edges into the new graph, it sees the constant
edge, treats it as constant-mode, and then chokes on the next
parameterised edge.

`SCCVertex::hash()` therefore *also fails* on Toy-D's SCC₂ (it
calls `as_graph()` internally).

**Implication:** WP-1's synthetic-graph constructor must:
1. Preserve `coefficients_length = 0` aux→parent edges as
   genuinely constant (not coerce them to parameterised with all-zero
   coefficients, which would change semantics). The synthetic graph
   needs to support mixed mode, OR the synthetic-source/absorbing
   edges need to also be genuinely constant in line with the aux
   convention.
2. Use vertex pointer (or numerical index) lookup, never state
   lookup. The current `as_graph()`'s `find_or_create_vertex(state)`
   call would silently merge `X` (the aux) with the synthetic
   source vertex (also all-zero state by default).

## Experiment 2 — eliminator output addressability

**Question:** does `ptd_graph_ex_absorbation_time_comp_graph_parameterized`
record per-vertex absorption probabilities in addressable slots,
or does the composer need a small extension to expose them?

**Result: outputs are already addressable.** No eliminator
extension required.

Reading the eliminator end-to-end at `src/c/phasic.c:6981–7995`:

- The eliminator produces a `ptd_desc_reward_compute_parameterized`
  (the PRC) — a list of `ptd_comp_graph_parameterized` commands.
- Each command's `from` and `to` fields are **original-graph vertex
  indices**, set via `original_indices[]` (`phasic.c:7014, 7028,
  7196, 7427` and similar sites).
- The non-parameterised PRC produced by
  `ptd_graph_build_ex_absorbation_time_comp_graph_parameterized`
  (`phasic.c:8008–8113`) preserves these indices unchanged.
- The consumer `ptd_expected_waiting_time` (`phasic.c:8337–8384`)
  does `result[command.from] += result[command.to] * command.multiplier`
  where `result[]` is indexed by original-graph vertex index, so
  after replay `result[v]` holds "expected time to absorption
  starting from vertex `v`."

For composition: when an SCC's synthetic graph is eliminated, its
PRC produces a `result[]` array indexed by *synthetic-graph* vertex
index. `result[upstream_connecting_vertex_idx]` after replay holds
"expected time to absorb starting at that upstream-connecting
vertex" — which is exactly the value the parent's elimination needs
when mass arrives at that upstream-connecting vertex from outside.

**Implication for WP-5:** the "exit-slot propagation" worry in §4.5
of the reference doc was overstated. The composer's job is
metadata bookkeeping — translate per-SCC vertex indices to
parent-graph indices when splicing commands — not exposing new
numerical outputs from the eliminator. Composition becomes
substantially simpler.

## Experiment 3 — `SCCVertex::as_graph()` callers

**Question:** are there production callers of `as_graph()` whose
behaviour we'd disturb by modifying or augmenting it?

**Result: zero production callers.** Safe to modify in place or
add a sibling method.

Found callers (via `grep -rn` in `src/`, `tests/`, `api/`):

| Site | File | Status |
|---|---|---|
| Wrapper `_scc_vertex_as_graph` | `src/phasic/__init__.py:8891–8904` | Wraps result in Python `Graph`; not a real caller |
| Hierarchical trace pipeline | `src/phasic/hierarchical_trace_cache.py:134, 218, ...` | Deprecated; behind `Graph.compute_trace` |
| Test suite | `tests/pytest/inference/test_trace_stitching.py`, `test_scc_api.py` | Tests of the deprecated path + low-level API tests |

The Python wrapper at `__init__.py:8891` exists but does only
type-coercion. The only real production-code call sites are inside
`hierarchical_trace_cache.py`, and that module is itself behind
the deprecated `Graph.compute_trace` (which emits a
`DeprecationWarning`).

**Decision:** WP-1 will **add a new sibling method**
`SCCVertex::as_synthetic_graph()` rather than modify `as_graph()`.
Reasons:
1. The two methods produce different artefacts (strict-internal vs.
   synthetic-source-and-absorbing-wrapped). Conflating them would
   confuse callers.
2. Test fixtures rely on `as_graph()`'s current "vertices_length()
   in (scc.size(), scc.size() + 1)" contract
   (`test_scc_api.py:121`). Modifying it would change a
   user-observable contract.
3. The new method is a fresh entry point; we don't have to
   preserve any backward-compatibility constraint.

**Side observation:** `failing_tests.md:67` claims `scc.hash()` causes
a segfault. **It does not** in current branch state — verified on
multiple toy variants. The `failing_tests.md` note appears stale.
The hash function is safe to use as-is in WP-2.

## Updated plan implications

Pulling the three experiments together:

1. **WP-1's hardest sub-problem is the duplicate-state /
   mixed-mode-edge issue, not the categorisation algorithm.** The
   Python `_find_*` family ports cleanly to C; the existing
   `as_graph()` is a partial template; but the synthetic-graph
   constructor must handle aux vertices and their constant edges
   correctly, which the current `as_graph()` does *not* do.
2. **WP-2 (canonical hashing) shrinks substantially.** The core
   hash invariant already holds for unique-state graphs; we only
   need to preserve it under synthetic-graph wrapping and handle
   the duplicate-state case. The "graph isomorphism" anxiety in §6.2
   of the reference doc was unfounded for this codebase's actual
   workload.
3. **WP-5 (composition) shrinks too.** No eliminator extension; just
   careful per-command index translation. The §4.5 "exit-slot
   propagation" sub-problem mostly disappears.
4. **Aux edges must remain genuinely constant** (Experiment 4).
   The existing graph layer permits `coefficients_length = 0`
   edges to coexist with parameterised edges *internally* (the
   `edge_mode` field locks the user-facing API but the C struct
   tolerates the mixed configuration). `ptd_graph_update_weights`
   (`src/c/phasic.c:4232–4236`) explicitly **skips** edges with
   `coefficients_length == 0`, preserving their weight unchanged.
   This is the design contract for aux edges. WP-1's synthetic
   graph constructor must:
   - **Not** convert aux→parent edges to all-zero-coefficient
     parameterised (that would change `update_weights` behaviour:
     the weight would be re-computed as `0·θ[0]+...=0` instead of
     remaining 1.0).
   - Build synthetic graphs at the **C level**, calling
     `ptd_graph_add_edge` directly so we can pass
     `coefficients_length = 0` for the aux edges and full
     `coefficients_length = param_length` for the rest. The
     Python wrapper at `src/cpp/phasic_pybind.cpp:3506–3543`
     locks mode via `add_edge_parameterized` vs `add_edge`,
     which is why the existing `as_graph()` chokes — it is
     forced into one mode.

## Experiment 4 detail — aux→parent edge mechanics

Inspected `build_toy_d()` directly:

- The aux vertex `X` (state `[0, 0]`, index 6) has one outgoing
  edge: `X → C` with `weight = 1.0`.
- C's outgoing edges include `C → X` with `weight = 0.5` (the
  parameterised parent→aux rate, which `add_aux_vertex` set up
  using the supplied coefficient vector `[0, 0.5, 0, 0]`).
- Inspection of `ptd_graph_update_weights` (`src/c/phasic.c:4229–4236`)
  confirms aux→parent edges with `coefficients_length = 0` are
  skipped during weight updates.

The graph itself reports `parameterized() == True` and
`param_length == 4`, consistent with the rest of the toy. The
aux→parent constant edge is *internally* mixed-mode but does not
break the eliminator (`ptd_graph_ex_absorbation_time_comp_graph_parameterized`
treats every edge's `weight` field as a `double *multiplierptr`
regardless of whether `coefficients` is empty).

This means the existing eliminator already handles aux edges
correctly. The only piece that breaks is the *Python-level*
`as_graph()` because it routes through the user-facing add_edge
wrapper that locks mode. WP-1 sidesteps this by building synthetic
graphs in C.
