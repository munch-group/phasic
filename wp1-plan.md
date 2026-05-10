# WP-1 — synthetic SCC graph constructor (detailed plan)

**Branch:** `hierar-elimin-cache`
**Status:** drafted, awaiting sign-off
**Predecessors:** Step 1 (toy fixture, commit `41e37bd`), Step 2
(pre-WP-1 experiments, commit `518460e`)
**Successor:** WP-2 (canonical hashing) — depends on WP-1's
synthetic graph being well-defined and stable.

## 1. Goal

Add a single C function and one C++ wrapper:

```c
struct ptd_graph *ptd_scc_build_synthetic_graph(
    const struct ptd_scc_graph *scc_graph,
    size_t scc_index,
    struct ptd_scc_synthetic_metadata *metadata_out);
```

Given a parent SCC decomposition and an SCC index, return a freshly
allocated `ptd_graph` that wraps the SCC's internal vertices with
a synthetic source vertex and a synthetic absorbing vertex, in the
canonical 5-part ordering. Populate `metadata_out` with the
vertex-category arrays needed by the composer (WP-5).

The returned graph must be a complete, self-contained
parameterised phase-type graph that
`ptd_graph_ex_absorbation_time_comp_graph_parameterized` can be
applied to without modification. WP-1 does **not** call the
eliminator — it only constructs the synthetic graph.

## 2. Why this WP first

Three reasons:

1. **Every other WP consumes its output.** WP-2 hashes the
   synthetic graph (so canonical ordering decisions land here);
   WP-3 serialises a PRC built from it (so the placeholder edges'
   format is settled here); WP-4/WP-5 read its metadata to
   compose. WP-1 is the foundation.
2. **It's the WP that exposes the most existing-code
   constraints.** Aux-vertex handling (Experiment 1b/4),
   mixed-mode edges (Experiment 4), `find_or_create_vertex` traps
   (§6.5 of the reference doc), Tarjan's reverse-topological
   output convention (§3.5) — all surface inside WP-1. Better to
   resolve them now under controlled conditions than to discover
   them mid-WP-5.
3. **It's testable in isolation.** The Toy-D variant is the
   regression anchor; we don't need a working composer to verify
   WP-1.

## 3. C struct definitions

Two new structs; both go in `api/c/phasic.h` near the existing
`ptd_scc_graph` declarations (around line 952).

```c
/**
 * Vertex categories within a synthetic SCC graph.
 *
 * The synthetic graph contains, in canonical 5-part order:
 *   index 0:                            synthetic source vertex
 *   indices [1..1+n_upstream]:          upstream-connecting vertices
 *   indices [...n_internal_only]:       pure-internal vertices
 *   indices [...n_downstream_connecting]: downstream-connecting vertices
 *   index N-1:                          synthetic absorbing vertex
 *
 * `parent_indices[k]` is the original-graph vertex index of the
 * synthetic-graph vertex at synthetic-graph index `k`. For the
 * synthetic source and absorbing vertices, `parent_indices[k]` is
 * SIZE_MAX (they have no original-graph counterpart).
 */
struct ptd_scc_synthetic_metadata {
    /* The source SCC's index in the parent's SCC decomposition.
     * Stored for cross-checks during composition. */
    size_t scc_index;

    /* Total number of vertices in the synthetic graph
     * (== n_upstream_connecting + n_internal_only +
     *    n_downstream_connecting + 2). */
    size_t n_vertices;

    /* Per-category counts. The synthetic source occupies index 0;
     * upstream-connecting occupy [1, 1 + n_upstream_connecting);
     * internal-only occupy
     *   [1 + n_upstream_connecting, 1 + n_upstream_connecting + n_internal_only);
     * downstream-connecting occupy
     *   [1 + n_upstream_connecting + n_internal_only,
     *    1 + n_upstream_connecting + n_internal_only + n_downstream_connecting);
     * the synthetic absorbing vertex occupies
     *   1 + n_upstream_connecting + n_internal_only + n_downstream_connecting. */
    size_t n_upstream_connecting;
    size_t n_internal_only;
    size_t n_downstream_connecting;

    /* Mapping from synthetic-graph vertex index to parent-graph
     * vertex index. Length == n_vertices. Sentinel SIZE_MAX for
     * synthetic source (index 0) and synthetic absorbing
     * (index n_vertices - 1). Owned by metadata; freed via
     * ptd_scc_synthetic_metadata_destroy. */
    size_t *parent_indices;

    /* Per-upstream-connecting vertex: list of (parent_vertex_idx,
     * parent_edge_idx) pairs identifying each parent-graph edge
     * that delivers mass into this upstream-connecting vertex
     * from outside the SCC. The composer (WP-5) uses this to wire
     * the synthetic source's placeholder edges to the parent's
     * actual external edge weights at composition time.
     *
     * `upstream_in_edges[k]` corresponds to the upstream-connecting
     * vertex at synthetic index 1 + k (k in [0, n_upstream_connecting)).
     * Each entry is a vector of (vertex, edge) pairs; multiple
     * pairs are possible when more than one external vertex feeds
     * the same upstream-connecting vertex. */
    struct ptd_scc_external_edge_ref **upstream_in_edges;
    size_t *upstream_in_edges_lengths;  /* length: n_upstream_connecting */

    /* Per-downstream-connecting vertex: list of (parent_vertex_idx,
     * parent_edge_idx) pairs identifying each parent-graph edge
     * that takes mass out of this downstream-connecting vertex to
     * outside the SCC. Same shape as upstream_in_edges, but for
     * outgoing edges. */
    struct ptd_scc_external_edge_ref **downstream_out_edges;
    size_t *downstream_out_edges_lengths;  /* length: n_downstream_connecting */
};

struct ptd_scc_external_edge_ref {
    size_t parent_vertex_idx;  /* original-graph vertex index */
    size_t parent_edge_idx;    /* index into that vertex's edges[] */
};

void ptd_scc_synthetic_metadata_destroy(
    struct ptd_scc_synthetic_metadata *metadata);
```

The metadata is verbose by design: every WP that follows will need
some subset of it, and getting it right once here is cheaper than
carrying multiple half-formed mappings forward.

## 4. The construction algorithm

### 4.1 High-level outline

```
1. Validate inputs (scc_graph != NULL, scc_index < n_sccs,
   metadata_out != NULL).
2. Identify vertex categories by scanning the parent graph's
   edges (§4.2 below).
3. Build the canonical synthetic-graph vertex order (§4.3).
4. Allocate the new ptd_graph (state_length and param_length
   from parent).
5. Create vertices in canonical order — using ptd_vertex_create
   to bypass the AVL tree, NEVER ptd_find_or_create_vertex.
6. Add edges (§4.4):
   - Synthetic source -> each upstream-connecting (placeholder
     coefficients).
   - Each internal vertex -> its in-SCC targets, copying the
     parent's coefficients_length and coefficients verbatim.
   - Each downstream-connecting -> synthetic absorbing
     (placeholder coefficients).
7. Populate metadata_out with category counts, parent_indices,
   and external edge references.
8. Return the new graph.
```

The construction is pure C — we never call into the C++ layer's
`add_edge` wrapper that locks edge mode (the cause of the
existing `as_graph()` Toy-D crash, per Experiment 1b).

### 4.2 Vertex categorisation

Five sets, computed from the parent graph and the SCC's internal
vertex pointer set:

| Set | Definition |
|---|---|
| `internal` | The SCC's internal vertices (`scc_vertex->internal_vertices`). |
| `upstream` | Parent vertices NOT in `internal` that have an outgoing edge whose target is in `internal`. |
| `upstream_connecting` | Vertices in `internal` that have at least one incoming edge from `upstream`. |
| `downstream_connecting` | Vertices in `internal` that have at least one outgoing edge whose target is NOT in `internal` (excluding aux-loop self-equivalents — see §4.5). |
| `internal_only` | `internal` minus (`upstream_connecting` ∪ `downstream_connecting`). |

`upstream` and the corresponding `downstream` (parent vertices
receiving from `internal`) are *not* vertices in the synthetic
graph; they are absorbed into the synthetic source and absorbing
vertex respectively. They appear in the metadata only via
`upstream_in_edges` and `downstream_out_edges` lookups.

Implementation: iterate parent vertices in two passes. First
pass builds `upstream`, `upstream_connecting`,
`downstream_connecting` as `bool[]` flags indexed by parent
vertex index, plus `external_edge_ref` lists. Second pass
computes `internal_only = internal \ (upstream_connecting ∪
downstream_connecting)`.

`internal` is identified by a `bool[]` flag indexed by parent
vertex index, populated from `scc_vertex->internal_vertices`.
Identity is by **vertex pointer** comparison (or equivalently,
parent vertex index — both are stable). Never by state.

### 4.3 Canonical synthetic-graph vertex order

Within each category, vertices are ordered by:

1. **Lexicographic state vector** (primary key).
2. **Sorted out-edge signature** (tiebreak): for each vertex,
   produce a canonical out-edge signature — a sorted list of
   `(target_category, target_within_category_position,
     coefficients_length, sorted_coefficients)`. Compare these
   lexicographically.
3. **Original parent vertex index** (last-resort tiebreak), with
   a `PTD_LOG_WARNING`.

Tier 1 alone suffices for graphs with unique state vectors
(verified in Experiment 1). Tier 2 handles aux vertices (Toy-D),
which all have all-zero state. Tier 3 is a safety net.

The ordering is a well-defined deterministic comparison function
suitable for `qsort`. Implementation: a static `qsort_r`-style
comparator that takes the parent graph as its `arg` so it can
read state vectors and edges.

### 4.4 Edge construction

Three edge types. Each is added via direct `realloc + struct
init`, mimicking `add_aux_vertex`'s manual construction path
(`src/cpp/phasiccpp.cpp:337–359`), so we can pass arbitrary
`coefficients_length` (including 0).

**Type A: synthetic source → upstream-connecting (placeholder).**

These are the cross-graph-reuse-invariant edges. Use a single
shared placeholder coefficient vector of length 1 with value 1.0.
Set `coefficients_length = 1` (NOT 0), so the eliminator's
multiplier-pointer scheme works (the placeholder coefficient
value is what gets multiplied into the elimination commands).

The actual parent coefficient values are not encoded here; they
live in the metadata's `upstream_in_edges` and are bound at
composition time via the `EXTERNAL` pointer kind (WP-3).

`should_free_coefficients = true` so the synthetic graph owns
the array.

**Type B: internal edges (verbatim copy).**

For every parent edge whose `from` and `to` are both in
`internal`, create a corresponding edge in the synthetic graph
with:
- `from` = synthetic vertex matching the parent's `from`.
- `to` = synthetic vertex matching the parent's `to`.
- `coefficients_length` = copied verbatim (may be 0 for aux
  edges, may be `param_length` for parameterised edges).
- `coefficients` = `malloc(...)` + `memcpy` from the parent edge
  (or `NULL` if `coefficients_length == 0`).
- `weight` = copied verbatim.
- `should_free_coefficients = true`.

This preserves the mixed-mode-edge behaviour exactly. Aux
vertices' constant in-SCC edges (which Experiment 4 confirmed
must remain `coefficients_length = 0` to be skipped by
`update_weights`) are copied unchanged.

**Type C: downstream-connecting → synthetic absorbing (placeholder).**

Same shape as Type A. Use the same length-1 placeholder
coefficient vector value `1.0`. Metadata's
`downstream_out_edges` records the parent-edge references for
later binding.

### 4.5 Self-loops and aux vertices

`ptd_graph_add_edge` rejects self-loops. Our manual edge
construction does too (we'll preserve this check via an explicit
`if (from == to) goto skip`). But aux vertices in
real-world graphs do *not* form self-loops with their parent —
they form a 2-cycle (`X → C` and `C → X` are distinct edges, not
a self-loop on `C`). So no special handling needed beyond the
self-loop guard.

Aux vertices' constant outgoing edge (`X → C`, weight 1.0,
coefficients_length 0) is just a Type B edge — copied verbatim.

### 4.6 The `param_length` and `state_length` of the synthetic graph

- `state_length` = parent's `state_length` (vertices keep their
  state vectors).
- `param_length` = parent's `param_length` (Type B edges keep
  their coefficient lengths).

Set via `ptd_graph_create(parent->state_length)` followed by
`ptd_graph_set_param_length(syn, parent->param_length)` *before*
adding any edges. This pre-locks the synthetic graph and avoids
any auto-detection from the first added edge.

Edge mode is set explicitly to `PTD_EDGE_MODE_PARAMETERIZED`
after creation (the synthetic graph is parameterised; placeholder
edges are array-style; Type B edges may be either, but the graph
mode is lockable to parameterised because the existence of *any*
parameterised edge is sufficient to lock it).

## 5. C++ wrapper

A new method on `phasic::SCCVertex`:

```cpp
class SCCVertex {
public:
    // ... existing ...

    /**
     * Build a synthetic-wrapped graph for this SCC.
     *
     * Unlike as_graph(), the result includes a synthetic source
     * vertex and a synthetic absorbing vertex with placeholder
     * edges, in the canonical 5-part vertex ordering. The
     * resulting graph can be passed directly to
     * ptd_graph_ex_absorbation_time_comp_graph_parameterized.
     *
     * @param metadata_out Receives vertex categorisation and
     *                     external-edge references. Caller owns;
     *                     must be destroyed with
     *                     ptd_scc_synthetic_metadata_destroy.
     * @return Newly owned synthetic graph.
     */
    Graph as_synthetic_graph(
        struct ptd_scc_synthetic_metadata **metadata_out) const;
};
```

The Pybind11 binding for `as_synthetic_graph` returns a
`std::tuple<Graph, py::capsule>` so Python tests can inspect the
metadata. The metadata struct is opaque from Python except via
helper accessors that read the category counts and per-vertex
parent index — also exposed as Pybind methods on a small
metadata wrapper class.

## 6. File layout

| Path | Purpose |
|---|---|
| `api/c/phasic.h` | `struct ptd_scc_synthetic_metadata` declaration; `ptd_scc_build_synthetic_graph` prototype; `ptd_scc_synthetic_metadata_destroy` prototype. |
| `src/c/scc_synthetic.c` | New file. Contains the full implementation. |
| `src/c/scc_synthetic_internal.h` | Header for cross-file helpers if needed (e.g. the comparator); private to `src/c`. |
| `api/cpp/scc_graph.h` | `SCCVertex::as_synthetic_graph` declaration. |
| `api/cpp/scc_graph.cpp` | Implementation: thin wrapper that calls the C function and wraps the result. |
| `src/cpp/phasic_pybind.cpp` | Pybind11 binding for `SCCVertex.as_synthetic_graph`; small wrapper class for the metadata. |
| `tests/pytest/test_synthetic_scc_graph.py` | New test file. Verifies WP-1 on all six toy variants. |

Minimal new files. The existing `api/cpp/scc_graph.cpp` already
has the pattern for translating `ptd_scc_graph` to `phasic::Graph`
on output.

## 7. Tests (pre-merge gate)

All in `tests/pytest/test_synthetic_scc_graph.py`. Each test
parametrises over the toy variants where applicable.

### 7.1 Structural tests

```python
def test_synthetic_graph_has_canonical_layout(toy_name, scc_idx):
    """Verify vertex count, categorisation counts, and
    synthetic source/absorbing positions match the canonical
    layout."""

def test_synthetic_graph_state_length_preserved(toy_name):
    """state_length and param_length match parent."""

def test_synthetic_graph_self_contained(toy_name, scc_idx):
    """No edges in the synthetic graph point outside the
    synthetic graph (no dangling pointers)."""

def test_synthetic_graph_no_state_collisions(toy_name, scc_idx):
    """Every non-starting non-absorbing vertex has a unique
    pointer (state collisions are allowed; we verify identity
    is preserved). Specifically targets Toy-D's aux vertex."""
```

### 7.2 Edge-handling tests

```python
def test_internal_edges_copied_verbatim(toy_name, scc_idx):
    """Each Type B edge has the same coefficients_length and
    coefficients values as the parent edge."""

def test_aux_edges_remain_constant(toy_d_specific):
    """The aux→parent edge in Toy-D's SCC2 has
    coefficients_length=0 in the synthetic graph too."""

def test_synthetic_source_and_absorbing_edges_are_placeholder(toy_name):
    """Type A and Type C edges have coefficients_length=1 and
    coefficient value 1.0."""
```

### 7.3 Metadata correctness

```python
def test_metadata_parent_indices_consistent(toy_name, scc_idx):
    """For every non-source non-absorbing synthetic vertex,
    metadata.parent_indices[k] points to a parent vertex with
    matching state and matching index in the SCC's internal
    vertices."""

def test_metadata_upstream_in_edges_complete(toy_name, scc_idx):
    """For each upstream-connecting vertex, the recorded
    external in-edges cover ALL parent edges that flow into
    that vertex from outside the SCC."""

def test_metadata_downstream_out_edges_complete(toy_name, scc_idx):
    """Symmetric: external out-edges from each
    downstream-connecting vertex are recorded fully."""
```

### 7.4 The smoke test (the most important single test)

```python
def test_synthetic_graph_is_eliminable(toy_name, scc_idx):
    """The most important WP-1 test: pass the synthetic graph
    through ptd_graph_ex_absorbation_time_comp_graph_parameterized
    and verify it returns a non-NULL PRC. Then call
    ptd_graph_build_ex_absorbation_time_comp_graph_parameterized
    and verify it produces a non-NULL reward_compute_graph.
    Then call expected_waiting_time and verify a finite
    non-NaN result vector.

    This does NOT verify numerical equivalence with the parent
    monolithic computation — that is WP-5's job. It only
    verifies the synthetic graph is structurally valid input
    for the existing eliminator."""
```

### 7.5 Toy-D-specific

```python
def test_toy_d_aux_vertex_in_synthetic_graph():
    """Toy-D's SCC₂ contains the aux vertex X (state [0,0]).
    The synthetic graph for that SCC must contain X as a
    distinct vertex (NOT merged with the synthetic source,
    which also has state [0,0]). Verify by counting vertices
    with all-zero state — should be exactly 2 (synthetic
    source + aux X), with distinct pointers."""

def test_toy_d_eliminable():
    """Toy-D's synthetic SCC₂ graph must pass the smoke test."""
```

### 7.6 Toy regression

The existing `test_toy_regression.py` continues to pass
unchanged. WP-1 introduces no behaviour change to public APIs.

### 7.7 Required pre-merge greps

A grep-based pre-merge check (manual, documented in the
working agreements):

```bash
# No state-based vertex lookups in WP-1 code.
grep -n "find_or_create_vertex\|ptd_find_or_create_vertex" \
    src/c/scc_synthetic.c src/c/scc_synthetic_internal.h \
    api/cpp/scc_graph.cpp:as_synthetic_graph

# No mode-locking add_edge calls (use direct ptd_graph_add_edge
# or manual struct init only).
grep -n "Vertex::add_edge\|Vertex::add_edge_parameterized" \
    src/c/scc_synthetic.c
```

Both must return zero hits.

## 8. Open questions and how they get resolved during the WP

These are the things I'll resolve while writing the code, not now:

1. **Should the placeholder coefficient be `1.0` or some
   sentinel?** Leaning towards `1.0` because it's neutral under
   the `weight = c · θ` formula and any conditioning analysis.
   Will confirm it doesn't trigger spurious warnings in
   `ptd_expected_waiting_time`'s conditioning pre-scan.
2. **What's the right error-reporting convention?** The C
   library uses `ptd_err` as a thread-local error buffer; the
   wrapper translates to Python exceptions. WP-1 follows the
   same convention.
3. **Where exactly does the `SCCVertex::as_synthetic_graph` Python
   binding return the metadata?** Either as a tuple or via an
   out-parameter object. Will pick the one that's cleanest in
   tests; the choice doesn't affect the C-level design.

## 9. Estimated scope

- C struct definitions and prototypes: ~50 LOC.
- `scc_synthetic.c` implementation: ~400 LOC, including the
  category-flagging passes, the canonical comparator, vertex
  construction, edge construction, and metadata population.
- C++ wrapper: ~30 LOC.
- Pybind binding and metadata wrapper: ~80 LOC.
- Tests: ~250 LOC across the function set above.

Total ~800 LOC, mostly in one new file. No existing code is
modified — WP-1 is purely additive. The existing `as_graph()`
remains, with its known Toy-D limitation, until WP-7 retires it
(or until a follow-up cleanup PR after the branch lands).

## 10. Out of scope for WP-1

- Actually eliminating the synthetic graph (the smoke test
  invokes the eliminator but the WP-1 deliverable does not own
  the elimination behaviour).
- Caching the per-SCC PRC. WP-4.
- Composing per-SCC PRCs. WP-5.
- Format extension for `EXTERNAL` pointers. WP-3.
- Modifying or deprecating the existing `SCCVertex::as_graph()`.
  Out of scope for the whole branch.
- Reading `~/.phasic_cache/`. WP-1 is a pure constructor; no I/O.
