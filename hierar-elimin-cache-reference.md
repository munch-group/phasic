# Hierarchical SCC-cached parameterised reward compute graph — reference

This is the working reference for the `hierar-elimin-cache` branch. It
documents the conceptual algorithms involved, what already exists in the
C, C++, and Python layers, and what each work package needs to build.
It is intended to be read end-to-end before detailed planning, then
referenced piecewise during implementation.

## 1. What is being built and why

### 1.1 Goal

Replace the current monolithic O(n³) Gaussian elimination behind
`Graph.expectation` / `Graph.moments` / `Graph.variance` / `Graph.covariance`
(and any other call that goes through `ptd_expected_waiting_time`) with
an SCC-decomposed pipeline that:

1. Decomposes the graph into strongly connected components.
2. Eliminates each SCC independently, treating it as a self-contained
   parameterised phase-type graph (see §3 for the framing).
3. Caches each SCC's symbolic elimination on disk under its own content
   hash, so that two parent graphs sharing an SCC reuse the cached
   result.
4. Composes per-SCC results into a whole-graph
   `parameterized_reward_compute_graph` (PRC) and caches that too,
   under the parent graph's content hash.
5. Optionally distributes per-SCC elimination across cores or
   processes.

The end-state is a cache that is operationally identical to today's
Stage A2 from the consumer's point of view (`ptd_expected_waiting_time`
loads the parent PRC and replays it), but is built up from
recomposable, independently-shareable SCC artefacts.

### 1.2 Non-goals for this branch

- **Replacing the FFI / uniformization paths.** PMF / PDF / sojourn
  paths do not go through `ptd_precompute_reward_compute_graph` and are
  out of scope for this branch.
- **Changing public Python API.** All work happens behind
  `ptd_precompute_reward_compute_graph`. Users should observe a
  speed-up on first compute of large structured graphs and on repeat
  computes across related graphs, with no API changes.
- **Reviving the deprecated Python `EliminationTrace` pipeline.** That
  pipeline is the reference implementation for the algorithms but its
  runtime path will not be re-enabled. Some of its components
  (`hierarchical_trace_cache.py` helpers) will be ported or
  consulted, not wired back in.

### 1.3 The single goal: parallel SCC elimination *and* cross-graph SCC reuse

The two capabilities are not separable ambitions; they are facets of
one goal, delivered together by the same machinery:

- **Parallel SCC elimination:** decompose the parent graph, eliminate
  independent SCCs concurrently across cores within a process, compose
  the per-SCC results into a parent PRC.
- **Cross-graph SCC reuse:** the per-SCC PRCs produced during that
  elimination are content-hashed and persisted to disk. A second
  parent graph that contains a structurally-identical SCC reads the
  cached PRC instead of re-eliminating.

These are the same artefacts traversing the same pipeline — there is
no "Level 1" path that builds in-memory-only per-SCC results and a
separate "Level 2" path that adds disk persistence. Every per-SCC
elimination produces an on-disk PRC keyed by its content hash, and
every per-SCC compute starts with a disk-cache lookup. Within a
process this gives parallel speedup on the first compute of a large
structured graph; across processes / runs / users it gives the
cross-graph reuse that motivates the work for users building many
related models.

This unification is what justifies the design choices below — in
particular the synthetic-graph framing (§3) and the canonical SCC
hashing (§6.2). Both are required for cross-graph reuse and both are
free for parallel elimination, so we get parallel-and-reuse together
or neither.

## 2. Existing infrastructure inventory

This is what already lives in the repo on this branch. Each entry
includes file:line and a one-line characterisation of how complete
the component is.

### 2.1 Graph elimination — already SCC-aware

| Component | Location | Status |
|---|---|---|
| Tarjan's SCC algorithm | `src/c/phasic.c:1994–2070` (`strongconnect2`) | Live, production |
| `ptd_find_strongly_connected_components` | `src/c/phasic.c:2400+` (caller of `strongconnect2`); declared `api/c/phasic.h:974` | Live, production |
| `ptd_isolate_starting_vertex_scc` | `src/c/phasic.c:2118–2300+` | Live, runs after SCC discovery; ensures starting vertex is alone in SCC 0 |
| Topological sort of SCCs | `ptd_scc_graph_topological_sort` at `src/c/phasic.c:4554`, declared `api/c/phasic.h:976` | Live |
| Topological sort of vertices | `ptd_graph_topological_sort` at `src/c/phasic.c:4550`, declared `api/c/phasic.h:361` | Live |
| `ptd_graph_ex_absorbation_time_comp_graph_parameterized` (static order) | `src/c/phasic.c:6981–7995` | Live; outer loop already iterates SCCs in topo order |
| `_dyn` variant (min-degree within SCC) | `src/c/phasic.c:7473–7995` | Live but opt-in via `PHASIC_DYN_ORDERING` env var |
| `ptd_graph_build_ex_absorbation_time_comp_graph_parameterized` | `src/c/phasic.c:8008–8113` | Live; converts `_parameterized` PRC to non-parameterised PRC by replaying θ-dependent commands |

**Implication for design:** the C eliminator is already structured as
"for each SCC in topological order, eliminate that SCC's internal
vertices." The Level 1 refactor is therefore not adding SCC awareness —
it is exposing the existing per-SCC work as a first-class subroutine.

### 2.2 PRC artefact and Stage A2 disk cache

| Component | Location | Status |
|---|---|---|
| `struct ptd_desc_reward_compute_parameterized` | `api/c/phasic.h:456–461` | Live; container for symbolic command list |
| `struct ptd_comp_graph_parameterized` (one command) | `api/c/phasic.h:446–454` | Live; six command types: `NEW_ADD/P/PP/INV/ZERO/ONE_MINUS/DIVIDE` |
| `ptd_save_parameterized_reward_compute_graph` | `src/c/phasic.c:3232+`; declared `api/c/phasic.h:493` | Live; atomic write-then-rename |
| `ptd_load_parameterized_reward_compute_graph` | `src/c/phasic.c:3495+`; declared `api/c/phasic.h:517` | Live; reconstructs pointers against supplied graph |
| `ptd_pcg_build_cache_path` | `src/c/phasic.c:3181+` | Live; uses `ptd_graph_content_hash` as filename |
| Cache disable env var (`PHASIC_DISABLE_CACHE`) | `src/c/phasic.c:3170+` (`ptd_pcg_cache_disabled`) | Live |
| Header / pointer encoding format | `src/c/phasic.c:2855–3100+` (`PTDPRMC1`, `PTD_PCG_PTR_*`, `ptd_pcg_disk_*`) | Live; v1 format |
| Build-and-save integration in `ptd_precompute_reward_compute_graph` | `src/c/phasic.c:1858–1985` | Live; load-on-miss-then-save pattern |
| `ptd_graph_build_ex_absorbation_time_comp_graph_parameterized` (replay layer) | `src/c/phasic.c:8008+` | Live; replays the θ-dependent commands into a θ-independent reward compute graph |

**Implication for design:** the on-disk format is already designed to
be relocatable (the encoder records edge pointers as
`(vertex_idx, edge_idx, byte_offset)`, the decoder resolves against a
supplied graph). The format supports two pointer kinds: `MEM` (offsets
into the flat scratch buffer) and `EDGE` (references to live
`edge->weight` slots).

### 2.3 Content hashing

| Component | Location | Status |
|---|---|---|
| `ptd_graph_content_hash` | `src/c/phasic_hash.c:216+`; declared `api/c/phasic.h` (search `phasic_hash.h`) | Live; SHA-256 over graph metadata + per-vertex (state, sorted edges) hashes |
| `compare_edges` (canonical edge ordering) | `src/c/phasic_hash.c:147–166` | Live; sorts edges by target index, then by coefficient length, then by coefficient values lexicographically |
| `hash_vertex_structure` | `src/c/phasic_hash.c:169+` | Live; hashes state vector, then sorted-canonical edges |
| `ptd_hash_result` struct | `api/c/phasic_hash.h` | Live; carries 256-bit hash, 64-bit prefix, hex string |

**Important caveat.** `ptd_graph_content_hash` walks vertices in their
**numerical index order** (`for (size_t i = 0; i < graph->vertices_length; i++)`)
and uses **`edge->to->index`** in the per-edge hash. So the hash is
**not invariant under vertex relabelling** — two graphs that are
isomorphic but use different vertex numberings hash differently.

This matters for SCC hashing because the same logical SCC extracted
from different parent graphs will have different vertex numberings.
See §6.2 for the SCC-canonicalisation strategy.

### 2.4 SCC subgraph extraction (C++ layer)

| Component | Location | Status |
|---|---|---|
| `phasic::SCCGraph` class | `api/cpp/scc_graph.h:34–111`, impl `api/cpp/scc_graph.cpp:14–116` | Live |
| `phasic::SCCVertex` class | `api/cpp/scc_graph.h:120–182`, impl `api/cpp/scc_graph.cpp:120+` | Live |
| `SCCVertex::as_graph()` | `api/cpp/scc_graph.cpp:138–222` | Live; **but only copies internal edges — no synthetic source/absorbing wrapping**, see §3.2 |
| `SCCVertex::hash()` | `api/cpp/scc_graph.cpp:236–250` | Live; hashes `as_graph()` output via `ptd_graph_content_hash` |
| `SCCVertex::internal_vertex_indices()` | `api/cpp/scc_graph.cpp:224–234` | Live |
| `SCCVertex::outgoing_scc_edges()` | `api/cpp/scc_graph.cpp:252–262` | Live |
| `SCCGraph::sccs_in_topo_order()` | `api/cpp/scc_graph.cpp:70–81` | Live |
| Pybind11 bindings | `src/cpp/phasic_pybind.cpp:3062–3110` | Live |

**Implication for design:** the wrapper class is in place but
`SCCVertex::as_graph()` produces a *strict* SCC subgraph (only
internal edges, no synthetic interface). Work package WP-2 modifies or
augments this to produce the synthetic-wrapped version that
`ptd_graph_ex_absorbation_time_comp_graph_parameterized` can be
applied to as if it were a standalone graph.

### 2.5 Python `hierarchical_trace_cache.py` (algorithm reference, deprecated)

The Python pipeline is the reference implementation for SCC-based
elimination orchestration. It is no longer part of any production
path (see `compute-paths.md`), but its algorithms are correct and the
implementation is well-tested. **Treat it as the algorithmic spec.**

| Component | Location | What it solves |
|---|---|---|
| `_find_upstream_vertices` | `src/phasic/hierarchical_trace_cache.py:880–925` | Identifies vertices outside the SCC that have edges *into* it |
| `_find_upstream_connecting` | `src/phasic/hierarchical_trace_cache.py:928–966` | Internal vertices that *receive* edges from upstream |
| `_find_downstream_connecting` | `src/phasic/hierarchical_trace_cache.py:969+` | Internal vertices that have edges *out of* the SCC |
| `_find_downstream_vertices` | `src/phasic/hierarchical_trace_cache.py:1007+` | Vertices outside the SCC that *receive* edges from it |
| `_build_scc_subgraph` | `src/phasic/hierarchical_trace_cache.py:1047–1233` | Builds the synthetic-wrapped subgraph for non-first SCCs with the canonical 5-part vertex ordering |
| `_build_first_scc_subgraph` | `src/phasic/hierarchical_trace_cache.py:1236+` | Special case: first SCC, contains the original starting vertex |
| `_identify_trace_vertices` | `src/phasic/hierarchical_trace_cache.py:1351+` | Maps trace-vertex indices back to original-graph indices |
| `_remap_operation` | `src/phasic/hierarchical_trace_cache.py:1502+` | Renumbers operations during stitching |
| `record_enhanced_scc_traces` | `src/phasic/hierarchical_trace_cache.py:1565–1616` | Per-SCC trace recording orchestrator |
| `stitch_scc_traces` | `src/phasic/hierarchical_trace_cache.py:1619+` | Sister-vertex merging during composition |
| `get_trace_hierarchical` | `src/phasic/hierarchical_trace_cache.py:2024+` | Main entry point, two-level cache lookup |
| `collect_missing_traces_batch` | `src/phasic/hierarchical_trace_cache.py:161+` | Walks the SCC tree, collects only the work units that need computing (cache-miss-driven) |
| `compute_missing_traces_parallel` | `src/phasic/hierarchical_trace_cache.py:565+` | vmap / pmap / sequential strategies |

The 5-part vertex ordering used by `_build_scc_subgraph`
(`{upstream, upstream-connecting, internal-only, downstream-connecting, downstream}`)
is the canonical layout that makes within-SCC elimination produce a
trace whose composition with sibling traces is straightforward. The
ordering is documented at `src/phasic/hierarchical_trace_cache.py:1058–1066`.

### 2.6 Components removed from production but useful as reference

| Component | Location | Note |
|---|---|---|
| `ptd_record_elimination_trace` (C version) | `src/c/phasic.c:12555–12760+` (commented out) | Original C trace recorder; commented out due to memory blow-up |
| `src/c/trace/trace_cache.c` | full file | Dormant; no live caller |
| `src/c/trace/trace_internal.h` | full file | Dormant |
| `ptd_load_trace_from_cache` / `ptd_save_trace_to_cache` | `src/c/phasic.c:988–1108` | Live functions but no producer |
| Python `EliminationTrace` evaluator | `src/phasic/trace_elimination.py` | Used by `Graph.compute_trace` (deprecated) |
| `instantiate_from_trace` | `src/phasic/trace_elimination.py:1324–1423` | Reference for "how a trace becomes a graph" |

## 3. The "SCC as a graph in its own right" framing

This is the conceptual core of the design. Reading this section
correctly is the difference between a clean implementation and a
nest of special cases.

### 3.1 Definition

For a parent graph `G` with SCC decomposition `{S₁, S₂, ..., Sₖ}` in
topological order, each SCC `Sᵢ` is augmented to form a self-contained
parameterised phase-type graph `Sᵢ′` as follows:

1. **Internal vertices**: all vertices of `Sᵢ` are copied verbatim
   with their state vectors. Their internal edges (edges to other
   vertices of `Sᵢ`) are copied with their full coefficient vectors.
2. **Synthetic source vertex**: a new starting vertex `src(Sᵢ′)` is
   added at index 0. It has one edge to each *upstream-connecting*
   vertex of `Sᵢ` (vertices that receive edges from outside `Sᵢ`).
   Edge weights are placeholders (see §3.3).
3. **Synthetic absorbing vertex**: a new absorbing vertex `abs(Sᵢ′)`
   is added at the highest index. Each *downstream-connecting* vertex
   of `Sᵢ` (those that have edges outside `Sᵢ`) gets one edge to
   `abs(Sᵢ′)`, again with placeholder weights.
4. **Vertex ordering**: vertices are arranged in the canonical 5-part
   order `{src, upstream-connecting, internal-only,
   downstream-connecting, abs}`. This ordering is the contract — it
   is what makes within-SCC elimination produce a result that
   composes correctly.

The result is a complete `ptd_graph` that
`ptd_graph_ex_absorbation_time_comp_graph_parameterized` can be
applied to without any modification, producing a PRC `.bin` that
encodes the symbolic elimination of *just this SCC*.

### 3.2 Difference from current `SCCVertex::as_graph()`

The existing `SCCVertex::as_graph()` (`api/cpp/scc_graph.cpp:138–222`)
produces step 1 only — it copies internal vertices and internal
edges, creates the new graph's starting vertex (which is *not* a
synthetic source for the SCC, it's whatever the original starting
vertex was if it happens to be in this SCC, otherwise an unused
default), and stops.

It is missing:

- The synthetic source vertex with edges to upstream-connecting
  internal vertices.
- The synthetic absorbing vertex.
- Edges from downstream-connecting internal vertices to the synthetic
  absorbing vertex.
- The canonical 5-part vertex ordering.

WP-2 either replaces `as_graph()` or adds a sibling method (proposed:
`as_synthetic_graph()`) that produces a fully-wrapped graph.

### 3.3 The placeholder-weight problem

This is the subtlest part of the framing. The synthetic source's
edges into the SCC, and the SCC's edges out to the synthetic absorbing
vertex, *come from the parent graph's external edges*. Two parents
sharing the same SCC structurally will typically have *different
external edge weights* (different coefficients into the SCC, different
coefficients out).

**If we encode the actual external coefficients into the synthetic
graph, the SCC's content hash depends on the parent and we get no
cross-graph reuse.** The whole point of caching SCCs is that two
parents with the same SCC topology hit the same cache entry.

The fix: **the synthetic graph encodes only the SCC's intrinsic
structure**. Source-edge and absorbing-edge weights are recorded as
*symbolic placeholders* — the cached SCC PRC has slots for "external
input from upstream-connecting vertex k" and "external output through
downstream-connecting vertex j", to be bound at composition time.

This requires a small, carefully-scoped extension to the PRC format
(WP-3): a new pointer kind, e.g. `PTD_PCG_PTR_EXTERNAL`, that points
into a per-SCC external-input table rather than into the live graph
or the scratch mem buffer. The table is resolved at composition time
by writing the appropriate parent-supplied values into it.

### 3.4 Why this framing is clean

- **One file format.** SCC-level PRCs and parent-level PRCs use the
  same `.bin` layout (header + commands + mem buffer), differing only
  in whether the `EXTERNAL` pointer kind is present. Same writer, same
  reader, same atomicity, same versioning.
- **Hash invariance.** If the SCC's synthetic graph is *constructed
  in a canonical way* (canonical vertex ordering, canonical edge
  ordering — see §6.2), then the same SCC structure produces the
  same `ptd_graph_content_hash` regardless of parent. Cross-graph
  reuse just works.
- **Recursion is free.** Nothing in the design assumes that the
  "SCC" being processed has no internal SCC structure. If a
  per-SCC graph happens to be large enough to have non-trivial
  SCC decomposition itself, the same machinery applies recursively.
- **One eliminator.** No special-case "per-SCC eliminator" — the
  existing `ptd_graph_ex_absorbation_time_comp_graph_parameterized`
  works as-is, given a synthetic-wrapped SCC graph.

### 3.5 Composition

Composition takes per-SCC PRCs and produces a parent PRC. Two
correctness invariants:

1. **SCCs compose in reverse-topological (sink-first) order.** This
   is a *correction* of an earlier claim in this doc. The C
   eliminator processes SCCs in the order Tarjan's algorithm
   produces them — which is **reverse topological**, i.e. sinks
   first. The C++ accessor `SCCGraph::sccs_in_topo_order()`
   (`api/cpp/scc_graph.cpp:70–81`) returns the C order verbatim
   (its name is historical and misleading; the comment in source
   even says "topological order" but the actual content is the
   reverse). Verified empirically on toy-base
   (2026-05-10): the SCC ordering returned was `[{s}, {C,D},
   {A,B}, {Ω}]`, where `{A,B} → {C,D}` in the condensation. The
   composer must therefore walk SCCs in this same reverse-topo
   order to match the eliminator's behaviour. Composition
   eliminates sink SCCs first, so by the time we reach an
   upstream SCC, downstream SCCs' absorption-probability
   expressions are already encoded in their PRCs as inputs that
   the upstream SCC's elimination references via `EXTERNAL`
   pointers. This is the same logic as Gaussian elimination on a
   DAG: solve leaves first, then back-substitute.
2. **Within-SCC elimination order does not affect correctness.** Any
   ordering of vertex elimination inside one SCC produces a
   mathematically equivalent symbolic result. This is why the
   `_dyn` (min-degree) variant is allowed to coexist with the static
   variant — they differ in command sequence but agree numerically.

**Renaming `sccs_in_topo_order` is out of scope** for this branch
to avoid touching unrelated callers. Treat the misnomer as a known
cosmetic issue; documentation in this file always specifies which
direction is meant.

Composition mechanically:

1. Sort SCCs in topological order.
2. For each SCC `Sᵢ`:
   1. Look up `Sᵢ`'s cached PRC (or compute and cache if missing).
   2. Allocate the SCC's external-input table.
   3. Bind the table's entries to the parent graph's actual external
      coefficient pointers (`&edge->coefficients[k]` for the parent's
      θ-dependent weight expression).
   4. Append `Sᵢ`'s commands to the parent PRC, with all `MEM`
      pointer offsets translated to the parent PRC's mem buffer
      layout, and `EXTERNAL` pointers resolved to either the parent's
      live edge slots or to mem slots populated by upstream SCCs'
      absorbing-side outputs.
3. Save the parent PRC to disk under the parent's content hash.

The translation in step 2.iv is mostly mechanical bookkeeping. It is
the most error-prone part of the implementation and warrants careful
test coverage (WP-7).

## 4. Algorithmic spec for each work package

This section gives the precise algorithm for each work package. Each
package is sized to be independently verifiable: it can be merged once
its tests pass without depending on later packages being complete.

### 4.1 WP-1: SCC synthetic-graph constructor (C-level helper)

**Goal:** A single C function
`ptd_scc_build_synthetic_graph(scc_graph, scc_index)` that returns a
fully-wrapped `ptd_graph` ready for elimination.

**Signature (proposed):**

```c
struct ptd_graph *ptd_scc_build_synthetic_graph(
    const struct ptd_scc_graph *scc_graph,
    size_t scc_index,
    struct ptd_scc_synthetic_metadata *metadata_out);
```

`metadata_out` carries the vertex-category arrays needed by the
composer (`upstream_connecting`, `downstream_connecting`,
`vertex_map`).

**Algorithm:**

1. Identify vertex categories using the same logic as the Python
   helpers (`_find_upstream_vertices`,
   `_find_upstream_connecting`, `_find_downstream_connecting`,
   `_find_downstream_vertices`):
   - `upstream`: vertices outside `Sᵢ` with at least one edge into
     an internal vertex.
   - `upstream_connecting`: internal vertices receiving from
     `upstream`.
   - `internal_only`: internal vertices not in any "connecting" set.
   - `downstream_connecting`: internal vertices with at least one
     edge to a vertex outside `Sᵢ`.
   - `downstream`: vertices outside `Sᵢ` receiving from
     `downstream_connecting`.

   For the synthetic graph, only `upstream_connecting`,
   `internal_only`, `downstream_connecting` matter for vertices;
   `upstream` and `downstream` are *not* vertices in the synthetic
   graph — they're folded into the synthetic source and absorbing
   vertex respectively.
2. Create a new `ptd_graph` with the parent's `state_length` and
   `param_length`.
3. Add vertices in canonical order:
   1. Synthetic source vertex (the new graph's `starting_vertex`).
   2. `upstream_connecting` vertices, in the canonical SCC-internal
      order (see §6.2 for what "canonical" means here).
   3. `internal_only` vertices, in canonical order.
   4. `downstream_connecting` vertices, in canonical order.
   5. Synthetic absorbing vertex.
4. Add edges:
   1. From synthetic source to each `upstream_connecting` vertex,
      with placeholder coefficients (see §6.3 for the placeholder
      scheme).
   2. From each internal vertex (any of the three internal categories)
      to its targets within the SCC, copying the parent's
      coefficients verbatim.
   3. From each `downstream_connecting` vertex to the synthetic
      absorbing vertex, with placeholder coefficients.
5. Return the synthetic graph and populate `metadata_out` with arrays
   that map synthetic-graph vertex indices back to parent-graph
   vertex indices.

**References:**
- Python equivalents: `_build_scc_subgraph` and `_build_first_scc_subgraph`
  in `src/phasic/hierarchical_trace_cache.py:1047–1350`. Algorithm
  is identical; just port to C.
- Existing `SCCVertex::as_graph()` in `api/cpp/scc_graph.cpp:138–222`
  is a good starting point for the vertex-and-edge-copy plumbing
  (the `vertex_map` pattern, the parameterised vs non-parameterised
  edge handling).
- The categorisation helpers (`_find_*` family in
  `hierarchical_trace_cache.py:880–1046`) are five short pure
  functions that translate near-verbatim to C.

**Tests:** verify on a hand-constructed graph with two SCCs that
the synthetic graph for each SCC has the correct vertex count,
correct edge structure, and that the categorisation arrays in
`metadata_out` agree with what the Python equivalents produce on
the same parent.

### 4.2 WP-2: SCC content-hash invariance

**Goal:** Make `ptd_graph_content_hash` of two synthetic SCC graphs
agree if and only if their underlying SCCs are structurally equivalent.

**Problem:** as documented in §2.3, `ptd_graph_content_hash` walks
vertices in `index` order. Two parents that contain the same logical
SCC will assign different vertex indices to its members. The
synthetic-graph constructor (WP-1) must therefore order vertices
deterministically *based on SCC-intrinsic properties*, not on parent
indices.

**Algorithm:** within each canonical category
(`upstream_connecting`, `internal_only`, `downstream_connecting`),
order vertices by:

1. **Primary:** lexicographic order of the state vector
   (`vertex->state[0..state_length-1]`).
2. **Secondary (tie-break):** if state vectors are equal, compare
   sorted out-edge signatures: list of
   `(target_canonical_position_within_scc, sorted_coefficients)`
   tuples. (Inter-SCC edges resolve to "synthetic-source" or
   "synthetic-absorbing" tokens.)
3. **Tertiary (last resort):** if both above are tied, the SCC has a
   non-trivial automorphism. Emit a warning. Fall back to original
   parent index. Document the corner case.

For phasic's typical workloads (state-vector-driven coalescent /
queuing models), the primary key alone is almost always sufficient
because state vectors are unique and meaningful.

**Tests:**
- Build two parent graphs that share an SCC structurally (different
  parent topologies, same internal SCC structure). Assert that
  `synthetic_graph_for(scc_in_parent_A).content_hash() ==
  synthetic_graph_for(scc_in_parent_B).content_hash()`.
- Build a parent graph with a small symmetry (e.g. two
  interchangeable internal vertices). Confirm the warning fires; do
  not block on it but document.

### 4.3 WP-3: PRC format extension — `EXTERNAL` pointer kind

**Goal:** Extend the `.bin` format so that a saved SCC PRC can encode
"this command's `multiplierptr` / `fromT` / `toT` is an external
input to be bound at composition time."

**Format change:**

Add a new value to the `enum ptd_pcg_ptr_kind` in
`src/c/phasic.c:2881+`:

```c
enum ptd_pcg_ptr_kind {
    PTD_PCG_PTR_NULL = 0,
    PTD_PCG_PTR_MEM = 1,
    PTD_PCG_PTR_EDGE = 2,
    PTD_PCG_PTR_EXTERNAL = 3,  // new
};
```

`PTD_PCG_PTR_EXTERNAL` payload: a `uint32_t external_table_index`
(reuses the existing `vertex_idx` slot in `struct ptd_pcg_disk_ptr`).

Bump `PTD_PCG_FORMAT_REVISION` from 1 to 2. Files written with
revision 1 remain readable; the loader handles both
revisions in the obvious way (revision 1 files cannot contain
`EXTERNAL` pointers, so no fallback logic is needed).

**Encoding semantics:**

When saving a synthetic-wrapped SCC graph's PRC, any pointer that
references the placeholder coefficients (the source-edge or
absorbing-edge coefficients in the synthetic graph) is encoded as
`EXTERNAL` with a unique table index per placeholder slot. The save
function needs to know which edges are placeholders — pass an
auxiliary `external_anchors` array to
`ptd_save_parameterized_reward_compute_graph` describing which
`(vertex_idx, edge_idx)` pairs in the synthetic graph correspond to
external slots.

**Loading semantics:**

`ptd_load_parameterized_reward_compute_graph` accepts an additional
parameter `external_table` (a `double *` array indexed by the table
index). At pointer-resolution time, `EXTERNAL` pointers are resolved
to `&external_table[index]`. The composer (WP-5) populates this table
before triggering the replay.

**Implementation files:**

- `src/c/phasic.c:2881–2920` — extend the enum and disk_ptr layout.
- `src/c/phasic.c:2940–3050+` — extend `ptd_pcg_encode_ptr` to handle
  the new kind.
- `src/c/phasic.c:3500+` — extend the loader. A new
  signature variant
  `ptd_load_parameterized_reward_compute_graph_ex(path, graph, external_table)`
  is preferable to changing the existing function's signature.
- `api/c/phasic.h:493+` — declare the new save/load variants.

**Tests:**
- Round-trip: save a hand-constructed PRC with `EXTERNAL` pointers,
  load it with a populated table, replay, verify the result agrees
  with a non-cached direct-memory equivalent.
- Format-revision compatibility: load an existing revision-1
  cache file, verify it still works.

### 4.4 WP-4: Per-SCC PRC computation and cache lookup

**Goal:** A function that, given an SCC, returns its PRC, hitting the
disk cache when possible and computing-then-saving on miss.

**Signature (proposed):**

```c
struct ptd_desc_reward_compute_parameterized *
ptd_scc_get_or_compute_prc(
    const struct ptd_scc_graph *scc_graph,
    size_t scc_index,
    struct ptd_scc_synthetic_metadata *metadata_out);
```

**Algorithm:**

1. Build the synthetic graph via `ptd_scc_build_synthetic_graph`
   (WP-1), populating `metadata_out`.
2. Compute its content hash using `ptd_graph_content_hash`.
3. Build the SCC cache file path:
   `~/.phasic_cache/parameterized_reward_compute/scc_<hash_hex>.bin`
   (the `scc_` prefix distinguishes SCC-level entries from
   parent-level entries; both can live in the same directory).
4. Try to load. On hit: return the loaded PRC. The metadata is
   regenerated from the parent graph (it's parent-dependent and not
   in the cache).
5. On miss: run
   `ptd_graph_ex_absorbation_time_comp_graph_parameterized` on the
   synthetic graph. Save the result to the cache via the new
   `_ex` variant of the save function (WP-3) with the appropriate
   `external_anchors` array. Return the result.
6. Destroy the synthetic graph (it was a temporary).

**Threading:** at this stage, all SCCs are computed sequentially.
Parallelism is WP-6.

**References:**
- The current `ptd_precompute_reward_compute_graph` at
  `src/c/phasic.c:1858–1985` is the structural model — same
  load-on-miss-then-save pattern, just per-SCC instead of
  per-parent-graph.

**Tests:**
- Cold cache: first call builds and saves. Verify file exists.
- Warm cache: second call loads. Verify by stat'ing file mtime.
- Invalidation: edit the synthetic graph (or change the parent),
  verify a different cache key is computed.

### 4.5 WP-5: Composition — assembling parent PRC from per-SCC PRCs

**Goal:** Walk the SCC condensation in topological order, splice
per-SCC PRCs into a single parent PRC, resolve `EXTERNAL` pointers
against the parent's actual edge weights and against scratch slots
populated by upstream SCCs.

**Signature (proposed):**

```c
struct ptd_desc_reward_compute_parameterized *
ptd_compose_scc_prcs(
    const struct ptd_scc_graph *scc_graph,
    struct ptd_desc_reward_compute_parameterized **per_scc_prcs,
    struct ptd_scc_synthetic_metadata **per_scc_metadata,
    size_t n_sccs);
```

**Algorithm:**

This is the most algorithmically delicate WP. Pseudo-code (note:
"topological order" here means *reverse topological* / sink-first,
matching what `SCCGraph::sccs_in_topo_order()` actually returns —
see §3.5 correction):

```
allocate parent_commands (initially empty), parent_mem_chain (empty)
allocate inter_scc_table — a per-SCC array of "this SCC's exit slots" pointers
for scc_i in reverse-topological (sink-first) order:
    prc_i, meta_i = per_scc_prcs[i], per_scc_metadata[i]
    allocate external_table_i (size = number of EXTERNAL slots in prc_i)
    for each EXTERNAL slot k in prc_i:
        if slot is an upstream-input slot (source-edge in synthetic graph):
            # the upstream SCC's exit slot drives this input
            external_table_i[k] = inter_scc_table[upstream_scc][upstream_slot]
            # may need a scratch mem slot if propagation is multi-hop;
            # see "exit-slot propagation" below
        elif slot is a parent-edge input slot (the SCC's source edges
             actually correspond to parent's external edge weights from
             vertices in OTHER SCCs to upstream-connecting):
            external_table_i[k] = &parent_graph->vertices[v]->edges[e]->coefficients[c]
    for each command in prc_i:
        translate command pointer slots:
            MEM offsets: shift by current parent_mem_chain offset; reserve a
                         block in parent_mem_chain for prc_i's mem
            EDGE refs: rewrite to refer to the corresponding edge in the
                       PARENT graph (synthetic-graph edge indices map back
                       to parent-graph edges via meta_i)
            EXTERNAL refs: resolve via external_table_i
        append translated command to parent_commands
    record this SCC's exit-slot pointers in inter_scc_table[i]
return parent PRC consisting of parent_commands and parent_mem_chain
```

**The hard part: exit-slot propagation.** When SCC `i`'s elimination
absorbs mass into its synthetic absorbing vertex, the C eliminator
records a sequence of commands that compute "the probability of
absorption at synthetic-absorb given starting at synthetic-source."
For the parent, this becomes "the probability mass routed through
SCC `i` from its upstream interface to its downstream interface."
Each downstream-connecting vertex in the SCC is a separate output
channel — the eliminator produces multiple absorption-probability
expressions, one per downstream-connecting vertex.

These expressions live in mem-buffer slots in `prc_i`. The composer
needs to know which mem-buffer slots correspond to which
downstream-connecting vertices, so it can wire them as inputs to
downstream SCCs' EXTERNAL slots.

This wiring information must therefore be recorded by
`ptd_graph_ex_absorbation_time_comp_graph_parameterized` for the
synthetic graph — specifically: "after replay, mem-slot offsets
`[o_1, o_2, ..., o_k]` hold the downstream-output values, in the
order corresponding to the synthetic graph's
downstream-connecting vertices."

This is a small extension to the eliminator's output: in addition to
the PRC, return an `output_offsets[]` array. Add this to
`struct ptd_desc_reward_compute_parameterized` or to a sibling
struct.

**References:**
- Python sister-vertex merging:
  `stitch_scc_traces` at `hierarchical_trace_cache.py:1619+`. The
  algorithm is the same shape — sort SCCs in topo order, splice
  command sequences, resolve cross-SCC dependencies.
- `_remap_operation` at
  `hierarchical_trace_cache.py:1502+` is the Python equivalent of
  the per-command pointer translation step.

**Tests:**
- Unit: compose two trivially-constructed SCC PRCs (two-vertex
  SCCs each), verify the composed PRC produces the same result as
  monolithic elimination of the four-vertex parent.
- Property: for ten random parameterised graphs, assert that
  composed-PRC and monolithic-PRC give numerically equal
  `expected_waiting_time` results to ≤ 1e-12 across many random θ.
  This is the canonical correctness test for the entire branch.

### 4.6 WP-6: Parallelism — per-SCC eliminations on a thread pool

**Goal:** Compute independent SCCs in parallel.

**Algorithm:**

1. Build the SCC condensation DAG.
2. Identify "level sets": SCCs at the same topological depth (no
   path between them in the condensation) can be eliminated
   concurrently.
3. For each level set, dispatch all SCCs in the set to OpenMP
   threads via `#pragma omp parallel for`. Each thread invokes
   `ptd_scc_get_or_compute_prc` (WP-4).
4. Synchronise at the level boundary. Move to the next level.
5. Composition (WP-5) remains sequential — it's fast compared to
   elimination, and parallelising it doesn't pay.

**Caveat: cache-write contention.** Two threads computing different
SCCs may try to write different cache files concurrently. The atomic
write-then-rename pattern in `ptd_save_parameterized_reward_compute_graph`
already handles this correctly — different filenames don't conflict.
Two threads computing the *same* SCC (which can happen if both are
in different level sets working on different parents that share an
SCC in this batch) will race on writing the same file; the
atomic-rename pattern means the loser's write is wasted but
correctness is preserved.

**References:**
- Existing OpenMP usage in
  `src/cpp/parameterized/graph_builder_ffi.cpp` — the FFI handlers
  already use OpenMP for vmap-parallelism. Same patterns apply.

**Tests:**
- Build a graph with 10 independent SCCs (no condensation edges
  between them). Time elimination with `omp_set_num_threads(1)`
  vs `omp_set_num_threads(8)`. Expect near-linear speedup.
- Verify numerical correctness is unchanged from sequential.
- Run with `PHASIC_DISABLE_CACHE=1` to confirm the parallel path
  works without disk involvement.

### 4.7 WP-7: Integration — wire into `ptd_precompute_reward_compute_graph`

**Goal:** Replace the monolithic elimination call inside
`ptd_precompute_reward_compute_graph` with the SCC-decomposed
pipeline, gated by an env-var (`PHASIC_HIERAR_ELIMINATION=1` initially,
flipped to default-on once the branch is stable).

**Changes:**

1. In `src/c/phasic.c:1934–1958`, replace the
   `ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph)` call
   with:
   ```c
   if (use_hierarchical) {
       // SCC-decomposed pipeline
       struct ptd_scc_graph *scc = ptd_find_strongly_connected_components(graph);
       // ... per-SCC compute (WP-4, parallelised by WP-6)
       // ... compose (WP-5)
       graph->parameterized_reward_compute_graph = composed;
       ptd_scc_graph_destroy(scc);
   } else {
       graph->parameterized_reward_compute_graph =
           ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
   }
   ```
2. Stage A2 (parent-level disk cache) remains unchanged — the
   composed PRC is still keyed and saved by the parent graph's
   content hash via the existing
   `ptd_save_parameterized_reward_compute_graph` call.

**Tests:**
- All existing integration tests must pass with both
  `PHASIC_HIERAR_ELIMINATION` set and unset. Numerical agreement to
  ≤ 1e-12 between the two paths is the gate.
- A new benchmark on a large structured coalescent (≥ 500 vertices)
  comparing wall-clock between hierarchical and monolithic.

### 4.8 WP-8: Telemetry and tooling

**Goal:** Make the cache observable so users (and we) can confirm
it's doing what it's supposed to.

**Components:**

- Extend `phasic.cache.param_compute_cache_info()`
  (`src/phasic/cache.py`) to count SCC-level entries (those with
  `scc_` prefix) separately from parent-level entries.
- Add a `phasic.cache.scc_cache_stats()` helper that summarises
  hit/miss patterns recorded during the most recent forward call.
  Implementation: a small ring buffer in C populated by
  `ptd_scc_get_or_compute_prc` (WP-4), exposed via pybind11.
- Add `PHASIC_LOG_LEVEL=DEBUG` log lines at SCC build, save, load,
  cache hit/miss boundaries.

**Tests:** mostly observability; verify that cache_info() returns
the expected counts after a controlled sequence of operations.

## 5. Verification strategy

The single most important verification is the
**equivalence test** between hierarchical and monolithic elimination.
This must pass for every WP that touches numerical output.

```python
def assert_hierarchical_equivalence(graph, n_random_thetas=100):
    """Run after WP-7 is in place."""
    import numpy as np
    rng = np.random.default_rng(42)
    for _ in range(n_random_thetas):
        theta = rng.lognormal(size=graph.param_length())
        with phasic_env(PHASIC_HIERAR_ELIMINATION='0'):
            graph_a = clone(graph)
            graph_a.update_weights(theta)
            r_mono = graph_a.expectation()
        with phasic_env(PHASIC_HIERAR_ELIMINATION='1'):
            graph_b = clone(graph)
            graph_b.update_weights(theta)
            r_hier = graph_b.expectation()
        assert np.isclose(r_mono, r_hier, rtol=1e-12, atol=1e-12), \
            f"theta={theta}: mono={r_mono}, hier={r_hier}"
```

Run this on:

- Small graphs (5–10 vertices) with hand-constructed SCC structures
  (single SCC, two-SCC chain, two-SCC-fan, two parallel SCCs, nested).
- Real coalescent at `nr_samples=5..20`.
- A custom graph with many small SCCs (worst case for composition
  bookkeeping).
- A graph with one giant SCC (worst case for per-SCC elimination —
  no parallelism win to be had).

## 6. Cross-cutting design decisions

### 6.1 Cache directory layout

Both SCC and parent PRCs live in
`~/.phasic_cache/parameterized_reward_compute/`, distinguished by
filename prefix:

```
~/.phasic_cache/parameterized_reward_compute/
    <parent_hash_hex>.bin       # parent PRC (current Stage A2)
    scc_<scc_hash_hex>.bin      # per-SCC PRC (new with this branch)
```

Rationale: same disk format, same cleanup tools, same env-var
control. The `scc_` prefix makes per-SCC artefacts trivially
identifiable for stats and selective clearing.

### 6.2 Canonical vertex ordering inside synthetic SCC graphs

Within each canonical category (`upstream_connecting`,
`internal_only`, `downstream_connecting`), order vertices by
lexicographic order of their state vectors. This is the simplest
canonicalisation that works for phasic's typical workload (where
state vectors are unique and structurally meaningful).

For graphs without unique state vectors, fall back to:
1. State vector lexicographic order.
2. Out-edge signature (sorted target-position-within-SCC + sorted coefficient-vector pairs).
3. (Final tiebreak) original parent-graph index, with a logged warning.

This is documented in WP-2.

### 6.3 Placeholder coefficients in synthetic graphs

The synthetic source's edges into the SCC, and the SCC's edges out
to the synthetic absorbing vertex, must be encoded as placeholders
that hash invariantly across parents.

Concrete scheme: the synthetic graph stores these edges with a
single placeholder coefficient of `1.0` and the placeholder edges
are marked (via an ancillary array passed alongside the graph) as
"do not encode this edge's `&coefficients[k]` as `EDGE` in the
on-disk PRC; encode it as `EXTERNAL` with a fresh table index."

This means `ptd_graph_content_hash` of the synthetic graph sees
identical placeholder weights regardless of parent, so the hash
captures only the SCC's intrinsic structure.

### 6.4 Backward compatibility

- Format revision 1 cache files (existing parent-level Stage A2
  entries) remain readable by the new loader.
- The hierarchical path is gated by `PHASIC_HIERAR_ELIMINATION` until
  it has burned in. The monolithic path is the reference until
  flipped.
- No public Python API changes. Cache-clearing tooling
  (`phasic.cache.clear_param_compute_cache`) clears both prefixes.

### 6.5 Robustness to non-unique vertex states (load-bearing invariant)

The starting vertex and any auxiliary vertex created via
`ptd_vertex_create()` (whether by user code calling
`add_aux_vertex()` or by phasic itself in
DPH-normalisation, see `src/c/phasic.c:2556`) are initialised with
an **all-zero state vector** (`src/c/phasic.c:3754–3763`,
`calloc(state_length, sizeof(int))`). Real workloads therefore
contain graphs with multiple non-starting vertices sharing the
all-zero state — and more generally, *any* state-vector collision
between non-starting vertices is structurally legal.

Every component of this branch must work correctly on graphs with
duplicate non-starting state vectors. Concretely:

- **Vertex identity is the pointer (`struct ptd_vertex *`) or the
  numerical index, never the state vector.** Any code that builds
  `vertex_map` lookups, vertex categorisations, or hash inputs must
  key on identity, not state.
- **`find_or_create_vertex(state)` cannot be used to "look up" an
  existing aux vertex.** The AVL tree behind it does not contain
  aux vertices (see `src/c/phasic.c:1291` and the comment at
  `api/cpp/scc_graph.cpp:144–151`); calling it on an all-zero
  state creates a *new* vertex rather than returning the existing
  aux. WP-1's synthetic-graph constructor must therefore use
  `ptd_vertex_create()` (or `vertex_create_state()` with explicit
  state copy) for every internal vertex, then add edges to the
  newly-created vertices via direct vertex pointers, never via
  state-based lookup.
- **The existing `SCCVertex::as_graph()`
  (`api/cpp/scc_graph.cpp:138–222`) already has a partial fix
  for the starting vertex** but does **not** handle non-starting
  aux vertices. Multiple aux vertices in one SCC would currently
  collide on `find_or_create_vertex([0,...,0])` and produce a
  malformed subgraph. WP-1 must close this gap.
- **Canonical SCC ordering (§6.2) cannot rely on state-vector
  uniqueness as a primary key.** The lexicographic-state-vector
  rule still applies, but for collision resolution we fall back
  to (a) sorted out-edge signatures, (b) is_aux flag, (c) parent
  index as last resort. SCCs containing aux vertices are
  expected — not a corner case — and the canonicalisation must
  produce stable hashes across parents that contain the same SCC
  with aux vertices in identical positions.
- **Toy regression coverage must include an aux-vertex variant.**
  Add a "Toy-D" variant alongside the others (§11.3): a graph
  identical to toy-base but with an explicit aux vertex inside
  SCC₂ that has all-zero state. Reference values captured the
  same way. This variant is the regression anchor for the
  duplicate-state invariant.
- **The `is_aux` flag is preserved by `ptd_clone_graph`
  (`src/c/phasic.c:1178`)**, so it survives the SCC subgraph
  extraction. WP-1 must preserve it on synthetic-graph copies.

This invariant is mentioned here once and applies to every WP. If
you find yourself writing code that uses
`find_or_create_vertex(state)` to identify a vertex during synthetic
graph construction, composition, or hash computation, you have
violated this invariant and the code must be rewritten.

### 6.6 SCC caching is moments-specific, not universal

`Stage A2` and the SCC-decomposed pipeline this branch builds are
hit only by code paths that go through
`ptd_precompute_reward_compute_graph` →
`ptd_expected_waiting_time`. That family is:

- `Graph.expectation` / `Graph.moments` / `Graph.variance` /
  `Graph.covariance` (eager).
- The moments half of `pmf_and_moments_from_graph` and its
  multivariate sibling (JAX-traceable, used by SVGD with rewards
  or with non-zero regularisation, and by SVGD's default
  `DataPrior` setup).
- `Graph.expected_sojourn_time` on parameterised graphs.

PMF/PDF/CDF computations (continuous *and* discrete) use
uniformization — the forward algorithm — and do **not** require
symbolic Gaussian elimination. Stage A2 is therefore **not** the
"universal C-side cache"; it is specifically a cache for the
moment-computation path.

Two practical consequences for this branch:

- **Discrete PMF does not benefit from this branch.** `Graph.dph_pmf`
  and the discrete branch of `Graph.pdf` go through uniformization,
  not elimination. The hierarchical pipeline does not affect them.
- **Reward transformation is structurally a graph rewrite
  (Algorithm 2), separate from moment computation.** A reward
  vector reshapes the graph; PMF on the rewritten graph still uses
  uniformization, no elimination. Today
  `pmf_and_moments_from_graph` performs the rewrite *and* a
  moments computation in one C++ trip; the moments computation is
  what triggers Stage A2, not the rewrite. There is no inherent
  reason the SCC pipeline must always run when rewards are present;
  it runs when moments are computed, period.

This framing matters for telemetry (WP-8): "Stage A2 hit rate per
SVGD iteration" is meaningful only for runs that compute moments.
Joint-prob SVGD (which uses sojourn times, not moments) bypasses
the cache entirely and reporting hit-rates on it would be
nonsensical.

It also matters for follow-up work. A small dispatcher
optimisation in `Graph.svgd` — routing
`rewards is None and regularization == 0` to `pmf_from_graph`
rather than `pmf_and_moments_from_graph` — would skip Stage A2
entirely for that configuration. That optimisation is
**out of scope for this branch** but its mention here is
deliberate: anyone evaluating the branch's wallclock impact on a
no-rewards/no-regularisation SVGD run should know that
`pmf_and_moments_from_graph` is paying the moments cost
unnecessarily, and therefore that improvements from this branch
on that workload reflect mostly redundant work being parallelised,
not work that needed doing in the first place.

### 6.7 What we don't have to solve in this branch

- **Distributed (cross-process) SCC computation.** Within-process
  parallelism (WP-6) is explicit; cross-process distribution is a
  separable later concern. The on-disk SCC cache makes
  cross-process trivially possible (job submits run, populate cache,
  later run picks up the work) but no orchestration code is required
  on this branch.
- **MPFR support.** The MPFR path (`ptd_graph_ex_absorbation_time_comp_graph_mpfr`,
  `ptd_expected_waiting_time_mpfr`) is structurally separate. SCC
  decomposition for MPFR is a follow-up.
- **JAX FFI integration.** The FFI handlers do not consume PRCs
  today and are not affected by this work.

## 7. Sequencing

The work packages are roughly ordered by dependency. Within each
phase, tests must pass before moving on. Note that disk persistence
of per-SCC PRCs is not deferred to a "later phase" — it lands in
Phase B alongside the per-SCC compute path, because parallel
elimination and cross-graph reuse are the same goal (§1.3).

**Phase A — synthetic graph and canonical hashing:**
- WP-1 (synthetic graph builder)
- WP-2 (canonical hashing)
- Tests that compare synthetic-graph content hashes across two
  hand-constructed parents — the cross-graph-reuse correctness
  invariant.

**Phase B — format extension and per-SCC disk cache:**
- WP-3 (format extension)
- WP-4 (per-SCC compute + cache lookup, with on-disk persistence
  from day one)
- Round-trip and cache-hit tests, including the cross-parent reuse
  case (parent A populates the cache, parent B containing a
  structurally-identical SCC hits it).

**Phase C — assembly:**
- WP-5 (composition)
- Equivalence test on small graphs (the single most important test
  in the branch).

**Phase D — production wire-up:**
- WP-7 (integration into `ptd_precompute_reward_compute_graph`)
- Equivalence test on real workloads.

**Phase E — parallelism and telemetry:**
- WP-6 (parallel per-SCC elimination)
- WP-8 (telemetry: cache hit/miss visibility, SCC-level cache stats)
- Benchmark suite.

Phases A–D are correctness-driven and must produce no behavioural
regressions when the env-var is unset. Phase E delivers the
wall-clock wins. Cross-graph reuse is a property of Phases B–D
combined — by the time Phase D lands, two parents sharing an SCC
already share its cached PRC; Phase E only adds parallelism and
observability on top.

## 8. Quick orientation index

Files most relevant to this branch:

| Concern | File |
|---|---|
| Eliminator (current monolithic path) | `src/c/phasic.c:6981–7995` |
| SCC discovery | `src/c/phasic.c:1994–2510` |
| PRC format & disk I/O | `src/c/phasic.c:2855–3700+` |
| Stage A2 cache integration | `src/c/phasic.c:1858–1985` |
| Graph content hashing | `src/c/phasic_hash.c:147–272` |
| SCC C++ wrappers | `api/cpp/scc_graph.h`, `api/cpp/scc_graph.cpp` |
| Pybind SCC bindings | `src/cpp/phasic_pybind.cpp:3062–3110` |
| Python algorithm reference | `src/phasic/hierarchical_trace_cache.py:880–2100` |
| Public C API surface | `api/c/phasic.h:165–520, 952–977` |
| `expected_waiting_time` consumer | `src/c/phasic.c:8247–8394` |
| `ptd_graph_build_ex_...` (PRC → reward compute graph) | `src/c/phasic.c:8008–8113` |

The single most useful chunk to internalise before writing new code
is **`src/c/phasic.c:6981–7995`** (the parameterised eliminator
itself) and **`src/phasic/hierarchical_trace_cache.py:1047–1350`**
(the synthetic-graph construction algorithm, as written in Python).
The C work is essentially porting the Python construction algorithm
and routing its output through the existing C eliminator.

## 9. Follow-up project — MPFR support for the SCC-decomposed pipeline

This is a separate project to be tackled after the main branch
lands. It is documented here because the design choices in the main
branch must not foreclose the MPFR path.

### 9.1 What MPFR does today

The current monolithic eliminator has two siblings:

- **`ptd_graph_ex_absorbation_time_comp_graph_parameterized`**
  (`src/c/phasic.c:6981`) — builds a θ-dependent PRC.
- **`ptd_graph_ex_absorbation_time_comp_graph_mpfr`**
  (`src/c/phasic.c:6486`, `static`) — builds a θ-independent
  high-precision compute graph using MPFR arithmetic. Each command
  stores its multiplier as a `char *multiplier_str`
  (`struct ptd_reward_increase_mpfr` at `api/c/phasic.h:440`).
- **`ptd_graph_build_ex_absorbation_time_comp_graph_parameterized`**
  (`src/c/phasic.c:8008`) — collapses the parameterised PRC to a
  non-parameterised one (`ptd_desc_reward_compute`) once concrete θ
  is known.
- **`ptd_expected_waiting_time_mpfr`**
  (`src/c/phasic.c:8124`, `static`) — MPFR-precision consumer.

The runtime decision is in
`ptd_expected_waiting_time` (`src/c/phasic.c:8247–8335`): pre-scan
the (θ-bound) `reward_compute_graph` for conditioning, and if
`condition_number > PHASIC_CONDITION_THRESHOLD` (default 1e12) or
`PHASIC_FORCE_MPFR=1`, lazily build the MPFR compute graph from the
*original* `ptd_graph` and route through `ptd_expected_waiting_time_mpfr`.

A subtle point: the MPFR path today does **not** consume the
parameterised PRC. It re-eliminates from scratch using the graph's
concrete (θ-bound) edge weights. So MPFR caching today is only
useful when the same θ recurs — `reward_compute_graph_mpfr` is
cached on the `ptd_graph` struct but invalidated whenever weights
change.

### 9.2 What "MPFR for the SCC-decomposed pipeline" means

Two distinct things, with very different scope:

**MPFR-A: high-precision consumer over the composed parent PRC.**
The hierarchical parent PRC is θ-dependent; once collapsed via
`ptd_graph_build_ex_absorbation_time_comp_graph_parameterized` it
becomes a `ptd_desc_reward_compute` whose multipliers are concrete
doubles. This is exactly the input the existing MPFR machinery
operates on after re-elimination today — except that with the
hierarchical pipeline we already *have* the collapsed PRC, so
re-elimination is unnecessary. We just need an MPFR consumer that
reads `reward_compute_graph` (collapsed, double-precision) and
performs the multiply-accumulate sweep at MPFR precision.

This is small: extend `ptd_expected_waiting_time_mpfr` with a path
that takes the existing `reward_compute_graph` rather than
allocating a fresh MPFR compute graph. The conditioning pre-scan
in `ptd_expected_waiting_time` already operates on
`reward_compute_graph`, so the trigger logic is unchanged.

MPFR-A is the one most workloads need. It picks up arbitrary
precision *consumption* without changing the elimination side.

**MPFR-B: high-precision per-SCC elimination with on-disk
high-precision SCC artefacts.** Eliminate each SCC at MPFR
precision, store per-SCC MPFR PRCs on disk, compose them into a
parent MPFR PRC. This is the analogue of the main branch but with
arbitrary-precision arithmetic throughout.

MPFR-B is real work and only worth doing if MPFR-A is found
insufficient — i.e. if conditioning is so bad that even the
elimination itself loses too much precision in double, and you
need MPFR multipliers preserved through composition.

### 9.3 Work packages — MPFR-A (the cheap, high-value path)

**MPFR-A-WP-1: hoist the MPFR consumer to operate on
`reward_compute_graph`.**

Currently `ptd_expected_waiting_time_mpfr`
(`src/c/phasic.c:8124`) consumes `reward_compute_graph_mpfr`. Add a
sibling function that consumes `reward_compute_graph` (concrete
doubles) and performs the multiply-accumulate at MPFR precision.
The arithmetic loop is the same shape; only the multiplier source
changes.

**MPFR-A-WP-2: route the conditioning trigger through the new
consumer.**

In `ptd_expected_waiting_time` (`src/c/phasic.c:8290–8327`),
replace the lazy build of `reward_compute_graph_mpfr` with the
new consumer. Preserve `PHASIC_FORCE_MPFR` and
`PHASIC_CONDITION_THRESHOLD` semantics.

**MPFR-A-WP-3: deprecate `reward_compute_graph_mpfr` and the
`_comp_graph_mpfr` builder.**

Once the new consumer is in place, the cached MPFR compute graph
becomes redundant — the consumer reads the regular
`reward_compute_graph` directly. Mark
`ptd_graph_ex_absorbation_time_comp_graph_mpfr` and the
`reward_compute_graph_mpfr` field as deprecated, behind a build
flag, then remove in a follow-up.

**Tests:** numerical agreement with current MPFR path on a poorly
conditioned graph (e.g. coalescent at large `nr_samples` where
condition numbers naturally exceed 1e12) to within `2^-precision`
relative tolerance.

### 9.4 Work packages — MPFR-B (the parallel-elimination MPFR path)

Only undertake if MPFR-A is found insufficient. The shape is
parallel to the main branch.

**MPFR-B-WP-1: per-SCC MPFR elimination.** Add an MPFR-precision
variant of the synthetic-SCC eliminator (mirrors §4.1 / §4.4).
Output: per-SCC MPFR command list with `char *multiplier_str`
fields.

**MPFR-B-WP-2: on-disk MPFR SCC PRC format.** Mirrors §4.3 / WP-3.
The disk format must encode multiplier strings (variable length).
Add a `mpfr_` filename prefix to distinguish from
double-precision SCC entries:
`~/.phasic_cache/parameterized_reward_compute/mpfr_scc_<hash>.bin`.

**MPFR-B-WP-3: MPFR composition.** Mirrors §4.5 / WP-5. String
multipliers do not participate in `EXTERNAL` pointer wiring (the
parent's θ-bound multipliers are still encoded as live edge
references — the `EDGE` pointer kind survives). What changes is
the per-command arithmetic: composition concatenates command
sequences; the underlying numerical replay is MPFR.

**MPFR-B-WP-4: MPFR consumer over composed MPFR PRC.** A
final-step MPFR sweep that reads the composed parent MPFR PRC and
produces a high-precision result vector.

**Trigger logic.** The existing `condition_number > threshold` path
(`src/c/phasic.c:8290`) decides between double and MPFR consumers
on a per-call basis. With MPFR-B the decision moves earlier:
either rebuild the parent PRC at MPFR precision (expensive, may
trigger per-SCC MPFR elimination — but cached on disk after first
time), or fall through to MPFR-A's hybrid (eliminate in double,
consume in MPFR) which is cheaper but less precise.

**Tests:** end-to-end equivalence on poorly-conditioned graphs
between MPFR-B and the existing monolithic MPFR path, to within
`2^-precision`.

### 9.5 Cache invalidation and naming

The main branch's cache directory layout (§6.1) reserves the
`scc_` prefix for double-precision SCC PRCs. MPFR-B's
`mpfr_scc_` prefix avoids collisions. Cache-clearing tooling
(`phasic.cache.clear_param_compute_cache`) must understand both
prefixes.

The format-revision number lives independently per file format. A
revision bump on `scc_*.bin` does not invalidate `mpfr_scc_*.bin`
and vice versa.

### 9.6 Why this is a separate project

- **Risk isolation.** MPFR is a build-optional dependency
  (`#ifdef HAVE_MPFR`). The main branch must not regress when
  built without MPFR. Keeping MPFR work to a follow-up keeps the
  compile-time conditional surface confined to one PR.
- **Empirical priority.** It is currently unclear whether real
  phasic workloads even need MPFR after the SCC decomposition
  lands, since per-SCC elimination on smaller subgraphs may itself
  reduce conditioning by avoiding large-cycle bypass-edge
  arithmetic. We should measure conditioning on real workloads
  after the main branch ships before scoping MPFR-B.
- **MPFR-A is small enough to be a single PR.** It can land
  without waiting for MPFR-B at all.

## 10. Follow-up project — distributed SCC computation across SLURM nodes

This is the second follow-up project. It treats SCC level sets as
units of distribution: an entire level set can be farmed out to
worker nodes on a cluster, computed in parallel, with results
collected through the shared `~/.phasic_cache/`.

### 10.1 Why this composes cleanly with the main branch

The main branch already produces what's needed:

- **Per-SCC PRCs are content-hashed and disk-persistent.** Any
  process can write a cache entry; any process can read it. The
  atomic write-then-rename in
  `ptd_save_parameterized_reward_compute_graph` means concurrent
  writers from different nodes converge to identical content.
- **Per-SCC eliminations are independent.** Within a level set
  there are no inter-SCC dependencies. The level-set
  decomposition is computed once on the orchestrator and
  distributed; each worker just receives "compute these SCC
  hashes" and writes the results to the shared cache.
- **Composition is fast** (§4.5). The orchestrator does composition
  locally after all workers in a level set have finished. There is
  nothing to distribute about composition.

So the distributed work is "submit SLURM tasks that populate the
SCC cache, then run composition locally." No protocol design, no
RPC, no message-passing — just shared filesystem access and the
existing on-disk cache.

### 10.2 Required orchestration components

**SLURM-WP-1: shared-filesystem cache discovery.**

`~/.phasic_cache/` is conventionally a per-home cache, which on a
SLURM cluster typically maps to a shared NFS home directory. Add
an env var `PHASIC_CACHE_DIR` that overrides the default location.
Workers and the orchestrator must agree on this path.

Implementation: extend `ptd_pcg_build_cache_path`
(`src/c/phasic.c:3181+`) to honour `PHASIC_CACHE_DIR` if set.
Document filesystem requirements (POSIX rename semantics; Lustre
and most NFS variants are fine, GPFS sometimes is not).

**SLURM-WP-2: orchestrator — level-set computation.**

Given a parent graph, compute the SCC condensation, identify
level sets (SCCs at the same topological depth that have no
inter-set dependencies). For each level set, identify which SCCs
are *missing* from the cache (
`!stat("~/.phasic_cache/.../scc_<hash>.bin")`) and emit a list of
work units `(scc_hash, parent_graph_serialised)` to compute.

The orchestrator code can live in Python
(`src/phasic/distributed_utils.py` already exists for related
purposes). Implementation: walk the SCCs in
`SCCGraph.sccs_in_topo_order()` order (reverse topological,
sink-first — see §3.5), group by level (BFS layers in the
condensation), filter by cache-miss, emit work units.

**SLURM-WP-3: worker — single-SCC compute task.**

A standalone CLI entry point (`python -m phasic.scc_worker
<work_unit_path>`) that:

1. Loads the work unit (parent graph JSON + target SCC hash).
2. Reconstructs the parent graph (or just the SCC subgraph if
   possible — see SLURM-WP-4).
3. Calls `ptd_scc_get_or_compute_prc` (WP-4 of the main branch),
   which writes the result to the shared cache.
4. Exits.

Each worker is one SLURM array task. Failures are isolated; the
orchestrator retries by re-checking the cache after the level set
completes.

**SLURM-WP-4: minimal SCC subgraph serialisation.**

Naively passing the entire parent graph to each worker is
wasteful — the worker only needs the SCC's vertices, internal
edges, and the synthetic-graph wrapping. Implement a
`serialise_scc_subgraph(parent_graph, scc_idx)` helper that
produces a self-contained JSON describing just the SCC's
synthetic graph. The worker rebuilds it via the existing JSON
graph constructor (`graph_from_json`).

This is pure plumbing — no new algorithms.

**SLURM-WP-5: SLURM job-script generator.**

A helper `phasic.distributed_utils.submit_scc_jobs(parent_graph,
sbatch_options)` that:

1. Writes work units to a temp directory on the shared FS.
2. Generates a SLURM array job script that invokes
   `python -m phasic.scc_worker $WORK_UNIT_PATH`.
3. Submits the array via `sbatch`.
4. Returns a future-like handle that the caller can wait on.

The caller's contract: after `wait()` returns, all SCCs in the
requested level set have on-disk cache entries (or one or more
workers failed, in which case the level-set is partially populated
and the local fallback eliminates whatever's missing).

**SLURM-WP-6: integration — distributed-aware
`ptd_precompute_reward_compute_graph` wrapper.**

A Python-level wrapper around the C call:

```python
def precompute_distributed(graph, slurm_options=None):
    if slurm_options is None:
        graph._precompute()  # local hierarchical, already in C
        return
    scc_decomp = graph.scc_decomposition()
    for level_set in level_sets_in_topo_order(scc_decomp):
        missing = [scc for scc in level_set if not _scc_cached(scc)]
        if missing:
            submit_scc_jobs(graph, missing, slurm_options).wait()
    graph._precompute()  # composition + parent cache; fast now
```

### 10.3 Performance considerations

- **SLURM submission overhead** (~seconds per array submission)
  dominates for small graphs. Set a threshold below which the
  distributed path is bypassed in favour of local in-process
  parallelism (WP-6 of the main branch). Default threshold:
  parent graphs with < 1000 vertices.
- **Filesystem contention.** Many workers writing to the same
  directory can hit metadata-server bottlenecks on shared
  filesystems. Mitigate by sharding the cache dir into hash-prefix
  subdirectories (`~/.phasic_cache/.../scc_<aa>/scc_<aabbcc...>.bin`)
  if profiling reveals contention. Defer until measured.
- **Worker placement.** SLURM allocations can be small (one core
  per worker, modest memory). Configure via `sbatch_options`.

### 10.4 What does *not* need to change

- The main branch's C code is unchanged. The distribution layer
  sits entirely above it, in Python orchestration and CLI worker
  scripts.
- The SCC PRC format is unchanged.
- The `ptd_graph_content_hash` function is unchanged. Workers and
  orchestrator must agree on the hash, which they automatically do
  because they share the C code.

### 10.5 Sequencing

SLURM-WP-1 (cache dir discovery) and SLURM-WP-4 (SCC
serialisation) are independent and can land first. SLURM-WP-2,
WP-3, WP-5 are interdependent and form one PR. SLURM-WP-6 is the
final integration.

### 10.6 Out of scope for this follow-up

- **Non-SLURM cluster systems** (PBS, LSF, k8s). The orchestration
  layer can be generalised, but the first cut targets SLURM only,
  matching `phasic.distributed_utils`'s existing focus.
- **Cross-host shared cache without shared filesystem.** Some
  clusters don't share home directories. A future extension could
  use object storage (S3-compatible) as the cache backend, but
  this is a substantial change to
  `ptd_save_parameterized_reward_compute_graph` and is deferred.
- **Dynamic load balancing.** The orchestrator submits a fixed
  array job per level set. Adaptive scheduling (steal-work, etc.)
  is overkill for the workload shape and not pursued.

## 11. Toy reference model — the four-vertex two-SCC graph

A canonical small graph used as the cross-stage debugging reference.
Small enough to enumerate by hand, large enough to exercise every
algorithmic surface in the main branch.

### 11.1 Topology

Four non-starting vertices `A, B, C, D`, plus the starting vertex
`s` and the absorbing vertex `Ω` (auto-created by phasic when an
edge is added to a vertex with no out-edges). Two SCCs:

- **SCC₁ = {A, B}** — a two-cycle.
- **SCC₂ = {C, D}** — a two-cycle.

Edges (parameterised; coefficients indicate how θ enters):

```
   s ──[c0·θ0]──> A
   s ──[c1·θ0]──> C        (initial probability vector)

   A ──[c2·θ1]──> B        (SCC₁ internal)
   B ──[c3·θ1]──> A        (SCC₁ internal, closes the cycle)

   A ──[c4·θ2]──> C        (SCC₁ → SCC₂; downstream-connecting in SCC₁)
   B ──[c5·θ2]──> D        (SCC₁ → SCC₂; downstream-connecting in SCC₁)

   C ──[c6·θ1]──> D        (SCC₂ internal)
   D ──[c7·θ1]──> C        (SCC₂ internal, closes the cycle)

   C ──[c8·θ3]──> Ω        (absorption from SCC₂)
   D ──[c9·θ3]──> Ω        (absorption from SCC₂)
```

Coefficient values to use as reference:
`c0..c9 = (1.0, 1.0, 2.0, 1.0, 0.5, 0.3, 1.5, 0.7, 1.0, 1.0)`.

State vectors (`state_length=2`) are arbitrary but should be unique
to allow canonical hashing without falling through to tie-breakers:

```
A: [0, 0]
B: [0, 1]
C: [1, 0]
D: [1, 1]
```

### 11.2 What this exercises

- **Two SCCs in a chain.** SCC₁ topologically precedes SCC₂ —
  composition order is non-trivial.
- **Each SCC has internal cycles** of length 2 — the eliminator
  must do non-trivial within-SCC work, not just bypass-edge
  shuffling.
- **Multiple upstream-connecting and downstream-connecting
  vertices.** SCC₁ has two downstream-connecting vertices (A and
  B, both with edges to SCC₂). SCC₂ has two upstream-connecting
  vertices (C and D, both reached from SCC₁). This forces the
  composer to wire multiple inter-SCC channels.
- **Multiple parameters.** Four θ slots (`θ0..θ3`) cover IPV (`θ0`),
  internal SCC dynamics (`θ1`), inter-SCC (`θ2`), and absorption
  (`θ3`) — the composer must propagate parameter dependence
  correctly across all categories.
- **Manual computation is feasible.** All 10 edges fit on one
  page; the symbolic expected-time can be derived by hand.

### 11.3 What it doesn't exercise (and the variants that do)

The base toy model has two SCCs of size 2 each. For full coverage,
maintain three additional variants that share the same coefficient
scheme but vary topology:

- **Toy-A: trivial-SCC-only.** Replace each 2-cycle with a single
  vertex (so `B` and `D` are removed, A→C, C→Ω). All SCCs are
  singletons. Tests the "trivial SCC" code path of the eliminator.
- **Toy-B: parallel SCCs.** Add a third SCC `{E, F}` parallel to
  SCC₂ (s also has edge to E; both SCC₂ and SCC₃ reach Ω
  independently). Tests level-set parallelism (SCC₂ and SCC₃ are
  in the same level set).
- **Toy-C: shared-SCC reuse.** Two parent graphs `P` and `P'`,
  where `P'` is obtained from `P` by replacing SCC₂'s coefficients
  for the *external* edges (A→C, B→D, C→Ω, D→Ω) but *keeping
  SCC₂'s internal coefficients identical*. SCC₂'s synthetic graph
  must content-hash the same in both parents — this is the
  cross-graph-reuse correctness test (§5).
- **Toy-D: aux-vertex with duplicate all-zero state.** Toy-base
  augmented with one auxiliary vertex `X` inside SCC₂, where `X`
  has the all-zero state vector (colliding with the starting
  vertex's state and any other aux that may exist). `X`
  participates in SCC₂'s cycle (e.g. extra edges `C → X` and
  `X → D`). This variant is the regression anchor for the
  duplicate-state invariant (§6.5): every WP must produce
  correct results on Toy-D, and any WP that uses
  `find_or_create_vertex(state)` to identify a vertex during
  synthetic graph construction will fail here.

### 11.4 Reference values

Compute `ptd_expected_waiting_time` (without rewards, defaulting
to ones) at `θ = (1.0, 1.0, 1.0, 1.0)` using the current
monolithic eliminator on the base toy and record the result vector.
Every refactor in the main branch must agree with this vector to
≤ 1e-12.

A small test fixture under `tests/c/test_toy_model.c` should
construct each toy variant programmatically (so the test file
itself documents the topology) and assert the reference values.
The fixture lives independently of the main test suite so it can
serve as a regression anchor.

### 11.5 How to use the toy model at each stage

| Stage | Toy use |
|---|---|
| WP-1 (synthetic graph) | Construct synthetic graphs for SCC₁ and SCC₂ from toy-base **and toy-D**. Visually inspect vertex categorisation. Toy-D verifies the constructor does not collapse aux vertices via state-based lookup. |
| WP-2 (canonical hashing) | Verify SCC₂ in toy-base and SCC₂ in toy-C hash identically (cross-graph reuse). Verify Toy-D's SCC₂-with-aux hashes deterministically. |
| WP-3 (format extension) | Round-trip a toy SCC PRC through save/load with `EXTERNAL` pointers. Include Toy-D to confirm aux vertices serialise correctly. |
| WP-4 (per-SCC compute) | Build cache entries for SCC₁ and SCC₂; verify on-disk filenames; verify cache hit on second compute. Toy-D in this stage exercises aux vertices through the full per-SCC compute path. |
| WP-5 (composition) | Compose toy-base SCC₁ ∘ SCC₂; assert numerical equivalence with monolithic. Repeat on Toy-D — aux vertices must compose correctly. |
| WP-6 (parallelism) | Use Toy-B (parallel SCCs); time elimination with 1 vs 2 threads. |
| WP-7 (integration) | Run toy-base, toy-A, toy-B, toy-C, **and toy-D** through the production wrapper; verify all reference values. |
| WP-8 (telemetry) | Confirm reported hit/miss counts match the expected pattern across toy-base and toy-C. |
| MPFR-A | Inflate one coefficient in toy-base to force ill-conditioning; verify MPFR consumer kicks in. |
| SLURM | Submit toy-B's level set as a 2-task array job; verify cache populates correctly. |

The toy model is the smallest harness that meaningfully exercises
each work package in isolation. It should be the *first* test
written for any new component and the *last* one consulted when
debugging.

## 12. Working agreements

These are the operating rules for the branch. They exist to keep
the work on solid ground and avoid trial-and-error on the actual
implementation.

### 12.1 Planning cadence

- **Plan one phase ahead, not the whole branch.** Detailed WP
  plans are written at the start of each phase, informed by what
  the previous phase actually revealed. The reference document
  (this file) is the long-term contract; per-phase plans live in
  separate `wpN-<topic>.md` files committed alongside the code.
- **Two pieces of detailed planning happen up front:** (a) the toy
  model fixture and reference values (§11), instantiated as a
  runnable test before any new C code; (b) WP-1's detailed plan,
  including C signatures and struct layouts.
- **Pre-WP experiments retire unknowns before the plan is
  drafted.** Each WP's plan is preceded by experiments that
  resolve any open questions about how existing code behaves. No
  WP plan claims a fact without either a `file:line` citation or
  an experiment backing it.

### 12.2 Verification gates

- **The toy model regression runs on every commit during this
  branch.** Cheap, catches mistakes early. Reference values
  computed against the current monolithic eliminator at fixed θ
  vectors and committed as
  `tests/pytest/toy_model_reference.json`.
- **Each WP must pass the toy-model regression and at least one
  targeted unit test for its new behaviour before merging.** Full
  pytest is not required (project test suite has known
  flakiness — see project memory).
- **Toy-D (aux vertex with duplicate all-zero state) is a hard
  gate for every WP that touches graph traversal or hashing.**
  Duplicate non-starting state vectors are a load-bearing
  invariant of phasic graphs (§6.5); a WP that passes other toy
  variants but fails Toy-D is silently broken for any user who
  calls `add_aux_vertex()` or whose graph went through DPH
  normalisation. Do not merge a WP without Toy-D coverage.
- **The cross-path equivalence test (§5) is the gate for any WP
  that touches numerical output.** Hierarchical and monolithic
  results must agree to ≤ 1e-12 across many random θ on the
  toy variants (including Toy-D) and on a real coalescent.
- **`PHASIC_HIERAR_ELIMINATION` env-var gate stays in place
  until Phase D's equivalence test passes on real workloads.**
  No flipping the default before that gate.

### 12.3 Tooling and isolation

- **Experiments run in temporary git worktrees** (`isolation:
  worktree`) so the `hierar-elimin-cache` branch stays in a
  known-good state. Worktrees are auto-cleaned if no changes are
  committed.
- **Read-only fact-finding goes to the Explore agent.** Anything
  of the form "find every caller of X" or "is symbol Y used
  outside file Z" is delegated to keep the main context focused
  on design and implementation.
- **Multi-step research goes to general-purpose.** Open-ended
  investigations that need both grep and reasoning (e.g. "find
  every caller of X and explain whether each tolerates change
  Y") use the general-purpose agent.
- **WP-5 composition gets a math-stats-checker audit pass.** The
  symbolic-elimination algebra during composition is the most
  algorithmically delicate part of the branch. After
  implementation lands and passes toy regression, the
  math-stats-checker agent reviews the derivation.
- **Each WP gets a code-improver review pass before merging.**
  After own implementation passes toy regression, the
  code-improver agent does one read-through. Concerns it raises
  are addressed (or explicitly accepted with reasoning) before
  merging.

### 12.4 Communication discipline

- **Every claim in a plan cites `file:line` or experiment.** No
  "I think this should work" sentences. Either a citation, or an
  experiment ID with its result, or a flag that says "open
  question to be resolved by experiment N before WP starts."
- **Reports stay short.** Experiment results in 5–10 lines; WP
  plans in 1–2 pages; long-form documents only when asked.
- **Phase boundaries are checkpoints.** End of each WP: short
  summary of what landed, what changed in the long-term reference
  doc, what the next WP needs from this one. Then wait for sign-off
  before the next phase begins.
- **Open questions surface immediately, not at end-of-phase.** If
  a WP plan reveals an assumption that doesn't hold, that's
  surfaced in the same message that drafts the plan, with a
  proposed experiment to resolve it — not buried.

### 12.5 What "standing on solid ground" means in practice

- **No guessing about how existing code behaves.** Read it, or run
  it, or both. The C eliminator and the on-disk PRC format are
  the load-bearing infrastructure for this branch; we cannot
  afford to misunderstand them.
- **Vertex identity is by pointer or numerical index, never by
  state.** State vectors collide legally — multiple non-starting
  vertices may share the all-zero state via `add_aux_vertex()`
  or DPH normalisation (§6.5). Code that uses
  `find_or_create_vertex(state)` to identify an *existing*
  vertex during synthetic graph construction, composition, or
  hash computation is broken. The pre-merge checklist for every
  WP includes a `grep` for `find_or_create_vertex` in the new
  code; each occurrence must be justified (creating a new
  unique-state vertex) or removed.
- **Function names and comments are NOT trustworthy substitutes
  for empirical verification.** Worked example from this branch
  (2026-05-10): `SCCGraph::sccs_in_topo_order()` is named and
  documented as returning topological order, but actually
  returns *reverse* topological order (sink-first). The doc was
  initially written assuming the name was accurate; the
  Toy-D experiment uncovered the discrepancy. Lesson: when a
  WP plan turns on which direction a function returns,
  *measure* the direction with a print-and-eyeball experiment
  before writing the WP plan. Names lie; behaviour doesn't.
- **Trial-and-error happens in experiments, not in committed
  code.** If we don't know whether approach X works, an
  experiment in a worktree answers it before any line of branch
  code is written.
- **Numerical agreement to ≤ 1e-12 is non-negotiable for
  refactors that should be no-ops.** "It looks close" is not a
  passing result.
- **The toy model is the canonical debugging anchor.** When
  something disagrees, the first move is to reproduce the
  disagreement on the smallest toy variant that exhibits it.

### 12.6 Scope discipline

- **No drive-by improvements.** This branch is for SCC-decomposed
  hierarchical elimination. Code-improver may notice unrelated
  cleanups; those go on a separate list, not into this branch's
  PRs.
- **No format changes beyond what WPs explicitly require.** The
  PRC `.bin` format is changed once (WP-3) to add the `EXTERNAL`
  pointer kind. Any further format work is a follow-up PR with
  its own version-bump rationale.
- **No new public Python API in this branch.** Internal C and C++
  APIs grow as needed; the user-visible surface stays the same.
  Telemetry helpers (WP-8) are the one exception and stay
  minimal.
- **Follow-ups stay in their own projects.** MPFR (§9) and SLURM
  (§10) do not begin until the main branch has landed and burned
  in.
