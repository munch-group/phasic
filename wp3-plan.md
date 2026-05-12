# WP-3 — PRC format extension for cross-SCC binding (detailed plan)

**Branch:** `hierar-elimin-cache`
**Status:** drafted, awaiting sign-off
**Predecessors:** WP-1 (commit `7d7447c`), WP-2 (commit `800ca6e`)
**Successor:** WP-4 (per-SCC compute + cache lookup)

## 1. The problem WP-3 solves

WP-1 produces a synthetic graph in which the SCC's placeholder
source/absorbing edges have parameter-independent coefficients
(`[1.0, 0.0, ..., 0.0]`). When the eliminator runs on this synthetic
graph, the resulting PRC contains `multiplierptr` references to
those placeholder coefficients.

If we save that PRC to disk and reload it later, the load function
needs to bind the multiplier pointers to *something*. Two options:

- **Option A:** bind to the synthetic graph's own placeholder edge
  slots (which contain the literal `[1.0, 0, ..., 0]`). Then the
  loaded PRC produces a numeric result that ignores the parent's
  external edge weights — wrong for composition.
- **Option B:** bind to the parent graph's actual external edge
  slots, so the PRC's elimination produces the correct result for
  this specific parent. Right for composition, but requires the
  load function to know which placeholder corresponds to which
  parent edge.

WP-3 is the format work that makes Option B possible.

## 2. Two competing designs

The WP-1 reference doc (§3.3, §6.3) and WP-1 plan §4.4 specified a
new `EXTERNAL` pointer kind. After implementing WP-1 and WP-2, an
alternative is now visible: **just reuse the existing `EDGE`
pointer kind, but resolve it against the parent graph at load
time.** Let me compare both before committing to one.

### 2.1 Design A: new `PTD_PCG_PTR_EXTERNAL` pointer kind

This is the original WP-1 plan.

- Add a new value to `enum ptd_pcg_ptr_kind` in `src/c/phasic.c:2881+`:
  `PTD_PCG_PTR_EXTERNAL = 3`. Payload: `uint32_t external_table_index`.
- The save function takes an auxiliary `external_anchors[]` array
  that says "this synthetic-edge pointer corresponds to external
  table slot `k`".
- The load function takes an auxiliary `external_table[]` array
  (a `double *`) that the composer populates with parent-supplied
  coefficient values before triggering the replay.
- Bumps `PTD_PCG_FORMAT_REVISION` from 1 to 2.

**Pros:**
- The cache file is *parent-independent*: same content hash → same
  bytes on disk, regardless of which parent originally produced
  the entry.
- Composition writes parent values into the table once per
  composition; replay reads them. No graph-side coupling.

**Cons:**
- New format revision; v1 readers cannot read v2 files (forward
  incompatibility, but acceptable since v2 is a strict superset).
- Composer must allocate and lifecycle the external table per
  per-SCC PRC use.
- Adds complexity to the save function: it must know which
  pointers are placeholders vs. real internal-edge references.

### 2.2 Design B: reuse `PTD_PCG_PTR_EDGE`, resolve against parent at load

In this design, the synthetic graph's placeholder edges *don't
exist* on disk as a separate concept. Instead:

- At save time, the per-SCC PRC encodes synthetic placeholder
  edges as `EDGE` pointers with synthetic vertex/edge indices.
- At load time, instead of resolving against the synthetic graph
  (which we no longer have at composition time), resolve against
  a *substitute graph* that we construct on the fly: a graph
  whose edge slots are the parent's actual external edges.

**Pros:**
- No format change; existing readers/writers unchanged.
- Conceptually simpler: only one pointer kind, one resolution path.

**Cons:**
- The on-disk file's `EDGE` references are encoded with synthetic
  graph vertex indices, so the loader must keep a translation
  table from synthetic-graph vertex IDs to "where to find that
  edge weight at composition time."
- Asymmetry: internal-edge `EDGE` pointers resolve to the
  synthetic graph (which we *do* have at compose time, since the
  composer rebuilds the synthetic graph anyway); placeholder-edge
  `EDGE` pointers resolve to the parent graph. Two semantically
  different resolutions of the same pointer kind, distinguished
  only by which synthetic-vertex index the edge belongs to.
- The cache file *is still parent-independent* — the synthetic
  vertex indices are intrinsic to the SCC, not to a specific
  parent — so this advantage isn't a tiebreaker.

### 2.3 Design C: don't cache the per-SCC PRC bytes; cache something else

A third option deferred for now: don't try to make the per-SCC
PRC bytes parent-independent. Instead, cache the *symbolic
elimination structure* (operations and their dependencies) in a
form that doesn't reference any particular memory layout, and
re-instantiate the pointer scheme at load time.

This is essentially what the Python `EliminationTrace` does. It
would be a much larger change — effectively the alternative path
the user already considered when discussing the trace pipeline.
Not a realistic option for this branch given the "C-native" goal.

### 2.4 Decision

**Design A** (new `EXTERNAL` pointer kind). Reasons:

1. The semantics are explicit. A reader of the cache file can tell
   that a pointer is an "external slot" without inspecting the
   synthetic graph topology to decide whether the edge is a
   placeholder.
2. The composer's binding is local and easy to reason about: write
   N values into a table, replay, done. No special-casing of
   pointer indices by category.
3. The format revision bump is small. The cache file remains a
   single self-describing blob; the loader's interface gains one
   parameter (`external_table`).
4. Design B's asymmetry — same pointer kind with two resolution
   semantics depending on which vertex index it references — is a
   trap. Future maintainers reading the cache loader would need to
   consult the synthetic-graph topology to understand which slot
   maps where. The clarity cost outweighs the format-stability win.

I'll proceed with Design A.

## 3. Format revision details

### 3.1 New pointer kind

In `src/c/phasic.c` near line 2881:

```c
enum ptd_pcg_ptr_kind {
    PTD_PCG_PTR_NULL = 0,
    PTD_PCG_PTR_MEM = 1,
    PTD_PCG_PTR_EDGE = 2,
    PTD_PCG_PTR_EXTERNAL = 3,  /* new: external table slot */
};
```

### 3.2 Disk layout

The existing `struct ptd_pcg_disk_ptr` (line 2899) is 8+8+4+4+8 =
32 bytes:

```c
struct ptd_pcg_disk_ptr {
    uint8_t  kind;
    uint8_t  pad[7];
    int64_t  doubles_offset;            /* MEM kind */
    uint32_t vertex_idx;                /* EDGE kind */
    uint32_t edge_idx;                  /* EDGE kind */
    int64_t  byte_offset_from_edge_weight;  /* EDGE kind */
};
```

For `EXTERNAL`, reuse the `vertex_idx` slot to hold the
external-table index. The other fields are written as 0. This
preserves the 32-byte size and avoids any layout change at the
struct level.

```c
/* EXTERNAL encoding:
 *   kind = PTD_PCG_PTR_EXTERNAL
 *   vertex_idx = external_table_index
 *   doubles_offset, edge_idx, byte_offset_from_edge_weight = 0
 */
```

### 3.3 Format revision

Bump `PTD_PCG_FORMAT_REVISION` from 1 to 2. Files written at
revision 2 are not loadable by revision-1 readers (the loader
would encounter `kind = 3` and not know how to resolve it). Files
written at revision 1 remain loadable by revision-2 readers if
they contain no `EXTERNAL` pointers (revision 2 is a strict
superset of revision 1).

In practice, this means: existing Stage A2 cache files (revision 1)
keep working. New per-SCC cache files (revision 2) are only
written by the new code path. The user's `~/.phasic_cache/` may
contain a mix; that's fine.

## 4. New API

Two new functions, both in `api/c/phasic.h`:

```c
/**
 * Save a per-SCC parameterised reward compute graph with EXTERNAL
 * pointer support.
 *
 * Like ptd_save_parameterized_reward_compute_graph, but takes an
 * additional list of external anchors describing which pointers
 * in the compute graph correspond to external slots. Any pointer
 * matching an entry in the anchor list is encoded as PTD_PCG_PTR_EXTERNAL
 * with that entry's index. All other pointers are encoded as
 * MEM/EDGE/NULL as in v1.
 *
 * @param path Destination cache file.
 * @param compute Symbolic compute graph.
 * @param graph The graph from which compute was built (typically
 *              the synthetic graph). Used to resolve EDGE pointers.
 * @param external_anchors Array of double* pointers; pointers in
 *              compute matching one of these get encoded as
 *              EXTERNAL with the matching array index.
 * @param n_external Length of external_anchors.
 * @return 0 on success, -1 on error.
 */
int ptd_save_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph,
        const double *const *external_anchors,
        size_t n_external);

/**
 * Load a per-SCC parameterised reward compute graph with EXTERNAL
 * pointer support.
 *
 * Like ptd_load_parameterized_reward_compute_graph, but takes an
 * external_table array that EXTERNAL pointers are resolved against.
 *
 * @param path Cache file path.
 * @param graph The graph to bind EDGE pointers against (typically
 *              a freshly-rebuilt synthetic graph at load time).
 * @param external_table Array of doubles, indexed by external
 *              table index. The loaded compute graph's EXTERNAL
 *              pointers are resolved to &external_table[index].
 *              Caller owns; must outlive the returned compute
 *              graph.
 * @param n_external Length of external_table.
 * @return Newly allocated compute graph, or NULL on error.
 */
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph_ex(
        const char *path,
        const struct ptd_graph *graph,
        const double *external_table,
        size_t n_external);
```

The existing v1 functions
(`ptd_save_parameterized_reward_compute_graph`,
`ptd_load_parameterized_reward_compute_graph`) remain unchanged.
They write/read v1 files; the new `_ex` functions write/read v2
files. The loader auto-detects which is which from the header
revision and refuses if mismatched.

## 5. Encoding logic

### 5.1 Save path

`ptd_save_parameterized_reward_compute_graph_ex` follows the same
flow as v1, with one extra step in `ptd_pcg_encode_ptr`:

```c
static void ptd_pcg_encode_ptr_ex(
        const double *ptr,
        const struct ll_of_a *mem_chain,
        const struct ptd_pcg_edge_anchor *edge_anchors,
        size_t n_edge_anchors,
        const double *const *external_anchors,
        size_t n_external,
        struct ptd_pcg_disk_ptr *out)
{
    memset(out, 0, sizeof(*out));
    if (ptr == NULL) { out->kind = PTD_PCG_PTR_NULL; return; }

    /* Try external anchors first (cheap O(n_external) scan). */
    for (size_t i = 0; i < n_external; ++i) {
        if (ptr == external_anchors[i]) {
            out->kind = PTD_PCG_PTR_EXTERNAL;
            out->vertex_idx = (uint32_t)i;
            return;
        }
    }

    /* Fall through to v1 logic: try mem chain, then edge anchors. */
    /* ... existing ptd_pcg_encode_ptr body ... */
}
```

The v1 encoder is preserved as-is for the v1 save function; the
v2 save function calls the new variant.

### 5.2 Load path

`ptd_load_parameterized_reward_compute_graph_ex` follows the v1
flow, with one extra case in pointer resolution:

```c
case PTD_PCG_PTR_EXTERNAL: {
    if (disk_ptr.vertex_idx >= n_external) {
        /* Bad table index — corrupt cache or wrong external_table. */
        snprintf(ptd_err, ..., "external index out of range");
        goto fail;
    }
    *resolved_out = (double *)&external_table[disk_ptr.vertex_idx];
    break;
}
```

If the file is revision 1, the loader rejects any
`PTD_PCG_PTR_EXTERNAL` it encounters (it shouldn't, since v1 never
wrote that kind). If the file is revision 2 but `external_table`
is NULL or `n_external` is 0 and `EXTERNAL` pointers are present,
the loader fails with a clear error.

## 6. Building synthetic-PRC + external-anchor list

WP-3 also needs a small helper that, given a synthetic graph and
its `ptd_scc_synthetic_metadata`, produces the `external_anchors[]`
array suitable for `ptd_save_parameterized_reward_compute_graph_ex`.
The anchors are pointers into the synthetic graph's placeholder
edge coefficient arrays.

```c
/**
 * Build an external-anchor list for a synthetic graph.
 *
 * Walks the synthetic graph and collects pointers to the
 * coefficient[0] field of every placeholder edge (Type A
 * synth-source-to-internal and Type C internal-to-synth-absorbing
 * edges). The result is a flat array suitable for passing as
 * external_anchors to ptd_save_parameterized_reward_compute_graph_ex.
 *
 * The order of anchors in the returned array matches the order in
 * which the metadata's external_in_edges and external_out_edges
 * are walked (synthetic vertex index, then per-vertex insertion
 * order).
 *
 * @param synth Synthetic graph.
 * @param meta Metadata produced by ptd_scc_build_synthetic_graph.
 * @param n_anchors_out Receives the number of anchors.
 * @return Newly allocated array of double*; caller must free().
 */
double **ptd_scc_collect_external_anchors(
        const struct ptd_graph *synth,
        const struct ptd_scc_synthetic_metadata *meta,
        size_t *n_anchors_out);
```

The anchor count equals `n_uc + (count of dual-category vertices'
out-edges) + (count of pure-downstream vertices' out-edges)` —
i.e. one anchor per Type A edge and one per Type C edge. The
metadata's `external_in_edges_lengths[v_synth]` summed over
upstream-connecting `v_synth` plus `external_out_edges_lengths[v_synth]`
summed over downstream-connecting `v_synth` gives the total —
*not* counting parent edges (each Type A/C synthetic edge
corresponds to exactly one anchor, even if the parent has
multiple external edges into the same internal vertex).

Wait — let me think about this more carefully. The synthetic graph
has:
- One Type A edge per upstream-connecting vertex (synthetic source
  -> that vertex). So `n_uc` Type A edges.
- One Type C edge per internal vertex that has any external
  out-edge (whether or not it also has external in-edges). So a
  count equal to `(set of internal vertices with at least one
  parent-external out-edge)`.

The number of *anchors* equals the number of synthetic placeholder
edges. The number of *parent edges* feeding through each anchor
varies (could be many parent edges feeding one upstream-connecting
vertex's single Type A anchor). At composition time, the composer
sums those parent edge weights and writes the sum into the
anchor's external table slot.

This per-edge-aggregation is the composer's responsibility (WP-5).
WP-3 only needs to provide:
1. A way to identify which synthetic-graph pointers are placeholder
   anchors at save time (Design A's `external_anchors`).
2. A way to inject parent-supplied values at load time (Design A's
   `external_table`).

The mapping from "external table slot" to "set of parent edges that
contribute to it" lives in the metadata (`external_in_edges` and
`external_out_edges`) plus the convention that anchor index `k`
corresponds to a specific synthetic edge.

## 7. File layout

| Path | Change |
|---|---|
| `api/c/phasic.h` | +60 lines: declarations for `_ex` variants and the anchor collector. |
| `src/c/phasic.c` | +200 lines: extended encoder, extended decoder, format-revision dispatch. Modify `enum ptd_pcg_ptr_kind` and bump `PTD_PCG_FORMAT_REVISION`. |
| `src/c/scc_synthetic.c` | +60 lines: implement `ptd_scc_collect_external_anchors`. |
| `tests/pytest/test_scc_prc_external.py` | new (~200 LOC): round-trip tests. |

## 8. Tests

### 8.1 Round-trip on a single SCC

```python
def test_save_load_roundtrip_single_scc():
    """Build a synthetic graph, eliminate it, save the PRC with
    external anchors, load it with a populated external_table,
    replay, verify the result matches a direct-memory replay
    using the same external values."""
```

### 8.2 Format revision compatibility

```python
def test_v1_file_loads_with_v2_loader():
    """A v1 cache file (no EXTERNAL pointers) must load via the
    v2 loader path, ignoring the n_external parameter."""

def test_v2_file_rejects_v1_loader():
    """A v2 file containing EXTERNAL pointers must fail to load
    via the v1 loader (clear error, no crash)."""
```

### 8.3 Anchor collection

```python
def test_collect_external_anchors_count():
    """For every toy variant, the number of collected anchors
    equals the number of synthetic placeholder edges (n_uc + count
    of distinct downstream-connecting vertices)."""
```

### 8.4 Smoke: external_table substitution affects elimination

```python
def test_external_table_substitution_changes_result():
    """Load the same v2 cache file twice with different
    external_table values, verify the elimination produces
    different numerical results — confirming the binding is
    actually live."""
```

## 9. Out of scope for WP-3

- Building the per-SCC cache directory entries (`scc_<hash>.bin`).
  WP-4.
- Wiring the parent's external coefficient values into the
  external_table at composition time. WP-5.
- Caching v2 PRCs alongside v1 in `~/.phasic_cache/parameterized_reward_compute/`.
  WP-4 decides the layout.
- Clearing or migration of pre-existing v1 cache entries. v1 files
  remain readable indefinitely; no migration needed.

## 10. Estimated scope

~520 LOC total: 60 in `api/c/phasic.h`, 200 in `src/c/phasic.c`,
60 in `src/c/scc_synthetic.c`, 200 in tests. About half a day of
focused work.

## 11. Open questions

1. **Should `external_anchors` accept duplicates?** If two
   placeholder edges happen to point to the same coefficient slot
   (shouldn't happen with WP-1's `should_free_coefficients = true`
   per-edge allocation, but worth verifying), what's the encoding?
   I'll assume no duplicates and assert; will revisit if WP-1's
   actual behaviour differs.
2. **What if `n_external == 0`?** The `_ex` save and load functions
   should degrade gracefully to v1-equivalent behaviour and write
   a v2 file with zero EXTERNAL pointers. (Useful edge case for
   SCCs with no external connections — pure isolated SCCs.)
3. **Should the format revision flag be per-pointer-kind or
   global?** Going with global for simplicity; a v2 file may
   contain only MEM and EDGE pointers if no EXTERNAL was used,
   which still parses correctly.
