# WP-4 — per-SCC PRC compute and disk cache (detailed plan)

**Branch:** `hierar-elimin-cache`
**Status:** drafted
**Predecessors:** WP-3 (commit `ff2c136`)
**Successor:** WP-5 (composition)

## 1. Goal

A C function `ptd_scc_get_or_compute_prc` that, given an SCC,
returns its parameterised reward compute graph (PRC), hitting the
on-disk cache when possible and computing-then-saving on miss.
This is the per-SCC analogue of the existing
`ptd_precompute_reward_compute_graph`, which operates on whole
parent graphs.

```c
struct ptd_desc_reward_compute_parameterized *
ptd_scc_get_or_compute_prc(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_graph **synth_out,
        struct ptd_scc_synthetic_metadata **metadata_out);
```

Returns the PRC, plus the synthetic graph and metadata that the
caller (WP-5's composer) will need.

## 2. The cache key

Per §6.1 of the reference doc, per-SCC PRCs live in the same
directory as Stage A2 (parent-level) PRCs, distinguished by
filename prefix:

```
~/.phasic_cache/parameterized_reward_compute/
    <parent_hash_hex>.bin       # parent (rev-1, existing Stage A2)
    scc_<scc_hash_hex>.bin      # per-SCC (rev-2 with EXTERNAL pointers)
```

The SCC hash is `ptd_graph_content_hash` of the synthetic graph —
which is what we already verified in WP-2 to be cross-parent
invariant.

## 3. Algorithm

1. Build the synthetic graph + metadata via
   `ptd_scc_build_synthetic_graph`.
2. Compute the synthetic graph's content hash via
   `ptd_graph_content_hash`.
3. Construct the cache file path:
   `~/.phasic_cache/parameterized_reward_compute/scc_<hash_hex>.bin`.
4. Try to load via `ptd_load_parameterized_reward_compute_graph_ex`.
   - On hit: return the loaded PRC alongside synth + metadata.
     The synth is needed because the loaded PRC's EDGE pointers
     reference synth's edge weights.
   - On miss: continue.
5. Run `ptd_graph_ex_absorbation_time_comp_graph_parameterized`
   on the synthetic graph to produce the PRC.
6. Collect external anchors via
   `ptd_scc_collect_external_anchors`.
7. Save the PRC via
   `ptd_save_parameterized_reward_compute_graph_ex`.
   Best-effort: log on failure but still return the in-memory PRC.
8. Return the PRC + synth + metadata.

## 4. Cache disable

Honour `PHASIC_DISABLE_CACHE=1` (same env var as Stage A2). When
disabled, skip steps 4 and 7 — always recompute, never save.

## 5. New API

In `api/c/phasic.h`:

```c
struct ptd_desc_reward_compute_parameterized *
ptd_scc_get_or_compute_prc(
        const struct ptd_scc_graph *scc_graph,
        size_t scc_index,
        struct ptd_graph **synth_out,
        struct ptd_scc_synthetic_metadata **metadata_out);
```

Plus a Python-side helper for inspection / cache-stats:
- A pybind binding on `SCCVertex`:
  `scc.get_or_compute_prc()` returns a tuple
  `(synth_graph, metadata, prc_present_bool)` for tests; the
  PRC itself isn't directly exposed (it's a C-private struct), but
  the side effect (cache file presence) is observable.

## 6. Cache stats

Extend `phasic.cache.param_compute_cache_info()` to return
separate counts for parent (`<hash>.bin`) and SCC (`scc_<hash>.bin`)
entries. Useful for users observing cache behaviour.

## 7. Tests

In `tests/pytest/test_scc_prc_cache.py`:

- **Cold cache → save:** delete any existing `scc_*.bin`, call
  `get_or_compute_prc` on a toy SCC, verify the file appeared.
- **Warm cache → load:** call again, verify the file mtime is
  unchanged (load path, no rewrite) AND the returned PRC produces
  the same numerical result as a direct elimination.
- **Cross-parent reuse (the headline test):** call on Toy-C P's
  SCC₂, then on Toy-C P''s SCC₂. The second call must hit the
  cache (mtime unchanged after second call) because the SCC
  content hash is the same.
- **Cache disable env var:** with `PHASIC_DISABLE_CACHE=1`, no
  file is written and no file is read.
- **Toy-D aux SCC:** the aux-vertex SCC computes and caches
  correctly (regression for the duplicate-state case through the
  whole pipeline).

## 8. File layout

| Path | Change |
|---|---|
| `api/c/phasic.h` | +20 lines: declaration. |
| `src/c/scc_synthetic.c` | +120 lines: implementation. |
| `src/cpp/phasic_pybind.cpp` | +50 lines: pybind binding. |
| `src/phasic/cache.py` | +30 lines: split SCC vs parent cache stats. |
| `tests/pytest/test_scc_prc_cache.py` | new (~150 LOC). |

## 9. Out of scope

- Composition. WP-5.
- Modifying `ptd_precompute_reward_compute_graph` to use the
  hierarchical pipeline. WP-7.
- Any change to the SCC iteration order. The existing
  reverse-topological order is what WP-5/WP-7 will use.
