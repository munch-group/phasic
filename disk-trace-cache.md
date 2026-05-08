# Plan: Disk-persistent symbolic elimination cache

## Context

After Stage A0 + A1 (landed 2026-05-05) the symbolic
`parameterized_reward_compute_graph` is built once per (thread,
GraphBuilder) and reused across SVGD theta calls within a process,
delivering ~32× per-theta speedup on the moments path. **Each fresh
process still pays the O(n^3) Gaussian elimination cost on the first
call**, however, because the cache is in-memory only.

This plan makes the cache process-persistent: serialise
`parameterized_reward_compute_graph` to disk under `~/.phasic_cache/`
keyed by graph content hash, and load it on first use in any
subsequent process. Benefits:

- Repeated SVGD invocations on the same model skip elimination after
  the first build.
- SLURM multi-node workers share the cache via the network filesystem
  (each worker still keeps its own thread-local in-process state, but
  the disk cache primes that state on first use).
- Notebook iteration: rebuild a model in a new kernel, the elimination
  is loaded from disk in milliseconds instead of seconds.

JAX compatibility, `update_weights`, `reward_transform`, and the
in-memory persistent graph are all unchanged. The disk cache is a
strict additive optimisation inside `ptd_precompute_reward_compute_graph`.

## What this plan is — and isn't

**Is**: a C-level disk serialisation for
`struct ptd_desc_reward_compute_parameterized` (the symbolic
elimination output, declared in `api/c/phasic.h:405`). New save/load
functions, hashed cache key, atomic write-then-rename, env-var
opt-out.

**Isn't**: a generic graph cache. The persistent in-memory `Graph`
machinery (Stage A1) handles the *hot* path. Stage A2 only addresses
the *first* call in each process — that's expensive once, then
amortised forever per process.

## Design

### Cache key

`ptd_graph_content_hash(graph)` (`src/c/phasic_hash.c:216`) returns a
deterministic SHA-256 over (vertices, states, edges, coefficients,
edge_mode). Theta-independent. Already used elsewhere; reuse it
directly. Cache file path:

```
~/.phasic_cache/parameterized_reward_compute/<hash_hex>.bin
```

### On-disk format

The struct has four parts, three trivially serialisable and one
needing a fix-up table:

| Part | Storage | Notes |
|---|---|---|
| `length` (size_t) | direct write | |
| `commands` array | each command serialised as: `{type, from, to, multiplier, fromT_offset, toT_offset, multiplierptr_kind, multiplierptr_payload}` | Pointers replaced by offsets; see fix-up details below |
| `mem` (linked list of `double[32768]` chunks) | flatten to `[chunk_count, chunk0_len, chunk0_data, chunk1_len, …]` | Loader rebuilds `ll_of_a` chain or coalesces into one big block (decision below) |
| `memr` (`double **` indexed by vertex) | each entry stored as a `mem_offset` (which `mem` slot it points into) | One offset per vertex |

#### Pointer fix-up table — three kinds

Each `ptd_comp_graph_parameterized` has three pointer fields. They
fall into one of three categories that need different storage:

1. **`fromT` and `toT`**: always point inside `mem`. Store as
   `mem_offset` = (chunk_index, slot_index), or as a single
   global-flat `slot_index` if we choose the single-block layout.
2. **`multiplierptr` pointing inside `mem`** (used by `INV`,
   `ONE_MINUS`, `DIVIDE`, etc.): same fix-up as fromT/toT.
3. **`multiplierptr` pointing at `&edge->weight`** (used by `PP`
   commands): store as `(vertex_idx, edge_idx)` — re-pointed at load
   time against the live graph's edge array.

A single byte tag per command (`multiplierptr_kind`) distinguishes
"points into mem" from "points into edge weights" from "unused".
`multiplierptr_payload` is a `(uint32, uint32)` pair that carries
either a mem offset or an `(vertex_idx, edge_idx)` pair depending on
the tag.

#### `mem` layout decision: single block vs. linked-list chain

The current `mem` is a linked list of 32768-double chunks
(`src/c/phasic.c:1315`). On disk we have two options:

- **Option A — Coalesce into a single allocation.** Loader does one
  `calloc(total_slots * sizeof(double))`, builds a single `ll_of_a`
  node wrapping it. Simpler; one less malloc per node.
- **Option B — Preserve the linked-list structure.** Loader rebuilds
  the chain block-for-block. Matches the current allocation pattern
  exactly.

**Recommendation: Option A** (coalesce). The chain structure exists
only because the recorder grows incrementally; the consumer
(`ptd_graph_build_ex_absorbation_time_comp_graph_parameterized`) and
the destroy function don't care about chunk boundaries. A single
contiguous block is simpler to serialise and faster to allocate. The
loader still wraps it in one `ll_of_a` node so the destroy function
works unchanged.

#### Endianness and binary stability

phasic builds are platform-specific (we don't ship cross-platform
binaries; users build via `pixi run install-dev` per machine). The
cache is per-machine. Use native-endian `fwrite`/`fread`. Include a
short header to detect format mismatches:

```
struct ptd_param_compute_disk_header {
    char magic[8];               // "PTDPRMC1"
    uint32_t version;             // 1
    uint32_t format_revision;     // 1
    uint64_t graph_hash_truncated; // first 8 bytes of hash for sanity check
    uint64_t commands_length;     // == res->length
    uint64_t mem_total_doubles;   // sum of all chunk lengths
    uint64_t memr_length;         // == graph->vertices_length
};
```

If `magic` doesn't match or `version`/`format_revision` differs from
the loader's expectation, treat as a cache miss and rebuild. (Don't
attempt schema migration.)

### New C functions

In `src/c/phasic.c`:

```c
/**
 * Save a symbolic compute graph to disk.
 *
 * Walks `compute->commands` building a pointer-fix-up table,
 * flattens `compute->mem` to a single-block layout, writes the
 * header + serialised body to a temp file, fsyncs, and atomically
 * renames into place.
 *
 * @param path Cache file path, e.g.
 *             ~/.phasic_cache/parameterized_reward_compute/<hash>.bin
 * @param compute The symbolic compute graph to save
 * @param graph The graph the compute came from (needed for
 *              edge-pointer fix-up — we serialise multiplierptr-into-
 *              edge-weight as (vertex_idx, edge_idx))
 * @return 0 on success, -1 on error (sets ptd_err)
 */
int ptd_save_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_desc_reward_compute_parameterized *compute,
        const struct ptd_graph *graph);

/**
 * Load a symbolic compute graph from disk.
 *
 * Reads the header, validates magic/version, allocates a single-
 * block mem buffer, reads commands and re-points all `fromT`,
 * `toT`, and `multiplierptr` references against the loaded mem and
 * the supplied graph's edge weights.
 *
 * @param path Cache file path (must exist)
 * @param graph The graph to bind multiplierptr-into-edge fields to
 * @return Newly allocated compute graph (caller owns via
 *         ptd_parameterized_reward_compute_graph_destroy), or NULL
 *         on error (sets ptd_err). Returns NULL silently for cache
 *         miss; the caller distinguishes between "file not present"
 *         (a cache miss; expected) and "file present but corrupt"
 *         (warning; rebuild).
 */
struct ptd_desc_reward_compute_parameterized *
ptd_load_parameterized_reward_compute_graph(
        const char *path,
        const struct ptd_graph *graph);
```

### Cache lookup site

Inside `ptd_precompute_reward_compute_graph`
(`src/c/phasic.c:1777`), at the spot where the symbolic elimination
runs (around line 1826):

```c
if (graph->parameterized_reward_compute_graph == NULL) {
    /* Try disk cache first */
    if (!getenv_disable_cache()) {
        char path[PATH_MAX];
        if (build_param_compute_cache_path(graph, path, sizeof(path)) == 0) {
            graph->parameterized_reward_compute_graph =
                    ptd_load_parameterized_reward_compute_graph(path, graph);
        }
    }

    /* Cache miss — run the elimination */
    if (graph->parameterized_reward_compute_graph == NULL) {
        if (graph->use_dyn_ordering) {
            graph->parameterized_reward_compute_graph =
                    ptd_graph_ex_absorbation_time_comp_graph_parameterized_dyn(graph);
        } else {
            graph->parameterized_reward_compute_graph =
                    ptd_graph_ex_absorbation_time_comp_graph_parameterized(graph);
        }

        /* Populate disk cache for next process */
        if (!getenv_disable_cache()) {
            char path[PATH_MAX];
            if (build_param_compute_cache_path(graph, path, sizeof(path)) == 0) {
                /* Best-effort save; failures are warnings only */
                ptd_save_parameterized_reward_compute_graph(
                        path, graph->parameterized_reward_compute_graph, graph);
            }
        }
    }
}
```

Helpers:

- `build_param_compute_cache_path(graph, buf, len)`: combines the
  existing `get_cache_dir`-style helper with `<hash>.bin`. Creates
  `~/.phasic_cache/parameterized_reward_compute/` on first use.
- `getenv_disable_cache()`: matches the Python convention
  (`PHASIC_DISABLE_CACHE=1` per `src/phasic/trace_serialization.py:81`).

### Concurrency

- **Write race**: two processes building the same hash race to write
  the same file. Use `O_EXCL | O_CREAT` on the temp file path then
  atomic rename — last writer wins; both produce identical content
  so the result is bit-identical. No corruption.
- **Read while write**: atomic rename guarantees readers see either
  the old file or the new one, never a partial write. Loader can
  always proceed.
- **Same-process concurrency**: serialised by the per-graph
  `compute_graph_lock` (Stage 3.2) — only one thread per graph
  reaches the cache code path at a time.
- **Cross-process file lock not needed**: the worst case is two
  processes both write the same content; that's wasted work, not
  corruption.

### Invalidation

The hash includes graph topology + coefficients. Any structural or
coefficient change produces a new key — never an in-place
invalidation. Stale entries from old format versions are detected by
the `magic`/`version` header check and treated as cache miss.

The cache directory may grow over time. Provide a Python helper
`phasic.cache.clear_param_compute_cache()` (rmtree on the
sub-directory). Do not auto-prune.

### Failure modes

| Failure | Behaviour |
|---|---|
| `HOME` not set | `build_param_compute_cache_path` returns -1; cache disabled silently |
| Cache dir not writable | save fails silently; load is unaffected; first call rebuilds and tries to save (which then warns once) |
| Disk full during save | partial temp file removed; rename never happens; correctness preserved |
| Corrupt cache file | header check fails → cache miss → rebuild + overwrite |
| Format version mismatch | header check fails → cache miss |
| `PHASIC_DISABLE_CACHE=1` | both load and save are skipped |

## Stages

### Stage B1 — Save / load functions, in isolation

Touch points:

- `src/c/phasic.c`: `ptd_save_parameterized_reward_compute_graph`,
  `ptd_load_parameterized_reward_compute_graph`,
  `build_param_compute_cache_path`, `getenv_disable_cache`. Plus a
  small helper to flatten the `mem` linked list into a contiguous
  buffer, and the inverse.
- `api/c/phasic.h`: declare the two public save/load functions.
  Internal helpers stay file-static.

The functions are testable in isolation: build a graph, run
elimination, save, destroy graph + recreate same graph from same JSON,
load, run replay, assert results match.

**Acceptance**: `tests/pytest/inference/test_disk_param_cache.py`'s
"save → reload → bit-identical replay" test passes (in-process; no
subprocess yet).

### Stage B2 — Hook into `ptd_precompute_reward_compute_graph`

Touch points:

- `src/c/phasic.c`: insert the cache-lookup-then-rebuild-then-save
  block at the symbolic-elimination call site (current line ~1826).
  Both `_dyn` and non-`_dyn` ordering variants take the same path.
  Honour `PHASIC_DISABLE_CACHE`.

**Acceptance**:

- Subprocess test: spawn process A that builds the graph and warms
  the cache, then process B that builds the same graph and is
  observed (via `_has_param_compute_graph_cache` + a timing assertion
  or a count-the-eliminations debug counter) to skip the elimination.
- All 314 existing inference tests still pass.
- Stage A1 verification still passes (in-memory cache and disk
  cache compose correctly: in-memory hits short-circuit before disk
  is consulted).

### Stage B3 — Python-side cache management

Touch points:

- `src/phasic/cache.py` (new file or extend `trace_serialization.py`):
  `clear_param_compute_cache()`, `param_compute_cache_size()`,
  `param_compute_cache_list()`. Thin wrappers around filesystem ops.
- `src/cpp/phasic_pybind.cpp`: optional bindings for cache inspection
  (size, count, clear). Not strictly required since the cache is
  transparent.

**Acceptance**: `clear_param_compute_cache()` removes all files;
`param_compute_cache_size()` returns total bytes.

## Stages C — Tests

### `tests/pytest/inference/test_disk_param_cache.py`

- **Save/load round trip in same process**: build, save, drop, load,
  compare replay output against fresh build. Bit-identical.
- **Subprocess round trip**: subprocess A populates cache; subprocess
  B loads and runs forward; results match A's.
- **Hash stability**: same Python-side graph construction → same
  hash → same cache file. Mutating any coefficient changes the hash.
- **Format-version mismatch**: write a file with wrong magic, verify
  load returns NULL and rebuild succeeds (overwriting the bad file).
- **`PHASIC_DISABLE_CACHE=1`**: cache is neither read nor written.
- **`HOME` unset**: cache disabled, computation unaffected.
- **In-memory + disk cache compose**: first call hits disk; second
  call (same process) hits in-memory; both produce identical results.
- **Concurrent process writers**: spawn two processes that both build
  the same uncached graph; assert the final cache file is valid and
  both processes produced correct results (no corruption from the
  race).

### `tests/pytest/inference/bench_disk_param_cache.py`

Not part of the regular pytest run. Measures:

- 200-vertex synthetic model: time from fresh process start to first
  forward result, with and without disk cache.
- Cache hit rate over 50 sequential SVGD-style runs (each in a fresh
  subprocess) on 10 different models.

## Critical files

- **`src/c/phasic.c`** — new save/load functions; cache hook in
  `ptd_precompute_reward_compute_graph`.
- **`api/c/phasic.h`** — declare new public save/load API.
- **`src/cpp/phasic_pybind.cpp`** — optional Python bindings for
  cache inspection.
- **`tests/pytest/inference/test_disk_param_cache.py`** — new file.
- **`tests/pytest/inference/bench_disk_param_cache.py`** — new
  benchmark, not part of pytest default.

## Existing utilities to reuse

- `ptd_graph_content_hash` (`src/c/phasic_hash.c:216`) — deterministic
  SHA-256 over graph topology + coefficients. Theta-independent. Use
  `result->hash_hex` (64-char string) as the file-name stem.
- `get_cache_dir`-style pattern (`src/c/phasic.c:357`) — clone for
  the new sub-directory `parameterized_reward_compute/`.
- `~/.phasic_cache/` root convention
  (`src/phasic/trace_serialization.py:232`).
- `PHASIC_DISABLE_CACHE=1` env var
  (`src/phasic/trace_serialization.py:81`).
- Per-graph `compute_graph_lock` (Stage 3.2,
  `api/c/phasic.h:180`) — already serialises in-process readers, no
  extra locking needed.
- `ptd_parameterized_reward_compute_graph_destroy`
  (`src/c/phasic.c:2712`) — works on loaded compute graphs unchanged
  because the loader sets `mem` to a `ll_of_a` chain (single node
  wrapping the coalesced block).

## Verification

```
pixi run -- pytest \
  tests/pytest/inference/test_disk_param_cache.py \
  tests/pytest/inference/test_persistent_graph_cache.py \
  tests/pytest/inference/test_trace_vs_direct.py \
  tests/pytest/inference/test_trace_jax_compat.py \
  tests/pytest/inference/test_multivariate_correctness.py \
  -v
```

Plus the full inference suite to confirm no regression on the 314
passing baseline established after Stage A1:

```
pixi run -- pytest tests/pytest/inference/ -q
```

Manual benchmark (not part of pytest):

```
pixi run -- python tests/pytest/inference/bench_disk_param_cache.py
```

## Acceptance criteria across the whole plan

When all stages land:

- Save → reload → replay produces bit-identical results to fresh
  elimination on a battery of graph sizes (10, 50, 200 vertices).
- Subprocess B loads cache populated by subprocess A, no rebuild.
- All 314 inference tests still pass.
- Stage A1's `test_persistent_graph_cache.py` still passes (in-memory
  fast path doesn't regress).
- `PHASIC_DISABLE_CACHE=1` round-trips identically (cache neither
  read nor written; correctness unchanged).
- Benchmark on 200-vertex synthetic model: subsequent-process startup
  cost dropped from "elimination time" to "load time" — at least an
  order of magnitude on n ≥ 200.

## Out of scope

- **The in-memory persistent graph (Stage A1)**. Already landed.
- **Caching the *concrete* `reward_compute_graph`**. It depends on
  current edge weights (theta), so it would need re-keying per theta
  — not worth it; the rebuild path from the symbolic cache is fast.
- **Cross-machine cache portability**. Phasic builds are
  platform-specific; the cache is, too. The header check fails on
  format mismatch, so the worst-case is "rebuild and overwrite".
- **Cache eviction / size cap**. `~/.phasic_cache/` is user-owned;
  the user (or `clear_param_compute_cache()`) manages it. We don't
  auto-prune.
- **Compression**. The compute graph is a few KB to a few MB for the
  models we care about. zstd or similar could shrink it 3-5×, but
  the disk-cache path is dominated by parse+fix-up, not bytes-on-disk.
  Skip until profiles say otherwise.
- **`reward_compute_graph_mpfr`**. The MPFR variant has its own
  caching; out of scope here. Could follow the same pattern in a
  later sub-stage if needed.
- **Python-side `cache_param_compute=True/False` toggle**. The cache
  is on by default and controlled by `PHASIC_DISABLE_CACHE`. A
  per-call toggle would require a thread-local override; YAGNI until
  someone asks for it.

## Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pointer fix-up table is wrong → segfault on replay | Medium | Stage B1 isolated tests catch this before B2 hooks the cache into the live forward path. |
| Hash collision (different graph, same hash) | Negligibly low (SHA-256) | Header includes 8-byte hash prefix as a sanity check; loader can also re-validate vertex count + edge count match the live graph and fall back to rebuild on mismatch. |
| File-format change breaks old caches silently | Medium | Header `version`/`format_revision`. Bump on every layout change; loader treats mismatch as cache miss. |
| Race between two processes writing the same file | Low (small window) | Atomic write-then-rename. Both processes produce identical bytes; last writer wins, no corruption. |
| Disk cache disabled in CI but tests assume it | Low | `PHASIC_DISABLE_CACHE=1` is the test env default for tests that exercise the no-cache path; tests that need the cache populate it explicitly. |
| Stage A1's in-memory cache hides a Stage A2 bug | Medium | Test must invoke compute through a *fresh* `GraphBuilder` (and ideally a fresh subprocess) so the in-memory cache is empty and the disk path is the only source of truth. |

## Estimated effort

- Stage B1 (save/load isolation): ~1 day. Pointer fix-up logic is
  the main complexity; once the format is settled, the code is
  mechanical.
- Stage B2 (hook into precompute): ~half a day. Small change at one
  site, with the env-var gate.
- Stage B3 (Python helpers): ~half a day. Thin wrappers.
- Tests + benchmark: ~half a day.

Total: 2–3 days of focused work. Risk-adjusted: 4 days including
debugging the pointer fix-up under valgrind / ASan.
