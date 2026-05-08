# phasic caching: in-memory and on-disk

A working reference for which cache holds what, who reads/writes it,
how the user controls each one, and what is shareable across
machines.

## Caches at a glance

| Cache | Location | What | Layer | Status |
|---|---|---|---|---|
| `parameterized_reward_compute_graph` (Stage A0) | in-memory, per-`ptd_graph` | symbolic elimination output | C | live, transparent |
| `tl_persistent_graphs` (Stage A1) | in-memory, thread-local per `GraphBuilder*` | full `phasic::Graph` object | C++ | live, transparent |
| `builder_cache` | in-memory, thread-local per JSON | parsed `GraphBuilder` instance | C++ | live, transparent |
| Param-compute disk cache (Stage A2) | `~/.phasic_cache/parameterized_reward_compute/<hash>.bin` | symbolic elimination output | C | live, transparent |
| GraphCache | `~/.phasic_cache/graphs/<callback_hash>.json` | serialised `Graph` object | Python | live, opt-in |
| Trace cache | `~/.phasic_cache/traces/<graph_hash>.json` (and `.pkl`) | Python `EliminationTrace` | Python | deprecated |
| Compiled-trace `.so` | `/tmp/trace_log_lik_<hash>.so` | compiled C++ from a trace | Python | deprecated |
| ctypes-loaded `.so` (in-memory) | `_lib_cache` dict, per process | dlopen handle for compiled `pmf_from_cpp` / `pmf_from_graph_parameterized` libs | Python | live, transparent |
| JAX compilation cache | `~/.jax_cache/` | XLA-compiled JIT artefacts | JAX | live, transparent |
| TraceRegistry / IPFS | `~/.phasic_cache/registry/` + IPFS | shared community traces | Python | live, opt-in |

Below: each cache in detail.

## In-memory caches (per-process / per-thread)

### Stage A0 — `parameterized_reward_compute_graph`
**Where:** `struct ptd_graph::parameterized_reward_compute_graph`
(C). One pointer per `ptd_graph`.
**What:** the output of `ptd_precompute_reward_compute_graph` — a
list of operations that replay the elimination given a `theta`
vector. Built lazily on the first call to a function that needs
moments / waiting time / reward-transformed quantities.
**Why it survives parameter updates:** the recorded operations
hold pointers (`multiplierptr`) directly into live edge-weight
slots. `ptd_graph_update_weights` and `ptd_graph_update_ipv` write
into those slots without invalidating the operations. The
symbolic structure depends only on topology + coefficients, which
neither operation mutates.
**Cost:** O(n³) once, then O(n) per replay.
**Invalidated by:** structural mutations (`ptd_graph_add_edge`,
`add_aux_vertex`, etc.), `ptd_graph_clone` (the clone gets none),
graph destruction.
**User control:** none. Always on. The output participates in the
A2 disk cache.

### Stage A1 — `tl_persistent_graphs`
**Where:** `src/cpp/parameterized/graph_builder.cpp:241` —
`thread_local std::unordered_map<const GraphBuilder*, unique_ptr<phasic::Graph>>`.
**What:** a full `phasic::Graph` instance per `(thread, GraphBuilder)`
pair, kept alive across calls.
**Triggered by:** `GraphBuilder::get_or_init_persistent_graph(theta)`.
On first call from a thread, builds the graph and stores it.
Subsequent calls reuse the cached graph and just call
`update_weights_parameterized(theta)` to refresh edge weights —
which preserves the A0 cache embedded in it.
**Why per-thread:** `phasic::Graph` is not thread-safe under
`update_weights` (it mutates edge weights in place), so each OS
thread under OpenMP/vmap gets its own.
**Used by:** the pybind11 `compute_pmf` / `compute_moments` etc.
methods on `GraphBuilder` (called from non-FFI Python paths). FFI
handlers do **not** use this cache — they call
`builder->build(theta, theta_len)` for a fresh graph per batch
element instead, because OpenMP threads each need an independent
graph.
**User control:** none. Always on. Cleared when the
`GraphBuilder` Python object is GC'd (its destructor erases the
thread's slot).

### `builder_cache`
**Where:** `src/cpp/parameterized/graph_builder_ffi.cpp:20` —
`thread_local std::unordered_map<std::string, shared_ptr<GraphBuilder>>`.
**What:** parsed `GraphBuilder` instances keyed by JSON string.
**Triggered by:** every FFI handler (`ComputePmfFfiImpl`,
`DaisyChainJointProbsFfiImpl`, etc.) — first call with a given
JSON parses it once, subsequent calls hit the cache.
**Why it matters:** parsing the JSON is O(n) in graph size and
happens on every FFI call (XLA invokes the handler with no
between-call state). Caching here saves O(n) JSON parsing per FFI
call.
**User control:** none. Always on, per-thread.

### `_lib_cache` (Python-side dlopen cache)
**Where:** `src/phasic/__init__.py:206` — `_lib_cache: dict`.
**What:** ctypes-loaded shared-library handles, keyed by `(callback_hash, discrete_flag)` or `(lib_name, discrete_flag)`.
**Used by:** `pmf_from_cpp` and `pmf_from_graph_parameterized`,
which compile a per-graph C++ wrapper to a `.so`, then use ctypes
to call it from inside a `jax.pure_callback`.
**Why:** prevents repeated dlopen-and-bind on every model
construction within a process.
**User control:** none. Process-lifetime.

## On-disk caches (persist across processes)

### Stage A2 — Param-compute disk cache
**Path:** `~/.phasic_cache/parameterized_reward_compute/<hash>.bin`.
**What:** the same A0 symbolic-elimination output, serialised to
a 64-byte-header binary file (magic `PTDPRMC1` + version +
truncated graph hash + lengths + the operation array).
**Cache key:** `ptd_graph_content_hash(graph)` — a SHA-256 over
graph topology + coefficients (`src/c/phasic_hash.c:216`).
Theta-independent: every Graph with the same structure hits the
same key regardless of update history.
**Triggered by:** every C-path forward that needs the symbolic
graph. The C entry point is at `phasic.c:1880` inside
`ptd_precompute_reward_compute_graph` — on a hit, `ptd_load_…`
deserialises into the in-memory A0 slot; on a miss the elimination
runs as before and writes back via `ptd_save_…`.
**User control:**
- `phasic.cache.clear_param_compute_cache()` — delete files.
- `phasic.cache.param_compute_cache_info()` — inspect.
- `PHASIC_DISABLE_CACHE=1` — skips both reads and writes.
- Format-version mismatch is detected via the file header and
  treated as a cache miss (the bad file is then overwritten).
**Sharing:** files are theta-independent SHA-256-keyed binaries —
in principle shareable across machines with the same byte order,
phasic version, and `PTD_PCG_FORMAT_REVISION`. There is no
built-in helper to publish or fetch them.
**Atomicity:** writes use write-then-rename, so concurrent writers
produce identical content (no partial files on disk).
**Saving cost:** O(n³) elimination on miss, O(file size I/O) on
hit (≈microseconds for a 67-vertex model).

### GraphCache
**Path:** `~/.phasic_cache/graphs/<callback_hash>.json`.
**What:** a serialised `Graph` object — JSON containing vertex
states, edges, parameterised-edge coefficients, IPV, etc.
**Cache key:** `phasic.callback_hash.hash_callback(callback, **construction_params)`
— an AST-level hash of the callback function plus the
construction kwargs (e.g. `nr_samples`, `theta`).
**Triggered by:** `Graph(callback_fn, ..., graph_cache=True)` —
opt-in via the `graph_cache=True` constructor flag. On hit, the
graph is rebuilt from JSON without re-running the callback. On
miss, the callback runs as normal and the result is saved.
**User control:**
- Constructor flag: `Graph(callback, graph_cache=True)`.
- `phasic.graph_cache.GraphCache().clear_graph_cache()` — clears.
- No env-var disable.
**Why opt-in:** the callback hash relies on AST stability; closures
or lambdas with captured variables can hash inconsistently across
sessions. Default-off avoids surprises.
**Sharing:** in principle, yes — the JSON is portable. But the
hash key requires **the same callback source on both ends**, so
sharing is more useful within a project than between users.

### Trace disk cache (deprecated)
**Path:** `~/.phasic_cache/traces/<graph_hash>.json` (preferred)
or `<graph_hash>.pkl` (fallback for older traces).
**What:** Python `EliminationTrace` objects — the per-vertex
operation list produced by `record_elimination_trace` (the
Python path) or the SCC-decomposed equivalent from
`hierarchical_trace_cache`.
**Cache key:** `ptd_graph_content_hash(graph)` — same SHA-256 the
A2 cache uses (topology + coefficients).
**Triggered by:** the Python `EliminationTrace` machinery.
Specifically: `Graph.compute_trace()` and friends called via the
now-disabled trace path through `Graph.moments` /
`Graph.expectation` / `Graph.variance`.
**Status: deprecated.** Nothing in the public API populates this
cache anymore (the trace path was retired from
moments/expectation/variance in favour of the C++ super() path).
The helpers remain to clean up leftover files.
**User control:**
- `phasic.cache.clear_trace_cache()` — clears, emits
  `DeprecationWarning`.
- `phasic.cache.trace_cache_info()` — inspects, emits
  `DeprecationWarning`.
- `PHASIC_DISABLE_CACHE=1` — would skip reads/writes if anything
  still wrote here.
**Sharing:** see "TraceRegistry / IPFS" below for the curated
shared trace surface; the per-machine disk cache is by-hash and
not designed for ad-hoc sharing.

### Compiled-trace `.so` (deprecated)
**Path:** `/tmp/trace_log_lik_<trace_hash>.so`.
**What:** dynamically compiled C++ libraries embedding a Python
`EliminationTrace`, used by `_wrap_trace_log_likelihood_for_jax`
to drive a `pure_callback` against a static array
representation of the trace.
**Triggered by:** `trace_to_log_likelihood(trace, …, use_cpp=True)`.
**Status: deprecated** along with the rest of the Python trace
pipeline. The compiled `.so`s land in `/tmp` and are wiped on
reboot.
**User control:** none. The `.so` files persist until `/tmp` is
cleared.

### JAX compilation cache
**Path:** `~/.jax_cache/` by default; configurable via
`JAX_COMPILATION_CACHE_DIR` env var or
`phasic.cache_manager.configure_layered_cache(local_cache_dir=…, shared_cache_dir=…)`.
**What:** XLA-compiled JIT/FFI artefacts (the compiled XLA HLO
modules, including the FFI calls phasic registers).
**Cache key:** internal to JAX/XLA — based on the traced
computation graph hash.
**Triggered by:** every `jax.jit` / `jax.pmap` (and indirectly
every `jax.grad` through `custom_vjp`) when JAX's persistent cache
is enabled. The cache is enabled when
`JAX_COMPILATION_CACHE_DIR` is set, which `phasic`'s
`CompilationConfig.apply()` does for you on import (see
`src/phasic/jax_config.py:174`).
**User control:**
- `phasic.clear_jax_cache()` (top-level export from `model_export.py`).
- `phasic.cache_manager.CacheManager(cache_dir).info()` — inspect.
- `phasic.cache_manager.configure_layered_cache(local, shared)` —
  configure local + shared (read-only) dirs.
- `phasic.CompilationConfig(cache_dir=..., shared_cache_dir=...,
  cache_strategy='layered')` — full configuration object.
- `JAX_COMPILATION_CACHE_DIR` env var — point JAX at a custom dir
  before phasic import.
**Sharing:** **yes.** This cache is the canonical "shareable" one
in phasic — `CompilationConfig.cache_strategy='layered'` is
designed for compute clusters where individual users have local
caches and a project shares a read-only cache on a network
filesystem. The JIT-compiled XLA modules are portable across
machines with the same XLA version and ABI.

### TraceRegistry / IPFS-backed traces
**Path:** local cache at `~/.phasic_cache/registry/`; remote
content addressed by IPFS hash with HTTP-gateway fallback.
**What:** a community-shared registry of pre-computed
`EliminationTrace` objects, intended for canonical population-
genetics and queueing models so users can skip the O(n³)
elimination entirely.
**Triggered by:** `phasic.trace_repository.get_trace("trace_id")`,
`TraceRegistry().list_traces()`, etc. Opt-in.
**Cache key:** `trace_id` (string) maps to an IPFS content hash;
the IPFS hash is the integrity check.
**Status: live but opt-in.** Sits in the codebase and is
documented; depends on IPFS daemon (auto-start) or a public HTTP
gateway. Note: the underlying `EliminationTrace` consumer is the
deprecated trace path — practically useful today only if the
caller drives trace functions directly. Will become more useful
again once an FFI-backed trace evaluator lands (or never, if the
trace path is fully retired).
**Sharing:** the **whole point** is sharing. Designed to be the
canonical answer to "can a colleague's expensive trace be
reused?" — yes, publish via `TraceRegistry().publish_trace(...)`
+ IPFS, then anyone can `get_trace(trace_id)`.

## Path-by-path: which caches each compute path uses

| Path | A0 | A1 | builder_cache | A2 disk | GraphCache | Trace disk | JAX cache |
|---|---|---|---|---|---|---|---|
| `Graph.expectation` / `moments` / `variance` | ✓ | — | — | ✓ | — | — | — |
| `Graph.pdf` / `stop_probability` | — | — | — | — | — | — | — |
| `pmf_from_graph` (FFI) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `pmf_and_moments_from_graph` (FFI, `use_ffi=True`) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `pmf_and_moments_from_graph` (`pure_callback` fallback, default) | ✓ | ✓ | — | ✓ | — | — | ✓ |
| `pmf_from_graph_joint_index` (FFI) | ✓ | — | ✓ | ✓ | — | — | ✓ |
| `daisy_chain_joint_probs` (FFI) | — (uses uniformization, not symbolic graph) | — | ✓ | — | — | — | ✓ |
| `pmf_from_cpp`, `pmf_from_graph_parameterized` | varies | — | — | varies | — | — | ✓ |
| `Graph(callback, graph_cache=True)` | — | — | — | — | ✓ | — | — |
| Trace path (deprecated) | — | — | — | — | — | ✓ | — |

Notes:
- "✓" = the path benefits from / writes to that cache as a side
  effect. "—" = the path doesn't touch that cache.
- `daisy_chain_joint_probs` uses uniformization
  (`stop_probability`) rather than the symbolic compute graph, so
  Stage A0/A2 are not on its critical path. The benefit it gets
  from the FFI builder cache is JSON-parsing amortisation only.
- `Graph.pdf` and `stop_probability` use a separate per-Graph
  cache for the `ptd_probability_distribution_context` (in-memory,
  invalidated by `weight_version`). It's listed as a cache only
  for completeness — it's reset by every weight update.

## User-facing cache controls — quick reference

```
# Disable/clear individual on-disk caches:
PHASIC_DISABLE_CACHE=1                                       # env: A2 + trace cache
phasic.cache.clear_param_compute_cache()                     # A2
phasic.cache.clear_trace_cache()                             # trace (deprecated; warns)
phasic.cache.clear_all_caches()                              # both above
phasic.graph_cache.GraphCache().clear_graph_cache()          # GraphCache
phasic.clear_jax_cache()                                     # JAX cache
phasic.clear_model_cache()                                   # JAX + GraphCache + legacy trace dir
phasic.clear_caches()                                        # everything above

# Inspection:
phasic.cache.param_compute_cache_info()
phasic.cache.trace_cache_info()                              # warns
phasic.graph_cache.GraphCache().get_stats()
phasic.cache_manager.CacheManager('~/.jax_cache').info()
phasic.get_all_cache_stats()                                 # all of the above

# Opt-in / configuration:
Graph(callback, graph_cache=True)                            # GraphCache
phasic.configure(ffi=True, jit=True, jax=True)               # routes through JAX cache
phasic.cache_manager.configure_layered_cache(                # layered JAX cache
    local_cache_dir='~/.jax_cache',
    shared_cache_dir='/shared/project/jax_cache',
)
phasic.CompilationConfig(...).apply()                        # full JAX cache config
JAX_COMPILATION_CACHE_DIR=/path                              # JAX cache dir before import

# Trace sharing (opt-in, deprecated downstream):
trace = phasic.trace_repository.get_trace("trace_id")
TraceRegistry().publish_trace(trace, trace_id, metadata, …)
```

## Can cached traces be shared?

Three different "trace caches" exist, with different sharing stories:

1. **Stage A2 param-compute cache** (`~/.phasic_cache/parameterized_reward_compute/`)
   — *content-addressed binary files keyed by SHA-256 of the
   graph*. Sharing is mechanically possible by copying files: same
   model + same phasic version + same byte order = same cache hit.
   No built-in publish/fetch; users would have to rsync or use a
   shared NFS mount. Format-version mismatches detected at load
   time, so a stale share is a miss, not a corruption.

2. **Python trace cache** (`~/.phasic_cache/traces/`) — same
   keying scheme but **deprecated**. No public path populates it
   anymore; the helpers remain to clean up leftover files. Sharing
   is technically possible but pointless because nothing reads the
   loaded traces.

3. **TraceRegistry / IPFS** — **the designed sharing surface**.
   `phasic.trace_repository.get_trace(trace_id)` fetches a
   pre-computed `EliminationTrace` from IPFS; the registry is a
   GitHub-hosted human-readable index. Currently exists in the
   codebase but its consumer (the trace-driven log-likelihood
   pipeline) is on the deprecation track. If/when an FFI-backed
   trace evaluator lands, this becomes the natural community
   path for "skip the O(n³) elimination, use someone else's
   trace."

## What `cache_trace` did and why it's deprecated

The `Graph(cache_trace=True)` kwarg is now a deprecation no-op
(force-set to `False`, emits a `DeprecationWarning`). What it
*used* to do: cache a Python `EliminationTrace` on the graph
instance and route `expectation()` / `moments()` / `variance()`
through `_ensure_trace` → `instantiate_from_trace` → C++ moments
on the rebuilt graph. That path was strictly worse than calling
the C++ super() method directly, since the Stage A0 cache already
handles the expensive part (the O(n³) elimination is done once,
in C, and survives parameter updates). Removing the trace
detour removed:

- A wasted O(n) trace recording per Python call.
- A known numerical bug on cyclic graphs
  (see `tests/pytest/failing_tests.md`).
- A second code path with its own caching state to maintain.

The trace files still on disk after upgrades are harmless;
`phasic.cache.clear_trace_cache()` cleans them up.

## Notes on `use_cache` parameter on `pmf_from_graph`

`Graph.pmf_from_graph(graph, ..., use_cache=True)` accepts a
`use_cache` keyword for backward compatibility; the value has
**no effect** (the parameter is declared in the signature but
not read inside the body — see `__init__.py:4215`). The original
"symbolic DAG cache" the parameter gated was removed
("symbolic_cache.py has been removed as obsolete" — see
`__init__.py:4298`). Treat the kwarg as a no-op.

## Summary: which caches matter for which workflow

- **Iterative SVGD / MCMC on a parameterised joint-prob graph:**
  A0 (per-process), A1 (in-memory persistent graph for the pybind
  path), A2 (across-process), JAX cache. The first run pays the
  O(n³) elimination + JAX compilation; subsequent runs (same
  process or fresh process, same graph) skip both.
- **Notebook iteration on a single model:** A0 amortises within a
  notebook session. JAX cache amortises across kernel restarts.
- **SLURM array on a shared filesystem:** layered JAX cache
  (`shared_cache_dir`) lets workers share compiled XLA modules.
  A2 amortises the C-side elimination if the shared cache is on
  the same filesystem.
- **Notebook with a slow-to-build callback:** opt into
  `Graph(callback, graph_cache=True)` to skip callback execution
  on subsequent runs; the cache key is AST-stable across sessions.
- **Publishing a model for the community:** TraceRegistry / IPFS
  (when consumer is functional) or just commit a script that
  rebuilds the graph; the A2 cache will populate on first run.
