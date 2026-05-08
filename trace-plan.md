# Plan: Persistent parameterized graph + cached symbolic elimination across runs

## Context

SVGD's per-theta cost is dominated by repeated O(n³) graph elimination, not by anything inherent to the inference. Investigation traced the cost to two layers:

1. **Per-call layer**. `GraphBuilder::build(theta, ...)` (`src/cpp/parameterized/graph_builder.cpp:118`) constructs a fresh `phasic::Graph` per call using the **scalar** `add_edge(to, w)` overload. The fresh graph is therefore *non-parameterized* (`graph->parameterized == false`), and `ptd_precompute_reward_compute_graph` (`src/c/phasic.c:1772`) takes the non-parameterized branch — running full O(n³) Gaussian elimination every time. The fresh `phasic::Graph` is destroyed at the end of `compute_pmf_and_moments`, taking the elimination cache with it.

2. **Per-process layer**. Even if the per-call layer were fixed (each thread keeping a persistent graph), the symbolic elimination (`parameterized_reward_compute_graph`) would still be re-built on every fresh process — every SVGD restart, every SLURM worker. For large models (n ≥ 100, repeated SVGD runs across many param settings, multi-node clusters), that O(n³) startup cost is the actual headline cost.

This plan addresses both layers. Per-call: thread-local persistent graph + `update_weights(theta)`. Per-process: serialize `parameterized_reward_compute_graph` to disk under `~/.phasic_cache/`, keyed by the existing graph content hash.

JAX compatibility is preserved: the change happens *inside* the existing `pure_callback` boundary, so `jit` / `vmap` / `pmap` / sharding / `custom_vjp` finite-difference grad keep working with no changes at the JAX layer.

### Empirical baseline

For a 51-vertex parameterised coalescent over 100 thetas:

| Path | Time |
|---|---|
| Current FFI sequential (fresh graph per theta) | 1113 ms |
| Current FFI vmap'd via OpenMP (fresh per thread) | 299 ms |
| Persistent graph + `update_weights` (sequential, moments only) | 4 ms |

Roughly 250×–400× speedup on moments by avoiding redundant elimination. The PDF/PMF forward computation is intrinsic and dominates total time once the elimination cost is amortised.

## Design

### Layer A — Persistent thread-local `phasic::Graph` inside `GraphBuilder`

`GraphBuilder` is already cached thread-locally at the FFI handler boundary (`src/cpp/parameterized/graph_builder_ffi.cpp:20`). We extend it so that each thread also caches a `phasic::Graph` instance (one per thread per `GraphBuilder`).

**Algorithm per `compute_pmf_and_moments(theta, ...)` call:**
1. If this thread does not have a `phasic::Graph` cached for this `GraphBuilder`: build one using `add_edge_parameterized` (so the graph is *parameterised*; `update_weights` will work; `parameterized_reward_compute_graph` will be built once on first use and reused thereafter), then call `update_weights(theta)`.
2. Else: call `update_weights(theta)` on the cached graph.
3. Call the existing forward functions (`g.expected_waiting_time`, `g.pdf`, `g.dph_pmf`).

The C-level `parameterized_reward_compute_graph` survives `update_weights` (it's only invalidated by `notify_change`, which `update_weights` doesn't trigger). So step 3's first call builds the parameterised compute graph (O(n³)); every subsequent step-3 call just runs the concrete-replay layer (O(commands)).

**Why thread-local rather than shared.** `update_weights` is not thread-safe: it mutates `edge->weight` in place and rebuilds the concrete `reward_compute_graph` (`src/c/phasic.c:3096`, `src/c/phasic.c:3022`). Sharing one `phasic::Graph` across OpenMP threads in the FFI batched loop would race. Thread-local storage means each OpenMP worker has its own graph and own elimination cache, which is consistent with how the FFI handler currently parallelises over batch elements.

**Why this preserves JAX semantics.** The change is *inside* the existing `pure_callback` (no change at the JAX boundary). Each call still produces output entirely determined by `theta`. JIT, vmap (with `vmap_method='expand_dims'`), pmap (one Python interpreter per device), `custom_vjp` finite-difference grad — all unaffected. `pure_callback`'s purity contract is about *output determinism*, not internal statelessness; an internal cache that depends only on the input theta satisfies it.

### Layer B — Disk-persistent `parameterized_reward_compute_graph` cache

After Layer A, the `parameterized_reward_compute_graph` is built once per (graph structure, thread) and reused across all thetas in that thread. Layer B persists it to disk so that:

- Subsequent SVGD runs on the same model don't pay the O(n³) elimination cost again.
- SLURM multi-node workers can share the cache via the network filesystem (caveat: each worker still has thread-local in-process state; the disk cache primes that state on first use per process).

**Cache key**. Use the existing `phasic.hash.compute_graph_hash(graph).hash_hex` (`src/cpp/phasic_pybind.cpp:5042`) as the key. The hash is deterministic SHA-256, includes structure + coefficients (which are theta-independent), and excludes per-call edge weights. So the same parameterised model always produces the same key regardless of theta history. Cache file: `~/.phasic_cache/parameterized_reward_compute/<hash_hex>.bin`.

**On-disk format**. The struct has three serializable parts plus one part needing fix-up:

| Part | Storage | Notes |
|---|---|---|
| `length` (size_t) | direct write | |
| `commands` array | direct write of opcodes (`type`, `from`, `to`, `multiplier`) plus offsets into `mem`/`memr` for `fromT`, `toT`, `multiplierptr` | Pointers replaced by offsets |
| `mem` (linked list) | flatten into a single `double[]` buffer | Linked-list structure preserved by storing per-block sizes; loader rebuilds the `ll_of_a` chain |
| `memr` | direct write of offsets into the flattened `mem` | |
| Edge-weight pointers | record `(vertex_idx_in_storage_order, edge_idx_within_vertex)` per command that needs one | Re-pointed on load against the live graph's edge array |

The pointer-fix-up table is the only complexity. Each `add_command_param_p` and `add_command_param_pp` site (`src/c/phasic.c:5873, 5949, 6372, 6449`) writes a `multiplierptr` pointing at `&vertex->edges[j]->weight`. We record `(i, j)` for each such command at recording time and replay the assignment at load time. The vertex/edge ordering is structural (set by `ptd_graph_create` + edge-insertion order at graph construction), so it's stable across processes that build the same graph from the same JSON serialization.

**Concurrency**. Use atomic write-then-rename (the same pattern as `trace_serialization.py:307-312`). Multiple processes racing to populate the same cache key produce duplicate work but not corruption. Each load is independent and lock-free.

**Invalidation**. The cache key embeds structure + coefficients via the graph hash. Any structural or coefficient change produces a new key — no in-place invalidation needed.

**Optional: cache disable**. Honour `PHASIC_DISABLE_CACHE=1` environment variable (already used by `trace_serialization.py:81`).

### Layer C — Tests + benchmark

Two new test files to gate Layer A and Layer B independently:

- `tests/pytest/inference/test_persistent_graph_cache.py` — Layer A correctness:
  - Persistent graph + `update_weights` produces bit-identical results to fresh-graph-per-call across a battery of (theta, observation) inputs.
  - Bit-identical under sequential, vmap (sequential), and pmap dispatch.
  - Thread-safety: pmap with N devices produces N independent persistent graphs; outputs match a sequential reference.
  - The `parameterized_reward_compute_graph` is built once per thread per model (instrument by counting calls via a logger or a temporary debug counter).
- `tests/pytest/inference/test_disk_param_cache.py` — Layer B correctness:
  - Build graph in process A, populate cache. Build same graph in process B (subprocess), verify cache hit, verify replay matches.
  - Cache key independent of theta history.
  - Mutating coefficients changes the cache key (recorded under a different file).
  - `PHASIC_DISABLE_CACHE=1` round-trips correctly.

Plus a benchmark `tests/pytest/inference/bench_persistent_graph.py` (not part of the regular pytest run) measuring per-theta cost on the 51-vertex coalescent and on a larger 200-vertex synthetic model, comparing fresh-graph vs persistent vs persistent+disk-cache.

## Stages

Stages 0 and 1-prep (`OpType.SUB` + PHASE 1 dedup) are already landed in commits `e583286` and `7c41edd`. The Stage 0 verification harness (`test_trace_vs_direct.py`, `test_trace_jax_compat.py`) remains green.

### Stage A1 — `GraphBuilder` thread-local persistent `phasic::Graph`

Touch points:

- **`src/cpp/parameterized/graph_builder.hpp`**: add `mutable thread_local std::unique_ptr<phasic::Graph> persistent_graph_` private member.
- **`src/cpp/parameterized/graph_builder.cpp`**:
  - In `build(theta, theta_len)`, switch from `from_v->add_edge(to_v, evaluated_weight)` (lines 171, 177, 184, 190) to `from_v->add_edge_parameterized(to_v, evaluated_weight, edge.coefficients)` for parameterized edges and the existing scalar overload for purely constant edges. The result is a graph with `edge_mode == PARAMETERIZED` so `update_weights` works.
  - New helper `Graph& get_or_init_persistent_graph(const double* theta, size_t theta_len)`:
    - If `persistent_graph_` is null, build a fresh graph via `build(theta, theta_len)` and store it in `persistent_graph_`.
    - Otherwise call `persistent_graph_->update_weights_parameterized(...)` with the new theta.
    - Return `*persistent_graph_`.
  - Refactor `compute_moments`, `compute_pmf`, `compute_pmf_and_moments` to call `get_or_init_persistent_graph(theta)` instead of `build(theta)`. The downstream `compute_moments_impl` / `g.pdf` / etc. work unchanged on the persistent graph.

- **`src/cpp/parameterized/graph_builder_ffi.cpp`**: no change needed for Stage A1. The OpenMP parallel-for already gives each thread its own `thread_local` `persistent_graph_` slot since the keyword is on the GraphBuilder member. (Each iteration of `#pragma omp parallel for` may run on a different thread, so iterations could see different `persistent_graph_` instances — this is correct behaviour because each thread has its own elimination cache.)

**Acceptance**: Stage 0 harness still green; new `test_persistent_graph_cache.py` passes; benchmark shows ≥10× speedup for sequential SVGD on n ≥ 50.

### Stage A2 — Disk-persistent `parameterized_reward_compute_graph` cache

Touch points:

- **`src/c/phasic.c`**: new functions
  - `int ptd_save_parameterized_reward_compute_graph(const char* path, const struct ptd_desc_reward_compute_parameterized* compute, const struct ptd_graph* graph)`:
    - Walks the commands building a pointer-fix-up table (for each `multiplierptr` and `toT`/`fromT` that points into edge weights, record the `(vertex_idx, edge_idx)` it points at).
    - Flattens `mem` (the `ll_of_a` linked list) to a single buffer, recording the linked-list block boundaries.
    - Writes header + commands array (with pointer offsets instead of pointers) + flat `mem` + `memr` offsets + the fix-up table to the file.
    - Atomic write-then-rename.
  - `struct ptd_desc_reward_compute_parameterized* ptd_load_parameterized_reward_compute_graph(const char* path, const struct ptd_graph* graph)`:
    - Reads file. Allocates `mem` as a single block (or rebuilds the linked-list chain). Allocates `memr`. Allocates `commands`.
    - Translates offsets back to pointers using the loaded `mem` base address and per-vertex/per-edge weight pointers from the graph.
    - Returns ready-to-use struct.

- **`src/c/phasic.c`**: extend `ptd_precompute_reward_compute_graph` (line 1772):
  - Before building from scratch (line 1796), check the disk cache. Compute the graph hash via `ptd_graph_content_hash`, look for `~/.phasic_cache/parameterized_reward_compute/<hash>.bin`, attempt to load.
  - On cache miss, build fresh as today, then save to disk.
  - Both behaviours gated by the existing `PHASIC_DISABLE_CACHE=1` env var.

- **`src/cpp/phasic_pybind.cpp`**: optional Python bindings for cache inspection (clear, list, size). Keep minimal — the cache should be transparent.

**Acceptance**: `test_disk_param_cache.py` passes (subprocess load matches in-process); benchmark shows on second SVGD run from a fresh process the elimination cost is amortised away (load time ≪ elimination time).

### Stage A3 — Wire into `Graph.svgd` (optional)

The Stage A1+A2 changes are transparent to all existing callers — no API change needed. Optionally add documentation noting that SVGD now caches the elimination structure across runs and across processes via `~/.phasic_cache/`, with the `PHASIC_DISABLE_CACHE` env var as the override.

If we want explicit user-facing control, we can add `Graph.svgd(..., cache_param_compute=True/False)` (default True) that passes through to a new C-level toggle. But this is optional polish; the cache being transparent and on-by-default is the simpler default.

## Critical files

- **`src/cpp/parameterized/graph_builder.hpp`** (Stage A1) — add persistent_graph_ member.
- **`src/cpp/parameterized/graph_builder.cpp`** (Stage A1) — switch build() to use `add_edge_parameterized`; add `get_or_init_persistent_graph` helper; refactor `compute_*` entry points.
- **`src/c/phasic.c`** (Stage A2) — new save/load functions for `parameterized_reward_compute_graph`; cache lookup in `ptd_precompute_reward_compute_graph`.
- **`tests/pytest/inference/test_persistent_graph_cache.py`** (Stage A1 verification) — new file.
- **`tests/pytest/inference/test_disk_param_cache.py`** (Stage A2 verification) — new file.
- **`tests/pytest/inference/bench_persistent_graph.py`** (optional benchmark) — not part of pytest default.

## Existing utilities to reuse

- `phasic.hash.compute_graph_hash(graph)` (`src/cpp/phasic_pybind.cpp:5042`) — graph content hashing, deterministic SHA-256, key for disk cache.
- `~/.phasic_cache/` directory convention (`src/phasic/trace_serialization.py:293`) — same root for the new sub-directory `parameterized_reward_compute/`.
- Atomic write-then-rename (`src/phasic/trace_serialization.py:307-312`) — same pattern for cache writes.
- `PHASIC_DISABLE_CACHE` env var (`src/phasic/trace_serialization.py:81`) — same opt-out.
- `Graph::update_weights_parameterized` (`api/cpp/phasiccpp.h:268`) — wraps `ptd_graph_update_weights`; preserves `parameterized_reward_compute_graph` cache.
- `add_edge_parameterized` (`src/cpp/phasiccpp.cpp:283`) — needed for the build() change in Stage A1.
- The existing thread_local pattern in `graph_builder_ffi.cpp:20` — template for the new thread_local persistent graph.

## Verification

End-to-end test command after each stage:

```
pixi run -- pytest \
  tests/pytest/inference/test_trace_vs_direct.py \
  tests/pytest/inference/test_trace_jax_compat.py \
  tests/pytest/inference/test_persistent_graph_cache.py \
  tests/pytest/inference/test_disk_param_cache.py \
  tests/pytest/inference/test_self_loop_correction.py \
  tests/pytest/inference/test_hierarchical_graph.py \
  tests/pytest/inference/test_trace_repository.py \
  tests/pytest/inference/test_trace_rewards.py \
  -v
```

Multi-core / SLURM-only commands:

```
PHASIC_NUM_CPU_DEVICES=4 pixi run -- pytest \
  tests/pytest/inference/test_trace_jax_compat.py -k "pmap or sharding" -v

sbatch scripts/test_slurm_trace.sh
```

Acceptance criteria:

- All Stage 0 tests still pass (no regression).
- New persistent-cache tests pass (correctness + thread-local isolation under pmap).
- New disk-cache tests pass (cross-process round-trip, hash stability, env-var opt-out).
- Benchmark on 51-vertex coalescent shows ≥10× sequential speedup.
- Benchmark on second-run-from-fresh-process shows amortised elimination cost (load time ≪ build time).
- `Graph.svgd` on the two-island notebook produces equivalent posterior to before.

## Out of scope

- Python `EliminationTrace` self-loop correction. The Python trace path is orthogonal to this plan; it's used only by `cache_trace=True` mode for direct `Graph.expectation()`/etc., not by SVGD. The xfailed tests in `test_self_loop_correction.py` and `test_trace_vs_direct.py` (cyclic) stay xfailed.
- True symbolic `jax.grad` (per-op tracing through the elimination). The current `custom_vjp` finite-difference path is preserved unchanged. If exact symbolic gradients become a requirement later, that's a separate project on the Python op-DAG side.
- The follow-on plans `update_ipv-plan.md` and `daisy-chain-plan.md` are unaffected; their assumptions (trace replay produces a `Graph`, `pdf`/`stop_probability` work) are now satisfied via the existing direct-C++ path on the cached parameterized graph.
