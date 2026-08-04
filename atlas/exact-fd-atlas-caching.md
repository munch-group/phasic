# B3 caching atlas: every cache that touches the parameterized elimination tape

**Purpose.** A reproduced segfault was caused by `ptd_load_pcg_rev3_mmap` leaving `input_specs` `NULL` in a way `ptd_sojourn_grad_theta_subset` didn't expect (fixed, guarded at `src/c/phasic.c:11432`). This document maps *every* cache layer that touches the parameterized elimination tape or its FFI wrapper objects, so the same "cache A populates differently than cache B, new code only tested against one of them" class of bug can be checked systematically. All claims are cited `file:line`, read directly from source in this session (2026-08-04).

**Headline finding.** Beyond the already-fixed `input_specs` bug, the on-disk caching subsystem has a *second, older, still-live* format (rev-1/rev-2, `PTDPRMC1`) that backs the public GitHub trace registry (`Graph.pull_cache()`/`push_cache()`) and was **never given the bounds-hardening that rev-3 got** — it has an unchecked arbitrary-offset pointer decode (`src/c/phasic.c:3244-3246`) and an unchecked command-type dispatch that aborts the whole process on garbage input (`src/c/phasic.c:9720`), even though the file's own comments say hardening was added "because a cache file can be... pulled from the community registry" (`src/c/phasic.c:3621-3628`) — that comment is attached to the *wrong* format. Separately, the registry's "parent" artifact is saved in a format (`PTDPRMC1`) the automatic loader can never read (`PTDPRMC3`-only), so `pull_cache()`'s documented promise silently doesn't hold for parent artifacts. And the per-SCC on-disk cache (`src/c/scc_synthetic.c`) shares the exact `input_specs`-NULL-on-mmap shape with zero guard today, dormant until the first SCC-level exact-gradient C function is written.

---

## 0. The five cache layers found

1. **Graph-level in-memory tape cache** — `graph->reward_compute_graph` / `graph->parameterized_reward_compute_graph` / `graph->parameterized_reward_compute_graph_off`, populated by `ptd_precompute_reward_compute_graph` (`src/c/phasic.c:1930`).
2. **Stage-A2 on-disk cache, rev-3 zero-copy format** — `ptd_save_pcg_rev3` / `ptd_load_pcg_rev3` (`src/c/phasic.c:3549`, `3844`), gated by `reward_compute_cache` / `PHASIC_REWARD_COMPUTE_CACHE` (default OFF).
3. **Per-SCC on-disk cache** — `src/c/scc_synthetic.c`, same rev-3 format and directory, `scc_` filename prefix, same env gate.
4. **Legacy rev-1/rev-2 on-disk format + the GitHub trace registry** — `ptd_save/load_parameterized_reward_compute_graph[_ex]` (`src/c/phasic.c:4048`, `4300`), exposed to Python only as "test-only" pybind methods, but actually used in production by `src/phasic/compute_repository.py` (`Graph.pull_cache()`/`push_cache()`).
5. **FFI/C++ per-thread caches** — `builder_cache`, `per_thread_graph_cache`, `daisy_chain_meta_cache` in `src/cpp/parameterized/graph_builder_ffi.cpp`.

---

## 1. Graph-level in-memory tape cache

### 1.1 Fields (`api/c/phasic.h:173-260`)

```
struct ptd_graph {
    ...
    struct ptd_desc_reward_compute *reward_compute_graph;                        // :182  numeric, theta-baked
    struct ptd_desc_reward_compute_parameterized *parameterized_reward_compute_graph;      // :183  symbolic, RAW pointer form
    struct ptd_desc_reward_compute_parameterized_off *parameterized_reward_compute_graph_off; // :187  symbolic, OFFSET/index form
    bool was_dph;                                                                // :196
    bool dph_compute_invalidated;                                                // :203
    ...
};
```

`parameterized_reward_compute_graph_off` is documented at `:184-186` as "the zero-copy offset form (dual-form cache load path)... When set (cache HIT), the offset executor runs against it and the raw `parameterized_reward_compute_graph` stays NULL."

### 1.2 Populator: `ptd_precompute_reward_compute_graph` (`src/c/phasic.c:1930-2155`)

- Takes `graph->compute_graph_lock` unconditionally (`:1944-1946`).
- **`dph_compute_invalidated` wipe** (`:1948-1975`): if set, frees and NULLs *all three* of `reward_compute_graph`, `parameterized_reward_compute_graph`, `parameterized_reward_compute_graph_off`, then clears the flag. One-shot; only runs when the flag is true.
- **Rebuild-vs-hit condition** (`:1977-2051`): only enters the parameterized-build branch when `graph->reward_compute_graph == NULL` (i.e. after a wipe, or on first use). Inside that, only builds/loads a symbolic tape when **both** `parameterized_reward_compute_graph == NULL && parameterized_reward_compute_graph_off == NULL` (`:1979-1980`) — so an already-populated tape (either form) is reused untouched.
  - If the disk cache is enabled (`!ptd_pcg_cache_disabled()`, `:1992`) it tries `ptd_load_pcg_rev3` first (`:1996-1997`). **A hit populates `parameterized_reward_compute_graph_off` and *only* that field** (`:2003`) — `parameterized_reward_compute_graph` (raw) is never touched.
  - On a miss, it builds fresh via `ptd_graph_ex_absorbation_time_comp_graph_parameterized[_dyn]` and **populates `parameterized_reward_compute_graph` (raw) and *only* that field** (`:2016-2021`), then best-effort saves via `ptd_save_pcg_rev3` (`:2024-2049`).
  - **These two fields are therefore mutually exclusive by construction** — never both non-NULL at once outside the transient self-check local variable at `:2063-2133`. Confirmed by grepping every assignment site to `parameterized_reward_compute_graph_off =` (`:1973`, `:2003`, `:2978`, `:4978` — only ever NULL-init or the rev-3 load) — it is **never** set from a fresh in-memory conversion.
- **Dual-form executor fork** (`:2075-2085`): if `_off` is set, run the offset executor (`ptd_graph_build_ex_absorbation_time_comp_graph_parameterized_off`); else run the raw executor. Result always lands in `graph->reward_compute_graph` (the numeric form used by ordinary PDF/moments evaluation).
- `PHASIC_PCG_SELFCHECK` env (`:2065-2133`) builds a *local* `ptd_pcg_convert_to_offset` copy of the raw PRC and diff-checks it against the offset executor's output — this is a debug/CI-only path, distinct from the two graph-level fields.

### 1.3 What `dph_compute_invalidated` is / who sets it

- Set by `Graph.set_was_dph(True)` on the **false→true transition only**, in the pybind layer: `if (value && !g.c_graph()->was_dph) { g.c_graph()->dph_compute_invalidated = true; } g.c_graph()->was_dph = value;` (`src/cpp/phasic_pybind.cpp:1368-1387`).
- Mirrored in the C++ parameterized `GraphBuilder::build()` path: `if (was_dph_ && !g.c_graph()->was_dph) { g.c_graph()->dph_compute_invalidated = true; } g.c_graph()->was_dph = was_dph_;` (`src/cpp/parameterized/graph_builder.cpp:228-231`) — a no-op in practice there since `build()` always starts from a fresh `ptd_graph_create()`.
- **Consumed exactly once**, lazily, inside `ptd_precompute_reward_compute_graph` (`src/c/phasic.c:1948-1975`) — *not* at flip time. This means there is a real window between `set_was_dph(True)` and the next call into `ptd_precompute_reward_compute_graph` during which `graph->parameterized_reward_compute_graph_off`/`parameterized_reward_compute_graph` still hold the **pre-was_dph** tape while `graph->was_dph` already reads `true`. See §4 for who is safe from this and why.

### 1.4 THE TABLE — `ptd_desc_reward_compute_parameterized_off` field-by-field across its 3 construction paths

Struct definition: `src/c/phasic.c:1906-1917`.

| Field | `ptd_pcg_convert_to_offset` (in-memory, fresh) `:3425-3527` | `ptd_load_pcg_rev3_copy` (disk, read+copy fallback) `:3683-3763` | `ptd_load_pcg_rev3_mmap` (disk, zero-copy) `:3771-3838` |
|---|---|---|---|
| `length` | set from `raw->length` (`:3433`) | set from file header `h.n_commands` (`:3696`) | set from mmap'd header `h->n_commands` (`:3811`) |
| `commands` | **heap**, `calloc`'d, owned (`:3438-3440`) | **heap**, `malloc`+`fread`, owned (`:3707-3711`) | **points INTO the mmap region** — not owned, not `free()`-able (`:3812`) |
| `mem_base` | **heap**, via `ptd_pcg_flatten_mem` (`:3434-3435`) | **heap**, `malloc`+`fread`, owned (`:3717-3721`) | **points INTO the mmap region**, writable via `MAP_PRIVATE` COW (`:3814`) |
| `mem_doubles` | set (`:3435`) | set from header (`:3697`) | set from header (`:3813`) |
| `mem_is_mmap` | `0` (calloc default; never set) | `0` (calloc default; never set) | **`1`, explicitly set** (`:3808`) |
| `inputs` | `malloc`'d array of **live** `&edge->weight[+byte]` / external-coeff pointers (`:3512-3524`) | `malloc`'d array, bound identically to live `&edge->weight[+byte]` (`:3730, 3743-3744`) | `malloc`'d array, bound identically to live `&edge->weight[+byte]` (`:3818-3829`) — **same semantics as the other two paths** |
| `n_inputs` | `n_spec` (dedup count) (`:3511`) | `h.n_inputs` (`:3698`) | `h->n_inputs` (`:3815`) |
| **`input_specs`** | **POPULATED** — heap array, owned, `spec` (`:3525`) | **POPULATED** — heap array, `malloc`'d and filled field-by-field from the file, semantically identical to the convert path (`:3728-3738`) | **`NULL`, unconditionally** — comment: `"a loaded descriptor is never re-saved"` (`:3816`) |
| `mmap_base` | `NULL` (calloc default) | `NULL` (never set) | the mmap'd file base, for `munmap` (`:3809`) |
| `mmap_len` | `0` (calloc default) | `0` (never set) | the mapping length (`:3810`) |

**This is the exact bug template.** `input_specs` is `Y` on two of three construction paths and `N` on the third, with **no field in the struct itself that flags "was this built via the mmap path"** other than `mem_is_mmap` (which callers would have to know to cross-check) — the field itself is silently absent. The fix, at the one call site that reads it off the graph-level cache, is a manual guard: `if (off->n_inputs > 0 && off->input_specs == NULL) return -1;` (`:11432`, inside `ptd_sojourn_grad_theta_subset`, `:11408-11492`).

Crucially: **`graph->parameterized_reward_compute_graph_off` is *only ever* populated via `ptd_load_pcg_rev3`** (§1.2) — meaning any code that reads this graph-level field is, by construction, always reading a disk-cache-loaded object, and *which* of the two disk-load sub-paths (mmap vs copy) it got is itself non-deterministic at the call site: it depends on the `PHASIC_PCG_DISABLE_MMAP` env var, the platform (`#ifndef _WIN32`, `:3765`, `3846`), and whether the mmap attempt happened to fail for an unrelated reason (`ptd_load_pcg_rev3` dispatcher, `:3844-3855`, silently falls back to copy on *any* mmap failure). So even "reads the graph-level off-cache" does not deterministically mean "input_specs is NULL" or "is populated" — it can flip per-process, per-platform, per-env-var.

### 1.5 Destroy correctly branches on ownership (`ptd_pcg_desc_off_destroy`, `:9608-9622`)

```c
if (off->mem_is_mmap) {
    if (off->mmap_base) munmap(off->mmap_base, off->mmap_len);
} else {
    if (off->commands) free(off->commands);
    if (off->mem_base) free(off->mem_base);
    if (off->input_specs) free(off->input_specs);
}
if (off->inputs) free(off->inputs);   /* inputs[] is always heap (bound at load) */
free(off);
```
This is the one place in the codebase that already treats `mem_is_mmap` as authoritative for ownership. Any *new* code that manually copies `off->commands`/`off->mem_base` out of the struct (rather than going through this destroy function) would silently do the wrong thing (leak on the mmap path, or `free()` a non-heap pointer) depending on which construction path built the struct it's holding — flagged as landmine L6 below.

---

## 2. Stage-A2 on-disk cache (rev-3 zero-copy format)

- **Gate**: `reward_compute_cache` (Python config) / `PHASIC_REWARD_COMPUTE_CACHE` env var, default **OFF**. Checked by `ptd_pcg_cache_disabled()` (`src/c/phasic.c:3862-3866`): unset or anything other than exactly `"1"` = disabled. Wired from `phasic.configure(reward_compute_cache=True)` in `src/phasic/config.py:194, 317-318, 456`.
- **Path**: `ptd_pcg_build_cache_path` (`:3906-3951`) → `<cache_root>/parameterized_reward_compute/<content_hash_hex>.bin`, where `<cache_root>` is `$PHASIC_CACHE_DIR` or `$HOME/.phasic_cache` (`ptd_cache_root_dir`, `:3878-3904`). Directories are `mkdir -p`'d on demand.
- **Cache key = `ptd_graph_content_hash(graph)`** (`src/c/phasic_hash.c:216-272`). What's hashed, precisely:
  - Graph metadata: `state_length`, `param_length`, `vertices_length`, and one flags byte with bit 0 = `parameterized`, bit 1 = **`was_dph`** (`:236-243`).
  - Per vertex (canonical index order): vertex `state` array, then each out-edge (sorted by target-vertex-index then coefficient length then coefficient values for canonical ordering, `compare_edges` `:147-166`) hashed as `(target_index, coefficients[])` (`hash_vertex_structure`, `:169-214`, `:190-198`).
  - **NOT hashed**: `edge->weight` (the current theta-evaluated numeric value — correct, since the symbolic tape is theta-independent, verified by the "Stage A0 invariant" comment at `api/c/phasic.h:382-384`), `weight_mode`/formula tape ops (correct too, for the same reason — the elimination structure only references `&edge->weight` by pointer, not by how that slot gets filled), and **`graph->use_dyn_ordering`** (not obviously correct/incorrect — see landmine L8).
- **Save**: `ptd_save_pcg_rev3` (`:3549-3605`) — converts raw PRC to offset form via `ptd_pcg_convert_to_offset`, writes `[header | commands_off[] | mem doubles | input-specs[]]` with atomic write-then-rename (`tmp.<pid>` → `rename()`, `:3559-3603`).
- **Load dispatcher**: `ptd_load_pcg_rev3` (`:3844-3855`) tries `ptd_load_pcg_rev3_mmap` first (unless `PHASIC_PCG_DISABLE_MMAP` is set or on Windows), falls back to `ptd_load_pcg_rev3_copy` on any mmap failure.
- **Field-drop on load**: see the table in §1.4 — `input_specs` is the one field silently dropped (set `NULL`) specifically on the mmap sub-path, not on the copy sub-path. No *other* field is silently dropped by either loader relative to the fresh-build path (both loaders populate `length`/`commands`/`mem_base`/`mem_doubles`/`inputs`/`n_inputs` identically in content, only `input_specs` and the storage class of `commands`/`mem_base` differ).
- **Validation against untrusted/corrupt files**: `ptd_pcg3_off_validate` (`:3630-3678`) runs on *both* rev-3 load sub-paths (`:3752` for copy, `:3833` for mmap) and rejects (treats as cache miss, not a crash) any command with an out-of-range `type` (would otherwise `DIE_ERROR`-abort the raw-command switch, `:3641-3644`), any `MEM` operand outside `[0, mem_doubles)` (`:3657-3663`), or any `INPUT` operand `>= n_inputs` (`:3664-3669`). Overflow-safe size arithmetic (`ptd_size_mul_ok`/`ptd_size_add_ok`, `:3610-3619`) guards the mmap section-offset math (`:3792-3804`) and the copy-path mallocs (`:3704, 3714, 3724`) against attacker-chosen header counts. **This hardening explicitly anticipates hostile input** — the comment at `:3621-3628` says the cache file "can be corrupt, from another user on a shared filesystem, or pulled from the community registry." (See §2c for why that comment is misattributed to the wrong file format in practice.)

---

## 2b. Per-SCC on-disk cache (`src/c/scc_synthetic.c`)

- Shares the **exact same rev-3 format and top-level directory** as §2, distinguished only by an `scc_` filename prefix: `ptd_scc_build_cache_path` (`scc_synthetic.c:90-136`) → `<cache_root>/parameterized_reward_compute/scc_<hash_hex>.bin`, comment at `:131-132`.
- Same gate: `ptd_scc_cache_disabled()` (`:69-73`) reads the identical `PHASIC_REWARD_COMPUTE_CACHE` env var. Additionally gated by a size floor, `PHASIC_MIN_SCC_SIZE_TO_CACHE` (default 4 vertices, `:75-88`).
- Cache key: `ptd_graph_content_hash(synth)` on the per-SCC **synthetic** graph (`:125`) — same hash function as §2, so it also encodes `was_dph` for the synthetic graph.
- `ptd_synth_get_or_compute_prc` (`:1133-1202`) and `ptd_scc_get_or_compute_prc` (`:1204-1239`) call **the identical `ptd_load_pcg_rev3` / `ptd_save_pcg_rev3`** entry points as the top-level graph (`:1173-1174`, `1198`), and on a hit install the loaded descriptor onto `synth->parameterized_reward_compute_graph_off` (`:1178`) — exactly the same field, on a different `ptd_graph*`.
- **This does NOT route through `ptd_precompute_reward_compute_graph`** at all — it manipulates `synth->parameterized_reward_compute_graph_off` directly, so `dph_compute_invalidated`/the was_dph latch (§1.3) has **no effect here**; the synth graph is always freshly built per call by `ptd_scc_build_synthetic_graph`, so this is currently safe, but it means the SCC path is architecturally decoupled from the top-level graph's invalidation machinery.
- **No consumer of `synth->parameterized_reward_compute_graph_off->input_specs` exists today** (confirmed: no `input_specs` reference anywhere in `scc_synthetic.c` or any other file besides `phasic.c`). This is a **dormant** instance of the exact same landmine as §1.4/L1 — see L5 below.

---

## 2c. Legacy rev-1/rev-2 on-disk format + the GitHub trace registry

This is a **separate, older on-disk format** for the same conceptual artifact (a serialized `ptd_desc_reward_compute_parameterized`), predating the rev-3 zero-copy format, and it is **not dead code** — it backs a real, network-facing production feature.

- **Format**: magic `PTD_PCG_MAGIC = "PTDPRMC1"` (`src/c/phasic.c:3053`), `PTD_PCG_VERSION = 1` (`:3054`), `PTD_PCG_FORMAT_REVISION` up to `2` (`:3061`; rev-2 adds `EXTERNAL`-pointer anchors for SCC composition). Produces the **raw** (`ptd_desc_reward_compute_parameterized`, `mem`+`memr` linked-list form), not the offset/`_off` form — it has no `input_specs` concept at all.
- **Save**: `ptd_save_parameterized_reward_compute_graph[_ex]` (`:4048` / rev-2 variant with `external_anchors`); **load**: `ptd_load_parameterized_reward_compute_graph_impl` (`:4300-4586`), wrapped by the v1/v2 entry points at `:4595-4617`.
- **Python exposure**: pybind methods `_has_param_compute_graph_cache`, `_save_param_compute_graph`, `_load_param_compute_graph`, `_save_synthetic_param_compute_graph_ex`, `_load_synthetic_param_compute_graph_ex` (`src/cpp/phasic_pybind.cpp:1092-1350`), all doc-labeled **"Test-only... not part of the public API."**
- **But it is production code**: `src/phasic/compute_repository.py` — which backs `Graph.pull_cache()`/`Graph.push_cache()` (assigned via `src/phasic/_graph_cache_transfer.py:10-155`) — calls `graph._save_param_compute_graph(str(parent))` directly for the registry's "parent" artifact (`compute_repository.py:272`), and the module's own docstring describes this as the artifact format (`compute_repository.py:1-33`, `58-76`). Only `grep`-confirmed callers of `_load_param_compute_graph` in the whole tree are two test files (`tests/pytest/test_scc_prc_external.py`, `tests/pytest/test_disk_param_cache_save_load.py`) — **no production Python code calls the loader** (see finding below).
- `compute_repository.py` explicitly supports **both** magics: `_PCG_HEADER_MAGIC = b"PTDPRMC1"` and `_PCG3_HEADER_MAGIC = b"PTDPRMC3"` (`:72, 76`), and `_peek_format_revision` (`:84-116`) distinguishes them by magic alone. Per-SCC artifacts pushed to the registry (`_fetch_scc`, `compute_repository.py:555-588`) are copied **verbatim** from the local `scc_*.bin` cache files (`_save_artifacts`, `:257-288`) — which, per §2b, are genuinely rev-3. **Only the "parent" artifact is forced through the rev-1 saver** (`:272`); there is no code path in `compute_repository.py` that ever writes a rev-3 *parent* file.

### Findings specific to this cache layer

**(a) No bounds check on `MEM`-kind pointer decode — arbitrary out-of-bounds write from an untrusted file.** `ptd_pcg_decode_ptr_impl` (`:3234-3269`):
```c
if (enc->kind == PTD_PCG_PTR_MEM) {
    return mem_base + enc->doubles_offset;      // :3244-3246 — NO bounds check
}
if (enc->kind == PTD_PCG_PTR_EDGE) {
    if (enc->vertex_idx >= graph->vertices_length) return NULL;   // :3248 — bounds-checked
    ...
    if (enc->edge_idx >= v->edges_length) return NULL;            // :3252 — bounds-checked
    ...
}
```
`enc->doubles_offset` is an `int64_t` read straight from the file and added unbounded to `mem_base`. This pointer subsequently gets **written through** at replay time (e.g. `case P: *(command.fromT) = *(command.fromT) + *command.toT * command.multiplier;`, `:9660-9663`, or the offset-form equivalent). This is the same class of vulnerability the rev-3 loader was explicitly hardened against — compare `ptd_pcg3_off_validate`'s `MEM` check (`:3657-3663`, "closes the arbitrary-write... the executor would otherwise perform," comment at `:3646-3648`) — but that hardening was **never applied to the rev-1/2 decoder**, which is the one format actually reachable from an internet-downloaded file today.

**(b) No validation of the command `type` byte before dispatch — unconditional process abort on any corrupt/malicious file.** `ptd_load_parameterized_reward_compute_graph_impl` copies `commands[i].type = encoded_cmds[i].type` (`:4460`) with no range check. The consuming executor, `ptd_graph_build_ex_absorbation_time_comp_graph_parameterized` (used for **both** freshly-built raw PRCs and rev-1/2-loaded ones — it is the single raw-form executor), has `default: DIE_ERROR(1, "Unknown command\n");` (`:9720`). Rev-3's `ptd_pcg3_off_validate` explicitly rejects out-of-range types for exactly this reason (comment: "An unknown type would `DIE_ERROR`-abort the whole process," `:3638-3644`) — again, that specific defense exists only for rev-3.

**(c) No overflow-safe size arithmetic on the rev-1/2 load path**, unlike rev-3's `ptd_size_mul_ok`/`ptd_size_add_ok`. `malloc(header.commands_length * sizeof(*encoded_cmds))` (`:4362-4363`) and `malloc(header.mem_total_doubles * sizeof(double))` (`:4383`) multiply attacker-controlled 64-bit counts directly.

**(d) The registry's "parent" artifact can never be auto-consumed — the documented promise doesn't hold.** `Graph.pull_cache()` downloads the parent `.bin` to `~/.phasic_cache/parameterized_reward_compute/<hash>.bin` (`_graph_cache_transfer.py:10-50`, docstring at `:14-17`: *"so the next call to expectation()/pdf()/moments() reuses the published elimination"*). But that parent file is **always** rev-1 (`PTDPRMC1`), and the **only** automatic consumer of that exact path — `ptd_precompute_reward_compute_graph` → `ptd_load_pcg_rev3` (§1.2/§2) — checks for magic `PTDPRMC3` (`ptd_load_pcg_rev3_copy`, `:3688-3691`; `ptd_load_pcg_rev3_mmap`, `:3785`) and silently treats a `PTDPRMC1` file as a miss. There is no production Python code calling `_load_param_compute_graph` (confirmed by grep, above) to bridge the gap. **Net effect: pulling a registry parent artifact writes an inert file that the automatic C-side cache will always ignore**; only the per-SCC artifacts (genuinely rev-3, §2b) get auto-consumed, and only for graphs that go through SCC/hierarchical elimination. This is a functional instance of the exact "cache A populates differently than cache B, and [the consumer] only tested against one of them" pattern — just manifesting as silent non-use rather than a crash.

---

## 3. FFI/C++ per-thread caches (`src/cpp/parameterized/graph_builder_ffi.cpp`)

| Cache | Scope | Key | Declared | What's cached |
|---|---|---|---|---|
| `builder_cache` | `thread_local` | full structure-JSON string | `:93` | `shared_ptr<GraphBuilder>` — parsed topology/weight-mode/coefficients |
| `daisy_chain_meta_cache` | process-wide, `std::mutex`-protected | structure-JSON string | `:118-127` | `DaisyChainMeta` — parsed `_daisy_chain` block + topology-only derived tables (`aux_set`, `t_to_aux`, `collapsed_pos`) |
| `per_thread_graph_cache` | `thread_local` | `const GraphBuilder*` (raw pointer identity) | `:129-134` | one persistent `phasic::Graph` per builder, built once, then mutated in place |

- **Invalidation**: `grep`-confirmed **zero** `.erase(`/`.clear()` calls anywhere in this file. All three caches are pure-accumulate for the life of the process (or, for the two `thread_local` ones, the life of the OMP worker thread — which `ensure_omp_full_width_once()`, `:140-154`, keeps pinned at full width via `omp_set_dynamic(0)`, so these threads are effectively process-lifetime too). All three caches are never invalidated within a process lifetime. This is safe today only because the key is the full structure content (a changed topology/coefficients/weight_mode/was_dph always produces a different JSON string and therefore a different `builder_cache` entry, and via that a different `per_thread_graph_cache` slot) — old entries just accumulate unused, not incorrectly reused. It would become a correctness risk only if the same JSON string could later represent a semantically different structure, which is not possible given the current serialization scheme.
- **`update_weights` interaction**: `per_thread_graph_cache`'s `phasic::Graph` is built once with dummy theta (`std::vector<double> dummy_theta(theta_len, 1.0)`, e.g. `:1073-1076`, `:1743-1751`) and every subsequent batch element calls `ptd_graph_update_weights(g.c_graph(), theta_b, theta_len, ...)` directly on the **same** live `ptd_graph*` (`:1081-1086`, `:1151-1156`). Per §1's design (comment at `phasic.c:5821-5845`), `update_weights` does **not** rebuild `parameterized_reward_compute_graph`/`_off` — it relies on those tapes holding live pointers into `edge->weight`, which `update_weights` mutates in place. So the FFI per-thread Graph gets the graph-level tape cache **for free**, with zero extra plumbing — this is the intended fast path (comment at `:1060-1068`: "the O(commands) replay path... runs without an O(n^3) symbolic rebuild on every batch element").
- **`was_dph` interaction**: baked into the structure JSON by `GraphBuilder::parse_structure` (`graph_builder.cpp:76-88`) — absent-key default is `is_discrete_` (`:87`), matching the Python-side serializer. Applied to the freshly-built `ptd_graph` in `GraphBuilder::build()` (`:228-231`). Since `was_dph` is part of the cache key (the JSON string), a graph that changes `was_dph` necessarily gets a **new** `builder_cache`/`per_thread_graph_cache` entry — there is no in-place was_dph flip anywhere in this file's cache objects, so no staleness gap analogous to §1.3/§4 exists at the FFI-cache level itself (the gap, if any, is entirely inside the one fresh `ptd_graph` each new entry builds, which is covered by §1/§4).

---

## 4. `was_dph` / `set_was_dph` — which caches it invalidates, and the gap

| Cache | Invalidated by `set_was_dph(True)`? | Mechanism | Timing |
|---|---|---|---|
| `graph->reward_compute_graph` (numeric) | Yes | `dph_compute_invalidated` wipe | **Lazy** — only at the next `ptd_precompute_reward_compute_graph` call |
| `graph->parameterized_reward_compute_graph` (raw) | Yes | same wipe | same lazy timing |
| `graph->parameterized_reward_compute_graph_off` (graph-level, disk-loaded) | Yes | same wipe | same lazy timing |
| Stage-A2 rev-3 file on disk (§2) | Not directly invalidated — but `was_dph` is one of the hashed bits in `ptd_graph_content_hash` (`phasic_hash.c:242`), so a was_dph-flipped graph reads/writes a **different file** than before the flip | keyed out, not wiped | immediate (different key) |
| Per-SCC rev-3 file (§2b) | Same as above (same hash function) | keyed out | immediate |
| `builder_cache`/`per_thread_graph_cache`/`daisy_chain_meta_cache` (§3) | Same as above — `was_dph` is baked into the JSON key | keyed out | immediate |

**The gap**: `dph_compute_invalidated = true` is set synchronously inside `set_was_dph(True)` (`phasic_pybind.cpp:1374-1376`), but the actual wipe of `graph->parameterized_reward_compute_graph_off` (etc.) doesn't happen until the *next* `ptd_precompute_reward_compute_graph(graph)` call (`phasic.c:1948`). Between those two points, `graph->parameterized_reward_compute_graph_off` (if it was already populated) **still points at the pre-was_dph tape**, even though `graph->was_dph` already reads `true`. Any C function that reads `graph->parameterized_reward_compute_graph_off` directly, without first calling `ptd_precompute_reward_compute_graph`, would silently operate on stale, pre-discretization elimination structure — not a crash, a **silently wrong gradient**. Checked: the one production function that reads this field, `ptd_sojourn_grad_theta_subset`, correctly calls `ptd_precompute_reward_compute_graph(graph)` as its very first action (`:11415`) before touching `_off` (`:11419`) — so this is not live today, but the field itself carries no self-describing "am I stale" marker; safety depends entirely on caller discipline (landmine L7 below).

---

## 5. Landmine list

**L1 — [FIXED, the reference case] `input_specs` NULL on mmap load.** `ptd_load_pcg_rev3_mmap` leaves `off->input_specs = NULL` unconditionally (`phasic.c:3816`) while `ptd_pcg_convert_to_offset` and `ptd_load_pcg_rev3_copy` always populate it. `ptd_sojourn_grad_theta_subset` was written assuming the in-memory-conversion pattern used by its three sibling functions (`ptd_moment0_grad_theta`, `ptd_moments_grad_theta`, `ptd_moments_grad_theta_log`, `ptd_moments_grad_theta_dph` — all of which build a **local**, never-cached `ptape` and convert it themselves, so `input_specs` is always populated for them), but it alone reads the **shared, possibly-disk-loaded** `graph->parameterized_reward_compute_graph_off` — inheriting the mmap gap. Fixed with an explicit guard at `:11432`. **Any future exact-gradient/exact-Jacobian C function that reads `graph->parameterized_reward_compute_graph_off->input_specs` needs the identical guard**; there is no shared helper enforcing it — the check is copy-pasted logic at one call site, not a property of the type.

**L2 — [NEW] Unbounded `MEM`-kind pointer offset in the rev-1/2 decoder.** `ptd_pcg_decode_ptr_impl` (`phasic.c:3244-3246`) does `mem_base + enc->doubles_offset` with zero bounds checking, in contrast to the `EDGE` branch two lines below it which *is* bounds-checked, and in contrast to rev-3's `ptd_pcg3_off_validate`. A future function that trusts "the loader already validated this file" (true for rev-3, false for rev-1/2) would inherit an out-of-bounds **write** primitive from any `.bin` file loaded via `_load_param_compute_graph`/`_load_synthetic_param_compute_graph_ex` — currently test-only call sites, but the format is the one written by the production `push_cache()`/read by a hypothetical future `pull_cache()`-integration that actually loads the parent artifact.

**L3 — [NEW] Unvalidated command `type` on the rev-1/2 load path → process-wide abort.** No check equivalent to rev-3's `c->type < 0 || c->type > 6` (`:3641`) exists before `commands[i].type = encoded_cmds[i].type` (`:4460`); the shared raw executor's `default: DIE_ERROR(...)` (`:9720`) will `DIE_ERROR`-abort the *entire process* (not just fail the call) on any corrupt or adversarial file of this format.

**L4 — [NEW] `pull_cache()`'s parent artifact is format-incompatible with the only automatic consumer.** Parent artifacts are always saved rev-1 (`compute_repository.py:272`); the automatic Stage-A2 loader only accepts rev-3 magic. A pulled parent `.bin` sits inertly at the exact path/filename the automatic loader checks, and is silently ignored (magic mismatch → miss), contradicting the `pull_cache()` docstring's claim that "the next call to expectation()/pdf()/moments() reuses the published elimination" (`_graph_cache_transfer.py:14-17`). Only per-SCC artifacts (genuinely rev-3) are actually auto-consumed.

**L5 — [NEW, dormant] The per-SCC rev-3 cache has the same `input_specs`-NULL-on-mmap shape as L1, with no guard yet.** `scc_synthetic.c` installs `synth->parameterized_reward_compute_graph_off` from the identical `ptd_load_pcg_rev3` (`:1173-1178`), so a hypothetical future per-SCC exact-gradient C function (analogous to `ptd_sojourn_grad_theta_subset` but operating on a synthetic SCC graph) would need the identical `input_specs == NULL` guard, and there is currently no `input_specs` reference anywhere in `scc_synthetic.c` to model the fix on — nothing today would remind a future author this landmine exists at the SCC layer too.

**L6 — [NEW, minor] `commands`/`mem_base` ownership silently differs by construction path.** Per §1.4, these two pointers are heap-owned on two of three paths and mmap-backed (must be `munmap`'d, never `free()`'d) on the third. `ptd_pcg_desc_off_destroy` already branches correctly on `mem_is_mmap` (`:9611-9619`), but any *new* code that reaches directly into the struct (e.g. to `realloc()` `commands` for an in-place patch, or to hand a raw pointer to another owner) rather than going through the destroy function would corrupt memory or leak, depending purely on which of the three paths produced the instance it's holding — indistinguishable at the call site without explicitly checking `mem_is_mmap`.

**L7 — [NEW, minor] `graph->parameterized_reward_compute_graph_off` carries no self-describing staleness marker.** Per §4, the field can be non-NULL yet stale (pre-`was_dph` topology) in the window between `set_was_dph(True)` and the next `ptd_precompute_reward_compute_graph` call. The only thing that currently makes this safe is that the one reader (`ptd_sojourn_grad_theta_subset`) happens to call `ptd_precompute_reward_compute_graph` first. A future function that copies the "check `_off`, else fall back to raw, else decline" pattern (`:11419-11439`) but omits the preceding `ptd_precompute_reward_compute_graph` call (easy to omit — it's not enforced by the type system or an assertion) would silently compute a gradient against stale, pre-discretization structure instead of crashing or declining.

**L8 — [NEW, low-confidence/informational] `use_dyn_ordering` is not part of `ptd_graph_content_hash`.** A cache file built with the static elimination order and one built with the dynamic min-degree order for the *same* topology+coefficients+was_dph hash to the same filename and would satisfy each other's cache lookup. This is very likely benign — Gaussian elimination order does not change the final numeric result, only performance/build path, and vertex indices in the saved commands are independent of ordering strategy — but it is an undocumented cross-path invariant, not a proven-safe or tested one, and is exactly the shape of thing ("a build-time flag that isn't part of the cache key") that has caused similar bugs elsewhere in this codebase.

---

## Summary

Beyond the already-fixed `input_specs`-NULL-on-mmap bug (`ptd_sojourn_grad_theta_subset`, guarded at `src/c/phasic.c:11432`), five new landmines of the same general shape were found, all in the on-disk caching layer:

1. **Unbounded pointer arithmetic in the legacy rev-1/2 loader** (`phasic.c:3244-3246`): a `MEM`-kind operand's file-supplied offset is added to `mem_base` with zero bounds checking (unlike the `EDGE` branch right next to it, and unlike rev-3's hardened validator) — an out-of-bounds **write** primitive from a crafted `.bin` file.
2. **Unvalidated command-type byte on the same path** (`phasic.c:4460`, consumed at `:9720`): a corrupt/malicious file triggers `DIE_ERROR`, aborting the whole process — rev-3 explicitly guards against exactly this, rev-1/2 does not.
3. **The public GitHub trace registry (`Graph.pull_cache()`) is the one real network-facing consumer of that unhardened rev-1/2 format** — the security comment justifying rev-3's hardening ("pulled from the community registry") is attached to the wrong file format in practice.
4. **`pull_cache()`'s "parent" artifact can never be auto-consumed**: it's saved as rev-1, but the only automatic loader (Stage-A2) only accepts rev-3 magic, so the download silently becomes dead weight, contradicting the docstring.
5. **The per-SCC on-disk cache (`scc_synthetic.c`) shares the exact `input_specs`-NULL-on-mmap hazard** with zero guard today — dormant until a future per-SCC exact-gradient C function is written.

Two more minor/informational findings: `commands`/`mem_base` ownership (heap vs. mmap) is invisible outside `mem_is_mmap`, so any future code bypassing `ptd_pcg_desc_off_destroy` risks a leak or corruption depending on construction path; and `graph->parameterized_reward_compute_graph_off` can be transiently stale (pre-`was_dph`) with no self-describing marker, safe today only because the sole reader happens to call `ptd_precompute_reward_compute_graph` first.
