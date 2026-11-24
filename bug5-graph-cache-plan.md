# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v2, 2026-11-24. Branch `fix/graph-cache-stale`, worktree
`/Users/kmt/phasic-bug5` (own pixi env), rebased on master `5ef0d76c`
(bugs 1-4 merged). No source changed yet. Python-only change. Backward
compatibility is not required (user decision, 2026-11-23).**

**v1 (dependency fingerprint) was rejected by review 1** (evidence in
`experiments/bug5_graph_cache/review1/`). No inspection of a callback can
see everything it depends on: helpers behind `functools.lru_cache` or
`np.vectorize` pickle identically after their body changes, objects
pickled by reference hide their contents, two lambdas on one line share
their source, a source file edited after import differs from the running
code, and a `StateIndexer` does not pickle at all. v2 stops trying to infer
the dependencies.

**User decisions taken for v2 (2026-11-24):**
- D1: caching happens only when the user names the entry with
  `Graph(..., cache_key="...")`. `graph_cache=` is removed.
- D2: the global switch (`phasic.configure(graph_cache=...)`,
  `PHASIC_DISABLE_GRAPH_CACHE`) is removed. A per-graph `cache_key` is
  the only way to cache, so there is no global/explicit conflict to resolve.

Evidence on unmodified code (HOME redirected; the real `~/.phasic_cache`
is never read): `experiments/bug5_graph_cache/b1.log`, `b2.log`.

## 1. The defect (read from source + measured)

`Graph(callback, ...)` looks the graph up on disk before building it
(`src/phasic/__init__.py:1682-1727`) and saves it afterwards
(`:1792-1803`). The cache is ON by default (`config.py:611`,
`graph_cache: bool = True`; the env table at `config.py:465` encodes the
same default). The key (`callback_hash.py:33-170`) hashes a version tag,
the Python version, the ipv, the callback's own source (AST) and `repr()`
of the keyword arguments. Anything else the callback reads is missing:
module variables, captured variables and helper functions changed between
two builds all return the stale graph silently (b1.log: 0.833333 returned,
0.083333 correct), across processes. Further faults:
- `repr()` of a long numpy array is abbreviated, so two arrays differing
  in a middle element share a key;
- a failed key computation is swallowed (`graph_cache.py:183-187`, and the
  save is wrapped in a warning at `__init__.py:1798-1803`);
- the key does not identify the phasic build;
- the graph cache ignores `cache_dir` / `PHASIC_CACHE_DIR`: it always
  uses `DEFAULT_CACHE_DIR = Path.home()/".phasic_cache"/"graphs"`
  (`graph_cache.py:32`), while the C-side caches honour `PHASIC_CACHE_DIR`
  (`cache.py:64-76`);
- the key is computed twice per construction (load and save).

What the cache is worth (b2.log): build 0.076 s vs load 0.026 s at 1,003
vertices, 3.47 s vs 0.98 s at 5,823 vertices.

## 2. Fix

**Contract.** The user names a cache entry. phasic adds to the key what it
can establish exactly: the callback's own source, the keyword arguments
and initial vector BY VALUE, and the phasic build. Everything else the
callback depends on (module globals, helper functions, files) is the
user's responsibility: change `cache_key` when it changes. That contract
is stated in the `cache_key` docstring with an example.

### 2.1 `Graph(..., cache_key=None)`

- Signature: `Graph(arg, ipv=None, cache_key=None, **kwargs)`; the
  `graph_cache` parameter is removed.
- `cache_key=None` (default): no lookup, no file written, no hashing.
- `cache_key` given:
  - it must be a non-empty `str`, otherwise `TypeError`/`ValueError`;
  - `arg` must be a callback, otherwise `ValueError` ("cache_key applies
    to graphs built from a callback");
  - the ENTRY KEY (§2.2) is computed ONCE, before anything is built. A
    key error (§2.3) propagates: nothing is built and nothing is written;
  - entry present and readable: the graph is loaded exactly as the
    current hit path does (`__init__.py:1689-1723`: callback, kwargs,
    weight mode, formula tape, trace state restored), plus
    `self._graph_cache_entry = entry_key`;
  - entry absent: build as today, then save under the same entry key
    (atomic write: temporary file in the same directory + `os.replace`,
    so a concurrent reader never sees a partial file). A write failure
    raises `OSError` naming the path (caching was requested and could
    not happen; the user re-runs with the cache directory fixed or
    without `cache_key`);
  - entry present but unreadable (bad JSON, wrong format field,
    `from_serialized` fails): `warnings.warn(UserWarning)` naming the
    file and the reason, rebuild, overwrite. The key was sound; only the
    stored copy is bad, so the result is still correct.
- `graph_cache=` passed: `TypeError` stating that it was removed and that
  `cache_key="<name>"` caches a graph. (Without this check the argument
  would fall into `**kwargs` and be handed to the callback.)
- `cache_key` is never passed to the callback.

### 2.2 The entry key (`callback_hash.py`, rewritten)

`cache_entry_key(cache_key, callback, **params) -> str`: the first 32 hex
characters of SHA-256 over:
1. `GRAPH_CACHE_FORMAT = "2"` (replaces `PHASIC_CALLBACK_VERSION`; no
   entry written before can match);
2. the Python major.minor version;
3. the phasic build: `_phasic_build_id()` = SHA-256 of the bytes of the
   installed `phasic_pybind` extension file and of every `*.py` file in
   the installed `phasic` package directory (sorted by relative path),
   computed once per process. A rebuild or re-install that changes any of
   them changes every key;
4. `cache_key`;
5. the callback's normalised AST (the existing `_normalize_ast`: no
   positions, no docstrings). The `_callback` / `with_ipv` wrapper is
   unwrapped as today and its `__ipv__` is value-fingerprinted.
   Unavailable source raises `GraphCacheKeyError`;
6. every keyword argument, sorted by name, by VALUE FINGERPRINT. The
   construction kwargs (`theta_dim`, `cache_trace`, `hierarchical`,
   `indexer`, user kwargs) are all included, since all of them
   affect the graph or its stored attributes.

VALUE FINGERPRINT (exact rules, recursive):
- `None`, `bool`, `int`, `str`, `bytes`: type name + `repr`;
  `float` via `float.hex`; `complex` via the hex of both parts;
- numpy scalar: dtype + bytes; numpy array: dtype, shape and C-order
  bytes (object dtype refused);
- `list`, `tuple`: type + element fingerprints in order; `dict` with
  string keys: (key, value) pairs sorted by key (other key types refused);
- an object with a callable `to_dict()` method: class module + qualified
  name, then the fingerprint of what `to_dict()` returns, under the same
  rules. This is the protocol the docs already describe for passing
  objects to a callback (`docs/pages/tutorial/state-space.ipynb` cell 44)
  and the one `StateIndexer` implements (`state_indexing.py:1850`, a
  complete structural description: property sets, properties, slots,
  layout order);
- anything else (functions, generators, arbitrary objects, sets,
  object arrays): NOT FINGERPRINTABLE.

### 2.3 Errors

`GraphCacheKeyError(ValueError)`, exported from `phasic`, raised when the
entry key cannot be computed. Its message names the keyword argument and
the path inside it (for example `model -> to_dict()['rates'][2]`), the
offending type, and the ways out: pass plain values, give the object a
`to_dict()`, or build without `cache_key`.

### 2.4 Global configuration removed

- `config.py`: field `graph_cache` (`:611`), its documentation
  (`:549-555`), the env-table row (`:465`), the from-env read
  (`:325-326`), the two env-var lists (`:201`, `:930`) and the report
  entry (`:914`) are removed.
- `phasic.configure(graph_cache=...)`: raises the existing "Unknown
  configuration option" `ValueError` with an added line saying the option
  was removed and pointing to `Graph(..., cache_key=...)`.
- `PHASIC_DISABLE_GRAPH_CACHE` set in the environment: one
  `UserWarning` when the configuration is first created, saying it no
  longer has an effect.
- `graph_cache.is_graph_cache_disabled` is deleted.

### 2.5 Cache location

`GraphCache()` with no argument resolves its directory at construction
time: `Path(os.environ["PHASIC_CACHE_DIR"]) / "graphs"` when that env var
is set (the variable `configure(cache_dir=...)` writes, the same one the C
caches read), otherwise the module attribute `DEFAULT_CACHE_DIR`. Reading
the attribute at call time keeps the existing tests' monkeypatch of
`graph_cache.DEFAULT_CACHE_DIR` working. `get_cache_dir`,
`get_graph_cache_stats`, `print_graph_cache_info` and
`clear_all_graph_caches` use the same resolution.

### 2.6 `GraphCache` API

- Kept: `GraphCache(cache_dir=None)`, `clear()`, `clear_graph_cache()`,
  `get_cache_stats()`, `_serialize_value` / `_deserialize_value` (used by
  `test_serialize_is_aux.py`).
- Replaced: `save_graph(graph, callback, **params)` and
  `load_graph(callback, **params)` become `save(graph, entry_key,
  meta)` and `load(entry_key) -> Graph | None` (None only for a missing
  file; an unreadable file raises `GraphCacheEntryError`, which
  `Graph.__init__` turns into the warning + rebuild of §2.1).
- Removed: `get_or_build` (`Graph(cb, cache_key=...)` does this) and
  `hash_callback` / `_detect_closures`.
- The JSON entry keeps its current fields plus `format`, `cache_key`
  and `callback_name` (for people inspecting the directory). On load,
  `format` must equal `GRAPH_CACHE_FORMAT`.

### 2.7 `Graph.clear_from_cache(graph_cache=True, ...)`

(`_graph_cache_mgmt.py:70-91`; the parameter name selects which cache
to clear and is unrelated to the removed option.) It uses
`self._graph_cache_entry`. A graph built without `cache_key` has no entry:
`removed["graph_cache"]` is 0, documented. A graph not built from a
callback still raises as today.

### 2.8 Old entries

Entries written before this change are never loaded (the format and the
key composition both differ). They stay on disk;
`phasic.clear_all_graph_caches()` removes them (docstring says so).

## 3. Tests

New file `tests/pytest/test_graph_cache_key.py`. Every test points
`graph_cache.DEFAULT_CACHE_DIR` at `tmp_path` via monkeypatch and clears
`PHASIC_CACHE_DIR`, except test 14, which sets it. A hit is proven by a
spy on the callback (not called); "stale" is judged on `expectation()`
against a build without `cache_key`.

Fail on master:
1. Default build is uncached: a module-level variable changed between two
   `Graph(cb, ipv=...)` builds gives the correct second graph, and no file
   appears under the cache directory (master: stale, file written).
2. `Graph(cb, ipv, graph_cache=True)` and `graph_cache=False` raise
   `TypeError` mentioning `cache_key`; `configure(graph_cache=True)`
   raises `ValueError` mentioning `cache_key`.
3. Same `cache_key`, same everything: hit; identical vertex count,
   `expectation()` and `serialize()` arrays; also across two subprocesses
   sharing the directory (`PYTHONHASHSEED` 0 and 1).
4. Different `cache_key`: miss.
5. Keyword argument changed (int); long numpy array differing in one
   middle element (identical abbreviated repr): miss and correct.
6. ipv changed (wrapper and `ipv=` forms): miss.
7. Callback body changed: miss. Comment, docstring and whitespace edits:
   hit.
8. `indexer=StateIndexer(...)` built afresh with the same layout: hit; a
   different property range: miss.
9. Object with `to_dict()`: hit when equal, miss when a field differs.
   Object without `to_dict()`: `GraphCacheKeyError` naming the argument
   and the class; callback not called, no file written.
10. A function or a set as a keyword argument: `GraphCacheKeyError`.
11. `_phasic_build_id` monkeypatched to another value: existing entry not
    loaded.
12. Corrupt entry (truncated JSON; wrong `format`): `UserWarning`,
    correct graph, file rewritten and loadable.
13. Write failure (`os.replace` monkeypatched to raise): `OSError` with
    the path.
14. `PHASIC_CACHE_DIR=<tmp>`: the entry lands in `<tmp>/graphs`, not
    under `DEFAULT_CACHE_DIR`.
15. `cache_key` validation: `cache_key=3`, `cache_key=""`, and
    `Graph(3, cache_key="x")` raise.
16. `clear_from_cache()` removes the entry of a `cache_key` graph and
    reports 0 for a graph without one.
17. `PHASIC_DISABLE_GRAPH_CACHE=1` (fresh configuration): `UserWarning`
    that it has no effect.

Contract pin (passes on both, documents the user's responsibility):
18. Same `cache_key`, module-level helper changed between builds: the
    cached graph is returned. The docstring example states exactly this.

Existing tests edited (behaviour asserted stays the same; only the way
the cache is requested changes):
- `test_graph_cache.py::TestCacheHitPreservesCallback` (2 tests):
  `graph_cache=True` -> `cache_key=...`; `graph_cache=False` -> argument
  removed.
- `test_serialize_constant_roundtrip.py`: lines 106-113 and 134 use
  `cache_key`; `load_graph(callback)` at 111 -> `load(entry_key)`; T6
  (`:219-240`) rewritten on the new API: write an entry under format "1"
  (monkeypatched `GRAPH_CACHE_FORMAT`), check that it is not loaded under
  "2" and that the rebuilt graph is constant.
- `test_theta_layout.py::test_graph_cache_hit_gets_default_layout`,
  `test_weight_formula_theta_dim.py::test_graph_cache_preserves_theta_dim`
  (drop `configure(graph_cache=...)`),
  `test_from_serialized_find_vertex.py::test_joint_prob_graph_via_warm_graph_cache`
  (spy moves to `GraphCache.load`): `cache_key`.
- `test_config_scc_controls.py::test_configure_disable_graph_cache_sets_env`
  -> asserts the `ValueError` instead.
- `test_constant_parameterized_consistency.py:53,59,232` and
  `_discretize_pins.py:47,51,86`: drop `graph_cache=False` (no change in
  behaviour: those builds were uncached and stay uncached).
A grep for `graph_cache`, `GraphCache`, `hash_callback`,
`PHASIC_CALLBACK_VERSION` and `PHASIC_DISABLE_GRAPH_CACHE` in `tests/`
and `src/` must come back clean (apart from the new names and the
unrelated `per_thread_graph_cache` / `param_compute_graph_cache` /
`clear_from_cache(graph_cache=...)`) before the implementation review.

## 4. Gate

Same tools and rules as the other fixes: `pixi run install-dev` in the
worktree, `gate2.sh /Users/kmt/phasic-bug5 bug5`, `gate_compare.py
/Users/kmt/phasic-bug4/.gate_bug4 /Users/kmt/phasic-bug5/.gate_bug5
--new-file tests/pytest/test_graph_cache_key.py`, three isolated reruns
for any regression. The new tests are also run on the master build and
the outcome recorded. Suite wall time before and after is reported (the
suite no longer gets cached construction by default). The cost of the
entry key at 1,003 and 5,823 vertices is measured against the load time
(b2 models; reported, not hidden). `_phasic_build_id` is measured on its
own once.

## 5. Notebooks (not edited — list for the user)

Scanned read-only from the main tree's current notebooks
(2026-11-24). Cells that raise after the change:
- `popgen/moran-coalescent-derived.ipynb` cell 1:
  `configure(graph_cache=True)` -> remove the line.
- `tutorial/matrix-solve.ipynb` cell 2: `phasic.configure(graph_cache=True)`
  -> remove.
- `popgen/old-moran-coalescent-derived.ipynb` cell 4: `graph_cache=False`
  -> remove.
- `tutorial/python_trace-and-jax-caching.ipynb` cells 8, 10, 12 and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cells 8, 10, 12:
  `graph_cache=True` -> `cache_key="two-locus-arg"` (cells 10/12 show
  that `dummy=42` vs `dummy=99` give separate entries; still true).
- `tutorial/state-space.ipynb` cells 36, 38, 45: `graph_cache=True` ->
  `cache_key=...`. Cell 45's `MyModel` keeps working through `to_dict()`;
  its `__repr__` is no longer needed.
Text to revise (not raising):
- `tutorial/state-space.ipynb` cells 31, 39, 44 (cache description;
  "if `graph_cache=True` is not passed"; the `__repr__` requirement).
- `tutorial/python_trace-and-jax-caching.ipynb` cells 1, 7 and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cells 1, 7 (how to
  enable; the cache-correctness sentence must state the user-key
  contract).
- `tutorial/distributed.ipynb` cell 31 (markdown example with
  `graph_cache=True`) and cell 37.
- `popgen/ghost.ipynb` cell 3, `popgen/two-locus-island.ipynb` cells 11,
  29, 37: commented-out `graph_cache.save_graph(...)` lines use the
  removed method (harmless while commented).
- `tutorial/svgd-joint-prob.ipynb` cell 78: commented-out
  `graph_cache=True`.
Unaffected: `GraphCache()` construction and `.clear()`
(`state-space` cell 33, `ghost` cell 1, `two-locus-island` cell 1),
`print_graph_cache_info` / `get_graph_cache_stats`, and
`clear_from_cache(graph_cache=False)` (`scc-decomposition` cells 26, 29).

## 6. Risks

- Construction is uncached unless a key is given (3.5 s instead of 1 s
  at 5,823 vertices on repeat builds).
- A user who changes a helper and keeps the key gets the old graph. That
  is the documented contract, pinned by test 18, and is no worse than
  today, where it happens with no key at all.
- `to_dict()` that omits a field the callback uses gives a stale hit for
  that field. The protocol docstring says `to_dict()` must describe
  everything the callback reads from the object.
- The build id hashes ~5 MB of files once per process (measured in the
  gate). A development edit to any installed `.py` file invalidates all
  entries; that is intended.
