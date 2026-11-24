# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v2.5, 2026-11-24. Branch `fix/graph-cache-stale`, worktree
`/Users/kmt/phasic-bug5` (own pixi env), rebased on master `5ef0d76c`
(bugs 1-4 merged). No source changed yet. Python-only change. Backward
compatibility is not required (user decision, 2026-11-23).**

History: v1 (dependency fingerprint) rejected by review 1
(`experiments/bug5_graph_cache/review1/`). v2 reviewed by review 4
(`experiments/bug5_graph_cache/review4/`): REVISE, 14 findings, all taken
in v2.1 (mapping in §7).

v1's lesson: no inspection of a callback can see everything it depends on
(helpers behind `lru_cache`/`np.vectorize` pickle identically after a body
change, objects pickled by reference, a source file edited after import, a
`StateIndexer` does not pickle). v2 stops inferring dependencies through
module globals and lets the user name the entry.

**User decisions (2026-11-24):**
- D1: caching happens only when the user names the entry with
  `Graph(..., cache_key="...")`. `graph_cache=` is removed.
- D2: the global switch (`phasic.configure(graph_cache=...)`,
  `PHASIC_DISABLE_GRAPH_CACHE`) is removed.

Evidence on unmodified code (HOME redirected; the real `~/.phasic_cache`
is never read): `experiments/bug5_graph_cache/b1.log`, `b2.log`,
`review4/p1_probes.log`.

## 1. The defect (read from source + measured)

`Graph(callback, ...)` looks the graph up on disk before building it
(`src/phasic/__init__.py:1682-1727`) and saves it afterwards
(`:1792-1803`). The cache is ON by default (`config.py:611`; the env table
at `config.py:465` encodes the same default). The key
(`callback_hash.py:33-170`) hashes a version tag, the Python version, the
ipv, the callback's source (AST) and `repr()` of the keyword arguments.
Anything else the callback reads is missing: module variables, captured
variables and helper functions changed between two builds all return the
stale graph silently (b1.log: 0.833333 returned, 0.083333 correct), across
processes. Further faults:
- `repr()` of a long numpy array is abbreviated, so two arrays differing
  in a middle element share a key;
- a failed key computation is swallowed (`graph_cache.py:183-187`; the
  save is wrapped in a warning at `__init__.py:1798-1803`); the whole hit
  path sits in `except Exception` (`:1688-1726`) and falls through to a
  second base-class `__init__`, which is silently ignored by pybind
  (review4 p3: `state_length` stays 3 after re-init with 5);
- the key does not identify the phasic build;
- a cache hit skips argument validation: with a warm cache,
  `Graph(decorated_cb, ipv=[9])` returns without the "ipv is ignored"
  `ValueError` that a cold build raises (review4 p1 H);
- the graph cache ignores `cache_dir` / `PHASIC_CACHE_DIR`
  (`graph_cache.py:32`), while the C caches honour it (`cache.py:64-76`);
- `phasic.clear_caches()` wipes the user's real graph cache
  (`model_export.py:79-83`) and the existing test
  `test_weight_formula_theta_dim.py::test_graph_cache_preserves_theta_dim`
  calls it without redirecting the directory.

What the cache is worth (b2.log): build 0.076 s vs load 0.026 s at 1,003
vertices, 3.47 s vs 0.98 s at 5,823 vertices.

## 2. Fix

**Contract.** The user names a cache entry. phasic adds to the key
everything it can read exactly from the call: the callback function object
(its running bytecode, constants, defaults, closure contents, bound
`self`, recursively), the keyword arguments and initial vector BY VALUE,
and the phasic build. What the callback reaches through MODULE GLOBALS
(helper functions, module variables, imported objects), files and the
environment is the user's responsibility: change `cache_key` when it
changes. The `cache_key` docstring states this with an example.

### 2.1 `Graph(..., cache_key=None)`

- Signature: `Graph(arg, ipv=None, cache_key=None, **kwargs)`; the
  `graph_cache` parameter is removed. `self._graph_cache_entry = None`
  is set at the top of `__init__`.
- `graph_cache=` passed: `TypeError` saying it was removed and that
  `cache_key="<name>"` caches a graph (otherwise it would reach the
  callback through `**kwargs`).
- `cache_key=None` (default): no lookup, no hashing, no file.
- `cache_key` given:
  - non-`str` -> `TypeError`, empty -> `ValueError`; `arg` not callable ->
    `ValueError` ("cache_key applies to graphs built from a callback");
  - ALL argument validation for a callback runs first: the
    ipv/decorator checks now at `:1747-1755` move above the lookup (for
    every call, cached or not; same errors, raised earlier);
  - the ENTRY KEY (§2.2) is computed once. A key error propagates:
    nothing is built, nothing written;
  - lookup: `GraphCache().load(entry_key)` returns a separate, fully
    built `Graph` or `None` (missing file), or raises
    `GraphCacheEntryError` (unreadable: bad JSON or encoding, wrong
    shape, wrong `format`; exact classes in §2.6). An exception from
    `from_serialized` itself is NOT an entry error: it is re-raised as
    `GraphCacheLoadError` naming the file (§2.6) and propagates: the
    build id guarantees the same phasic build wrote the entry, so a
    well-formed entry that fails to load is a phasic bug or tampering,
    and must not be hidden behind a rebuild. ONLY that call is inside a `try`, and
    only `GraphCacheEntryError` is caught: `warnings.warn(UserWarning)`
    naming the file and reason, then treated as a miss. The broad
    `except Exception` is removed. Nothing after
    `super().__init__(cached_graph)` is inside a `try`: an error while
    restoring attributes on a hit propagates;
  - hit: the current restore sequence (`:1689-1723`) runs, plus
    `self._graph_cache_entry = entry_key`;
  - miss: build as today, then `GraphCache().save(self, entry_key, meta)`
    and `self._graph_cache_entry = entry_key`. A write failure raises
    `OSError` naming the path (the graph built correctly, but caching
    was requested and could not happen).
- `cache_key` is never passed to the callback.

### 2.2 The entry key (`callback_hash.py`, rewritten)

`cache_entry_key(cache_key, callback, **params) -> str`: the first 32 hex
characters of SHA-256 over a TOKEN STREAM. Every token is written as
`<tag>:<len>:<payload>` with the payload's byte length, so no two token
sequences can produce the same bytes. Components in order:
1. `GRAPH_CACHE_FORMAT = "2"` (replaces `PHASIC_CALLBACK_VERSION`);
2. Python `sys.version_info[:2]` (bytecode is version-specific);
3. the phasic build id (§2.2.3);
4. `cache_key`;
5. FUNCTION FINGERPRINT of the callback as passed to the C builder, i.e.
   after the `_callback(ipv)` wrapping (`:1666-1671`). The wrapper's
   closure holds the user function and the ipv, so both are covered by
   the generic rule. No special case for phasic wrappers;
6. the keyword arguments that reach the builder or are stored on the
   graph: all of `kwargs` after `cache_trace` and `hierarchical` are
   popped (`:~1608`; they are forced no-ops). This includes `theta_dim`,
   `indexer` and user kwargs. In the order given (insertion order of
   `**kwargs`), name + VALUE FINGERPRINT.

#### 2.2.1 FUNCTION FINGERPRINT (exact, recursive, visited set; a
function already visited contributes a back-reference token)

- Python function `f`: CODE FINGERPRINT of `f.__code__`; VALUE
  FINGERPRINT of `f.__defaults__` and `f.__kwdefaults__`; for each name in
  `co_freevars`, the cell contents (empty cell -> token `empty`) by VALUE
  FINGERPRINT, where a cell holding a function is fingerprinted as a
  function (so decorator wrappers and captured helpers are covered).
- bound method: FUNCTION FINGERPRINT of `__func__` + VALUE FINGERPRINT of
  `__self__`.
- `functools.partial`: function, `args`, `keywords`.
- builtin function, numpy ufunc / array-function dispatcher: module +
  qualified name token, ONLY IF `__self__` is `None` or a module AND
  resolving `__module__` + `__qualname__` (or `__name__` for ufuncs)
  through `sys.modules` returns the very same object. This admits
  `math.sqrt`, `len`, `np.exp`, `np.linalg.solve` and refuses builtin
  bound methods of instances (`{}.get`, `Random(1).random`, which carry
  state in `__self__`) and `np.frompyfunc` ufuncs ("<lambda>
  (vectorized)"), which would otherwise share a name token (review 5 q1
  D). It also refuses every `scipy.special` ufunc (`gamma`, `binom`:
  their `__module__` is `None`, review 6 A); the error advises moving
  such a function to a module global (then covered by the user's
  `cache_key`, as for any imported object). Note the asymmetry, which is
  consistent with the contract: compiled dispatchers are keyed by name,
  while Python-level library functions reached through a cell (for
  example `scipy.special.comb`) are keyed by their code.
- anything else callable (class, callable instance without `to_dict`):
  NOT FINGERPRINTABLE.
- Recursion: the visited set maps each function to its first-visit
  index; a revisit contributes a back-reference token carrying that
  index, so different recursion shapes give different streams.

CODE FINGERPRINT of a code object: `co_code`, `co_exceptiontable`,
`co_names`, `co_varnames`, `co_freevars`, `co_cellvars`, `co_argcount`,
`co_posonlyargcount`, `co_kwonlyargcount`, `co_flags`, `co_name`, and
`co_consts` element by element (nested code objects by CODE FINGERPRINT;
`frozenset` constants by their elements' fingerprints, sorted; `Ellipsis`
token; `slice` constants (3.14 folds constant slices) by start/stop/step;
other constants by VALUE FINGERPRINT; a constant type none of these
covers raises `GraphCacheKeyError`, never a crash). `co_exceptiontable`
is read with `getattr(co, "co_exceptiontable", b"")` (absent before
3.11; `requires-python >= 3.10`, pixi pins < 3.14; only 3.13 is
tested here). Excluded: `co_filename`,
`co_firstlineno`, `co_linetable`, `co_qualname`. Consequences (all
tested): this is the code that RUNS, so editing the file on disk without
reloading changes nothing (review4 p1 D); comment and whitespace edits
hit; a docstring edit misses (docstrings are constants; a harmless miss);
two lambdas on one line get distinct keys; `inspect.getsource` is no
longer used, so callbacks without accessible source can be cached.

#### 2.2.2 VALUE FINGERPRINT (exact types checked first, in this order)

1. `None`, `bool`, `int`, `str`, `bytes` (exact types): type name +
   canonical repr; `float` via `float.hex`; `complex` via the hex of both
   parts;
2. numpy scalar (`np.generic` subclass): dtype str + `tobytes()`;
3. `type(x) is np.ndarray` exactly (masked arrays, `np.matrix` and other
   subclasses are refused): dtype str, shape, C-order bytes; object dtype
   refused;
4. exact `list` / `tuple`: type + element fingerprints in order;
5. exact `dict`: entries in INSERTION order (a callback iterating a dict
   sees that order, so it can change the graph); keys must be `str`;
6. any callable that §2.2.1 admits (Python function, bound method,
   partial, and builtins/ufuncs under exactly §2.2.1's conditions):
   FUNCTION FINGERPRINT; §2.2.1 and this rule are one rule;
7. a class object (`isinstance(x, type)`): NOT FINGERPRINTABLE (so a
   `__class__` cell from `super()` or a class passed as an argument is
   refused explicitly, never called through an unbound `to_dict`);
8. any other object with a callable `to_dict()` (including `dict`/`list`
   subclasses that have one): class module + qualified name, then the
   VALUE FINGERPRINT of what `to_dict()` returns; any `Exception` (not
   `BaseException`: `KeyboardInterrupt` / `SystemExit` pass through)
   raised inside `to_dict()` is re-raised as `GraphCacheKeyError` with
   the path and the original exception chained. This is the protocol
   the docs already describe for objects passed to a callback
   (`docs/pages/tutorial/state-space.ipynb` cell 44) and that
   `StateIndexer` implements (`state_indexing.py:1850`, which covers the
   same structure as its `__eq__`/`__hash__`, review4);
9. anything else: NOT FINGERPRINTABLE.
Recursion is bounded: a depth limit of 100, and a cycle check against the
ids of the containers / `to_dict()` objects on the CURRENT PATH from the
root only (pushed on entry, popped on exit). A process-wide seen-set is
wrong: it flags a list referenced twice, an ipv reusing one state list,
and freed temporaries whose ids are reused (review 5 q2). Both limits
raise `GraphCacheKeyError` (never `RecursionError`).

#### 2.2.3 Build id

At `import phasic`, `graph_cache` records `(path, size, mtime_ns)` for
the extension module located by `phasic.phasic_pybind.__file__` (not by
globbing; if it has no `__file__` or the file is missing, the first key
computation raises `GraphCacheKeyError`, so an editable or source-tree
layout can never silently leave the extension out of the id) and every
`*.py` file in the package directory of `phasic.__file__` (a few dozen
`stat` calls; no
reading). EVERY key computation re-stats those files (measured 2.1 ms,
review 5) and compares with the record; if any stat changed since import
(for example `install-dev` ran under a live kernel), `GraphCacheKeyError`:
"phasic was reinstalled after it was imported; restart the interpreter".
Only the content hash (measured: 52 .py + the .so, 3.49 MB, 11.9 ms;
review 4) is memoised, computed on the first key computation. `.cpp` /
`.h` files shipped in the package directory are not part of the id
(graph construction goes through the compiled extension only).

### 2.3 Errors

- `GraphCacheKeyError(ValueError)`, exported from `phasic`: the key
  cannot be computed. The message names the path to the offending value
  (for example `kwargs['model'] -> to_dict()['rates'][2]` or
  `callback -> closure 'helper' -> defaults[0]`), its type, and the ways
  out: pass plain values, give the object a `to_dict()`, move the value
  to a module global and change `cache_key` yourself, or build without
  `cache_key`.
- `GraphCacheLoadError(RuntimeError)`, exported: a well-formed entry
  failed in `from_serialized` (§2.6); names the file and the ways out.
- `GraphCacheEntryError(Exception)`, internal to `graph_cache`: an entry
  file is unreadable (caught only as described in §2.1).

### 2.4 Global configuration removed

- `config.py`: field `graph_cache` (`:611`) and its documentation
  (`:549-555`); the env-table row (`:465`) and its comment (`:444`, the
  convention text that cites it); the from-env read (`:325-326`); the
  env-var lists (`:201`, `:930`); the report entry (`:914`); the
  docstring mentions at `:239`, `:308`, `:607-609`; the inverted-bool
  example comment at `:1040-1041` (reworded to a remaining field).
- `phasic.configure(graph_cache=...)`: the existing unknown-option path
  raises `PTDConfigError` (`exceptions.py:14`; NOT a `ValueError`), with
  an added line saying the option was removed and pointing to
  `Graph(..., cache_key=...)`.
- `PHASIC_DISABLE_GRAPH_CACHE` set in the environment: one `UserWarning`
  at `import phasic`, next to the import-time `get_config()`
  (`__init__.py:133`), saying it no longer has an effect.
- `graph_cache.is_graph_cache_disabled` is deleted.
- `cache.py:26-28` docstring ("on by default") reworded.

### 2.5 Cache location

`GraphCache()` with no argument resolves its directory when constructed:
`Path(os.environ["PHASIC_CACHE_DIR"]) / "graphs"` when that variable is
set (written by `configure(cache_dir=...)` via `validate()`,
`config.py:819-845`, and read by the C caches), otherwise the module
attribute `DEFAULT_CACHE_DIR`, read at call time. Every test that
monkeypatches `DEFAULT_CACHE_DIR` also does
`monkeypatch.delenv("PHASIC_CACHE_DIR", raising=False)`, since the env
var takes precedence. `get_cache_dir`, `get_graph_cache_stats`,
`print_graph_cache_info`, `clear_all_graph_caches` and
`model_export.clear_caches` use the same resolution.

### 2.6 `GraphCache` API and entry file

- Kept: `GraphCache(cache_dir=None)`, `clear()`, `clear_graph_cache()`,
  `get_cache_stats()`, `_serialize_value` / `_deserialize_value`.
  `clear()` / `clear_graph_cache()` (and through them
  `clear_all_graph_caches` and `model_export.clear_caches`) also remove
  leftover `.tmp-*.partial` files, which a writer killed between
  `os.open` and `os.replace` (SIGKILL, OOM kill, SLURM time limit) leaves
  behind (review 7 C), but ONLY those whose mtime is more than one hour
  old, so a concurrent writer's in-flight file in a shared directory is
  never removed under it (review 8); `get_cache_stats` still counts only `*.json`.
- Replaced: `save_graph(graph, callback, **params)` /
  `load_graph(callback, **params)` -> `save(graph, entry_key, meta)` /
  `load(entry_key)`.
- Removed: `get_or_build`, `hash_callback`, `_detect_closures`,
  `_normalize_ast`.
- Entry content: `format`, `phasic_version`, `created_at`, `cache_key`,
  `callback_name`, `graph_data`. `construction_params` is DROPPED: the
  key already encodes the parameters, and serialising them could fail
  after the build for values the key accepts (np.bool_, complex, bytes:
  review4 p1 E).
- Write: `os.open(<cache dir>/f".tmp-{secrets.token_hex(8)}.partial",
  O_CREAT | O_EXCL | O_WRONLY, 0o666)` (unique per writer, never matching
  `*.json`; the kernel applies the process umask and any default ACL of
  the directory, as for today's `open()`, so no umask is read or changed
  (reading it with `os.umask` would briefly change it process-wide while
  JAX/OpenMP threads run), and no `mkstemp` 0600 mode), write, close,
  `os.replace` to `<entry_key>.json`; on any exception the
  temporary file is removed before the error propagates. Two concurrent
  writers each replace atomically; a reader sees a complete file or none
  (shared SLURM directories are a documented use,
  `distributed.ipynb` cell 31).
- On load, `format` must equal `GRAPH_CACHE_FORMAT`, otherwise
  `GraphCacheEntryError`.
- Read errors are classed exactly as:
  - `FileNotFoundError` on open -> `None` (miss);
  - from `json.load`: `ValueError` (covers `JSONDecodeError` and
    `UnicodeDecodeError` from invalid UTF-8, review 6 B) ->
    `GraphCacheEntryError`;
  - shape checks done explicitly before use: top level not a `dict`,
    a required field missing or of the wrong type, `format` not equal to
    `GRAPH_CACHE_FORMAT` -> `GraphCacheEntryError` (implemented as
    explicit checks, so no `KeyError`/`TypeError` escapes raw);
  - `graph_data` is decoded by a RESTRICTED decoder: plain JSON values
    plus the `{"__type__": "ndarray", ...}` tag only (the only tag
    `serialize()` output needs now that `construction_params` is gone).
    Any other `__type__` tag -> `GraphCacheEntryError` naming the file and
    the tag, WITHOUT importing anything. The general `_deserialize_value`
    (which imports the module a tag names and calls its `from_dict`,
    review 7 A) is no longer used on the load path; it stays only for
    `test_serialize_is_aux.py`;
  - `from_serialized`: NOT caught as an entry error (§2.1). An
    `Exception` from it, other than `MemoryError` and `RecursionError`
    (which say nothing about the file and propagate unchanged, as do
    `BaseException`s such as `KeyboardInterrupt`), is re-raised as
    `GraphCacheLoadError(RuntimeError)` (exported), whose
    message names the entry file and the ways out (delete that file, run
    `phasic.clear_all_graph_caches()`, or change `cache_key`), with the
    original exception chained (`raise ... from e`; not `add_note`,
    which needs Python 3.11);
  - any other `OSError` (for example `PermissionError` on an entry another
    user wrote under umask 077, common on HPC) PROPAGATES, re-raised as
    the same `OSError` subclass with the original chained, and a message
    naming the file, its owner (best effort: `os.stat` or `pwd.getpwuid`
    failing, for example in a non-searchable directory, gives "owner
    unknown" and never raises from the handler) and the two ways out (another `cache_key`, or another cache
    directory via `configure(cache_dir=...)`). Two users therefore never
    overwrite each other's entries back and forth silently.
  `GraphCacheEntryError` -> warn, rebuild, overwrite (§2.1).

### 2.7 `Graph.clear_from_cache(graph_cache=True, ...)`

(`_graph_cache_mgmt.py:70-91`; this parameter selects which cache to
clear and is unrelated to the removed option, so it stays.) Uses
`getattr(self, "_graph_cache_entry", None)`. No entry (built without
`cache_key`) -> `removed["graph_cache"] = 0`, documented. A graph not
built from a callback still raises as today.

### 2.8 Old entries

Entries written before this change are never loaded (different format
and key composition). They stay on disk; `phasic.clear_all_graph_caches()`
removes them (docstring says so).

## 3. Tests

New file `tests/pytest/test_graph_cache_key.py`. Every test
monkeypatches `graph_cache.DEFAULT_CACHE_DIR` to `tmp_path` and deletes
`PHASIC_CACHE_DIR` (test 14 sets it instead). Every subprocess gets
`HOME` and `PHASIC_CACHE_DIR` pointing into `tmp_path`, so the gate
(which runs with the real HOME) never touches the real cache. A hit is
proven by a spy on the callback (not called); "stale" is judged on
`expectation()` against a build without `cache_key`. Callbacks used with
`cache_key` accept `**kw`, so that on master the argument reaches them
harmlessly and the outcome recorded on master reflects the real
behaviour (on master `cache_key` is just ignored, and the default cache
is on).

Expected to fail on master:
1. Default build is uncached: a module-level variable changed between two
   builds gives the correct second graph, and no file appears.
2. `graph_cache=True` and `graph_cache=False` raise `TypeError`
   mentioning `cache_key`; `configure(graph_cache=True)` raises
   `PTDConfigError` mentioning `cache_key`.
3. Same `cache_key`, same everything: hit (spy not called), identical
   vertex count, `expectation()` and `serialize()` arrays; across two
   isolated subprocesses (`PYTHONHASHSEED` 0 and 1) too.
4. Different `cache_key`: miss.
5. Keyword argument changed (int); long numpy array differing in one
   middle element; dict with the same items in another insertion order:
   miss, and correct.
6. ipv changed (`ipv=` and `with_ipv` forms): miss.
7. Callback body changed (new function object from source with one
   constant changed): miss. Comment- and whitespace-only edit: hit.
   Two lambdas defined on one line: distinct keys.
8. Closure: a factory `make(rate)` called with 1.0 and 5.0, same
   `cache_key`: miss, both correct (review4 p1 C). Default-bound loop
   idiom `def cb(state, rate=n)`: miss when `n` changes. Closure holding
   an object without `to_dict()`: `GraphCacheKeyError` naming
   `closure '<name>'`.
9. Source file edited on disk, module not reloaded: same key (the
   running code is keyed).
10. `indexer=StateIndexer(...)` built afresh with the same layout: hit;
    a different property range: miss.
11. Object with `to_dict()`: hit when equal, miss when a field differs.
    Object without `to_dict()`, a `set`, a masked array, a cyclic
    `to_dict()`, a class object with a `to_dict` attribute, a `to_dict()`
    that raises, a closure cell holding `{}.get` or an `np.frompyfunc`
    ufunc: `GraphCacheKeyError` (not `RecursionError` / `TypeError`),
    callback not called, no file written. Not cycles: a list referenced
    twice in one kwarg, an ipv reusing one state list, a list of 200
    `to_dict()` objects: these hash and hit. A bound-method callback
    calling `super()`, on an instance that HAS a valid `to_dict()` (so
    `__self__` is fingerprintable): `GraphCacheKeyError` whose path names
    the `__class__` cell (proves the refusal comes from the class rule,
    not from `__self__`).
12. Values the key accepts that the old entry format could not store
    (`np.bool_`, complex, bytes keyword arguments): build, save and hit
    work.
13. `_phasic_build_id` monkeypatched to another value: miss. A recorded
    stat changed after import, simulated by monkeypatching the RECORDED
    stat table (never by touching installed files), both before the first
    key computation and after one: `GraphCacheKeyError` mentioning
    restart.
14. `PHASIC_CACHE_DIR=<tmp>`: entry in `<tmp>/graphs`.
15. Corrupt entry (truncated JSON; invalid UTF-8 bytes; a valid JSON
    list; a dict missing `graph_data`; wrong `format`): `UserWarning`,
    correct graph, file rewritten and then hit. A hit whose attribute
    restore raises: a FORMULA-mode entry is written by hand under the
    computed key (a graph with `weight_formula` set, serialised and
    passed to `GraphCache.save`, since `Graph.__init__` only ever saves
    linear-mode graphs and the tape restore at `__init__.py:1716` runs in
    formula mode only). `_set_weight_tape` is monkeypatched to pass
    through on its FIRST call (the one inside `from_serialized`,
    `_graph_serialize.py:733-741`, review 6 D) and raise on its SECOND
    (the restore on the live graph); the test asserts that the second
    call happened, that the error propagates, and that no `UserWarning`
    was emitted (no silent rebuild). An unreadable entry file (`chmod 000`, skipped when running
    as root) propagates `PermissionError` whose message names the file;
    with the directory made non-searchable instead (permissions restored
    in a `finally`, skipped when running as root), the message says
    "owner unknown" and the original error is chained. A `MemoryError`
    raised from a monkeypatched `from_serialized` propagates unchanged
    (not wrapped). A
    well-formed entry whose `from_serialized` raises (monkeypatched)
    gives `GraphCacheLoadError` naming the file, original chained, and no
    `UserWarning`. An entry
    whose `graph_data` carries `{"__type__": "<module>.<Class>"}` for a
    module not yet imported (a throwaway module written to `tmp_path` on
    `sys.path`): `UserWarning` + rebuild, and that module is NOT in
    `sys.modules` afterwards.
16. Write failure (`os.replace` monkeypatched to raise): `OSError` with
    the path, and no `.partial` file left in the directory. A stray
    `.tmp-x.partial` file with an mtime two hours old is removed by
    `GraphCache().clear()`; one with a current mtime is kept; neither is
    counted by `get_cache_stats()`.
17. Validation before lookup: warm the cache for a `with_ipv`-decorated
    callback, then `Graph(cb, ipv=[9], cache_key=...)` raises the "ipv is
    ignored" `ValueError`.
18. `cache_key=3`, `cache_key=""`, `Graph(3, cache_key="x")` raise.
19. `clear_from_cache()` removes the entry of a `cache_key` graph (built
    on a miss and on a hit) and reports 0 for a graph without one.
20. `PHASIC_DISABLE_GRAPH_CACHE=1`: subprocess `import phasic` emits the
    `UserWarning`.

Contract pin (documents the user's responsibility):
21. Same `cache_key`, a MODULE-LEVEL helper changed between builds: the
    cached graph is returned. Expected on master: also the cached graph.

Existing tests edited (asserted behaviour unchanged; only the way the
cache is requested and isolated changes):
- `test_graph_cache.py::TestCacheHitPreservesCallback` (2 tests):
  `graph_cache=True` -> `cache_key=...`, `graph_cache=False` -> removed,
  plus `delenv("PHASIC_CACHE_DIR")`.
- `test_serialize_constant_roundtrip.py`: `:106-113`, `:134` use
  `cache_key`; `load_graph(callback)` at `:111` -> `load(entry_key)`
  (from `g._graph_cache_entry`). T6 (`:219-240`) is rewritten to exercise
  the load-time format check: write a valid entry, rewrite its `format`
  field to `"1"`, assert the `UserWarning` and a rebuilt CONSTANT graph.
  Fixture `tmp_cache` gains `delenv`.
- `test_theta_layout.py::test_graph_cache_hit_gets_default_layout`,
  `test_from_serialized_find_vertex.py::test_joint_prob_graph_via_warm_graph_cache`
  (spy moves to `GraphCache.load`): `cache_key`, `delenv`.
- `test_weight_formula_theta_dim.py::test_graph_cache_preserves_theta_dim`:
  `configure(graph_cache=...)` removed, `cache_key` used, and the cache
  directory redirected with monkeypatch (today it wipes the real cache
  via `clear_caches()`).
- `test_config_scc_controls.py::test_configure_disable_graph_cache_sets_env`
  -> asserts `PTDConfigError`; its env-var list (`:49`) and those in
  `test_config_conflicts.py:29` and `test_configure_context.py:23` drop
  `PHASIC_DISABLE_GRAPH_CACHE`.
- `test_constant_parameterized_consistency.py:53,59,232` and
  `_discretize_pins.py:47,51,86`: drop `graph_cache=False` (those builds
  were uncached and stay uncached).

Grep gate before the implementation review: `graph_cache`, `GraphCache`,
`hash_callback`, `PHASIC_CALLBACK_VERSION`, `PHASIC_DISABLE_GRAPH_CACHE`,
`is_graph_cache_disabled`, `get_or_build`, `load_graph`, `save_graph`
over `src/`, `tests/`, `scripts/`, `experiments/` and `docs/api/`
(not notebooks). The only remaining hits allowed are the new API,
`per_thread_graph_cache` / `param_compute_graph_cache`,
`clear_from_cache(graph_cache=...)`, and module/function names of the
graph-cache helpers. Known now: `experiments/dr_discretize_cpp_sizes.py:26`
and `experiments/dr_d1_e0_scale.py:424-426` (drop the argument);
`docs/api/Graph.qmd`, `configure.qmd`, `get_config.qmd`,
`PhasicConfig.qmd` are generated (regenerated with `pixi run api` if that
runs cleanly in the worktree; otherwise listed for the user).

## 4. Gate

`pixi run install-dev` in the worktree, `gate2.sh /Users/kmt/phasic-bug5
bug5`, `gate_compare.py /Users/kmt/phasic-bug4/.gate_bug4
/Users/kmt/phasic-bug5/.gate_bug5 --new-file
tests/pytest/test_graph_cache_key.py`, three isolated reruns for any
regression. The new tests are run on the master build (bug 4 worktree)
and the outcome recorded. Suite wall time before and after is reported.
The entry-key cost at 1,003 and 5,823 vertices is measured against the
load time (b2 models), and the import-time stat recording on its own.
The untracked `fakehome*` and `review3` directories are never committed
(review evidence is committed only as scripts + logs, under 1 MB).

## 5. Notebooks (not edited — list for the user)

Scanned read-only from the main tree's current notebooks (2026-11-24;
confirmed independently by review 4, `review4/p2_nb_scan.log`).
Cells that raise after the change:
- `popgen/moran-coalescent-derived.ipynb` cell 1 and
  `tutorial/matrix-solve.ipynb` cell 2: `configure(graph_cache=True)` ->
  remove.
- `popgen/old-moran-coalescent-derived.ipynb` cell 4:
  `graph_cache=False` -> remove.
- `tutorial/python_trace-and-jax-caching.ipynb` cells 8, 10, 12 and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cells 8, 10, 12:
  `graph_cache=True` -> `cache_key="two-locus-arg"` (`dummy=42` vs
  `dummy=99` still give separate entries).
- `tutorial/state-space.ipynb` cells 36, 38, 45: `graph_cache=True` ->
  `cache_key=...`. Cell 45's `MyModel` works through `to_dict()`; its
  `__repr__` is no longer needed.
Text to revise (not raising):
- `tutorial/state-space.ipynb` cells 31, 39, 44.
- `tutorial/python_trace-and-jax-caching.ipynb` cells 1, 7 and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cells 1, 7 (the
  cache-correctness sentence must state the user-key contract).
- `tutorial/distributed.ipynb` cells 31, 37.
- Commented-out lines only: `popgen/ghost.ipynb` cell 3,
  `popgen/two-locus-island.ipynb` cells 11, 29, 37
  (`graph_cache.save_graph`), `tutorial/svgd-joint-prob.ipynb` cell 78.
Unaffected: `GraphCache()` and `.clear()` (`state-space` cell 33,
`ghost` cell 1, `two-locus-island` cell 1), `print_graph_cache_info` /
`get_graph_cache_stats`, `clear_from_cache(graph_cache=False)`
(`scc-decomposition` cells 26, 29).

## 6. Risks

- Construction is uncached unless a key is given (3.5 s instead of 1 s at
  5,823 vertices on repeat builds).
- A user who changes a module-level helper and keeps the key gets the old
  graph: the documented contract, pinned by test 21; no worse than today,
  where it happens with no key at all.
- `to_dict()` that omits a field the callback uses gives a stale hit for
  that field; the protocol docstring says it must describe everything the
  callback reads from the object.
- Callbacks that capture unfingerprintable objects in closures can no
  longer be cached (explicit error with ways out).
- A docstring edit misses (constant in the code object): extra build,
  never a stale graph.
- Callbacks capturing a `scipy.special` ufunc (or any compiled callable
  that does not resolve back to itself by name) in a closure cell are
  refused; such functions must be module globals to be cached.
- A shared, group-writable cache directory can be poisoned: anyone who
  can compute an entry key can plant a JSON graph there. With the
  restricted decoder (§2.6) it is data only, never an import or a call
  (unlike pickle caches, and unlike today's loader, which imports any
  module an entry names), but a peer could plant a wrong graph; this is existing behaviour, and the `cache_key` docstring says
  to share a cache directory only with trusted users.
- Review 5 findings -> v2.2: 1 builtin arm §2.2.1; 2 classes and
  `to_dict()` errors §2.2.2 rules 7-8; 3 current-path cycle check and
  indexed back-references; 4 `fchmod` and read-error classes §2.6;
  5 test 15 rewritten; 6 version portability §2.2.1; 7 stat check per
  key computation §2.2.3, test 13; 8 this risk. Tests 11, 13, 15
  extended.

## 7. Review 4 findings -> v2.1

1 closures/defaults/self -> §2.2.1, tests 8. 2 broad except -> §2.1,
test 15. 3 source vs running code -> code-object fingerprint §2.2.1, build
id stat check §2.2.3, tests 9, 13. 4 entry serialisation -> §2.6, test 12.
5 PTDConfigError -> §2.4, test 2. 6 validation order -> §2.1, test 17.
7 fingerprint holes -> §2.2 framing, §2.2.2 order/exact types/insertion
order/recursion, tests 5, 11. 8 atomic write -> §2.6, test 16. 9 test
design -> §3 preamble, `**kw`, subprocess isolation, delenv, T6 rewrite,
`clear_caches` redirect. 10 popped kwargs -> §2.2 item 6. 11
`_graph_cache_entry` -> §2.1, §2.7, test 19. 12 warning site -> §2.4,
test 20. 13 missing references -> §2.4, §3 edits and grep gate.
14 hygiene -> §4.
- Review 6 findings -> v2.3: 1 test 15 (second-call raise, no-warning
  assertion); 2 `os.open(O_EXCL, 0o666)` instead of `mkstemp`+`fchmod`
  §2.6; 3 exact read-error classes incl. `UnicodeDecodeError` and shape
  checks §2.6, test 15; 4 `from_serialized` failures propagate §2.1/§2.6;
  5 scipy.special refusal documented §2.2.1/§6; 6 extension located by
  `__file__` §2.2.3, `Exception` not `BaseException` rule 8, permission
  message §2.6; 7 test 11 `super()` case with a valid `to_dict()`.
- Review 7 findings -> v2.4: 1 restricted decoder for `graph_data`
  (no imports), §2.6, §6, test 15; 2 best-effort owner lookup, chained
  error, §2.6, test 15; 3 `.partial` cleanup in `clear()`, test 16;
  4 `GraphCacheLoadError` naming the file and ways out, test 15.
- Review 8 findings -> v2.5: 1 `GraphCacheLoadError` wraps `Exception`
  except `MemoryError`/`RecursionError`; 2 `.partial` cleanup only after
  one hour; 3 §2.1 names `GraphCacheLoadError`; 4 test 15 permission
  restore and root skip. Restricted decoder measured sufficient for all
  graph kinds (review 8 t1).
