# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v3.2, 2026-11-24. Branch `fix/graph-cache-stale`, worktree
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
- D3 (after four independent reviews each found a wrong-result case in
  the generic "key any object by content" machinery): NARROW what the key
  accepts automatically. No user-defined protocol (`to_dict()`) is
  trusted; only plain values, numpy arrays, containers of those,
  functions/closures (recursively) and phasic's own `StateIndexer` /
  `Property`. Anything else raises `GraphCacheKeyError` with the advice
  to pass plain values.
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
(helper functions, module variables, imported objects), installed
library versions (numpy, scipy, ...: compiled functions are keyed by
name, Python functions captured in a cell by their code, never by the
module globals they read), files and the environment is the user's
responsibility: change `cache_key` when it changes. The key also
reflects arguments and captured state AT CALL TIME, so a callback must
not change its arguments or captured state in ways that affect the graph
(for example mutating one kwarg and reading an alias of it: aliases are
keyed by content). On a hit the callback does NOT run, so any side
effect it has (filling a passed container such as a state-to-label map)
does not happen; and a callback whose output order depends on hash
randomisation (iterating a set of strings) gets the WRITER's vertex
order on a hit, an isomorphic graph with different vertex indices. The
`cache_key` docstring states all this with an example.

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
    every call, cached or not; same errors, raised earlier). They read
    `getattr(arg, "__name__", None)` instead of `arg.__name__`
    (`:1667`, `:1750`), so a `partial` callback gets the normal ipv
    error instead of today's `AttributeError` (review E 7d);
  - the ENTRY KEY (§2.2) is computed once. A key error propagates:
    nothing is built, nothing written;
  - lookup: `cache = GraphCache()` once per call, then
    `cache.load(entry_key)` returns a separate, fully
    built `Graph`, or `None` (missing file), or raises (§2.6). ONE policy
    for every entry that exists but cannot be used (malformed JSON or
    encoding, wrong shape or `format`, a non-ndarray type tag,
    `from_serialized` failing): `GraphCacheLoadError` naming the file and
    the ways out. Rationale: writes are atomic with best-effort
    durability (§2.6) and the format and the phasic build are
    inside the key, so an unusable entry can only come from disk or
    file-system corruption, tampering or a phasic bug; none of these should
    be hidden behind a silent rebuild, and the user decides (delete the
    file, clear the cache, or change `cache_key`). There is no
    warn-and-rebuild path. Nothing inside `Graph.__init__` catches
    anything from `load()`; the broad `except Exception` is removed.
    Nothing after `super().__init__(cached_graph)` is inside a `try`: an
    error while restoring attributes on a hit propagates;
  - hit: the current restore sequence (`:1689-1723`) runs; then
    `self.dyn_ordering = os.environ.get("PHASIC_DYN_ORDERING") is not
    None`, mirroring the C rule applied at graph creation
    (`phasic.c:2996`), because `serialize()` persists the WRITER's flag
    (`_graph_serialize.py:222`, restored at `:698`) and a hit must behave
    like a fresh build in the reader's environment (review C p9); plus
    `self._graph_cache_entry = <entry path>` (the ABSOLUTE file path,
    `Path.resolve()`, so a relative `PHASIC_CACHE_DIR` plus a later
    `chdir` cannot break it, and so
    `clear_from_cache` finds it even if `PHASIC_CACHE_DIR` changes later);
  - miss: build as today, then `cache.save(self, entry_key, meta)` on the
    SAME `GraphCache` instance `cache` used for the lookup (one per call,
    §2.6);
    only after it returns, `self._graph_cache_entry = <entry path>`. ANY
    `Exception` from `save()` (an `OSError` from the write, or for example
    `serialize()` failing or a `MemoryError` while encoding a 600k-vertex
    graph) propagates with its type unchanged (an `OSError` keeps its
    subclass and gains a message naming the path), with the built graph
    attached as `e.graph` and the message saying so, so an hours-long
    build is not lost (still an error, not a silent fallback);
    `self._graph_cache_entry` stays `None` on that object. When the
    failure comes from `serialize()` itself (for example a callback
    returning edges with mixed coefficient lengths builds but cannot be
    serialised, review D d2), the note says this graph cannot be cached
    and to drop `cache_key`.
- `cache_key` is never passed to the callback.

### 2.2 The entry key (`callback_hash.py`, rewritten)

`cache_entry_key(cache_key, callback, /, **params) -> str` (the first two
parameters positional-only, so a user kwarg named `callback` or
`cache_key` cannot collide with them): the first 32 hex
characters of SHA-256 over a TOKEN STREAM. Every token is written as
`<tag>:<len>:<payload>` with the payload's byte length, so no two token
sequences can produce the same bytes. That alone does not make the
VALUE -> token-sequence map one-to-one (a literal implementation gives
`[[1],2]` and `[[1,2]]` the same stream, review B p2), so EVERY composite
token carries its element count before its elements: list, tuple, dict
(entry count), frozenset constant, `co_consts`, `co_names` and the other
tuple fields of a code object, `__defaults__`, `__kwdefaults__`, the
closure (cell count), partial `args` / `keywords`, the kwargs block and
an array's shape (dimension count). Together with the tags this makes
the stream prefix-free, so distinct structures always give distinct
streams. Components in order:
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
  FINGERPRINT of `f.__defaults__`, `f.__kwdefaults__` and `f.__dict__`
  (function attributes, e.g. `cb.rate = r` read back through the
  function's own cell; phasic's wrapper attributes `__wrapped__` and
  `__ipv__` are fingerprintable); for each name in
  `co_freevars`, the cell contents (empty cell -> token `empty`) by VALUE
  FINGERPRINT, where a cell holding a function is fingerprinted as a
  function (so decorator wrappers and captured helpers are covered).
- bound method: FUNCTION FINGERPRINT of `__func__` + VALUE FINGERPRINT of
  `__self__`.
- `functools.partial`, EXACT type only (`type(x) is functools.partial`;
  a subclass may override `__call__` and falls through to the later
  rules or is refused): function, `args`, `keywords`.
- builtin function, numpy ufunc / array-function dispatcher, gated on
  EXACT type first: `type(x) is types.BuiltinFunctionType`,
  `type(x) is np.ufunc`, or `type(x) is type(np.sum)`
  (`numpy._ArrayFunctionDispatcher`); any other object (for example a
  module-level `functools.lru_cache` function, or an object made with
  `functools.update_wrapper`, both of which satisfy the conditions below
  and would otherwise be keyed by NAME, review F p6) is NOT
  FINGERPRINTABLE. For these exact types: module +
  qualified name token, ONLY IF `__self__` is `None` or a module AND
  resolving `__module__` + `__qualname__` (or `__name__` for ufuncs)
  through `sys.modules` returns the very same object. This admits
  `math.sqrt`, `len`, `np.exp`, `np.linalg.solve` and refuses builtin
  bound methods of instances (`{}.get`, `Random(1).random`, which carry
  state in `__self__`) and `np.frompyfunc` ufuncs ("<lambda>
  (vectorized)"), which would otherwise share a name token (review 5 q1
  D). It also refuses every `scipy.special` ufunc (`gamma`, `binom`:
  they have NO `__module__` / `__qualname__` attribute, review C p8, so
  every attribute in these rules is read with `getattr(obj, name, None)`
  and a missing one means refusal with `GraphCacheKeyError`, never an
  `AttributeError`); the error advises moving
  such a function to a module global (then covered by the user's
  `cache_key`, as for any imported object). Note the asymmetry, which is
  consistent with the contract: compiled dispatchers are keyed by name,
  while Python-level library functions reached through a cell (for
  example `scipy.special.comb`) are keyed by their code.
- an immutable compiled type (for example
  `itertools.combinations_with_replacement`, the usual
  `all_pairs = partial(combinations_with_replacement, r=2)` idiom found in
  8 test files and the notebooks): module + qualified name token, ONLY IF
  `__flags__ & (1 << 8)` (`Py_TPFLAGS_IMMUTABLETYPE`, Python 3.10+) is
  set AND `__module__` + `__qualname__` resolve through `sys.modules` to
  the very same object. User-defined classes are mutable heap types and
  stay refused (so the `__class__` cell of a `super()` call is still
  refused).
- anything else callable (other class, callable instance): NOT
  FINGERPRINTABLE.
- a MODULE object (not callable; for example the cell a local
  `import numpy as np` inside a factory creates): module-name token,
  ONLY IF `sys.modules[m.__name__] is m`. It behaves exactly like a
  module global, whose contents are the user's responsibility under the
  contract. (Value rule 6b below.)
- Recursion: the visited map maps each function to its first-visit
  index; a revisit contributes a back-reference token carrying that
  index, so different recursion shapes give different streams. The map
  is keyed by `id()` but HOLDS A STRONG REFERENCE to every visited
  function until the key computation finishes (since v3.0 the only
  temporaries walked are the dicts `StateIndexer.to_dict()` returns,
  which hold no functions; the strong references stay as a guard), so a
  temporary freed mid-walk can
  never have its id reused by a different function (which would turn it
  into a back-reference and give two different inputs one stream,
  review D d7).

CODE FINGERPRINT of a code object: `co_code`, `co_exceptiontable`,
`co_names`, `co_varnames`, `co_freevars`, `co_cellvars`, `co_argcount`,
`co_posonlyargcount`, `co_kwonlyargcount`, `co_flags`, `co_name`, and
`co_consts` element by element (nested code objects by CODE FINGERPRINT;
`frozenset` constants by their elements' fingerprints, sorted; `tuple`
constants element by element under these same CONSTANT rules (so
`state[..., 0]`, whose constant is `(Ellipsis, 0)`, and on 3.14
`M[1:3, 0]` are covered, review C p4); `Ellipsis` token; `slice`
constants by start/stop/step; other constants by VALUE FINGERPRINT
(which also has the `Ellipsis` and `slice` rules, so a default
`idx=(..., 0)` is covered); a constant type none of these
covers raises `GraphCacheKeyError`, never a crash). `co_exceptiontable`
is read with `getattr(co, "co_exceptiontable", b"")` (absent before
3.11; harmless defensiveness: `pyproject.toml:29` says
`requires-python >= 3.10` but phasic already needs 3.11 to import
(`from typing import Self`, `__init__.py:11`); pixi pins < 3.14; only
3.13 is tested here). Excluded: `co_filename`,
`co_firstlineno`, `co_linetable`, `co_qualname`. Consequences (all
tested): this is the code that RUNS, so editing the file on disk without
reloading changes nothing (review4 p1 D); comment lines, blank lines,
indentation and spacing within a line hit, but splitting an expression
or call across lines can change `co_code` on 3.13 (line boundaries stop
instruction fusion such as `LOAD_FAST_LOAD_FAST`, review C p2) and then
misses (harmless); a docstring edit misses (docstrings are constants; a harmless miss);
two lambdas on one line get distinct keys; `inspect.getsource` is no
longer used, so callbacks without accessible source can be cached.

#### 2.2.2 VALUE FINGERPRINT (exact types checked first, in this order)

1. EXACT types only (`type(x) is ...`): `None`, `bool`, `str`, `bytes`:
   type name + canonical repr; `int` via `hex()` (no 4300-digit `repr`
   limit); `float` via `float.hex`; `complex` via the hex of both parts.
   Subclasses do not match here: `np.float64` (a `float` subclass) goes
   to rule 2, and `IntEnum` members and other subclasses are refused;
2. numpy scalar (`np.generic` subclass): `x.dtype.str` + `tobytes()`;
   `np.void` scalars and any dtype with `.fields` or `.hasobject` are
   refused;
3. `type(x) is np.ndarray` exactly (masked arrays, `np.matrix` and other
   subclasses are refused): `x.dtype.str`, shape, C-order bytes. Refused:
   `dtype.hasobject` (covers object dtype AND structured dtypes with an
   object field, whose bytes are pointers) and `dtype.fields is not
   None` (structured dtypes: different field layouts share
   `dtype.str == '|V16'` and bytes, review A C);
4. exact `list` / `tuple`: type + element fingerprints in order;
5. exact `dict`: entry count, then entries in INSERTION order (a
   callback iterating a dict sees that order, so it can change the
   graph); each key by VALUE FINGERPRINT (any fingerprintable key: tuple
   state keys, int keys), then its value. The key
   reflects captured state AT CALL TIME: a closure-captured memo dict
   that the build fills is keyed as it was before that build (stated in
   the docstring);
5b. `Ellipsis` (token) and exact `slice` (start, stop, step by these
   rules);
6. any callable that §2.2.1 admits (Python function, bound method,
   partial, and builtins/ufuncs under exactly §2.2.1's conditions):
   FUNCTION FINGERPRINT; §2.2.1 and this rule are one rule;
   6b. a module object under §2.2.1's module rule: module-name token;
7. a class object (`isinstance(x, type)`): the immutable-compiled-type
   token of §2.2.1 if it qualifies; otherwise NOT FINGERPRINTABLE (so a
   `__class__` cell from `super()` or a user class passed as an argument
   is refused explicitly);
8. phasic's own value types, EXACT class only, and only in their
   plain, unextended form:
   - `Property`: `type(p) is Property`; keyed by ALL four dataclass
     fields (`name`, `min_value`, `max_value`, `offset`), since `offset`
     takes part in `__eq__` and is readable by a callback
     (`state_indexing.py:97`).
   - `StateIndexer`: admitted only if ALL of:
     (a) `type(ix) is StateIndexer`, every PropertySet has
     `type(ps) is PropertySet` or `type(ps) is StateSpace`
     (`state_indexing.py:220`, `:2204`), and every Property passes the
     Property rule;
     (b) attribute NAMES within allow-lists: for the indexer and for each
     PropertySet, the attributes of a freshly built object PLUS the
     documented lazily added ones (`_props_dataclass`, added on the first
     `index_to_props`, `state_indexing.py:427-434`; any other lazy
     attribute found at implementation time is listed the same way). The
     pin test compares the allow-lists with `set(vars(...))` of an object
     that has been USED (after `index_to_props`), so it neither misses a
     lazy attribute nor refuses an indexer a build has already used
     (review F 4);
     (c) BEHAVIOURAL STATE equal to a canonical rebuild: let
     `c = StateIndexer.from_dict(ix.to_dict())`; then `ix._offsets`,
     `ix._total_state_length`, the order lists, and for each PropertySet
     its `properties` list, `list(property_dict.items())`, `_bases`,
     `_radix_powers` and (if present) the field names of
     `_props_dataclass` / `_result_class` must equal those of `c`. This
     catches every route by which `to_dict()` and behaviour drift apart:
     a property list edited after construction (`PropertySet` keeps the
     caller's list uncopied, `:280`, while `property_dict`, `_bases`,
     `_radix_powers`, `_offsets` and `_total_state_length` are computed
     once, `:281`, `:288-289`, `:906-924`; measured: equal `to_dict()`,
     5 vs 7 vertices, review F p3) or allow-listed values overwritten.
     Keyed as: class token, PropertySet class names in order, then the
     VALUE FINGERPRINT of `to_dict()` with each Property's `offset`
     added (which, by (c), now determines the behaviour). Anything that
     fails (a)-(c) is NOT FINGERPRINTABLE, with an error naming the
     failing check (an extra attribute, a subclass, or "indexer state
     differs from its to_dict(); was its property list changed after
     construction?"). Exceptions from `to_dict()` / `from_dict()` are
     re-raised as `GraphCacheKeyError` (path chained). Cost: one
     `from_dict` per key computation, negligible next to a build.
   NO other object is keyed through `to_dict()` or any other
   user-defined method: a user's `to_dict()` cannot be checked to be
   one-to-one (pandas' is not: it collapses duplicate index labels,
   review D d3);
9. anything else: NOT FINGERPRINTABLE. This includes user class
   instances (with or without `to_dict()`), pandas objects, dataclass
   instances (phasic's `HexCell` / `HexGrid` too) and bound methods of
   such instances. The error advises passing the values the callback
   needs as plain values (numbers, strings, lists, dicts, numpy arrays),
   for example `model.to_dict()` or `df.to_numpy()` plus labels as lists,
   or building without `cache_key`.
Recursion is bounded in EVERY recursive arm (values, functions, code
objects and their constants, `to_dict()` results), and `cache_entry_key`
additionally catches `RecursionError` at its top level and re-raises it
as `GraphCacheKeyError`. Limits: a depth limit of 100, and a cycle check against the
ids of the containers / `StateIndexer` objects on the CURRENT PATH from the
root only (pushed on entry, popped on exit). A process-wide seen-set is
wrong: it flags a list referenced twice, an ipv reusing one state list,
and freed temporaries whose ids are reused (review 5 q2). Both limits
raise `GraphCacheKeyError` (never `RecursionError`).

#### 2.2.3 Build id

Neither `graph_cache` nor the rewritten `callback_hash` may run anything
at module level that loads `phasic_pybind`. On master this already
happens: `graph_cache.py:29` calls `get_logger(__name__)` at module
level, whose first use runs `setup_logging()`
(`logging_config.py:252-253`), which imports `phasic_pybind`
(`:200`), before `OMP_NUM_THREADS` is set at `__init__.py:107` (review C
p6; a pre-existing defect fixed here). Both modules therefore use a lazy
logger: `logging.getLogger("phasic.graph_cache")` /
`("phasic.callback_hash")` at module level (stdlib only), with phasic's
handler setup left to the existing package-level call. (In this env
libomp is already loaded by numpy/pandas, so the guarded property is the
order of the `phasic_pybind` import, not libomp's load time.)
At `import phasic`, AFTER `from .phasic_pybind import *`
(`__init__.py:279`), a call `graph_cache._record_build_stats()` at the end
of `phasic/__init__.py` (which NEVER raises at import: any failure, for
example a file vanishing between listing and `stat`, is stored and
raised as `GraphCacheKeyError` by the first key computation, so
`import phasic` cannot break for users who never cache; a missing
record, for example after `importlib.reload(phasic.graph_cache)`, also
raises `GraphCacheKeyError` rather than skipping the check) records
`(path, size, mtime_ns)` for
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
Recording in `graph_cache`'s module body would be wrong: that module is
imported at `__init__.py:53`, before `OMP_NUM_THREADS` is set at
`:107-108`, and touching `phasic_pybind` there would load the native
library and libomp too early (review A F). Only the content hash (measured: 52 .py + the .so, 3.49 MB, 11.9 ms;
review 4) is memoised, computed on the first key computation. `.cpp` /
`.h` files shipped in the package directory are not part of the id
(graph construction goes through the compiled extension only).

### 2.3 Errors

- `GraphCacheKeyError(ValueError)`, exported from `phasic`: the key
  cannot be computed. The message names the path to the offending value
  (for example `kwargs['model']` or
  `callback -> closure 'helper' -> defaults[0]`), its type, and the ways
  out: pass plain values (for example `model.to_dict()` instead of
  `model`), move the value
  to a module global and change `cache_key` yourself, or build without
  `cache_key`.
- `clear_all_graph_caches` is added to the `phasic` namespace
  (`__init__.py:53` imports only `GraphCache`, `get_graph_cache_stats`,
  `print_graph_cache_info` today, so the advice below and
  `_graph_cache_mgmt.py:60` name a function users cannot reach, review C).
- `GraphCacheLoadError(RuntimeError)`, exported: an entry exists but
  cannot be used (§2.1, §2.6); names the file and the ways out (delete
  the file, `phasic.clear_all_graph_caches()`, or another `cache_key`),
  original exception chained. (`GraphCacheEntryError` of v2.5 is gone.)

### 2.4 Global configuration removed

- `config.py`: field `graph_cache` (`:611`) and its documentation
  (`:549-555`); the env-table row (`:465`) with its comment (`:463`),
  and the convention comment (`:444`, the
  convention text that cites it); the from-env read (`:325-326`); the
  env-var lists (`:201`, `:930`); the report entry (`:914`); the
  docstring mentions at `:239`, `:308`; the cache-layout comment at
  `:607-609` is REWORDED (it still holds, minus the "graph_cache" field
  name), not deleted; the inverted-bool
  example comment at `:1040-1041` (reworded to a remaining field).
- `phasic.configure(graph_cache=...)`: the existing unknown-option path
  raises `PTDConfigError` (`exceptions.py:14`; NOT a `ValueError`), with
  an added line saying the option was removed and pointing to
  `Graph(..., cache_key=...)`.
- `PHASIC_DISABLE_GRAPH_CACHE` set in the environment: one `UserWarning`
  at `import phasic`, next to the import-time `get_config()`
  (`__init__.py:133`), saying it no longer has an effect.
- `graph_cache.is_graph_cache_disabled` is deleted.
- `PHASIC_DISABLE_CACHE` is read NOWHERE in `src/` (only set by
  `profile.py:158-173`, mentioned in comments at `api/c/phasic.h:1021,
  1034, 1326`; `trace_serialization.is_cache_disabled` reads
  `PHASIC_REWARD_COMPUTE_CACHE`, `:85-102`). It therefore does not affect
  `cache_key`, which always caches; the docstring says so and the test
  only guards against a future reader.
- While the `Graph.__init__` docstring is edited: its examples at
  `:1579-1581` use a nonexistent `cache=True`; they are corrected.
- `cache.py:26-28` docstring ("on by default") reworded.

### 2.5 Cache location

`GraphCache()` with no argument resolves its directory when constructed:
`Path(env) / "graphs"` when `PHASIC_CACHE_DIR` is set AND
non-empty, implemented INLINE in `graph_cache` (3 lines, with a comment)
and NOT by importing `phasic.cache`: that module imports
`trace_serialization` at module level, which imports `phasic_pybind`
(`cache.py:48-50`, `trace_serialization.py:47-48`), and `graph_cache` is
imported at `__init__.py:53`, before `OMP_NUM_THREADS` is set (review B
p1). Same rule as `cache.py:73-74` and the C side,
`phasic.c:3891`, which treat an empty value as unset. (The variable is
written by `configure(cache_dir=...)` via `validate()`,
`config.py:819-845`, and read by the C caches.) Otherwise the module
attribute `DEFAULT_CACHE_DIR`, read at call time. Tests isolate with
`monkeypatch.setenv("PHASIC_CACHE_DIR", <tmp>)`, which also redirects
the C-side caches that `clear_from_cache` touches (§3). `get_cache_dir`, `get_graph_cache_stats`,
`print_graph_cache_info`, `clear_all_graph_caches` and
`model_export.clear_caches` use the same resolution.

### 2.6 `GraphCache` API and entry file

- Kept: `GraphCache(cache_dir=None)`, `clear()`, `clear_graph_cache()`,
  `get_cache_stats()`. EDITED: `_serialize_value`
  (`graph_cache.py:304-378`) loses its generic `to_dict()` arm (which tags
  any object with `__type__: module.Class`) and its docstring/error text
  advertising the `to_dict()`/`from_dict()` protocol; it encodes plain
  JSON values plus the `ndarray` tag only and raises on anything else, so
  the save side is symmetric with the restricted decoder and no entry
  can be written that every later lookup would refuse. Deleted:
  `_deserialize_value` (see below). `Graph.__init__` constructs ONE `GraphCache()` per
  call, so lookup and save resolve the same directory.
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
  `callback_name` (informational: the user function's name through
  phasic's wrapper, `getattr(getattr(cb, "__wrapped__", cb), "__name__",
  None)`, falling back to the type name, so a wrapped `partial` does not
  crash at save time, review C p10), `graph_data`. `construction_params` is DROPPED: the
  key already encodes the parameters, and serialising them could fail
  after the build for values the key accepts (np.bool_, complex, bytes:
  review4 p1 E).
- Write: `os.open(<cache dir>/f".tmp-{secrets.token_hex(8)}.partial",
  O_CREAT | O_EXCL | O_WRONLY, 0o666)` (unique per writer, never matching
  `*.json`; the kernel applies the process umask and any default ACL of
  the directory, as for today's `open()`, so no umask is read or changed
  (reading it with `os.umask` would briefly change it process-wide while
  JAX/OpenMP threads run), and no `mkstemp` 0600 mode), write,
  `os.fsync(fd)`, close, `os.replace` to `<entry_key>.json`. The
  temporary file is removed in a `finally` that covers `BaseException`
  too (a Ctrl-C during a minute-long 600k-vertex save must not leave a
  ~0.85 GB `.partial` behind for an hour). Durability is best effort: no
  directory `fsync`, and macOS `fsync` is not `F_FULLFSYNC`; the worst
  outcome of a crash is a miss or a `GraphCacheLoadError`, never a wrong
  graph. A save error is amended IN PLACE: the same exception object is
  re-raised with `e.graph` set and `e.add_note(...)` naming the path and
  `e.graph` (phasic already requires Python 3.11: `from typing import
  Self`, `__init__.py:11`). Two concurrent
  writers each replace atomically; a reader sees a complete file or none
  (shared SLURM directories are a documented use,
  `distributed.ipynb` cell 31).
- Load errors are classed exactly as:
  - `FileNotFoundError` on open -> `None` (miss);
  - any other `OSError` on open/read (for example `PermissionError` on an
    entry another user wrote under umask 077, common on HPC): re-raised
    as the same `OSError` subclass, original chained, with a message
    naming the file, its owner (best effort: `os.stat` or `pwd.getpwuid`
    failing, for example in a non-searchable directory, gives "owner
    unknown" and never raises from the handler) and the ways out
    (another `cache_key`, or another directory via
    `configure(cache_dir=...)`);
  - `GraphCacheLoadError` (original chained) for: `json.load` raising
    `ValueError` (covers `JSONDecodeError` and `UnicodeDecodeError`);
    explicit shape checks failing (top level not a `dict`, a required
    field missing or of the wrong type, `format` != `GRAPH_CACHE_FORMAT`;
    explicit checks, so no raw `KeyError`/`TypeError`); the RESTRICTED
    decoder meeting any `__type__` tag other than `ndarray` (plain JSON
    plus that tag is all `serialize()` output needs, measured for every
    graph kind in review 8 and review A D; nothing is imported); and any
    `Exception` from `from_serialized` other than `MemoryError` and
    `RecursionError`, which say nothing about the file and propagate
    unchanged, as do `BaseException`s such as `KeyboardInterrupt`.
    The restricted decoder wraps EVERY `Exception` it meets (unknown tag,
    `__type__` without `__data__`, ragged `__data__` making numpy raise),
    except `MemoryError` / `RecursionError`, in `GraphCacheLoadError`
    (a deeply nested tampered file can make `json.load` raise
    `RecursionError`, which therefore propagates raw, by the same
    exception rule). Chaining uses `raise ... from e`.
  The general `_deserialize_value` (which imports the module a tag names
  and calls its `from_dict`, review 7 A) is DELETED (no caller remains;
  `test_serialize_is_aux.py` moves to the restricted decoder), and
  `_serialize_value` stays for the save path, edited as described above.

### 2.7 `Graph.clear_from_cache(graph_cache=True, ...)`

(`_graph_cache_mgmt.py:70-91`; this parameter selects which cache to
clear and is unrelated to the removed option, so it stays.) Uses
`getattr(self, "_graph_cache_entry", None)`, which holds the entry's
file PATH resolved at construction (not just the key). No entry (built without
`cache_key`) -> `removed["graph_cache"] = 0`, documented. Graphs
DERIVED from a cached graph (`discretize`, widening, `laplace_transform`,
`add_epoch`: `__init__.py:3445-3447, 10136-10138, 3003-3005, 3276`) do
not inherit `_graph_cache_entry`, so `clear_from_cache()` on them reports
0. On master the same call re-hashed the copied `_callback` and deleted
the BASE graph's entry; the new behaviour is intended (a derived graph
has no entry of its own) and documented. A graph not built from a
callback still raises as today.

### 2.8 Old entries

Entries written before this change are never loaded (different format
and key composition). They stay on disk; `phasic.clear_all_graph_caches()`
removes them (docstring says so).

## 3. Tests

New file `tests/pytest/test_graph_cache_key.py`. An autouse fixture sets
`PHASIC_CACHE_DIR` to a directory under `tmp_path` (this redirects the
graph cache AND the C-side caches, so `clear_from_cache()`, whose default
also clears `parameterized_reward_compute`, never reaches the real HOME),
and additionally monkeypatches `graph_cache.DEFAULT_CACHE_DIR` to another
`tmp_path` directory as a backstop. Test 14 checks both resolutions
explicitly. Every subprocess gets `HOME` and `PHASIC_CACHE_DIR` pointing
into `tmp_path`, so the gate (which runs with the real HOME) never
touches the real cache. All subprocesses share one session-scoped
`MPLCONFIGDIR` (from `tmp_path_factory`), so a fresh HOME does not
rebuild matplotlib's font cache in each (15 s per import against 1.9 s,
review B p4) and its stderr notice cannot interfere with output checks.
The new names (`GraphCacheKeyError`, `GraphCacheLoadError`,
`cache_entry_key`, ...) are imported INSIDE the tests, so on master the
file collects and each test records its own outcome (test 21 can show
its master pass). No test calls `phasic.clear_caches()` (it wipes
`$HOME/.phasic_cache` and `$HOME/.jax_cache` regardless of
`PHASIC_CACHE_DIR`, `model_export.py:49,72`). A hit is
proven by a spy on the callback (not called); "stale" is judged on
`expectation()` against a build without `cache_key`. Callbacks used with
`cache_key` accept `**kw`, so that on master the argument reaches them
harmlessly and the outcome recorded on master reflects the real
behaviour (on master `cache_key` is just ignored, and the default cache
is on).

Expected to fail on master, EXCEPT these REGRESSION GUARDS that also
pass on master and are labelled as guards in the file: tests 3, 4 and 6
(master already puts `cache_key` into `repr(kwargs)` and the ipv into
its key, `callback_hash.py:95-103,142-160`), test 24 (master's
`repr(kwargs)` is exact for small dicts with tuple or int keys), the
`state[..., 0]` part of test 25 (master hashes the AST), and the
`PHASIC_DISABLE_CACHE` half of test 20 (master ignores `cache_key` but
caches by default), test 10 (master's `StateIndexer.__repr__` is exact
for ranges and layout, `state_indexing.py:1570-1593`), and the
`Property` and "not cycles" parts of test 11:
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
   middle element; dict with the same items in another insertion order;
   nested structures with equal flattenings (`[[1],2]` vs `[[1,2]]`,
   `{'a':{'b':1},'c':2}` vs `{'a':{'b':1,'c':2}}`, `[[[3,0],1]]` vs
   `[[[3,0]],1]`): miss, and correct.
6. ipv changed (`ipv=` and `with_ipv` forms): miss.
7. Callback body changed (new function object from source with one
   constant changed): miss. Comment-line, blank-line and in-line spacing
   edits: hit (no line splits, see §2.2.1).
   Two lambdas defined on one line: distinct keys.
8. Closure: a factory `make(rate)` called with 1.0 and 5.0, same
   `cache_key`: miss, both correct (review4 p1 C). Default-bound loop
   idiom `def cb(state, rate=n)`: miss when `n` changes. Closure holding
   a user-class instance: `GraphCacheKeyError` naming
   `closure '<name>'`. The factory idiom with a local
   `all_pairs = partial(combinations_with_replacement, r=2)` captured in
   the closure: hits on rebuild (immutable compiled type admitted).
9. Source file edited on disk, module not reloaded: same key (the
   running code is keyed).
10. `indexer=StateIndexer(...)` built afresh with the same layout: hit;
    a different property range: miss.
11. Refused with `GraphCacheKeyError` (not `RecursionError` /
    `TypeError` / `AttributeError`), callback not called, no file
    written: a user-class instance WITH a valid `to_dict()` (the docs'
    `MyModel`) as a kwarg and in a closure cell; a pandas DataFrame and a
    Series; a dataclass instance; a user class passed as a kwarg (error
    path names the class); a bound method of a user instance as a cell;
    a `set`; a masked array; a list containing itself; a closure cell
    holding `{}.get` or an `np.frompyfunc` ufunc; a structured-dtype
    array; a structured array with an object field; an `np.void` scalar.
    Not cycles (hash and hit): a list referenced twice in one kwarg, an
    ipv reusing one state list, a list of 200 `StateIndexer` objects.
    `Property` kwargs: hit when equal, miss when `max_value` differs;
    `Property(min_value=1, offset=0)` vs `Property(min_value=1,
    offset=3)`: different keys. Refused: a `StateIndexer` with an extra
    attribute (`ix.rate = 2`), one whose PropertySet carries an extra
    attribute, one built from a PropertySet subclass other than
    `StateSpace`, one holding a `Property` subclass overriding
    `encode_value`, a `Property` subclass as a kwarg, an indexer whose
    caller-owned property list was edited after construction (element
    replaced; element appended), an indexer with `_bases` overwritten,
    a module-level `functools.lru_cache` function and an
    `update_wrapper` object as kwargs/cells, an `IntEnum` member. An
    indexer already used by a build (`index_to_props` called): hits.
    `np.float64(1.0)` vs `1.0` inside a list: different keys. A
    `StateSpace`-based indexer: hits. The allow-lists equal
    `set(vars(...))` of freshly built objects (pins them against future
    attribute additions in `state_indexing.py`).
12. Values the key accepts that the old entry format could not store
    (`np.bool_`, complex, bytes keyword arguments) and an int above 4300
    digits held INSIDE a list kwarg and as a closure default (a top-level
    int kwarg is converted to float before the callback, `__init__.py:
    1743-1745`, which overflows with or without a cache): build, save and
    hit work. A module captured in a closure cell (local
    `import numpy as np` in a factory): hits.
13. `graph_cache._phasic_build_id()` (the memoised content hash of
    §2.2.3, named here) monkeypatched to another value: miss. A recorded
    stat changed after import, simulated by monkeypatching the RECORDED
    stat table (never by touching installed files), both before the first
    key computation (the content-hash memo is monkeypatched back to
    `None` first, since an earlier test may have filled it) and after
    one: `GraphCacheKeyError` mentioning restart. A failure stored by
    `_record_build_stats` (monkeypatched) and a missing record: both
    `GraphCacheKeyError` at the first key computation, while
    `import phasic` itself succeeded.
14. `PHASIC_CACHE_DIR=<tmp>`: entry in `<tmp>/graphs`. With it unset,
    and with it set to `""` (HOME also monkeypatched to `tmp_path` for
    these cases, so nothing can reach the real HOME): entry under the
    monkeypatched `DEFAULT_CACHE_DIR` (never a relative `graphs/`). A
    relative `PHASIC_CACHE_DIR`: `_graph_cache_entry` is absolute. Import order: a
    subprocess whose env has `OMP_NUM_THREADS` REMOVED (the pytest process
    has it set by phasic, `__init__.py:107-108`, and children inherit it)
    runs `import phasic` with a `sys.meta_path` hook that records, when
    `phasic.phasic_pybind` is first imported by anyone, whether
    `OMP_NUM_THREADS` is already set; the test asserts it is. The check is
    shown to work by running the same hook once on a deliberate violation
    (a stubbed package importing `phasic.cache` first, as in review B p1),
    where it must report the violation.
15. Unusable entries raise `GraphCacheLoadError` naming the file, with
    the original chained, the callback NOT called and the file left in
    place: truncated JSON; invalid UTF-8 bytes; a valid JSON list; a dict
    missing `graph_data`; wrong `format`; a `graph_data` carrying
    `{"__type__": "<module>.<Class>"}` for a throwaway module written to
    `tmp_path` on `sys.path` (that module is NOT in `sys.modules`
    afterwards); a well-formed entry whose `from_serialized` raises
    (monkeypatched). A `MemoryError` from a monkeypatched
    `from_serialized` propagates unchanged. A hit whose attribute restore
    raises: a FORMULA-mode entry is written by hand under the computed key
    (a graph with `weight_formula` set, serialised and passed to
    `GraphCache.save`, since `Graph.__init__` only ever saves linear-mode
    graphs and the tape restore at `__init__.py:1716` runs in formula
    mode only); `_set_weight_tape` passes through on its FIRST call (the
    one inside `from_serialized`, `_graph_serialize.py:733-741`) and
    raises on its SECOND (the restore on the live graph); the test asserts
    the second call happened and that exactly that error propagates. An
    unreadable entry file (`chmod 000`, skipped as root) propagates
    `PermissionError` whose message names the file; with the directory
    made non-searchable instead (permissions restored in a `finally`,
    skipped as root), the message says "owner unknown" and the original
    is chained. A ragged `__data__` and a `__type__` without `__data__`:
    `GraphCacheLoadError`. The advice's way out works: after
    `phasic.clear_all_graph_caches()` the same call rebuilds and saves.
16. Write failure (`os.replace` monkeypatched to raise
    `PermissionError`): the same `PermissionError` subclass, the path in
    the message, `e.graph` equal to a fresh build (same `serialize()`),
    `e.graph._graph_cache_entry is None`, and no `.partial` file left in
    the directory. `serialize` monkeypatched to raise `RuntimeError`
    during save: `RuntimeError` with `e.graph` set. A stray
    `.tmp-x.partial` file with an mtime two hours old is removed by
    `GraphCache().clear()`; one with a current mtime is kept; neither is
    counted by `get_cache_stats()`.
17. Validation before lookup: warm the cache for a `with_ipv`-decorated
    callback, then `Graph(cb, ipv=[9], cache_key=...)` raises the "ipv is
    ignored" `ValueError`.
18. `cache_key=3`, `cache_key=""`, `Graph(3, cache_key="x")` raise.
19. `clear_from_cache()` removes the entry of a `cache_key` graph (built
    on a miss and on a hit) and reports 0 for a graph without one (its
    `parameterized_reward_compute` half runs against the redirected
    `PHASIC_CACHE_DIR`); after changing `PHASIC_CACHE_DIR` between build
    and clear, the entry is still found and removed (path stored).
20. `PHASIC_DISABLE_GRAPH_CACHE=1`: subprocess `import phasic` emits the
    `UserWarning`. `PHASIC_DISABLE_CACHE=1` with `cache_key`: the entry
    is still written and hit.

22. `PHASIC_DYN_ORDERING`: an entry written in a subprocess with the
    variable set and read in a subprocess without it: the hit has
    `dyn_ordering` False, like a fresh build there (and vice versa).
23. Function attributes: a factory setting `cb.rate = r` that the
    callback reads through its own cell: miss when `r` changes.
24. A dict kwarg with tuple keys and one with int keys: hit when equal,
    miss when one entry differs.
25. `state[..., 0]` in the callback and a default `idx=(..., 0)`: key
    computed, hit on rebuild. `scipy.special.gamma` captured in a cell:
    `GraphCacheKeyError` (not `AttributeError`). `with_ipv([5])(partial(model,
    k=2.0))`: saves (callback_name fallback) and hits.

26. A `partial` subclass overriding `__call__`: refused. A user kwarg
    named `callback`: keyed normally (no `TypeError`).
    A callback with mixed coefficient lengths under `cache_key`: the save
    error carries `e.graph` and the "cannot be cached" note.

Contract pin (documents the user's responsibility):
21. Same `cache_key`, a MODULE-LEVEL helper changed between builds: the
    cached graph is returned. Expected on master: also the cached graph.

Existing tests edited (asserted behaviour unchanged; only the way the
cache is requested and isolated changes). Each edited test that relies on
a HIT gains a spy on `GraphCache.load` (non-`None` result) or on the
callback (not called), so a silent miss can no longer pass (review B 12):
- `test_graph_cache.py::TestCacheHitPreservesCallback` (2 tests):
  `graph_cache=True` -> `cache_key=...`, `graph_cache=False` -> removed,
  plus `setenv("PHASIC_CACHE_DIR", tmp)`. The first test's callback
  captures a local `all_pairs = partial(combinations_with_replacement,
  r=2)`; it keys under the immutable-compiled-type rule (§2.2.1), so the
  test body is otherwise unchanged (review A 1).
- `test_serialize_constant_roundtrip.py`: `:106-113`, `:134` use
  `cache_key`; `load_graph(callback)` at `:111` -> `load(entry_key)`
  (from `g._graph_cache_entry`). T6 (`:219-240`) is rewritten to exercise
  the load-time format check: write a valid entry, rewrite its `format`
  field to `"1"`, assert `GraphCacheLoadError` naming the file; after
  deleting the file, the rebuild is a CONSTANT graph. Fixture `tmp_cache`
  (today `tmp_path/"graphs"`) sets `PHASIC_CACHE_DIR` to `tmp_path` (so
  entries land in `tmp_path/graphs`, not `graphs/graphs`), its
  `delenv("PHASIC_DISABLE_GRAPH_CACHE")` at `:68` is removed, and T1's
  `any(tmp_cache.iterdir())` becomes a check for a `*.json` file (a bare
  directory would pass vacuously).
- `test_theta_layout.py::test_graph_cache_hit_gets_default_layout`,
  `test_from_serialized_find_vertex.py::test_joint_prob_graph_via_warm_graph_cache`
  (spy moves to `GraphCache.load`; its callback also captures a local
  `all_pairs` partial, keyed as above): `cache_key`,
  `setenv("PHASIC_CACHE_DIR", tmp)`.
- `test_serialize_is_aux.py::test_is_aux_survives_the_graph_cache_json_path`:
  round-trips through the RESTRICTED decoder used on the load path
  (`graph_cache._decode_graph_data`) instead of `_deserialize_value`, so
  it keeps testing the real path (review A 5).
- `test_weight_formula_theta_dim.py::test_graph_cache_preserves_theta_dim`:
  `configure(graph_cache=...)` removed, `cache_key` used,
  `setenv("PHASIC_CACHE_DIR", tmp)`, and the `phasic.clear_caches()`
  call at line 112 DELETED (it wipes `$HOME/.phasic_cache` and
  `$HOME/.jax_cache` whatever the redirection, measured review A E;
  `tmp` starts empty, so it is not needed).
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
(not notebooks), EXCLUDING committed review evidence (`experiments/bug*/`,
`experiments/fd_step_guard/`, and `*.log` files), which records past
behaviour and is never edited. The only remaining hits allowed are the new API,
`per_thread_graph_cache` / `param_compute_graph_cache`,
`clear_from_cache(graph_cache=...)`, and module/function names of the
graph-cache helpers. Known now: `experiments/dr_discretize_cpp_sizes.py:26`
(drop the argument) and `experiments/dr_d1_e0_scale.py:371` and
`:424-426` (`PHASIC_DISABLE_GRAPH_CACHE` in an env-var list and in
`prep_env`: remove the variable, otherwise the child prints the new
import warning). `tests/pytest/test_slurm_multi_node_ffi.py:311, 328,
367, 384` also call `phasic.clear_caches()`; they are skipped without
SLURM and outside this fix, listed in `found-defects` instead;
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
The entry-key cost is measured against build and load time at 1,003
and 5,823 vertices (b2 models) AND at a typical size (coalescent
~18k and ~53k vertices; review A measured on master's JSON format:
build 3.2 s / 11.3 s, save 1.4 s / 4.6 s, entry 22 MB / 75 MB, load
0.75 s / 2.7 s, peak RSS 0.47 / 0.8 GB), plus the import-time stat
recording on its own. Expectation stated in the docstring, with file
size and memory given separately: roughly linear; at 600k sparse
vertices an entry file of about 0.85 GB, ~35 s per load, and a load-time
process peak of about 5 GB (extrapolated from 0.8 GB at 53k, review B 8);
more for dense (two-locus) models. The 53k measurement is repeated on the
new format and reported.
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
- `tutorial/state-space.ipynb` cells 36, 38: `graph_cache=True` ->
  `cache_key=...`.
- `tutorial/state-space.ipynb` cell 45: `Graph(callback, model=model,
  graph_cache=True)` with a user `MyModel` object. After the change a
  user object cannot be keyed (D3): pass `model.to_dict()` (and read it
  as a dict in the callback) with `cache_key=...`, or drop caching.
Text to revise (not raising):
- `tutorial/state-space.ipynb` cells 31, 39, 44 (cell 44 describes the
  `to_dict()` / deterministic-`__repr__` protocol for custom objects,
  which no longer applies: only plain values, numpy arrays and phasic's
  `StateIndexer` / `Property` are keyed).
- `tutorial/python_trace-and-jax-caching.ipynb` cells 1, 7 and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cells 1, 7 (the
  cache-correctness sentence must state the user-key contract).
- `tutorial/distributed.ipynb` cells 31, 37; and cell 36,
  `tutorial/python_trace-and-jax-caching.ipynb` cell 20 and
  `tutorial/configuration.ipynb` cell 29, which describe
  `PHASIC_DISABLE_CACHE` / a nonexistent `cache_enabled` field as
  disabling on-disk caching; also `tutorial/configuration.ipynb` cells 5
  and 6 (text and table row) and
  `tutorial/trace-and-jax-caching-c-path.ipynb` cell 26 ("honours
  PHASIC_DISABLE_CACHE=1"). The variable has no effect anywhere in
  phasic; the text should say so.
- Saved OUTPUTS only (harmless, refreshed on re-run):
  `tutorial/configuration.ipynb` cells 4, 19, `tutorial/mpfr.ipynb`
  cell 6 and `popgen/coalescent-derived.ipynb` cell 2 show `graph_cache`
  in printed configuration; logged "Saved graph to cache / Cache hit /
  Cleared" lines appear in `tutorial/svgd-basics.ipynb` (2, 6, 62),
  `tutorial/svgd-multi-feature.ipynb` (1, 2, 12, 24),
  `tutorial/svgd-learning-rate.ipynb`, `tutorial/svgd-multi-param.ipynb`,
  and the untracked `tutorial/mcmc.ipynb` (2, 4) and
  `tutorial/svgd-priors.ipynb` (cell indices 0-based).
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
- Callbacks that take or capture user objects (class instances,
  dataclasses incl. phasic's `HexCell` / `HexGrid`, pandas objects) can
  no longer be cached as they are (explicit error; pass plain values).
  This is the price of D3, which removes the class of stale hits a
  user-defined `to_dict()` can cause.
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

## 7. Review history

Entries are historical; where they disagree with §2-§6, the current text
wins (e.g. review 5's `fchmod` was replaced by `os.open(0o666)` in v2.3,
and review A's `cache._cache_root` by the inline rule in v2.7).

Review 4 findings -> v2.1

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
- Independent review A -> v2.6: 1 immutable compiled types admitted by
  name (§2.2.1, rule 7), existing tests keep their `all_pairs` closures;
  2 `clear_caches()` removed from `test_weight_formula_theta_dim`, no
  test calls it; 3 build stats recorded after `phasic_pybind` import;
  4 structured / object-field dtypes refused, `dtype.str` spelt out;
  5 `test_serialize_is_aux` uses the restricted decoder; 6 empty
  `PHASIC_CACHE_DIR` = unset via `cache._cache_root`; 7 tests isolate
  with `setenv("PHASIC_CACHE_DIR")`; 8 grep gate excludes evidence;
  9 ints via `hex`; 10 typical-size measurements, `e.graph` on write
  failure; 11 `PHASIC_DISABLE_CACHE` does not affect `cache_key`, doc
  cells listed; nits (`cache=True` docstring, `callback_name`). One
  SIMPLIFICATION beyond the findings: every unusable entry now raises
  `GraphCacheLoadError` (no warn-and-rebuild path; `GraphCacheEntryError`
  removed), for the reason stated in §2.1.
- Independent review B -> v2.7: 1 element counts in every composite
  token (prefix-free stream), test 5 nested cases; 2 cache-dir rule
  inlined, `phasic.cache` not imported by `graph_cache`; 3 import-order
  test without inherited `OMP_NUM_THREADS`, with a negative control;
  4 big int inside a list / default; 5 `PHASIC_DISABLE_CACHE` is read
  nowhere, three more notebook cells; 6 `e.graph` on any `Exception`
  from `save`, entry stays `None`; 7 shared `MPLCONFIGDIR`; 8 file size
  vs peak RSS at 600k; 9 `fsync` before rename; 10 modules in cells
  keyed by name, dataclass refusal documented; 11 grep list and SLURM
  `clear_caches` callers; 12 hit spies in edited tests; 13 entry path
  stored; 14 new names imported inside tests.
- Independent review C -> v2.8: 1 lazy stdlib loggers in `graph_cache`
  and `callback_hash` (fixes the pre-existing early `phasic_pybind`
  import; test 14 now meaningful); 2 `dyn_ordering` set from the reader's
  env on a hit, test 22; 3 `clear_all_graph_caches` exported, test 15;
  4 tuple constants recursed, Ellipsis/slice value rules, test 25;
  5 `callback_name` fallback; 6 whitespace claim narrowed, test 7;
  7 scipy.special attributes absent, `getattr` everywhere; 8 save errors
  amended in place with `add_note` (phasic needs 3.11), `finally` covers
  BaseException, durability worded as best effort; 9 decoder wraps every
  Exception; 10 tests 3/4/6 labelled guards; 11 `f.__dict__` keyed,
  test 23; 12 any fingerprintable dict key, call-time semantics, test 24;
  nits (fixture paths, `:68`, `:463`, `_phasic_build_id` named,
  `_deserialize_value` deleted, one `GraphCache()` per call).
- Independent review D -> v2.9: A1 pandas objects refused in rule 8
  (`to_dict()` collapses duplicate labels), one-to-one requirement on
  `to_dict()` documented; A2 visited map holds strong references,
  `to_dict()` results kept alive; A3 exact `partial` only; B1 build-stat
  recording never raises at import, failures and missing records raise
  at first key computation; B2 positional-only `cache_entry_key`
  parameters; B3 guards relabelled (tests 20 half, 24, 25 part); B4 test
  13 resets the memo; B5 "cannot be cached" note for unserialisable
  graphs; B6 Python-version wording, no dtype-string example, json
  RecursionError sentence. New test 26.
- v3.0 (user decision D3): generic `to_dict()` protocol removed; rule 8
  admits only phasic's `StateIndexer` / `Property`; user objects, pandas
  and dataclasses refused with plain-value advice; tests 8, 11, 26 and
  the notebook list (state-space cell 45 now raises) updated. Review D's
  A1/A2 are subsumed (no user `to_dict()` results are walked; only
  `StateIndexer.to_dict()` results are).
- Independent review E -> v3.1: 1 `StateIndexer` / `Property` admitted
  only in plain form (exact classes, attribute allow-lists, `offset`
  keyed), refusals in test 11; 2 alias/mutation and library-version
  responsibility in the contract; 3 `_serialize_value` made symmetric
  with the decoder; 4 temporaries wording; 5 more guards labelled;
  6 saved-output notebook list completed; 7 derived graphs documented in
  §2.7, absolute entry path, HOME patched in test 14, `getattr` for
  `__name__`, library versions in the contract.
- Independent review F -> v3.2: 1 `StateIndexer` admitted only if its
  behavioural state equals a canonical `from_dict(to_dict())` rebuild
  (catches edited property lists and overwritten internals); 2 builtin
  arm gated on exact types (no name-keying of `lru_cache` /
  `update_wrapper` objects); 3 exact numeric types; 4 allow-lists include
  lazy attributes, pinned on a used object; 5 cosmetics (§2.5
  parenthesis, one `GraphCache` instance in §2.1, `:607-609` reworded,
  §7 marked historical); 6-7 no side effects on a hit and hash-seed
  vertex order documented; 8 recursion bounded in every arm plus a
  top-level `RecursionError` catch.
