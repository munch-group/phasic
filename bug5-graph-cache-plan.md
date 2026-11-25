# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v6.1, 2026-11-25. Branch `fix/graph-cache-stale`, worktree
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
(helper functions, module variables, imported objects) or through a
MODULE captured in a closure cell (keyed by its name only, so its
attributes, and edits followed by `importlib.reload`, are not seen),
installed
library versions (numpy, scipy, ...: compiled functions are keyed by
name, Python functions captured in a cell by their code, never by the
module globals they read), files and the environment is the user's
responsibility: change `cache_key` when it changes. The key also
reflects arguments and captured state AT CALL TIME, so a callback must
not change its arguments or captured state in ways that affect the graph
(for example mutating one kwarg and reading an alias of it: aliases are
keyed by content). Object IDENTITY (`is`, `id()`) is not part of the
key: a callback that distinguishes equal-content objects by identity is
outside the contract (dicts keyed by function objects are refused
outright). A kwarg dict or list built by iterating a SET of
strings has a hash-seed-dependent order, so it misses across processes
(sort it first). The key's cost scales with the argument and captured
data and is paid on hits too. Functions ANYWHERE (callback, closure cells, defaults,
keyword arguments) are keyed by their code, defaults and closures, not
by the module globals they read; those globals are the user's
responsibility wherever the function is passed. On a hit the callback does NOT run, so any side
effect it has (filling a passed container such as a state-to-label map)
does not happen; and a callback whose output order depends on hash
randomisation (iterating a set of strings, using `hash()`) gets the
WRITER's graph on a hit: isomorphic to the reader's fresh build if the
randomisation only affects enumeration order, otherwise possibly a
different graph; such callbacks are outside the contract (they are not
deterministic functions of their inputs). The
`cache_key` docstring states all this with an example.

### 2.1 `Graph(..., cache_key=None)`

- Signature: `Graph(arg, ipv=None, *, cache_key=None, **kwargs)`
  (`cache_key` keyword-only); the
  `graph_cache` parameter is removed. `self._graph_cache_entry = None`
  is set at the top of `__init__`.
- `graph_cache=` passed: `TypeError` saying it was removed and that
  `cache_key="<name>"` caches a graph (otherwise it would reach the
  callback through `**kwargs`).
- `cache_key=None` (default): no lookup, no hashing, no file.
- `cache_key` given:
  - `type(cache_key) is not str` -> `TypeError` with the advice "pass
    str(name)" (so `np.str_` and `StrEnum` members are rejected clearly
    rather than tripping the key computation), empty -> `ValueError`;
    `arg` not callable ->
    `ValueError` ("cache_key applies to graphs built from a callback");
  - ALL argument validation for a callback runs first: the
    ipv/decorator checks now at `:1750-1759` move above the lookup (for
    every call, cached or not; same errors, raised earlier). They read
    `getattr(arg, "__name__", None)` instead of `arg.__name__`
    (`:1669`, `:1753`), so a `partial` callback gets the normal ipv
    error instead of today's `AttributeError` (review E 7d);
  - the ENTRY KEY (§2.2) is computed once. A key error propagates:
    nothing is built, nothing written;
  - lookup: `cache = GraphCache(create=True)` once per call, then
    `cache.load(digest)` (the full digest from §2.2) returns a separate, fully
    built `Graph`, or `None` (missing file), or raises (§2.6). ONE policy
    for every entry whose CONTENT cannot be used (OS errors such as
    `PermissionError` are re-raised with a message naming the file, and
    `MemoryError` / `RecursionError` propagate unchanged, §2.6)
    (malformed JSON or
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
    `self.dyn_ordering` is set to the flag of a freshly created native
    graph, `_Graph(1).get_dyn_ordering()` (the pybind base exposes
    `get_dyn_ordering` / `set_dyn_ordering`, `phasic_pybind.cpp:1550,
    1533`, not the `dyn_ordering` property, which
    exists only on the Python subclass, `__init__.py:1956-1971`, review
    X x4), which applies the C rule at graph
    creation exactly (`phasic.c:2996`, C `getenv`, which `os.environ`
    can diverge from after `os.putenv` or a C library's `setenv`, review
    W 2), because `serialize()` persists the WRITER's flag
    (`_graph_serialize.py:222`, restored at `:698`) and a hit must behave
    like a fresh build in the reader's environment (review C p9); plus
    `self._graph_cache_entry = <entry path>` (the ABSOLUTE file path,
    `Path.resolve()`, so a relative `PHASIC_CACHE_DIR` plus a later
    `chdir` cannot break it, and so
    `clear_from_cache` finds it even if `PHASIC_CACHE_DIR` changes later);
  - miss: BEFORE building, the cache directory's writability is probed by
    creating and removing a `.tmp-<hex>.partial` file (`O_CREAT |
    O_EXCL`); failure raises the save-path `OSError` (message naming the
    directory and the ways out) before any build work, so a read-only
    cache directory never costs an hours-long build (hits from a
    read-only cache keep working, since the probe runs on a miss only).
    Disk-full or quota errors can still only surface at save time
    (stated in the docstring). Then build as today, then
    `cache.save(self, digest, meta)` on the
    SAME `GraphCache` instance `cache` used for the lookup (one per call,
    §2.6);
    only after it returns, `self._graph_cache_entry = <entry path>`. ANY
    exception from `save()`, including `BaseException`s such as a
    `KeyboardInterrupt` aimed at a slow 600k-vertex save, gets `e.graph`
    and the note attached before it propagates (an `OSError` from the
    write, or for example
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
`cache_key` cannot collide with them): the FULL SHA-256 hex digest
(64 characters) of a TOKEN STREAM; the entry file name is its first 32
characters, and `GraphCache.load(digest)` / `save(graph, digest, meta)`
take the full digest and derive the file name. Every token is written as
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
3. the phasic build id (§2.2.3) and `numpy.__version__` (the state
   arrays the callback receives are numpy arrays phasic creates, and
   scalar promotion changed between numpy 1.x and 2.x, NEP 50, while the
   same phasic build can run against either). Bit-equality of a hit with
   a fresh build is otherwise only promised on the same machine type
   (numpy's SIMD math can differ by a few ULPs between CPUs); stated in
   the docstring;
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

#### 2.2.1 FUNCTION FINGERPRINT (exact, recursive descent; sharing handled
by sub-digests and cycles refused, §2.2.2)

- Python function `f`, EXACT type only (`type(f) is
  types.FunctionType`): `f.__module__` encoded through the VALUE
  FINGERPRINT, with `'__mp_main__'` normalised to `'__main__'` (spawn
  and forkserver workers import the main script as `__mp_main__`, which
  is the same module as the parent's `__main__`; without this, every
  multiprocessing worker would miss the parent's entry and rebuild,
  review N p3; the docstring notes the consequence: globals the parent
  sets inside `if __name__ == "__main__":` are absent in the worker, yet
  the worker gets the parent's graph, which is within the contract,
  since module globals are the user's responsibility) (it can be `None`, for example for functions made by
  `exec(src, {})` or `types.FunctionType(code, {})`, review M m5; the
  same applies to every other name-like token, and `cache_key` itself
  goes through the `str` value rule) (so two same-bodied callbacks
  from different modules, each reading its own module global, are kept
  apart even under one `cache_key`; it is `'__main__'` in notebooks, so
  notebook hits are unaffected; `co_filename` stays excluded); CODE
  FINGERPRINT of `f.__code__`; `f.__name__` and `f.__qualname__` (writable
  slots separate from `co_name` and absent from `__dict__`, changed by
  `functools.wraps` or assignment, which a callback could branch on,
  review AA 2; `__doc__` is not keyed and is named in the contract as
  not part of the key); VALUE
  FINGERPRINT of `f.__defaults__`, `f.__kwdefaults__` and `f.__dict__`
  (function attributes, e.g. `cb.rate = r` read back through the
  function's own cell; phasic's wrapper attributes `__wrapped__` and
  `__ipv__` are fingerprintable); for each name in
  `co_freevars`, the cell contents (empty cell -> token `empty`) by VALUE
  FINGERPRINT, where a cell holding a function is fingerprinted as a
  function (so decorator wrappers and captured helpers are covered).
- bound method, EXACT type only (`type(m) is types.MethodType`):
  FUNCTION FINGERPRINT of `__func__` + VALUE FINGERPRINT of `__self__`.
- NOT FINGERPRINTABLE: any other object that merely exposes `__code__`,
  in particular Cython's `cython_function_or_method`, whose `__code__`
  is a placeholder (all-zero `co_code`, empty `co_names`): distinct
  cyfunctions share full code fingerprints (two groups of 12 across 257
  scanned modules, review G g4), so a recompiled Cython helper in a cell
  would keep its key.
- `functools.partial`, EXACT type only (`type(x) is functools.partial`;
  a subclass may override `__call__` and falls through to the later
  rules or is refused): function, `args`, `keywords`, and `vars(p)`
  (a partial accepts attributes that a callback can read back through
  the cell holding it, review I i5; keyed like a function's `__dict__`).
- builtin function, numpy ufunc / array-function dispatcher, gated on
  EXACT type first: `type(x) is types.BuiltinFunctionType`,
  `type(x) is np.ufunc`, or `type(x) is type(np.sum)`
  (`numpy._ArrayFunctionDispatcher`); any other object (for example a
  module-level `functools.lru_cache` function, or an object made with
  `functools.update_wrapper`, both of which satisfy the conditions below
  and would otherwise be keyed by NAME, review F p6) is NOT
  FINGERPRINTABLE. (A Cython extension compiled with `binding=False`
  produces exact builtin functions that pass this arm by name, so a
  user's own recompiled extension keeps its key; this falls under the
  contract's "installed libraries" clause and is stated in the
  docstring.) For these exact types: module +
  qualified name token, ONLY IF `__self__` is `None` or exactly a module
  (`type(__self__) is types.ModuleType`) AND
  resolving `__module__` + `__qualname__` (or `__name__` for ufuncs)
  through `sys.modules` returns the very same object. This admits
  `math.sqrt`, `len`, `np.exp`, `np.linalg.solve` and refuses builtin
  bound methods of instances (`{}.get`, `Random(1).random`, which carry
  state in `__self__`) and `np.frompyfunc` ufuncs ("<lambda>
  (vectorized)"), which would otherwise share a name token (review 5 q1
  D). It also refuses every `scipy.special` ufunc (`gamma`, `binom`:
  they have NO `__module__` / `__qualname__` attribute, review C p8, so
  every attribute in these rules is read with `getattr(obj, name, None)`
  and a missing `__module__` / `__qualname__` (or `__name__` for ufuncs)
  means refusal with `GraphCacheKeyError`, never an `AttributeError`; a
  missing `__self__` reads as `None`, since numpy ufuncs and
  dispatchers such as `np.exp`, `np.sum`, `np.linalg.solve` have no
  `__self__` at all, review Y y5); the error advises moving
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
  ONLY IF `type(m) is types.ModuleType` (exact: a `ModuleType` subclass
  could run user code when `__name__` is read; numpy, scipy, jax, pandas,
  math, itertools and phasic are all exactly `ModuleType`, review X x6)
  AND `type(m.__name__) is str` (checked before the lookup, since a
  module's `__name__` is writable) AND `sys.modules[m.__name__] is m`.
  It behaves exactly like a
  module global, whose contents are the user's responsibility under the
  contract. (Value rule 6b below.) Admitted ONLY when a closure cell's
  DIRECT content is the module object (a module nested in a container
  inside a cell, in a function default `def cb(state, np=np)` or
  `__kwdefaults__`, or in phasic's own `ipv` wrapper cell, is refused
  like any other position); a
  module passed as a keyword argument or inside the ipv is NOT
  FINGERPRINTABLE, because there the contract promises keying BY VALUE
  and a module's attributes can change (`mod.RATE = 4.0`) or be reloaded
  (`importlib.reload` keeps the same object and name) without changing
  the token (measured: E[T] 1.6 / 0.4 / 0.16 under one token, review M
  m6). The error advises passing the needed values as plain kwargs.
- Sharing: see "Sub-digests" at the end of §2.2.2; a function is
  represented in its parent's stream by its sub-digest. CYCLES ARE
  REFUSED EVERYWHERE, including recursion through closure cells (a
  nested helper that calls itself, mutually recursive nested helpers, a
  dispatch table): advice, define such helpers at module level, where
  they are module globals and not walked.

CODE FINGERPRINT of a code object: `co_code`, `co_exceptiontable`,
`co_names`, `co_varnames`, `co_freevars`, `co_cellvars`, `co_argcount`,
`co_posonlyargcount`, `co_kwonlyargcount`, `co_flags`, `co_name`, and
`co_consts` element by element (nested code objects by CODE FINGERPRINT;
`frozenset` constants by their elements' TOKEN BYTES (a scalar's token,
or a container's sub-digest token, §2.2.2; computed from content alone,
so independent of the hash-seed-dependent iteration order and of object
sharing), sorted by those bytes; `tuple`
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

EVERY type gate in §2.2.1 and §2.2.2 tests `type(x)` (identity, or
`issubclass(type(x), ...)` where a family is meant), NEVER
`isinstance(x, ...)`: `isinstance` consults `x.__class__`, which a user
class can define as a property, so a spoofed object would pass and its
own attributes would then be keyed, running user code during the walk
(review V v6). FIRST of all, before any test that hashes or compares
`type(x)` (set or dict membership, dispatch tables), the walker checks
`type(type(x)) is type`: a user METACLASS could define `__hash__` /
`__eq__` and make `type(x) in ALLOWED` true for a spoof, running user
code (review AB p2); every admitted class has metaclass `type`, so this
check refuses nothing legitimate. Allow-set tests are written as
identity tests (`any(type(x) is t for t in ALLOWED)`, or a lookup made
safe by the metaclass check).

(Token tags: Python-scalar tokens (rule 1) and numpy-scalar tokens
(rule 2) use DISJOINT tag namespaces, `py:` and `np:` prefixes, so
`bool` and `np.bool`, whose type names are both `bool`, can never share
a tag, whatever the payloads.)
1. EXACT types only (`type(x) is ...`): `None`, `bool`, `str`, `bytes`:
   type name + canonical repr; `int` via `hex()` (no 4300-digit `repr`
   limit); `float` via its 8 IEEE-754 bytes (`struct.pack("<d", x)`;
   `float.hex` prints `nan` for every NaN regardless of sign and
   payload, review S s8); `complex` via the 16 bytes of both parts.
   Subclasses do not match here: `np.float64` (a `float` subclass) goes
   to rule 2, and `IntEnum` members and other subclasses are refused;
2. numpy scalar: admitted ONLY if `type(x)` is in a fixed frozenset of
   numpy's own scalar classes, built once at import from
   `np.sctypeDict.values()` with `__module__ == "numpy"`, minus
   `np.void`, `np.object_`, `np.longdouble` and `np.clongdouble`. The
   membership test reads no attribute of `x`, so no user code can run:
   any Python subclass of a numpy scalar is refused (it could override
   arithmetic and carries a `__dict__`, review H p6; a `dtype` property
   on a subclass would run if `x.dtype` were read, reviews Z z9 and AA
   p2, and `np.void` subclasses and numpy's own `np.record` even pass a
   `np.dtype(type(x)).type` identity test). The excluded kinds: `np.void`
   (structured, with `.fields`), object, and long double (`g` / `G`: on
   Linux x86-64 their 80-bit values sit in 16-byte slots with
   uninitialised padding, so equal values can have different bytes).
   User-defined dtypes such as `ml_dtypes`' float8 / int4 / bfloat16
   (installed alongside jax; they report `dtype.str` `<V1` / `<V2`, so
   two with equal bytes would collide, review K k3) are not in the set
   and are refused. For an admitted scalar, `type(x).__name__` +
   `x.dtype.str` + `tobytes()` (numpy's C code for these classes; the
   class name separates `np.int64` from `np.longlong`, which share
   `dtype.str` and bytes and differ only in `type()` / `repr`);
3. `type(x) is np.ndarray` exactly (masked arrays, `np.matrix` and other
   subclasses are refused): `x.dtype.str`, shape, a LAYOUT token (`C`
   if C-contiguous, else `F` if F-contiguous, else `S` followed by the
   strides IN BYTES (strides need not be multiples of the itemsize, for
   example field views of packed records, review T t4); a C- and an F-ordered copy of one
   array give different last-bit results in reductions such as
   `sum(axis=1)` and different `ravel('K')` order, review R p5;
   `DataFrame.to_numpy()` returns F-ordered arrays), C-order bytes (hashed
   from `memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))`,
   which copies only non-contiguous arrays and also works for datetime /
   timedelta kinds `M` / `m`, which numpy's buffer protocol rejects, and
   for 0-d arrays, via `reshape(-1)`; not via `tobytes()`). The dtype
   token also carries `x.dtype.type.__name__` (as rule 2 does for
   scalars; `int64` vs `longlong` arrays share `dtype.str` and bytes).
   The dtype gate: `x.dtype.type` must be identical to a class in rule 2's
   allow-set (reading `x.dtype` is numpy's C code once `type(x) is
   np.ndarray`); this refuses StringDType (its `.type` is `str`),
   user-defined dtypes, structured, object and long-double dtypes.
   Refused: a dtype that fails this gate,
   long-double dtype chars `g` / `G` (as in rule 2; on arm64 their
   `dtype.str` equals float64 / complex128), `dtype.hasobject` (covers object dtype AND structured dtypes with an
   object field, whose bytes are pointers) and `dtype.fields is not
   None` (structured dtypes: different field layouts share
   `dtype.str == '|V16'` and bytes, review A C);
4. exact `list` / `tuple`: type + element fingerprints in order;
5. exact `dict`: entry count, then entries in INSERTION order (a
   callback iterating a dict sees that order, so it can change the
   graph); identity-hashed objects (functions, bound methods,
   partials, and tuples or frozensets containing them, at any depth) are
   REFUSED as dict keys, because a callback looking them up by identity
   can tell apart two objects whose content encodings are equal (review
   Z z4: `scale={f: 1.0, g: 5.0}` with `f`, `g` equal-content closures);
   each other key by VALUE FINGERPRINT (any fingerprintable key: tuple
   state keys, int keys), then its value. The key
   reflects captured state AT CALL TIME: a closure-captured memo dict
   that the build fills is keyed as it was before that build (stated in
   the docstring, with the consequence spelled out: closure state that
   the build itself changes, such as a hand-written memo dict or a
   `nonlocal` call counter, makes every repeat call in the same process
   MISS and write a new entry; keep such memo state in a module global,
   which is not keyed);
5b. `Ellipsis` (token) and exact `slice` (start, stop, step by these
   rules);
6. any callable that §2.2.1 admits (Python function, bound method,
   partial, and builtins/ufuncs under exactly §2.2.1's conditions):
   FUNCTION FINGERPRINT; §2.2.1 and this rule are one rule;
   6b. a module object under §2.2.1's module rule, and only when it sits
   in a closure cell: module-name token (in the kwargs block or the ipv
   it is refused, see §2.2.1);
7. a class object (`issubclass(type(x), type)`): the immutable-compiled-type
   token of §2.2.1 if it qualifies, which additionally requires
   `type(x) is type` (no user metaclass, whose `__getattribute__` would
   run user code when `__flags__` / `__module__` / `__qualname__` are
   read, review Y 6); otherwise NOT FINGERPRINTABLE (so a
   `__class__` cell from `super()` or a user class passed as an argument
   is refused explicitly);
8. phasic's own value types, EXACT class only, and only in their
   plain, unextended form:
   - `Property` (standalone or inside an indexer, same rule everywhere):
     `type(p) is Property` AND `set(vars(p))` equal to its four dataclass
     fields (a frozen dataclass still has a `__dict__`, so an instance
     attribute such as an `encode_value` override set via
     `object.__setattr__` is otherwise invisible); keyed by ALL four
     fields (`name`, `min_value`, `max_value`, `offset`), since `offset`
     takes part in `__eq__` and is readable by a callback
     (`state_indexing.py:97`).
   - `StateIndexer`: admitted only if ALL of:
     (a) (run FIRST, before `to_dict()` is called) `type(ix) is
     StateIndexer`; exact container types for its internals:
     `type(...) is dict` for `_property_sets`, `_slots`, `_offsets` and
     every PropertySet's `property_dict`, `type(...) is list` for
     `_pset_order`, `_slot_order`, `_entity_order`, exact `np.ndarray`
     for EACH PROPERTYSET's `_bases` / `_radix_powers` (they are
     PropertySet attributes, `state_indexing.py:288-289`; reading them on
     the indexer raises), exact `str` keys in `_property_sets`, `_slots`
     and every `property_dict` (checked by `type(k) is str` before any
     hashing or comparison of a key), and exact `str` / `int` keys and
     values in `_offsets` (a dict subclass would run user code inside
     `to_dict()` itself, or lie to check (c), review AB 2); every PropertySet's `properties` is
     exactly a `list` or `tuple` (review Y 7a), every PropertySet has
     `type(ps) is PropertySet` or `type(ps) is StateSpace`
     (`state_indexing.py:220`, `:2204`), and EVERY Property the indexer
     can reach passes the Property rule: the entries of each
     `properties` list AND the values of each `property_dict` (which
     `props_to_index` validates through, `:533`); every slot reachable
     through `_slots` has `type(slot) is Slot`, `set(vars(slot))` equal
     to that of a fresh `Slot`, and `slot.name` equal to its key;
     (b) attribute NAMES within allow-lists: for the indexer and for each
     PropertySet, the attributes of a freshly built object PLUS the
     documented lazily added ones (`_props_dataclass`, added on the first
     `index_to_props`, `state_indexing.py:427-434`; any other lazy
     attribute found at implementation time is listed the same way). The
     pin test compares the allow-lists with `set(vars(...))` of an object
     that has been USED (after `index_to_props`), so it neither misses a
     lazy attribute nor refuses an indexer a build has already used
     (review F 4);
     (c) (run only AFTER `to_dict()` has been fingerprinted, which
     refuses any non-plain field value, so no user operator can run
     during the rebuild) BEHAVIOURAL STATE equal to a canonical rebuild: let
     `c = StateIndexer.from_dict(ix.to_dict())`; then `ix._offsets`,
     `ix._total_state_length`, `_pset_order`, `_slot_order`,
     `_entity_order`, and for each PropertySet
     `[(p.name, p.min_value, p.max_value) for p in ps.properties]`
     (compared as lists, so a caller's TUPLE of properties is admitted,
     and on folded values only, so a deprecated `offset`, which
     `to_dict()` drops after `__post_init__` folded it into min/max,
     `state_indexing.py:99-106`, does not cause a refusal), the same
     triples for `property_dict.items()` with their keys, and each
     PropertySet's `_bases` and `_radix_powers` must equal those of the
     matching PropertySet of `c`, where equal for these
     numpy arrays means same `type`, `dtype` and `shape` and
     `np.array_equal` (a bare `!=` would raise on multi-element arrays,
     and value-only equality would admit a float array with different
     behaviour). The GENERATED classes (`ix._result_class`, created in
     every indexer's constructor, `state_indexing.py:878, 976`, and each
     PropertySet's lazily created `_props_dataclass`, `:429`) are NOT
     compared entry by entry with the rebuild's (v5.7's entry-by-entry
     rule could not pass on any indexer: the dataclass machinery,
     `__dataclass_fields__`, `__dataclass_params__`, per-class
     descriptors, annotation objects, a `FrozenInstanceError` cell and a
     `repr_running` set, is neither identical nor fingerprintable,
     review AF p3). Instead, an ADDITIVE registry in `state_indexing.py`,
     installed as a module-level wrapper that rebinds the module's
     `make_dataclass` name (only lines 429 and 976 call it; their bodies
     are untouched, matching the no-modify-existing preference; the
     generated class's `__module__` stays `phasic.state_indexing`): the
     wrapper calls the real `make_dataclass`, then stores a SNAPSHOT
     `dict(vars(cls))` (the same entry objects) ON THE CLASS ITSELF as
     `cls.__phasic_snapshot__`, together with the field names, wrapped in
     a tiny holder object `_Snap` (`__slots__`, and a `__reduce__`
     returning `(_Snap, (None, None))`): a bare dict there makes
     `dill.dumps(indexer)` fail for EVERY indexer, used or not, a
     regression for all users, including phasic's own `%%srun` magic,
     which dill-dumps notebook globals (`srun_magic.py:147, 219-224`) and
     would silently lose such an indexer; with the holder, dill and
     cloudpickle copy a harmless empty stub, and the rebuilt class stays
     unregistered (review AI p4-p6), and adds
     the class to a module-level `weakref.WeakSet` `_GENERATED_CLASSES`.
     (A dict of weakrefs plus a snapshot held outside the class, as in
     the first v5.8 text, leaks: the snapshot's `__dict__` / `__weakref__`
     descriptors and the frozen `__setattr__` closure reference the
     class, so it is never freed; measured 55 MB over 1000
     joint-probability-graph constructions; with the snapshot on the
     class and a WeakSet, 0.16 MB and no live classes, review AG p3/p4.)
     The check requires: the class is in `_GENERATED_CLASSES`;
     `cls.__bases__ == (object,)` and `cls.__name__` is the name phasic
     generates (`IndexResult` / `<Name>Props`); the field names equal the
     ones the indexer's properties / slots imply; and `vars(cls)`, minus
     the `__phasic_snapshot__` key and minus `__slotnames__` (which
     CPython's copyreg adds to a class the first time one of its
     instances is copied or pickled, e.g. a `deepcopy` of a
     state-to-label dict holding `index_to_props` results; it only
     affects copying of instances, review AI p3), has exactly the
     snapshot's keys with every entry the IDENTICAL object (`is`). A user-made class is not
     registered; replacing or adding a class attribute (a swapped
     `__init__`, an added `__post_init__`) changes an entry's identity or
     the key set; changing the bases or the name is caught by the first
     two conditions. Not detected (documented as outside the contract,
     like other deliberately altered internals): in-place edits INSIDE an
     entry (`cls.__init__.__code__ = ...`, mutating
     `__dataclass_fields__` contents). No entry is fingerprinted, so no
     dataclass internals need rules, and the check reads only the class's
     own `__dict__`, `__bases__` and `__name__` (no user code: the
     class's metaclass is `type`, checked first). An indexer sent through
     cloudpickle (joblib/loky) or dill arrives with a REBUILT, unregistered
     class and is refused with an error naming this case and the way
     out: rebuild it in the worker with `StateIndexer.from_dict(
     ix.to_dict())`, or construct the indexer there (review AG p6).
     (Review AC p8's swap and review AE 1's swaps are refused.) Each Property must have
     `set(vars(p))` equal to its four dataclass fields (a frozen
     dataclass still has a `__dict__`, so extra attributes are
     otherwise invisible; same check as (a)), and each slot is checked as
     in (a) (`state_indexing.py:193`; subclasses are accepted by the
     constructor at `:855-862`). Any exception raised
     while running ANY of (a), (b) and (c) (for example `AttributeError` for a missing
     attribute on an old pickled indexer, since `__getattr__` raises for
     `_` names, `:996-998`) is re-raised as `GraphCacheKeyError` (path
     chained). This catches the known routes by which `to_dict()` and
     behaviour drift apart (an indexer whose private attributes are
     deliberately replaced in ways not listed here is outside the
     contract):
     a property list edited after construction (`PropertySet` keeps the
     caller's list uncopied, `:280`, while `property_dict`, `_bases`,
     `_radix_powers`, `_offsets` and `_total_state_length` are computed
     once, `:281`, `:288-289`, `:906-924`; measured: equal `to_dict()`,
     5 vs 7 vertices, review F p3) or the compared values overwritten.
     Keyed as: class token, PropertySet class names in order, then the
     VALUE FINGERPRINT of `to_dict()`, followed by a counted list of
     every Property's `offset` in `to_dict()` property order (behaviour
     is fixed by the folded min/max that (c) checks; `offset` is keyed
     because a callback can read it). Anything that
     fails (a)-(c) is NOT FINGERPRINTABLE, with an error naming the
     failing check (an extra attribute, a subclass, or "indexer state
     differs from its to_dict(); was its property list changed after
     construction?"). Exceptions from `to_dict()` / `from_dict()` are
     re-raised as `GraphCacheKeyError` (path chained). Cost: one
     `from_dict` per DISTINCT indexer per key computation (indexers are
     memoised after their checks pass, §2.2.2), negligible next to a
     build.
   NO other object is keyed through `to_dict()` or any other
   user-defined method: a user's `to_dict()` cannot be checked to be
   one-to-one (pandas' is not: it collapses duplicate index labels,
   review D d3);
9. anything else: NOT FINGERPRINTABLE. This includes user class
   instances (with or without `to_dict()`), pandas objects, dataclass
   instances (phasic's `HexCell` / `HexGrid` too) and bound methods of
   such instances. The error advises passing the values the callback
   needs as plain values (numbers, strings, lists, dicts, numpy arrays),
   for example `model.to_dict()`, or numeric DataFrame columns via
   `to_numpy()` with labels as lists (a mixed or string frame's
   `to_numpy()` is an object array, refused), or building without
   `cache_key`. The docstring lists the common refusals so they are not
   a surprise: `range`, `frozenset`, `defaultdict` / `OrderedDict`
   values, `pathlib.Path`, `jax.Array`, `operator.itemgetter`, a local
   `lru_cache` helper in a closure, numba `@njit` / `jax.jit` helpers
   and `np.vectorize` objects in a closure, `collections.namedtuple`
   instances (pass `nt._asdict()` or a tuple), `np.memmap` arrays (pass
   `np.asarray(mm)`; note this reads the file to hash it), the class
   `collections.Counter` captured in a cell, phasic's own
   `index_to_props` results (precompute lookup tables as plain ints or
   dicts), long-double arrays, functions whose `__dict__` holds an `inspect.Signature`
   (set by some decorator libraries as `__signature__`), numpy 2
   `StringDType` arrays, numpy dtype INSTANCES such as
   `np.dtype('int64')` (capture `np.int64` or the dtype string instead),
   `fractions.Fraction`,
   `decimal.Decimal`, nesting deeper than 100. The docstring also says
   that a function made with `exec(src, ns)` or `types.FunctionType(code,
   ns)` reads its globals from `ns`, which counts as MODULE GLOBALS
   (not keyed; change `cache_key` when `ns` changes). The docstring also says that entries
   are never evicted: every distinct argument set under one `cache_key`
   writes its own file (22-75 MB at typical sizes, §4), so a parameter
   sweep under one name accumulates files; clear them with
   `phasic.clear_all_graph_caches()` or `Graph.clear_from_cache()`.
Recursion is bounded in EVERY recursive arm (values, functions, code
objects and their constants, `to_dict()` results), and `cache_entry_key`
additionally catches `RecursionError` at its top level and re-raises it
as `GraphCacheKeyError`.

SUB-DIGESTS (v4.4; replaces the back-reference tokens of v4.1, which
made frozenset constants hash-seed dependent, and the recursion
back-references of v4.2-v4.3, whose soundness depended on the walk's
entry point, review R p2). Every container (exact list, tuple, dict, and
the dicts `StateIndexer.to_dict()` returns), every function, bound
method and partial, every numpy array and every `StateIndexer` that
passed rule 8 is represented in its parent's stream by ONE fixed-length
token: its SUB-DIGEST, the SHA-256 of its own token stream, which begins
with a KIND TAG unique to its kind (list, tuple, dict, function, bound
method, partial, CODE OBJECT, array, indexer; code objects are memoised
too, so one `def` in a loop producing many closures hashes its code
once) and in which its children are in turn
sub-digests. A tuple has ONE encoding whether it is reached as a code
constant or as a value (the constant-only additions, `Ellipsis` and
`slice`, are also value rules, rule 5b), so the shared memo can never
hold two meanings for one object.
NO CYCLES: the current-path check applies to EVERY memoisable object
(containers, functions, bound methods, partials, indexers); meeting an
object that is already on the current path raises `GraphCacheKeyError`
naming the path, whatever the entry point. Consequently every sub-digest
is a pure function of the object's content: it never depends on where
else the object occurs, on sharing, on walk order or on the hash seed.
Within one key computation sub-digests are memoised by `id()` in ONE
memo shared by values and code constants, holding a strong reference to
each object until the computation ends (so a freed temporary's id is
never reused). Each memoised object is hashed exactly once, so the walk
is linear in the number of distinct objects (a DAG such as `a = [a, a]`
nested k times takes k steps; 200 references to one 8 MB array hash it
once; shared state tuples cost nothing extra); no budget or re-walk cap
is needed. Order of checks at each memoisable object: (1) current-path
check, (2) memo lookup, (3) walk. The identity-hashed-key refusal of
rule 5 is a SEPARATE predicate run on every dict key, ignoring the
memo (a key tuple memoised earlier from a value position must still be
refused as a key, review AB 3). Modules (direct cell content only) are
never memoised: their admission depends on position (§2.2.1). Scalar
tokens have their own tags, distinct from all sub-digest kind tags.
Limits: a depth limit of 100, and the current-path check above (a
process-wide seen-set is wrong: it flags a list referenced twice, an ipv
reusing one state list, and freed temporaries whose ids are reused,
review 5 q2). Both raise `GraphCacheKeyError` (never `RecursionError`).

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
(`__init__.py:279`), a call `callback_hash._record_build_stats(s0)` (taking
the top-of-import snapshot `s0` as a parameter) at the
end of `phasic/__init__.py` records ONCE PER PROCESS (the record is
stored as an attribute on the `phasic_pybind` extension module, the one
object `importlib.reload` can never replace; a second call, from
`importlib.reload(phasic)`, COMPARES the
files against that record instead of replacing it (a reload of
`callback_hash` alone does not call `_record_build_stats`; the next key
computation checks against the record surviving on the extension
module), and stores a
"restart" failure if they differ, because the old code keeps running
after a reload, review P p2), for the extension module located by
`phasic.phasic_pybind.__file__` (not by globbing; no `__file__` or a
missing file is a recording failure, so an editable or source-tree layout
can never silently leave the extension out of the id), every OTHER file
under the package directory ending in one of
`importlib.machinery.EXTENSION_SUFFIXES` (one `.so` today; a second
compiled module added later must not fall out of the id silently), and
every `*.py` file under the package directory of `phasic.__file__`,
RECURSIVELY, excluding `__pycache__` (the package is flat today, 52
files, but a future subpackage must not fall out of the id silently):
(i) a stat record `(path, size, mtime_ns, ctime_ns, ino, dev)` per file,
and (ii) the CONTENT HASH of those files, computed right there, at import
(measured 11.9 ms for 52 .py + the .so, 3.49 MB, review 4; paid by every
`import phasic`, accepted as the price of an unambiguous id). The content
hash uses the same counted, length-prefixed framing as the key: for each
file in sorted relative-path order its relative path, its size and its
bytes, so moving text across a file boundary (an extraction refactor)
changes it. This content hash IS `_phasic_build_id()`. To make sure it
describes the code the process actually imported, a stdlib-only stat
snapshot is taken at the TOP of `phasic/__init__.py` (immediately after
`from __future__ import annotations`, which must stay the first
statement), before any
submodule or `phasic_pybind` import: `os.scandir` of the package
directory for `*.py` (recursive, as above) and for EVERY file ending in
one of `importlib.machinery.EXTENSION_SUFFIXES` (never
`importlib.util.find_spec`, which would call every `sys.meta_path`
finder and trip test 14's import hook); at the end of import,
`phasic.phasic_pybind.__file__` must be among the snapshotted files
(otherwise a recording failure is stored, whose message says
"cache_key needs a regular install of phasic (the compiled extension is
outside the package directory, e.g. an editable install); use `pixi run
install-dev` or a normal pip install", and the docstring lists editable
installs among the refusals; scikit-build-core's editable redirect mode
places the `.so` outside `src/phasic`). Also at the end of import,
iterating `list(sys.modules.items())` and skipping `None` values: every
loaded `phasic.*` module's source file must be among the hashed files,
where the source file is `__spec__.origin` if `__spec__` and its origin
are not `None`, else `__file__` (pybind11 submodules such as
`phasic.phasic_pybind.hash` and `phasic.phasic_pybind.parameterized`
have `__spec__ is None` and `__file__` equal to the hashed `.so`, review
AD p1); a module with neither is exempt only if its name starts with
`phasic.phasic_pybind.`; otherwise a recording failure is stored (a sourceless
`.pyc` layout, or a module loaded from outside the package directory,
would otherwise run code the id does not cover; the docstring states
that the id covers the package's source and extension files). ORDER at the end of import:
first the content hash H is computed from all the files, THEN they are
re-stat'ed (S1); S1 must equal the top-of-import snapshot S0 in size,
`mtime_ns`, inode and device (a difference in `ctime_ns` ONLY, from a
hard-link count change or an xattr write during the 3-15 s import, is
accepted and S1 is recorded), otherwise a recording failure is stored
(raised at the first key computation); the stored record is (S0 or S1,
H). (Re-hashing at import to accept a stats change would re-open the
window review W 1 closed, so it is not done.) Hashing before the final stat closes the window
in which a concurrent reinstall could change a file after its last stat
but before it is read, which would record a hash of code the process is
not running and let the later "stats changed, bytes equal" rule accept
it (review W 1). So no reinstall during or after import can go
unnoticed (review I 2, review M 3, review W 1).
Recording NEVER raises at import: any failure (a file vanishing between
listing and reading, a missing extension file) is stored and raised as
`GraphCacheKeyError` by the first key computation, so `import phasic`
cannot break for users who never cache; a missing record (for example
if the record attribute on the extension module was deleted, or
`_record_build_stats` never ran because `phasic/__init__.py` failed
part-way) also raises `GraphCacheKeyError` rather than skipping the
check.
On EVERY key computation the package directory is re-listed and the files
re-stat'ed (measured 2.1 ms, review 5). If nothing changed, the import
hash is used. If a file was added or removed: `GraphCacheKeyError`
"phasic was reinstalled after it was imported; restart the interpreter"
(by design; the error text and the `cache_key` docstring say that keyed
runs must not overlap a reinstall, e.g. `pixi run install-dev`, of the
same environment, since a rebuilt `.so` never re-hashes equal).
If only stats changed (any of size, mtime, ctime, inode, device), the
files are RE-HASHED and compared with the import hash: equal means the
bytes are unchanged (for example a hard-linked file in a shared conda /
pixi package cache whose `ctime` moved because another environment
linked or unlinked it, or an xattr write on Lustre/GPFS, review J p5),
and the stat record is refreshed silently; different means a reinstall,
and `GraphCacheKeyError` as above. So a ctime-only metadata change, at
import or later, never produces a spurious "restart" error, and a real
reinstall is always caught.
Recording in `graph_cache`'s module body would be wrong: that module is
imported at `__init__.py:53`, before `OMP_NUM_THREADS` is set at
`:107-108`, and touching `phasic_pybind` there would load the native
library and libomp too early (review A F). OWNERSHIP: the build id lives
in `callback_hash` (`_record_build_stats`, `_phasic_build_id`, the
recorded table), which imports nothing from `graph_cache`; `graph_cache`
imports `callback_hash` (as today, `graph_cache.py:26`), so there is no
import cycle. `.cpp` / `.h` files shipped in the package directory are
not part of the id (graph construction goes through the compiled
extension only).

### 2.3 Errors

- Both cache exceptions are DEFINED in `callback_hash` (which imports
  nothing from `graph_cache`; `graph_cache` imports them from there) and
  re-exported from `phasic`.
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

`GraphCache()` with no argument resolves its directory when constructed
and makes it ABSOLUTE (`Path.resolve()`, the same call as for the entry
path in §2.1), so a `chdir` by the callback
or another thread during a long build cannot send `save()` elsewhere:
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
`monkeypatch.setenv("PHASIC_CACHE_DIR", <tmp>)` (the docstring notes that
a RELATIVE `PHASIC_CACHE_DIR` is resolved once by the graph cache but
used verbatim, relative to the current directory, by the C-side caches,
so set it absolute), which also redirects
the C-side caches that `clear_from_cache` touches (§3). `get_cache_dir`, `get_graph_cache_stats`,
`print_graph_cache_info` and `clear_all_graph_caches` use the same
resolution. `model_export.clear_caches` / `clear_model_cache` do NOT: their
graph-cache part keeps using the legacy default directory
(`graph_cache.DEFAULT_CACHE_DIR`, read at call time), consistent with
their other part, which clears `$HOME/.phasic_cache` regardless of
`PHASIC_CACHE_DIR` (`model_export.py:72`; pre-existing, found-defects
item P). So a `clear_caches()` call never reaches a user-chosen, possibly
shared `PHASIC_CACHE_DIR` (review AH 2); entries there are cleared only by
`clear_all_graph_caches()` or `Graph.clear_from_cache()`. Both docstrings
say so.

### 2.6 `GraphCache` API and entry file

- Kept: `GraphCache(cache_dir=None)`, `clear()`, `clear_graph_cache()`,
  `get_cache_stats()`. The constructor takes `create: bool = False`
  (DEFAULT False): only `Graph.__init__`'s `cache_key` path constructs
  `GraphCache(create=True)`, which creates the directory (the miss-path
  writability probe needs it to exist); so the documented user pattern
  `GraphCache().clear()` (notebooks listed as unaffected in §5) never
  runs `mkdir`. The management helpers
  (`clear()`, `clear_graph_cache()`, `get_cache_stats()`,
  `clear_all_graph_caches()`, `get_graph_cache_stats()`,
  `print_graph_cache_info()`, `get_cache_dir()`) RESOLVE the directory
  without `mkdir` (the default `create=False`), so a missing
  directory gives 0 removed / empty stats and a read-only or missing HOME
  never makes them fail or creates a directory as a side effect
  (pre-existing on master, `graph_cache.py:88-89, 485, 523`, review U 1).
  The constructor's `mkdir(parents=True,
  exist_ok=True)` (`graph_cache.py:88-89`) failing re-raises the same
  `OSError` subclass with a message naming the directory and the ways
  out (another directory via `configure(cache_dir=...)`, or no
  `cache_key`); this happens before the lookup, so nothing is built.
  `save()` repeats `mkdir(parents=True, exist_ok=True)` before
  `os.open`, so a cache root removed during a long build (for example
  `rm -rf ~/.phasic_cache`) is recreated rather than failing after the
  build. (`phasic.clear_caches()` deletes files but keeps directories,
  `cache_manager.py:142-152`; it can delete a concurrent writer's
  in-flight `.partial`, whose `os.replace` then fails with
  `FileNotFoundError`: a save error with `e.graph` attached, the build
  kept, nothing saved.) EDITED: `_serialize_value`
  (`graph_cache.py:304-378`) loses its generic `to_dict()` arm (which tags
  any object with `__type__: module.Class`) and its docstring/error text
  advertising the `to_dict()`/`from_dict()` protocol; it encodes plain
  JSON values plus the `ndarray` tag only and raises on anything else, so
  the save side is symmetric with the restricted decoder and no entry
  can be written that every later lookup would refuse. Deleted:
  `_deserialize_value` (see below). `Graph.__init__` constructs ONE
  `GraphCache(create=True)` per call, whose directory is absolute (§2.5)
  and exists (the miss-path writability probe relies on that), so lookup
  and save use the same directory. `GraphCache.entry_path(digest)`
  returns the absolute entry path (used for `_graph_cache_entry`).
  `clear()` / `clear_graph_cache()` delete ONLY files whose names match
  `^[0-9a-f]{32}\.json$` (entry names, old and new; the directory may now
  be user-chosen via `PHASIC_CACHE_DIR`, so other JSON files there must
  survive, review M 6); `get_cache_stats` counts the same pattern. The
  `int` they return counts removed entry files only, not `.partial`
  files. They
  (and through them `clear_all_graph_caches` and
  `model_export.clear_caches`, which uses the legacy default directory,
  §2.5) also remove
  leftover `.tmp-*.partial` files, which a writer killed between
  `os.open` and `os.replace` (SIGKILL, OOM kill, SLURM time limit) leaves
  behind (review 7 C), but ONLY those whose mtime is more than one hour
  old, so a concurrent writer's in-flight file in a shared directory is
  never removed under it (review 8; a `.partial` counts as stale when
  (cache directory mtime) - (file mtime) exceeds one hour, both read
  from the same file server rather than compared with the local clock,
  so NFS clock skew cannot shorten it; creating the `.partial` itself
  bumps the directory mtime, so the difference only grows through later
  activity in the directory and a file is never judged older than it
  is, at the price of stale files lingering in an idle directory; a writer
  suspended for over an hour mid-save, e.g. SLURM preemption by
  SUSPEND, can still lose its `.partial`, which is only a save error
  with `e.graph` attached, documented); `get_cache_stats` counts only
  entry-named files (the same pattern as `clear()`), never `.partial`
  files or other JSON files.
- Replaced: `save_graph(graph, callback, **params)` /
  `load_graph(callback, **params)` -> `save(graph, digest, meta)` /
  `load(digest)` (full 64-character digest; file name = first 32), where
  `meta` is a `dict` with exactly the keys `cache_key` (str) and
  `callback_name` (str or None); `save` adds `entry_digest`, `format`,
  `phasic_version`, `created_at` and `graph_data` itself.
- Removed: `get_or_build`, `hash_callback`, `_detect_closures`,
  `_normalize_ast`.
- Entry content: `entry_digest` (the FULL SHA-256 hex of the key token
  stream; the file name holds its first 32 characters), `format`,
  `phasic_version`, `created_at`, `cache_key`,
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
  JAX/OpenMP threads run), and no `mkstemp` 0600 mode), then
  the file object is a BINARY buffered writer,
  `graph_cache._open_entry_writer(fd)` = `os.fdopen(fd, "wb")`; the
  entry's small fields are written first, `format` FIRST, as the exact
  bytes of one shared
  constant `_ENTRY_HEADER = b'{"format":'` followed by the format string
  (so a reader can tell an entry's format from its first bytes, and the
  legacy sweep, §2.8, tests the same constant; a natural `json.dumps`
  header would write `{"format": ` with a space), then EACH large `graph_data` field encoded with
  `json.dumps(value, separators=(",", ":"))` (ASCII, since
  `ensure_ascii` is on by default; the one-shot call uses CPython's C
  encoder; `json.dump` streaming always uses the pure-Python encoder,
  2.8x slower, review O p6) and written into the same JSON object in
  8 MiB slices, each slice encoded to bytes just before it is written
  (`f.write(s[i:i+CH].encode("ascii"))`), so the bytes copy never
  exceeds one slice (a whole-string write, text or binary, costs a full
  extra copy, review Y y8), then `f.flush(); os.fsync(f.fileno())` (an
  `OSError` from `fsync` with errno `EINVAL`, `ENOTSUP` / `EOPNOTSUPP` or
  `EROFS`, which some FUSE, network and overlay filesystems return, is
  logged and ignored: durability is best effort; any other error
  propagates)
  (memory stays bounded per field), then
  `os.replace` to `<first 32 hex>.json`. The buffered writer loops over
  short writes (a single `os.write` of more than 2 GiB writes only
  `INT_MAX` bytes on macOS and about 2 GiB on Linux, so a natural
  `os.write(fd, data)` would store a truncated entry and report success,
  review N p8), and the sliced encoding bounds the extra memory to one
  slice per field (the field's string itself is still held while it is
  written). Compact separators replace master's `indent=2`
  (`graph_cache.py:153`), which was 3.6x larger (2.45 MB vs 0.68 MB at
  7,382 vertices, review N p6). The
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
  (concurrent writers are a documented use: `distributed.ipynb` cell 31
  builds several graphs with a `ProcessPoolExecutor`; shared cluster
  directories add more).
- On load, `entry_digest` must equal the full digest passed to `load()`;
  a mismatch is a `GraphCacheLoadError` naming the file (a file copied or
  planted under the wrong name, or a truncated-name collision, would
  otherwise be loaded silently as a wrong graph, review M 2).
- Load errors are classed exactly as:
  - `FileNotFoundError` on open -> `None` (miss);
  - `ESTALE` ("Stale file handle", on NFS when another client replaced the
    entry with `os.replace` while this reader had it open): re-open and
    re-read ONCE (an entry under the same name is equivalent by key); a
    second `ESTALE` falls through to the next case;
  - any other `OSError` on open/read (for example `PermissionError` on an
    entry another user wrote under umask 077, common on HPC): re-raised
    as the same `OSError` subclass, original chained, with a message
    naming the file, its owner (best effort: `os.stat` or `pwd.getpwuid`
    failing, for example in a non-searchable directory, gives "owner
    unknown" and never raises from the handler) and the ways out
    (another `cache_key`, or another directory via
    `configure(cache_dir=...)`);
  - (entries are opened with `encoding="utf-8"`, so the classification
    below does not depend on the locale)
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

(`_graph_cache_mgmt.py:70-91`, plus its docstring at `:21`
("<callback_hash>.json") and error text at `:81-83` ("callback-hash
key"), both reworded; this parameter selects which cache to
clear and is unrelated to the removed option, so it stays.) Uses
`getattr(self, "_graph_cache_entry", None)`, which holds the entry's
file PATH resolved at construction (not just the key). No entry (built without
`cache_key`) -> `removed["graph_cache"] = 0`, documented. Graphs
DERIVED from a cached graph (`discretize`, widening, `laplace_transform`,
`add_epoch`: `__init__.py:3445-3447, 10136-10138, 3003-3005, 3276`) do
not inherit `_graph_cache_entry`, so `clear_from_cache()` on them reports
0. On master the same call re-hashed the copied `_callback` and deleted
the BASE graph's entry; the new behaviour is intended (a derived graph
has no entry of its own) and documented. `clone()` / `copy()`
(`__init__.py:10063-10077`) construct via `Graph(super().clone())`, so
their `_callback` is `None` and `clear_from_cache()` on them RAISES "not
built from a callback", as on master, as for any graph not built from a
callback. The `graph_cache.py` docstrings at `:7`, `:52`, `:86`
(`~/.phasic_cache/graphs/`) are reworded to name `PHASIC_CACHE_DIR`.

### 2.8 Old entries

Entries written before this change are never loaded (different format
and key composition). They stay on disk; `phasic.clear_all_graph_caches()`
removes them: it sweeps entry-named files (`^[0-9a-f]{32}\.json$`) from
the current graph directory, and ALSO from the legacy default directory
(master always wrote there, ignoring `PHASIC_CACHE_DIR`,
`graph_cache.py:32, 88`), so users who set `cache_dir` do not keep
orphaned old entries. The legacy directory is `graph_cache.DEFAULT_CACHE_DIR`
READ AT CALL TIME (never `Path.home()` or a fixed constant, so tests
redirect it), and if it differs from the current directory: it is never
created (absent means skipped; no `GraphCache(legacy)` constructor with
its `mkdir` is used, so a read-only or missing HOME on a compute node
cannot make the call fail); only files that are clearly OLD-format are
deleted there, i.e. whose first bytes are NOT `_ENTRY_HEADER` (§2.6; new entries
write `format` first, §2.6, and any future format will too), so valid
entries of this or a newer phasic that other runs wrote to the default
directory survive (review S 1, review T 1); a read or unlink error on a
single file is skipped and counted, never fatal. Return types:
`GraphCache.clear()` and `clear_graph_cache()` keep returning an `int`
(files removed), so `model_export.clear_model_cache`'s `count > 0`
(`model_export.py:83-84`) keeps working; `clear_all_graph_caches()`
returns a `dict` `{"removed": int, "skipped": int}` (removed over both
directories, skipped = per-file errors); it has no internal caller (only
a docstring mention at `_graph_cache_mgmt.py:60`). Docstring says so.

## 3. Tests

New file `tests/pytest/test_graph_cache_key.py`. An autouse fixture sets
`HOME` to a directory under `tmp_path` (so no code path, however it
resolves a home-relative default, can reach the real
`~/.phasic_cache`), and `PHASIC_CACHE_DIR` to a directory under `tmp_path` (this redirects the
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
`state[..., 0]` part of test 25 (master hashes the AST; the guard is
spy-observed: hit on rebuild, not a key-level assertion), and the
`PHASIC_DISABLE_CACHE` half of test 20 (master ignores `cache_key` but
caches by default; this half and test 3 prove the hit with a spy on the
callback, not by looking for a file under `$PHASIC_CACHE_DIR/graphs`,
which master ignores), test 10 (master's `StateIndexer.__repr__` is exact
for ranges and layout, `state_indexing.py:1570-1593`), and the
`Property` hit/miss and "not cycles" parts of test 11, phrased as
spy-observed hit/miss; the key-level assertions of test 11 (for example
`Property(min_value=1, offset=0)` vs `offset=3` giving different keys)
use `cache_entry_key` and are expected to FAIL on master (no such
function):
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
   Two lambdas with DIFFERENT bodies defined on one line: distinct
   keys (two identical lambdas share code, name and qualname, so they
   share a key, correctly).
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
    `StateSpace`-based indexer: hits. Refused: a Python subclass of
    `np.float64` (alone and in a list), a `Property` subclass placed in a
    `property_dict` with the same triple, a standalone `Property` kwarg
    with an instance attribute set via `object.__setattr__`, a slot whose
    `name` differs from its key. Admitted and hit: an indexer with
    a deprecated `offset` Property, one built from a TUPLE of
    properties, `a + b` composition with an interleaved slot,
    `append(prefix_other=...)`, and the `add_epoch` / `joint_prob_graph`
    constructions. Refused: a Cython function in a closure cell, a
    `Slot` subclass in an indexer, a Property with an extra attribute
    (set via `object.__setattr__`), an indexer missing `_entity_order`.
    The allow-lists equal `set(vars(...))` of USED objects (after
    `index_to_props`), as §2.2.2 rule 8 (b) says (pins them against
    future attribute additions in `state_indexing.py`).
12. Values the key accepts that the old entry format could not store
    (`np.bool_`, complex, bytes keyword arguments) and an int above 4300
    digits held INSIDE a list kwarg and as a closure default (a top-level
    int kwarg is converted to float before the callback, `__init__.py:
    1745-1747`, which overflows with or without a cache): build, save and
    hit work. A module captured in a closure cell (local
    `import numpy as np` in a factory): hits.
13. (Every case below that calls `_record_build_stats(s0)` or alters
    the build record runs in a SUBPROCESS, which first deletes the record
    attribute from `phasic_pybind` and passes a constructed `s0`, because the record lives on the
    extension module for the whole process and `monkeypatch` cannot undo
    a stored "restart" failure, review R 6.)
    `callback_hash._phasic_build_id()` (the import-time content hash of
    §2.2.3) monkeypatched to another value: miss. With the recorded stat
    table monkeypatched (never touching installed files): a changed stat
    whose re-hash equals the import hash (only `ctime_ns` differs, the
    hard-link case) is ACCEPTED, the record refreshed, and the key
    unchanged; a changed stat with the re-hash helper monkeypatched to
    return different content: `GraphCacheKeyError` mentioning restart. A
    module file added to the directory listing (monkeypatched):
    `GraphCacheKeyError`. A recording failure stored by
    `_record_build_stats` (monkeypatched) and a missing record: both
    `GraphCacheKeyError` at the first key computation, while
    `import phasic` itself succeeded. A file change between the top-of-
    import snapshot and the end of import (monkeypatched snapshot):
    recording failure, `GraphCacheKeyError` at first use; a ctime-only
    difference between S0 and S1 (monkeypatched): accepted, keyed builds
    work. A file
    "changed" between the end-of-import hash and the final stat
    (monkeypatched, subprocess): recording failure. The content hash differs when bytes
    move across a file boundary (two temporary files, unit test of the
    hashing helper). `numpy.__version__` monkeypatched: miss.
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
    graphs and the tape restore at `__init__.py:1719` runs in formula
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
    `.tmp-x.partial` file whose mtime is two hours before the cache
    directory's mtime (both set with `os.utime`) is removed by
    `GraphCache().clear()`; one with the directory's mtime is kept;
    neither is counted by `get_cache_stats()`.
17. Validation before lookup: warm the cache for a `with_ipv`-decorated
    callback, then `Graph(cb, ipv=[9], cache_key=...)` raises the "ipv is
    ignored" `ValueError`.
18. POSITIVE CONTROLS for rule 8: a plain StateIndexer, fresh and after
    `index_to_props` (both generated classes present), a StateSpace-based
    one, and one built by `a + b`: all pass the checks and key (a check
    that refuses every indexer must fail this test). A registered class
    with `__init__` replaced, one with an added `__post_init__`, one whose
    `__bases__` were changed, and an unregistered same-field dataclass
    swapped in: refused. LEAK GUARD: build 1000 indexers and
    joint-probability base + reward indexer pairs with `index_to_props`
    and key computations, `gc.collect()`: the number of live registered
    classes stays bounded (not growing with N). An indexer round-tripped
    through `dill` in-process (dill rebuilds the generated class; skipped
    if dill is absent), and one dumped with `joblib.externals.cloudpickle`
    and loaded in a SUBPROCESS (an in-process cloudpickle round trip
    returns the same registered class and is admitted, correctly):
    refused with the rebuild advice. Not gated on a top-level
    `import cloudpickle`, which is not installed here (review AH 1).
    `dill.loads(dill.dumps(ix))` itself WORKS (fresh and used indexer;
    no regression for `%%srun` / pathos users), and the loaded copy is
    refused for keying. After `copy.deepcopy` of a dict holding
    `index_to_props` results (adds `__slotnames__`), a keyed build with
    the original indexer still hits. On the REAL install (subprocess, no monkeypatching), the
    end-of-import coverage check stores no recording failure, and a
    keyed build succeeds (guards against a check that refuses every
    normal install). A `StateIndexer` whose PropertySet `_props_dataclass` (and,
    separately, whose `_result_class`) was replaced by a same-field
    frozen dataclass with a shifting `__post_init__`: refused. A module
    whose `__name__` was set to a non-str: refused before any lookup.
    `np.int64` vs `np.longlong` arrays with equal bytes: different keys.
    An object whose class has a user metaclass defining `__hash__` /
    `__eq__` that mimic `np.float64`: refused, and neither method runs.
    A `StateIndexer` whose `property_dict` was replaced by a dict
    subclass: refused before `to_dict()` runs. kwargs `keys=[(f,),(g,)]`
    plus `scale={keys[0]: 1.0, keys[1]: 5.0}` (a key tuple first seen
    as a value): refused. `np.int64(3)` vs `np.longlong(3)`: different
    keys. A Python subclass of `np.void` with a `dtype` property and an
    `np.record` value: refused, the property never runs. Two
    equal-content functions differing only in `__name__` (one renamed by
    assignment), captured and branched on: different keys.
    A dict keyed by function objects (`{f: 1.0, g: 5.0}`) as a kwarg:
    `GraphCacheKeyError`. A `np.float64` subclass with a `dtype` property
    that records being called: refused, and the property never runs.
    `cache_key=3`, `cache_key=""`, `cache_key=np.str_("x")`,
    `Graph(3, cache_key="x")` raise (`np.str_`: `TypeError` advising
    `str(name)`); an object whose `__class__` property claims
    `np.float64` or `type` (a spoof): `GraphCacheKeyError`;
    `Graph(cb, ipv, "name")` (positional) raises `TypeError`.
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

26. Two captured partials identical in func/args/keywords but differing
    in a hand-set attribute the callback reads: different keys. A cache directory that does not exist under a read-only
    parent (skipped as root): `mkdir` fails, `OSError` naming
    it, callback not called. An EXISTING read-only cache directory
    (skipped as root): the pre-build writability probe raises the
    `OSError`, callback not called; a HIT from a read-only directory
    holding a valid entry works. Short writes: the save path's file
    object is constructed through a small internal factory
    (`graph_cache._open_entry_writer(fd)`), which the test replaces with
    a `BufferedWriter` over a `RawIOBase` whose `write` stores at most 7
    bytes per call and which implements `fileno()` (returning the real
    fd, so `os.fsync` works); the saved entry is complete and loads (patching
    `os.write` would test nothing, since `os.fdopen` text writes never
    call it, review S s9). A subprocess SCRIPT executed as `__main__` (so its
    callback's module is `__main__`) builds the graph, then a
    `multiprocessing` worker with start method `spawn` (which imports
    the script as `__mp_main__`) calls the same `Graph(cb, ...,
    cache_key=...)`: hit (callback not called in the worker); this is the
    case the normalisation exists for (a callback in the pytest module
    would not exercise it). A 60k-entry dict of state tuples as a
    kwarg (the step counter proves linearity; 600k would cost several
    seconds of suite time), and a 60k x 20 adjacency table whose neighbour tuples are the
    same objects as the keys, plus entries holding a literal tuple
    constant: all keyed (memoised sub-digests), with walk steps
    counted by an instrumented counter and asserted linear (no timing
    assertions). Hash-seed independence: a callback containing a set literal of
    string tuples plus a reuse of one element (`if key in {("a",1),
    ("b",2), ("c",3)}: if key == ("a",1)`) gives EQUAL digests in
    subprocesses under `PYTHONHASHSEED` 0-3; editing the reused element
    to `("c",3)` gives a DIFFERENT digest. A self-recursive nested
    function, two mutually recursive closures, a dispatch table entered
    at the container AND entered at a function (callback = `f2` whose
    cell holds `c2 = [f2]`), and a partial held in a list read by its
    function: all `GraphCacheKeyError` naming the cycle. The same
    recursive helper defined at module level: keyed (a global). In a SUBPROCESS (a reload inside the
    pytest process replaces `phasic.Graph` and breaks later tests,
    review Q p5): `importlib.reload(phasic)` after the package files
    change (monkeypatched re-hash) makes the next key computation raise
    `GraphCacheKeyError` (restart). 200 references to one 8 MB array
    in a kwarg: hashed once, asserted with an instrumented counter of
    array hashes (not timing). A tuple that is both a code constant and
    a default of the same function: keyed, hit. A C- and an F-ordered
    copy of one array as the kwarg: different keys. An `ESTALE` raised
    once by the load path's internal opener
    (`graph_cache._open_entry_reader(path)`, named for this purpose like
    `_open_entry_writer`): the entry is re-read and hits. Module positions: direct
    cell content keyed; module as a default, inside a container in a
    cell, and inside the ipv refused. The cache ROOT (parent of `graphs/`)
    deleted between lookup and save: the save succeeds. A recording
    failure for an extension outside the package directory
    (monkeypatched `__file__`, subprocess): `GraphCacheKeyError` with the
    regular-install message. `clear_all_graph_caches()` with an ABSENT
    legacy directory: no directory created, no error; with HOME
    read-only, `PHASIC_CACHE_DIR` unset AND `DEFAULT_CACHE_DIR` pointed
    explicitly at `<read-only home>/.phasic_cache/graphs` (otherwise the
    autouse fixture's tmp directory makes the case vacuous, review AA 3): `clear_all_graph_caches()`
    returns `{"removed": 0, "skipped": 0}` and `get_graph_cache_stats()`
    returns empty stats, neither raising nor creating a directory;
    `GraphCache().clear()` likewise; a legacy-directory entry written by
    the real `GraphCache.save` (not hand-made bytes) survives the sweep;
    `os.fsync` monkeypatched to raise `OSError(EINVAL)`: the save
    succeeds; with a legacy file
    whose first bytes are `{"format":"3"` : kept. `clear_all_graph_caches()` with
    `PHASIC_CACHE_DIR` set and `DEFAULT_CACHE_DIR` monkeypatched to a
    second tmp directory holding one old-format and one new-format entry
    file: the old one is deleted, the new one survives. `nan` vs `-nan`
    as a kwarg: different keys. A `np.longdouble` array: refused. A `StateIndexer` with a PropertySet
    named `dtype` as a kwarg: keyed normally. `ml_dtypes` arrays
    `float8_e4m3fn` vs `float8_e4m3b11fnuz` with equal bytes, and an
    `int4` scalar: `GraphCacheKeyError` (skipped if `ml_dtypes` is not
    installed). A `datetime64` array, a `timedelta64` array and a 0-d
    array: keyed, hit on rebuild, miss when one value differs. An indexer
    whose `_slots` attribute was deleted: `GraphCacheKeyError` (not
    `AttributeError`). A module passed as a kwarg and one inside the ipv:
    `GraphCacheKeyError`; a module in a closure cell: keyed by name (a
    contract-pin case: after `mod.RATE` changes, the hit returns the old
    graph, as documented). A function made by `exec(src, {})`
    (`__module__` is `None`): keyed. A copied entry file renamed to
    another call's entry name: `GraphCacheLoadError`. A JSON file with a
    non-entry name in the cache directory survives `GraphCache().clear()`.
    A shared-container DAG nested 40 deep: keyed, walk-step counter
    equal to 40 containers (memoised sub-digests). A `KeyboardInterrupt` raised inside `save()`
    (monkeypatched): propagates with `e.graph` attached, no `.partial`
    left. A `partial` subclass
    overriding `__call__`: refused. A user kwarg
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
  `cache_key`; `load_graph(callback)` at `:111` -> `load(digest)`, the
  digest obtained from `cache_entry_key(...)` for the same call (the
  file name holds only its first 32 characters). T6 (`:219-240`) is rewritten to exercise
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

Source change outside the cache modules: `state_indexing.py` gains the
additive registry (an `import weakref`, the `WeakSet`, and a module-level
wrapper rebinding `make_dataclass`; no existing function body changes).
`state_indexing.py` is CRLF, so the CRLF note below applies to it too.

Implementation note: `graph_cache.py`, `callback_hash.py`,
`state_indexing.py`, `model_export.py` and `config.py` use CRLF line
endings (`__init__.py` and `_graph_cache_mgmt.py` are LF); edits must preserve them (binary-mode edits, as
in bug 4) so the diff stays readable.

## 4. Gate

(Gate scripts live in `/Users/kmt/phasic/.git/claude-scratch/`.)
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
recording on its own, and the SAVE-time peak RSS at 53k (serialize()
builds lists of lists, then `.tolist()` and `json.dumps` add copies;
reported next to the load peak). Expectation stated in the docstring, with file
size and memory given separately: roughly linear; at 600k sparse
vertices an entry file of about 0.85 GB, ~35 s per load, and a load-time
process peak of about 5 GB (extrapolated from 0.8 GB at 53k, review B 8);
more for dense (two-locus) models. The 53k measurement is repeated on the
new format (compact JSON, per-field C encoder) and reported, and the
600k expectation is updated from it.
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
  user-defined `to_dict()` can cause. A bare `PropertySet` (a factory
  capturing `ps = indexer.lineage`, a PropertySet kwarg, or a bound
  method such as `indexer.lineage.index_to_props` in a cell) is refused
  too; pass the whole indexer instead. Recursive NESTED helpers
  (closures that call themselves or each other, dispatch tables held in
  closures) are refused; define them at module level.
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
  to share a cache directory only with trusted users. The more likely
  accident is two cooperating users or projects choosing the same
  `cache_key` for same-code callbacks that read different module
  globals: the second silently gets the first's graph (within the D1
  contract); the docstring advises project-unique key names and a
  per-user cache directory.

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
  `_deserialize_value` deleted, one `GraphCache(create=True)` per call).
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
- (Out of chronological order; belongs after review 4.) Review 5
  findings -> v2.2: 1 builtin arm §2.2.1; 2 classes and
  `to_dict()` errors §2.2.2 rules 7-8; 3 current-path cycle check and
  indexed back-references; 4 `fchmod` and read-error classes §2.6;
  5 test 15 rewritten; 6 version portability §2.2.1; 7 stat check per
  key computation §2.2.3, test 13; 8 this risk. Tests 11, 13, 15
  extended.
- Independent review G -> v3.3: 1 exact `FunctionType` / `MethodType`;
  Cython functions refused; 2-4 canonical check compares folded
  (name, min, max) triples as lists, specifies `_props_dataclass` /
  `_result_class`, checks Property `vars` and exact `Slot`, wraps
  exceptions; 5 bare PropertySet refusal in §6; 6 hash-randomised
  callbacks outside the contract; 7 pin test on used objects, review-5
  history moved out of §6, line 1670; 8 `_graph_cache_mgmt.py:21, 81-83`.
- Independent review H -> v3.4: 1 numpy scalars exact type only
  (`type(x) is x.dtype.type`); 2 the Property / Slot checks cover every
  reachable object (`property_dict` values, slots) and a standalone
  Property; 3 build id hashed per file with path and size framing,
  directory re-listed on every check; 4 `numpy.__version__` in the key,
  machine-type caveat documented; 5 common refusals listed in the
  docstring, DataFrame advice corrected; 6 build id owned by
  `callback_hash` (no import cycle); 7 offsets as a counted list after
  `to_dict()`; 8 `cache_key` keyword-only.
- Independent review I -> v3.5 (no ordinary-code wrong result found):
  1 a partial's `__dict__` keyed; 2 build-id stat tuple adds `ctime_ns`,
  inode, device; 3 rule 2 gated on `isinstance(np.generic)` before the
  exact check; 4 `mkdir` errors specified, `save()` recreates a removed
  directory; 5 order lists named; 6 §7 order note; docs list of common
  refusals extended (numba / `jax.jit` / `np.vectorize`); CRLF note.
- Independent review J -> v3.6 (no wrong-result case found): 1 build id
  = content hash computed at import; stat changes trigger a re-hash and
  are accepted when the bytes are unchanged (no spurious "restart" on
  hard-linked conda / pixi installs), a real change still raises;
  2 cache directory made absolute; 3 `f.__module__` keyed; 4 `clone()` /
  `copy()` and `graph_cache.py` docstrings in §2.7; 5 arrays hashed from
  a memoryview.
- Independent review K -> v3.7 (no ordinary-code wrong result found):
  1 only built-in numpy dtypes keyed (`ml_dtypes` refused); 2 arrays
  hashed via a uint8 view (datetime / timedelta / 0-d work); 3 exception
  wrapping covers all of the StateIndexer checks (a)-(c); 4 test 26
  unwritable-directory cases split; 5 wording: `clear_caches` scope,
  `.stem`, guard tests use spies, recursive build-id listing.
- Independent review L -> v3.8 (no wrong result or data loss found):
  1 more common refusals in the docstring (namedtuple, `np.memmap`,
  `Counter` class, `index_to_props` results); 2 guard tests phrased as
  spy-observed, key-level parts expected to fail on master; 3 `e.graph`
  attached to any exception from `save()`, including KeyboardInterrupt;
  4 no-eviction note; 5 line references updated (1669, 1753, 1750-1759,
  1745-1747); 6 entries opened with `encoding="utf-8"`.
- Independent review M -> v3.9: 1 modules admitted only in closure cells
  (refused in kwargs / ipv, where by-value keying is promised), contract
  says cell modules are keyed by name; 2 full digest stored in the entry
  and checked on load; 3 top-of-import stat snapshot closes the import
  window; 4 `__module__` via the value rule (can be `None`); exceptions
  defined in `callback_hash`; 5 array equality in the StateIndexer check
  specified; 6 `clear()` deletes only entry-named files; 7 total node
  budget for the key walk.
- Independent review N -> v4.0 (no wrong graph found): 1 save streams
  through a buffered file object (no truncated >2 GiB writes); 2
  writability probed before building on a miss; 3 `__mp_main__`
  normalised to `__main__`; 4 budget counts container re-entries only
  (10^6), so large plain lookup tables are keyed; 5 compact JSON
  (3.6x smaller than master's indent=2); 6 full 64-hex digest passed to
  `load` / `save`; 7 top-of-import snapshot finds the extension via
  `os.scandir` + `EXTENSION_SUFFIXES`, never `find_spec`; 8 module
  admitted only as a cell's direct content; §2.1 "one policy" wording.
- Independent review O -> v4.1 (no wrong result or data loss found):
  1 re-entry budget replaced by container back-reference tokens (linear,
  no budget, shared tuples keyed); 2 reinstall-overlap documented;
  3 spawn test runs a `__main__` script; 4 large fields encoded with the
  C encoder (`json.dumps`) and written piecewise; 5 exec / FunctionType
  namespaces are module globals (docs); 6 Cython `binding=False` under
  the installed-libraries clause; 7 more documented refusals; 8 numpy >=
  1.25 for keyed arrays.
- Independent review P -> v4.2: 1 container back-references replaced by
  content-only SUB-DIGESTS (hash-seed and sharing independent; memoised
  by id with strong references; linear); frozenset constants use them;
  functions' recursion via relative current-path back-references, with
  open subtrees not memoised; 2 build-id record anchored on the
  extension module, reload compares instead of re-recording; 3 one tag
  space per token kind, modules / StateIndexer never memoised; 4
  `to_dict()` fingerprinted before the canonical check; 5 cycle check
  before memo lookup; 7-8 docs (functions keyed by code everywhere,
  `__signature__` refusal).
- Independent review Q -> v4.3 (no wrong result or data loss found):
  1 re-walk cap (10^5) for dense recursion groups, "no budget" claim
  removed; 2 reload test in a subprocess; 3 partials / bound methods get
  back-references, container-mediated recursion refused and documented;
  4 closure-state misses documented; 5 walk-order sentence corrected;
  6 stale reload example replaced; 7 arrays and checked StateIndexers
  memoised, cost statements corrected; 8 frozenset sorted by token
  bytes, one shared memo, line 1719.
- Independent review R -> v4.4: 1 ALL cycles refused (functions
  included), so every sub-digest is pure content, memoised
  unconditionally, linear, with no re-walk cap (removes the entry-point
  dependence); 2 array layout token (C / F / strides); 3 kind tag at the
  head of every sub-digest, one tuple encoding for constants and
  values; 4 one re-read on `ESTALE`; 5 cap removed with the recursion
  machinery; 6 build-record tests in subprocesses; 7 legacy cache
  directory swept by `clear_all_graph_caches`; 8 counters instead of
  timing, §2.1 wording.
- Independent review S -> v4.5 (no stale hit found): 1 legacy sweep reads
  `DEFAULT_CACHE_DIR` at call time and deletes only non-format-2 files
  there (`format` written first), autouse fixture sets HOME; 2 floats
  keyed by IEEE bytes (NaN sign / payload); 3 short-write test via an
  injected raw writer; 4 code objects memoised, test sizes 60k, cost
  documented; 5 long-double kinds refused, set-order misses documented;
  6 accepted as stated.
- Independent review T -> v4.6 (no wrong result or data loss found):
  1 legacy sweep never creates the directory, deletes only files not
  starting with `{"format":`, per-file errors counted; 2 `save()` mkdir
  with `parents=True`, corrected reason; 3 editable installs refused
  with a specific message, documented; 4 strides keyed in bytes; 5 load
  path opener named for the ESTALE test.
- Independent review U -> v4.7 (no wrong result or data loss found;
  reviewer: "effectively at sign-off"): 1 management helpers resolve the
  directory without `mkdir`; 2 return types named (`int` for
  `clear()` / `clear_graph_cache()`, dict for `clear_all_graph_caches()`);
  3 `get_cache_stats` wording made consistent; 4 every extension module
  in the package hashed into the build id.
- Independent review V -> v4.8 (no realistic wrong result or data loss):
  1 every type gate uses `type(x)` / `issubclass(type(x), ...)`, never
  `isinstance`; 2 `cache_key` must be exactly `str`; 3 reload wording;
  4 multiprocessing globals note.
- Independent review W -> v4.9 (no realistic wrong result or data loss):
  1 end-of-import order fixed (hash, then final stat equal to the
  top-of-import snapshot; record = (S0, H)); 2 hit-path `dyn_ordering`
  read from a freshly created native graph; 3 line 485.
- Independent review X -> v5.0 (no wrong result or data loss found):
  1 hit-path flag via `_Graph(1).get_dyn_ordering()` (the property does
  not exist on the base); 2 `GraphCache(create=False)` by default,
  `create=True` only on the `cache_key` path; 3 shared `_ENTRY_HEADER`
  constant for writer and sweep; 4 `fsync` EINVAL / ENOTSUP / EROFS
  ignored; 5 exact `ModuleType` checks; 6 dtype instances listed among
  refusals.
- Independent review Y -> v5.1 (no wrong result or data loss found):
  1 `GraphCache(create=True)` on the cache_key path stated everywhere,
  `entry_path(digest)` named; 2 entry writer is binary (`"wb"`),
  header constant is bytes, test injects a binary writer; 3 large fields
  written in encoded 8 MiB slices, memory claim corrected; 4 missing
  `__self__` reads as None (ufuncs, dispatchers); 5 long-double arrays
  refused in rule 3; 6 immutable-type token requires `type(x) is type`;
  7 `clear()` count excludes `.partial` files.
- Independent review Z -> v5.2 (no ordinary-use wrong result or data
  loss): 1 identity-hashed dict keys refused, identity documented as not
  keyed; 2 numpy-scalar exact check reads no instance attribute;
  3 `_record_build_stats(s0)` signature and test-13 setup; 4 injected
  writer implements `fileno()`.
- Independent review AA -> v5.3 (no realistic wrong result or data loss):
  1 numpy scalars admitted only from a fixed frozenset of numpy's own
  classes (no `np.void` subclass or `np.record`); 2 `__name__` /
  `__qualname__` keyed, `__doc__` named as not keyed; 3 read-only-HOME
  test points `DEFAULT_CACHE_DIR` at the read-only home; 4 wording on
  `get_` / `set_dyn_ordering`.
- Independent review AB -> v5.4 (no realistic wrong result or data
  loss): 1 global `type(type(x)) is type` check before any hashed
  lookup on a type, identity allow-set tests; 2 StateIndexer internal
  container types pinned before `to_dict()`; 3 identity-key refusal is a
  separate predicate, not memoised; 4 array dtype gate stated
  explicitly against the rule-2 allow-set, stale numpy-version sentence
  removed; 5 `Path.resolve()` throughout; 6 scalar token includes the
  class name.
- Independent review AC -> v5.5 (no realistic wrong result or data
  loss): 1 generated indexer classes compared with the rebuild's (mro,
  attribute keys, method code), "every route" claim narrowed; 2 exact
  str dict keys pinned before hashing, module `__name__` type checked;
  3 `_bases` / `_radix_powers` are per-PropertySet; 4 array token
  carries the dtype class name; 5 loaded `phasic.*` modules must be
  among the hashed files; 6 lambda test wording, gate-script path.
- Independent review AD -> v5.6 (no wrong result or data loss): 1 the
  loaded-module coverage check falls back to `__file__` for pybind
  submodules (`__spec__ is None`), skips `None` entries, and a test runs
  it on the real install; 2 `.partial` age judged against the directory
  mtime, suspension case documented; 3 relative `PHASIC_CACHE_DIR`
  documented.
- Independent review AE -> v5.7 (no realistic wrong result or data
  loss): 1 generated-class check driven from the reference entries
  (full function fingerprints, non-function entries); 2 ctime-only
  change during import accepted; 3 `meta` keys specified; 4 snapshot
  placed after the `__future__` import; 5 distributed.ipynb citation
  corrected; 6 accidental shared-key collisions documented.
- Independent review AF -> v5.8 (no wrong result or data loss): 1 the
  generated-class check (which v5.7 made unsatisfiable) replaced by an
  additive registry in `state_indexing.py` (identity of every class
  entry against its creation-time snapshot), with positive controls in
  test 18; 2 save-time peak memory measured and stated; 3 positive
  control added.
- Independent review AG -> v5.9 (no wrong result or data loss; checks
  (a)-(c) and the registry measured to pass on real indexers): 1 registry
  leak fixed (snapshot on the class, WeakSet, module-level wrapper),
  leak-guard test; 2 cloudpickle / dill indexers refused with rebuild
  advice; 3 bases / name checked, in-place entry edits named outside the
  contract; 4 §3 source-change note and CRLF; 5 tests extended.
- Independent review AH -> v6.0 (no wrong result or data loss;
  reviewer would sign off once fixed): 1 pickled-indexer test uses dill
  in-process and joblib's cloudpickle across a subprocess; 2
  `clear_caches()` keeps its graph part on the legacy default directory,
  never reaching `PHASIC_CACHE_DIR`; 3 disjoint `py:` / `np:` scalar tag
  namespaces.
- Independent review AI -> v6.1 (no wrong result or data loss): 1 the
  snapshot is held in a `_Snap` object whose `__reduce__` yields an
  empty stub, so dill pickling of indexers keeps working (it broke for
  all users under v5.9); 2 `__slotnames__` ignored in the class
  key-set comparison; 3 CRLF file list.
