# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v4.4, 2026-11-25. Branch `fix/graph-cache-stale`, worktree
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
keyed by content). Functions ANYWHERE (callback, closure cells, defaults,
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
  - non-`str` -> `TypeError`, empty -> `ValueError`; `arg` not callable ->
    `ValueError` ("cache_key applies to graphs built from a callback");
  - ALL argument validation for a callback runs first: the
    ipv/decorator checks now at `:1750-1759` move above the lookup (for
    every call, cached or not; same errors, raised earlier). They read
    `getattr(arg, "__name__", None)` instead of `arg.__name__`
    (`:1669`, `:1753`), so a `partial` callback gets the normal ipv
    error instead of today's `AttributeError` (review E 7d);
  - the ENTRY KEY (§2.2) is computed once. A key error propagates:
    nothing is built, nothing written;
  - lookup: `cache = GraphCache()` once per call, then
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
    `self.dyn_ordering = os.environ.get("PHASIC_DYN_ORDERING") is not
    None`, mirroring the C rule applied at graph creation
    (`phasic.c:2996`), because `serialize()` persists the WRITER's flag
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
  review N p3) (it can be `None`, for example for functions made by
  `exec(src, {})` or `types.FunctionType(code, {})`, review M m5; the
  same applies to every other name-like token, and `cache_key` itself
  goes through the `str` value rule) (so two same-bodied callbacks
  from different modules, each reading its own module global, are kept
  apart even under one `cache_key`; it is `'__main__'` in notebooks, so
  notebook hits are unaffected; `co_filename` stays excluded); CODE
  FINGERPRINT of `f.__code__`; VALUE
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

1. EXACT types only (`type(x) is ...`): `None`, `bool`, `str`, `bytes`:
   type name + canonical repr; `int` via `hex()` (no 4300-digit `repr`
   limit); `float` via `float.hex`; `complex` via the hex of both parts.
   Subclasses do not match here: `np.float64` (a `float` subclass) goes
   to rule 2, and `IntEnum` members and other subclasses are refused;
2. numpy scalar: gated on `isinstance(x, np.generic)` FIRST (so `.dtype`
   is never read from an arbitrary object, e.g. a `StateIndexer` whose
   PropertySet is named `dtype`, or an object whose `dtype` property
   raises), then EXACT type only (`type(x) is x.dtype.type`; a Python
   subclass of a numpy scalar can override arithmetic and carries a
   `__dict__`, review H p6, and is refused), with a BUILT-IN numpy dtype
   (`type(x.dtype).__module__ == "numpy.dtypes"`; user-defined dtypes
   such as `ml_dtypes`' float8 / int4 / bfloat16, installed alongside
   jax, all report `dtype.str` `<V1` / `<V2`, so two of them with equal
   bytes would collide, review K k3, and are refused):
   `x.dtype.str` + `tobytes()`;
   `np.void` scalars and any dtype with `.fields` or `.hasobject` are
   refused;
3. `type(x) is np.ndarray` exactly (masked arrays, `np.matrix` and other
   subclasses are refused): `x.dtype.str`, shape, a LAYOUT token (`C`
   if C-contiguous, else `F` if F-contiguous, else `S` followed by the
   strides divided by the itemsize; a C- and an F-ordered copy of one
   array give different last-bit results in reductions such as
   `sum(axis=1)` and different `ravel('K')` order, review R p5;
   `DataFrame.to_numpy()` returns F-ordered arrays), C-order bytes (hashed
   from `memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))`,
   which copies only non-contiguous arrays and also works for datetime /
   timedelta kinds `M` / `m`, which numpy's buffer protocol rejects, and
   for 0-d arrays, via `reshape(-1)`; not via `tobytes()`). Refused: a
   dtype that is not built in (the same `numpy.dtypes` gate as rule 2),
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
7. a class object (`isinstance(x, type)`): the immutable-compiled-type
   token of §2.2.1 if it qualifies; otherwise NOT FINGERPRINTABLE (so a
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
     (a) `type(ix) is StateIndexer`, every PropertySet has
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
     triples for `property_dict.items()` with their keys, and `_bases`
     and `_radix_powers` must equal those of `c`, where equal for these
     numpy arrays means same `type`, `dtype` and `shape` and
     `np.array_equal` (a bare `!=` would raise on multi-element arrays,
     and value-only equality would admit a float array with different
     behaviour); `_result_class`'s field
     names must equal `c`'s; `_props_dataclass`, if present (it is
     created lazily, so `c` never has it), must have field names
     `[p.name for p in ps.properties]`. Each Property must have
     `set(vars(p))` equal to its four dataclass fields (a frozen
     dataclass still has a `__dict__`, so extra attributes are
     otherwise invisible; same check as (a)), and each slot is checked as
     in (a) (`state_indexing.py:193`; subclasses are accepted by the
     constructor at `:855-862`). Any exception raised
     while running ANY of (a), (b) and (c) (for example `AttributeError` for a missing
     attribute on an old pickled indexer, since `__getattr__` raises for
     `_` names, `:996-998`) is re-raised as `GraphCacheKeyError` (path
     chained). `_result_class` / `_props_dataclass` are generated classes
     checked by field names only; their other behaviour is phasic code
     covered by the build id. This
     catches every route by which `to_dict()` and behaviour drift apart:
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
   dicts), functions whose `__dict__` holds an `inspect.Signature`
   (set by some decorator libraries as `__signature__`), numpy 2
   `StringDType` arrays, `fractions.Fraction`,
   `decimal.Decimal`, nesting deeper than 100. The docstring also says
   that a function made with `exec(src, ns)` or `types.FunctionType(code,
   ns)` reads its globals from `ns`, which counts as MODULE GLOBALS
   (not keyed; change `cache_key` when `ns` changes), and that numpy >=
   1.25 is required for keyed arrays (the built-in dtype gate uses
   `numpy.dtypes`; without it every array and numpy scalar is refused
   with a message naming the requirement). The docstring also says that entries
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
method, partial, array, indexer) and in which its children are in turn
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
check, (2) memo lookup, (3) walk. Modules (direct cell content only) are
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
(`__init__.py:279`), a call `callback_hash._record_build_stats()` at the
end of `phasic/__init__.py` records ONCE PER PROCESS (the record is
stored as an attribute on the `phasic_pybind` extension module, the one
object `importlib.reload` can never replace; a second call, from
`importlib.reload(phasic)` or a reload of `callback_hash`, COMPARES the
files against that record instead of replacing it, and stores a
"restart" failure if they differ, because the old code keeps running
after a reload, review P p2), for the extension module located by
`phasic.phasic_pybind.__file__` (not by globbing; no `__file__` or a
missing file is a recording failure, so an editable or source-tree layout
can never silently leave the extension out of the id) and every `*.py`
file under the package directory of `phasic.__file__`, RECURSIVELY,
excluding `__pycache__` (the package is flat today, 52 files, but a
future subpackage must not fall out of the id silently):
(i) a stat record `(path, size, mtime_ns, ctime_ns, ino, dev)` per file,
and (ii) the CONTENT HASH of those files, computed right there, at import
(measured 11.9 ms for 52 .py + the .so, 3.49 MB, review 4; paid by every
`import phasic`, accepted as the price of an unambiguous id). The content
hash uses the same counted, length-prefixed framing as the key: for each
file in sorted relative-path order its relative path, its size and its
bytes, so moving text across a file boundary (an extraction refactor)
changes it. This content hash IS `_phasic_build_id()`. To make sure it
describes the code the process actually imported, a stdlib-only stat
snapshot is taken at the TOP of `phasic/__init__.py`, before any
submodule or `phasic_pybind` import: `os.scandir` of the package
directory for `*.py` (recursive, as above) and for `phasic_pybind*`
files ending in one of `importlib.machinery.EXTENSION_SUFFIXES` (never
`importlib.util.find_spec`, which would call every `sys.meta_path`
finder and trip test 14's import hook); at the end of import,
`phasic.phasic_pybind.__file__` must be among the snapshotted files
(otherwise a recording failure is stored); at the end the files are
re-stat'ed, and if anything changed during the import (a concurrent
reinstall on a shared cluster), a recording failure is stored (raised at
the first key computation). So no reinstall during or after import can
go unnoticed (review I 2, review M 3).
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
and `GraphCacheKeyError` as above. So a harmless metadata change never
produces a spurious "restart" error, and a real one is always caught.
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
and makes it ABSOLUTE (`Path.absolute()`), so a `chdir` by the callback
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
`monkeypatch.setenv("PHASIC_CACHE_DIR", <tmp>)`, which also redirects
the C-side caches that `clear_from_cache` touches (§3). `get_cache_dir`, `get_graph_cache_stats`,
`print_graph_cache_info`, `clear_all_graph_caches` and
the GRAPH-CACHE part of `model_export.clear_caches` use the same
resolution (its `clear_model_cache` part still `rmtree`s
`$HOME/.phasic_cache` regardless of `PHASIC_CACHE_DIR`,
`model_export.py:72`; pre-existing, found-defects item P, unchanged
here).

### 2.6 `GraphCache` API and entry file

- Kept: `GraphCache(cache_dir=None)`, `clear()`, `clear_graph_cache()`,
  `get_cache_stats()`. The constructor's `mkdir(parents=True,
  exist_ok=True)` (`graph_cache.py:88-89`) failing re-raises the same
  `OSError` subclass with a message naming the directory and the ways
  out (another directory via `configure(cache_dir=...)`, or no
  `cache_key`); this happens before the lookup, so nothing is built.
  `save()` repeats `mkdir(exist_ok=True)` before `os.open`, so a
  directory removed during a long build (for example by
  `phasic.clear_caches()`, `model_export.py:120-138`) is recreated
  rather than failing after the build. EDITED: `_serialize_value`
  (`graph_cache.py:304-378`) loses its generic `to_dict()` arm (which tags
  any object with `__type__: module.Class`) and its docstring/error text
  advertising the `to_dict()`/`from_dict()` protocol; it encodes plain
  JSON values plus the `ndarray` tag only and raises on anything else, so
  the save side is symmetric with the restricted decoder and no entry
  can be written that every later lookup would refuse. Deleted:
  `_deserialize_value` (see below). `Graph.__init__` constructs ONE `GraphCache()` per
  call, whose directory is absolute (§2.5), so lookup and save use the
  same directory.
  `clear()` / `clear_graph_cache()` delete ONLY files whose names match
  `^[0-9a-f]{32}\.json$` (entry names, old and new; the directory may now
  be user-chosen via `PHASIC_CACHE_DIR`, so other JSON files there must
  survive, review M 6); `get_cache_stats` counts the same pattern. They
  (and through them `clear_all_graph_caches` and
  `model_export.clear_caches`) also remove
  leftover `.tmp-*.partial` files, which a writer killed between
  `os.open` and `os.replace` (SIGKILL, OOM kill, SLURM time limit) leaves
  behind (review 7 C), but ONLY those whose mtime is more than one hour
  old, so a concurrent writer's in-flight file in a shared directory is
  never removed under it (review 8); `get_cache_stats` still counts only `*.json`.
- Replaced: `save_graph(graph, callback, **params)` /
  `load_graph(callback, **params)` -> `save(graph, digest, meta)` /
  `load(digest)` (full 64-character digest; file name = first 32).
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
  `with os.fdopen(fd, "w", encoding="utf-8") as f:` write the entry's
  small fields, then EACH large `graph_data` field encoded with
  `json.dumps(value, separators=(",", ":"))` (the one-shot call uses
  CPython's C encoder; `json.dump` streaming always uses the
  pure-Python encoder, 2.8x slower, review O p6) and written piecewise
  into the same JSON object, then `f.flush(); os.fsync(f.fileno())`
  (memory stays bounded per field), then
  `os.replace` to `<first 32 hex>.json`. The buffered writer loops over
  short writes (a single `os.write` of more than 2 GiB writes only
  `INT_MAX` bytes on macOS and about 2 GiB on Linux, so a natural
  `os.write(fd, data)` would store a truncated entry and report success,
  review N p8), and streaming avoids holding the string and its bytes at
  once. Compact separators replace master's `indent=2`
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
  (shared SLURM directories are a documented use,
  `distributed.ipynb` cell 31).
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
the current graph directory AND from the legacy default
`~/.phasic_cache/graphs` (master always wrote there, ignoring
`PHASIC_CACHE_DIR`, `graph_cache.py:32, 88`), so users who set
`cache_dir` do not keep orphaned old entries (docstring says so).

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
13. (Every case below that calls `_record_build_stats` or alters the
    build record runs in a SUBPROCESS, because the record lives on the
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
    recording failure, `GraphCacheKeyError` at first use. The content hash differs when bytes
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
    `.tmp-x.partial` file with an mtime two hours old is removed by
    `GraphCache().clear()`; one with a current mtime is kept; neither is
    counted by `get_cache_stats()`.
17. Validation before lookup: warm the cache for a `with_ipv`-decorated
    callback, then `Graph(cb, ipv=[9], cache_key=...)` raises the "ipv is
    ignored" `ValueError`.
18. `cache_key=3`, `cache_key=""`, `Graph(3, cache_key="x")` raise;
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
    holding a valid entry works. `os.write` / the file object's `write`
    monkeypatched to perform short writes: the saved entry is complete
    and loads. A subprocess SCRIPT executed as `__main__` (so its
    callback's module is `__main__`) builds the graph, then a
    `multiprocessing` worker with start method `spawn` (which imports
    the script as `__mp_main__`) calls the same `Graph(cb, ...,
    cache_key=...)`: hit (callback not called in the worker); this is the
    case the normalisation exists for (a callback in the pytest module
    would not exercise it). A 600k-entry dict of state tuples as a
    kwarg, and a 60k x 20 adjacency table whose neighbour tuples are the
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
    once by a monkeypatched `open`: the entry is re-read and hits. Module positions: direct
    cell content keyed; module as a default, inside a container in a
    cell, and inside the ipv refused. The cache directory deleted between lookup
    and save: the save succeeds. A `StateIndexer` with a PropertySet
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

Implementation note: `graph_cache.py` (and other files in this repo)
use CRLF line endings; edits must preserve them (binary-mode edits, as
in bug 4) so the diff stays readable.

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
  to share a cache directory only with trusted users.

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
