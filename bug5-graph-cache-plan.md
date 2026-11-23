# Bug 5: the on-disk graph cache returns a stale graph — fix plan

**Status: v1, 2026-11-23. Branch `fix/graph-cache-stale`, worktree
`/Users/kmt/phasic-bug5` (own pixi env). No source changed yet. Backward
compatibility is not required (user decision). Python-only change.
Independent of bugs 1-4 in code; gated after them (rebased first).**

Evidence on unmodified code, with HOME redirected to a scratch directory
(the real `~/.phasic_cache` is never read): `experiments/bug5_graph_cache/`
(`b1_repro.py`, `b1.log`, `b2_value.py`, `b2.log`).

## 1. The defect (read from source + measured)

`Graph(callback, ...)` looks the graph up on disk before building it
(`src/phasic/__init__.py:1680-1727`) and saves it afterwards (`:1790-1800`).
The cache is ON by default (`config.py:611`, `graph_cache: bool = True`).
The key (`src/phasic/callback_hash.py:33-170`) is a hash of: a version tag,
the Python version, the initial state (ipv), the callback's own source
(AST), and `repr()` of the keyword arguments.

Everything else the callback reads is missing from the key:

| the callback depends on | changed between two builds | result (b1.log) |
|---|---|---|
| a module-level variable | 1 -> 10 | stale: 0.833333, correct 0.083333 |
| a variable captured from an enclosing function | 1 -> 10 | stale |
| a helper function it calls | body changed | stale |
| a closure, passed with `ipv=` | 1 -> 10 | stale |

The entry is on disk, so the wrong graph is also returned in later
processes and after the code has been corrected back and forth. No
message is produced. A closure check exists (`_detect_closures`,
`callback_hash.py:219`) but is skipped for every callback that reaches the
cache through `Graph(...)`, because `Graph.__init__` wraps the callback
first (`__init__.py:1667-1671`) and the hash skips the check for wrapped
callbacks (`callback_hash.py:110-112`).

Further faults in the same path:
- keyword arguments are keyed by `repr()`; an object whose repr contains a
  memory address only produces a warning (`callback_hash.py:146-158`), and
  a long numpy array's repr is abbreviated ("...") so two different arrays
  can share a key;
- a failed key computation is swallowed: `load_graph` returns None
  (`graph_cache.py:183-187`) and the save failure is caught in
  `Graph.__init__`, so `graph_cache=True` can silently do nothing;
- the key does not identify the phasic build, so a rebuilt library with an
  unchanged version number reloads graphs written by the old one.

What the cache is worth (b2.log): build vs. load 0.076 s vs 0.026 s at
1,003 vertices, 0.387 vs 0.128 at 1,166, 3.47 vs 0.98 at 5,823 — about
3.5x on construction, and a first build that is ~45% slower because it
also writes the entry.

## 2. Fix

**Principle. A cached graph may be returned only when the key covers
everything the build can depend on, as far as that can be established by
inspection. When it cannot be established, caching that graph is refused
with an error that says why. Because inspection can never cover every
Python program (see residuals, §2.5), caching is something the user asks
for, not a default.**

### 2.1 Caching becomes opt-in

`graph_cache` defaults to `False` in the configuration (`config.py:611`
and the documentation at `:549-553`); `Graph(..., graph_cache=True)` and
`phasic.configure(graph_cache=True)` turn it on;
`PHASIC_DISABLE_GRAPH_CACHE=1` still forces it off.

### 2.2 A dependency fingerprint replaces the source-only key

`callback_hash.py` is rewritten around one function,
`fingerprint_callback(callback, **params) -> str`, which hashes:

1. a key-format version (new value, so no existing entry can match);
2. the Python version;
3. the phasic build: `phasic.__version__` plus size and modification time
   of the installed `phasic_pybind` extension file and of
   `phasic/__init__.py` (a re-install during development invalidates the
   cache);
4. the initial state (ipv), by value fingerprint;
5. the callback, by FUNCTION FINGERPRINT;
6. every keyword argument, by VALUE FINGERPRINT (replacing `repr()`).

FUNCTION FINGERPRINT of a Python function `f` (recursive, with a visited
set so mutual recursion terminates):
- the normalised AST of its source, as today;
- the value fingerprints of `f.__defaults__` and `f.__kwdefaults__`;
- for every free variable (`f.__code__.co_freevars` with `f.__closure__`):
  name and value fingerprint of the cell's content;
- for every global the code can load — the `LOAD_GLOBAL` / `LOAD_NAME`
  operands of `f.__code__` and of every code object nested in it
  (`dis.get_instructions`) — that resolves in `f.__globals__`: name and
  value fingerprint. Names that resolve only in builtins contribute their
  name. A name that resolves nowhere contributes "unresolved".

VALUE FINGERPRINT of an object, by exact type:
- `None`, `bool`, `int`, `float`, `complex`, `str`, `bytes`: type name and
  `repr` (floats via `float.hex`);
- `tuple`, `list`: element fingerprints in order; `dict`: (key, value)
  fingerprints sorted by key fingerprint; `set`, `frozenset`: sorted
  element fingerprints;
- numpy array or scalar: dtype, shape and raw bytes (object dtype refused);
- Python function: its function fingerprint; `functools.partial`: function,
  args, keywords;
- module: its name and `__version__` if any — allowed only for an
  INSTALLED module (file under the interpreter's `sys.prefix` or the
  standard library, or a built-in);
- class: qualified name — allowed only if defined in an installed module;
- instance of a class defined in an installed module (for example a
  `StateIndexer`): class qualified name plus `pickle.dumps` bytes;
- anything else — a module or class from the user's own files, an instance
  of a user-defined class, a generator, an open file, a lock, a bound
  method of such an object, anything that fails to pickle — is NOT
  FINGERPRINTABLE.

The phasic `_callback(ipv)` wrapper is recognised as today and the wrapped
function fingerprinted; `_detect_closures` is deleted (closures are now
fingerprinted, not rejected).

### 2.3 Not fingerprintable means an error, not a silent skip

`fingerprint_callback` raises `GraphCacheKeyError` (a `ValueError`
subclass, exported) naming the callback, the dependency path (for example
`rates -> helper -> CONFIG`), the offending type, and the two ways out:
pass the value as a keyword argument / make it a plain value, or build
with `graph_cache=False`. `Graph.__init__` lets it propagate: with
caching requested, an un-keyable callback fails before anything is built.
`GraphCache.load_graph` and `save_graph` no longer swallow key errors.
A corrupt or unreadable cache FILE still logs a warning and rebuilds (the
key was sound; only the stored copy is bad), and the rebuilt graph
overwrites it.

`Graph.clear_from_cache(graph_cache=True)` (`_graph_cache_mgmt.py:86-91`)
uses the same fingerprint.

### 2.4 Entries written before this change

The new key-format version means none can be loaded. They are left on
disk (the code never deletes user files unasked);
`phasic.clear_all_graph_caches()` removes them. Stated in the docstring.

### 2.5 Residual cases the fingerprint cannot see (documented)

Stated in the `graph_cache` documentation as the contract for opting in:
the build must be a deterministic function of the callback's arguments
and of the values described in §2.2. Not detected: values reached only
through attribute access on an installed module or object that the
program mutates at run time (`np.some_global = ...`), names looked up
dynamically (`globals()[name]`, `getattr` with computed names, `eval`),
files, environment variables and random numbers read inside the callback,
and edits to an installed third-party library that keep its version. With
the cache off — the default — none of this matters.

## 3. Tests (new file `tests/pytest/test_graph_cache_fingerprint.py`)

Every test redirects HOME / the cache directory to `tmp_path` and passes
`graph_cache=True` explicitly. "Stale" is judged on a model quantity
(`expectation()`), compared with a `graph_cache=False` build.

Fail on master:
1. Module-level variable changed between two builds (plain callback and
   `ipv=` form): second build equals the uncached build.
2. Captured variable changed (callback defined in a factory function,
   called with two values): same.
3. Helper function replaced by one with a different body; helper of a
   helper changed; mutual recursion between two helpers terminates.
4. Default argument bound to a module variable (`def cb(state, n=N)`)
   changed.
5. Keyword argument that is a long numpy array differing in one middle
   element (abbreviated repr identical): different graphs.
6. A dependency that is an instance of a class defined in the test module:
   `Graph(..., graph_cache=True)` raises `GraphCacheKeyError` whose message
   names the variable and the class; nothing is written to the cache
   directory (today: builds, silently uncached or stale).
7. Default: with no `graph_cache` argument and default configuration, two
   builds write no file under the cache directory.
8. Phasic-build component: with the build-identity function monkeypatched
   to a different value, an existing entry is not loaded.

Guards (pass on master and after):
9. Unchanged callback and dependencies, `graph_cache=True`: second build
   is a cache hit (asserted by spying on the build callback: not called)
   and gives the same `expectation()`, vertex count and `serialize()`
   arrays as the first; holds across a subprocess (two `python -c` runs
   sharing the cache directory, hash randomisation seeds 0 and 1).
10. Formatting-only edits (comment, docstring, whitespace) still hit.
11. A dependency on `numpy`/`math` functions and on a `StateIndexer`
    instance is fingerprintable (hit on the second build); changing the
    indexer's layout misses.
12. A corrupt cache file: warning, rebuild, correct result, file
    rewritten.
13. `PHASIC_DISABLE_GRAPH_CACHE=1` with `graph_cache=True`: no file
    written, no error even for an un-keyable callback (cache is off).
14. Existing behaviours kept: the cache-hit graph keeps its callback
    (`extend`, `add_epoch` work) — the two tests in
    `tests/pytest/test_graph_cache.py::TestCacheHitPreservesCallback`,
    which are edited only to pass `graph_cache=True` if they relied on the
    default.

Existing tests that rely on the default being on, or on the closure
error text, are found by the gate (§4) and by grep before implementing
(`graph_cache`, `hash_callback`, `closures` in `tests/pytest/`); each
edit is listed in the plan before the implementation review.

## 4. Gate

Same tools and rules as the other fixes: rebase, `pixi run install-dev`,
`gate2.sh`, `gate_compare.py --new-file
tests/pytest/test_graph_cache_fingerprint.py`, baseline = the previous
fix's gate run, three isolated reruns for any regression. The new tests
are also run on the master build and the outcome recorded. Every gate
process runs with HOME as it is today (the suite's own configuration);
the new tests never touch the real cache.

Additional check, because the default changes: total wall time of the
gate before and after is recorded (construction is no longer cached by
default), and the b2 timings are re-measured with `graph_cache=True` to
record the cost of fingerprinting (target: under 5% of the load time at
1,003 vertices; if not met, reported, not hidden).

## 5. Notebooks (not edited — list for the user)

18 notebook cells pass `graph_cache=True` and 3 pass `graph_cache=False`
(`docs/pages/tutorial/`: mpfr, svgd-multi-param, svgd-learning-rate,
configuration, state-space, svgd-basics, scc-decomposition,
python_trace-and-jax-caching, svgd-joint-prob, matrix-solve). Before the
merge each `graph_cache=True` cell's callback is fingerprinted by a script
(extracting the cell sources into a scratch directory, not executing the
notebooks in place) and the cells that would now raise are listed in the
plan with the reason. Text that describes the cache as on by default is
listed too.

## 6. Risks

- Construction is no longer cached unless requested: repeated builds of
  the same model across processes take the full build time (3.5 s instead
  of 1 s at 5,823 vertices).
- A callback that was cached before (silently, possibly wrongly) may now
  be refused when caching is requested, with an explicit error.
- The fingerprint walks the callback's dependencies on every cached
  build; a callback that references a very large array pays for hashing
  it (measured in the gate).
- Pickle bytes of an installed-class instance that contains a set of
  strings vary with hash randomisation: a miss, never a stale hit
  (guard 9 runs under two seeds for the supported cases).
