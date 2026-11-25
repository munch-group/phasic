"""
Entry keys for the on-disk graph cache (``Graph(..., cache_key=...)``).

The user names a cache entry with ``cache_key``. :func:`cache_entry_key`
adds to that name everything phasic can read EXACTLY from the call:

- the cache format, the Python minor version, the phasic build (a content
  hash of the installed package's ``*.py`` and extension files, recorded at
  ``import phasic``) and ``numpy.__version__``;
- the callback function object as passed to the C builder: its running
  bytecode, constants, defaults, ``__dict__``, closure contents and bound
  ``self``, recursively (functions are keyed by their code, never by the
  module globals they read);
- the keyword arguments and the initial probability vector, BY VALUE.

Only plain values are keyed automatically: ``None``, ``bool``, ``int``,
``float``, ``complex``, ``str``, ``bytes``, numpy scalars and arrays of
numpy's own numeric/str/datetime dtypes, exact ``list`` / ``tuple`` /
``dict`` containers of those, ``Ellipsis`` / ``slice``, functions /
bound methods / ``functools.partial`` objects, compiled functions that
resolve back to themselves by name, immutable compiled types (for example
``itertools.combinations_with_replacement``), a module held DIRECTLY in a
closure cell (keyed by name) and phasic's own ``StateIndexer`` /
``Property`` in their plain form. Anything else raises
:class:`GraphCacheKeyError`. What a callback reaches through module
globals, installed libraries, files or the environment is the user's
responsibility: change ``cache_key`` when it changes.

The key is the full SHA-256 hex digest of a prefix-free token stream:
every token is ``<tag>:<len>:<payload>``, every composite carries its
element count before its elements, and every container, function, code
object, array and indexer is represented in its parent's stream by the
SHA-256 of its own stream (a "sub-digest", memoised by ``id()`` within
one key computation). Cycles are refused.

This module imports nothing from phasic at module level (it is imported
before ``OMP_NUM_THREADS`` is set and must not load ``phasic_pybind``)
and uses a stdlib logger only.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.machinery
import logging
import os
import struct
import sys
import types
from typing import Any

import numpy as np

_logger = logging.getLogger("phasic.callback_hash")

# Version of the entry key composition AND of the entry file format.
# Replaces the old callback version tag; entries written under any other value
# are never loaded.
GRAPH_CACHE_FORMAT = "2"

# Nesting limit for every recursive arm of the walk.
_MAX_DEPTH = 100

# Name of the attribute on the phasic_pybind extension module that holds
# the build record (the one object importlib.reload can never replace).
_RECORD_ATTR = "_phasic_build_record"

_WAYS_OUT = (
    "Ways out: pass plain values the callback needs (numbers, strings, "
    "lists, dicts, numpy arrays; for example model.to_dict() instead of "
    "model), move the value to a module global and change cache_key "
    "yourself whenever it changes, or build without cache_key."
)

_RESTART = (
    "phasic was reinstalled after it was imported; restart the interpreter. "
    "Keyed runs (Graph(..., cache_key=...)) must not overlap a reinstall of "
    "the same environment (for example `pixi run install-dev`)."
)


class GraphCacheKeyError(ValueError):
    """The entry key for ``Graph(..., cache_key=...)`` cannot be computed.

    The message names the path to the offending value, its type and the
    ways out (pass plain values, move the value to a module global and
    change ``cache_key`` yourself, or build without ``cache_key``).
    Nothing is built and nothing is written when it is raised.
    """


class GraphCacheLoadError(RuntimeError):
    """A graph cache entry exists but cannot be used.

    The message names the file and the ways out (delete the file, call
    ``phasic.clear_all_graph_caches()``, or use another ``cache_key``);
    the original exception, if any, is chained.
    """


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
# Tag strings never contain a ':' followed by a digit-only remainder that
# could be read as another tag + length, and no tag equals a ':'-delimited
# prefix of another tag, so <tag>:<len>:<payload> streams are uniquely
# decodable. Python-scalar tags use the 'py:' namespace and numpy-scalar
# tags the 'np:' namespace.

def _tok(tag: str, payload: bytes) -> bytes:
    return b"%s:%d:%s" % (tag.encode("ascii"), len(payload), payload)


def _count(n: int) -> bytes:
    return _tok("count", b"%d" % n)


def _str_payload(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


_NONE_TOK = _tok("py:none", b"")
_TRUE_TOK = _tok("py:bool", b"True")
_FALSE_TOK = _tok("py:bool", b"False")
_ELLIPSIS_TOK = _tok("py:ellipsis", b"")
_EMPTY_CELL_TOK = _tok("empty", b"")


def _scalar_np_types() -> frozenset:
    """numpy's own scalar classes, minus void, object and long double."""
    excluded = {np.void, np.object_, np.longdouble, np.clongdouble}
    out = set()
    for t in set(np.sctypeDict.values()):
        if type(t) is type and t.__module__ == "numpy" and not any(
                t is x for x in excluded):
            out.add(t)
    return frozenset(out)


_NP_SCALAR_TYPES = _scalar_np_types()
_NP_DISPATCHER_TYPE = type(np.sum)
_BUILTIN_TYPES = (types.BuiltinFunctionType, np.ufunc, _NP_DISPATCHER_TYPE)
_IMMUTABLETYPE_FLAG = 1 << 8   # Py_TPFLAGS_IMMUTABLETYPE (3.10+)
_MISSING = object()

# Allow-lists of instance attribute names (§2.2.2 rule 8 (b)): the
# attributes of a freshly built object plus the documented lazily added
# ones. Pinned against set(vars(...)) of USED objects by
# tests/pytest/test_graph_cache_key.py.
_INDEXER_ATTRS = frozenset({
    "_property_sets", "_slots", "_offsets", "_pset_order", "_slot_order",
    "_entity_order", "_total_state_length", "_result_class",
})
_PSET_ATTRS = frozenset({
    "name", "properties", "property_dict", "_bases", "_radix_powers",
    "_props_dataclass",   # lazily added by PropertySet._props_class()
})


def _type_name(t: Any) -> str:
    """Qualified name of a type, read through type's own descriptors (no
    metaclass __getattribute__ can run)."""
    try:
        qn = type.__dict__["__qualname__"].__get__(t, type)
        mod = type.__dict__["__module__"].__get__(t, type)
        if type(qn) is str and type(mod) is str:
            return f"{mod}.{qn}" if mod != "builtins" else qn
    except Exception:
        pass
    return "<unknown type>"


_PHASIC_TYPES = None


def _phasic_types():
    """(Property, PropertySet, Slot, StateIndexer, StateSpace), imported
    lazily (state_indexing imports only numpy)."""
    global _PHASIC_TYPES
    if _PHASIC_TYPES is None:
        from .state_indexing import (Property, PropertySet, Slot,
                                     StateIndexer, StateSpace)
        _PHASIC_TYPES = (Property, PropertySet, Slot, StateIndexer, StateSpace)
    return _PHASIC_TYPES


# Last walk's instrumentation counters (tests assert linearity with them).
_LAST_WALK_STATS: dict[str, int] = {}


class _KeyWalker:
    """One key computation: memo, current path, depth and counters."""

    def __init__(self) -> None:
        self._memo: dict[int, tuple[Any, bytes]] = {}
        self._path: dict[int, str] = {}
        self._labels: list[str] = []
        self._depth = 0
        self.stats: dict[str, int] = {"steps": 0, "array_hashes": 0}

    # -- errors ------------------------------------------------------------
    def _where(self) -> str:
        return " -> ".join(self._labels) if self._labels else "<top>"

    def refuse(self, x: Any, reason: str) -> GraphCacheKeyError:
        return GraphCacheKeyError(
            f"cache_key: cannot key {self._where()} "
            f"(type {_type_name(type(x))}): {reason}. {_WAYS_OUT}"
        )

    # -- memoised composites -------------------------------------------------
    def _memoised(self, x: Any, kind: str, walker) -> bytes:
        i = id(x)
        if i in self._path:
            raise GraphCacheKeyError(
                f"cache_key: cycle: {self._where()} refers back to "
                f"{self._path[i]} (type {_type_name(type(x))}). Recursive "
                "structures and recursive NESTED helpers (closures that call "
                "themselves or each other, dispatch tables held in closures) "
                "cannot be keyed; define such helpers at module level, where "
                f"they are module globals and are not walked. {_WAYS_OUT}"
            )
        hit = self._memo.get(i)
        if hit is not None:
            return hit[1]
        if self._depth >= _MAX_DEPTH:
            raise self.refuse(x, f"nesting deeper than {_MAX_DEPTH}")
        self._depth += 1
        self._path[i] = self._where()
        try:
            h = hashlib.sha256()
            h.update(_tok("kind", kind.encode("ascii")))
            walker(x, h)
            token = _tok("sub", h.digest())
        finally:
            del self._path[i]
            self._depth -= 1
        self.stats["steps"] += 1
        self.stats[kind] = self.stats.get(kind, 0) + 1
        self._memo[i] = (x, token)
        return token

    # -- entry points ------------------------------------------------------
    def value(self, x: Any, label: str, const: bool = False) -> bytes:
        self._labels.append(label)
        try:
            return self._value(x, const)
        finally:
            self._labels.pop()

    def cell(self, x: Any, label: str) -> bytes:
        """Closure-cell content: a module is admitted here only."""
        self._labels.append(label)
        try:
            if type(x) is types.ModuleType:
                return self._module(x)
            return self._value(x, False)
        finally:
            self._labels.pop()

    # -- the value rules (§2.2.2) ------------------------------------------
    def _value(self, x: Any, const: bool) -> bytes:
        t = type(x)
        # Before ANY hashing or comparison of type(x): a user metaclass
        # could define __hash__/__eq__ (review AB p2).
        if type(t) is not type:
            raise self.refuse(x, "its class has a user-defined metaclass")
        # 1. exact Python scalars
        if x is None:
            return _NONE_TOK
        if t is bool:
            return _TRUE_TOK if x else _FALSE_TOK
        if t is str:
            return _tok("py:str", _str_payload(x))
        if t is int:
            return _tok("py:int", hex(x).encode("ascii"))
        if t is float:
            return _tok("py:float", struct.pack("<d", x))
        if t is complex:
            return _tok("py:complex", struct.pack("<dd", x.real, x.imag))
        if t is bytes:
            return _tok("py:bytes", x)
        # 2. numpy scalars (membership test reads no attribute of x)
        if t in _NP_SCALAR_TYPES:
            return _tok("np:scalar",
                        _tok("name", t.__name__.encode("ascii"))
                        + _tok("dtype", x.dtype.str.encode("ascii"))
                        + _tok("bytes", x.tobytes()))
        # 3. exact ndarray
        if t is np.ndarray:
            return self._memoised(x, "array", self._walk_array)
        # 4. exact list / tuple
        if t is list:
            return self._memoised(x, "list", self._walk_list)
        if t is tuple:
            if const:
                return self._memoised(x, "tuple", self._walk_tuple_const)
            return self._memoised(x, "tuple", self._walk_tuple)
        # 5. exact dict
        if t is dict:
            return self._memoised(x, "dict", self._walk_dict)
        # 5b. Ellipsis / slice
        if x is Ellipsis:
            return _ELLIPSIS_TOK
        if t is slice:
            return _tok("py:slice",
                        self.value(x.start, "slice.start")
                        + self.value(x.stop, "slice.stop")
                        + self.value(x.step, "slice.step"))
        # code-constant-only rules
        if const:
            if t is types.CodeType:
                return self._memoised(x, "code", self._walk_code)
            if t is frozenset:
                return self._frozenset_const(x)
        # 6. callables (§2.2.1)
        if t is types.FunctionType:
            return self._memoised(x, "function", self._walk_function)
        if t is types.MethodType:
            return self._memoised(x, "bound method", self._walk_method)
        if t is functools.partial:
            return self._memoised(x, "partial", self._walk_partial)
        if any(t is b for b in _BUILTIN_TYPES):
            return self._builtin(x)
        # 6b. a module outside a closure cell
        if t is types.ModuleType:
            raise self.refuse(
                x, "a module is keyed (by name only) only when it sits "
                "directly in a closure cell; as a keyword argument, inside "
                "the ipv, in a default or inside a container its attributes "
                "could change without changing the key; pass the needed "
                "values as plain keyword arguments instead")
        # 7. class objects
        if issubclass(t, type):
            return self._class(x)
        # 8. phasic's own value types
        Property, _ps, _sl, StateIndexer, _ss = _phasic_types()
        if t is Property:
            return self._property(x)
        if t is StateIndexer:
            return self._memoised(x, "indexer", self._walk_indexer)
        # 9. anything else
        if t is frozenset or t is set:
            raise self.refuse(x, "sets are not keyed (their iteration order "
                                 "depends on the hash seed); pass a sorted list")
        if t is types.CodeType:
            raise self.refuse(x, "a code object is keyed only as a constant "
                                 "of a function")
        raise self.refuse(
            x, "only plain values, numpy arrays, containers of those, "
            "functions and phasic's StateIndexer/Property are keyed "
            "automatically (no user-defined protocol such as to_dict() is "
            "trusted)")

    # -- containers ----------------------------------------------------------
    def _walk_list(self, x: list, h) -> None:
        h.update(_count(len(x)))
        for i, v in enumerate(x):
            h.update(self.value(v, f"[{i}]"))

    def _walk_tuple(self, x: tuple, h) -> None:
        h.update(_count(len(x)))
        for i, v in enumerate(x):
            h.update(self.value(v, f"[{i}]"))

    def _walk_tuple_const(self, x: tuple, h) -> None:
        # Same encoding as _walk_tuple; elements under the CONSTANT rules
        # (which add only code objects and frozensets).
        h.update(_count(len(x)))
        for i, v in enumerate(x):
            h.update(self.value(v, f"[{i}]", const=True))

    def _frozenset_const(self, x: frozenset) -> bytes:
        # Elements by their token bytes (scalar tokens or sub-digest
        # tokens, computed from content alone), sorted: independent of the
        # hash-seed-dependent iteration order.
        if self._depth >= _MAX_DEPTH:
            raise self.refuse(x, f"nesting deeper than {_MAX_DEPTH}")
        self._depth += 1
        try:
            toks = sorted(self.value(v, "frozenset element", const=True)
                          for v in x)
        finally:
            self._depth -= 1
        return _tok("fset", _count(len(toks)) + b"".join(toks))

    @staticmethod
    def _identity_hashed(k: Any, depth: int = 0) -> bool:
        """True for dict keys that hash by identity (functions, bound
        methods, partials, builtins, classes, modules, indexers) or tuples /
        frozensets containing them at any depth. Type identity tests only."""
        t = type(k)
        if (t is types.FunctionType or t is types.MethodType
                or t is functools.partial or t is types.ModuleType
                or any(t is b for b in _BUILTIN_TYPES)):
            return True
        if type(t) is type and issubclass(t, type):
            return True
        if t is tuple or t is frozenset:
            if depth > _MAX_DEPTH:
                return True
            return any(_KeyWalker._identity_hashed(e, depth + 1) for e in k)
        return t is _phasic_types()[3]

    def _walk_dict(self, x: dict, h) -> None:
        h.update(_count(len(x)))
        for i, (k, v) in enumerate(x.items()):
            if self._identity_hashed(k):
                self._labels.append(f"dict key #{i}")
                try:
                    raise self.refuse(
                        k, "dict keys that are functions, methods, classes, "
                        "modules or other identity-hashed objects (or tuples "
                        "containing them) are refused: a callback looking "
                        "them up by identity can tell apart objects whose "
                        "content encodings are equal")
                finally:
                    self._labels.pop()
            h.update(self.value(k, f"dict key #{i}"))
            if type(k) is str or type(k) is int:
                label = f"[{k!r}]"
            else:
                label = f"[<key #{i}>]"
            h.update(self.value(v, label))

    # -- arrays ----------------------------------------------------------------
    def _walk_array(self, x: np.ndarray, h) -> None:
        dt = x.dtype
        if not any(dt.type is t for t in _NP_SCALAR_TYPES):
            raise self.refuse(x, f"arrays of dtype {dt.str!r} are not keyed "
                                 "(only numpy's own numeric, bool, str, bytes "
                                 "and datetime dtypes)")
        if dt.char in "gG":
            raise self.refuse(x, "long-double arrays are not keyed (their "
                                 "padding bytes are undefined)")
        if dt.hasobject:
            raise self.refuse(x, "arrays holding Python objects are not keyed")
        if dt.fields is not None:
            raise self.refuse(x, "structured arrays are not keyed")
        h.update(_tok("dtype", _tok("name", dt.type.__name__.encode("ascii"))
                      + _tok("str", dt.str.encode("ascii"))))
        h.update(_tok("shape", _count(x.ndim)
                      + b"".join(_tok("dim", b"%d" % n) for n in x.shape)))
        if x.flags.c_contiguous:
            layout = b"C"
        elif x.flags.f_contiguous:
            layout = b"F"
        else:
            layout = b"S" + b",".join(b"%d" % s for s in x.strides)
        h.update(_tok("layout", layout))
        mv = memoryview(np.ascontiguousarray(x).reshape(-1).view(np.uint8))
        h.update(b"data:%d:" % mv.nbytes)
        h.update(mv)
        self.stats["array_hashes"] += 1

    # -- functions (§2.2.1) ------------------------------------------------
    def _walk_function(self, f: types.FunctionType, h) -> None:
        mod = f.__module__
        if type(mod) is str and mod == "__mp_main__":
            # spawn / forkserver workers import the main script as
            # __mp_main__, the same module as the parent's __main__.
            mod = "__main__"
        h.update(self.value(mod, "__module__"))
        h.update(self._memoised_code(f.__code__))
        h.update(self.value(f.__name__, "__name__"))
        h.update(self.value(f.__qualname__, "__qualname__"))
        h.update(self.value(f.__defaults__, "defaults"))
        h.update(self.value(f.__kwdefaults__, "kwdefaults"))
        h.update(self.value(f.__dict__, "__dict__"))
        freevars = f.__code__.co_freevars
        closure = f.__closure__ or ()
        if len(closure) != len(freevars):
            raise self.refuse(f, "closure does not match its code's free "
                                 "variables")
        h.update(_count(len(freevars)))
        for name, c in zip(freevars, closure):
            h.update(self.value(name, "free variable name"))
            try:
                content = c.cell_contents
            except ValueError:
                h.update(_EMPTY_CELL_TOK)
                continue
            h.update(self.cell(content, f"closure {name!r}"))

    def _memoised_code(self, co: types.CodeType) -> bytes:
        self._labels.append("code")
        try:
            return self._memoised(co, "code", self._walk_code)
        finally:
            self._labels.pop()

    def _walk_code(self, co: types.CodeType, h) -> None:
        h.update(_tok("co_code", co.co_code))
        h.update(_tok("co_exceptiontable",
                      getattr(co, "co_exceptiontable", b"")))
        for fld in ("co_names", "co_varnames", "co_freevars", "co_cellvars"):
            h.update(self.value(getattr(co, fld), fld))
        for fld in ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount",
                    "co_flags"):
            h.update(self.value(getattr(co, fld), fld))
        h.update(self.value(co.co_name, "co_name"))
        consts = co.co_consts
        h.update(_count(len(consts)))
        for i, c in enumerate(consts):
            h.update(self.value(c, f"co_consts[{i}]", const=True))

    def _walk_method(self, m: types.MethodType, h) -> None:
        h.update(self.value(m.__func__, "__func__"))
        h.update(self.value(m.__self__, "__self__"))

    def _walk_partial(self, p: functools.partial, h) -> None:
        h.update(self.value(p.func, "partial.func"))
        h.update(self.value(p.args, "partial.args"))
        h.update(self.value(p.keywords, "partial.keywords"))
        h.update(self.value(vars(p), "partial.__dict__"))

    def _resolves_to_itself(self, x: Any, mod: Any, qn: Any) -> bool:
        if type(mod) is not str or type(qn) is not str:
            return False
        obj = sys.modules.get(mod)
        if obj is None:
            return False
        for part in qn.split("."):
            obj = getattr(obj, part, _MISSING)
            if obj is _MISSING:
                return False
        return obj is x

    def _builtin(self, x: Any) -> bytes:
        owner = getattr(x, "__self__", None)
        if not (owner is None or type(owner) is types.ModuleType):
            raise self.refuse(
                x, "a compiled method bound to an instance carries state in "
                "its __self__")
        mod = getattr(x, "__module__", None)
        qn = getattr(x, "__qualname__", None)
        if type(x) is np.ufunc and type(qn) is not str:
            qn = getattr(x, "__name__", None)
        if not self._resolves_to_itself(x, mod, qn):
            raise self.refuse(
                x, "a compiled function is keyed by its module and name, so "
                "it must resolve back to the very same object through "
                "sys.modules (scipy.special ufuncs, np.frompyfunc objects, "
                "lru_cache wrappers and other compiled callables do not); "
                "move it to a module global (then covered by your cache_key, "
                "as any imported object)")
        return _tok("fn:builtin", _tok("module", _str_payload(mod))
                    + _tok("qualname", _str_payload(qn)))

    def _class(self, x: type) -> bytes:
        if type(x) is not type:
            raise self.refuse(x, "classes with a user-defined metaclass are "
                                 "not keyed")
        if not (x.__flags__ & _IMMUTABLETYPE_FLAG):
            raise self.refuse(
                x, f"class {_type_name(x)} is a mutable (user-defined) class; "
                "only immutable compiled types such as "
                "itertools.combinations_with_replacement are keyed")
        mod = x.__module__
        qn = x.__qualname__
        if not self._resolves_to_itself(x, mod, qn):
            raise self.refuse(x, "an immutable type must resolve back to "
                                 "itself by module and name")
        return _tok("fn:type", _tok("module", _str_payload(mod))
                    + _tok("qualname", _str_payload(qn)))

    def _module(self, m: types.ModuleType) -> bytes:
        name = m.__dict__.get("__name__")
        if type(name) is not str:
            raise self.refuse(m, "a module whose __name__ is not a str")
        if sys.modules.get(name) is not m:
            raise self.refuse(m, f"module {name!r} is not the module "
                                 "registered under that name in sys.modules")
        return _tok("module", _str_payload(name))

    # -- phasic value types (rule 8) ---------------------------------------
    def _property_fields(self) -> frozenset:
        import dataclasses
        Property = _phasic_types()[0]
        return frozenset(f.name for f in dataclasses.fields(Property))

    def _check_property_shape(self, p: Any) -> None:
        Property = _phasic_types()[0]
        if type(p) is not Property:
            raise self.refuse(p, "only the exact class Property is keyed "
                                 "(subclasses may override its behaviour)")
        d = vars(p)
        if not all(type(k) is str for k in d) or set(d) != self._property_fields():
            raise self.refuse(p, "a Property with extra instance attributes "
                                 "is not keyed")

    def _property(self, p: Any) -> bytes:
        self._check_property_shape(p)
        d = vars(p)
        return _tok("phasic:property",
                    self.value(d["name"], "Property.name")
                    + self.value(d["min_value"], "Property.min_value")
                    + self.value(d["max_value"], "Property.max_value")
                    + self.value(d["offset"], "Property.offset"))

    def _walk_indexer(self, ix: Any, h) -> None:
        Property, PropertySet, Slot, StateIndexer, StateSpace = _phasic_types()
        try:
            psets = self._indexer_checks_ab(ix, Property, PropertySet, Slot,
                                            StateSpace)
        except GraphCacheKeyError:
            raise
        except Exception as e:
            raise self.refuse(ix, f"indexer check failed ({type(e).__name__}: "
                                  f"{e})") from e
        try:
            td = ix.to_dict()
        except Exception as e:
            raise self.refuse(ix, f"to_dict() failed ({type(e).__name__}: "
                                  f"{e})") from e
        h.update(_tok("class", b"StateIndexer"))
        h.update(_count(len(psets)))
        for ps in psets:
            h.update(_tok("psetclass", type(ps).__name__.encode("ascii")))
        h.update(self.value(td, "to_dict()"))
        offsets = [p.offset for ps in psets for p in ps.properties]
        h.update(_count(len(offsets)))
        for i, o in enumerate(offsets):
            h.update(self.value(o, f"Property offset #{i}"))
        # (c) only after to_dict() has been fingerprinted (which refuses any
        # non-plain field value, so no user operator runs below).
        try:
            c = StateIndexer.from_dict(td)
            same = self._indexer_matches(ix, c)
        except GraphCacheKeyError:
            raise
        except Exception as e:
            raise self.refuse(ix, f"indexer check failed ({type(e).__name__}: "
                                  f"{e})") from e
        if not same:
            raise self.refuse(ix, "indexer state differs from its to_dict(); "
                                  "was its property list changed after "
                                  "construction?")

    def _indexer_checks_ab(self, ix, Property, PropertySet, Slot, StateSpace):
        """Checks (a) and (b) of §2.2.2 rule 8, run before to_dict()."""
        d = vars(ix)
        if not all(type(k) is str for k in d):
            raise self.refuse(ix, "an indexer with non-str attribute names")
        extra = set(d) - _INDEXER_ATTRS
        if extra:
            raise self.refuse(ix, f"an indexer with extra attribute(s) "
                                  f"{sorted(extra)}")
        ps_map = d["_property_sets"]
        slots = d["_slots"]
        offsets = d["_offsets"]
        for nm, obj in (("_property_sets", ps_map), ("_slots", slots),
                        ("_offsets", offsets)):
            if type(obj) is not dict:
                raise self.refuse(ix, f"indexer attribute {nm} is not exactly "
                                      "a dict")
        for nm in ("_pset_order", "_slot_order", "_entity_order"):
            if type(d[nm]) is not list:
                raise self.refuse(ix, f"indexer attribute {nm} is not exactly "
                                      "a list")
        for k, v in offsets.items():
            if type(k) is not str or type(v) is not int:
                raise self.refuse(ix, "indexer _offsets must map str to int")
        psets = []
        for k, ps in ps_map.items():
            if type(k) is not str:
                raise self.refuse(ix, "indexer _property_sets has a non-str key")
            if not (type(ps) is PropertySet or type(ps) is StateSpace):
                raise self.refuse(ix, f"PropertySet {k!r} is a "
                                      f"{_type_name(type(ps))}, not exactly "
                                      "PropertySet or StateSpace")
            pd = vars(ps)
            if not all(type(a) is str for a in pd):
                raise self.refuse(ix, f"PropertySet {k!r} has non-str attribute "
                                      "names")
            extra = set(pd) - _PSET_ATTRS
            if extra:
                raise self.refuse(ix, f"PropertySet {k!r} has extra "
                                      f"attribute(s) {sorted(extra)}")
            props = pd["properties"]
            if not (type(props) is list or type(props) is tuple):
                raise self.refuse(ix, f"PropertySet {k!r}.properties is not a "
                                      "list or tuple")
            pdict = pd["property_dict"]
            if type(pdict) is not dict:
                raise self.refuse(ix, f"PropertySet {k!r}.property_dict is not "
                                      "exactly a dict")
            for a in ("_bases", "_radix_powers"):
                if type(pd[a]) is not np.ndarray:
                    raise self.refuse(ix, f"PropertySet {k!r}.{a} is not "
                                          "exactly a numpy array")
            for p in props:
                self._check_indexer_property(p, k)
            for pk, p in pdict.items():
                if type(pk) is not str:
                    raise self.refuse(ix, f"PropertySet {k!r}.property_dict has "
                                          "a non-str key")
                self._check_indexer_property(p, k)
            psets.append(ps)
        fresh_slot = set(vars(Slot("x")))
        for k, slot in slots.items():
            if type(k) is not str:
                raise self.refuse(ix, "indexer _slots has a non-str key")
            if type(slot) is not Slot:
                raise self.refuse(ix, f"slot {k!r} is a {_type_name(type(slot))}"
                                      ", not exactly Slot")
            sd = vars(slot)
            if not all(type(a) is str for a in sd) or set(sd) != fresh_slot:
                raise self.refuse(ix, f"slot {k!r} has extra attributes")
            if type(sd["name"]) is not str or sd["name"] != k:
                raise self.refuse(ix, f"slot {k!r} has a name different from "
                                      "its key")
        return psets

    _PLAIN_FIELD_TYPES = (str, int, bool)

    def _check_indexer_property(self, p: Any, pset_name: str) -> None:
        self._check_property_shape(p)
        # Inside an indexer, Property fields must be plain scalars, so the
        # behavioural comparison (c) can never run a user __eq__.
        for f, v in vars(p).items():
            t = type(v)
            if not (any(t is pt for pt in self._PLAIN_FIELD_TYPES)
                    or (type(t) is type and t in _NP_SCALAR_TYPES)):
                raise self.refuse(p, f"Property field {f!r} in PropertySet "
                                     f"{pset_name!r} is not a plain scalar")

    @staticmethod
    def _arrays_equal(a: Any, b: Any) -> bool:
        return (type(a) is np.ndarray and type(b) is np.ndarray
                and a.dtype == b.dtype and a.shape == b.shape
                and bool(np.array_equal(a, b)))

    def _indexer_matches(self, ix: Any, c: Any) -> bool:
        """Check (c): behavioural state equals the canonical rebuild."""
        d, cd = vars(ix), vars(c)
        if list(d["_property_sets"]) != list(cd["_property_sets"]):
            return False
        if d["_offsets"] != cd["_offsets"]:
            return False
        tsl = d["_total_state_length"]
        if type(tsl) is not int or tsl != cd["_total_state_length"]:
            return False
        for nm in ("_pset_order", "_slot_order", "_entity_order"):
            if d[nm] != cd[nm]:
                return False

        def triple(p):
            pv = vars(p)
            return (pv["name"], pv["min_value"], pv["max_value"])

        for name, ps in d["_property_sets"].items():
            cps = cd["_property_sets"][name]
            pd, cpd = vars(ps), vars(cps)
            if [triple(p) for p in pd["properties"]] != \
                    [triple(p) for p in cpd["properties"]]:
                return False
            if [(k, triple(p)) for k, p in pd["property_dict"].items()] != \
                    [(k, triple(p)) for k, p in cpd["property_dict"].items()]:
                return False
            for a in ("_bases", "_radix_powers"):
                if not self._arrays_equal(pd[a], cpd[a]):
                    return False
        return True


def cache_entry_key(cache_key: str, callback: Any, /, **params: Any) -> str:
    """Full SHA-256 hex digest (64 characters) identifying a cache entry.

    Parameters
    ----------
    cache_key : str
        The user's entry name.
    callback : callable
        The callback exactly as passed to the C builder (after phasic's
        ``with_ipv`` wrapping, so the ipv is covered through its closure).
    **params
        The keyword arguments that reach the builder or are stored on the
        graph, in the order given.

    Raises
    ------
    GraphCacheKeyError
        If any part of the call cannot be keyed exactly, or the phasic
        build record is missing, failed or no longer describes the
        installed files.
    """
    global _LAST_WALK_STATS
    w = _KeyWalker()
    try:
        build = _phasic_build_id()
        import numpy
        top = hashlib.sha256()
        top.update(_tok("format", GRAPH_CACHE_FORMAT.encode("ascii")))
        top.update(_tok("python", b"%d.%d" % sys.version_info[:2]))
        top.update(_tok("build", str(build).encode("ascii")))
        top.update(_tok("numpy", _str_payload(str(numpy.__version__))))
        top.update(w.value(cache_key, "cache_key"))
        top.update(w.value(callback, "callback"))
        top.update(_tok("kwargs", _count(len(params))))
        for name, v in params.items():
            top.update(w.value(name, "keyword name"))
            top.update(w.value(v, f"kwargs[{name!r}]"))
    except RecursionError as e:
        raise GraphCacheKeyError(
            "cache_key: the call is nested too deeply to be keyed "
            f"(RecursionError). {_WAYS_OUT}") from e
    finally:
        _LAST_WALK_STATS = dict(w.stats)
    return top.hexdigest()


# ---------------------------------------------------------------------------
# Build id (§2.2.3)
# ---------------------------------------------------------------------------

def _scan_package(root: str) -> dict[str, tuple[int, int, int, int, int]]:
    """Stat every ``*.py`` file and every file ending in an extension-module
    suffix under ``root``, recursively, skipping ``__pycache__``.

    Returns ``{relative path: (size, mtime_ns, ctime_ns, ino, dev)}``.
    MUST apply exactly the rules of the stdlib-only snapshot at the top of
    ``phasic/__init__.py``.
    """
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    out: dict[str, tuple[int, int, int, int, int]] = {}
    stack = [""]
    while stack:
        rel = stack.pop()
        with os.scandir(os.path.join(root, rel) if rel else root) as it:
            for e in it:
                r = os.path.join(rel, e.name) if rel else e.name
                if e.is_dir(follow_symlinks=False):
                    if e.name != "__pycache__":
                        stack.append(r)
                elif e.name.endswith(".py") or e.name.endswith(suffixes):
                    if e.is_file():
                        st = e.stat()
                        out[r] = (st.st_size, st.st_mtime_ns, st.st_ctime_ns,
                                  st.st_ino, st.st_dev)
    return out


def _hash_package_files(root: str, rels: list[str]) -> str:
    """Content hash of the package files, counted and length-prefixed: for
    each file in sorted relative-path order its path, size and bytes."""
    h = hashlib.sha256()
    rels = sorted(rels)
    h.update(_count(len(rels)))
    for r in rels:
        with open(os.path.join(root, r), "rb") as f:
            data = f.read()
        h.update(_tok("path", _str_payload(r)))
        h.update(_tok("size", b"%d" % len(data)))
        h.update(_tok("bytes", data))
    return h.hexdigest()


def _same_but_ctime(a: dict, b: dict) -> bool:
    """True when two stat tables agree in file set, size, mtime_ns, inode
    and device (ctime_ns may differ)."""
    if set(a) != set(b):
        return False
    for k, (size, mtime, _ctime, ino, dev) in a.items():
        s2, m2, _c2, i2, d2 = b[k]
        if (size, mtime, ino, dev) != (s2, m2, i2, d2):
            return False
    return True


def _module_source(mod: Any) -> Any:
    spec = getattr(mod, "__spec__", None)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if origin is not None:
        return origin
    return getattr(mod, "__file__", None)


def _compute_record(s0: Any) -> dict:
    """Build the record at the end of ``import phasic`` (never raises for
    an expected failure: returns ``{"error": msg}``)."""
    regular = (
        "cache_key needs a regular install of phasic (the compiled "
        "extension is outside the package directory, e.g. an editable "
        "install); use `pixi run install-dev` or a normal pip install"
    )
    if not isinstance(s0, dict) or "root" not in s0 or "files" not in s0:
        return {"error": "the phasic build record could not be made: no "
                         "top-of-import snapshot"}
    root = s0["root"]
    files0 = s0["files"]
    pkg = sys.modules.get("phasic")
    ext = getattr(pkg, "phasic_pybind", None) if pkg is not None else None
    ext_file = getattr(ext, "__file__", None) if ext is not None else None
    if not isinstance(ext_file, str) or not os.path.isfile(ext_file):
        return {"error": f"the phasic build record could not be made: the "
                         f"compiled extension file is missing ({ext_file!r}). "
                         f"{regular}"}
    real_root = os.path.realpath(root)

    def rel_in_pkg(path: str):
        rp = os.path.realpath(path)
        rel = os.path.relpath(rp, real_root)
        if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
            return None
        return rel

    rel_ext = rel_in_pkg(ext_file)
    if rel_ext is None or rel_ext not in files0:
        return {"error": f"the phasic build record could not be made: "
                         f"{ext_file} is not among the package files. {regular}"}
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == "phasic" or name.startswith("phasic.")):
            continue
        src = _module_source(mod)
        if src is None:
            if name.startswith("phasic.phasic_pybind."):
                continue
            return {"error": f"the phasic build record could not be made: "
                             f"module {name} has no source file, so the build "
                             "id (which covers the package's source and "
                             "extension files) cannot cover it"}
        rel = rel_in_pkg(src) if isinstance(src, str) else None
        if rel is None or rel not in files0:
            return {"error": f"the phasic build record could not be made: "
                             f"module {name} was loaded from {src!r}, which is "
                             "not among the package's hashed source and "
                             f"extension files. {regular}"}
    # Hash FIRST, then re-stat (review W 1).
    h = _hash_package_files(root, list(files0))
    s1 = _scan_package(root)
    if not _same_but_ctime(files0, s1):
        return {"error": "the phasic build record could not be made: package "
                         "files changed while phasic was being imported. "
                         + _RESTART}
    return {"root": root, "stats": s1, "hash": h, "error": None}


def _check_record(rec: dict) -> str:
    """Re-list and re-stat the package; return the import hash or raise
    GraphCacheKeyError (restart). Refreshes the stat table on a
    stats-only change whose re-hash equals the import hash."""
    root = rec["root"]
    cur = _scan_package(root)
    stats = rec["stats"]
    if set(cur) != set(stats):
        raise GraphCacheKeyError(_RESTART + " (a package file was added or "
                                            "removed)")
    if cur != stats:
        h2 = _hash_package_files(root, list(cur))
        if h2 != rec["hash"]:
            raise GraphCacheKeyError(_RESTART + " (package file contents "
                                                "changed)")
        rec["stats"] = cur
    return rec["hash"]


def _get_record() -> Any:
    ext = sys.modules.get("phasic.phasic_pybind")
    return getattr(ext, _RECORD_ATTR, None) if ext is not None else None


def _record_build_stats(s0: Any) -> None:
    """Record the phasic build id ONCE per process (called at the end of
    ``phasic/__init__.py`` with the top-of-import snapshot ``s0``).

    The record is stored on the ``phasic_pybind`` extension module. A second
    call (``importlib.reload(phasic)``) compares the files against the
    existing record and stores a "restart" failure if they differ. Never
    raises: failures are stored and raised as GraphCacheKeyError by the
    first key computation.
    """
    try:
        ext = sys.modules.get("phasic.phasic_pybind")
        if ext is None:
            return   # nothing to store on; a missing record raises later
        existing = getattr(ext, _RECORD_ATTR, None)
        if existing is not None:
            if existing.get("error"):
                return
            try:
                _check_record(existing)
            except GraphCacheKeyError as e:
                setattr(ext, _RECORD_ATTR, {"error": str(e)})
            return
        try:
            rec = _compute_record(s0)
        except Exception as e:   # a file vanishing between list and read, ...
            rec = {"error": f"the phasic build record could not be made "
                            f"({type(e).__name__}: {e})"}
        setattr(ext, _RECORD_ATTR, rec)
    except Exception as e:   # pragma: no cover - must never break import
        try:
            _logger.debug("phasic build record failed: %r", e)
        except Exception:
            pass


def _phasic_build_id() -> str:
    """The content hash of the installed phasic package, recorded at import.

    Re-lists and re-stats the package on every call: a file added or
    removed, or changed contents, raise GraphCacheKeyError (restart); a
    stats-only change with equal contents (hard-link ctime, xattr writes)
    is accepted silently.
    """
    rec = _get_record()
    if rec is None:
        raise GraphCacheKeyError(
            "cache_key: the phasic build record is missing (phasic was not "
            "imported completely, or the record was removed); restart the "
            "interpreter.")
    if rec.get("error"):
        raise GraphCacheKeyError(f"cache_key: {rec['error']}")
    try:
        return _check_record(rec)
    except GraphCacheKeyError:
        raise
    except Exception as e:
        raise GraphCacheKeyError(
            f"cache_key: the phasic package files could not be checked "
            f"({type(e).__name__}: {e}). {_RESTART}") from e
