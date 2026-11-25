"""Graph cache keyed by a user-named entry: ``Graph(..., cache_key=...)``.

Plan: bug5-graph-cache-plan.md §3 (tests 1-26 and the contract pin 21).

Isolation: an autouse fixture redirects HOME, PHASIC_CACHE_DIR and
``graph_cache.DEFAULT_CACHE_DIR`` into ``tmp_path`` and unsets
PHASIC_COMPILATION_CACHE_DIR / JAX_COMPILATION_CACHE_DIR, so no code path
can reach the real ``~/.phasic_cache`` or ``~/.jax_cache``. Every
subprocess gets the same treatment plus a shared session MPLCONFIGDIR.
Every file a test writes is under ``tmp_path``.

The new names (``GraphCacheKeyError``, ``GraphCacheLoadError``,
``cache_entry_key``, ...) are imported INSIDE the tests, so the file also
collects on master and each test records its own outcome there. Tests
labelled REGRESSION GUARD also pass on master.

A hit is proven by the callback NOT being called (module-level CALLS
counter); "stale" is judged on expectation() against a build without
cache_key. Callbacks used with cache_key accept ``**kw``.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import errno
import functools
import io
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import types
import warnings
import zlib
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest

import phasic
import phasic.graph_cache as graph_cache
from phasic import Graph, Property, StateIndexer, with_ipv

REL = 1e-14
CALLS: list = []
RATE = 1.0
_DELETE_ROOT: dict = {"path": None}
THIS = sys.modules[__name__]
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# ----------------------------------------------------------------------------
# isolation
# ----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PHASIC_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    # backstop: the legacy/default directory also points into tmp_path
    monkeypatch.setattr(graph_cache, "DEFAULT_CACHE_DIR", tmp_path / "default_graphs")
    yield


@pytest.fixture(scope="session")
def mpl_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("mplconfig"))


def _sub_env(tmp_path, mpl_dir, extra=None):
    env = dict(os.environ)
    env["HOME"] = str(tmp_path / "home")
    env["PHASIC_CACHE_DIR"] = str(tmp_path / "subcache")
    env["MPLCONFIGDIR"] = mpl_dir
    env.pop("PHASIC_COMPILATION_CACHE_DIR", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def _run(tmp_path, mpl_dir, code, name="script.py", env=None, ok=True, timeout=500):
    path = tmp_path / name
    path.write_text(textwrap.dedent(code))
    r = subprocess.run([sys.executable, str(path)], env=_sub_env(tmp_path, mpl_dir, env),
                       cwd=str(tmp_path), capture_output=True, text=True, timeout=timeout)
    if ok:
        assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-4000:])
    return r


def _results(r):
    return [json.loads(line[len("RESULT:"):]) for line in r.stdout.splitlines()
            if line.startswith("RESULT:")]


def _entry_files(tmp_path):
    out = []
    for d in (tmp_path / "cache" / "graphs", tmp_path / "default_graphs"):
        if d.is_dir():
            out += sorted(d.glob("*.json"))
    return out


# ----------------------------------------------------------------------------
# callbacks and helpers
# ----------------------------------------------------------------------------
def _hit_build(cb, *args, **kw):
    """Build; return (graph, hit) where hit means the callback did not run."""
    n0 = len(CALLS)
    g = Graph(cb, *args, **kw)
    return g, len(CALLS) == n0


def _close(a, b, rel=REL):
    assert abs(a - b) <= rel * max(abs(a), abs(b)), (a, b)


def _ser_equal(g1, g2):
    s1, s2 = g1.serialize(), g2.serialize()
    assert s1.keys() == s2.keys()
    for k in s1:
        a, b = s1[k], s2[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert np.array_equal(np.asarray(a), np.asarray(b)), k
        else:
            assert a == b, k


def _sig(p):
    """A deterministic rate >= 1 that depends on the whole value."""
    if isinstance(p, np.ndarray):
        return 1.0 + zlib.crc32(p.tobytes()) % 97 / 10.0
    return 1.0 + zlib.crc32(repr(p).encode()) % 97 / 10.0


@with_ipv([4])
def coal_global(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2 * RATE]]


@with_ipv([3])
def cb_param(state, p=None, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], _sig(p) * n]]


@with_ipv([3])
def coal_param_cb(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], [float(n)]]]


@with_ipv([3])
def cb_ignore(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], float(n)]]


def coal_undecorated(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2]]


def _helper_rate():
    return 1.0


@with_ipv([4])
def cb21(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2 * _helper_rate()]]


def _rec_helper(n):
    return 0 if n <= 0 else _rec_helper(n - 1)


@with_ipv([3])
def cb_module_recursive(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    _rec_helper(2)
    if n <= 1:
        return []
    return [[[n - 1], float(n)]]


@with_ipv([3])
def cb_delroot(state, **kw):
    p = _DELETE_ROOT.get("path")
    if p is not None:
        shutil.rmtree(p, ignore_errors=True)
        _DELETE_ROOT["path"] = None
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], float(n)]]


@with_ipv([3])
def cb_mixed(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    if n == 3:
        return [[[n - 1], [1.0]]]
    return [[[n - 1], [1.0, 2.0]]]


def model25(state, k=1.0, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * k]]


@with_ipv([3])
def cb25(state, idx=(..., 0), **kw):
    CALLS.append(1)
    n = int(state[..., 0])
    assert int(state[idx]) == n
    if n <= 1:
        return []
    return [[[n - 1], float(n)]]


@functools.lru_cache(maxsize=None)
def _lru_helper(x):
    return x


class _CallableW:
    def __call__(self, x):
        return x


def _wrapped_target(x):
    return x


_UPDATE_WRAPPER_OBJ = functools.update_wrapper(_CallableW(), _wrapped_target)


def _key(cache_key, cb, **kw):
    from phasic.callback_hash import cache_entry_key
    return cache_entry_key(cache_key, cb, **kw)


def _stats():
    from phasic import callback_hash as ch
    return dict(ch._LAST_WALK_STATS)


def _refused(cb=cb_param, match=None, **kw):
    """Graph(cb, cache_key=...) refuses with GraphCacheKeyError, callback not
    called, no file written. Returns the error."""
    from phasic import GraphCacheKeyError
    n0 = len(CALLS)
    with pytest.raises(GraphCacheKeyError, match=match) as ei:
        Graph(cb, cache_key="refused", **kw)
    assert len(CALLS) == n0
    return ei.value


def _coal_E(n, rate):
    return sum(2.0 / (k * (k - 1)) for k in range(2, n + 1)) / rate


# ----------------------------------------------------------------------------
# 1. default build is uncached
# ----------------------------------------------------------------------------
def test_01_default_build_is_uncached(tmp_path, monkeypatch):
    monkeypatch.setattr(THIS, "RATE", 1.0)
    g1 = Graph(coal_global)
    _close(g1.expectation(), _coal_E(4, 1.0))
    monkeypatch.setattr(THIS, "RATE", 2.0)
    g2 = Graph(coal_global)
    _close(g2.expectation(), _coal_E(4, 2.0))
    assert _entry_files(tmp_path) == []


# ----------------------------------------------------------------------------
# 2. graph_cache= and configure(graph_cache=) are removed
# ----------------------------------------------------------------------------
def test_02_graph_cache_argument_removed():
    from phasic import PTDConfigError
    for v in (True, False):
        with pytest.raises(TypeError, match="cache_key"):
            Graph(coal_global, graph_cache=v)
    with pytest.raises(PTDConfigError, match="cache_key"):
        phasic.configure(graph_cache=True)


# ----------------------------------------------------------------------------
# 3. same key: hit (REGRESSION GUARD)
# ----------------------------------------------------------------------------
def test_03_same_key_hits(tmp_path):
    """REGRESSION GUARD (passes on master)."""
    g1, hit1 = _hit_build(coal_global, cache_key="t03")
    assert not hit1
    g2, hit2 = _hit_build(coal_global, cache_key="t03")
    assert hit2
    assert g1.vertices_length() == g2.vertices_length()
    _ser_equal(g1, g2)
    _close(g1.expectation(), g2.expectation())


T03_SCRIPT = '''
import json
import phasic
from phasic import Graph, with_ipv
CALLS = []

@with_ipv([5])
def cb(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * (n - 1) / 2]]

g = Graph(cb, cache_key="t03-sub")
ser = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in g.serialize().items()}
print("RESULT:" + json.dumps({"calls": len(CALLS), "E": g.expectation(),
                              "n": g.vertices_length(), "ser": ser}))
'''


def test_03_same_key_hits_across_processes(tmp_path, mpl_dir):
    """REGRESSION GUARD (passes on master)."""
    r0 = _results(_run(tmp_path, mpl_dir, T03_SCRIPT, env={"PYTHONHASHSEED": "0"}))[0]
    r1 = _results(_run(tmp_path, mpl_dir, T03_SCRIPT, env={"PYTHONHASHSEED": "1"}))[0]
    assert r0["calls"] > 0 and r1["calls"] == 0
    assert r0["n"] == r1["n"] and r0["ser"] == r1["ser"]
    _close(r0["E"], r1["E"])


# ----------------------------------------------------------------------------
# 4. different key: miss (REGRESSION GUARD)
# ----------------------------------------------------------------------------
def test_04_different_key_misses():
    """REGRESSION GUARD (passes on master)."""
    _hit_build(coal_global, cache_key="t04-a")
    _, hit = _hit_build(coal_global, cache_key="t04-b")
    assert not hit


# ----------------------------------------------------------------------------
# 5. arguments by value
# ----------------------------------------------------------------------------
_LONG = np.arange(2001, dtype=float)
_LONG2 = _LONG.copy()
_LONG2[1000] = -5.0


@pytest.mark.parametrize("a,b", [
    (1, 2),
    (_LONG, _LONG2),
    ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    ([[1], 2], [[1, 2]]),
    ({"a": {"b": 1}, "c": 2}, {"a": {"b": 1, "c": 2}}),
    ([[[3, 0], 1]], [[[3, 0]], 1]),
], ids=["int", "long-array", "dict-order", "nest1", "nest2", "nest3"])
def test_05_argument_change_misses_and_is_correct(a, b):
    _hit_build(cb_param, p=a, cache_key="t05")
    g, hit = _hit_build(cb_param, p=b, cache_key="t05")
    assert not hit
    _close(g.expectation(), Graph(cb_param, p=b).expectation())


# ----------------------------------------------------------------------------
# 6. ipv changed (REGRESSION GUARD)
# ----------------------------------------------------------------------------
def test_06_ipv_change_misses():
    """REGRESSION GUARD (passes on master)."""
    _hit_build(coal_undecorated, ipv=[3], cache_key="t06")
    _, hit = _hit_build(coal_undecorated, ipv=[4], cache_key="t06")
    assert not hit
    _hit_build(with_ipv([3])(coal_undecorated), cache_key="t06b")
    _, hit = _hit_build(with_ipv([4])(coal_undecorated), cache_key="t06b")
    assert not hit


# ----------------------------------------------------------------------------
# 7. callback body
# ----------------------------------------------------------------------------
SRC7 = '''
def cb7(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    if n <= 1:
        return []
    return [[[n - 1], n * 1.5]]
'''


def _exec_cb(src):
    ns = {"CALLS": CALLS, "__name__": "t07src"}
    exec(src, ns)
    return with_ipv([3])(ns["cb7"])


def test_07_callback_body():
    _hit_build(_exec_cb(SRC7), cache_key="t07")
    _, hit = _hit_build(_exec_cb(SRC7.replace("1.5", "2.5")), cache_key="t07")
    assert not hit, "constant change must miss"
    _, hit = _hit_build(_exec_cb(SRC7), cache_key="t07")
    assert hit
    comment = SRC7.replace("    n = int", "    # a comment line\n    n = int")
    blank = SRC7.replace("    n = int", "\n\n    n = int")
    spacing = SRC7.replace("n * 1.5", "n*1.5").replace("n - 1", "n-1")
    for src in (comment, blank, spacing):
        _, hit = _hit_build(_exec_cb(src), cache_key="t07")
        assert hit
    f, g = (lambda state, **kw: [] if state[0] <= 1 else [[[state[0] - 1], 1.0]]), (lambda state, **kw: [] if state[0] <= 1 else [[[state[0] - 1], 2.0]])  # noqa: E501
    assert _key("t07l", with_ipv([3])(f)) != _key("t07l", with_ipv([3])(g))


# ----------------------------------------------------------------------------
# 8. closures
# ----------------------------------------------------------------------------
def _make_rate(rate):
    def cb(state, **kw):
        CALLS.append(1)
        n = int(state[0])
        if n <= 1:
            return []
        return [[[n - 1], n * rate]]
    return with_ipv([3])(cb)


class _UserObj:
    rate = 2.0


def test_08_closures():
    from phasic import GraphCacheKeyError
    _hit_build(_make_rate(1.0), cache_key="t08")
    g5, hit = _hit_build(_make_rate(5.0), cache_key="t08")
    assert not hit
    _close(g5.expectation(), Graph(_make_rate(5.0)).expectation())
    g1, hit = _hit_build(_make_rate(1.0), cache_key="t08")
    assert hit
    _close(g1.expectation(), Graph(_make_rate(1.0)).expectation())

    # default-bound loop idiom
    cbs = []
    for n_ in (1.0, 3.0):
        def cbl(state, rate=n_, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], n * rate]]
        cbs.append(with_ipv([3])(cbl))
    _hit_build(cbs[0], cache_key="t08l")
    _, hit = _hit_build(cbs[1], cache_key="t08l")
    assert not hit

    # closure holding a user-class instance
    obj = _UserObj()

    def cbo(state, **kw):
        n = int(state[0])
        return [] if n <= 1 else [[[n - 1], n * obj.rate]]
    err = _refused(with_ipv([3])(cbo))
    assert "closure 'obj'" in str(err)
    assert isinstance(err, GraphCacheKeyError)

    # factory with a local all_pairs partial (immutable compiled type)
    def make_pairs():
        all_pairs = partial(combinations_with_replacement, r=2)

        def cb(state, **kw):
            CALLS.append(1)
            n = int(state[0])
            k = len(list(all_pairs(range(2))))
            return [] if n <= 1 else [[[n - 1], float(n * k)]]
        return with_ipv([3])(cb)
    _hit_build(make_pairs(), cache_key="t08p")
    _, hit = _hit_build(make_pairs(), cache_key="t08p")
    assert hit


# ----------------------------------------------------------------------------
# 9. source file edited on disk, module not reloaded
# ----------------------------------------------------------------------------
def test_09_source_edit_without_reload_keeps_key(tmp_path, monkeypatch):
    mod_dir = tmp_path / "mods9"
    mod_dir.mkdir()
    src = ("from phasic import with_ipv\n\n@with_ipv([3])\ndef cb(state, **kw):\n"
           "    n = int(state[0])\n    return [] if n <= 1 else [[[n - 1], n * 1.5]]\n")
    (mod_dir / "t09_mod.py").write_text(src)
    monkeypatch.syspath_prepend(str(mod_dir))
    monkeypatch.delitem(sys.modules, "t09_mod", raising=False)
    import importlib
    mod = importlib.import_module("t09_mod")
    try:
        k1 = _key("t09", mod.cb)
        (mod_dir / "t09_mod.py").write_text(src.replace("1.5", "9.5"))
        k2 = _key("t09", mod.cb)
        assert k1 == k2
    finally:
        sys.modules.pop("t09_mod", None)


# ----------------------------------------------------------------------------
# 10. indexer (REGRESSION GUARD)
# ----------------------------------------------------------------------------
def test_10_indexer_by_layout():
    """REGRESSION GUARD (passes on master)."""
    def ix(m):
        return StateIndexer(lineage=[Property("n", max_value=m)])
    _hit_build(cb_param, p=0.0, indexer=ix(3), cache_key="t10")
    _, hit = _hit_build(cb_param, p=0.0, indexer=ix(3), cache_key="t10")
    assert hit
    _, hit = _hit_build(cb_param, p=0.0, indexer=ix(4), cache_key="t10")
    assert not hit


# ----------------------------------------------------------------------------
# 11. refusals, non-cycles, Property and StateIndexer rule
# ----------------------------------------------------------------------------
class _ModelWithToDict:
    def __init__(self, r):
        self.r = r

    def to_dict(self):
        return {"r": self.r}

    @classmethod
    def from_dict(cls, d):
        return cls(d["r"])


@dataclasses.dataclass
class _DC:
    x: float = 1.0


class _Bound:
    def rate(self):
        return 2.0


def _cell_cb(value):
    def cb(state, **kw):
        _ = value
        n = int(state[0])
        return [] if n <= 1 else [[[n - 1], float(n)]]
    return with_ipv([3])(cb)


def _self_list():
    a = [1]
    a.append(a)
    return a


class _IE(enum.IntEnum):
    A = 1


class _F64Sub(np.float64):
    pass


def _structured():
    return np.zeros(2, dtype=[("a", "i4"), ("b", "f8")])


def _structured_obj():
    return np.zeros(2, dtype=[("a", "i4"), ("b", "O")])


@pytest.mark.parametrize("kw", [
    lambda: {"p": _ModelWithToDict(1.0)},
    lambda: {"p": __import__("pandas").DataFrame({"a": [1.0, 2.0]})},
    lambda: {"p": __import__("pandas").Series([1.0, 2.0])},
    lambda: {"p": _DC()},
    lambda: {"p": {1, 2}},
    lambda: {"p": np.ma.masked_array([1.0, 2.0], mask=[0, 1])},
    lambda: {"p": _self_list()},
    lambda: {"p": _structured()},
    lambda: {"p": _structured_obj()},
    lambda: {"p": np.zeros(1, dtype=[("a", "i4")])[0]},
    lambda: {"p": _lru_helper},
    lambda: {"p": _UPDATE_WRAPPER_OBJ},
    lambda: {"p": _IE.A},
    lambda: {"p": _F64Sub(1.0)},
    lambda: {"p": [_F64Sub(1.0)]},
], ids=["to_dict-obj", "dataframe", "series", "dataclass", "set", "masked",
        "self-list", "structured", "structured-object", "void-scalar", "lru",
        "update_wrapper", "intenum", "f64-subclass", "f64-subclass-in-list"])
def test_11_refused_kwargs(kw):
    _refused(**kw())


def test_11_user_class_kwarg_names_class():
    err = _refused(p=_ModelWithToDict)
    assert "_ModelWithToDict" in str(err)


@pytest.mark.parametrize("value", [
    lambda: _ModelWithToDict(1.0),
    lambda: _Bound().rate,
    lambda: {}.get,
    lambda: np.frompyfunc(lambda x: x, 1, 1),
    lambda: _lru_helper,
    lambda: _UPDATE_WRAPPER_OBJ,
], ids=["to_dict-obj", "bound-method", "dict.get", "frompyfunc", "lru", "update_wrapper"])
def test_11_refused_cells(value):
    _refused(_cell_cb(value()))


def test_11_cython_function_in_cell_refused():
    import importlib
    found = None
    for modname in ("pandas._libs.lib", "pandas._libs.algos", "pandas._libs.missing"):
        try:
            m = importlib.import_module(modname)
        except ImportError:
            continue
        for name in dir(m):
            o = getattr(m, name)
            if type(o).__name__ == "cython_function_or_method":
                found = o
                break
        if found is not None:
            break
    if found is None:
        pytest.skip("no Cython function available")
    _refused(_cell_cb(found))


def test_11_not_cycles_hash_and_hit():
    """REGRESSION GUARD (passes on master): spy-observed hits."""
    def build_shared():
        lst = [1, 2]
        return _hit_build(cb_param, p=[lst, lst], cache_key="t11-shared")
    build_shared()
    assert build_shared()[1]

    def build_ipv():
        s = [3]
        return _hit_build(cb_ignore.__wrapped__, ipv=[[s, 0.5], [s, 0.5]], cache_key="t11-ipv")
    build_ipv()
    assert build_ipv()[1]

    def build_many():
        many = [StateIndexer(lineage=[Property("n", max_value=3)]) for _ in range(200)]
        return _hit_build(cb_param, p=0.0, many=many, cache_key="t11-many")
    build_many()
    assert build_many()[1]


def test_11_property_kwargs_hit_and_miss():
    """Planned as a REGRESSION GUARD (spy-observed hit/miss). Measured on
    master: FAILS, because master cannot SAVE a Property kwarg (its entry
    stores construction_params via to_dict()/JSON and the save error is only
    a warning), so the second build misses."""
    _hit_build(cb_param, p=Property("n", max_value=3), cache_key="t11-prop")
    _, hit = _hit_build(cb_param, p=Property("n", max_value=3), cache_key="t11-prop")
    assert hit
    _, hit = _hit_build(cb_param, p=Property("n", max_value=4), cache_key="t11-prop")
    assert not hit


def test_11_property_offset_keyed():
    a = _key("k", cb_param, p=Property("n", max_value=3, min_value=1, offset=0))
    b = _key("k", cb_param, p=Property("n", max_value=3, min_value=1, offset=3))
    assert a != b


def _ix():
    return StateIndexer("epoch", lineage=[Property("n", max_value=3)])


@dataclasses.dataclass(frozen=True)
class _PropSub(Property):
    def encode_value(self, value):
        return 0


def test_11_indexer_refusals():
    from phasic.state_indexing import PropertySet, Slot

    ix = _ix()
    ix.rate = 2
    _refused(p=0.0, indexer=ix)

    ix = _ix()
    ix.lineage.extra = 1
    _refused(p=0.0, indexer=ix)

    class MyPS(PropertySet):
        pass
    _refused(p=0.0, indexer=StateIndexer(property_sets=[MyPS("lineage", [Property("n", max_value=3)])]))

    _refused(p=0.0, indexer=StateIndexer(lineage=[_PropSub("n", max_value=3)]))
    _refused(p=_PropSub("n", max_value=3))

    props = [Property("n", max_value=3)]
    ix = StateIndexer(lineage=props)
    props[0] = Property("n", max_value=5)
    _refused(p=0.0, indexer=ix, match="property list changed")

    props = [Property("n", max_value=3)]
    ix = StateIndexer(lineage=props)
    props.append(Property("m", max_value=2))
    _refused(p=0.0, indexer=ix)

    ix = _ix()
    ix.lineage._bases = np.array([9])
    _refused(p=0.0, indexer=ix)

    ix = _ix()
    ix.lineage.property_dict["n"] = _PropSub("n", max_value=3)
    _refused(p=0.0, indexer=ix)

    p = Property("n", max_value=3)
    object.__setattr__(p, "extra", 1)
    _refused(p=p)

    ix = _ix()
    ix._slots["epoch"] = Slot("other")
    _refused(p=0.0, indexer=ix)

    @dataclasses.dataclass(frozen=True)
    class MySlot(Slot):
        pass
    _refused(p=0.0, indexer=StateIndexer(lineage=[Property("n", max_value=3)],
                                         slots=[MySlot("epoch")]))

    ix = _ix()
    object.__setattr__(ix.lineage.properties[0], "extra", 1)
    _refused(p=0.0, indexer=ix)

    ix = _ix()
    object.__delattr__(ix, "_entity_order")
    _refused(p=0.0, indexer=ix)


def _coal_ix_cb(state, indexer=None, **kw):
    CALLS.append(1)
    t = []
    for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t


def test_11_indexer_admitted_and_hit():
    def hit_twice(make, key):
        _hit_build(cb_param, p=0.0, indexer=make(), cache_key=key)
        _, hit = _hit_build(cb_param, p=0.0, indexer=make(), cache_key=key)
        assert hit, key

    def used():
        ix = _ix()
        ix.index_to_props(1)
        ix.lineage.index_to_props(1)
        return ix
    hit_twice(used, "t11-used")

    def statespace():
        from phasic.state_indexing import StateSpace
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return StateIndexer(property_sets=[StateSpace([Property("n", max_value=3)], name="lineage")])
    hit_twice(statespace, "t11-ss")

    hit_twice(lambda: StateIndexer(lineage=[Property("n", max_value=3, offset=1)]), "t11-offset")
    hit_twice(lambda: StateIndexer(lineage=(Property("n", max_value=3),)), "t11-tuple")

    def added():
        a = StateIndexer("epoch", lineage=[Property("n", max_value=3)])
        b = StateIndexer(reward=[Property("r", max_value=2)])
        c = a + b
        assert c._entity_order == ["lineage", "epoch", "reward"]
        return c
    hit_twice(added, "t11-add")

    def appended():
        a = StateIndexer("epoch", lineage=[Property("n", max_value=3)])
        b = StateIndexer(lineage=[Property("r", max_value=2)])
        return a.append(b, prefix_other="x")
    hit_twice(appended, "t11-append")

    def epoch_and_joint():
        nr = 3
        ix = StateIndexer(lineages=[Property("ton", min_value=1, max_value=nr)])
        ipv = [nr] + [0] * (nr - 1)
        g = Graph(_coal_ix_cb, ipv=ipv, indexer=ix)
        g.update_weights([1.0])
        ge = g.add_epoch(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            jpg = g.joint_prob_graph(ix, reward_limit=2, mutation_rate=1e-2)
        return ge._indexer, jpg._indexer
    e1, j1 = epoch_and_joint()
    e2, j2 = epoch_and_joint()
    assert e1 is not None and j1 is not None
    for a, b, key in ((e1, e2, "t11-epoch"), (j1, j2, "t11-joint")):
        _hit_build(cb_param, p=0.0, indexer=a, cache_key=key)
        _, hit = _hit_build(cb_param, p=0.0, indexer=b, cache_key=key)
        assert hit, key


def test_11_scalar_types_keyed_apart():
    assert _key("k", cb_param, p=[np.float64(1.0)]) != _key("k", cb_param, p=[1.0])


def test_11_allow_lists_match_used_objects():
    from phasic import callback_hash as ch
    ix = _ix()
    ix.index_to_props(1)
    ix.lineage.index_to_props(1)
    assert set(vars(ix)) == ch._INDEXER_ATTRS
    assert set(vars(ix.lineage)) == ch._PSET_ATTRS


# ----------------------------------------------------------------------------
# 12. values the old entry format could not store
# ----------------------------------------------------------------------------
def test_12_values_old_format_could_not_store(tmp_path):
    kw = dict(a=np.bool_(True), c=1 + 2j, b=b"x", big=[10 ** 5000])
    g, hit = _hit_build(cb_ignore, cache_key="t12", **kw)
    assert not hit and g._graph_cache_entry is not None and Path(g._graph_cache_entry).exists()
    _, hit = _hit_build(cb_ignore, cache_key="t12", **kw)
    assert hit

    def make_big():
        def cb(state, big=10 ** 5000, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return with_ipv([3])(cb)
    _hit_build(make_big(), cache_key="t12b")
    _, hit = _hit_build(make_big(), cache_key="t12b")
    assert hit

    def make_np():
        import numpy as np_local

        def cb(state, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(np_local.sqrt(n * n))]]
        return with_ipv([3])(cb)
    _hit_build(make_np(), cache_key="t12m")
    _, hit = _hit_build(make_np(), cache_key="t12m")
    assert hit


# ----------------------------------------------------------------------------
# 13. phasic build id
# ----------------------------------------------------------------------------
def test_13_build_id_and_numpy_version_in_key(monkeypatch):
    from phasic import callback_hash as ch
    _hit_build(cb_ignore, cache_key="t13")
    with monkeypatch.context() as m:
        m.setattr(ch, "_phasic_build_id", lambda: "0" * 64)
        _, hit = _hit_build(cb_ignore, cache_key="t13")
        assert not hit
    _hit_build(cb_ignore, cache_key="t13n")
    import numpy
    with monkeypatch.context() as m:
        m.setattr(numpy, "__version__", "0.0.0-test")
        _, hit = _hit_build(cb_ignore, cache_key="t13n")
        assert not hit


def test_13_file_added_raises(monkeypatch):
    from phasic import GraphCacheKeyError
    from phasic import callback_hash as ch
    orig = ch._scan_package

    def scan(root):
        s = dict(orig(root))
        s["zz_added_module.py"] = (1, 1, 1, 1, 1)
        return s
    monkeypatch.setattr(ch, "_scan_package", scan)
    with pytest.raises(GraphCacheKeyError, match="restart"):
        _key("t13a", cb_ignore)


def test_13_content_hash_across_file_boundary(tmp_path):
    from phasic import callback_hash as ch
    a, b = tmp_path / "h1", tmp_path / "h2"
    a.mkdir()
    b.mkdir()
    (a / "x.py").write_bytes(b"ab")
    (a / "y.py").write_bytes(b"c")
    (b / "x.py").write_bytes(b"a")
    (b / "y.py").write_bytes(b"bc")
    assert ch._hash_package_files(str(a), ["x.py", "y.py"]) != \
        ch._hash_package_files(str(b), ["x.py", "y.py"])


T13_PRELUDE = '''
import copy, json, sys
import phasic
from phasic import callback_hash as ch, with_ipv, GraphCacheKeyError, Graph
ext = phasic.phasic_pybind

@with_ipv([3])
def cb(state, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

def key():
    return ch.cache_entry_key("t13", cb)

def keyerr():
    try:
        key()
        return None
    except GraphCacheKeyError as e:
        return str(e)

def out(**d):
    print("RESULT:" + json.dumps(d), flush=True)
'''


def test_13_record_stats_refresh_and_restart(tmp_path, mpl_dir):
    code = T13_PRELUDE + '''
k0 = key()
rec = getattr(ext, ch._RECORD_ATTR)
rel = sorted(rec["stats"])[0]
size, m, c, i, d = rec["stats"][rel]
rec["stats"][rel] = (size, m, c - 1, i, d)      # ctime only: re-hash equal
k1 = key()
out(case="ctime", same=(k0 == k1), refreshed=(rec["stats"][rel][2] == c))
rec["stats"][rel] = (size, m - 1, c, i, d)
ch._hash_package_files = lambda root, rels: "f" * 64
out(case="rehash", err=keyerr())
'''
    res = {r["case"]: r for r in _results(_run(tmp_path, mpl_dir, code))}
    assert res["ctime"]["same"] and res["ctime"]["refreshed"]
    assert res["rehash"]["err"] and "restart" in res["rehash"]["err"]


def test_13_recording_failure_and_missing_record(tmp_path, mpl_dir):
    hook = '''
import importlib.abc, importlib.machinery, sys

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "phasic.callback_hash":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        orig = spec.loader.exec_module

        def exec_module(m):
            orig(m)
            def boom(s0):
                raise OSError("a file vanished")
            m._compute_record = boom
        spec.loader.exec_module = exec_module
        return spec

sys.meta_path.insert(0, Hook())
'''
    code = hook + T13_PRELUDE + '''
out(case="failure", imported=True, err=keyerr())
'''
    r = _results(_run(tmp_path, mpl_dir, code, name="s13f.py"))[0]
    assert r["imported"] and r["err"] and "vanished" in r["err"]

    code = T13_PRELUDE + '''
delattr(ext, ch._RECORD_ATTR)
out(case="missing", err=keyerr())
'''
    r = _results(_run(tmp_path, mpl_dir, code, name="s13m.py"))[0]
    assert r["err"] and "missing" in r["err"]


def test_13_snapshot_differences(tmp_path, mpl_dir):
    code = T13_PRELUDE + '''
s0 = phasic._phasic_build_snapshot
rel = sorted(s0["files"])[0]

def with_change(idx, delta):
    s = copy.deepcopy(s0)
    t = list(s["files"][rel]); t[idx] += delta
    s["files"][rel] = tuple(t)
    return s

delattr(ext, ch._RECORD_ATTR)
ch._record_build_stats(with_change(1, -1000))     # mtime differs: failure
out(case="mtime", rec_err=getattr(ext, ch._RECORD_ATTR).get("error"), err=keyerr())

delattr(ext, ch._RECORD_ATTR)
ch._record_build_stats(with_change(2, -1000))     # ctime only: accepted
g = Graph(cb, cache_key="t13g")
out(case="ctime", rec_err=getattr(ext, ch._RECORD_ATTR).get("error"), err=keyerr(),
    n=g.vertices_length())

delattr(ext, ch._RECORD_ATTR)
orig_scan = ch._scan_package
ch._scan_package = lambda root: {k: (v[0] + 1,) + tuple(v[1:]) for k, v in orig_scan(root).items()}
ch._record_build_stats(s0)                         # changed after the hash: failure
ch._scan_package = orig_scan
out(case="final", rec_err=getattr(ext, ch._RECORD_ATTR).get("error"), err=keyerr())
'''
    res = {r["case"]: r for r in _results(_run(tmp_path, mpl_dir, code))}
    assert res["mtime"]["rec_err"] and res["mtime"]["err"]
    assert res["ctime"]["rec_err"] is None and res["ctime"]["err"] is None and res["ctime"]["n"] > 0
    assert res["final"]["rec_err"] and res["final"]["err"]


# ----------------------------------------------------------------------------
# 14. cache location and import order
# ----------------------------------------------------------------------------
def test_14_cache_location(tmp_path, monkeypatch):
    g, _ = _hit_build(cb_ignore, cache_key="t14")
    assert Path(g._graph_cache_entry).parent == (tmp_path / "cache" / "graphs").resolve()
    assert Path(g._graph_cache_entry).exists()
    for value in (None, ""):
        if value is None:
            monkeypatch.delenv("PHASIC_CACHE_DIR")
        else:
            monkeypatch.setenv("PHASIC_CACHE_DIR", value)
        g, _ = _hit_build(cb_ignore, cache_key=f"t14-{value!r}")
        assert Path(g._graph_cache_entry).parent == (tmp_path / "default_graphs").resolve()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHASIC_CACHE_DIR", "relcache")
    g, _ = _hit_build(cb_ignore, cache_key="t14-rel")
    assert os.path.isabs(g._graph_cache_entry)
    assert Path(g._graph_cache_entry).parent == (tmp_path / "relcache" / "graphs").resolve()


T14_HOOK = '''
import importlib.abc, json, os, sys
seen = {}

class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "phasic.phasic_pybind" and "omp" not in seen:
            seen["omp"] = "OMP_NUM_THREADS" in os.environ
        return None

sys.meta_path.insert(0, Hook())
'''


def test_14_import_order(tmp_path, mpl_dir):
    env = {"OMP_NUM_THREADS": None}
    r = _run(tmp_path, mpl_dir, T14_HOOK + '''
import phasic
print("RESULT:" + json.dumps({"omp": seen.get("omp")}))
''', name="s14a.py", env=env)
    assert _results(r)[0]["omp"] is True
    # The check works: a stubbed package that imports phasic.cache first
    # loads phasic_pybind before OMP_NUM_THREADS is set (review B p1).
    r = _run(tmp_path, mpl_dir, T14_HOOK + '''
import importlib.util, types
spec = importlib.util.find_spec("phasic")
stub = types.ModuleType("phasic")
stub.__path__ = list(spec.submodule_search_locations)
sys.modules["phasic"] = stub
try:
    import phasic.cache
except Exception as e:
    pass
print("RESULT:" + json.dumps({"omp": seen.get("omp")}))
''', name="s14b.py", env=env)
    assert _results(r)[0]["omp"] is False


# ----------------------------------------------------------------------------
# 15. unusable entries
# ----------------------------------------------------------------------------
def _real_entry(key):
    g, hit = _hit_build(coal_global, cache_key=key)
    assert not hit
    return Path(g._graph_cache_entry)


def _rewrite(path, entry):
    path.write_bytes(json.dumps(entry, separators=(",", ":")).encode())


def _assert_load_error(path, key, cause=None):
    from phasic import GraphCacheLoadError
    n0 = len(CALLS)
    with pytest.raises(GraphCacheLoadError) as ei:
        Graph(coal_global, cache_key=key)
    assert len(CALLS) == n0
    assert str(path) in str(ei.value)
    assert path.exists()
    if cause is not None:
        assert isinstance(ei.value.__cause__, cause), repr(ei.value.__cause__)
    return ei.value


def test_15_truncated_and_bad_utf8():
    p = _real_entry("t15a")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])
    _assert_load_error(p, "t15a", cause=json.JSONDecodeError)
    p = _real_entry("t15b")
    data = p.read_bytes()
    p.write_bytes(data.replace(b'"created_at":"', b'"created_at":"\xff\xfe', 1))
    _assert_load_error(p, "t15b", cause=UnicodeDecodeError)


def test_15_shape_checks():
    p = _real_entry("t15c")
    entry = json.loads(p.read_bytes())
    p.write_bytes(json.dumps([entry]).encode())
    e = _assert_load_error(p, "t15c")
    assert "top level" in str(e)

    p = _real_entry("t15d")
    entry = json.loads(p.read_bytes())
    del entry["graph_data"]
    _rewrite(p, entry)
    assert "graph_data" in str(_assert_load_error(p, "t15d"))

    p = _real_entry("t15e")
    entry = json.loads(p.read_bytes())
    entry["format"] = "1"
    _rewrite(p, entry)
    assert "format" in str(_assert_load_error(p, "t15e"))


def test_15_unknown_type_tag_imports_nothing(tmp_path, monkeypatch):
    mods = tmp_path / "mods15"
    mods.mkdir()
    (mods / "t15_evil.py").write_text("class Evil:\n    @classmethod\n    def from_dict(cls, d):\n        return cls()\n")
    monkeypatch.syspath_prepend(str(mods))
    p = _real_entry("t15f")
    entry = json.loads(p.read_bytes())
    k = next(iter(entry["graph_data"]))
    entry["graph_data"][k] = {"__type__": "t15_evil.Evil", "__data__": {}}
    _rewrite(p, entry)
    e = _assert_load_error(p, "t15f", cause=Exception)
    assert "t15_evil" in str(e) or "type tag" in str(e.__cause__)
    assert "t15_evil" not in sys.modules

    for bad in ({"__type__": "ndarray"}, {"__type__": "ndarray", "__data__": [[1, 2], [3]]}):
        p = _real_entry("t15g" + str(len(bad)))
        entry = json.loads(p.read_bytes())
        k = next(iter(entry["graph_data"]))
        entry["graph_data"][k] = bad
        _rewrite(p, entry)
        _assert_load_error(p, "t15g" + str(len(bad)), cause=Exception)


def test_15_from_serialized_errors(monkeypatch):
    p = _real_entry("t15h")

    def boom(cls, data):
        raise ValueError("bad graph data")
    monkeypatch.setattr(Graph, "from_serialized", classmethod(boom))
    _assert_load_error(p, "t15h", cause=ValueError)

    def oom(cls, data):
        raise MemoryError("oom")
    monkeypatch.setattr(Graph, "from_serialized", classmethod(oom))
    with pytest.raises(MemoryError):
        Graph(coal_global, cache_key="t15h")


def test_15_restore_error_propagates(monkeypatch):
    """A FORMULA-mode entry written by hand under the computed key; the tape
    restore on the live graph (the SECOND _set_weight_tape call) raises."""
    key = "t15-formula"
    g = Graph(coal_param_cb)
    g.weight_formula = "c0*t0"
    digest = _key(key, coal_param_cb)
    graph_cache.GraphCache(create=True).save(
        g, digest, {"cache_key": key, "callback_name": "coal_param_cb"})
    calls = {"n": 0}
    orig = Graph._set_weight_tape

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("restore failed (second call)")
        return orig(self, *a, **k)
    monkeypatch.setattr(Graph, "_set_weight_tape", flaky)
    with pytest.raises(RuntimeError, match="second call"):
        Graph(coal_param_cb, cache_key=key)
    assert calls["n"] == 2


@pytest.mark.skipif(IS_ROOT, reason="permissions are not enforced for root")
def test_15_unreadable_entry(tmp_path):
    p = _real_entry("t15p")
    os.chmod(p, 0)
    try:
        with pytest.raises(PermissionError) as ei:
            Graph(coal_global, cache_key="t15p")
        assert str(p) in str(ei.value)
    finally:
        os.chmod(p, 0o644)
    q = _real_entry("t15q")
    d = q.parent
    os.chmod(d, 0o600)
    try:
        with pytest.raises(PermissionError) as ei:
            Graph(coal_global, cache_key="t15q")
        assert "owner unknown" in str(ei.value)
        assert ei.value.__cause__ is not None
    finally:
        os.chmod(d, 0o755)


def test_15_clear_is_the_way_out():
    p = _real_entry("t15w")
    p.write_bytes(p.read_bytes()[:40])     # truncated: the entry header survives
    _assert_load_error(p, "t15w")
    phasic.clear_all_graph_caches()
    g, hit = _hit_build(coal_global, cache_key="t15w")
    assert not hit and Path(g._graph_cache_entry).exists()


# ----------------------------------------------------------------------------
# 16. write failures and stale partial files
# ----------------------------------------------------------------------------
def _partials(d):
    return sorted(Path(d).glob(".tmp-*.partial")) if Path(d).is_dir() else []


def test_16_write_failure_keeps_the_build(tmp_path, monkeypatch):
    def boom(src, dst):
        raise PermissionError(errno.EACCES, "denied")
    with monkeypatch.context() as m:
        m.setattr(graph_cache.os, "replace", boom)
        with pytest.raises(PermissionError) as ei:
            Graph(coal_global, cache_key="t16")
    e = ei.value
    path = graph_cache.GraphCache().entry_path(_key("t16", coal_global))
    assert any(str(path) in n for n in e.__notes__)
    assert e.graph._graph_cache_entry is None
    _ser_equal(e.graph, Graph(coal_global))
    assert _partials(tmp_path / "cache" / "graphs") == []


def test_16_serialize_failure(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("cannot serialize")
    monkeypatch.setattr(Graph, "serialize", boom)
    with pytest.raises(RuntimeError, match="cannot serialize") as ei:
        Graph(coal_global, cache_key="t16s")
    assert ei.value.graph is not None


def test_16_stale_partials(tmp_path):
    d = tmp_path / "cache" / "graphs"
    d.mkdir(parents=True)
    old, new = d / ".tmp-old.partial", d / ".tmp-new.partial"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    t = 1_700_000_000
    os.utime(old, (t - 7200, t - 7200))
    os.utime(new, (t, t))
    os.utime(d, (t, t))
    assert graph_cache.GraphCache().get_cache_stats()["num_graphs"] == 0
    graph_cache.GraphCache().clear()
    assert not old.exists() and new.exists()


# ----------------------------------------------------------------------------
# 17. validation before lookup
# ----------------------------------------------------------------------------
def test_17_validation_runs_before_lookup():
    Graph(coal_global, cache_key="t17")
    with pytest.raises(ValueError, match="ipv argument is ignored"):
        Graph(coal_global, ipv=[9], cache_key="t17")


# ----------------------------------------------------------------------------
# 18. clearing, positive controls, spoofs, argument checks
# ----------------------------------------------------------------------------
def test_18_clear_leaves_foreign_files(tmp_path):
    g, _ = _hit_build(coal_global, cache_key="t18c")
    d = Path(g._graph_cache_entry).parent
    foreign = d / ("0123456789abcdef" * 2 + ".json")
    foreign.write_text('{"my": "own data"}')
    c = graph_cache.GraphCache()
    assert c.clear() == 1
    assert foreign.exists() and c.last_clear_skipped == 1
    assert not Path(g._graph_cache_entry).exists()


def test_18_clear_model_cache_spares_phasic_cache_dir(tmp_path):
    from phasic.model_export import clear_model_cache
    g, _ = _hit_build(coal_global, cache_key="t18m")
    assert not str(tmp_path / "cache").startswith(os.environ["HOME"])
    clear_model_cache(verbose=False)
    assert Path(g._graph_cache_entry).exists()


def test_18_indexer_positive_controls():
    import dill
    plain = _ix()
    k_plain = _key("k", cb_param, p=0.0, indexer=plain)
    used = _ix()
    used.index_to_props(1)
    used.lineage.index_to_props(1)
    assert _key("k", cb_param, p=0.0, indexer=used) == k_plain
    from phasic.state_indexing import StateSpace
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ss = StateIndexer(property_sets=[StateSpace([Property("n", max_value=3)], name="lineage")])
    _key("k", cb_param, p=0.0, indexer=ss)
    _key("k", cb_param, p=0.0, indexer=StateIndexer(a=[Property("n", max_value=2)])
         + StateIndexer(b=[Property("m", max_value=2)]))
    dc = _ix()
    copy.deepcopy({i: dc.index_to_props(i) for i in range(3)})
    assert _key("k", cb_param, p=0.0, indexer=dc) == k_plain
    rt = dill.loads(dill.dumps(_ix()))
    assert _key("k", cb_param, p=0.0, indexer=rt) == k_plain


def test_18_real_install(tmp_path, mpl_dir):
    r = _run(tmp_path, mpl_dir, '''
import json
import phasic
from phasic import callback_hash as ch, with_ipv, Graph
rec = getattr(phasic.phasic_pybind, ch._RECORD_ATTR)

@with_ipv([3])
def cb(state, **kw):
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

g = Graph(cb, cache_key="t18-real")
print("RESULT:" + json.dumps({"error": rec.get("error"), "n": g.vertices_length(),
                              "entry": g._graph_cache_entry}))
''')
    res = _results(r)[0]
    assert res["error"] is None and res["n"] > 0 and res["entry"]


def test_18_module_with_nonstr_name_refused(monkeypatch):
    mod = types.ModuleType("t18_mod")
    monkeypatch.setitem(sys.modules, "t18_mod", mod)
    mod.__name__ = 3
    loads = []
    orig = graph_cache.GraphCache.load
    monkeypatch.setattr(graph_cache.GraphCache, "load",
                        lambda self, d: loads.append(d) or orig(self, d))

    def make():
        m = mod

        def cb(state, **kw):
            _ = m
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return with_ipv([3])(cb)
    _refused(make())
    assert loads == []


def test_18_dtype_and_scalar_classes():
    a = np.arange(4, dtype=np.int64)
    b = a.view(np.longlong)
    assert type(b.dtype.type) is type and a.tobytes() == b.tobytes()
    assert _key("k", cb_param, p=a) != _key("k", cb_param, p=b)
    assert _key("k", cb_param, p=np.int64(3)) != _key("k", cb_param, p=np.longlong(3))


def test_18_metaclass_spoof_runs_no_user_code():
    calls = []

    class Meta(type):
        def __hash__(cls):
            calls.append("hash")
            return hash(np.float64)

        def __eq__(cls, other):
            calls.append("eq")
            return True

    class Fake(metaclass=Meta):
        pass
    _refused(p=Fake())
    assert calls == []


def test_18_dict_subclass_property_dict_refused_before_to_dict(monkeypatch):
    called = []
    orig = StateIndexer.to_dict
    monkeypatch.setattr(StateIndexer, "to_dict", lambda self: called.append(1) or orig(self))

    class D(dict):
        pass
    ix = _ix()
    ix.lineage.property_dict = D(ix.lineage.property_dict)
    _refused(p=0.0, indexer=ix)
    assert called == []


def test_18_identity_hashed_dict_keys():
    def f(x):
        return x

    def g(x):
        return x
    keys = [(f,), (g,)]
    _refused(keys=keys, scale={keys[0]: 1.0, keys[1]: 5.0})
    _refused(scale={f: 1.0, g: 5.0})
    # Hashable wrappers around functions (implementation review M1).
    _refused(scale={slice(f): 1.0, slice(g): 5.0})
    _refused(scale={Property(f, max_value=1): 1.0,
                    Property(g, max_value=1): 5.0})


def test_18_float_property_fields_in_indexer_are_keyed():
    ix = StateIndexer(lineage=[Property("n", max_value=3.0)])
    ix2 = StateIndexer(lineage=[Property("n", max_value=3.0)])
    assert _key("t18f", coal_global, indexer=ix) == _key("t18f", coal_global, indexer=ix2)


def test_18_void_subclass_and_record():
    called = []

    class V(np.void):
        @property
        def dtype(self):
            called.append(1)
            return np.dtype("f8")
    try:
        v = V(8)
    except Exception:
        v = None
    if v is not None:
        _refused(p=v)
    rec = np.rec.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])[0]
    assert type(rec) is np.record
    _refused(p=rec)
    assert called == []


def test_18_float64_subclass_dtype_property_never_runs():
    called = []

    class F(np.float64):
        @property
        def dtype(self):
            called.append(1)
            return np.dtype("f8")
    _refused(p=F(1.0))
    assert called == []


def test_18_name_assignment_is_keyed():
    def make(name):
        def h():
            return 1
        h.__name__ = name

        def cb(state, **kw):
            n = int(state[0])
            r = 2.0 if h.__name__ == "a" else 1.0
            return [] if n <= 1 else [[[n - 1], n * r]]
        return with_ipv([3])(cb)
    assert _key("k", make("a")) != _key("k", make("b"))


def test_18_cache_key_argument_checks():
    with pytest.raises(TypeError):
        Graph(coal_global, cache_key=3)
    with pytest.raises(ValueError):
        Graph(coal_global, cache_key="")
    with pytest.raises(TypeError, match=r"str\(name\)"):
        Graph(coal_global, cache_key=np.str_("x"))
    with pytest.raises(ValueError):
        Graph(3, cache_key="x")
    with pytest.raises(TypeError):
        Graph(coal_undecorated, [3], "name")


def test_18_class_spoofs():
    class S1:
        @property
        def __class__(self):
            return np.float64

    class S2:
        @property
        def __class__(self):
            return type
    _refused(p=S1())
    _refused(p=S2())


# ----------------------------------------------------------------------------
# 19. clear_from_cache
# ----------------------------------------------------------------------------
def test_19_clear_from_cache(tmp_path, monkeypatch):
    g1, _ = _hit_build(coal_param_cb, cache_key="t19")
    p = Path(g1._graph_cache_entry)
    assert p.exists()
    assert g1.clear_from_cache()["graph_cache"] == 1 and not p.exists()
    _hit_build(coal_param_cb, cache_key="t19")
    g2, hit = _hit_build(coal_param_cb, cache_key="t19")
    assert hit
    assert g2.clear_from_cache()["graph_cache"] == 1 and not p.exists()
    g3 = Graph(coal_param_cb)
    assert g3.clear_from_cache()["graph_cache"] == 0
    g4, _ = _hit_build(coal_param_cb, cache_key="t19")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert g4.clear_from_cache()["graph_cache"] == 1 and not p.exists()


# ----------------------------------------------------------------------------
# 20. removed / unrelated environment variables
# ----------------------------------------------------------------------------
def test_20_disable_graph_cache_env_warns(tmp_path, mpl_dir):
    r = _run(tmp_path, mpl_dir, '''
import warnings
warnings.simplefilter("always")
import phasic
''', env={"PHASIC_DISABLE_GRAPH_CACHE": "1"})
    assert "PHASIC_DISABLE_GRAPH_CACHE no longer has an effect" in r.stderr


def test_20_disable_cache_env_has_no_effect(monkeypatch):
    """The PHASIC_DISABLE_CACHE half is a REGRESSION GUARD (passes on master)."""
    monkeypatch.setenv("PHASIC_DISABLE_CACHE", "1")
    _, hit = _hit_build(cb_ignore, cache_key="t20")
    assert not hit
    _, hit = _hit_build(cb_ignore, cache_key="t20")
    assert hit


# ----------------------------------------------------------------------------
# 21. contract pin: module-level helpers are the user's responsibility
# ----------------------------------------------------------------------------
def test_21_contract_pin_module_helper(monkeypatch):
    """CONTRACT PIN (documented behaviour; also the cached graph on master)."""
    g1, _ = _hit_build(cb21, cache_key="t21")
    monkeypatch.setattr(THIS, "_helper_rate", lambda: 3.0)
    g2, hit = _hit_build(cb21, cache_key="t21")
    assert hit
    _close(g2.expectation(), g1.expectation())
    assert abs(Graph(cb21).expectation() - g1.expectation()) > 1e-6


# ----------------------------------------------------------------------------
# 22. PHASIC_DYN_ORDERING follows the reader's environment
# ----------------------------------------------------------------------------
def _t22_script(key):
    return '''
import json
import phasic
from phasic import Graph, with_ipv
CALLS = []

@with_ipv([4])
def cb(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

g = Graph(cb, cache_key=%r)
n_cached = len(CALLS)
fresh = Graph(cb)
print("RESULT:" + json.dumps({"calls": n_cached, "dyn": bool(g.dyn_ordering),
                              "fresh": bool(fresh.dyn_ordering)}))
''' % key


def test_22_dyn_ordering_follows_reader(tmp_path, mpl_dir):
    for key, writer, reader in (("t22-a", "1", None), ("t22-b", None, "1")):
        w = _results(_run(tmp_path, mpl_dir, _t22_script(key), name="s22w.py",
                          env={"PHASIC_DYN_ORDERING": writer}))[0]
        r = _results(_run(tmp_path, mpl_dir, _t22_script(key), name="s22r.py",
                          env={"PHASIC_DYN_ORDERING": reader}))[0]
        assert w["calls"] > 0 and r["calls"] == 0
        assert w["dyn"] is (writer is not None)
        assert r["dyn"] is r["fresh"] is (reader is not None)


# ----------------------------------------------------------------------------
# 23. function attributes
# ----------------------------------------------------------------------------
def test_23_function_attributes():
    """Function attributes are keyed. The plan's literal case (a callback
    reading ``cb.rate`` through its OWN cell) is a self-reference, which the
    cycle rule refuses; the attribute is therefore read from a captured
    helper function, and the self-referencing form is asserted to be
    refused (implementation decision, see the bug 5 implementation report)."""
    from phasic import GraphCacheKeyError

    def make(r):
        def helper():
            return None
        helper.rate = r

        def cb(state, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], n * helper.rate]]
        return with_ipv([3])(cb)
    _hit_build(make(1.0), cache_key="t23")
    g, hit = _hit_build(make(2.0), cache_key="t23")
    assert not hit
    _close(g.expectation(), Graph(make(2.0)).expectation())

    def make_self(r):
        def cb(state, **kw):
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], n * cb.rate]]
        cb.rate = r
        return with_ipv([3])(cb)
    with pytest.raises(GraphCacheKeyError, match="cycle"):
        _key("t23s", make_self(1.0))


# ----------------------------------------------------------------------------
# 24. dicts with tuple / int keys (REGRESSION GUARD)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [
    ({(0, 1): 1.0, (1, 0): 2.0}, {(0, 1): 1.0, (1, 0): 3.0}),
    ({1: 1.0, 2: 2.0}, {1: 1.0, 2: 3.0}),
], ids=["tuple-keys", "int-keys"])
def test_24_dict_keys(a, b):
    """Planned as a REGRESSION GUARD. Measured on master: int-keys passes;
    tuple-keys FAILS, because master's entry stores construction_params as
    JSON, which cannot hold tuple keys (the save error is only a warning),
    so the second build misses."""
    _hit_build(cb_param, p=a, cache_key="t24")
    _, hit = _hit_build(cb_param, p=dict(a), cache_key="t24")
    assert hit
    _, hit = _hit_build(cb_param, p=b, cache_key="t24")
    assert not hit


# ----------------------------------------------------------------------------
# 25. Ellipsis constants, scipy ufunc, partial callback
# ----------------------------------------------------------------------------
def test_25_ellipsis_constants_hit():
    """REGRESSION GUARD for the state[..., 0] part (passes on master)."""
    _hit_build(cb25, cache_key="t25")
    _, hit = _hit_build(cb25, cache_key="t25")
    assert hit


def test_25_scipy_ufunc_in_cell_refused():
    from phasic import GraphCacheKeyError

    def make():
        from scipy.special import gamma

        def cb(state, **kw):
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(gamma(n))]]
        return with_ipv([3])(cb)
    with pytest.raises(GraphCacheKeyError):
        _key("t25g", make())


def test_25_partial_callback():
    cb = with_ipv([5])(partial(model25, k=2.0))
    g, hit = _hit_build(cb, cache_key="t25p")
    assert not hit
    entry = json.loads(Path(g._graph_cache_entry).read_bytes())
    assert entry["callback_name"] == "partial"
    _, hit = _hit_build(with_ipv([5])(partial(model25, k=2.0)), cache_key="t25p")
    assert hit


# ----------------------------------------------------------------------------
# 26. assorted
# ----------------------------------------------------------------------------
def test_26_partial_attributes_keyed():
    def make(v):
        p = partial(model25, k=1.0)
        p.extra = v

        def cb(state, **kw):
            _ = p.extra
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return with_ipv([3])(cb)
    assert _key("k", make(1)) != _key("k", make(2))


@pytest.mark.skipif(IS_ROOT, reason="permissions are not enforced for root")
def test_26_read_only_locations(tmp_path, monkeypatch):
    parent = tmp_path / "ro_parent"
    parent.mkdir()
    os.chmod(parent, 0o555)
    try:
        monkeypatch.setenv("PHASIC_CACHE_DIR", str(parent / "sub"))
        n0 = len(CALLS)
        with pytest.raises(OSError) as ei:
            Graph(cb_ignore, cache_key="t26ro")
        assert str(parent / "sub") in str(ei.value) and len(CALLS) == n0
    finally:
        os.chmod(parent, 0o755)

    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "rocache"))
    g, _ = _hit_build(cb_ignore, cache_key="t26-ro-hit")
    d = Path(g._graph_cache_entry).parent
    os.chmod(d, 0o555)
    try:
        n0 = len(CALLS)
        with pytest.raises(OSError) as ei:
            Graph(cb_ignore, cache_key="t26-ro-miss")
        assert str(d) in str(ei.value) and len(CALLS) == n0
        _, hit = _hit_build(cb_ignore, cache_key="t26-ro-hit")
        assert hit
    finally:
        os.chmod(d, 0o755)


class _Raw7(io.RawIOBase):
    """Writes at most 7 bytes per call."""

    def __init__(self, fd):
        self._fd = fd

    def writable(self):
        return True

    def write(self, b):
        return os.write(self._fd, bytes(memoryview(b)[:7]))

    def fileno(self):
        return self._fd

    def close(self):
        if not self.closed:
            os.close(self._fd)
        super().close()


def test_26_short_writes(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(graph_cache, "_open_entry_writer",
                  lambda fd: io.BufferedWriter(_Raw7(fd)))
        _hit_build(coal_global, cache_key="t26sw")
    g, hit = _hit_build(coal_global, cache_key="t26sw")
    assert hit
    _ser_equal(g, Graph(coal_global))


def test_26_spawn_worker_hits(tmp_path, mpl_dir):
    r = _run(tmp_path, mpl_dir, '''
import json, multiprocessing as mp
import phasic
from phasic import Graph, with_ipv
CALLS = []

@with_ipv([4])
def cb(state, **kw):
    CALLS.append(1)
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

def worker(q):
    g = Graph(cb, cache_key="t26-mp")
    q.put([len(CALLS), cb.__wrapped__.__module__, g.expectation()])

if __name__ == "__main__":
    g = Graph(cb, cache_key="t26-mp")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(q,))
    p.start()
    res = q.get(timeout=400)
    p.join()
    print("RESULT:" + json.dumps({"parent": len(CALLS), "worker": res[0],
                                  "module": res[1], "E": [g.expectation(), res[2]]}))
''', name="s26mp.py")
    res = _results(r)[0]
    assert res["module"] == "__mp_main__"
    assert res["parent"] > 0 and res["worker"] == 0
    _close(res["E"][0], res["E"][1])


@with_ipv([3])
def cb_const_tuple(state, adj=None, **kw):
    n = int(state[0])
    if adj is not None and adj.get((7, 7)) == [(7, 7)]:
        pass
    return [] if n <= 1 else [[[n - 1], float(n)]]


def test_26_large_kwargs_linear():
    N = 60_000
    d = {(i, i + 1): float(i) for i in range(N)}
    _key("t26big", cb_param, p=d)
    s = _stats()
    assert s["tuple"] >= N and s["steps"] <= N + 200
    keys = [(i, i % 7) for i in range(N)]
    adj = {k: [keys[(i * 31 + j) % N] for j in range(20)] for i, k in enumerate(keys)}
    const = next(c for c in cb_const_tuple.__wrapped__.__code__.co_consts
                 if type(c) is tuple and c == (7, 7))
    adj[const] = [const]
    _key("t26adj", cb_const_tuple, adj=adj)
    s = _stats()
    assert s["steps"] <= 2 * N + 300, s


T26_SEED = '''
import json
from phasic import with_ipv
from phasic.callback_hash import cache_entry_key

@with_ipv([3])
def cb(state, key=None, **kw):
    if key in {("a", 1), ("b", 2), ("c", 3)}:
        if key == ("a", 1):
            return []
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

@with_ipv([3])
def cb_edit(state, key=None, **kw):
    if key in {("a", 1), ("b", 2), ("c", 3)}:
        if key == ("c", 3):
            return []
    n = int(state[0])
    return [] if n <= 1 else [[[n - 1], float(n)]]

print("RESULT:" + json.dumps({"d": cache_entry_key("t26-seed", cb),
                              "e": cache_entry_key("t26-seed", cb_edit)}))
'''


def test_26_hash_seed_independence(tmp_path, mpl_dir):
    res = [_results(_run(tmp_path, mpl_dir, T26_SEED, name="s26seed.py",
                         env={"PYTHONHASHSEED": str(s)}))[0] for s in range(4)]
    assert len({r["d"] for r in res}) == 1
    assert len({r["e"] for r in res}) == 1
    assert res[0]["d"] != res[0]["e"]


def test_26_cycles_refused():
    from phasic import GraphCacheKeyError

    def selfrec():
        def helper(n):
            return 0 if n <= 0 else helper(n - 1)

        def cb(state, **kw):
            helper(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return cb

    def mutual():
        def a(n):
            return 0 if n <= 0 else b(n - 1)

        def b(n):
            return 0 if n <= 0 else a(n - 1)

        def cb(state, **kw):
            a(2)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return cb

    def table_container():
        table = []

        def f1(n):
            return table[0](n - 1) if n > 0 else 0
        table.append(f1)

        def cb(state, **kw):
            table[0](1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return cb

    def table_function():
        c2 = []

        def f2(state, **kw):
            _ = c2[0]
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        c2.append(f2)
        return f2

    def partial_in_list():
        lst = []

        def g(state, k=1.0, **kw):
            _ = lst[0]
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        lst.append(partial(g, k=2.0))
        return g

    for make in (selfrec, mutual, table_container, table_function, partial_in_list):
        with pytest.raises(GraphCacheKeyError, match="cycle"):
            _key("t26c", with_ipv([3])(make()))
    _key("t26c", cb_module_recursive)   # a module-level recursive helper is a global


def test_26_reload_after_reinstall_raises(tmp_path, mpl_dir):
    code = T13_PRELUDE + '''
import importlib
k0 = key()
orig_scan, orig_hash = ch._scan_package, ch._hash_package_files

def scan(root):
    s = dict(orig_scan(root))
    r = sorted(s)[0]
    t = list(s[r]); t[1] += 1
    s[r] = tuple(t)
    return s
ch._scan_package = scan
ch._hash_package_files = lambda root, rels: "e" * 64
importlib.reload(phasic)
ch._scan_package, ch._hash_package_files = orig_scan, orig_hash
out(case="reload", err=keyerr())
'''
    r = _results(_run(tmp_path, mpl_dir, code, name="s26reload.py"))[0]
    assert r["err"] and "restart" in r["err"]


def test_26_shared_array_hashed_once():
    a = np.zeros(1_000_000)
    _key("t26arr", cb_param, p=[a] * 200)
    assert _stats()["array_hashes"] == 1


@with_ipv([3])
def cb_tuple_default(state, idx=(1, 2), **kw):
    CALLS.append(1)
    n = int(state[0])
    if idx == (1, 2):
        pass
    return [] if n <= 1 else [[[n - 1], float(n)]]


def test_26_tuple_constant_and_default():
    _hit_build(cb_tuple_default, cache_key="t26td")
    _, hit = _hit_build(cb_tuple_default, cache_key="t26td")
    assert hit


def test_26_array_layout_keyed():
    a = np.arange(12.0).reshape(3, 4)
    assert _key("k", cb_param, p=a) != _key("k", cb_param, p=np.asfortranarray(a))


def test_26_estale_retried_once(monkeypatch):
    _hit_build(coal_global, cache_key="t26es")
    calls = {"n": 0}
    orig = graph_cache._open_entry_reader

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ESTALE, "Stale file handle")
        return orig(path)
    monkeypatch.setattr(graph_cache, "_open_entry_reader", flaky)
    _, hit = _hit_build(coal_global, cache_key="t26es")
    assert hit and calls["n"] == 2


def test_26_module_positions(monkeypatch):
    from phasic import GraphCacheKeyError

    def cell():
        import math as m_local

        def cb(state, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], m_local.sqrt(n * n)]]
        return with_ipv([3])(cb)
    _hit_build(cell(), cache_key="t26mod")
    _, hit = _hit_build(cell(), cache_key="t26mod")
    assert hit

    def default(state, m=math, **kw):
        n = int(state[0])
        return [] if n <= 1 else [[[n - 1], float(n)]]
    _refused(with_ipv([3])(default))

    def container():
        mods = [math]

        def cb(state, **kw):
            _ = mods[0]
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], float(n)]]
        return with_ipv([3])(cb)
    _refused(container())
    _refused(p=math)
    w = with_ipv([[[3], 1.0]])(cb_ignore.__wrapped__)
    w.__ipv__.append(math)      # a module inside the ipv (after validation)
    with pytest.raises(GraphCacheKeyError):
        _key("t26ipv", w)

    # contract pin: a module in a cell is keyed by name only
    mod = types.ModuleType("t26_ratemod")
    mod.RATE = 1.0
    monkeypatch.setitem(sys.modules, "t26_ratemod", mod)

    def make():
        m = mod

        def cb(state, **kw):
            CALLS.append(1)
            n = int(state[0])
            return [] if n <= 1 else [[[n - 1], n * m.RATE]]
        return with_ipv([3])(cb)
    g1, _ = _hit_build(make(), cache_key="t26pin")
    mod.RATE = 4.0
    g2, hit = _hit_build(make(), cache_key="t26pin")
    assert hit
    _close(g2.expectation(), g1.expectation())


def test_26_cache_root_deleted_during_build(tmp_path):
    root = tmp_path / "cache"
    _DELETE_ROOT["path"] = str(root)
    try:
        g, hit = _hit_build(cb_delroot, cache_key="t26root")
    finally:
        _DELETE_ROOT["path"] = None
    assert not hit and Path(g._graph_cache_entry).exists()


def test_26_extension_outside_package(tmp_path, mpl_dir):
    code = T13_PRELUDE + '''
delattr(ext, ch._RECORD_ATTR)
ext.__file__ = sys.executable
ch._record_build_stats(phasic._phasic_build_snapshot)
out(case="outside", err=keyerr())
'''
    r = _results(_run(tmp_path, mpl_dir, code, name="s26ext.py"))[0]
    assert r["err"] and "regular install" in r["err"]


def test_26_clear_all_legacy_absent_and_read_only_home(tmp_path, monkeypatch):
    legacy = Path(graph_cache.DEFAULT_CACHE_DIR)
    assert not legacy.exists()
    assert phasic.clear_all_graph_caches() == {"removed": 0, "skipped": 0,
                                               "foreign": 0}
    assert not legacy.exists()
    if IS_ROOT:
        return
    ro = tmp_path / "rohome"
    ro.mkdir()
    os.chmod(ro, 0o555)
    try:
        monkeypatch.setenv("HOME", str(ro))
        monkeypatch.delenv("PHASIC_CACHE_DIR")
        monkeypatch.setattr(graph_cache, "DEFAULT_CACHE_DIR", ro / ".phasic_cache" / "graphs")
        assert phasic.clear_all_graph_caches() == {"removed": 0, "skipped": 0,
                                                   "foreign": 0}
        assert phasic.get_graph_cache_stats()["num_graphs"] == 0
        assert graph_cache.GraphCache().clear() == 0
        assert not (ro / ".phasic_cache").exists()
    finally:
        os.chmod(ro, 0o755)


def test_26_legacy_sweep(tmp_path, monkeypatch):
    legacy = tmp_path / "default_graphs"
    legacy.mkdir()
    g = Graph(coal_global)
    digest = _key("t26leg", coal_global)
    new_path = graph_cache.GraphCache(legacy).save(
        g, digest, {"cache_key": "t26leg", "callback_name": "coal_global"})
    old = legacy / ("ab" * 16 + ".json")
    old.write_bytes(b'{\n  "version": "0.20.0",\n  "callback_hash": "x"\n}')
    future = legacy / ("cd" * 16 + ".json")
    future.write_bytes(b'{"format":"3","x":1}')
    foreign = legacy / ("ef" * 16 + ".json")
    foreign.write_bytes(b'{"my": "own md5-named file"}')
    res = phasic.clear_all_graph_caches()
    assert not old.exists()
    assert new_path.exists() and future.exists()
    assert foreign.exists()   # implementation review L2
    assert res["removed"] == 1 and res["foreign"] == 1


def test_26_fsync_einval_ignored(monkeypatch):
    def boom(fd):
        raise OSError(errno.EINVAL, "fsync not supported")
    monkeypatch.setattr(graph_cache.os, "fsync", boom)
    g, _ = _hit_build(coal_global, cache_key="t26fs")
    assert Path(g._graph_cache_entry).exists()


def test_26_value_details():
    from phasic import GraphCacheKeyError
    assert _key("k", cb_param, p=float("nan")) != _key("k", cb_param, p=-float("nan"))
    with pytest.raises(GraphCacheKeyError):
        _key("k", cb_param, p=np.zeros(3, dtype=np.longdouble))
    _key("k", cb_param, p=StateIndexer(dtype=[Property("n", max_value=3)]))
    ix = _ix()
    object.__delattr__(ix, "_slots")
    with pytest.raises(GraphCacheKeyError):
        _key("k", cb_param, indexer=ix)
    ns = {}
    exec("def f(state, **kw):\n    n = int(state[0])\n"
         "    return [] if n <= 1 else [[[n - 1], float(n)]]\n", ns)
    assert ns["f"].__module__ is None
    _key("k", with_ipv([3])(ns["f"]))
    _key("k", cb_param, callback=1.0)   # a user kwarg named callback


def test_26_ml_dtypes_refused():
    from phasic import GraphCacheKeyError
    ml = pytest.importorskip("ml_dtypes")
    a = np.zeros(3, dtype=ml.float8_e4m3fn)
    b = np.zeros(3, dtype=ml.float8_e4m3b11fnuz)
    for v in (a, b, ml.int4(1)):
        with pytest.raises(GraphCacheKeyError):
            _key("k", cb_param, p=v)


@pytest.mark.parametrize("make", [
    lambda v: np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[D]") + v,
    lambda v: np.array([1, 2], dtype="timedelta64[s]") + v,
    lambda v: np.array(1.0 + v),
], ids=["datetime64", "timedelta64", "0-d"])
def test_26_datetime_and_0d_arrays(make):
    _hit_build(cb_param, p=make(0), cache_key="t26dt")
    _, hit = _hit_build(cb_param, p=make(0), cache_key="t26dt")
    assert hit
    _, hit = _hit_build(cb_param, p=make(1), cache_key="t26dt")
    assert not hit


def test_26_renamed_entry_refused():
    from phasic import GraphCacheLoadError
    g, _ = _hit_build(coal_global, cache_key="t26ren-a")
    target = graph_cache.GraphCache().entry_path(_key("t26ren-b", coal_global))
    shutil.copy(g._graph_cache_entry, target)
    with pytest.raises(GraphCacheLoadError, match="entry_digest"):
        Graph(coal_global, cache_key="t26ren-b")


def test_26_non_entry_json_survives_clear(tmp_path):
    g, _ = _hit_build(coal_global, cache_key="t26json")
    other = Path(g._graph_cache_entry).parent / "notes.json"
    other.write_text("{}")
    graph_cache.GraphCache().clear()
    assert other.exists()


def test_26_shared_dag_nested_40():
    base = _key("k", cb_param, p=0)
    n0 = _stats().get("list", 0)
    x = 0
    for _ in range(40):
        x = [x, x]
    _key("k", cb_param, p=x)
    assert _stats().get("list", 0) - n0 == 40
    assert base


def test_26_keyboard_interrupt_in_save(tmp_path, monkeypatch):
    def boom(fd):
        raise KeyboardInterrupt
    monkeypatch.setattr(graph_cache.os, "fsync", boom)
    with pytest.raises(KeyboardInterrupt) as ei:
        Graph(coal_global, cache_key="t26ki")
    assert ei.value.graph is not None
    assert _partials(tmp_path / "cache" / "graphs") == []


def test_26_partial_subclass_refused():
    class P(partial):
        def __call__(self, *a, **k):
            return super().__call__(*a, **k)
    _refused(_cell_cb(P(model25, k=1.0)))


def test_26_mixed_coefficients_cannot_be_cached():
    with pytest.raises(ValueError) as ei:
        Graph(cb_mixed, cache_key="t26mixed")
    e = ei.value
    assert e.graph is not None
    assert any("cannot be cached" in n for n in e.__notes__)
