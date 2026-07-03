"""Regression test: PropertySet.index_to_props must not recompile a dataclass.

In the default mode (dataclass return), index_to_props called make_dataclass
— which runs exec/compile to synthesize a new frozen class — on EVERY scalar
call, and StateIndexer.index_to_props routes each element of an ndarray one at
a time through that scalar path. Decoding an array of N states therefore
compiled N throwaway classes (~660x slower than one build + O(N) instantiation).

The class is now memoized on the PropertySet. The key invariant: the number of
make_dataclass calls is independent of how many indices are decoded.
"""
import numpy as np

import phasic.state_indexing as si
from phasic.state_indexing import Property, PropertySet, StateIndexer


def _pset():
    return PropertySet(
        "lineage",
        [Property("descendants", max_value=20), Property("population", min_value=1, max_value=3)],
    )


def _count_make_dataclass(monkeypatch):
    calls = {"n": 0}
    real = si.make_dataclass

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(si, "make_dataclass", counting)
    return calls


def test_single_pset_builds_class_once(monkeypatch):
    calls = _count_make_dataclass(monkeypatch)
    pset = _pset()
    for i in range(pset.state_length):
        pset.index_to_props(i)
    pset.index_to_props(np.arange(pset.state_length))
    assert calls["n"] == 1, f"expected 1 build for one PropertySet, got {calls['n']}"


def test_indexer_array_decode_calls_independent_of_N(monkeypatch):
    idx = StateIndexer(
        lineage=[Property("descendants", max_value=20),
                 Property("population", min_value=1, max_value=3)]
    )
    calls = _count_make_dataclass(monkeypatch)

    # Warm caches (first decode may build the pset class + result class once).
    idx.index_to_props(np.arange(5))
    calls["n"] = 0

    idx.index_to_props(np.arange(10))
    c10 = calls["n"]
    idx.index_to_props(np.arange(idx.state_length))
    c_full = calls["n"] - c10

    assert c10 == c_full, (
        f"make_dataclass scaled with array size ({c10} for 10 vs {c_full} for "
        f"{idx.state_length}) — should be per-decode constant, not per element"
    )


def test_results_are_correct_and_stable():
    pset = _pset()
    # descendants base = 21 (0..20), population base = 3 (1..3)
    # index 5 -> descendants=5, population=1
    p = pset.index_to_props(5)
    assert (p.descendants, p.population) == (5, 1)

    arr = pset.index_to_props(np.array([0, 5, 7]))
    assert (arr[0].descendants, arr[0].population) == (0, 1)
    assert (arr[1].descendants, arr[1].population) == (5, 1)
    s7 = pset.index_to_props(7)
    assert (arr[2].descendants, arr[2].population) == (s7.descendants, s7.population)

    # cached class reused across scalar and array calls on the same pset
    assert type(arr[0]) is type(p)
