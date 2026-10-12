"""The C++ phasic::Graph::add_epoch widens the coefficient layout losslessly.

Part 3 of constant-parameterized-consistency-plan.md: its widening helper
(epoch_rebuild_wider in src/cpp/phasiccpp.cpp) used to copy only
coefficient-carrying edges, so a coefficient-less constant edge was dropped.
Compiled via cppimport like test_cpp_from_callback.py.
"""
import os

import numpy as np
import pytest

import phasic  # import before creating numpy arrays used by the x64 FFI
from phasic import Graph

# See test_cpp_from_callback.py: promote phasic_pybind's symbols (incl. the
# TLS ptd_err) to the global namespace so the fixture can bind them.
if hasattr(os, "RTLD_GLOBAL"):
    import ctypes as _ctypes
    import glob as _glob

    _pb = _glob.glob(
        os.path.join(os.path.dirname(phasic.__file__), "phasic_pybind*.so")
    )
    if _pb:
        _ctypes.CDLL(_pb[0], mode=os.RTLD_NOW | os.RTLD_GLOBAL)

cppimport = pytest.importorskip("cppimport")

_FIXTURE = os.path.join(os.path.dirname(__file__), "_cpp_add_epoch_fixture.cpp")
N = 4


@pytest.fixture(scope="module")
def cpp():
    return cppimport.imp_from_filepath(_FIXTURE, "_cpp_add_epoch_fixture")


def _constant_edges(g):
    return [(v.index(), e.to().index(), e.weight())
            for v in g.vertices() for e in v.edges()
            if e.coefficients_length() == 0]


def test_constant_edge_survives_cpp_add_epoch(cpp):
    base = Graph(cpp.base_graph(N, True))
    assert len(_constant_edges(base)) == 1
    epoch = Graph(cpp.epoch_graph(N, True, 1.0))
    src, dst, w = _constant_edges(base)[0]
    assert (src, dst, w) in _constant_edges(epoch)


def test_cpp_add_epoch_keeps_vertex_indices_and_counts(cpp):
    with_c = Graph(cpp.epoch_graph(N, True, 1.0))
    without = Graph(cpp.epoch_graph(N, False, 1.0))
    assert with_c.vertices_length() == without.vertices_length()
    base = Graph(cpp.base_graph(N, False))
    for v in base.vertices():
        w = without.vertex_at(v.index())
        assert list(w.state())[:base.state_length()] == list(v.state())


def test_cpp_add_epoch_unchanged_for_ordinary_graph(cpp):
    """Control (passes before and after): no constant edges, unique states."""
    g = Graph(cpp.epoch_graph(N, False, 1.0))
    assert g.param_length() == 4
    assert _constant_edges(g) == []


def test_cpp_add_epoch_keeps_duplicate_zero_states_and_is_aux(cpp):
    base = Graph(cpp.base_with_aux(N))
    epoch = Graph(cpp.epoch_with_aux(N, 1.0))
    aux_base = [v.index() for v in base.vertices() if v.is_aux()]
    assert len(aux_base) == 2
    # both aux vertices survive at their indices, still marked aux
    for i in aux_base:
        assert epoch.vertex_at(i).is_aux()
        assert list(epoch.vertex_at(i).state())[:base.state_length()] == [0] * base.state_length()
    # their coefficient-less back-edges survive too
    assert len(_constant_edges(epoch)) >= len(_constant_edges(base)) == 2


def test_cpp_add_epoch_refuses_constant_graph(cpp):
    with pytest.raises(ValueError, match="needs a parameterized model"):
        cpp.constant_epoch(N, 1.0)


@pytest.mark.parametrize("builder,phrase", [
    ("ragged_epoch", "add_epoch needs every edge's coefficient vector to have length param_length"),
    ("was_dph_epoch", "add_epoch is not supported on a discretize()'d graph"),
    ("formula_epoch", "add_epoch() adds coefficient slots, which is only defined for weight_mode='linear'"),
])
def test_cpp_add_epoch_refusals(cpp, builder, phrase):
    """The checks moved from the old widening helper to add_epoch itself
    (discretize-cpp-ssot-plan.md, primitive contract P4); messages kept."""
    with pytest.raises(ValueError) as exc:
        getattr(cpp, builder)(N)
    assert phrase in str(exc.value)
