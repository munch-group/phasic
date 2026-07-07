"""GATE G3 — serialize() round-trip, two parsers (Stage-3 duplication #4).

``Graph.serialize()`` emits a JSON dict consumed by two independent
reconstructors:
  Path A: ``Graph.from_serialized`` (pure Python, src/phasic/__init__.py:4082)
  Path B: ``GraphBuilder::parse_structure``/``build`` (C++, graph_builder.cpp:19,172)

On a graph WITHOUT the divergent features both rebuild identically and, sharing
the native ``Graph::pdf`` kernel, produce bit-identical PMF.  Two known
divergences (Stage-3 Q6) are pinned as xfail(strict):
  D1 (Q6a): ``from_serialized`` DROPS ``constant_edges`` while ``GraphBuilder``
            rebuilds them (graph_builder.cpp:276-306).
  D2 (Q6b): ``from_serialized`` merges duplicate-state vertices via
            ``find_or_create_vertex`` while ``GraphBuilder`` ignores
            ``vertex_indices`` and preserves identity via ``create_vertex_p``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import phasic
from phasic import Graph
from phasic.phasic_pybind import parameterized as P

pytestmark = [pytest.mark.equivalence]

THETA = np.array([2.0, 3.0], dtype=np.float64)
TIMES = np.array([0.5, 1.0, 2.0], dtype=np.float64)
GRAN = 100  # MUST equal GraphBuilder.compute_pmf default (phasic_pybind.cpp:5199)


def _to_json(ser: dict) -> str:
    return json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ser.items()}
    )


def _path_A(ser) -> np.ndarray:
    """Python parser: from_serialized -> update_weights -> native pdf."""
    g = Graph.from_serialized(ser)
    # pure-Python parser ran (NOT the C++ GraphBuilder). Post-WS-C from_serialized
    # lives in phasic._graph_serialize (assigned onto Graph); the C++ parser would
    # be phasic.phasic_pybind.parameterized.
    assert "phasic_pybind" not in Graph.from_serialized.__module__
    g.update_weights(THETA)
    return np.array([g.pdf(float(t), granularity=GRAN) for t in TIMES])


def _path_B(ser) -> np.ndarray:
    """C++ parser: GraphBuilder(json).compute_pmf."""
    assert hasattr(P, "get_compute_pmf_ffi_capsule")            # FFI-compiled build sanity
    b = P.GraphBuilder(_to_json(ser))
    assert type(b).__module__ == "phasic.phasic_pybind.parameterized"  # C++ parser ran
    return np.asarray(b.compute_pmf(THETA, TIMES, discrete=False, granularity=GRAN))


# ---- fresh graph builders (function scope: no module-level sharing) ----
def _g_match() -> Graph:
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v3, [0.0, 1.0])
    v1.add_edge(v3, [0.5, 0.5])
    return g


def _g_const() -> Graph:
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2, [1.0, 1.0])
    v1.add_aux_vertex_constant(0.7)          # -> constant_edges (phasiccpp.cpp:1645)
    return g


def _g_dup() -> Graph:
    g = Graph(1)
    s = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2a = g.create_vertex([2])               # create_vertex = NO dedup
    v2b = g.create_vertex([2])               # SECOND distinct vertex, same state [2]
    v3 = g.find_or_create_vertex([3])
    s.add_edge(v1, 1.0)
    v1.add_edge(v2a, [1.0, 0.0])
    v1.add_edge(v2b, [0.0, 1.0])
    v2a.add_edge(v3, [0.5, 0.0])
    v2b.add_edge(v3, [0.0, 0.5])
    return g


def test_g3_roundtrip_bit_identical():
    ser = _g_match().serialize()
    assert ser["constant_edges"].shape[0] == 0                       # no divergent feature
    assert len({tuple(r) for r in ser["states"].tolist()}) == ser["n_vertices"]  # no dup states
    a, b = _path_A(ser), _path_B(ser)
    np.testing.assert_array_equal(a, b)     # EXACT: shared native kernel, identical structure


@pytest.mark.xfail(strict=True, reason=(
    "Q6a: from_serialized DROPS constant_edges (src/phasic/__init__.py never reads the "
    "'constant_edges' key) while GraphBuilder::build rebuilds them "
    "(graph_builder.cpp:276-306). Stage-3 must teach from_serialized to read them."))
def test_g3_constant_edges_divergence():
    ser = _g_const().serialize()
    assert ser["constant_edges"].shape[0] > 0     # feature actually exercised
    a, b = _path_A(ser), _path_B(ser)
    np.testing.assert_array_equal(a, b)           # currently FALSE -> xfail


@pytest.mark.xfail(strict=True, reason=(
    "Q6b: from_serialized merges duplicate-state vertices via find_or_create_vertex "
    "(__init__.py:4342) while GraphBuilder ignores vertex_indices and preserves identity "
    "via create_vertex_p (graph_builder.cpp:233). Stage-3 must unify how vertex_indices "
    "is honored."))
def test_g3_vertex_indices_divergence():
    ser = _g_dup().serialize()
    assert len({tuple(r) for r in ser["states"].tolist()}) < ser["n_vertices"]  # dup states present
    a, b = _path_A(ser), _path_B(ser)
    np.testing.assert_array_equal(a, b)           # currently FALSE -> xfail
