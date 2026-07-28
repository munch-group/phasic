"""GATE G5 — parameterized elimination equivalence (Stage-3 duplication #1).

Pins the Python ``EliminationTrace`` record/replay engine
(``record_elimination_trace`` -> ``instantiate_from_trace``, trace_elimination.py)
against the C ``GraphBuilder`` reward-compute engine
(``GraphBuilder.compute_pmf``/``compute_moments``) for the SAME parameterized
graph + theta.

Both follow the "record once, replay per theta" pattern but are independent
engines: the Python side pre-eliminates to a reduced constant graph in NumPy,
the C side eliminates inside the forward kernel.  They never share operand
order, so equivalence is asserted to a tight rtol (not ``==``).

Cyclic and formula-mode graphs are xfail(strict): the Python trace refuses both
today (Stage-3 Q1 unifies the engines).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from phasic import Graph
from phasic.trace_elimination import (
    EliminationTrace,
    record_elimination_trace,
    instantiate_from_trace,
)
from phasic.phasic_pybind import parameterized
from phasic.ffi_wrappers import _make_json_serializable

pytestmark = [pytest.mark.equivalence]

GRAN = 100
TIMES = np.array([0.1, 0.5, 1.0, 2.0])

# ---- graph library (shape from tests/pytest/inference/test_trace_vs_direct.py) ----
ACYCLIC, CYCLIC, FORMULA = [], [], []
BUILDERS = {}


def _reg(name, kind):
    def deco(fn):
        BUILDERS[name] = fn
        {"acyclic": ACYCLIC, "cyclic": CYCLIC, "formula": FORMULA}[kind].append(name)
        return fn
    return deco


@_reg("acyclic_coalescent_n3", "acyclic")
def _coal():
    g = Graph(1)
    s = g.starting_vertex()
    v3, v2, v1 = (g.find_or_create_vertex([k]) for k in (3, 2, 1))
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [3.0])
    v2.add_edge(v1, [1.0])
    return g, np.array([2.0])


@_reg("acyclic_erlang3", "acyclic")
def _erlang():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c, d = (g.find_or_create_vertex([k]) for k in (1, 2, 3, 4))
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])
    b.add_edge(c, [1.0])
    c.add_edge(d, [1.0])
    return g, np.array([3.0])


@_reg("acyclic_branching", "acyclic")
def _branch():
    g = Graph(1)
    s = g.starting_vertex()
    A, B, C, ab = (g.find_or_create_vertex([k]) for k in (1, 2, 3, 4))
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0])
    A.add_edge(C, [0.0, 1.0])
    B.add_edge(ab, [1.0, 1.0])
    C.add_edge(ab, [1.0, 1.0])
    return g, np.array([2.0, 1.5])


@_reg("cyclic_back_edge", "cyclic")
def _back():
    g = Graph(1)
    s = g.starting_vertex()
    v1, v2, ab = (g.find_or_create_vertex([k]) for k in (2, 3, 1))
    s.add_edge(v1, [1.0])
    v1.add_edge(v2, [0.5])
    v2.add_edge(v1, [0.3])
    v2.add_edge(ab, [0.7])
    return g, np.array([2.0])


@_reg("cyclic_triangle", "cyclic")
def _tri():
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c, ab = (g.find_or_create_vertex([k]) for k in (1, 2, 3, 4))
    s.add_edge(a, [1.0])
    a.add_edge(b, [1.0])
    b.add_edge(c, [1.0])
    c.add_edge(a, [0.4])
    c.add_edge(ab, [0.6])
    return g, np.array([1.5])


@_reg("formula_coalescent", "formula")
def _formula():
    g, theta = _coal()
    g.weight_formula = "c0*t0"        # -> _weight_mode='formula'; Python trace refuses this
    return g, theta


# ---- helpers -------------------------------------------------------------
def _c_builder(graph, theta):
    js = json.dumps(_make_json_serializable(graph.serialize(theta_dim=len(theta))))
    return parameterized.GraphBuilder(js)


def _assert_c_engine(builder, theta):
    assert isinstance(builder, parameterized.GraphBuilder)
    assert builder.param_length == len(theta)
    probe = builder.build(np.asarray(theta, np.float64))
    probe.update_weights(np.asarray(theta, np.float64))
    _ = probe.expected_waiting_time()
    assert probe._has_param_compute_graph_cache() is True   # C parameterized engine ran


def _assert_py_engine(trace, rebuilt, theta):
    # The pure-Python recorder produced the trace, and instantiate_from_trace
    # (Python) produced `rebuilt`.  The load-bearing discriminator vs the C engine
    # is that `rebuilt` (a Python-eliminated graph) carries NO C parameterized
    # reward-compute cache, whereas the C engine's graph does (see _assert_c_engine).
    assert isinstance(trace, EliminationTrace)
    assert len(trace.operations) > 0 and trace.param_length == len(theta)
    _ = rebuilt.pdf(1.0, granularity=GRAN)
    assert rebuilt._has_param_compute_graph_cache() is False


def _param(names):
    def mark(n):
        if n in CYCLIC:
            return pytest.mark.xfail(strict=True, reason=(
                "G5/Q1: Python trace refuses cycles (self-loop 1/(1-q) unimplemented, "
                "trace_elimination.py:845); the C GraphBuilder engine handles them. "
                "Resolved when Stage-3 Q1 unifies the engines."))
        if n in FORMULA:
            return pytest.mark.xfail(strict=True, reason=(
                "G5/Q1: Python trace refuses weight_mode='formula' "
                "(trace_elimination.py:442); the C engine evaluates the formula tape. "
                "Resolved when Stage-3 Q1 unifies the engines."))
        return ()
    return pytest.mark.parametrize("name", [pytest.param(n, marks=mark(n)) for n in names])


# ---- gates ---------------------------------------------------------------
@_param(ACYCLIC + FORMULA + CYCLIC)
def test_g5_pmf_equivalence(name):
    build = BUILDERS[name]
    # Impl A (raises here for cyclic/formula -> xfail before any comparison)
    src, theta = build()
    trace = record_elimination_trace(src, theta_dim=len(theta))
    rebuilt = instantiate_from_trace(trace, params=np.asarray(theta, np.float64), use_log=False)
    _assert_py_engine(trace, rebuilt, theta)
    # Impl B
    cg, _ = build()
    builder = _c_builder(cg, theta)
    _assert_c_engine(builder, theta)
    # numbers
    py = np.array([rebuilt.pdf(float(t), granularity=GRAN) for t in TIMES])
    c = np.asarray(builder.compute_pmf(np.asarray(theta, np.float64), TIMES,
                                       discrete=False, granularity=GRAN))
    np.testing.assert_allclose(py, c, rtol=1e-6, atol=1e-12,
                               err_msg=f"{name}: Python trace pdf != C GraphBuilder pdf")


@_param(ACYCLIC + FORMULA + CYCLIC)
def test_g5_moments_equivalence(name):
    build = BUILDERS[name]
    src, theta = build()
    trace = record_elimination_trace(src, theta_dim=len(theta))
    rebuilt = instantiate_from_trace(trace, params=np.asarray(theta, np.float64), use_log=False)
    _assert_py_engine(trace, rebuilt, theta)
    cg, _ = build()
    builder = _c_builder(cg, theta)
    _assert_c_engine(builder, theta)
    m = np.asarray(builder.compute_moments(np.asarray(theta, np.float64), 2))  # [E[T], E[T^2]]
    np.testing.assert_allclose(rebuilt.expectation(), m[0], rtol=1e-9, atol=1e-12,
                               err_msg=f"{name}: mean mismatch")
    np.testing.assert_allclose(rebuilt.variance(), m[1] - m[0] ** 2, rtol=1e-9, atol=1e-12,
                               err_msg=f"{name}: variance mismatch")
