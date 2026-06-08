"""Batch 2: the C/C++ tape VM, driven through the real FFI build path
(serialize -> JSON with weight_mode='formula' -> GraphBuilder.compute_pmf).

Equivalence gates:
  * formula "c0*t0 + c1*t1"  reproduces weight_mode='linear'  (bit-identical)
  * formula "c0*t0 * c1*t1"  reproduces weight_mode='log'      (bit-identical)
  * transcendental/logic formulas (exp, logistic, select) match a constant
    reference graph whose edge weight is the Python reference VM's value
    -> transitively proves the C VM == weight_formula.eval_tape == numpy.
  * negative / non-finite results and out-of-range indices raise.
  * strict mode/tape pairing raises (no silent fallback).
"""
from __future__ import annotations

import json
import numpy as np
import pytest

import phasic                       # before jax / array creation
from phasic import Graph
from phasic import phasic_pybind as cpp
from phasic.ffi_wrappers import _make_json_serializable
from phasic.weight_formula import compile_formula, eval_tape

TIMES = np.linspace(0.1, 4.0, 40)
GRAN = 128                          # fixed -> identical discretisation both sides


def _builder_pmf(graph, theta, *, mode=None, tape=None):
    s = _make_json_serializable(graph.serialize())
    if mode is not None:
        s["weight_mode"] = mode
    if tape is not None:
        s["weight_formula_tape"] = tape
    b = cpp.parameterized.GraphBuilder(json.dumps(s))
    th = np.zeros(0) if theta is None else np.asarray(theta, dtype=float)
    return np.asarray(b.compute_pmf(th, TIMES, discrete=False, granularity=GRAN))


# --- fixtures ---------------------------------------------------------------
def tiny_param():
    """start -(1.0)-> [2] -(coeffs [2.0, 0.5])-> [1] absorbing."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5])
    return g, [2.0, 0.5]


def ref_linear(weight):
    """Same topology as tiny_param but a single coefficient = ``weight`` and
    theta=[1.0], so the linear inner product yields exactly ``weight``. Keeps
    the reference on the identical GraphBuilder.compute_pmf path."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [float(weight)])
    return g


def coalescent_param(n0):
    g = Graph(1)
    s = g.starting_vertex()
    v = {k: g.find_or_create_vertex([k]) for k in range(1, n0 + 1)}
    s.add_edge(v[n0], 1.0)
    for k in range(n0, 1, -1):
        v[k].add_edge(v[k - 1], [k * (k - 1) / 2.0, 1.0])   # 2 coeffs
    return g


# --- formula reproduces linear / log across MANY edges ----------------------
def test_formula_reproduces_linear():
    g = coalescent_param(6)                 # 5 parameterized edges
    theta = [1.3, 0.7]
    pmf_lin = _builder_pmf(g, theta)                         # default linear
    pmf_f = _builder_pmf(g, theta, mode="formula",
                         tape=compile_formula("c0*t0 + c1*t1"))
    np.testing.assert_allclose(pmf_f, pmf_lin, rtol=0, atol=1e-12)


def test_formula_reproduces_log():
    g = coalescent_param(6)
    theta = [1.3, 0.7]                       # all c_k*theta_k > 0
    pmf_log = _builder_pmf(g, theta, mode="log")
    pmf_f = _builder_pmf(g, theta, mode="formula",
                         tape=compile_formula("c0*t0 * c1*t1"))
    np.testing.assert_allclose(pmf_f, pmf_log, rtol=0, atol=1e-12)


# --- C VM == Python reference VM (eval_tape) for transcendental + logic -----
@pytest.mark.parametrize("formula", [
    "exp(c0*t0)",
    "logistic(c0*t0) + c1",
    "select(c0 == 2, c1*t0, t1)",          # c0==2 -> takes c1*t0
    "delta(c0, 2)*c1*t0 + (1 - delta(c0, 2))*t1",
    "sqrt(c0) + c1*t0**2",
])
def test_c_vm_matches_python_reference(formula):
    g, coeffs = tiny_param()
    theta = [0.6, 1.1]
    tape = compile_formula(formula)
    w_ref = eval_tape(tape, theta, coeffs)          # Python reference VM
    assert np.isfinite(w_ref) and w_ref >= 0.0
    pmf_formula = _builder_pmf(g, theta, mode="formula", tape=tape)
    pmf_ref = _builder_pmf(ref_linear(w_ref), [1.0])
    np.testing.assert_allclose(pmf_formula, pmf_ref, rtol=0, atol=1e-12)


# --- error handling: no silent fallback -------------------------------------
def test_negative_weight_raises():
    g, _ = tiny_param()
    with pytest.raises((RuntimeError, ValueError)):
        _builder_pmf(g, [0.6, 1.1], mode="formula",
                     tape=compile_formula("-c0*t0"))      # -1.2 < 0


def test_out_of_range_coeff_index_raises():
    g, _ = tiny_param()                                   # edge has 2 coeffs
    with pytest.raises((RuntimeError, ValueError)):
        _builder_pmf(g, [0.6, 1.1], mode="formula",
                     tape=compile_formula("c5*t0"))


def test_nonfinite_weight_raises():
    g, _ = tiny_param()
    with pytest.raises((RuntimeError, ValueError)):
        # log of a non-positive value -> nan/-inf -> rejected
        _builder_pmf(g, [0.6, 1.1], mode="formula",
                     tape=compile_formula("log(c0*t0 - 10)"))


def test_strict_pairing_formula_without_tape_raises():
    g, _ = tiny_param()
    s = _make_json_serializable(g.serialize())
    s["weight_mode"] = "formula"            # no weight_formula_tape
    with pytest.raises((RuntimeError, ValueError)):
        cpp.parameterized.GraphBuilder(json.dumps(s))


def test_strict_pairing_tape_without_formula_raises():
    g, _ = tiny_param()
    s = _make_json_serializable(g.serialize())
    s["weight_formula_tape"] = compile_formula("c0*t0")   # mode left 'linear'
    with pytest.raises((RuntimeError, ValueError)):
        cpp.parameterized.GraphBuilder(json.dumps(s))
