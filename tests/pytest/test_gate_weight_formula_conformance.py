"""GATE G4 — weight-formula conformance (Stage-3 duplication #3).

Reachable gate (G4a): Python ``compile_formula`` -> C tape VM
(``ptd_weight_tape_eval_arrays``, via the ``update_weights`` residual path) must
equal the Python ``eval_tape`` reference oracle, over the full opcode/function
corpus, at the WEIGHT level (bit-identity) and at the PMF level through the real
``GraphBuilder`` C VM.

COVERAGE GAP (G4b, documented, NOT closeable in pytest): the C++
``wf_tokenize``/``WFParser``/``wf_emit`` compiler in src/cpp/phasiccpp.cpp is
UNREACHABLE from Python — ``phasic::Graph::weight_formula(const std::string&)``
is unbound and ``GraphBuilder`` accepts only a precompiled ``weight_formula_tape``,
never a source string.  The ``tape_py == tape_cpp`` compiler-equivalence leg can
only be gated by a new tests/cpp CTest, which additionally needs a C++ tape
reader that does not exist today (Stage-3 Q3/Q4). See the Stage-2 handoff.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import phasic                                   # import BEFORE jax / array creation
from phasic import Graph
from phasic import phasic_pybind as cpp
from phasic.ffi_wrappers import _make_json_serializable
from phasic.weight_formula import compile_formula, eval_tape, OPCODES

pytestmark = [pytest.mark.equivalence]

TIMES = np.linspace(0.1, 4.0, 40)
GRAN = 128


def test_ffi_is_compiled_in_this_build():
    assert hasattr(cpp.parameterized, "get_compute_pmf_ffi_capsule")


# ---- reachable C-VM driver (weight level: strongest, bit-identity) ---------
def _cvm_edge_weight(formula, theta, coeffs):
    """Drive the reachable C tape VM (residual path) and read back the edge weight."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, list(map(float, coeffs)))   # parameterized (non-IPV) edge
    g.weight_formula = formula                  # Python compile_formula -> _set_weight_tape
    assert g.weight_mode == "formula"           # tape installed (path proof)
    assert g._weight_formula_tape is not None
    g.update_weights(list(map(float, theta)))   # C VM ptd_weight_tape_eval (residual)
    (edge,) = list(v2.edges())
    return edge.weight()


CORPUS = [
    "c0*t0 + c1*t1", "c0*t0 * c1*t1", "c0*t0 - c1*t1 + 3.5",
    "exp(c0*t0)", "log(c0*t0 + c1)", "sqrt(c0) + c1*t0**2", "pow(c0*t0, 2)",
    "logistic(c0*t0) + c1", "logistic(-c0*t0) + c1", "c0/(c1+1)",
    "(c0 == 2)*t0 + (c0 != 2)*t1", "(c0 <= 2)*t0 + (c0 > 2)*t1",
    "(c0 >= 2)*t0 + (c0 < 2)*t1", "and(c0>0, c1>0)*t0", "or(c0>5, c1>0)*t0",
    "not(c0==0)*t1", "select(c0 == 2, c1*t0, t1)",
    "delta(c0,2)*c1*t0 + (1-delta(c0,2))*t1",
]


@pytest.mark.parametrize("formula", CORPUS)
def test_cvm_weight_bit_identical_to_oracle(formula):
    theta, coeffs = [0.6, 1.1], [2.0, 0.5]
    w_ref = eval_tape(compile_formula(formula), theta, coeffs)
    assert np.isfinite(w_ref) and w_ref >= 0.0
    w_c = _cvm_edge_weight(formula, theta, coeffs)
    assert w_c == w_ref, f"{formula}: C VM {w_c!r} != oracle {w_ref!r}"


def test_formula_path_not_linear_or_log():
    # exp path value is distinct from both linear (2.95) and log (0.91)
    w = _cvm_edge_weight("exp(c0*t0)", [0.6, 1.1], [2.0, 0.5])
    assert w == eval_tape(compile_formula("exp(c0*t0)"), [0.6, 1.1], [2.0, 0.5])
    assert w != 2.95 and w != 0.91


def test_opcode_coverage_is_complete():
    # every opcode name in OPCODES appears in at least one corpus tape
    emitted = set()
    inv = {v: k for k, v in OPCODES.items()}
    for f in CORPUS + ["-c0*t0", "select(c0==2, t0, t1)"]:
        ops = compile_formula(f)["ops"]
        i = 0
        while i < len(ops):
            name = inv[ops[i]]
            emitted.add(name)
            i += 2 if name in ("PUSH_THETA", "PUSH_COEFF", "PUSH_CONST") else 1
    assert set(OPCODES) - emitted == set(), f"uncovered opcodes: {set(OPCODES) - emitted}"


# ---- PMF-level cross-check through the FFI-build GraphBuilder ----------------
def _builder_pmf(graph, theta, *, mode=None, tape=None):
    s = _make_json_serializable(graph.serialize())
    if mode is not None:
        s["weight_mode"] = mode
    if tape is not None:
        s["weight_formula_tape"] = tape
    b = cpp.parameterized.GraphBuilder(json.dumps(s))
    th = np.zeros(0) if theta is None else np.asarray(theta, dtype=float)
    return np.asarray(b.compute_pmf(th, TIMES, discrete=False, granularity=GRAN))


def _tiny_param():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5])
    return g


def _ref_linear(weight):
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [float(weight)])
    return g


@pytest.mark.parametrize("formula",
    ["exp(c0*t0)", "logistic(c0*t0) + c1", "select(c0 == 2, c1*t0, t1)",
     "sqrt(c0) + c1*t0**2"])
def test_pmf_matches_oracle_constant_graph(formula):
    theta = [0.6, 1.1]
    tape = compile_formula(formula)
    w_ref = eval_tape(tape, theta, [2.0, 0.5])
    assert np.isfinite(w_ref) and w_ref >= 0.0
    pmf_f = _builder_pmf(_tiny_param(), theta, mode="formula", tape=tape)
    pmf_r = _builder_pmf(_ref_linear(w_ref), [1.0])
    np.testing.assert_allclose(pmf_f, pmf_r, rtol=0, atol=1e-12)


# ---- error paths: no silent fallback; widen exception tuple ------------------
_ERRS = (RuntimeError, ValueError, ZeroDivisionError)


@pytest.mark.parametrize("formula,theta",
    [("-c0*t0", [0.6, 1.1]),          # negative
     ("c5*t0", [0.6, 1.1]),           # coeff index out of range (edge has 2 coeffs)
     ("log(c0*t0 - 10)", [0.6, 1.1])])  # non-finite
def test_bad_weight_rejected(formula, theta):
    with pytest.raises(_ERRS):
        _builder_pmf(_tiny_param(), theta, mode="formula", tape=compile_formula(formula))


def test_strict_pairing_formula_without_tape_raises():
    s = _make_json_serializable(_tiny_param().serialize())
    s["weight_mode"] = "formula"
    with pytest.raises(_ERRS):
        cpp.parameterized.GraphBuilder(json.dumps(s))


def test_strict_pairing_tape_without_formula_raises():
    s = _make_json_serializable(_tiny_param().serialize())
    s["weight_formula_tape"] = compile_formula("c0*t0")
    with pytest.raises(_ERRS):
        cpp.parameterized.GraphBuilder(json.dumps(s))


# ---- known divergences (xfail strict; Stage-3 registry) ---------------------
@pytest.mark.xfail(strict=True, reason=(
    "Q-G4-1: eval_tape eagerly evaluates untaken select() arms (Python math.log "
    "domain error) while the C residual VM (ptd_weight_tape_specialize) prunes "
    "dead arms and succeeds. Stage-3 must unify dead-select-arm semantics."))
def test_select_dead_arm_oracle_divergence():
    f = "select(c0 == 2, c1*t0, log(0 - c0))"
    w_ref = eval_tape(compile_formula(f), [0.6, 1.1], [2.0, 0.5])   # raises -> xfail
    assert w_ref == _cvm_edge_weight(f, [0.6, 1.1], [2.0, 0.5])


@pytest.mark.xfail(strict=True, reason=(
    "Q-G4-2: Python float.__pow__ returns complex for negative base + fractional "
    "exponent while the C libm pow -> NaN -> reject. Oracle and runtime disagree "
    "on the pow domain. Stage-3 must define pow domain semantics."))
def test_pow_negative_base_fractional_divergence():
    w_ref = eval_tape(compile_formula("pow(0 - c0, 0.5)"), [0.6, 1.1], [2.0, 0.5])
    assert isinstance(w_ref, float)                                # complex -> xfail
