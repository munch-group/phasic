"""Batch 1 tests for src/phasic/weight_formula.py (parser + tape + ref VM).

Pure Python, no build needed: the module is stdlib-only, so we load it
directly from the working-tree source (avoids testing a stale installed copy
and lets Batch 1 iterate without `pixi run install-dev`).
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "src" / "phasic" / "weight_formula.py")
_spec = importlib.util.spec_from_file_location("wf_under_test", _SRC)
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)

compile_formula = wf.compile_formula
eval_tape = wf.eval_tape
WeightFormulaError = wf.WeightFormulaError
OPCODES = wf.OPCODES


def ev(formula, theta, coeff):
    return eval_tape(compile_formula(formula), theta, coeff)


# ---------------------------------------------------------------------------
# Arithmetic + functions vs an independent Python reference
# ---------------------------------------------------------------------------
TH = [0.7, 1.3, 2.1, -0.4, 0.0, 3.0]
CO = [1.5, 2.0, 0.5, 4.0, 3.0, 1.0]


def _logi(a):
    return 1.0 / (1.0 + math.exp(-a)) if a >= 0 else math.exp(a) / (1.0 + math.exp(a))


ARITH_CASES = [
    ("c0*t0 + c1*t1", lambda t, c: c[0] * t[0] + c[1] * t[1]),
    ("c0*t0*c1*t1", lambda t, c: c[0] * t[0] * c[1] * t[1]),   # log-mode equiv
    ("exp(c0*t0 + c1*t1) + c2", lambda t, c: math.exp(c[0] * t[0] + c[1] * t[1]) + c[2]),
    ("t0**2", lambda t, c: t[0] ** 2),
    ("pow(t0, c0)", lambda t, c: t[0] ** c[0]),
    ("sqrt(c0)", lambda t, c: math.sqrt(c[0])),
    ("logistic(c0*t0)", lambda t, c: _logi(c[0] * t[0])),
    ("log(c0) + exp(-t0)", lambda t, c: math.log(c[0]) + math.exp(-t[0])),
    ("-t0 + c0", lambda t, c: -t[0] + c[0]),
    ("2*t0 - 3.5*c0", lambda t, c: 2 * t[0] - 3.5 * c[0]),
    ("t1**-c2", lambda t, c: t[1] ** (-c[2])),          # unary minus in exponent
    ("-2**2", lambda t, c: -(2 ** 2)),                   # -> -4 (** binds tighter)
    ("2**3**2", lambda t, c: 2 ** (3 ** 2)),             # -> 512 (right-assoc)
    ("(c0 + c1) * t0", lambda t, c: (c[0] + c[1]) * t[0]),
    ("c0 + c1 * t0", lambda t, c: c[0] + c[1] * t[0]),   # precedence
    ("1.0 / (1.0 + c0*t0)", lambda t, c: 1.0 / (1.0 + c[0] * t[0])),
    ("1e2 * t0 + .5", lambda t, c: 1e2 * t[0] + 0.5),    # number formats
]


@pytest.mark.parametrize("formula,ref", ARITH_CASES)
def test_arithmetic_matches_reference(formula, ref):
    assert ev(formula, TH, CO) == pytest.approx(ref(TH, CO), rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Logic: comparisons, delta, and/or/not, select
# ---------------------------------------------------------------------------
def test_delta_kronecker():
    assert ev("delta(c0, c1)", TH, [3.0, 3.0]) == 1.0
    assert ev("delta(c0, c1)", TH, [3.0, 4.0]) == 0.0
    # delta is exactly '=='
    t = compile_formula("delta(c0,c1)")
    assert t["ops"] == compile_formula("c0 == c1")["ops"]


def test_comparisons():
    assert ev("c0 < c1", TH, [1.0, 2.0]) == 1.0
    assert ev("c0 > c1", TH, [1.0, 2.0]) == 0.0
    assert ev("c0 <= c1", TH, [2.0, 2.0]) == 1.0
    assert ev("c0 >= c1", TH, [2.0, 2.0]) == 1.0
    assert ev("c0 != c1", TH, [1.0, 2.0]) == 1.0


def test_select_branches_by_condition():
    # c0 == 3 -> take c1*t0 ; else c2*t1
    co_true = [3.0, 2.0, 5.0]
    co_false = [9.0, 2.0, 5.0]
    assert ev("select(c0 == 3, c1*t0, c2*t1)", TH, co_true) == pytest.approx(2.0 * TH[0])
    assert ev("select(c0 == 3, c1*t0, c2*t1)", TH, co_false) == pytest.approx(5.0 * TH[1])


def test_boolean_and_or_not():
    assert ev("and(c0 < 2, c1 > 0)", TH, [1.0, 1.0]) == 1.0
    assert ev("and(c0 < 2, c1 > 0)", TH, [5.0, 1.0]) == 0.0
    assert ev("or(c0 < 2, c1 > 0)", TH, [5.0, 1.0]) == 1.0
    assert ev("not(c0)", TH, [0.0]) == 1.0
    assert ev("not(c0)", TH, [7.0]) == 0.0


def test_compound_logic():
    f = "select(and(c0 == 3, c1 < 2), c2*t0, t1)"
    assert ev(f, TH, [3.0, 1.0, 4.0]) == pytest.approx(4.0 * TH[0])   # cond true
    assert ev(f, TH, [3.0, 9.0, 4.0]) == pytest.approx(TH[1])         # cond false


def test_partition_of_unity_pattern():
    # delta(c0,c1)*t0 + (1-delta(c0,c1))*t1  -> t0 if c0==c1 else t1
    f = "delta(c0,c1)*t0 + (1 - delta(c0,c1))*t1"
    assert ev(f, TH, [5.0, 5.0]) == pytest.approx(TH[0])
    assert ev(f, TH, [5.0, 6.0]) == pytest.approx(TH[1])


# ---------------------------------------------------------------------------
# theta-independence guard
# ---------------------------------------------------------------------------
ACCEPTED = [
    "select(c0 == 3, c1*t0, c2*t1)",
    "delta(c0,c1)*t0 + (1-delta(c0,c1))*t1",
    "select(c0 > c1, t0, t1)",          # condition uses only coeffs
    "select(c0 == 3, t0, t1)",          # bare coeff condition is fine
    "and(c0 < 2, c1 > 0) * t0",
]


@pytest.mark.parametrize("formula", ACCEPTED)
def test_guard_accepts_theta_independent_conditions(formula):
    compile_formula(formula)   # must not raise


REJECTED = [
    "select(t0 > c0, t1, t2)",   # theta in select condition
    "delta(t0, c0)",             # theta in delta
    "t0 == c0",                  # theta in comparison
    "and(t0 < 1, c0 > 0)",       # theta in boolean operand
    "not(t0)",                   # theta in not()
    "select(c0 < t0, t1, t2)",   # theta on the right of the comparison
]


@pytest.mark.parametrize("formula", REJECTED)
def test_guard_rejects_theta_dependent_conditions(formula):
    with pytest.raises(WeightFormulaError) as ei:
        compile_formula(formula)
    msg = str(ei.value)
    assert "theta-independent" in msg
    assert "logistic" in msg


def test_guard_is_a_value_error():
    # WeightFormulaError subclasses ValueError
    with pytest.raises(ValueError):
        compile_formula("t0 == c0")


# ---------------------------------------------------------------------------
# Syntax / arity / unknown-identifier errors
# ---------------------------------------------------------------------------
BAD = [
    "",                # empty
    "   ",             # blank
    "c0 +",            # dangling operator
    "(c0",             # unbalanced
    "c0)",             # unbalanced
    "foo(t0)",         # unknown function
    "x0",              # unknown identifier
    "t",               # 't' without index
    "exp(t0, t1)",     # arity: exp takes 1
    "select(c0, t0)",  # arity: select takes 3
    "exp",             # function used as value
    "t0 t1",           # trailing token
    "== c0",           # leading operator
    "c0 @ t0",         # invalid char
    "delta(c0)",       # arity: delta takes 2
]


@pytest.mark.parametrize("formula", BAD)
def test_syntax_errors_raise(formula):
    with pytest.raises(WeightFormulaError):
        compile_formula(formula)


# ---------------------------------------------------------------------------
# Tape metadata: n_theta / n_coeff / stack_depth / const pool / structure
# ---------------------------------------------------------------------------
def test_index_counts():
    tape = compile_formula("t5 * c3")
    assert tape["n_theta"] == 6
    assert tape["n_coeff"] == 4


def test_stack_depth_tight():
    assert compile_formula("c0*t0 + c1*t1")["stack_depth"] == 3
    assert compile_formula("t0")["stack_depth"] == 1
    assert compile_formula("exp(c0*t0)")["stack_depth"] == 2


def test_const_pool_dedup():
    tape = compile_formula("2*t0 + 2*c0")     # the literal 2 appears twice
    assert tape["consts"].count(2.0) == 1


def test_tape_structure_simple():
    tape = compile_formula("c0*t0")
    assert tape["ops"] == [
        OPCODES["PUSH_COEFF"], 0,
        OPCODES["PUSH_THETA"], 0,
        OPCODES["MUL"],
    ]
    assert tape["src"] == "c0*t0"


def test_tape_is_json_native():
    import json
    tape = compile_formula("exp(c0*t0 + c1*t1) + c2")
    s = json.dumps(tape)              # must serialize with no custom encoder
    back = json.loads(s)
    assert back["ops"] == tape["ops"]
    assert back["consts"] == tape["consts"]
