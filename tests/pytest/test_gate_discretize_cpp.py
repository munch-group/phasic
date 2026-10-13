"""Gate: the C++ discretize (phasic::Graph::discretize, bound as
_discretize) reproduces the phase-0 records of Python discretize exactly
(phase 3 of discretize-cpp-ssot-plan.md).

Every pinned case and error case is re-run with Graph.discretize replaced by
a thin shim that only marshals arguments (type checks, kwargs, numpy state
for the callback, float conversion of its return) and calls C++. Python-only
attributes (is_discrete, indexer, callback, epoch attributes) are the
wrapper's job (phase 4) and are not compared here.
"""
import json
import numbers

import numpy as np
import pytest

from phasic import Graph
import _discretize_pins as pins
from test_gate_discretize_pins import REC, _json, _same

PYTHON_SIDE = {"is_discrete", "indexer_state_length", "has_callback", "attrs"}


def _shim(self, rate, skip_existing=False, **kwargs):
    """What the phase-4 Python wrapper will do around the C++ core."""
    if not callable(rate):
        if (isinstance(rate, (bool, np.bool_))
                or not isinstance(rate, (int, float, np.integer, np.floating))):
            raise TypeError(f"rate must be a number or callable, got {type(rate).__name__}")
        Graph._discretize_check_rate(float(rate))   # C++; before the kwargs check
        if kwargs:
            raise TypeError(
                f"discretize(): keyword arguments {sorted(kwargs)} are passed "
                f"to a callable rate, but rate is a fixed number.")
        arg = float(rate)
    else:
        def arg(state):
            s = np.asarray(state, dtype=np.int32)
            value = rate(s, **kwargs)
            if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
                return float(value)
            arr = np.asarray(value)
            if arr.ndim != 0 or arr.dtype.kind not in 'iuf':
                raise TypeError(
                    f"discretize(rate=callable): the callable must return ONE number, "
                    f"the discretization rate at the given state, but it returned "
                    f"{type(value).__name__} {value!r} at state {[int(x) for x in s]}.")
            return float(arr)
    mode = getattr(self, '_weight_mode', 'linear')
    g, rewards = super(Graph, self)._discretize(arg, skip_existing, bool(self.is_discrete), mode)
    d = Graph(g)
    d.is_discrete = True
    d.rewards = np.array(rewards, dtype=int)
    return d


@pytest.fixture
def via_cpp(monkeypatch):
    monkeypatch.setattr(Graph, "discretize", _shim)


@pytest.mark.parametrize("name", sorted(pins.CASES))
def test_case_through_cpp(via_cpp, name):
    build, call, theta = pins.CASES[name]
    got = _json(pins.snapshot(call(build()), theta))
    want = REC["cases"][name]
    for key in want:
        if key in PYTHON_SIDE:
            continue
        assert _same(got[key], want[key]), f"{name}: field {key!r} differs"


@pytest.mark.parametrize("name", sorted(pins.ERRORS))
def test_error_through_cpp(via_cpp, name):
    call, phrase = pins.ERRORS[name]
    assert pins.error_record(call, phrase) == REC["errors"][name]


def test_callback_argument_through_cpp(via_cpp):
    assert pins.callback_arg_record() == REC["callback_arg"]


@pytest.mark.parametrize("name", sorted(pins.CASES))
def test_theta_layout_through_cpp_matches_python(name, monkeypatch):
    """C++ records the same 'discretization' block (or none) as the current
    Python discretize, incl. skip_existing with nothing new and epoch graphs."""
    build, call, _ = pins.CASES[name]
    want = call(build()).theta_layout
    monkeypatch.setattr(Graph, "discretize", _shim)
    assert call(build()).theta_layout == want
