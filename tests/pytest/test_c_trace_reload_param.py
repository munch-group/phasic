"""Regression test: PARAM ops must decode with param_idx set on C-trace reload.

_c_trace_to_python rebuilt a PARAM op as Operation(op_type=PARAM,
operands=[param_idx]), leaving the dedicated param_idx field None. Both trace
executors read op.param_idx (never op.operands), so replay did
extended_params[None] -> a spurious newaxis, crashing multi-parameter models
with "setting an array element with a sequence".

The C JSON trace cache is only written by the cross-language C API (the Python
writer uses pickle), so we drive the exact decoder _c_trace_to_python directly
with the C getters monkeypatched.
"""
import numpy as np
import pytest

import phasic.trace_serialization as ts
from phasic.trace_elimination import OpType


@pytest.fixture
def _mock_c_trace(monkeypatch):
    """Two-op trace: op0 = PARAM(param_idx=1), op1 = CONST(0.5)."""
    ops = [
        {"op_type": 1, "param_idx": 1, "const_value": 0.0,
         "coefficients": [], "operands": []},          # PTD_OP_PARAM
        {"op_type": 0, "param_idx": -1, "const_value": 0.5,
         "coefficients": [], "operands": []},          # PTD_OP_CONST
    ]

    def setattrs(**kw):
        for name, fn in kw.items():
            monkeypatch.setattr(ts, name, fn, raising=False)

    setattrs(
        _c_trace_get_n_vertices=lambda p: 2,
        _c_trace_get_state_length=lambda p: 1,
        _c_trace_get_param_length=lambda p: 2,
        _c_trace_get_starting_vertex_idx=lambda p: 0,
        _c_trace_get_is_discrete=lambda p: False,
        _c_trace_get_operations_length=lambda p: len(ops),
        _c_trace_get_states=lambda p: np.array([[0], [1]], dtype=np.int32),
        _c_trace_get_vertex_rates=lambda p: np.array([0, 1], dtype=np.int32),
        _c_trace_get_edge_probs=lambda p: [[], []],
        _c_trace_get_vertex_targets=lambda p: [[], []],
        _c_trace_get_operation=lambda p, i: ops[i],
    )


def test_param_op_decodes_into_param_idx_field(_mock_c_trace):
    trace = ts._c_trace_to_python(1)
    assert trace is not None, "_c_trace_to_python returned None (decode failed)"

    param_ops = [op for op in trace.operations if op.op_type == OpType.PARAM]
    assert len(param_ops) == 1
    op = param_ops[0]

    # The fix: index lands in the dedicated field that executors read...
    assert op.param_idx == 1
    # ...and NOT stuffed into operands (which used to leave param_idx=None).
    assert list(op.operands) == []


def test_replay_indexes_params_without_newaxis(_mock_c_trace):
    """param_idx=None caused extended_params[None]; a real int must index."""
    trace = ts._c_trace_to_python(1)
    op = next(o for o in trace.operations if o.op_type == OpType.PARAM)

    theta = np.array([0.3, 0.7])
    # This is exactly what evaluate_trace does at the PARAM branch.
    value = theta[op.param_idx]
    assert np.ndim(value) == 0 and value == pytest.approx(0.7)
