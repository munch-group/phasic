"""Gate: Python ``Graph.discretize`` matches the records pinned on the
starting commit of the discretize C++ refactor (discretize-cpp-ssot-plan.md
phase 0). Structure, coefficients, weights and distribution values are
compared EXACTLY: the C++ port must reproduce them bit for bit. Error
types and key phrases are compared too (frozen API).

Records: tests/pytest/data/discretize_pins.json, written by
tests/pytest/_discretize_pins.py. Regenerate only for an intended change.
"""
import json
import math

import pytest

import _discretize_pins as pins

with open(pins.RECORDS) as _f:
    REC = json.load(_f)


def _same(a, b):
    """Bit-exact equality: same JSON types (bool, int and float distinct) and
    identical float bits (so -0.0 != 0.0, NaN == NaN)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a.hex() == b.hex()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def _json(x):
    return json.loads(json.dumps(x, sort_keys=True))


@pytest.mark.parametrize("name", sorted(pins.CASES))
def test_case_matches_record(name):
    build, call, theta = pins.CASES[name]
    got = _json(pins.snapshot(call(build()), theta))
    want = REC["cases"][name]
    allowed = pins.EXPECTED_CHANGES.get(name, ())
    for key in want:
        if key in allowed:
            continue
        assert _same(got[key], want[key]), f"{name}: field {key!r} differs"
    assert got.keys() == want.keys()


@pytest.mark.parametrize("name", sorted(pins.INPLACE_CASES))
def test_inplace_matches_record(name):
    build, rate = pins.INPLACE_CASES[name]
    g = build().discretize(rate)
    rewards = g.rewards
    got = _json({"rewards": [int(x) for x in rewards],
                 "states": [[int(x) for x in v.state()] for v in g.vertices()],
                 "edges": [pins._edges(v) for v in g.vertices()],
                 "expected_aux_visits": float(g.expectation_discrete(rewards))})
    assert _same(got, REC["inplace"][name])


@pytest.mark.parametrize("name", sorted(pins.ERRORS))
def test_error_matches_record(name):
    call, phrase = pins.ERRORS[name]
    got = pins.error_record(call, phrase)
    want = REC["errors"][name]
    assert want["type"] is not None and want["phrase_in_message"], \
        f"record for {name} does not show the expected error"
    assert got == want, f"{name}: {got} != {want}"


@pytest.mark.parametrize("name,builder", [("param_fixed", pins.coal_param),
                                          ("constant_fixed", pins.coal_constant)])
def test_roundtrip_matches_record(name, builder):
    got = _json(pins.serialize_roundtrip(builder().discretize(0.1)))
    assert _same(got, REC["roundtrip"][name])


def test_epoch_output_matches_record():
    got = _json(pins.snapshot_graph(pins.epoch_graph()))
    assert _same(got, REC["epoch_output"])


def test_callback_argument_matches_record():
    """What a rate callback receives (type, dtype, ndim, writeable) is part
    of the frozen API: callbacks use state.sum(), slicing, etc."""
    assert pins.callback_arg_record() == REC["callback_arg"]


@pytest.mark.parametrize("name", sorted(pins.EXPECTED_CHANGES))
def test_intended_changes_have_their_new_values(name):
    """Decision 8: a discretized constant graph keeps its callback (and its
    indexer), like a discretized parameterized graph."""
    build, call, theta = pins.CASES[name]
    got = pins.snapshot(call(build()), theta)
    assert got["has_callback"] is True
    if "indexer_state_length" in pins.EXPECTED_CHANGES[name]:
        assert got["indexer_state_length"] == REC["cases"]["indexed_param"]["indexer_state_length"]
