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
    if name in pins.EXPECTED_ERRORS:
        etype, phrase = pins.EXPECTED_ERRORS[name]
        assert pins.error_record(lambda: call(build()), phrase) == \
            {"type": etype, "phrase_in_message": True}
        return
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
    d = builder().discretize(0.1)
    got = _json(pins.serialize_roundtrip(d))
    want = REC["roundtrip"][name]
    # Intended change (phase 5): is_aux now survives the round trip; it was
    # all False before. Everything else is compared exactly.
    for key in want:
        if key in pins.ROUNDTRIP_EXPECTED_CHANGES.get(name, ()):
            continue
        assert _same(got[key], want[key]), f"{name}: field {key!r} differs"
    assert got.keys() == want.keys()
    # the new value: the flags of the recorded discretize output itself
    assert got["is_aux"] == REC["cases"][name]["is_aux"]
    assert any(got["is_aux"]) and not any(want["is_aux"])


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
    if "edges" in pins.EXPECTED_CHANGES[name]:
        # re-discretizing a discretize() result: stored weights equal the
        # first discretize's (per-vertex normalised)
        assert _same(_json(got["edges"]), REC["cases"]["param_fixed"]["edges"])
        return
    assert got["has_callback"] is True
    if "indexer_state_length" in pins.EXPECTED_CHANGES[name]:
        assert got["indexer_state_length"] == REC["cases"]["indexed_param"]["indexer_state_length"]


@pytest.mark.parametrize("builder", [pins.coal_param, pins.coal_constant])
def test_skip_existing_on_discretize_result_returns_the_same_graph(builder):
    """Re-discretizing a discretize() result with skip_existing adds nothing
    and leaves every stored weight as it was (per-vertex normalisation)."""
    d = builder().discretize(0.1)
    d2 = d.discretize(0.1, skip_existing=True)
    edges = lambda g: [pins._edges(v) for v in g.vertices()]
    assert edges(d2) == edges(d)
    assert list(d2.rewards) == list(d.rewards)


def test_native_dph_input_is_refused_but_discretize_result_is_not():
    with pytest.raises(ValueError, match="native discrete \\(DPH\\) graph"):
        pins.native_dph().discretize(0.1)
    d = pins.coal_param().discretize(0.1)
    assert d.is_discrete and d.get_was_dph()
    d.discretize(0.1, skip_existing=True)          # allowed
