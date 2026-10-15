"""serialize() carries the per-vertex aux flag (phase 5 of
discretize-cpp-ssot-plan.md, decision 1b): additive key 'is_aux'."""
import numpy as np
import pytest

from phasic import Graph
import _discretize_pins as pins


@pytest.mark.parametrize("builder", [pins.coal_param, pins.coal_constant])
def test_is_aux_survives_round_trip(builder):
    d = builder().discretize(0.1)
    ser = d.serialize()
    assert list(ser["is_aux"]) == [v.is_aux() for v in d.vertices()]
    d2 = Graph.from_serialized(ser)
    assert [v.is_aux() for v in d2.vertices()] == [v.is_aux() for v in d.vertices()]


def test_data_without_key_loads_without_aux_flags():
    """Older caches have no 'is_aux' key: loaded as before (no flags)."""
    ser = dict(pins.coal_param().discretize(0.1).serialize())
    ser.pop("is_aux")
    d2 = Graph.from_serialized(ser)
    assert not any(v.is_aux() for v in d2.vertices())


def test_wrong_length_is_refused():
    ser = dict(pins.coal_param().discretize(0.1).serialize())
    ser["is_aux"] = np.array([True])
    with pytest.raises(ValueError, match="'is_aux' has 1 entries"):
        Graph.from_serialized(ser)


def test_skip_existing_on_reloaded_graph_adds_nothing():
    """A reloaded parameterized discretized graph keeps its aux flags, so
    re-discretizing it with skip_existing works as on the original: the aux
    back-edges (constant edges on aux vertices) are not refused, and no
    vertex or slot is added. The skip rule itself (zero state + back edge)
    is unchanged."""
    d = pins.coal_param().discretize(0.1)
    d2 = Graph.from_serialized(d.serialize())
    d3 = d2.discretize(0.1, skip_existing=True)
    assert d3.vertices_length() == d.vertices_length()
    assert d3.param_length() == d.param_length()


def test_without_flags_reloaded_graph_is_refused_as_before():
    """Behaviour change pinned (phase 5): data without 'is_aux' (an older
    cache) reloads without flags, and then the aux back-edges count as
    constant edges on transient vertices, which discretize refuses -- what
    every reloaded discretized parameterized graph did before phase 5."""
    ser = dict(pins.coal_param().discretize(0.1).serialize())
    ser.pop("is_aux")
    d2 = Graph.from_serialized(ser)
    with pytest.raises(ValueError, match="coefficient-less constant"):
        d2.discretize(0.1, skip_existing=True)


def test_is_aux_survives_the_graph_cache_json_path():
    """The graph cache stores serialize() output as JSON (lists) and loads it
    back with np.array; the flags and their bool dtype survive."""
    import json
    from phasic.graph_cache import _serialize_value, _deserialize_value
    d = pins.coal_param().discretize(0.1)
    blob = json.dumps(_serialize_value(d.serialize()))
    data = _deserialize_value(json.loads(blob))
    assert np.asarray(data["is_aux"]).dtype == bool
    d2 = Graph.from_serialized(data)
    assert [v.is_aux() for v in d2.vertices()] == [v.is_aux() for v in d.vertices()]


@pytest.mark.parametrize("bad", [np.array([2] * 10), np.array([0.5] * 10),
                                 np.array(["no"] * 10), np.array([[True] * 10])])
def test_is_aux_values_must_be_boolean(bad):
    ser = dict(pins.coal_param().discretize(0.1).serialize())
    ser["is_aux"] = bad
    with pytest.raises(ValueError, match="'is_aux' must be a 1-D array of booleans"):
        Graph.from_serialized(ser)
