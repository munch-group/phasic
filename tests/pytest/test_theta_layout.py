"""Graph.theta_layout / set_parameter_name (phase 2 of
discretize-cpp-ssot-plan.md, section 4c): the meaning of every theta slot,
owned by the C++ graph, carried by every copy path, not persisted."""
import pytest

from phasic import Graph
import _discretize_pins as pins


def _kinds(g):
    return [(s['name'], s['kind'], s['epoch']) for s in g.theta_layout]


def test_base_layout_of_a_model():
    g = pins.two_param_chain()
    assert _kinds(g) == [('theta0', 'base', None), ('theta1', 'base', None)]
    assert [s['index'] for s in g.theta_layout] == [0, 1]


def test_constant_graph_has_no_layout_and_cannot_be_named():
    g = pins.coal_constant()
    assert g.theta_layout == []
    with pytest.raises(ValueError, match="no parameterized edges"):
        g.set_parameter_name(0, "x")


def test_discretize_appends_a_discretization_slot():
    d = pins.coal_param().discretize(0.1)
    assert _kinds(d) == [('theta0', 'base', None),
                         ('discretization', 'discretization', None)]
    assert len(d.theta_layout) == d.param_length()


def test_discretize_skip_existing_adds_no_slot():
    d = pins.coal_param().discretize(0.1).discretize(0.1, skip_existing=True)
    assert [s['kind'] for s in d.theta_layout] == ['base', 'discretization']


def test_names_are_kept_by_every_copy_path():
    g = pins.coal_param()
    g.set_parameter_name(0, 'coalescent')
    d = g.discretize(0.1)
    d.set_parameter_name(1, 'mutation')
    want = [('coalescent', 'base', None), ('mutation', 'discretization', None)]
    assert _kinds(d) == want
    assert _kinds(d.clone()) == want
    assert _kinds(d.copy()) == want
    assert _kinds(d._rebuild_with_wider_layout()) == want


def test_not_persisted_reloaded_graph_gets_default():
    d = pins.coal_param().discretize(0.1)
    d.set_parameter_name(1, 'mutation')
    d2 = Graph.from_serialized(d.serialize())
    assert _kinds(d2) == [('theta0', 'base', None), ('theta1', 'base', None)]


def test_epoch_graph_reports_full_length_base_layout():
    """add_epoch does not record its blocks yet (later work in the plan, with
    the add_epoch delegation), so an epoch graph reports base slots only."""
    e = pins.epoch_graph()
    layout = e.theta_layout
    assert len(layout) == e.param_length()
    assert all(s['kind'] == 'base' and s['epoch'] is None for s in layout)


def test_index_out_of_range():
    with pytest.raises(IndexError):
        pins.coal_param().set_parameter_name(5, 'x')


@pytest.mark.parametrize("mode", ['formula', 'callback'])
def test_undefined_for_formula_and_callback(mode):
    g = pins._bad_param_graph_with(mode)
    with pytest.raises(ValueError, match="undefined for weight_mode"):
        g.theta_layout


def test_log_mode_reports_plain_base_slots_keeping_user_names():
    g = pins.coal_param().discretize(0.1)
    g.set_parameter_name(0, 'coalescent')
    g.weight_mode = 'log'
    assert _kinds(g) == [('coalescent', 'base', None), ('theta1', 'base', None)]


def test_block_keeps_its_position_after_further_widening():
    d = pins.coal_param().discretize(0.1)
    w = d._rebuild_with_wider_layout(extra_coeff_slots=1)
    assert [s['kind'] for s in w.theta_layout] == ['base', 'discretization', 'base']
    assert [s['name'] for s in w.theta_layout] == ['theta0', 'discretization', 'theta2']


def test_formula_graph_refused_also_through_a_plain_wrap():
    """The C++ side checks the formula tape, so a Graph(g) wrap (which resets
    the Python weight-mode attribute) is refused too."""
    g = pins._bad_param_graph_with('formula')
    with pytest.raises(ValueError, match="undefined for weight_mode='formula'"):
        Graph(g).theta_layout


def test_recording_a_block_is_validated():
    g = pins.coal_param()
    with pytest.raises(ValueError, match="count must be between 1"):
        g._append_theta_block('x', 'x', 99)
    with pytest.raises(ValueError, match="count must be between 1"):
        g._append_theta_block('x', 'x', 0)
    with pytest.raises(ValueError, match="not parameterized"):
        pins.coal_constant()._append_theta_block('x', 'x', 1)


def test_repeated_block_name_gets_a_suffix():
    d = pins.coal_param().discretize(0.1)
    w = d._rebuild_with_wider_layout(extra_coeff_slots=1)
    w._append_theta_block('discretization', 'discretization', 1)
    assert [s['name'] for s in w.theta_layout] == ['theta0', 'discretization', 'discretization_2']


def test_negative_index_raises_index_error():
    with pytest.raises(IndexError):
        pins.coal_param().set_parameter_name(-1, 'x')


def test_graph_cache_hit_gets_default_layout(tmp_path, monkeypatch):
    import phasic.graph_cache as graph_cache
    monkeypatch.setattr(graph_cache, "DEFAULT_CACHE_DIR", tmp_path)
    cb = pins._coalescent_cb(4, True)
    first = Graph(cb, graph_cache=True)
    first.set_parameter_name(0, 'coalescent')
    hit = Graph(cb, graph_cache=True)
    assert _kinds(hit) == [('theta0', 'base', None)]
