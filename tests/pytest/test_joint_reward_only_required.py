"""Several indexer properties require an explicit reward_only (ticket T9).

Each counted property gets its own independent mutation process at the
full rate, one block of reward columns per property. That is right for
separate loci (two-locus joint SFS) and wrong for attributes of the same
lineage (a population label), so joint_prob_graph no longer picks "all
properties" implicitly (user decision 2026-11-20).
"""
import numpy as np
import pytest

from phasic import Graph, Property, StateIndexer, with_ipv
from test_notebook_model_recovery import coalescent_indexed


def _two_property_model():
    idx = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=2),
                                Property('pop', min_value=0, max_value=1)])
    L = idx.lineage
    s0 = np.zeros(idx.state_length, int)
    s0[L.props_to_index(descendants=1, pop=0)] = 2

    @with_ipv(list(s0))
    def cb(state, indexer=None):
        L = indexer.lineage
        i = L.props_to_index(descendants=1, pop=0)
        if state[i] == 2:
            new = np.zeros_like(state)
            new[L.props_to_index(descendants=2, pop=0)] = 1
            return [[new, [1.0]]]
        return []
    return Graph(cb, indexer=idx), idx


def test_several_properties_without_reward_only_refused():
    g, idx = _two_property_model()
    with pytest.raises(ValueError, match=r"several properties.*reward_only=\[\.\.\.\]"):
        g.joint_prob_graph(idx, discrete=True, reward_limit=2)


def test_callback_route_also_requires_reward_only():
    g, idx = _two_property_model()

    def cb(current_state, base_indexer, reward_indexer, current_rewards, **kw):
        return np.zeros(reward_indexer.state_length), 0.0

    with pytest.raises(ValueError, match="several properties"):
        g.joint_prob_graph(idx, discrete=True, reward_limit=2, reward_rates_callback=cb)


@pytest.mark.parametrize("reward_only,n_cols", [(['descendants'], 2),
                                                (['descendants', 'pop'], 4)])
def test_explicit_reward_only_accepted(reward_only, n_cols):
    g, idx = _two_property_model()
    j = g.joint_prob_graph(idx, discrete=True, reward_limit=2, reward_only=reward_only)
    j.update_weights([1.0, 0.5])
    t = j.joint_prob_table()
    assert t.shape[1] - 1 == n_cols
    assert 0 < t['prob'].sum() <= 1


def test_single_property_needs_no_reward_only():
    g, idx = coalescent_indexed(3)
    g.joint_prob_graph(idx, discrete=True, reward_limit=2)


# ---- review follow-ups (2026-11-20) ----

def _three_prop(order):
    # Properties with different value ranges, in a caller-chosen order.
    P = {'a': Property('a', min_value=1, max_value=2),
         'b': Property('b', min_value=0, max_value=3),
         'c': Property('c', min_value=0, max_value=1)}
    idx = StateIndexer(lineage=[P[o] for o in order])
    L = idx.lineage
    s0 = np.zeros(idx.state_length, int)
    s0[L.props_to_index(a=1, b=0, c=0)] = 2

    @with_ipv(list(s0))
    def cb(state, indexer=None):
        L = indexer.lineage
        i = L.props_to_index(a=1, b=0, c=0)
        if state[i] == 2:
            new = np.zeros_like(state)
            new[L.props_to_index(a=2, b=1, c=1)] = 1
            return [[new, [1.0]]]
        return []
    return Graph(cb, indexer=idx), idx


def _canonical_table(order, reward_only):
    g, idx = _three_prop(order)
    j = g.joint_prob_graph(idx, reward_only=reward_only, tot_reward_limit=2,
                           discrete=False)
    j.update_weights([1.0, 0.7])
    t = j.joint_prob_table()
    t = t[sorted(t.columns)]
    return t.sort_values(list(t.columns)).reset_index(drop=True)


@pytest.mark.parametrize("order", [('c', 'b', 'a'), ('b', 'a', 'c'), ('b', 'c', 'a')])
@pytest.mark.parametrize("reward_only", [['a', 'b'], ['b', 'a'], ['a', 'c']])
def test_subset_merge_independent_of_property_order(order, reward_only):
    # The merge zipped the value ranges (property order) with the SORTED
    # reward_only, so a non-alphabetical order raised "value out of range".
    ref = _canonical_table(('a', 'b', 'c'), reward_only)
    got = _canonical_table(order, reward_only)
    assert list(got.columns) == list(ref.columns)
    np.testing.assert_allclose(got.to_numpy(float), ref.to_numpy(float), rtol=1e-12)


@pytest.mark.parametrize("bad,exc,match", [
    (['descendants', 'popp'], ValueError, r"\['popp'\] are not properties"),
    (['descendant'], ValueError, "are not properties"),
    ([], ValueError, "reward_only is empty"),
    ('descendants', TypeError, "list of property names"),
    ([0], TypeError, "list of property names"),
])
def test_bad_reward_only_refused(bad, exc, match):
    g, idx = _two_property_model()
    with pytest.raises(exc, match=match):
        g.joint_prob_graph(idx, discrete=True, reward_limit=2, reward_only=bad)


def test_dict_reward_limit_refused_on_default_path():
    # It used to build a graph sending every mutation to trash.
    g, idx = coalescent_indexed(3)
    with pytest.raises(ValueError, match=r"dict \(per-property\) reward_limit"):
        g.joint_prob_graph(idx, discrete=False, reward_limit={'descendants': 2})
