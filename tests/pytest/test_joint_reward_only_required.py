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
