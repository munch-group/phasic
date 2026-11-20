"""joint_prob_graph(reward_rates_callback=...) works again (ticket T2).

The route forwarded to a reference implementation that had been
commented out in the same commit that introduced the fast builder
(89661051, 2026-05-21), so every callback raised AttributeError. The
callback now runs inside the fast builder: it supplies the mutation
rates per reward counter (and optional extra trash rate), the builder
validates them and enforces reward_limit / tot_reward_limit itself.
"""
import numpy as np
import pytest

from test_notebook_model_recovery import coalescent_indexed
from test_joint_prob_tot_cap import _p_n2


def _default_like(current_state, base_indexer, reward_indexer, current_rewards,
                  mutation_rate=1.0, reward_limit=None, tot_reward_limit=np.inf):
    # coalescent_indexed: one property, so base index i <-> reward counter i.
    n = reward_indexer.state_length
    return np.asarray(current_state[:n], dtype=float) * mutation_rate, 0.0


def _scaled(factor):
    def cb(current_state, base_indexer, reward_indexer, current_rewards, **kw):
        r, t = _default_like(current_state, base_indexer, reward_indexer,
                             current_rewards, **kw)
        return factor * r, t
    return cb


def _structure(j):
    # The full serialization: vertices, insertion order, edges, coefficients.
    import json
    from phasic.ffi_wrappers import _make_json_serializable
    return json.dumps(_make_json_serializable(j.serialize()), sort_keys=True)


@pytest.mark.parametrize("kw", [dict(reward_limit=2), dict(tot_reward_limit=2),
                                dict(reward_limit=2, tot_reward_limit=3)],
                         ids=["rl2", "tot2", "rl2_tot3"])
@pytest.mark.parametrize("discrete", [False, True])
def test_default_like_callback_builds_the_same_graph(kw, discrete):
    # The callback never sends anything to trash itself: the builder's cap
    # enforcement must reproduce the default graph exactly (and terminate).
    g, idx = coalescent_indexed(3)
    a = g.joint_prob_graph(idx, discrete=discrete, **kw)
    b = g.joint_prob_graph(idx, discrete=discrete, reward_rates_callback=_default_like, **kw)
    assert _structure(a) == _structure(b)


def test_scaled_rates_match_closed_form():
    # Doubling every rate is the same model with mutation rate 2*mu.
    c, mu = 1.0, 0.35
    g, idx = coalescent_indexed(2)
    j = g.joint_prob_graph(idx, discrete=False, tot_reward_limit=3,
                           reward_rates_callback=_scaled(2.0))
    j.update_weights([c, mu])
    t = j.joint_prob_table()
    k = t.iloc[:, 0].to_numpy().astype(int)
    np.testing.assert_allclose(t['prob'].to_numpy(), [_p_n2(x, c, 2 * mu) for x in k],
                               rtol=1e-12)


def test_callback_not_called_at_outcome_vertices():
    g, idx = coalescent_indexed(2)
    seen = []

    def cb(current_state, base_indexer, reward_indexer, current_rewards, **kw):
        seen.append(tuple(int(x) for x in current_state))
        return _default_like(current_state, base_indexer, reward_indexer, current_rewards, **kw)

    g.joint_prob_graph(idx, discrete=False, reward_limit=2, reward_rates_callback=cb)
    absorbing = {tuple(int(x) for x in v.state()) for v in g.vertices()
                 if not v.edges() and v.index() != g.starting_vertex().index()}
    assert seen and not (set(seen) & absorbing)


@pytest.mark.parametrize("ret,exc,match", [
    (lambda n: (np.ones(n + 1), 0.0), TypeError, "shape"),
    (lambda n: (np.ones(n), [0.0, 1.0]), TypeError, "scalar"),
    (lambda n: np.ones(n), TypeError, "return \\(rates, trash_rate\\)"),
    (lambda n: (-np.ones(n), 0.0), ValueError, "non-negative"),
    (lambda n: (np.full(n, np.nan), 0.0), ValueError, "finite"),
    (lambda n: (np.ones(n), -1.0), ValueError, "non-negative"),
], ids=["long", "trash_vector", "not_tuple", "negative", "nan", "neg_trash"])
def test_bad_callback_output_is_refused(ret, exc, match):
    g, idx = coalescent_indexed(2)

    def cb(current_state, base_indexer, reward_indexer, current_rewards, **kw):
        return ret(reward_indexer.state_length)

    with pytest.raises(exc, match=match):
        g.joint_prob_graph(idx, discrete=False, reward_limit=2, reward_rates_callback=cb)


def test_dict_reward_limit_refused_with_callback():
    g, idx = coalescent_indexed(2)
    with pytest.raises(ValueError, match="dict reward_limit"):
        g.joint_prob_graph(idx, discrete=False, reward_limit={'descendants': 2},
                           reward_rates_callback=_default_like)


def test_epoch_graph_callback_sees_epoch_slot():
    # On an add_epoch graph the base part of the state includes the epoch
    # slot (documented); a callback that ignores it reproduces the default.
    g, _ = coalescent_indexed(2)
    ge = g.add_epoch(0.5)
    lens = set()

    def cb(current_state, base_indexer, reward_indexer, current_rewards, **kw):
        lens.add(len(current_state))
        return _default_like(current_state, base_indexer, reward_indexer, current_rewards, **kw)

    a = ge.joint_prob_graph(ge._indexer, discrete=True, reward_limit=2)
    b = ge.joint_prob_graph(ge._indexer, discrete=True, reward_limit=2, reward_rates_callback=cb)
    assert lens == {ge.state_length()} and ge.state_length() == g.state_length() + 1
    assert _structure(a) == _structure(b)
