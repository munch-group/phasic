"""joint_prob_graph(event_rate_coefficient=...) and its deprecated alias.

``event_rate_coefficient`` (formerly ``mutation_rate``) is a FIXED scale
stored in the coefficients of the event edges (the edges that increase a
counter by one), so the effective event rate is
``event_rate_coefficient * theta[-1]``. It is not replaced by
``update_weights``. The old name still works with a DeprecationWarning.
"""
import json
import warnings

import numpy as np
import pytest

from test_notebook_model_recovery import coalescent_indexed


def _structure(j):
    from phasic.ffi_wrappers import _make_json_serializable
    return json.dumps(_make_json_serializable(j.serialize()), sort_keys=True)


def _probs(g, idx, theta, discrete, **kw):
    j = g.joint_prob_graph(idx, reward_limit=3, discrete=discrete, **kw)
    j.update_weights(theta)
    t = j.joint_prob_table()
    return t.iloc[:, -1].to_numpy()


@pytest.mark.parametrize("discrete", [True, False])
def test_coefficient_scales_last_theta_entry(discrete):
    # coefficient 2 at theta[-1]=1 is the same model as coefficient 1 at
    # theta[-1]=2, and differs from coefficient 1 at theta[-1]=1.
    g, idx = coalescent_indexed(4)
    scaled = _probs(g, idx, [1.0, 1.0], discrete, event_rate_coefficient=2.0)
    moved = _probs(g, idx, [1.0, 2.0], discrete)
    unit = _probs(g, idx, [1.0, 1.0], discrete)
    np.testing.assert_allclose(scaled, moved, rtol=0, atol=1e-14)
    assert np.abs(scaled - unit).max() > 1e-3


@pytest.mark.parametrize("discrete", [True, False])
def test_default_is_unit_coefficient(discrete):
    g, idx = coalescent_indexed(3)
    a = g.joint_prob_graph(idx, reward_limit=2, discrete=discrete)
    b = g.joint_prob_graph(idx, reward_limit=2, discrete=discrete,
                           event_rate_coefficient=1.0)
    assert _structure(a) == _structure(b)


def test_new_name_and_default_emit_no_deprecation_warning():
    g, idx = coalescent_indexed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        g.joint_prob_graph(idx, reward_limit=2)
        g.joint_prob_graph(idx, reward_limit=2, event_rate_coefficient=0.5)


@pytest.mark.parametrize("discrete", [True, False])
def test_mutation_rate_alias_warns_and_builds_same_graph(discrete):
    g, idx = coalescent_indexed(3)
    new = g.joint_prob_graph(idx, reward_limit=2, discrete=discrete,
                             event_rate_coefficient=0.35)
    with pytest.warns(DeprecationWarning, match="event_rate_coefficient"):
        old = g.joint_prob_graph(idx, reward_limit=2, discrete=discrete,
                                 mutation_rate=0.35)
    assert _structure(new) == _structure(old)


def test_both_names_raise():
    g, idx = coalescent_indexed(3)
    with pytest.raises(TypeError, match="both"):
        g.joint_prob_graph(idx, reward_limit=2, event_rate_coefficient=1.0,
                           mutation_rate=1.0)


@pytest.mark.parametrize("bad, exc", [
    (0.0, ValueError), (-1.0, ValueError), (np.inf, ValueError),
    (np.nan, ValueError), (True, TypeError), ("1", TypeError),
])
def test_invalid_coefficient_rejected(bad, exc):
    g, idx = coalescent_indexed(3)
    with pytest.raises(exc, match="event_rate_coefficient"):
        g.joint_prob_graph(idx, reward_limit=2, event_rate_coefficient=bad)


def test_invalid_value_through_alias_is_rejected_too():
    g, idx = coalescent_indexed(3)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="event_rate_coefficient"):
            g.joint_prob_graph(idx, reward_limit=2, mutation_rate=-1.0)


def test_callback_receives_resolved_coefficient_as_mutation_rate():
    # The callback keyword keeps its old name for compatibility.
    seen = set()

    def cb(current_state, base_indexer, reward_indexer, current_rewards,
           mutation_rate=None, reward_limit=None, tot_reward_limit=np.inf):
        seen.add(mutation_rate)
        n = reward_indexer.state_length
        return np.asarray(current_state[:n], dtype=float) * mutation_rate, 0.0

    g, idx = coalescent_indexed(3)
    j = g.joint_prob_graph(idx, reward_limit=2, event_rate_coefficient=0.7,
                           reward_rates_callback=cb)
    ref = g.joint_prob_graph(idx, reward_limit=2, event_rate_coefficient=0.7)
    assert seen == {0.7}
    assert _structure(j) == _structure(ref)
