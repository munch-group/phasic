"""Input validation of the internal probability_matching function.

It takes vertex indices (Graph.probability_matching maps feature-count
tuples to them). Passing the tuples directly used to be flattened by
np.unique and fitted as if the counts were vertex indices, silently
giving a wrong estimate.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_notebook_model_recovery import coalescent_indexed  # noqa: E402

from phasic.probability_matching import probability_matching  # noqa: E402


@pytest.fixture(scope="module")
def jpg_and_obs():
    g, idx = coalescent_indexed(4)
    jc = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    jc.update_weights([5.0, 3.0])
    table = jc.joint_prob_table()
    p = table["prob"].to_numpy()
    p = p / p.sum()
    rows = table.iloc[:, :-1].to_numpy()
    pick = np.random.default_rng(0).choice(len(p), 2000, p=p)
    obs = [tuple(int(x) for x in rows[k]) for k in pick]
    return jc, obs


def test_tuples_rejected(jpg_and_obs):
    jc, obs = jpg_and_obs
    with pytest.raises(ValueError, match="1-D sequence of vertex indices"):
        probability_matching(jc, obs, verbose=False)


def test_float_indices_rejected(jpg_and_obs):
    jc, obs = jpg_and_obs
    idx = jc._map_joint_observations_to_indices(obs)
    with pytest.raises(TypeError, match="integer vertex indices"):
        probability_matching(jc, np.asarray(idx, dtype=float), verbose=False)


def test_non_outcome_vertex_rejected(jpg_and_obs):
    jc, _ = jpg_and_obs
    # Vertex 0 is the starting vertex, never an outcome vertex.
    with pytest.raises(ValueError, match="not outcome vertices"):
        probability_matching(jc, [0, 0, 0], verbose=False)


def test_empty_rejected(jpg_and_obs):
    jc, _ = jpg_and_obs
    with pytest.raises(ValueError, match="empty"):
        probability_matching(jc, np.array([], dtype=np.int64), verbose=False)


def test_mapped_indices_fit_ratio(jpg_and_obs):
    """The wrapper path still works and recovers the identified ratio mu/c."""
    jc, obs = jpg_and_obs
    r = jc.probability_matching(obs, verbose=False)
    c, mu = np.ravel(r.theta)
    assert abs(mu / c - 0.6) < 0.05
