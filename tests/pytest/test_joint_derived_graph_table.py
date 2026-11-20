"""joint_prob_table() refuses graphs DERIVED from a joint-prob graph.

joint_stop_prob_graph() and joint_sojourn_graph() copy the joint-prob
metadata (indexer, rewarded properties) but change the outcome
structure: t-vertices end in trap loops (stop-probability graph) and
trash edges are redirected to absorption (both). Before ticket T7
(joint-prob-followups-plan.md) the table on such a graph listed the
wrong vertices as outcomes, with NaN probabilities, and svgd mapped
observations through it (KeyError or "probabilities contain NaN").
"""
import numpy as np
import pytest

from test_notebook_model_recovery import coalescent_indexed


@pytest.fixture(scope="module")
def graphs():
    g, idx = coalescent_indexed(3)
    j = g.joint_prob_graph(idx, discrete=False, reward_limit=2)
    jd = g.joint_prob_graph(idx, discrete=True, reward_limit=2)
    jd.update_weights([1.0, 0.5])
    rows = jd.joint_prob_table().iloc[:, :-1].to_numpy().astype(int).tolist()
    obs = [rows[i] for i in np.random.RandomState(0).randint(0, len(rows), 40)]
    return dict(jpg=j, jsp=j.joint_stop_prob_graph(),
                soj=j.joint_sojourn_graph(), obs=obs, n_outcomes=len(rows))


@pytest.mark.parametrize("kind,builder", [("jsp", "joint_stop_prob_graph"),
                                          ("soj", "joint_sojourn_graph")])
def test_table_refused_on_derived_graph(graphs, kind, builder):
    with pytest.raises(ValueError, match=rf"{builder}\(\).*source joint-probability graph"):
        graphs[kind].joint_prob_table()


@pytest.mark.parametrize("kind,builder", [("jsp", "joint_stop_prob_graph"),
                                          ("soj", "joint_sojourn_graph")])
def test_svgd_refused_on_derived_graph(graphs, kind, builder):
    with pytest.raises(ValueError, match=rf"{builder}\(\)"):
        graphs[kind].svgd(graphs["obs"], n_particles=4, n_iterations=2,
                          fixed=[(1, 0.5)], progress=False)


def test_source_graph_table_unaffected(graphs):
    t = graphs["jpg"].joint_prob_table()
    assert len(t) == graphs["n_outcomes"]
    assert np.all(np.isfinite(t["prob"].to_numpy()))
