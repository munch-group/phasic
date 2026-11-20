"""prior=None must not fall back silently to a standard-normal prior (T5).

When the default data-informed prior could not be built, svgd warned
and continued with a standard normal on the unconstrained scale, the
epoch path did the same with no message at all, and mcmc did it
silently too. All three now raise a ValueError naming the cause (user
decision 2026-11-20, joint-prob-followups-plan.md ticket T5).
"""
import numpy as np
import pytest

import phasic.svgd
import phasic.probability_matching
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph


class _Boom(RuntimeError):
    pass


def _boom(*args, **kwargs):
    raise _Boom("prior builder failed on purpose")


@pytest.fixture(scope="module")
def joint():
    g, idx = coalescent_indexed(3)
    j = g.joint_prob_graph(idx, discrete=False, reward_limit=2)
    jd = g.joint_prob_graph(idx, discrete=True, reward_limit=2)
    jd.update_weights([1.0, 0.5])
    rows = jd.joint_prob_table().iloc[:, :-1].to_numpy().astype(int).tolist()
    obs = [rows[i] for i in np.random.RandomState(0).randint(0, len(rows), 40)]
    return j, obs


def test_svgd_raises_when_dataprior_fails(joint, monkeypatch):
    j, obs = joint
    monkeypatch.setattr(phasic.svgd, "DataPrior", _boom)
    with pytest.raises(ValueError, match="default data-informed prior.*_Boom.*on purpose") as ei:
        j.svgd(obs, n_particles=4, n_iterations=2, fixed=[(1, 0.5)], progress=False)
    assert isinstance(ei.value.__cause__, _Boom)


def test_epoch_path_raises_when_probability_matching_fails(joint, monkeypatch):
    j, obs = joint
    monkeypatch.setattr(phasic.probability_matching, "probability_matching", _boom)
    with pytest.raises(ValueError, match="default data-informed epoch prior.*on purpose"):
        j.epoch_model(obs, [0.0, 0.5])


def test_explicit_prior_skips_the_default(joint, monkeypatch):
    # With prior= given, the failing default builder is never called.
    from phasic import GaussPrior
    j, obs = joint
    monkeypatch.setattr(phasic.probability_matching, "probability_matching", _boom)
    em = j.epoch_model(obs, [0.0, 0.5], prior=GaussPrior(ci=[0.5, 2.0]))
    assert em.theta_dim == 4


def test_mcmc_raises_when_dataprior_fails(monkeypatch):
    g = coalescent_graph(3)
    g.update_weights([1.0])
    data = np.asarray(g.sample(50), dtype=float)
    monkeypatch.setattr(phasic.svgd, "DataPrior", _boom)
    with pytest.raises(ValueError, match="default data-informed prior.*on purpose"):
        g.mcmc(data, n_samples=10, n_chains=1, burn_in=5)


def test_mcmc_default_prior_receives_rewards(monkeypatch):
    # T12: Graph.mcmc built its default DataPrior without rewards= (svgd
    # passes it), so with rewards the prior was fitted to the unrewarded
    # model.
    g = coalescent_graph(3)
    g.update_weights([1.0])
    data = np.asarray(g.sample(50), dtype=float)
    rewards = np.ones(g.vertices_length())
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise _Boom("stop after recording the call")

    monkeypatch.setattr(phasic.svgd, "DataPrior", spy)
    with pytest.raises(ValueError):
        g.mcmc(data, rewards=rewards, n_samples=10, n_chains=1, burn_in=5)
    assert seen.get("rewards") is not None
    np.testing.assert_array_equal(np.asarray(seen["rewards"]), rewards)
