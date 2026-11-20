"""joint_prob_table() on a continuous joint graph reports probabilities (T3).

Before ticket T3 (joint-prob-followups-plan.md) the table read the
expected sojourn time at each t-vertex, which on a continuous graph is
probability / (the t-vertex's rate into absorption) = probability / mu.
The raw (unnormalised) table must equal the closed-form outcome
probabilities on both continuous and discrete graphs.
"""
import numpy as np
import pytest

from test_notebook_model_recovery import coalescent_indexed
from test_joint_prob_tot_cap import _p_n2, _p_n3, _CAPS

_CASES = {2: (_p_n2, 1.0, 0.7), 3: (_p_n3, 1.1, 0.6)}


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('kw', _CAPS, ids=lambda d: ','.join(f'{k}={v}' for k, v in d.items()))
@pytest.mark.parametrize('n', [2, 3])
def test_table_equals_closed_form(n, kw, discrete):
    cf, c, mu = _CASES[n]
    g, idx = coalescent_indexed(n)
    j = g.joint_prob_graph(idx, discrete=discrete, **kw)
    j.update_weights([c, mu])
    t = j.joint_prob_table()
    feats = t.iloc[:, :n - 1].to_numpy().astype(int)
    assert np.all(t.iloc[:, n - 1:-1].to_numpy() == 0)  # size-n lineage never mutates
    ref = np.array([cf(*r, c, mu) for r in feats])
    np.testing.assert_allclose(t['prob'].to_numpy(), ref, rtol=1e-12)


@pytest.mark.parametrize('mu', [0.3, 0.7, 2.5])
def test_continuous_and_discrete_tables_agree(mu):
    g, idx = coalescent_indexed(3)
    tabs = []
    for discrete in (False, True):
        j = g.joint_prob_graph(idx, discrete=discrete, reward_limit=3)
        j.update_weights([1.0, mu])
        tabs.append(j.joint_prob_table().sort_index())
    cols = list(tabs[0].columns[:-1])
    a = tabs[0].set_index(cols)['prob'].sort_index()
    b = tabs[1].set_index(cols)['prob'].sort_index()
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)
