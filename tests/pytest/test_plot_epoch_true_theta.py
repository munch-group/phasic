"""SVGD plotting methods accept a per-epoch true_theta / param_names.

An epoch fit has a flat theta of n_epochs * param_length values, epoch by
epoch. plot_trace (and the other plots taking true_theta) indexed a
per-model-parameter true_theta by flat index and raised IndexError. They now
repeat a true_theta / param_names of length param_length for every epoch,
as a prior of that length is, and raise a clear ValueError for any other
length.
"""
from itertools import combinations_with_replacement

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phasic import ExpStepSize, Graph, GaussPrior, Property, StateIndexer, with_ipv

N_EPOCHS = 2


@pytest.fixture(scope="module")
def epoch_fit():
    nr = 3
    indexer = StateIndexer(
        lineage=[Property('descendants', min_value=1, max_value=nr)])

    @with_ipv([nr] + [0] * (nr - 1))
    def coal(state):
        tr = []
        for i, j in combinations_with_replacement(
                range(indexer.lineage.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            d = (indexer.lineage.index_to_props(i).descendants
                 + indexer.lineage.index_to_props(j).descendants)
            new[indexer.lineage.props_to_index(descendants=d)] += 1
            tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return tr

    jg = Graph(coal).joint_prob_graph(indexer, reward_limit=2, discrete=False)
    obs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [1, 1, 0]] * 2
    svgd = jg.svgd(
        obs,
        fixed=[(1, 1.0)],
        prior=GaussPrior(ci=[0.2, 3.0]),
        learning_rate=ExpStepSize(first_step=0.01, last_step=0.005, tau=5.0),
        epoch_starts=[0.0, 0.5],
        n_iterations=3,
        n_particles=4,
        progress=False,
        return_history=True,
        quiet_assumptions=True,
    )
    assert svgd.theta_dim == 2 * N_EPOCHS
    return svgd


def _close(out):
    plt.close('all')


def test_resolve_true_theta_repeats_per_epoch(epoch_fit):
    np.testing.assert_array_equal(
        epoch_fit._resolve_true_theta([1.5, 1.0]), [1.5, 1.0, 1.5, 1.0])
    np.testing.assert_array_equal(
        epoch_fit._resolve_true_theta([1.5, 1.0, 2.0, 1.0]),
        [1.5, 1.0, 2.0, 1.0])
    assert epoch_fit._resolve_true_theta(None) is None


def test_resolve_param_names_repeats_per_epoch(epoch_fit):
    assert epoch_fit._resolve_param_names(['c', 'u']) == [
        'c (epoch 1)', 'u (epoch 1)', 'c (epoch 2)', 'u (epoch 2)']
    assert epoch_fit._resolve_param_names(list('abcd')) == list('abcd')
    assert epoch_fit._resolve_param_names(None) is None


@pytest.mark.parametrize("bad", [[1.0], [1.0, 1.0, 1.0], [1.0] * 5])
def test_wrong_length_true_theta_raises(epoch_fit, bad):
    with pytest.raises(ValueError, match="true_theta has"):
        epoch_fit.plot_trace(true_theta=bad, return_fig=True)


def test_wrong_length_param_names_raises(epoch_fit):
    with pytest.raises(ValueError, match="param_names has"):
        epoch_fit.plot_trace(param_names=['a', 'b', 'c'], return_fig=True)


@pytest.mark.parametrize("true_theta", [[1.5, 1.0], [1.5, 1.0, 2.0, 1.0]],
                         ids=["per_epoch", "flat"])
def test_plots_accept_epoch_true_theta(epoch_fit, true_theta):
    names = ['coal', 'mut']
    _close(epoch_fit.plot_trace(true_theta=true_theta, param_names=names,
                                return_fig=True))
    _close(epoch_fit.plot_posterior(true_theta=true_theta, param_names=names,
                                    return_fig=True))
    _close(epoch_fit.plot_ci(true_theta=true_theta, return_fig=True))
    _close(epoch_fit.plot_pairwise(true_theta=true_theta, param_names=names,
                                   return_fig=True))
    _close(epoch_fit.plot_trace_accel(param_names=names, return_fig=True))


def test_trace_true_line_is_labelled_true(epoch_fit):
    fig, axes = epoch_fit.plot_trace(true_theta=[1.5, 1.0], return_fig=True)
    labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
    plt.close(fig)
    assert any(lbl == 'True = 1.5' for lbl in labels), labels


def test_non_epoch_fit_unchanged_and_strict():
    # Non-epoch fit: full-length true_theta works; a wrong length raises
    # ValueError (it used to raise IndexError or silently ignore extras).
    from phasic.svgd import SVGD

    class _Stub:
        theta_dim = 2
        model = object()

        def _plot_epoch_layout(self):
            return SVGD._plot_epoch_layout(self)

    s = _Stub()
    assert s._plot_epoch_layout() is None
    np.testing.assert_array_equal(
        SVGD._resolve_true_theta(s, [1.0, 2.0]), [1.0, 2.0])
    with pytest.raises(ValueError, match="expected 2 values"):
        SVGD._resolve_true_theta(s, [1.0])
    with pytest.raises(ValueError, match="expected 2 values"):
        SVGD._resolve_true_theta(s, [1.0, 2.0, 3.0])
