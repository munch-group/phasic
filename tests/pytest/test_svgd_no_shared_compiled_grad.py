"""A fit must not depend on earlier fits that used the same model object.

SVGD used to keep compiled gradient functions in a class-level dict keyed on
``id(model)`` plus shapes. The compiled function closes over the whole SVGD
instance (data, prior list, transform), so a later fit with the same model
object and shapes silently reused the first fit's gradient: different data
returned the first fit's answer, a different prior was ignored, and
``FreeEpochModel.fit`` calls with different priors / non-nested ``fixed``
sets were wrong. The cache (and its on-disk half) is gone; these tests pin
that. Plan: bug1-compiled-cache-plan.md.

Reference for every comparison: the same fit on a FRESHLY BUILT model
object. Fits are bit-reproducible in one process, so equality is exact.
"""
import gc
import pickle
import random
import weakref

import jax.numpy as jnp
import numpy as np
import pytest

from phasic import Graph, GaussPrior, SVGD

from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed

KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False,
          seed=1)


def _data(theta, n=150, seed=3):
    random.seed(seed)
    np.random.seed(seed)
    g = coalescent_graph(4)
    g.update_weights([theta])
    return np.asarray(g.sample(n))


DATA_A, DATA_B = _data(8.0), _data(1.0, seed=4)


def _model():
    return Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2,
                                            theta_dim=1)


def _wide():
    return GaussPrior(ci=[0.5, 20], lower=0.0)


def _narrow():
    return GaussPrior(ci=[0.01, 0.2], lower=0.0)


def _fit(model, data, prior, **kw):
    opt = {k: kw.pop(k) for k in ('return_history',) if k in kw}
    s = SVGD(model=model, observed_data=data, theta_dim=1, prior=prior,
             **{**KW, **kw})
    s.optimize(**opt)
    return s


def _particles(s):
    p = np.asarray(s.particles)
    assert np.all(np.isfinite(p))
    return p


def _assert_second_fit_independent(first, second):
    """first/second: callables model -> fitted SVGD."""
    shared = _model()
    p_first = _particles(first(shared))
    p_after = _particles(second(shared))
    p_fresh = _particles(second(_model()))
    # power: the two fits really are different problems
    assert np.max(np.abs(p_first - p_fresh)) > 1e-3
    np.testing.assert_array_equal(p_after, p_fresh)


# ------------------------------------------------------------ direct SVGD
@pytest.mark.parametrize("kw", [
    dict(),                                    # value_and_grad (history kept)
    dict(return_history=False),                # plain grad
    dict(regularization=1.0, nr_moments=2),    # sample moments differ with data
], ids=["default", "no_history", "regularized"])
def test_different_data_same_shape(kw):
    assert DATA_A.shape == DATA_B.shape
    _assert_second_fit_independent(
        lambda m: _fit(m, DATA_A, _wide(), **kw),
        lambda m: _fit(m, DATA_B, _wide(), **kw))


def test_different_prior():
    _assert_second_fit_independent(
        lambda m: _fit(m, DATA_A, _wide()),
        lambda m: _fit(m, DATA_A, _narrow()))


def test_mutated_data_then_optimize_again():
    # Without regularization: sample_moments is computed in __init__ and is
    # not refreshed by mutation, which is a separate matter.
    s = _fit(_model(), DATA_A, _wide())
    p_first = _particles(s).copy()
    s.observed_data = jnp.asarray(DATA_B, dtype=s.observed_data.dtype)
    s.optimize()
    p_fresh = _particles(_fit(_model(), DATA_B, _wide()))
    assert np.max(np.abs(p_first - p_fresh)) > 1e-3
    np.testing.assert_array_equal(_particles(s), p_fresh)


def test_same_instance_optimize_twice_is_reproducible():
    s = _fit(_model(), DATA_A, _wide())
    p1 = _particles(s).copy()
    s.optimize()
    np.testing.assert_array_equal(_particles(s), p1)


def test_graph_svgd_calls_are_independent():
    g = coalescent_graph(4)
    kw = dict(prior=_wide(), **KW)
    a = _particles(g.svgd(DATA_A, **kw))
    b = _particles(g.svgd(DATA_B, **{**kw, 'prior': _wide()}))
    b_fresh = _particles(coalescent_graph(4).svgd(DATA_B, **{**kw, 'prior': _wide()}))
    assert np.max(np.abs(a - b_fresh)) > 1e-3
    np.testing.assert_array_equal(b, b_fresh)


# ----------------------------------------------------------- FreeEpochModel
L = 3
EPOCHS = [0.0, 0.4]
EKW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False,
           seed=1)


@pytest.fixture(scope="module")
def joint():
    g, idx = coalescent_indexed(2)
    jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
    p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EPOCHS[1])
                    for k in range(L + 1)])
    rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 400)))]
    return jc, rows


def _em(joint):
    jc, rows = joint
    return jc.epoch_model(rows, epoch_starts=EPOCHS)


def _prior4(make=_wide):
    return [make(), None, make(), None]


FULL = [(1, 1.0), (3, 1.0)]                    # mu fixed in both epochs
NESTED = [(1, 1.0), (2, 0.5), (3, 1.0)]        # epoch-2 coalescence also fixed


def _epoch_independent(joint, first, second, compare_log_prob=False,
                       need_power=True):
    shared = _em(joint)
    s_first = first(shared)
    s_after = second(shared)
    s_fresh = second(_em(joint))
    p_after, p_fresh = _particles(s_after), _particles(s_fresh)
    if need_power:
        assert np.max(np.abs(_particles(s_first) - p_fresh)) > 1e-6
    np.testing.assert_array_equal(p_after, p_fresh)
    if compare_log_prob:
        np.testing.assert_array_equal(np.asarray(s_after.history_log_prob),
                                      np.asarray(s_fresh.history_log_prob))


def test_epoch_model_prior_override(joint):
    _epoch_independent(
        joint,
        lambda em: em.fit(fixed=FULL, prior=_prior4(_wide), **EKW),
        lambda em: em.fit(fixed=FULL, prior=_prior4(_narrow), **EKW))


def test_epoch_model_non_nested_fixed(joint):
    # Fixed slot 1 then fixed slot 3: the first fit's compiled function had no
    # prior on slot 1, which is free in the second fit.
    pri = lambda: [_wide(), _wide(), _wide(), _wide()]
    _epoch_independent(
        joint,
        lambda em: em.fit(fixed=[(1, 1.0)], prior=pri(), **EKW),
        lambda em: em.fit(fixed=[(3, 1.0)], prior=pri(), **EKW))


def test_epoch_model_nested_then_full(joint):
    _epoch_independent(
        joint,
        lambda em: em.fit(fixed=NESTED, prior=_prior4(), **EKW),
        lambda em: em.fit(fixed=FULL, prior=_prior4(), **EKW))


def test_epoch_model_default_prior_with_different_fixed_values(joint):
    # prior=None: the default prior is built from the fixed values.
    _epoch_independent(
        joint,
        lambda em: em.fit(fixed=[(1, 1.0), (3, 1.0)], **EKW),
        lambda em: em.fit(fixed=[(1, 3.0), (3, 3.0)], **EKW))


def test_epoch_model_full_then_nested_including_log_posterior(joint):
    # The likelihood-ratio order. Particles were already right; the recorded
    # log posterior carried the full fit's prior term for the now-fixed slot.
    _epoch_independent(
        joint,
        lambda em: em.fit(fixed=FULL, prior=_prior4(), **EKW),
        lambda em: em.fit(fixed=NESTED, prior=_prior4(), **EKW),
        compare_log_prob=True)


# ------------------------------------------------------ no retention / disk
@pytest.mark.parametrize("kw", [dict(), dict(return_history=False)],
                         ids=["default", "no_history"])
def test_fitted_instance_is_released(kw):
    s = _fit(_model(), DATA_A, _wide(), **kw)
    ref = weakref.ref(s)
    del s
    gc.collect()
    assert ref() is None


def test_fitted_epoch_instance_is_released(joint):
    s = _em(joint).fit(fixed=FULL, prior=_prior4(), **EKW)
    ref = weakref.ref(s)
    del s
    gc.collect()
    assert ref() is None


def test_no_compiled_gradient_files_written_or_read(joint, tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    cache = tmp_path / '.phasic_cache'
    cache.mkdir()
    # A non-empty file of the kind the old code would have loaded.
    planted = cache / 'compiled_svgd_0123456789abcdef.pkl'
    with open(planted, 'wb') as f:
        pickle.dump({'grad': 'not a function'}, f)
    before = planted.read_bytes()

    loaded = []
    real_load = pickle.load

    def recording_load(fh, *a, **k):
        loaded.append(str(getattr(fh, 'name', '')))
        return real_load(fh, *a, **k)

    monkeypatch.setattr(pickle, 'load', recording_load)

    _fit(_model(), DATA_A, _wide())
    _fit(_model(), DATA_A, _wide(), return_history=False)
    _em(joint).fit(fixed=FULL, prior=_prior4(), **EKW)
    coalescent_graph(4).svgd(DATA_A, prior=_wide(), **KW)

    assert not [p for p in loaded if 'compiled_svgd' in p]
    assert sorted(p.name for p in tmp_path.rglob('compiled_svgd_*')) == [planted.name]
    assert planted.read_bytes() == before


def test_class_level_cache_is_gone():
    assert not hasattr(SVGD, '_compiled_cache')
