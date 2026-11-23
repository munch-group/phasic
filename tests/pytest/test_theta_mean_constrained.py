"""SVGD.theta_mean / theta_std are reported on the parameter scale.

They used to be the mean/std of the optimiser's unconstrained particles, so
under positive_params=True a rate fixed at mu showed as log(expm1(mu))
(-9.028 for 1.2e-4) and free rates were off by softplus curvature. They now
equal get_results()['theta_mean'] / ['theta_std']; `particles` stay
unconstrained (get_results and the plots transform them from there).
"""
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phasic import GaussPrior

from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed

L = 3
EPOCHS = [0.0, 0.4]
MU = 1.2e-2                # small enough that softplus^-1(MU) is far from MU


def _rows(probs, n):
    counts = np.round(np.asarray(probs) * n).astype(int)
    return [[k, 0] for k in range(L + 1) for _ in range(counts[k])]


@pytest.fixture(scope="module")
def epoch_fit():
    g, idx = coalescent_indexed(2)
    jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
    p = normalised([p_n2_two_epoch(k, 2.0, MU, 6.0, MU, EPOCHS[1]) for k in range(L + 1)])
    prior = [GaussPrior(ci=[0.5, 20.0], lower=0.0), None] * 2
    return jc.svgd(_rows(p, 5_000), epoch_starts=EPOCHS, fixed=[(1, MU)],
                   prior=prior, n_particles=8, n_iterations=5, progress=False,
                   seed=1)


def test_fixed_slots_report_the_fixed_value(epoch_fit):
    tm = np.asarray(epoch_fit.theta_mean)
    assert tm.shape == (4,)
    np.testing.assert_allclose(tm[[1, 3]], MU, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(epoch_fit.theta_std)[[1, 3]], 0.0, atol=1e-12)
    # The old (unconstrained) value was log(expm1(MU)), about -4.43.
    assert np.all(np.asarray(epoch_fit.particles)[:, [1, 3]] < -4)


def test_attributes_equal_get_results(epoch_fit):
    r = epoch_fit.get_results()
    np.testing.assert_array_equal(np.asarray(epoch_fit.theta_mean), np.asarray(r['theta_mean']))
    np.testing.assert_array_equal(np.asarray(epoch_fit.theta_std), np.asarray(r['theta_std']))


def test_free_slots_are_means_of_transformed_particles(epoch_fit):
    P = jnp.asarray(epoch_fit.particles)
    want = jax.nn.softplus(P[:, [0, 2]])
    np.testing.assert_allclose(np.asarray(epoch_fit.theta_mean)[[0, 2]],
                               np.asarray(want.mean(axis=0)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(epoch_fit.theta_std)[[0, 2]],
                               np.asarray(want.std(axis=0)), rtol=1e-6)


def test_no_transform_reports_particle_moments():
    random.seed(4)
    np.random.seed(4)
    g = coalescent_graph(3)
    g.update_weights([2.0])
    s = g.svgd(g.sample(100), prior=GaussPrior(ci=[0.5, 5.0]), positive_params=False,
               n_particles=8, n_iterations=3, progress=False, seed=2)
    assert s.param_transform is None
    P = np.asarray(s.particles)
    np.testing.assert_allclose(np.asarray(s.theta_mean), P.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(s.theta_std), P.std(axis=0), rtol=1e-12)
