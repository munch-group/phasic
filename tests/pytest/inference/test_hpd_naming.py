"""HPD/HDR naming consistency: new names, deprecated aliases, fixes.

See hpd-hdr-naming-plan.md. "hpd" is an interval per parameter, "hdr" a
joint region, and "level" the enclosed mass. Every deprecated name must
warn and give output identical to its replacement.
"""
import warnings

import numpy as np
import pytest

from phasic.svgd import _compute_hpd, _compute_kde_hpd, _compute_kde_hdr


# ---------------------------------------------------------------------------
# Batch 1: helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_compute_hpd_alpha_alias(self):
        x = np.random.default_rng(1).normal(size=200)
        with pytest.warns(DeprecationWarning, match="level"):
            old = _compute_hpd(x, alpha=0.5)
        assert old == _compute_hpd(x, level=0.5)

    def test_compute_hpd_both_raises(self):
        with pytest.raises(TypeError, match="only 'level'"):
            _compute_hpd(np.arange(20.0), level=0.5, alpha=0.5)

    def test_compute_hpd_new_name_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _compute_hpd(np.arange(20.0), level=0.5)
            _compute_hpd(np.arange(20.0))

    def test_kde_hdr_alias(self):
        x = np.random.default_rng(2).normal(size=50)
        with pytest.warns(DeprecationWarning, match="_compute_kde_hpd"):
            old = _compute_kde_hdr(x, alpha=0.6)
        assert old == _compute_kde_hpd(x, level=0.6)

    def test_plot_module_has_no_own_copy(self):
        # the former phasic.plot copy returned (0, 19) for arange(20) at 0.95:
        # one sample too many; plot_posterior now uses the svgd helper
        import phasic.plot
        assert not hasattr(phasic.plot, "_compute_hpd")


# ---------------------------------------------------------------------------
# Fixtures for the SVGD / MCMC level tests
# ---------------------------------------------------------------------------

def _coalescent_graph():
    from phasic import Graph, with_ipv

    @with_ipv([4, 0, 0, 0])
    def coalescent(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions

    return Graph(coalescent)


def _fit(positive_params):
    import random
    random.seed(42)  # Graph.sample() seeds from stdlib random
    np.random.seed(42)
    graph = _coalescent_graph()
    graph.update_weights([7.0])
    observed = graph.sample(500)
    return graph.svgd(observed, learning_rate=0.002, n_iterations=20,
                      n_particles=12, progress=False,
                      positive_params=positive_params)


@pytest.fixture(scope="module")
def svgd_fit():
    return _fit(positive_params=True)


@pytest.fixture(scope="module")
def svgd_fit_no_transform():
    return _fit(positive_params=False)


def _mcmc(positive_params):
    import jax.numpy as jnp
    from phasic.mcmc import MCMC
    from phasic.svgd import GaussPrior

    def log_lik(theta):
        return -0.5 * jnp.sum((theta - 3.0) ** 2 / 0.5 ** 2)

    mcmc = MCMC(log_prob_fn=log_lik, prior=GaussPrior(mean=1.0, std=2.0),
                theta_dim=2, n_samples=300, n_chains=2, burn_in=50,
                proposal_scale=0.5, positive_params=positive_params,
                seed=42, verbose=False, progress=False)
    mcmc.run()
    return mcmc


# ---------------------------------------------------------------------------
# Batch 2: hpd_intervals() and the 'hpd_level' key
# ---------------------------------------------------------------------------

class TestHpdIntervalsSVGD:

    @pytest.mark.parametrize("fit", ["svgd_fit", "svgd_fit_no_transform"])
    def test_equals_get_results_at_095(self, fit, request):
        svgd = request.getfixturevalue(fit)
        r = svgd.get_results()
        lower, upper = svgd.hpd_intervals(0.95)
        assert np.array_equal(lower, r['hpd_lower'])
        assert np.array_equal(upper, r['hpd_upper'])
        assert r['hpd_level'] == 0.95
        assert lower.shape == upper.shape == (svgd.theta_dim,)

    def test_matches_manual_snippet(self, svgd_fit):
        P = np.asarray(svgd_fit.get_results()['particles'])
        lo, hi = np.array([_compute_hpd(P[:, i], level=0.5)
                           for i in range(P.shape[1])]).T
        lower, upper = svgd_fit.hpd_intervals(0.5)
        assert np.array_equal(lower, lo) and np.array_equal(upper, hi)

    def test_nested(self, svgd_fit):
        l50, u50 = svgd_fit.hpd_intervals(0.5)
        l95, u95 = svgd_fit.hpd_intervals(0.95)
        assert np.all(l95 <= l50) and np.all(u50 <= u95)

    def test_unconstrained(self, svgd_fit, svgd_fit_no_transform):
        P = np.asarray(svgd_fit.particles)
        lower, upper = svgd_fit.hpd_intervals(0.95, unconstrained=True)
        assert (lower[0], upper[0]) == _compute_hpd(P[:, 0], level=0.95)
        with pytest.raises(ValueError, match="unconstrained"):
            svgd_fit_no_transform.hpd_intervals(unconstrained=True)

    @pytest.mark.parametrize("level", [0.0, -0.1, 1.5])
    def test_bad_level(self, svgd_fit, level):
        with pytest.raises(ValueError, match="level"):
            svgd_fit.hpd_intervals(level)


class TestHpdIntervalsMCMC:

    @pytest.mark.parametrize("positive_params", [True, False])
    def test_equals_get_results_at_095(self, positive_params):
        mcmc = _mcmc(positive_params)
        r = mcmc.get_results()
        lower, upper = mcmc.hpd_intervals(0.95)
        assert np.array_equal(lower, r['hpd_lower'])
        assert np.array_equal(upper, r['hpd_upper'])
        assert r['hpd_level'] == 0.95
        l50, u50 = mcmc.hpd_intervals(0.5)
        assert np.all(lower <= l50) and np.all(u50 <= upper)
        if positive_params:
            mcmc.hpd_intervals(0.5, unconstrained=True)
        else:
            with pytest.raises(ValueError, match="unconstrained"):
                mcmc.hpd_intervals(unconstrained=True)

    def test_not_fitted(self):
        import jax.numpy as jnp
        from phasic.mcmc import MCMC
        mcmc = MCMC(log_prob_fn=lambda t: -jnp.sum(t ** 2), theta_dim=1,
                    positive_params=False, verbose=False, progress=False)
        with pytest.raises(RuntimeError, match="run"):
            mcmc.hpd_intervals()


# ---------------------------------------------------------------------------
# Batch 3: ci_level / ci_method names
# ---------------------------------------------------------------------------

def _no_deprecation(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return fn(*args, **kwargs)


def _ribbons(ax):
    return [np.asarray(c.get_paths()[0].vertices) for c in ax.collections]


def _patches(axes):
    return [np.asarray(p.get_verts()) for ax in np.ravel(axes) for p in ax.patches]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class TestCiNames:

    def test_plot_ci_ci_alias(self, svgd_fit):
        new = _no_deprecation(svgd_fit.plot_ci, return_fig=True, ci_level=0.5)
        with pytest.warns(DeprecationWarning, match="ci_level"):
            old = svgd_fit.plot_ci(return_fig=True, ci=0.5)
        a, b = _ribbons(new), _ribbons(old)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert new.get_title() == old.get_title()

    def test_plot_ci_both_raises(self, svgd_fit):
        with pytest.raises(TypeError, match="only 'ci_level'"):
            svgd_fit.plot_ci(return_fig=True, ci=0.5, ci_level=0.8)

    @pytest.mark.parametrize("old,new", [("kde", "kde_hpd"),
                                         ("percentile", "equal_tailed")])
    def test_plot_ci_method_alias(self, svgd_fit, old, new):
        ax_new = _no_deprecation(svgd_fit.plot_ci, return_fig=True, ci_method=new)
        with pytest.warns(DeprecationWarning, match=new):
            ax_old = svgd_fit.plot_ci(return_fig=True, ci_method=old)
        a, b = _ribbons(ax_new), _ribbons(ax_old)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert ax_new.get_title() == ax_old.get_title()

    def test_plot_ci_labels(self, svgd_fit):
        titles = {m: _no_deprecation(svgd_fit.plot_ci, return_fig=True,
                                     ci_method=m).get_title()
                  for m in ("kde_hpd", "hpd", "equal_tailed")}
        assert "KDE HPD" in titles["kde_hpd"]
        assert "HPD" in titles["hpd"] and "KDE" not in titles["hpd"]
        assert "CI" in titles["equal_tailed"]

    def test_plot_ci_rejects_unknown(self, svgd_fit):
        with pytest.raises(ValueError, match="kde_hpd"):
            svgd_fit.plot_ci(ci_method="bogus")

    def test_plot_posterior_percentile_alias(self, svgd_fit):
        fig_new, axes_new = _no_deprecation(
            svgd_fit.plot_posterior, return_fig=True, ci_method="equal_tailed")
        with pytest.warns(DeprecationWarning, match="equal_tailed"):
            fig_old, axes_old = svgd_fit.plot_posterior(
                return_fig=True, ci_method="percentile")
        a, b = _patches(axes_new), _patches(axes_old)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_summary_percentile_alias(self, svgd_fit, capsys):
        _no_deprecation(svgd_fit.summary, ci_method="equal_tailed", ci_level=0.5)
        new = capsys.readouterr().out
        with pytest.warns(DeprecationWarning, match="equal_tailed"):
            svgd_fit.summary(ci_method="percentile", ci_level=0.5)
        assert capsys.readouterr().out == new
        with pytest.raises(ValueError, match="equal_tailed"):
            svgd_fit.summary(ci_method="bogus")

    def test_shared_plot_posterior_alias(self):
        from phasic.plot import plot_posterior
        r = _mcmc(positive_params=False).get_results()
        fig_new, axes_new = _no_deprecation(
            plot_posterior, r, ci_method="equal_tailed", return_fig=True)
        with pytest.warns(DeprecationWarning, match="equal_tailed"):
            fig_old, axes_old = plot_posterior(r, ci_method="percentile",
                                               return_fig=True)
        a, b = _patches(axes_new), _patches(axes_old)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        with pytest.raises(ValueError, match="equal_tailed"):
            plot_posterior(r, ci_method="bogus", return_fig=True)

    def test_shared_plot_posterior_band_is_hpd(self):
        # F1: the shaded band equals MCMC's own hpd_intervals
        from phasic.plot import plot_posterior
        mcmc = _mcmc(positive_params=False)
        fig, axes = _no_deprecation(plot_posterior, mcmc.get_results(),
                                    ci_level=0.5, return_fig=True)
        lower, upper = mcmc.hpd_intervals(0.5)
        for i, ax in enumerate(np.ravel(axes)[:mcmc.theta_dim]):
            band = ax.patches[-1]  # axvspan is added after the histogram bars
            xs = band.get_path().transformed(band.get_patch_transform()).vertices[:, 0]
            assert np.allclose([xs.min(), xs.max()], [lower[i], upper[i]])


# ---------------------------------------------------------------------------
# Batch 4: plot_pairwise_hdr() and joint_hdr_particles()
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def svgd_fit_2d():
    import random
    from phasic import Graph, with_ipv

    @with_ipv([4, 0, 0, 0])
    def coalescent2(state):
        # theta[0]: merging two singletons, theta[1]: any other merge
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                rate = state[i] * (state[j] - same) / (1 + same)
                coef = [rate, 0.0] if (i == 0 and j == 0) else [0.0, rate]
                transitions.append([new, coef])
        return transitions

    random.seed(7)
    np.random.seed(7)
    graph = Graph(coalescent2)
    graph.update_weights([7.0, 3.0])
    observed = graph.sample(300)
    return graph.svgd(observed, learning_rate=0.002, n_iterations=15,
                      n_particles=16, progress=False)


def _contour_levels(ax):
    from matplotlib.contour import ContourSet
    return [list(c.levels) for c in ax.collections if isinstance(c, ContourSet)]


class TestPairwiseHdr:

    def test_plot_hdr_alias(self, svgd_fit_2d):
        new = _no_deprecation(
            svgd_fit_2d.plot_pairwise_hdr, levels=[0.9, 0.5], hexgrid=False,
            show_marginal_hpd=True, marginal_hpd_level=0.8, return_fig=True)
        with pytest.warns(DeprecationWarning, match="plot_pairwise_hdr"):
            old = svgd_fit_2d.plot_hdr(
                alphas=[0.9, 0.5], hexgrid=False, show_hpd=True,
                hpd_alpha=0.8, return_fig=True)
        assert _contour_levels(new) == _contour_levels(old) != []
        a, b = _patches(new), _patches(old)
        assert len(a) == len(b) == 2  # the two marginal HPD bands
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert new.get_xlim() == old.get_xlim()
        assert new.get_ylim() == old.get_ylim()

    def test_marginal_bands_are_hpd_intervals(self, svgd_fit_2d):
        ax = _no_deprecation(
            svgd_fit_2d.plot_pairwise_hdr, hexgrid=False,
            show_marginal_hpd=True, marginal_hpd_level=0.8, return_fig=True)
        lower, upper = svgd_fit_2d.hpd_intervals(0.8)
        vband, hband = ax.patches
        xs = vband.get_path().transformed(vband.get_patch_transform()).vertices[:, 0]
        ys = hband.get_path().transformed(hband.get_patch_transform()).vertices[:, 1]
        assert np.allclose([xs.min(), xs.max()], [lower[0], upper[0]])
        assert np.allclose([ys.min(), ys.max()], [lower[1], upper[1]])


class TestJointHdrParticles:

    def test_estimate_hdr_alias_with_transform(self, svgd_fit):
        new, t_new = _no_deprecation(svgd_fit.joint_hdr_particles,
                                     0.5, unconstrained=True)
        with pytest.warns(DeprecationWarning, match="joint_hdr_particles"):
            old, t_old = svgd_fit.estimate_hdr(0.5)
        assert np.array_equal(old, new) and float(t_old) == float(t_new)

    def test_constrained_default(self, svgd_fit):
        phi, t1 = svgd_fit.joint_hdr_particles(0.5, unconstrained=True)
        theta, t2 = svgd_fit.joint_hdr_particles(0.5)
        expected = np.array([np.asarray(svgd_fit.param_transform(p)) for p in phi])
        assert np.allclose(theta, expected, rtol=1e-12)
        assert float(t1) == float(t2)
        assert len(theta) == int(len(svgd_fit.particles) * 0.5)

    def test_no_transform(self, svgd_fit_no_transform):
        new, _ = _no_deprecation(svgd_fit_no_transform.joint_hdr_particles, 0.5)
        with pytest.warns(DeprecationWarning):
            old, _ = svgd_fit_no_transform.estimate_hdr(0.5)
        assert np.array_equal(old, new)
        with pytest.raises(ValueError, match="unconstrained"):
            svgd_fit_no_transform.joint_hdr_particles(unconstrained=True)

    def test_level(self, svgd_fit):
        with pytest.raises(ValueError, match="level"):
            svgd_fit.joint_hdr_particles(0.0)
        kept, _ = svgd_fit.joint_hdr_particles(0.01)  # rounds to 0 particles
        assert len(kept) == 1

    def test_estimate_hdr_keeps_old_edge_behaviour(self, svgd_fit):
        # before the rename, alpha < 1/n kept every particle; the alias must too
        with pytest.warns(DeprecationWarning):
            kept, _ = svgd_fit.estimate_hdr(0.01)
        assert np.array_equal(kept, svgd_fit.particles)


# ---------------------------------------------------------------------------
# Batch 5: MCMC summary header
# ---------------------------------------------------------------------------

def test_mcmc_summary_hpd_labels(capsys):
    mcmc = _mcmc(positive_params=False)
    mcmc.summary()
    out = capsys.readouterr().out
    assert "HPD 95% lo" in out and "HPD 95% hi" in out
    assert "2.5%" not in out and "97.5%" not in out
