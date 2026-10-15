"""Tests for SVGD.plot_ci: the KDE highest-density ribbon and the MAP line."""
import random

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from phasic.svgd import _compute_hpd, _compute_kde_hdr


# ---------------------------------------------------------------------------
# Unit tests for _compute_kde_hdr
# ---------------------------------------------------------------------------

class TestComputeKdeHdr:

    def test_normal_matches_closed_form(self):
        """Large normal sample: HDR ~ +-z * sd, widened by the KDE smoothing."""
        rng = np.random.default_rng(0)
        n = 20_000
        x = rng.normal(0.0, 1.0, n)
        expected = 1.959964 * np.sqrt(1 + n ** (-2 / 5))  # KDE var = sd^2 + h^2
        lo, hi = _compute_kde_hdr(x, alpha=0.95)
        assert lo == pytest.approx(-expected, abs=0.05)
        assert hi == pytest.approx(expected, abs=0.05)
        lo, hi = _compute_kde_hdr(x, alpha=0.6)
        assert lo == pytest.approx(-0.8416, abs=0.05)
        assert hi == pytest.approx(0.8416, abs=0.05)

    def test_continuous_where_sample_hpd_flips(self):
        """An outlier overtaking the opposite one flips the sample HPD between
        dropping the low and the high particle; the KDE region must not jump."""
        base = np.linspace(-1.0, 1.0, 18)
        s_values = np.linspace(0.0, 0.4, 401)
        hpd, kde = [], []
        for s in s_values:
            x = np.r_[base, -1.3 - s, 1.3 + (0.4 - s)]
            hpd.append(_compute_hpd(x, alpha=0.95))
            kde.append(_compute_kde_hdr(x, alpha=0.95))
        hpd_steps = np.abs(np.diff(np.array(hpd), axis=0)).max(axis=1)
        kde_steps = np.abs(np.diff(np.array(kde), axis=0)).max(axis=1)
        # the fixture really does make the sample HPD jump ...
        assert hpd_steps.max() > 50 * np.median(hpd_steps)
        # ... while the KDE region moves no faster than the particles do
        particle_step = s_values[1] - s_values[0]
        assert kde_steps.max() <= 2 * particle_step

    def test_clipped_to_sample_range(self):
        """KDE tails must not carry the region past the samples (e.g. below
        zero for a positive parameter)."""
        rng = np.random.default_rng(1)
        x = rng.exponential(1.0, 5000)
        lo, hi = _compute_kde_hdr(x, alpha=0.95)
        assert lo >= x.min()
        assert hi <= x.max()
        assert hi == pytest.approx(-np.log(0.05), abs=0.15)

    def test_degenerate_samples(self):
        assert _compute_kde_hdr(np.full(20, 7.0)) == (7.0, 7.0)
        assert _compute_kde_hdr(np.array([3.0])) == (3.0, 3.0)
        collapsed = np.r_[np.full(17, 7.06), 6.9, 7.2, 7.46]
        lo, hi = _compute_kde_hdr(collapsed)
        assert 6.9 <= lo <= 7.06 <= hi <= 7.46


# ---------------------------------------------------------------------------
# plot_ci integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_svgd():
    from phasic import Graph, with_ipv

    random.seed(42)
    np.random.seed(42)

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

    graph = Graph(coalescent)
    graph.update_weights([7.0])
    observed = graph.sample(500)
    svgd = graph.svgd(observed, learning_rate=0.002, n_iterations=20,
                      n_particles=10, progress=False)
    svgd._test_graph, svgd._test_observed = graph, observed  # for further fits
    return svgd


class TestPlotCi:

    def test_map_line_ends_at_map_estimate(self, fitted_svgd):
        ax = fitted_svgd.plot_ci(return_fig=True)
        line = ax.lines[0].get_ydata()
        assert len(line) == len(fitted_svgd.history)
        assert line[-1] == pytest.approx(
            fitted_svgd.map_estimate_from_particles()[0][0], rel=1e-12)
        assert ax.get_ylabel().startswith("MAP")

    def test_mean_line(self, fitted_svgd):
        ax = fitted_svgd.plot_ci(return_fig=True, estimate="mean")
        assert ax.lines[0].get_ydata()[-1] == pytest.approx(
            float(fitted_svgd.get_results()["theta_mean"][0]), rel=1e-12)

    @pytest.mark.parametrize("ci_method,label", [
        ("kde", "HDR"), ("hpd", "HPD"), ("percentile", "CI")])
    def test_ci_methods(self, fitted_svgd, ci_method, label):
        ax = fitted_svgd.plot_ci(return_fig=True, ci_method=ci_method)
        assert label in ax.get_ylabel()

    def test_rejects_unknown_options(self, fitted_svgd):
        with pytest.raises(ValueError, match="estimate"):
            fitted_svgd.plot_ci(estimate="median")
        with pytest.raises(ValueError, match="ci_method"):
            fitted_svgd.plot_ci(ci_method="bogus")


class TestPlotTrace:

    def _estimate_line(self, axes):
        return [l for l in axes[0].lines if l.get_linestyle() == "--"][0]

    def test_map_line_ends_at_map_estimate(self, fitted_svgd):
        fig, axes = fitted_svgd.plot_trace(return_fig=True)
        line = self._estimate_line(axes)
        assert len(line.get_ydata()) == len(fitted_svgd.history)
        assert line.get_ydata()[-1] == pytest.approx(
            fitted_svgd.map_estimate_from_particles()[0][0], rel=1e-12)
        assert line.get_label().startswith("MAP = ")

    def test_mean_line(self, fitted_svgd):
        fig, axes = fitted_svgd.plot_trace(return_fig=True, estimate="mean")
        line = self._estimate_line(axes)
        assert line.get_ydata()[-1] == pytest.approx(
            float(fitted_svgd.get_results()["theta_mean"][0]), rel=1e-12)
        assert line.get_label().startswith("Mean = ")

    def test_rejects_unknown_estimate(self, fitted_svgd):
        with pytest.raises(ValueError, match="estimate"):
            fitted_svgd.plot_trace(estimate="median")


class TestHistorySnapshots:

    def test_snapshot_k_is_after_k_steps(self, fitted_svgd):
        """Snapshot 0 is the initial particles, snapshot k the particles after
        k steps, labelled k; the final particles appear once, as the last one."""
        n_steps = 20  # fitted_svgd's n_iterations
        history = np.asarray(fitted_svgd.history)
        assert list(fitted_svgd.history_iterations) == list(range(n_steps + 1))
        assert len(history) == n_steps + 1
        np.testing.assert_array_equal(history[-1], np.asarray(fitted_svgd.particles))
        assert not np.array_equal(history[-1], history[-2])


# ---------------------------------------------------------------------------
# Log posteriors recorded during the fit (read by plot_ci's MAP line)
# ---------------------------------------------------------------------------

def _direct_log_probs(svgd):
    """Log posterior of every history particle, evaluated directly."""
    from functools import partial
    import jax.numpy as jnp
    log_prob = partial(svgd._log_prob_unified, rewards=svgd.rewards)
    return np.array([[float(log_prob(p)) for p in jnp.asarray(h)] for h in svgd.history])


class TestRecordedLogProb:

    def test_recorded_values_match_direct_evaluation(self, fitted_svgd):
        recorded = np.asarray(fitted_svgd.history_log_prob)
        assert recorded.shape == (len(fitted_svgd.history), fitted_svgd.particles.shape[0])
        np.testing.assert_allclose(recorded, _direct_log_probs(fitted_svgd), rtol=1e-12)

    def test_recording_does_not_change_the_fit(self, fitted_svgd):
        """Recording swaps grad for value_and_grad; the particles must be
        bit-identical to a fit that records nothing (no history kept)."""
        graph, observed = fitted_svgd._test_graph, fitted_svgd._test_observed
        kw = dict(learning_rate=0.002, n_iterations=20, n_particles=10,
                  progress=False, seed=11)
        with_record = graph.svgd(observed, return_history=True, **kw)
        without = graph.svgd(observed, return_history=False, **kw)
        assert with_record.history_log_prob is not None
        assert without.history_log_prob is None
        np.testing.assert_array_equal(np.asarray(with_record.particles),
                                      np.asarray(without.particles))

    def test_regularized_fit_computed_by_plot_ci(self, fitted_svgd):
        """With moment regularization the fit's objective is not the log
        posterior, so nothing is recorded; plot_ci computes it, keeps it,
        and its MAP line still ends at map_estimate_from_particles."""
        graph, observed = fitted_svgd._test_graph, fitted_svgd._test_observed
        svgd = graph.svgd(observed, learning_rate=0.002, n_iterations=20,
                          n_particles=10, progress=False, seed=11,
                          regularization=0.5, nr_moments=2)
        assert svgd.history_log_prob is None
        ax = svgd.plot_ci(return_fig=True)
        np.testing.assert_allclose(np.asarray(svgd.history_log_prob),
                                   _direct_log_probs(svgd), rtol=1e-12)
        assert ax.lines[0].get_ydata()[-1] == pytest.approx(
            svgd.map_estimate_from_particles()[0][0], rel=1e-12)

    def test_inconsistent_arguments_rejected(self):
        import jax.numpy as jnp
        from phasic.svgd import svgd_step, run_svgd, SVGDKernel
        particles = jnp.ones((3, 1))
        log_prob = lambda theta: -jnp.sum(theta ** 2)
        kernel = SVGDKernel()
        with pytest.raises(ValueError, match="compiled_value_and_grad"):
            svgd_step(particles, log_prob, kernel, 0.1,
                      compiled_grad=lambda t: t, return_log_prob=True)
        with pytest.raises(ValueError, match="requires return_log_prob"):
            svgd_step(particles, log_prob, kernel, 0.1,
                      compiled_value_and_grad=lambda t: (0.0, t))
        with pytest.raises(ValueError, match="return_history"):
            run_svgd(log_prob, particles, 2, learning_rate=0.1, kernel=kernel,
                     return_history=False, record_log_prob=True, verbose=False)
        with pytest.raises(ValueError, match="requires record_log_prob"):
            run_svgd(log_prob, particles, 2, learning_rate=0.1, kernel=kernel,
                     compiled_value_and_grad=lambda t: (0.0, t), verbose=False)
