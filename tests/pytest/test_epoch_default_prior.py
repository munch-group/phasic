"""Default (prior=None) prior of epoch fits, and the "no rate fixed" error.

Plan: epoch-default-prior-and-truncated-gauss-plan.md, Batch 2.

- The default epoch prior comes from a time-homogeneous reference fit
  (probability_matching) that now uses the fit's fixed rates and exposure
  (uniform or varying), with sigma = max(sd * se, |mean|) and
  lower=0.0 under positive_params. It is built where it is used (svgd,
  FreeEpochModel.fit), never in the model builder.
- On a joint graph with no rate fixed, multiplying every rate by the same
  factor leaves every joint probability unchanged, so every default prior
  raises (probability_matching, DataPrior, svgd, mcmc, the epoch default).
- An epoch fit with nothing pinning the last epoch's scale gets a notice.

Data come from the independent n=2 closed forms in test_epoch_normalization
(no phasic in the loop), mostly as deterministic expected counts so the
estimates are exact up to rounding. Comparing the default prior with the
true values is allowed here (user decision D6, 2026-11-22): this tests how
the default prior is built, not a recovery from a user prior.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

from phasic import DataPrior, GaussPrior
from phasic.exceptions import SvgdAssumptionWarning
from phasic.probability_matching import probability_matching

from test_epoch_normalization import normalised, p_n2, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_indexed

L = 3                      # reward_limit: outcomes k = 0..3 singletons
EPOCHS = [0.0, 0.4]
SD = 5.0
FIT = dict(n_particles=4, n_iterations=2, progress=False)


@pytest.fixture(scope="module")
def jc():
    g, idx = coalescent_indexed(2)
    return g.joint_prob_graph(idx, discrete=False, reward_limit=L)


def _rows(probs, n):
    """Deterministic expected counts -> observation rows [k, 0]."""
    counts = np.round(np.asarray(probs) * n).astype(int)
    return [[k, 0] for k in range(L + 1) for _ in range(counts[k])]


def _ls_estimate_c(rows, mu, alpha=1.0):
    """Independent time-homogeneous least-squares estimate of c (mu fixed),
    the same estimator probability_matching uses, from the closed form."""
    k = np.array([r[0] for r in rows])
    emp = np.bincount(k, minlength=L + 1) / k.size
    res = least_squares(
        lambda c: normalised([p_n2(j, c[0], mu, alpha) for j in range(L + 1)]) - emp,
        x0=[1.0], bounds=(1e-10, np.inf), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(res.x[0])


# ---------------------------------------------- differing per-epoch truths
C1, C2, MU = 2.0, 6.0, 1.0


@pytest.fixture(scope="module")
def two_epoch_rows():
    p = normalised([p_n2_two_epoch(k, C1, MU, C2, MU, EPOCHS[1]) for k in range(L + 1)])
    return _rows(p, 20_000)


@pytest.fixture(scope="module")
def default_fit(jc, two_epoch_rows):
    return jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, fixed=[(1, MU)], **FIT)


def test_default_prior_centred_on_reference_estimate(default_fit, two_epoch_rows):
    priors = default_fit.prior_list
    assert priors[1] is None and priors[3] is None          # mu fixed
    c_ref = _ls_estimate_c(two_epoch_rows, MU)
    for p in (priors[0], priors[2]):
        assert isinstance(p, GaussPrior) and p.lower == 0.0
        assert p.mu == pytest.approx(c_ref, rel=1e-4)
        assert np.isfinite(p.sigma) and p.sigma > 0


def test_default_prior_covers_each_epochs_truth(default_fit):
    # D6: allowed for a test of how the default prior is built.
    for p, c_true in ((default_fit.prior_list[0], C1), (default_fit.prior_list[2], C2)):
        assert abs(c_true - p.mu) < 1.96 * p.sigma, (p.mu, p.sigma, c_true)


def test_default_prior_width_rule(jc, default_fit, two_epoch_rows):
    # sigma = max(DataPrior width sd*se, |mean|): the same width
    # convention as the non-epoch default, plus the relative floor
    # (user decision 2026-11-23: per-epoch rates up to ~3x the
    # time-homogeneous estimate inside the 95% interval).
    dp = DataPrior(jc, two_epoch_rows, sd=SD, fixed=[(1, MU)])
    p = default_fit.prior_list[0]
    assert p.sigma == pytest.approx(max(dp[0].sigma, abs(dp[0].mu)), rel=1e-9)
    assert dp[0].mu == pytest.approx(p.mu, rel=1e-12)


def test_default_epoch_fit_runs(default_fit):
    # The hang regression: the default prior used to ignore fixed and be
    # ~±3e6 wide. Asserted through the fit's prior and particles, not a
    # timeout.
    assert np.all(np.isfinite(np.asarray(default_fit.particles)))
    assert all(p is None or p.sigma < 10 * p.mu for p in default_fit.prior_list)


def test_per_epoch_fixed_lists(jc, two_epoch_rows, default_fit):
    same = jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, fixed=[(1, [MU, MU])], **FIT)
    assert same.prior_list[0].mu == default_fit.prior_list[0].mu
    with pytest.raises(ValueError, match="fixed at different values"):
        jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, fixed=[(1, [MU, 2 * MU])], **FIT)


def test_default_prior_follows_positive_params(jc, two_epoch_rows):
    # Checked on the fit-time builder rather than through a fit: with
    # positive_params=False an untruncated prior can start rate particles
    # below zero, which the solver rightly refuses.
    em = jc.epoch_model(two_epoch_rows, EPOCHS)
    flat = [(1, MU), (3, MU)]
    assert em._default_prior_factory(flat, False)[0][0].lower is None
    assert em._default_prior_factory(flat, True)[0][0].lower == 0.0


# ------------------------------------------------------------- exposure
def test_uniform_exposure_notebook_case(jc):
    # Notebook cell: mu fixed at 1.2e-4 per site, exposure 10000 per window.
    c, mu, alpha = 5.0, 1.2e-4, 10_000.0
    rows = _rows(normalised([p_n2(k, c, mu, alpha) for k in range(L + 1)]), 20_000)
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, mu)], exposure=alpha,
                exposure_param_index=1, **FIT)
    assert s.prior_list[0].mu == pytest.approx(_ls_estimate_c(rows, mu, alpha), rel=1e-4)
    assert s.prior_list[0].mu == pytest.approx(c, rel=1e-2)


def test_varying_exposure_few_values(jc):
    c, mu = 3.0, 0.5
    rows, alpha = [], []
    for a in (0.2, 1.0, 5.0):
        r = _rows(normalised([p_n2(k, c, mu, a) for k in range(L + 1)]), 10_000)
        rows += r
        alpha += [a] * len(r)
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, mu)], exposure=np.array(alpha),
                exposure_param_index=1, **FIT)
    assert s.prior_list[0].mu == pytest.approx(c, rel=1e-2)
    # Ignoring the exposure would put the estimate elsewhere.
    assert abs(_ls_estimate_c(rows, mu) / c - 1) > 0.05


def test_free_exposure_parameter_with_another_rate_fixed(jc):
    # c fixed, mu (the exposure parameter) free: estimated in unscaled units.
    c, mu = 3.0, 0.5
    rows, alpha = [], []
    for a in (0.5, 2.0):
        r = _rows(normalised([p_n2(k, c, mu, a) for k in range(L + 1)]), 10_000)
        rows += r
        alpha += [a] * len(r)
    s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(0, c)], exposure=np.array(alpha),
                exposure_param_index=1, **FIT)
    assert s.prior_list[0] is None
    assert s.prior_list[1].mu == pytest.approx(mu, rel=1e-2)


def test_many_distinct_exposures_are_binned_with_notice(jc):
    c, mu, n = 3.0, 0.5, 6_000
    rng = np.random.default_rng(4)
    alpha = rng.uniform(0.5, 2.0, n)               # all distinct (> 100)
    rows = [[int(rng.choice(L + 1, p=normalised([p_n2(k, c, mu, a) for k in range(L + 1)]))), 0]
            for a in alpha]
    with pytest.warns(SvgdAssumptionWarning, match="quantile bins"):
        s = jc.svgd(rows, epoch_starts=EPOCHS, fixed=[(1, mu)], exposure=alpha,
                    exposure_param_index=1, **FIT)
    assert s.prior_list[0].mu == pytest.approx(c, rel=0.15)
    idx = jc._map_joint_observations_to_indices(rows)
    r = probability_matching(jc, idx, fixed=[(1, mu)], exposure=alpha,
                             exposure_param_index=1, verbose=False)
    assert r.exposure_binned and r.exposure_groups == 100


def test_probability_matching_without_exposure_unchanged(jc, two_epoch_rows):
    # The no-exposure path is the pre-change code path (one group, unscaled).
    idx = jc._map_joint_observations_to_indices(two_epoch_rows)
    r = probability_matching(jc, idx, fixed=[(1, MU)], verbose=False)
    assert r.exposure_groups == 1 and not r.exposure_binned
    r1 = probability_matching(jc, idx, fixed=[(1, MU)], verbose=False,
                              exposure=1.0, exposure_param_index=1)
    # Same estimator; the one-group path reads the table through a batched
    # call, so agreement is to optimiser precision, not bitwise.
    assert r1.theta[0] == pytest.approx(r.theta[0], rel=1e-7)
    assert r1.std[0] == pytest.approx(r.std[0], rel=1e-6)


# ------------------------------------------------- no rate fixed: raise
SCALE_MSG = "cannot determine the overall scale"


def test_no_rate_fixed_raises_everywhere(jc, two_epoch_rows):
    idx = jc._map_joint_observations_to_indices(two_epoch_rows)
    with pytest.raises(ValueError, match=SCALE_MSG):
        probability_matching(jc, idx, verbose=False)
    with pytest.raises(ValueError, match=SCALE_MSG):
        DataPrior(jc, two_epoch_rows)
    with pytest.raises(ValueError, match=SCALE_MSG):
        jc.svgd(two_epoch_rows, **FIT)
    with pytest.raises(ValueError, match=SCALE_MSG):
        jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, **FIT)
    # (Graph.mcmc does not take joint observations as rows, so its default
    # prior cannot be reached on a joint graph; it uses DataPrior, above.)


def test_discrete_joint_graph_raises_the_same(two_epoch_rows):
    g, idx = coalescent_indexed(2)
    jd = g.joint_prob_graph(idx, discrete=True, reward_limit=L)
    with pytest.raises(ValueError, match=SCALE_MSG):
        probability_matching(jd, jd._map_joint_observations_to_indices(two_epoch_rows),
                             verbose=False)


def test_former_silent_std_fallbacks_raise(jc, two_epoch_rows, monkeypatch):
    idx = jc._map_joint_observations_to_indices(two_epoch_rows)

    def fail(*a, **k):
        raise np.linalg.LinAlgError("forced")
    monkeypatch.setattr(np.linalg, "pinv", fail)
    with pytest.raises(ValueError, match="delta-method standard errors could not"):
        probability_matching(jc, idx, fixed=[(1, MU)], verbose=False)
    monkeypatch.setattr(np.linalg, "pinv", lambda J: np.zeros((J.shape[1], J.shape[0])))
    with pytest.raises(ValueError, match="zero or not finite"):
        probability_matching(jc, idx, fixed=[(1, MU)], verbose=False)


# ------------------------------------------------- FreeEpochModel / notices
def test_epoch_model_builds_without_fixed_and_fits_with_fixed(jc, two_epoch_rows, default_fit):
    em = jc.epoch_model(two_epoch_rows, EPOCHS)            # nothing fixed: no raise
    assert em.prior is None
    s = em.fit(fixed=[(1, MU), (3, MU)], **FIT)
    assert s.prior_list[1] is None and s.prior_list[3] is None
    assert s.prior_list[0].mu == pytest.approx(default_fit.prior_list[0].mu, rel=1e-12)
    with pytest.raises(ValueError, match=SCALE_MSG):
        em.fit(**FIT)


def _notices(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [str(x.message) for x in w if issubclass(x.category, SvgdAssumptionWarning)]


def test_last_epoch_scale_notice(jc, two_epoch_rows):
    prior = GaussPrior(ci=[0.5, 8.0], lower=0.0)
    msgs = _notices(lambda: jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, prior=prior, **FIT))
    assert any("last epoch" in m for m in msgs), msgs
    msgs = _notices(lambda: jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, prior=prior,
                                    fixed=[(1, MU)], **FIT))
    assert not any("last epoch" in m for m in msgs), msgs
    msgs = _notices(lambda: jc.svgd(two_epoch_rows, epoch_starts=EPOCHS, prior=prior,
                                    tied=[(0, [0, 1])], **FIT))
    assert not any("last epoch" in m for m in msgs), msgs
    em = jc.epoch_model(two_epoch_rows, EPOCHS, prior=prior)
    msgs = _notices(lambda: em.fit(fixed=[(1, MU)], **FIT))      # epoch 0 only
    assert any("last epoch" in m for m in msgs), msgs
    msgs = _notices(lambda: em.fit(fixed=[(3, MU)], **FIT))
    assert not any("last epoch" in m for m in msgs), msgs
