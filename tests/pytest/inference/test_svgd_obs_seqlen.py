"""Tests for the obs_seqlen / mu_index per-observation rescaling in SVGD.

Per-observation sequence-length correction lets users fit
coalescent-with-mutation models to segments of different lengths.
For each observation i, the wrapped model evaluates PMF at theta_i
where theta_i[mu_index] = theta[mu_index] * L_i.

Tests cover:
- Backward compatibility (obs_seqlen=None and obs_seqlen=1.0).
- The wrapped model matches a manual per-obs PMF computation.
- Scalar L != 1 shifts the posterior in the expected direction.
- Construction-time validation rejects misuse.
"""

import numpy as np
import pytest

# Enable JAX x64 before importing jax.numpy. The library's deferred
# JAX init (phasic._ensure_jax_active) only enables x64 when phasic
# is the first to import jax, which is fragile under pytest. Setting
# the env var here is reliable.
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from phasic import Graph, SVGD
from phasic.svgd import (
    _wrap_model_with_obs_seqlen,
    SparseObservations,
)


@pytest.fixture(autouse=True)
def _reset_jax_state():
    """Match the cache-reset pattern from test_svgd_correctness.py so
    inference outputs are deterministic regardless of test order."""
    jax.clear_caches()
    yield
    jax.clear_caches()


def _build_two_param_exponential():
    """Build a parameterized graph where rate = theta[0] + 0 * theta[1].

    The second parameter is a dummy that does not enter any edge weight.
    This lets us pick mu_index=0 cleanly while keeping theta_dim=2 so
    out-of-range mu_index validation has something to fail against.
    """
    g = Graph(1, parameterized=True)
    v_start = g.starting_vertex()
    v_transient = g.find_or_create_vertex([1])
    v_absorb = g.find_or_create_vertex([0])
    v_start.add_edge(v_transient, 1.0)
    # Coefficients [1.0, 0.0]: weight = 1.0*theta[0] + 0.0*theta[1] = theta[0].
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0, 0.0])
    return g


def _build_one_param_exponential():
    """Single-parameter exponential. Used to drive a 2-feature dataset
    so we can also test mu_index validation against theta_dim."""
    g = Graph(1, parameterized=True)
    v_start = g.starting_vertex()
    v_transient = g.find_or_create_vertex([1])
    v_absorb = g.find_or_create_vertex([0])
    v_start.add_edge(v_transient, 1.0)
    v_transient.add_edge_parameterized(v_absorb, 0.0, [1.0])
    return g


def _make_model_two_param():
    g = _build_two_param_exponential()
    return Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2
    )


def _make_model_one_param():
    g = _build_one_param_exponential()
    return Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=1
    )


def _flat_prior(phi):
    """Very wide normal prior in unconstrained space; effectively flat."""
    return -0.5 * jnp.sum((phi / 10.0) ** 2)


def test_obs_seqlen_none_no_overhead_attribute():
    """obs_seqlen=None stores None and leaves the model unwrapped."""
    model = _make_model_two_param()
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=0.5, size=20)

    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=8,
        n_iterations=1,
        learning_rate=0.01,
        seed=0,
        verbose=False,
    )
    assert svgd.obs_seqlen is None
    assert svgd.mu_index is None
    # Model attribute is the original user-supplied object.
    assert svgd.model is model


def test_obs_seqlen_one_does_not_change_log_prob():
    """L=1 means theta[mu_index]*1 = theta[mu_index]: the wrapped model
    must produce log-probabilities identical to the un-wrapped baseline."""
    model = _make_model_two_param()
    rng = np.random.default_rng(1)
    data = jnp.asarray(rng.exponential(scale=0.5, size=15))

    # positive_params=False so the rescaling acts directly on theta,
    # not on softplus(theta). Comparing math-equivalent paths.
    svgd_baseline = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=4,
        n_iterations=1,
        learning_rate=0.01,
        seed=1,
        verbose=False,
        positive_params=False,
    )
    svgd_scaled = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=4,
        n_iterations=1,
        learning_rate=0.01,
        seed=1,
        verbose=False,
        positive_params=False,
        obs_seqlen=1.0,
        mu_index=0,
    )

    theta = jnp.array([2.0, 1.0])
    lp_base = svgd_baseline._log_prob_unified(theta)
    lp_scaled = svgd_scaled._log_prob_unified(theta)
    np.testing.assert_allclose(lp_base, lp_scaled, rtol=1e-10, atol=1e-10)


def test_obs_seqlen_scalar_matches_rescaled_theta():
    """With a scalar L applied to all observations, the wrapped model
    log-prob at theta must equal the unwrapped model log-prob evaluated
    at theta with theta[mu_index] replaced by theta[mu_index]*L."""
    model = _make_model_two_param()
    rng = np.random.default_rng(2)
    data = jnp.asarray(rng.exponential(scale=0.5, size=12))
    L = 2.5

    # positive_params=False so the rescaling acts on theta, not on
    # softplus(theta) — this lets us compare against a manual
    # theta-space computation.
    svgd_baseline = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=4,
        n_iterations=1,
        learning_rate=0.01,
        seed=2,
        verbose=False,
        positive_params=False,
    )
    svgd_scaled = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=4,
        n_iterations=1,
        learning_rate=0.01,
        seed=2,
        verbose=False,
        positive_params=False,
        obs_seqlen=L,
        mu_index=0,
    )

    theta = jnp.array([2.0, 1.0])
    theta_rescaled = theta.at[0].multiply(L)

    lp_scaled = svgd_scaled._log_prob_unified(theta)
    lp_baseline_at_rescaled = svgd_baseline._log_prob_unified(theta_rescaled)

    # log_prob = log_lik + log_prior. log_prior is computed on the raw
    # theta (not theta_eff), so we must subtract the baseline prior at
    # theta_rescaled and add it back at theta to compare the
    # likelihood-only contribution. Easier: assert log-likelihood is
    # equal by recomputing log-prior under each call.
    log_pri_theta = _flat_prior(theta)
    log_pri_rescaled = _flat_prior(theta_rescaled)
    log_lik_scaled = lp_scaled - log_pri_theta
    log_lik_baseline = lp_baseline_at_rescaled - log_pri_rescaled
    np.testing.assert_allclose(
        log_lik_scaled, log_lik_baseline, rtol=1e-9, atol=1e-9
    )


def test_obs_seqlen_per_obs_matches_manual_sum():
    """With a per-observation L vector, the wrapped log-likelihood must
    equal sum_i log PMF(c_i | theta_with_mu = theta_mu * L_i) computed
    via the unwrapped model one observation at a time."""
    model = _make_model_two_param()
    rng = np.random.default_rng(3)
    data = jnp.asarray(rng.exponential(scale=0.5, size=8))
    L = jnp.asarray(rng.uniform(0.5, 3.0, size=8))

    svgd_scaled = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=4,
        n_iterations=1,
        learning_rate=0.01,
        seed=3,
        verbose=False,
        positive_params=False,
        obs_seqlen=L,
        mu_index=0,
    )

    theta = jnp.array([2.0, 1.0])
    lp_scaled = svgd_scaled._log_prob_unified(theta)

    # Manual reference: sum log PMF(c_i | theta_eff_i).
    pmf_per_obs = []
    for i in range(len(data)):
        theta_eff = theta.at[0].multiply(L[i])
        pmf_i, _ = model(theta_eff, data[i:i+1], rewards=None)
        pmf_per_obs.append(pmf_i[0])
    pmf_per_obs = jnp.array(pmf_per_obs)
    manual_log_lik = jnp.sum(jnp.log(pmf_per_obs + 1e-10))
    manual_log_prob = manual_log_lik + _flat_prior(theta)

    np.testing.assert_allclose(
        lp_scaled, manual_log_prob, rtol=1e-9, atol=1e-9
    )


def test_obs_seqlen_shifts_posterior_inverse_to_L():
    """Doubling L should halve the inferred theta[mu_index].

    The data is generated at rate r_true. Fitting with obs_seqlen=L
    interprets each observation as if its rate were softplus(theta[0])*L,
    so the inferred softplus(theta_mean[0]) should be approximately
    r_true / L.
    """
    model = _make_model_two_param()
    rng = np.random.default_rng(4)
    r_true = 4.0
    n_samples = 200
    data = jnp.asarray(rng.exponential(scale=1.0 / r_true, size=n_samples))
    L = 2.0

    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=_flat_prior,
        theta_dim=2,
        n_particles=40,
        n_iterations=400,
        learning_rate=0.05,
        seed=4,
        verbose=False,
        obs_seqlen=L,
        mu_index=0,
    )
    svgd.optimize()

    # SVGD reports theta in unconstrained (phi) space; convert via
    # softplus to get the effective rate.
    inferred_phi = float(svgd.theta_mean[0])
    inferred_rate = float(jax.nn.softplus(svgd.theta_mean[0]))
    expected_rate = r_true / L
    # 30% tolerance: SVGD with a finite particle count + iterations
    # does not converge exactly. The order of magnitude must be right.
    assert abs(inferred_rate - expected_rate) / expected_rate < 0.3, (
        f"Expected softplus(theta[0]) near {expected_rate} (= r_true/L), "
        f"got softplus({inferred_phi})={inferred_rate}."
    )


def test_validation_missing_mu_index():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"mu_index must be provided"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=2.0,
        )


def test_validation_mu_index_without_obs_seqlen():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"mu_index was provided but obs_seqlen"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            mu_index=0,
        )


def test_validation_mu_index_out_of_range():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"mu_index=5 is out of range"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=2.0,
            mu_index=5,
        )


def test_validation_wrong_length_vector():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"obs_seqlen length \(2\) does not match"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=np.array([1.0, 2.0]),
            mu_index=0,
        )


def test_validation_negative_L():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"strictly positive"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=np.array([1.0, -2.0, 3.0]),
            mu_index=0,
        )


def test_validation_2d_obs_seqlen_rejected():
    model = _make_model_two_param()
    data = np.array([0.5, 1.0, 1.5])
    with pytest.raises(ValueError, match=r"None, a scalar, or a 1D array"):
        SVGD(
            model=model,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=np.array([[1.0], [2.0], [3.0]]),
            mu_index=0,
        )


def test_multivariate_obs_seqlen_one_no_change():
    """Multivariate 2D observations: obs_seqlen=1 must reproduce the
    unwrapped multivariate model exactly."""
    g = _build_two_param_exponential()
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(
        g, nr_moments=2, discrete=False
    )

    rng = np.random.default_rng(10)
    data = jnp.asarray(rng.exponential(scale=0.5, size=(6, 2)))
    n_vertices = 3
    rewards_2d = jnp.array(
        [[1.0, 1.0, 0.0],
         [1.0, 0.5, 0.0]],
        dtype=jnp.float64,
    )
    # Provide positive theta_init so the model-validation block doesn't
    # try to evaluate at negative theta (which would create negative
    # edge weights). The math comparison happens at a fixed theta below.
    theta_init = jnp.array([[2.0, 1.0]] * 4, dtype=jnp.float64)
    n_particles = theta_init.shape[0]

    svgd_baseline = SVGD(
        model=model_mv,
        observed_data=data,
        prior=_flat_prior,
        theta_init=theta_init,
        theta_dim=2,
        n_particles=n_particles,
        n_iterations=1,
        learning_rate=0.01,
        seed=10,
        verbose=False,
        positive_params=False,
        rewards=rewards_2d,
    )
    svgd_scaled = SVGD(
        model=model_mv,
        observed_data=data,
        prior=_flat_prior,
        theta_init=theta_init,
        theta_dim=2,
        n_particles=n_particles,
        n_iterations=1,
        learning_rate=0.01,
        seed=10,
        verbose=False,
        positive_params=False,
        rewards=rewards_2d,
        obs_seqlen=1.0,
        mu_index=0,
    )

    theta = jnp.array([2.0, 1.0])
    lp_base = svgd_baseline._log_prob_unified(theta, rewards=rewards_2d)
    lp_scaled = svgd_scaled._log_prob_unified(theta, rewards=rewards_2d)
    np.testing.assert_allclose(lp_base, lp_scaled, rtol=1e-10, atol=1e-10)


def test_multivariate_obs_seqlen_per_segment_matches_manual():
    """Multivariate with per-segment L: wrapped log-likelihood must
    equal a manual per-segment per-feature computation using
    theta-rescaled PMFs."""
    g = _build_two_param_exponential()
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(
        g, nr_moments=2, discrete=False
    )

    rng = np.random.default_rng(11)
    n_segments = 5
    n_features = 2
    data = jnp.asarray(rng.exponential(scale=0.5, size=(n_segments, n_features)))
    L = jnp.asarray(rng.uniform(0.7, 2.5, size=n_segments))
    n_vertices = 3
    rewards_2d = jnp.array(
        [[1.0, 1.0, 0.0],
         [1.0, 0.5, 0.0]],
        dtype=jnp.float64,
    )

    theta_init = jnp.array([[2.0, 1.0]] * 4, dtype=jnp.float64)
    n_particles = theta_init.shape[0]
    svgd = SVGD(
        model=model_mv,
        observed_data=data,
        prior=_flat_prior,
        theta_init=theta_init,
        theta_dim=2,
        n_particles=n_particles,
        n_iterations=1,
        learning_rate=0.01,
        seed=11,
        verbose=False,
        positive_params=False,
        rewards=rewards_2d,
        obs_seqlen=L,
        mu_index=0,
    )

    theta = jnp.array([2.0, 1.0])
    lp_scaled = svgd._log_prob_unified(theta, rewards=rewards_2d)

    # Manual: per-segment PMF with theta_eff = theta.at[0].multiply(L_i),
    # evaluated on the segment row.
    pmf_per_seg = []
    for i in range(n_segments):
        theta_eff = theta.at[0].multiply(L[i])
        pmf_i, _ = model_mv(theta_eff, data[i:i+1, :], rewards=rewards_2d)
        pmf_per_seg.append(pmf_i[0])
    pmf_per_seg = jnp.stack(pmf_per_seg)  # (n_segments, n_features)
    manual_log_lik = jnp.sum(jnp.log(pmf_per_seg + 1e-10))
    manual_log_prob = manual_log_lik + _flat_prior(theta)

    np.testing.assert_allclose(lp_scaled, manual_log_prob, rtol=1e-9, atol=1e-9)


def test_multivariate_obs_seqlen_length_check_uses_n_segments():
    """For 2D observed_data, obs_seqlen must align with axis 0
    (segments), not the total number of elements."""
    g = _build_two_param_exponential()
    model_mv = Graph.pmf_and_moments_from_graph_multivariate(
        g, nr_moments=2, discrete=False
    )
    data = jnp.asarray(np.array([[0.5, 1.0], [0.7, 1.2], [0.9, 1.4]]))
    n_vertices = 3
    rewards_2d = jnp.array(
        [[1.0, 1.0, 0.0], [1.0, 0.5, 0.0]], dtype=jnp.float64
    )
    # Length-6 (= n_segments * n_features) is wrong; must be length-3.
    with pytest.raises(ValueError, match=r"obs_seqlen length \(6\) does not match"):
        SVGD(
            model=model_mv,
            observed_data=data,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            positive_params=False,
            rewards=rewards_2d,
            obs_seqlen=np.ones(6),
            mu_index=0,
        )


def test_validation_sparse_observations_rejected():
    model = _make_model_two_param()
    sparse = SparseObservations(
        values=jnp.array([1.0, 2.0, 3.0]),
        features=jnp.array([0, 0, 1], dtype=jnp.int32),
        n_features=2,
        slices=((0, 2), (2, 3)),
    )
    with pytest.raises(NotImplementedError, match=r"SparseObservations"):
        SVGD(
            model=model,
            observed_data=sparse,
            theta_dim=2,
            n_particles=4,
            n_iterations=1,
            verbose=False,
            obs_seqlen=2.0,
            mu_index=0,
        )
