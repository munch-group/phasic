"""Epoch (daisy-chain) likelihood is the CONDITIONAL likelihood.

Plan of record: epoch-normalization-plan.md (v2). Observations beyond a
joint-probability graph's reward caps cannot be passed to inference, so
every observation's probability must be normalised by the table mass
Z(theta, alpha) of its own (exposure-scaled) parameters. These tests pin
that against INDEPENDENT closed forms (not phasic's own tables):

* n=2 coalescent, one feature: K ~ Geometric, P(k) = (1-q) q^k with
  q = 2 alpha mu / (c + 2 alpha mu); two epochs by conditioning on the
  coalescence epoch (verified against brute-force integration to 1e-14
  during planning).
* n=3 coalescent, two features (singleton, doubleton counts), which
  separates per-feature caps from total caps.

Plus: gradients (FD and exact_final_grad, with exposure, under vmap), a
deterministic expected-log-likelihood gate that recovers the truth and
that the old raw likelihood / a shared-Z normalisation would fail, a
replicate-MLE gate on data from an independent simulator with
exposure-correlated dropout, the zero-mass exact-block limit with
exposure, and rejection of final_read='stopprob' for fits (decision D2a).
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import comb, gammainc
from scipy.stats import poisson

from phasic import Graph

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from test_notebook_model_recovery import coalescent_indexed  # noqa: E402

L = 3


# ------------------------------------------------------------ closed forms
def p_n2(k, c, mu, alpha=1.0):
    m = 2 * alpha * mu
    q = m / (c + m)
    return (1 - q) * q ** k


def p_n2_two_epoch(k, c1, mu1, c2, mu2, t1, alpha=1.0):
    m1, m2 = 2 * alpha * mu1, 2 * alpha * mu2
    q1, q2 = m1 / (c1 + m1), m2 / (c2 + m2)
    a = (1 - q1) * q1 ** k * gammainc(k + 1, (c1 + m1) * t1)
    b = np.exp(-c1 * t1) * sum(
        poisson.pmf(k - j, m1 * t1) * (1 - q2) * q2 ** j for j in range(k + 1))
    return a + b


def p_n3(k1, k2, c, mu):
    a = mu / (c + mu)
    q = 2 * mu / (c + 2 * mu)
    tot = 0.0
    for n in range(k1 + 1):
        s = k1 - n + k2
        tot += (1 - a) * a ** n * (1 - q) * q ** s * comb(s, k2) * 0.5 ** s
    return tot


def normalised(ps):
    ps = np.asarray(ps, dtype=float)
    return ps / ps.sum()


# ------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def g2():
    return coalescent_indexed(2)


def _jc(g_idx, **caps):
    g, idx = g_idx
    return g.joint_prob_graph(idx, discrete=False, **caps)


def _builder(jc, rows, epoch_starts, **kw):
    mapped = jc._map_joint_observations_to_indices(rows, seed=0)
    model, *_ = jc._daisy_chain_svgd_model(
        observed_indices=np.asarray(mapped), epoch_starts=epoch_starts,
        t_eval=200.0, **kw)
    return model


def _assert_no_degenerate_outcomes(jc):
    """Every outcome must map to exactly one t-vertex (the conditional
    likelihood numerator is otherwise one vertex's mass, plan §3)."""
    T = jc.joint_prob_table()
    keys = [tuple(r) for r in T.iloc[:, :-1].to_numpy()]
    assert len(keys) == len(set(keys)), "degenerate outcomes in fixture"


ROWS_N2 = [[k, 0] for k in range(L + 1)]


# ------------------------------------------------------------ forward
@pytest.mark.parametrize("caps", [dict(reward_limit=L), dict(tot_reward_limit=L)],
                         ids=["reward_limit", "tot_reward_limit"])
def test_n2_one_epoch_matches_closed_form_with_exposure(g2, caps):
    jc = _jc(g2, **caps)
    _assert_no_degenerate_outcomes(jc)
    c, mu = 1.3, 0.7
    em = jc.epoch_model(ROWS_N2, [0.0])
    out = np.asarray(em.model(jnp.asarray([c, mu]))[0])
    ref = normalised([p_n2(k, c, mu) for k in range(L + 1)])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0)

    alphas = [0.5, 1.0, 3.0]
    rows = ROWS_N2 * len(alphas)
    al = np.repeat(alphas, L + 1)
    m = _builder(jc, rows, [0.0], exposure_arr=al, exposure_param_index=1)
    out = np.asarray(m(jnp.asarray([c, mu]))[0])
    ref = np.concatenate([normalised([p_n2(k, c, mu, a) for k in range(L + 1)])
                          for a in alphas])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("alpha", [1.0, 3.0])
def test_n2_two_epochs_converge_to_closed_form(g2, alpha):
    """Intermediate epochs use a discretised stop probability (first
    order in 1/granularity): measured rel err 4.2e-5 at 1e4 and 4.2e-6
    at 1e5 (planning, Batch 0)."""
    jc = _jc(g2, reward_limit=L)
    th = [2.0, 0.7, 7.0, 1.5]
    ref = normalised([p_n2_two_epoch(k, *th, 0.5, alpha) for k in range(L + 1)])
    errs = {}
    for gran in (10_000, 100_000):
        m = _builder(jc, ROWS_N2, [0.0, 0.5], granularity=gran,
                     exposure_arr=np.full(L + 1, alpha), exposure_param_index=1)
        out = np.asarray(m(jnp.asarray(th))[0])
        errs[gran] = np.max(np.abs(out / ref - 1))
    assert errs[100_000] < 1e-5, errs
    assert errs[100_000] < errs[10_000] / 5, errs  # first-order convergence


@pytest.mark.parametrize("caps,limit,tot", [
    (dict(reward_limit=2), 2, np.inf),
    (dict(tot_reward_limit=2), None, 2),
    (dict(reward_limit=2, tot_reward_limit=3), 2, 3),
], ids=["per_feature", "total", "both"])
def test_n3_caps_match_closed_form(caps, limit, tot):
    g3 = coalescent_indexed(3)
    jc = _jc(g3, **caps)
    _assert_no_degenerate_outcomes(jc)
    c, mu = 1.1, 0.6
    allowed = [(k1, k2) for k1 in range(12) for k2 in range(12)
               if (limit is None or (k1 <= limit and k2 <= limit))
               and k1 + k2 <= tot]
    rows = [[k1, k2, 0] for k1, k2 in allowed]
    em = jc.epoch_model(rows, [0.0])
    out = np.asarray(em.model(jnp.asarray([c, mu]))[0])
    ref = normalised([p_n3(k1, k2, c, mu) for k1, k2 in allowed])
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0)
    # every reachable outcome is in the closed-form allowed set and v.v.
    T = jc.joint_prob_table()
    assert len(T) == len(allowed)


def test_raw_forward_seam_and_row_normalisation(g2):
    jc = _jc(g2, reward_limit=L)
    th = jnp.asarray([2.0, 0.7, 7.0, 1.5])
    m = _builder(jc, ROWS_N2, [0.0, 0.5])
    raw = np.asarray(m._raw_forward(th))
    np.testing.assert_allclose(np.asarray(m(th)[0]), raw / raw.sum(),
                               rtol=1e-14, atol=0)
    al = np.array([0.5, 3.0, 0.5, 3.0])
    me = _builder(jc, ROWS_N2, [0.0, 0.5], exposure_arr=al,
                  exposure_param_index=1)
    rows = np.asarray(me._raw_forward(th))  # (n_unique, n_t)
    assert rows.shape[0] == 2
    out = np.asarray(me(th)[0])
    tvi = [int(x) for x in jc.joint_stop_prob_graph()._t_vertex_indices]
    mapped = jc._map_joint_observations_to_indices(ROWS_N2, seed=0)
    pos = [tvi.index(v) for v in mapped]
    u = [0, 1, 0, 1]  # np.unique([0.5, 3.0]) order
    expect = [rows[u[i], pos[i]] / rows[u[i]].sum() for i in range(4)]
    np.testing.assert_allclose(out, expect, rtol=1e-14, atol=0)


# ------------------------------------------------------------ gradients
@pytest.mark.parametrize("epochs", [1, 2])
@pytest.mark.parametrize("use_exposure", [False, True], ids=["noexp", "exp"])
def test_gradients_match_closed_form(g2, epochs, use_exposure):
    jc = _jc(g2, reward_limit=L)
    rows = ROWS_N2 * 3
    counts = jnp.asarray([5., 3., 2., 1.] * 3)
    al = np.repeat([0.5, 1.0, 3.0], L + 1) if use_exposure else np.ones(len(rows))
    es = [0.0] if epochs == 1 else [0.0, 0.5]
    th = np.array([1.3, 0.7]) if epochs == 1 else np.array([2.0, 0.7, 7.0, 1.5])
    kw = dict(exposure_arr=al, exposure_param_index=1) if use_exposure else {}

    def cf(t):
        tot = 0.0
        for i, (r, a) in enumerate(zip(rows, al)):
            if epochs == 1:
                ps = [p_n2(k, t[0], t[1], a) for k in range(L + 1)]
            else:
                ps = [p_n2_two_epoch(k, *t, 0.5, a) for k in range(L + 1)]
            tot += float(counts[i]) * np.log(normalised(ps)[r[0]])
        return tot

    h = 1e-5
    g_cf = np.array([(cf(th + h * e) - cf(th - h * e)) / (2 * h)
                     for e in np.eye(len(th))])
    scale = np.max(np.abs(g_cf))
    grads = {}
    for efg in (False, True):
        m = _builder(jc, rows, es, exact_final_grad=efg, granularity=100_000, **kw)

        def loglik(t, m=m):
            return jnp.sum(counts * jnp.log(m(t)[0]))

        grads[efg] = np.asarray(jax.grad(loglik)(jnp.asarray(th)))
        P = jnp.asarray(th * np.exp(np.random.default_rng(0).normal(0, .1, (3, len(th)))))
        vg = np.asarray(jax.vmap(jax.grad(loglik))(P))
        loop = np.stack([np.asarray(jax.grad(loglik)(p)) for p in P])
        assert np.max(np.abs(vg - loop)) / np.max(np.abs(loop)) < 1e-7
    tol_cf = 1e-8 if epochs == 1 else 2e-5  # two epochs: granularity error
    for efg in (False, True):
        assert np.max(np.abs(grads[efg] - g_cf)) / scale < tol_cf, (efg, grads, g_cf)
    assert np.max(np.abs(grads[True] - grads[False])) / scale < 1e-7


def test_zero_mass_exposure_exact_block_gives_zero_not_nan(g2, monkeypatch):
    """At a zero handoff mass the exact block returns zero rows and zero
    J; the exposure quotient rule must keep the exact zero (0/0 guard)."""
    jc = _jc(g2, reward_limit=L)
    al = np.array([0.5, 3.0, 0.5, 3.0])
    m = _builder(jc, ROWS_N2, [0.0, 0.5], exact_final_grad=True,
                 exposure_arr=al, exposure_param_index=1)
    th = jnp.asarray([2.0, 0.7, 7.0, 1.5])
    calls = []
    orig = Graph._sojourn_grad_theta_subset

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(Graph, "_sojourn_grad_theta_subset", spy)
    monkeypatch.setattr(Graph, "stop_probability",
                        lambda self, *a, **k: np.zeros(self.vertices_length()))
    out, vjp = jax.vjp(lambda t: m(t)[0], th)
    (g,) = vjp(jnp.ones_like(out))
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), g
    assert np.array_equal(g[2:], np.zeros(2)), g
    assert not calls


# ------------------------------------------------------------ estimation
def test_deterministic_gate_recovers_truth_and_rejects_raw_and_shared_z(g2):
    """Maximise sum_k p_true(k) log p_impl(k | c) with p_true from the
    closed form (no Monte-Carlo error). mu fixed: one epoch identifies
    only the ratio. The raw likelihood and a shared-Z normalisation are
    evaluated on the same closed form to show they would miss by far
    more than the tolerance."""
    jc = _jc(g2, reward_limit=L)
    c_true, mu = 1.3, 0.7
    alphas = [0.3, 1.0, 4.0]
    rows, al, w = [], [], []
    for a in alphas:
        p = normalised([p_n2(k, c_true, mu, a) for k in range(L + 1)])
        for k in range(L + 1):
            rows.append([k, 0]); al.append(a); w.append(p[k])
    w = np.asarray(w)
    m = _builder(jc, rows, [0.0], exposure_arr=np.asarray(al), exposure_param_index=1)

    def neg(cc):
        return -float(np.sum(w * np.log(np.asarray(m(jnp.asarray([cc, mu]))[0]))))

    r = minimize_scalar(neg, bounds=(0.1, 10), method="bounded", options=dict(xatol=1e-10))
    assert abs(r.x / c_true - 1) < 1e-6, r.x

    def variant(cc, kind):
        Zs = {a: sum(p_n2(k, cc, mu, a) for k in range(L + 1)) for a in alphas}
        d = {"raw": lambda a: 1.0, "shared": lambda a: np.mean(list(Zs.values()))}[kind]
        return -sum(wi * np.log(p_n2(r_[0], cc, mu, a) / d(a))
                    for r_, a, wi in zip(rows, al, w))

    for kind in ("raw", "shared"):
        rv = minimize_scalar(lambda cc: variant(cc, kind), bounds=(0.1, 10),
                             method="bounded", options=dict(xatol=1e-10))
        assert abs(rv.x / c_true - 1) > 0.1, (kind, rv.x)


def test_tied_two_epoch_model_recovers_truth_deterministically():
    """The reproducer model (n=4, coal 5, mu 3 fixed, reward_limit 3,
    ~11% overflow): the raw epoch likelihood's maximiser was 6.377; the
    normalised one must be 5."""
    g4 = coalescent_indexed(4)
    jd = g4[0].joint_prob_graph(g4[1], reward_limit=3, discrete=True)
    jd.update_weights([5.0, 3.0])
    T = jd.joint_prob_table()
    p = normalised(T["prob"].to_numpy())
    rows = [list(r) for r in T.iloc[:, :-1].to_numpy()]
    jc = _jc(g4, reward_limit=3)
    em = jc.epoch_model(rows, [0.0, 1.0])

    def neg(cc):
        return -float(np.sum(p * np.log(np.asarray(
            em.model(jnp.asarray([cc, 3.0, cc, 3.0]))[0]))))

    r = minimize_scalar(neg, bounds=(1, 20), method="bounded", options=dict(xatol=1e-8))
    assert abs(r.x - 5.0) < 1e-5, r.x


def test_replicate_mle_independent_simulator_with_exposure_dropout(g2):
    """Data from an independent simulator (exponential coalescence time,
    Poisson mutations, rows over the cap DROPPED; dropout more likely
    at large exposure). 50 replicate MLEs must be unbiased (|z| < 3.5);
    the raw likelihood must be strongly biased on the same data."""
    jc = _jc(g2, reward_limit=L)
    c_true, mu = 1.3, 0.7
    alphas = np.array([0.3, 1.0, 4.0])
    rows = ROWS_N2 * len(alphas)
    al = np.repeat(alphas, L + 1)
    m = _builder(jc, rows, [0.0], exposure_arr=al, exposure_param_index=1)
    raw_rows_fn = m._raw_forward
    mapped = jc._map_joint_observations_to_indices(ROWS_N2, seed=0)
    tvi = [int(x) for x in jc.joint_stop_prob_graph()._t_vertex_indices]
    pos = np.asarray([tvi.index(v) for v in mapped])
    rng = np.random.default_rng(12345)
    R, N = 50, 3000
    est, est_raw = [], []
    for _ in range(R):
        a_draw = rng.choice(alphas, size=N)
        T = rng.exponential(1 / c_true, size=N)
        K = rng.poisson(2 * a_draw * mu * T)
        keep = K <= L
        counts = np.zeros(len(rows))
        for a, k in zip(a_draw[keep], K[keep]):
            counts[int(np.where(alphas == a)[0][0]) * (L + 1) + k] += 1
        cj = jnp.asarray(counts)

        def neg(cc):
            return -float(jnp.sum(cj * jnp.log(m(jnp.asarray([cc, mu]))[0])))

        def neg_raw(cc):
            rr = np.asarray(raw_rows_fn(jnp.asarray([cc, mu])))  # (3, n_t)
            return -float(np.sum(counts * np.log(rr[:, pos].ravel())))

        est.append(minimize_scalar(neg, bounds=(0.2, 10), method="bounded",
                                   options=dict(xatol=1e-8)).x)
        est_raw.append(minimize_scalar(neg_raw, bounds=(0.2, 10), method="bounded",
                                       options=dict(xatol=1e-8)).x)
    est, est_raw = np.asarray(est), np.asarray(est_raw)
    z = (est.mean() - c_true) / (est.std(ddof=1) / np.sqrt(R))
    z_raw = (est_raw.mean() - c_true) / (est_raw.std(ddof=1) / np.sqrt(R))
    assert abs(z) < 3.5, (z, est.mean())
    assert abs(z_raw) > 10, (z_raw, est_raw.mean())


# ------------------------------------------------------------ D2(a)
def test_stopprob_rejected_for_epoch_fits(g2):
    from phasic.svgd_config import SvgdConfigError
    jc = _jc(g2, reward_limit=L)
    with pytest.raises(SvgdConfigError, match="not supported for epoch fits"):
        jc.svgd(ROWS_N2, epoch_starts=[0.0, 0.5], final_read="stopprob",
                fixed=[(1, 0.7)], n_iterations=1, n_particles=2)
    with pytest.raises(ValueError, match="not supported for epoch models"):
        jc.epoch_model(ROWS_N2, [0.0, 0.5], final_read="stopprob")
