"""Discrete models: one row-sum rule, and stepping never goes negative (bug 4).

In a discrete phase-type model each vertex's outgoing weights are per-step
transition probabilities; the remainder stays put. Before this fix:

- rows summing up to 1.0001 were accepted (pmf above 1, negative pmf, cdf
  above 1), the elimination read-outs had no check at all (variance -0.222
  at p = 1.5), and negative weights were never refused;
- the step subtracted each outflow from its source, so rows summing to
  exactly 1 (what discretize() and joint-probability graphs have) left
  -1e-16 residuals: negative masses and negative pmf values.

Now every weight must be finite and >= 0 and every row (starting vertex
included) must sum to at most 1 + 1e-9, and the step caps each outflow at
the mass still present at its source.
Plan: bug4-discrete-rowsum-plan.md (test numbers refer to its section 3).
"""
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import phasic
from phasic import Graph

from test_notebook_model_recovery import (coalescent_graph, coalescent_indexed,
                                          two_island_graph)

ROW_MSG = "outgoing rate <= 1"
WEIGHT_MSG = "finite, non-negative outgoing weights"


# ----------------------------------------------------------------- builders
def geom(p):
    """Geometric chain: start -> v (1), v -> absorbing (p)."""
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0)
    v.add_edge(a, p)
    g.is_discrete = True
    return g


def pgeom():
    """Parameterized geometric chain: the absorption probability is theta[0]."""
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0])
    g.is_discrete = True
    return g


def two_edge(w1, w2):
    """start -> v; v -> a (theta0), v -> b (theta1)."""
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    a = g.find_or_create_vertex([0])
    b = g.find_or_create_vertex([2])
    s.add_edge(v, 1.0)
    v.add_edge(a, [1.0, 0.0])
    v.add_edge(b, [0.0, 1.0])
    g.is_discrete = True
    g.update_weights([w1, w2])
    return g


def start_split(w1, w2):
    """Initial distribution over two transient vertices, then absorption."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1])
    b = g.find_or_create_vertex([2])
    z = g.find_or_create_vertex([0])
    s.add_edge(a, w1)
    s.add_edge(b, w2)
    a.add_edge(z, 0.5)
    b.add_edge(z, 0.25)
    g.is_discrete = True
    return g


def safe_discrete():
    """Native discrete graph, every row sums to <= 0.6."""
    g = Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([3])
    z = g.find_or_create_vertex([0])
    s.add_edge(a, 0.6)
    a.add_edge(b, 0.35)
    a.add_edge(c, 0.2)
    b.add_edge(a, 0.1)
    b.add_edge(z, 0.45)
    c.add_edge(z, 0.3)
    c.add_edge(a, 0.25)
    g.is_discrete = True
    return g


def row_sums(g):
    return [sum(e.weight() for e in g.vertex_at(i).edges())
            for i in range(g.vertices_length())]


def discretized(g):
    d = g.discretize(0.2)
    return d[0] if isinstance(d, tuple) else d


def legit_graphs():
    gi, idx = coalescent_indexed(5)
    J = gi.joint_prob_graph(idx, reward_limit=3)
    J.update_weights([2.0, 0.05])
    c = coalescent_graph(10)
    c.update_weights([0.37])
    t = two_island_graph(8)
    t.update_weights([1.0, 0.3])
    return {"joint_n5": J, "coalescent_n10_discretize": discretized(c),
            "two_island_n8_discretize": discretized(t)}


def readouts(g):
    return {
        "pdf": lambda: g.pdf(2),
        "cdf": lambda: g.cdf(2),
        "stop_probability": lambda: g.stop_probability(2),
        "accumulated_visits": lambda: g.accumulated_visits(2),
        "sample": lambda: g.sample(5),
        "reward_transform_discrete": lambda: g.reward_transform_discrete(
            np.array([0, 2, 0])).pdf(2),
        "expectation": lambda: g.expectation(),
        "variance": lambda: g.variance(),
        "covariance": lambda: g.covariance(np.ones(3), np.ones(3)),
        "moments_discrete": lambda: g.moments(2, discrete=True),
    }


ELIMINATION = ("expectation", "variance", "covariance", "moments_discrete")


# ------------------------------------- test 1: tolerance is 1e-9, not 1e-4
@pytest.mark.parametrize("excess", [1e-6, 1e-5])
@pytest.mark.parametrize("name", list(readouts(geom(0.5))))
def test_small_excess_raises_everywhere(excess, name):
    p = 1.0 + excess
    with pytest.raises(Exception, match=ROW_MSG) as ei:
        readouts(geom(p))[name]()
    msg = str(ei.value)
    assert "index 1" in msg
    printed = float(re.search(r"Is '([^']+)'", msg).group(1))
    assert abs(printed - p) <= 1e-15


# ------------------------- test 2: elimination read-outs at large excess
@pytest.mark.parametrize("excess", [2e-4, 0.5])
@pytest.mark.parametrize("name", ELIMINATION)
def test_elimination_readouts_raise(excess, name):
    with pytest.raises(Exception, match=ROW_MSG):
        readouts(geom(1.0 + excess))[name]()


# ------------------------------ test 3: rounding-level excess is accepted
@pytest.mark.parametrize("excess", [2.2e-16, 1e-15, 1e-12])
def test_rounding_excess_accepted_and_never_negative(excess):
    g = geom(1.0 + excess)
    pdf = np.array([g.pdf(k) for k in range(1, 21)])
    assert np.all(pdf >= 0.0)
    assert pdf[0] == 1.0
    assert g.cdf(20) <= 1.0
    for k in range(1, 21):
        assert np.all(np.asarray(g.stop_probability(k)) >= 0.0)
    assert abs(g.variance()) <= 2e-12
    assert abs(g.expectation() - 1.0) <= 2e-12


# ------------------------ test 4: legitimate graphs through the step
@pytest.mark.parametrize("name", list(legit_graphs()))
def test_step_never_negative_on_legitimate_graphs(name):
    g = legit_graphs()[name]
    rs = row_sums(g)
    assert max(rs) <= 1.0 + 1e-9
    if name == "joint_n5":
        assert max(rs) > 1.0, "precondition: rows above 1 by rounding"
    ctx = g.distribution_context()
    cdf_prev = ctx.cdf()
    for _ in range(400):
        ctx.step()
        assert ctx.pmf() >= 0.0
        assert np.all(np.asarray(ctx.stop_probability(), dtype=float) >= 0.0)
        assert ctx.cdf() >= cdf_prev
        assert ctx.cdf() <= 1.0 + 1e-12
        cdf_prev = ctx.cdf()


# ------------------------------------------- test 5: the starting vertex
def test_starting_vertex_row_sum():
    with pytest.raises(Exception, match=r"\(starting vertex\)"):
        start_split(1.0, 0.5).pdf(1)
    with pytest.raises(Exception, match=r"\(starting vertex\)"):
        start_split(0.5, 0.5 + 1e-5).pdf(1)
    g = start_split(0.5, 0.5 + 1e-12)
    assert g.cdf(200) <= 1.0 + 1e-12
    assert all(g.pdf(k) >= 0.0 for k in range(1, 50))


# -------------------------- test 6: parameterized native discrete paths
def _jax_models():
    import jax.numpy as jnp
    models = {"pmf_from_graph": Graph.pmf_from_graph(pgeom(), discrete=True)}
    for ffi in (False, True):
        mm = Graph.pmf_and_moments_from_graph(pgeom(), discrete=True,
                                              use_ffi=ffi)
        models[f"pmf_and_moments_ffi{ffi}"] = (
            lambda t, k, _m=mm: _m(t, k)[0])
    return models, jnp


@pytest.mark.parametrize("name", ["pmf_from_graph", "pmf_and_moments_ffiFalse",
                                  "pmf_and_moments_ffiTrue"])
def test_parameterized_paths(name):
    models, jnp = _jax_models()
    m = models[name]
    k = jnp.array([1.0, 2.0, 3.0])
    with pytest.raises(Exception, match=ROW_MSG):
        np.asarray(m(jnp.array([1.0 + 1e-5]), k))
    vals = np.asarray(m(jnp.array([1.0 + 1e-12]), k))
    assert np.all(vals >= 0.0)


# ------------------------------- tests 7 and 8: no crash, in a subprocess
_PRELUDE = """
import numpy as np, phasic
from phasic import Graph
def geom(p):
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0); v.add_edge(a, p); g.is_discrete = True
    return g
"""


def _run(body):
    return subprocess.run([sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
                          capture_output=True, text=True, timeout=600)


def _assert_caught(res):
    assert res.returncode == 0, (res.returncode, res.stderr[-500:])
    assert "CAUGHT" in res.stdout and ROW_MSG in res.stdout, res.stdout


def test_distribution_context_does_not_crash():
    _assert_caught(_run("""
        try:
            ctx = geom(1.5).distribution_context()
            ctx.step(); ctx.pmf()
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', e)
    """))


def test_sample_multivariate_discrete_does_not_crash():
    _assert_caught(_run("""
        g = geom(1.5)
        try:
            g.sample_multivariate_discrete(5, np.ones((3, 2)))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', e)
    """))


def test_batched_ffi_does_not_abort():
    _assert_caught(_run("""
        import jax, jax.numpy as jnp
        g = Graph(1); s = g.starting_vertex()
        v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
        s.add_edge(v, 1.0); v.add_edge(a, [1.0]); g.is_discrete = True
        mm = Graph.pmf_and_moments_from_graph(g, discrete=True, use_ffi=True)
        k = jnp.array([1.0, 2.0])
        th = jnp.array([[1.0 + 1e-5], [1.0 + 1e-5]])
        try:
            np.asarray(jax.vmap(lambda t: mm(t, k)[0])(th))
            print('NO_RAISE')
        except Exception as e:
            print('CAUGHT', e)
    """))


# --------------------------------- test 9: a context held by the user
def test_held_context_revalidates_after_update():
    g = pgeom()
    g.update_weights([0.5])
    ctx = g.distribution_context()
    for _ in range(3):
        ctx.step()
    jumps = ctx.jumps()
    g.update_weights([1.5])
    with pytest.raises(Exception, match=ROW_MSG):
        ctx.step()
    assert ctx.jumps() == jumps            # a failed step leaves it untouched
    mass = np.asarray(ctx.stop_probability(), dtype=float)
    g.update_weights([0.25])
    cdf0 = ctx.cdf()
    ctx.step()
    np.testing.assert_allclose(ctx.pmf(), mass[1] * 0.25, rtol=1e-15)
    np.testing.assert_allclose(ctx.cdf(), cdf0 + mass[1] * 0.25, rtol=1e-15)
    np.testing.assert_allclose(np.asarray(ctx.stop_probability(), dtype=float)[1],
                               mass[1] * 0.75, rtol=1e-15)


# ------------------------------------ test 9b: negative and NaN weights
@pytest.mark.parametrize("name", ["pdf", "stop_probability", "sample",
                                  "expectation", "variance"])
def test_negative_weight_raises(name):
    g = two_edge(-0.5, 0.9)
    with pytest.raises(Exception, match=WEIGHT_MSG + ".*negative weight -0.5"):
        readouts(g)[name]()


def test_nan_weight_raises():
    g = Graph(1)
    s = g.starting_vertex()
    v = g.find_or_create_vertex([1])
    a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0)
    v.add_edge(a, float("nan"))
    g.is_discrete = True
    with pytest.raises(Exception, match=WEIGHT_MSG + ".*non-finite"):
        g.pdf(2)
    with pytest.raises(Exception):
        pgeom().update_weights([float("nan")])


# ---------------------------------- test 9c: empty times, moments only
@pytest.mark.parametrize("ffi", [False, True])
@pytest.mark.parametrize("p", [1.0 + 1e-5, 1.5])
def test_empty_times_still_validates(ffi, p):
    import jax.numpy as jnp
    mm = Graph.pmf_and_moments_from_graph(pgeom(), discrete=True, use_ffi=ffi)
    with pytest.raises(Exception, match=ROW_MSG):
        np.asarray(mm(jnp.array([p]), jnp.array([]))[1])


# ------------------------------------------ test 9d: gradient at the bound
def test_gradient_at_the_bound():
    import jax
    import jax.numpy as jnp
    m = Graph.pmf_from_graph(pgeom(), discrete=True)
    k = jnp.array([1.0, 2.0, 3.0])
    loss = lambda t: jnp.sum(jnp.log(m(t, k)))
    with pytest.raises(Exception, match=ROW_MSG):
        np.asarray(jax.grad(loss)(jnp.array([1.0 - 5e-8])))
    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(jnp.array([1.0 - 2e-7])))))
    assert np.all(np.isfinite(np.asarray(m(jnp.array([1.0 - 5e-8]), k))))


def test_gradient_at_the_lower_bound():
    """The finite-difference probe (theta - 1e-7) crosses 0 below 1e-7. A
    native discrete forward refuses a negative weight, so this used to raise;
    since Plan A Batch 1 (fd-domain-steps-plan.md v2.10 §4.4, §7) a refusing
    model probes a slot in [0, 1e-7] with the capped positive rule, so the
    gradient is finite and matches the geometric closed form."""
    import jax
    import jax.numpy as jnp
    m = Graph.pmf_from_graph(pgeom(), discrete=True)
    k = jnp.array([1.0, 2.0, 3.0])
    loss = lambda t: jnp.sum(jnp.log(m(t, k)))
    for th in (5e-8, 2e-7):
        g = float(np.asarray(jax.grad(loss)(jnp.array([th])))[0])
        exact = float(np.sum(1.0 / th - (np.asarray(k) - 1.0) / (1.0 - th)))
        assert g == pytest.approx(exact, rel=1e-6)
    assert np.all(np.isfinite(np.asarray(m(jnp.array([5e-8]), k))))


# -------------------------------------------- test 9e: not fail-soft
def test_row_sum_error_is_not_a_rate_blowup():
    import jax.numpy as jnp
    m = Graph.pmf_from_graph(pgeom(), discrete=True)
    with pytest.raises(Exception) as ei:
        np.asarray(m(jnp.array([1.0 + 1e-5]), jnp.array([1.0, 2.0])))
    assert ROW_MSG in str(ei.value)
    assert phasic._is_rate_blowup(ei.value) is False


# ======================================================== guards
# ------------------------------------- guard 10: geometric closed forms
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 1.0])
def test_geometric_closed_forms(p):
    g = geom(p)
    for k in range(1, 30):
        assert abs(g.pdf(k) - p * (1 - p) ** (k - 1)) <= 1e-15
        assert abs(g.cdf(k) - (1 - (1 - p) ** k)) <= 1e-15
    np.testing.assert_allclose(g.expectation(), 1 / p, rtol=1e-14)
    np.testing.assert_allclose(g.variance(), (1 - p) / p ** 2, rtol=1e-14,
                               atol=1e-300)


# ---------------------- guard 11: fresh joint graph, elimination table
def test_fresh_joint_graph_table_unvalidated():
    gi, idx = coalescent_indexed(3)
    fresh = gi.joint_prob_graph(idx, reward_limit=2)
    t_fresh = np.asarray(fresh.joint_prob_table(), dtype=float)
    gi2, idx2 = coalescent_indexed(3)
    upd = gi2.joint_prob_graph(idx2, reward_limit=2)
    upd.update_weights([1.0, 1.0])
    t_upd = np.asarray(upd.joint_prob_table(), dtype=float)
    np.testing.assert_allclose(t_fresh, t_upd, rtol=0, atol=1e-14)


# -------------------- guards 12 and 13: cached contexts re-validate
def test_cached_pdf_context_revalidates():
    g = pgeom()
    g.update_weights([0.5])
    assert g.pdf(3) == 0.125
    g.update_weights([1.5])
    with pytest.raises(Exception, match=ROW_MSG):
        g.pdf(3)


def test_accumulated_visits_revalidates():
    g = pgeom()
    g.update_weights([0.5])
    g.accumulated_visits(3)
    g.update_weights([1.5])
    with pytest.raises(Exception, match=ROW_MSG):
        g.accumulated_visits(3)


# ------------------------ guard 14: rows <= 0.6 against the matrix form
def test_safe_native_discrete_matches_matrix_form():
    g = safe_discrete()
    n = g.vertices_length()
    P = np.zeros((n, n))
    for i in range(n):
        for e in g.vertex_at(i).edges():
            P[i, e.to().index()] = e.weight()
    absorbing = [i for i in range(n) if not g.vertex_at(i).edges()]
    transient = [i for i in range(1, n) if i not in absorbing]
    alpha = P[0, transient]
    T = P[np.ix_(transient, transient)]
    # The remainder of each transient row stays put.
    T = T + np.diag(1.0 - P[transient].sum(axis=1))
    exit_ = P[np.ix_(transient, absorbing)].sum(axis=1)
    state = alpha.copy()
    for k in range(1, 201):
        assert abs(g.pdf(k) - state @ exit_) <= 1e-15
        state = state @ T


# ------------------------- guard 15: continuous closed forms still hold
def test_continuous_closed_forms():
    g = Graph(1)
    s = g.starting_vertex()
    w = g.find_or_create_vertex([1])
    z = g.find_or_create_vertex([0])
    s.add_edge(w, 1.0)
    w.add_edge(z, 2.0)
    e = Graph(1)
    s2 = e.starting_vertex()
    a = e.find_or_create_vertex([1])
    b = e.find_or_create_vertex([2])
    z2 = e.find_or_create_vertex([0])
    s2.add_edge(a, 1.0)
    a.add_edge(b, 2.0)
    b.add_edge(z2, 2.0)
    for t in (0.1, 0.5, 1.0, 1.5):
        np.testing.assert_allclose(g.pdf(t), 2 * np.exp(-2 * t), rtol=3e-3)
        np.testing.assert_allclose(g.cdf(t), 1 - np.exp(-2 * t), atol=1e-3)
    # Erlang from t = 0.25: near 0 its pdf rises steeply and the default
    # uniformization is 0.9% off at t = 0.1 (same on the unfixed build).
    for t in (0.25, 0.5, 1.0, 1.5):
        np.testing.assert_allclose(e.pdf(t), 4 * t * np.exp(-2 * t), rtol=3e-3)
        np.testing.assert_allclose(
            e.cdf(t), 1 - np.exp(-2 * t) * (1 + 2 * t), atol=1e-3)
