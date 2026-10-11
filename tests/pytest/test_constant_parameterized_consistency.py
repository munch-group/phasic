"""Constant vs parameterized models: consistent user-facing behaviour.

Plan: constant-parameterized-consistency-plan.md.

Part 2 (this section): every entry point that takes theta refuses a
constant model with one standard message. A constant model is one built
with scalar rates (add_edge(v, 3.0), or a callback returning floats), or a
derived graph with baked-in weights (reward_transform() output). Before
this, several entry points (svgd, mcmc, method_of_moments, joint_prob_graph
...) silently treated it as a 1-parameter model scaling every rate.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph, with_ipv

jnp = pytest.importorskip("jax.numpy")

GUARD_MSG = "must have parameterized edges"
N = 4


def _coalescent(state, *, as_list):
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
            r = state[i] * (state[j] - same) / (1 + same)
            transitions.append([new, [r] if as_list else r])
    return transitions


@with_ipv([N] + [0] * (N - 1))
def _coal_constant(state):
    return _coalescent(state, as_list=False)


@with_ipv([N] + [0] * (N - 1))
def _coal_param(state):
    return _coalescent(state, as_list=True)


def constant_graph():
    g = Graph(_coal_constant, graph_cache=False)
    assert not g.parameterized()
    return g


def param_graph():
    g = Graph(_coal_param, graph_cache=False)
    assert g.parameterized()
    return g


DATA = np.array([0.3, 0.5, 0.9, 1.2, 0.4])


def _rewards(g):
    r = np.zeros(g.vertices_length())
    r[2] = 1.0
    return r


# entry point -> call(graph). Each must raise the guard error on a constant
# graph BEFORE doing any work.
SITES = {
    "svgd": lambda g: g.svgd(DATA, n_particles=2, n_iterations=1, theta_dim=1,
                             verbose=False),
    "mcmc": lambda g: g.mcmc(DATA, n_samples=2, n_chains=1, burn_in=1,
                             theta_dim=1),
    "Graph.method_of_moments": lambda g: g.method_of_moments(DATA, theta_dim=1),
    "method_of_moments module": lambda g: __import__(
        "phasic.method_of_moments", fromlist=["x"]).method_of_moments(
            g, DATA, theta_dim=1),
    "Graph.probability_matching": lambda g: g.probability_matching(DATA),
    "probability_matching module": lambda g: __import__(
        "phasic.probability_matching", fromlist=["x"]).probability_matching(
            g, DATA),
    "epoch_model": lambda g: g.epoch_model(DATA, [0.0]),
    "daisy_chain_joint_probs": lambda g: g.daisy_chain_joint_probs(
        epoch_thetas=np.ones((1, 1)), epoch_dts=[1.0], initial_ipv=[1.0]),
    "pmf_and_moments_from_graph": lambda g: Graph.pmf_and_moments_from_graph(g),
    "moments_from_graph": lambda g: Graph.moments_from_graph(g),
    "pmf_and_moments_from_graph_multivariate":
        lambda g: Graph.pmf_and_moments_from_graph_multivariate(g),
    "pmf_from_graph_joint_index": lambda g: Graph.pmf_from_graph_joint_index(g),
    "add_epoch": lambda g: g.add_epoch(1.0),
    "joint_prob_graph": lambda g: g.joint_prob_graph(reward_limit=1),
    "joint_stop_prob_graph": lambda g: g.joint_stop_prob_graph(),
    "joint_sojourn_graph": lambda g: g.joint_sojourn_graph(),
    "reward_visit_probability(theta=jax)": lambda g: g.reward_visit_probability(
        _rewards(g), theta=jnp.asarray([1.0])),
    "reward_visit_probability(theta=numpy)": lambda g: g.reward_visit_probability(
        _rewards(g), theta=np.asarray([1.0])),
    "profile(theta)": lambda g: g.profile(theta=[1.0], probe_dyn=False),
    "profile_graph(theta)": lambda g: __import__(
        "phasic.profile", fromlist=["x"]).profile_graph(
            g, theta=[1.0], probe_dyn=False),
    "bffg.path_exit_rates_by_param": lambda g: __import__(
        "phasic.bffg", fromlist=["x"]).path_exit_rates_by_param(g, {}),
    "bffg.bffg_log_prob": lambda g: __import__(
        "phasic.bffg", fromlist=["x"]).bffg_log_prob(
            g, g, np.ones(1), lambda t: t, DATA),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_constant_graph_refused(site):
    with pytest.raises(ValueError, match=GUARD_MSG) as exc:
        SITES[site](constant_graph())
    assert "needs a parameterized model" in str(exc.value)


@pytest.mark.parametrize("site", sorted(SITES))
def test_parameterized_graph_passes_the_guard(site):
    """The parameterized twin may fail for other reasons (e.g. "requires a
    joint-probability graph"), but never at the guard."""
    try:
        SITES[site](param_graph())
    except Exception as e:  # noqa: BLE001 - only the guard message matters
        assert "needs a parameterized model" not in str(e), site


def test_reward_transform_output_is_refused():
    p = param_graph()
    p.update_weights([1.0])
    rt = p.reward_transform(np.ones(p.vertices_length()))
    assert not rt.parameterized()
    with pytest.raises(ValueError, match=GUARD_MSG):
        rt.svgd(DATA, n_particles=2, n_iterations=1, theta_dim=1, verbose=False)


def test_constant_graph_without_theta_still_works():
    """Entry points used WITHOUT theta keep working on constant graphs."""
    g = constant_graph()
    p = g.reward_visit_probability(_rewards(g))
    assert 0.0 <= float(p) <= 1.0
    prof = g.profile(probe_dyn=False)     # _max_rate used to update_weights
    assert prof is not None
    np.testing.assert_allclose(g.expectation(), 1.5, rtol=1e-12)



# =========================================================== Part 3
# Lossless layout widening (_rebuild_with_wider_layout and its callers
# discretize / add_epoch / joint_prob_graph).
from itertools import combinations_with_replacement

from phasic import StateIndexer, Property


def _chain_with_constant_middle(constant=True):
    """start -> a; a -> b [2.0], a -> c (constant 0.5 or [0.5]), a -> d [1.0];
    b, c, d absorbing. With constant=False the middle edge is a
    parameterized [0.5], which equals the constant at theta0 = 1."""
    g = Graph(1)
    s = g.starting_vertex()
    a, b, c, d = (g.find_or_create_vertex([k]) for k in (10, 1, 2, 3))
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0])
    if constant:
        a.add_edge_constant(c, 0.5)
    else:
        a.add_edge(c, [0.5])
    a.add_edge(d, [1.0])
    return g


def _n_constant_edges(g):
    return sum(1 for v in g.vertices() for e in v.edges()
               if e.coefficients_length() == 0)


def test_discretize_refuses_constant_edge_on_parameterized_graph():
    """A discretized graph renormalises every edge per theta; a constant
    edge next to the new theta-dependent aux edge would drift between
    update_weights calls, so discretize() refuses it."""
    g = _chain_with_constant_middle(constant=True)
    with pytest.raises(ValueError, match="coefficient-less constant"):
        g.discretize(0.1)
    with pytest.raises(ValueError, match="coefficient-less constant"):
        g.discretize(lambda state: 0.1)


def test_add_epoch_keeps_constant_edge_in_place():
    """add_epoch output is continuous (no renormalisation), so a constant
    edge is carried losslessly, in its original position."""
    g = param_graph()
    v = g.vertex_at(2)
    absorbing = [u for u in g.vertices() if not u.edges()
                 and u.index() != g.starting_vertex().index()][0]
    v.add_edge_constant(absorbing, 0.25)
    kinds = [e.coefficients_length() == 0 for e in v.edges()]
    g.update_weights([1.0])
    e = g.add_epoch(1.0)
    ev = e.vertex_at(2)
    assert [ed.coefficients_length() == 0 for ed in ev.edges()][:len(kinds)] == kinds
    assert (absorbing.index(), 0.25) in [(ed.to().index(), ed.weight())
                                          for ed in ev.edges()
                                          if ed.coefficients_length() == 0]


def _coal_indexed(nr):
    indexer = StateIndexer(lineages=[Property('descendants', min_value=1, max_value=nr)])
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = nr

    @with_ipv(ipv)
    def coal(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            out.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return out
    return Graph(coal, indexer=indexer, graph_cache=False), indexer


def test_add_epoch_then_discretize_keeps_epoch_metadata_and_indexer():
    g, _ = _coal_indexed(4)
    g.update_weights([1.0])
    e = g.add_epoch(1.0)
    d = e.discretize(0.1)
    for attr in ('_n_epochs', '_epoch_state_index', '_base_param_length'):
        assert getattr(d, attr) == getattr(e, attr), attr
    assert d._indexer is e._indexer


def test_discretize_skip_existing_adds_nothing():
    d = param_graph().discretize(0.1)
    d2 = d.discretize(0.1, skip_existing=True)
    assert d2.vertices_length() == d.vertices_length()
    assert d2.param_length() == d.param_length()
    assert np.array_equal(d2.rewards, d.rewards)
    d.update_weights([1.0, 1.0])
    d2.update_weights([1.0, 1.0])
    for n in (1, 2, 4):
        np.testing.assert_allclose(d2.pdf(n), d.pdf(n), rtol=1e-12)


def test_widening_joint_prob_graph_keeps_duplicate_zero_states():
    g, indexer = _coal_indexed(3)
    jg = g.joint_prob_graph(indexer, reward_limit=1, mutation_rate=1.0)
    w = jg._rebuild_with_wider_layout(extra_coeff_slots=1)
    assert w.vertices_length() == jg.vertices_length()
    assert _n_constant_edges(w) == _n_constant_edges(jg)


def test_widening_keeps_ipv_and_is_aux():
    d = param_graph().discretize(0.1)
    w = d._rebuild_with_wider_layout(extra_coeff_slots=1)
    sv, wsv = d.starting_vertex(), w.starting_vertex()
    assert [(e.to().index(), e.weight()) for e in sv.edges()] == \
           [(e.to().index(), e.weight()) for e in wsv.edges()]
    assert [v.is_aux() for v in d.vertices()] == [v.is_aux() for v in w.vertices()]
    assert w.get_was_dph() and np.array_equal(w.rewards, d.rewards)


def _old_rebuild_coefficients(graph, extra):
    """Reference: the coefficient layout the pre-fix helper produced, for a
    graph with unique states and no coefficient-less edges."""
    P = graph.param_length()
    out = []
    for v in graph.vertices():
        for e in v.parameterized_edges():
            out.append((v.index(), e.to().index(),
                        tuple(list(e.edge_state(P)) + [0.0] * extra)))
    return out


def _coefficients(graph):
    P = graph.param_length()
    return [(v.index(), e.to().index(), tuple(e.edge_state(P)))
            for v in graph.vertices() for e in v.parameterized_edges()]


def test_bit_identity_for_ordinary_graph():
    """Control: unique states, no constant edges -> exactly today's layout."""
    g = param_graph()
    w = g._rebuild_with_wider_layout(extra_coeff_slots=3)
    assert _coefficients(w) == _old_rebuild_coefficients(g, 3)
    assert [list(v.state()) for v in w.vertices()] == [list(v.state()) for v in g.vertices()]


def test_ragged_coefficients_refused():
    g = Graph(1)
    s = g.starting_vertex()
    a, b = g.find_or_create_vertex([2]), g.find_or_create_vertex([1])
    g.set_param_length(1)
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0, 7.0])
    with pytest.raises(ValueError, match="coefficient vector to have"):
        g.discretize(0.1)


def _set_mode(g, mode):
    if mode == 'formula':
        g.weight_formula = "t0*c0"
    elif mode == 'callback':
        g.weight_callback = lambda theta, coeffs: float(theta[0] * coeffs[0])
    else:
        g.weight_mode = mode
    assert g.weight_mode == mode
    return g


@pytest.mark.parametrize("mode", ['formula', 'callback', 'log'])
def test_widening_refuses_non_linear_weight_modes(mode):
    with pytest.raises(ValueError, match="weight_mode='linear'"):
        _set_mode(param_graph(), mode).discretize(0.1)
    with pytest.raises(ValueError, match="weight_mode='linear'"):
        _set_mode(param_graph(), mode).add_epoch(1.0)
    gi, indexer = _coal_indexed(3)
    with pytest.raises(ValueError, match="weight_mode='linear'"):
        _set_mode(gi, mode).joint_prob_graph(indexer, reward_limit=1)


def test_add_epoch_refuses_discrete_graph():
    d = param_graph().discretize(0.1)
    with pytest.raises(NotImplementedError, match="discrete graph"):
        d.add_epoch(1.0)


def test_joint_prob_graph_refuses_discretized_graph():
    g, indexer = _coal_indexed(3)
    d = g.discretize(0.1)
    with pytest.raises(ValueError, match="discretized graph"):
        d.joint_prob_graph(indexer, reward_limit=1)


def test_joint_prob_graph_refuses_constant_base_edge():
    g, indexer = _coal_indexed(3)
    src = g.starting_vertex().edges()[0].to()
    dst = [v for v in g.vertices() if v.index() != src.index() and not v.edges()
           and v.index() != g.starting_vertex().index()][0]
    src.add_edge_constant(dst, 0.25)
    with pytest.raises(ValueError, match="coefficient-less constant"):
        g.joint_prob_graph(indexer, reward_limit=1, mutation_rate=1.0)


# =========================================================== Part 4
# discretize(rate=callable): the callable returns ONE number for any graph;
# on a parameterized graph it goes into the widened slot, exactly as a
# fixed rate does.

def _hand_discretized(graph, rate_of_state):
    """Independent oracle: E[number of aux visits] for a CONSTANT graph,
    via the closed form of the discretized chain (aux rate r(s) at each
    transient s): E[N] = sum over transient s of r(s) * E[time in s]."""
    sojourn = np.asarray(graph.expected_sojourn_time())
    total = 0.0
    for v in graph.vertices():
        if v.index() == graph.starting_vertex().index() or not v.edges():
            continue
        total += rate_of_state(v.state()) * sojourn[v.index()]
    return total


def test_scalar_callback_on_parameterized_graph_equals_fixed_rate():
    d_fixed = param_graph().discretize(0.1)
    d_cb = param_graph().discretize(lambda state: 0.1)
    assert d_cb.param_length() == d_fixed.param_length() == 2
    for d in (d_fixed, d_cb):
        d.update_weights([1.0, 1.0])
    np.testing.assert_allclose(d_cb.expectation_discrete(d_cb.rewards),
                               d_fixed.expectation_discrete(d_fixed.rewards),
                               rtol=1e-14)


@pytest.mark.parametrize("mu", [0.3, 1.0])
def test_state_dependent_callback_matches_constant_twin(mu):
    """nr_lineages * mu (the discrete.ipynb cell 17 callback): the
    parameterized graph at theta = [1, 1] equals the constant graph, and
    both match the closed-form oracle."""
    def mutation(state):
        return sum(state) * mu
    dp = param_graph().discretize(mutation)
    dc = constant_graph().discretize(mutation)
    dp.update_weights([1.0, 1.0])
    ep = dp.expectation_discrete(dp.rewards)
    ec = dc.expectation_discrete(dc.rewards)
    np.testing.assert_allclose(ep, ec, rtol=1e-12)
    oracle = _hand_discretized(constant_graph(), lambda s: sum(s) * mu)
    np.testing.assert_allclose(ec, oracle, rtol=1e-12)


def test_theta_last_scales_the_callback_rate():
    """Effective rate = rate(state) * theta[-1]: callback mu at theta[-1]=2
    equals callback 2*mu at theta[-1]=1."""
    a = param_graph().discretize(lambda s: 0.3 * sum(s))
    b = param_graph().discretize(lambda s: 0.6 * sum(s))
    a.update_weights([1.0, 2.0])
    b.update_weights([1.0, 1.0])
    np.testing.assert_allclose(a.expectation_discrete(a.rewards),
                               b.expectation_discrete(b.rewards), rtol=1e-12)


@pytest.mark.parametrize("make", [param_graph, constant_graph])
def test_callback_zero_rate_for_some_states(make):
    """Singleton-only mutation 0.3 * s[0]: 0 at states without singletons.
    Reviewer-measured value E = 0.6 for this 4-sample coalescent."""
    d = make().discretize(lambda s: 0.3 * s[0])
    if d.parameterized():
        d.update_weights([1.0, 1.0])
    np.testing.assert_allclose(d.expectation_discrete(d.rewards), 0.6, rtol=1e-10)


@pytest.mark.parametrize("ret", [np.float64(0.2), np.array(0.2), np.int64(1), 1,
                                 __import__('fractions').Fraction(1, 5)])
def test_callback_numpy_scalars_accepted(ret):
    """Any real scalar works and gives the same graph as the fixed rate."""
    d = param_graph().discretize(lambda s: ret)
    ref = param_graph().discretize(float(ret))
    assert d.param_length() == ref.param_length() == 2
    d.update_weights([1.0, 1.0])
    ref.update_weights([1.0, 1.0])
    np.testing.assert_allclose(d.expectation_discrete(d.rewards),
                               ref.expectation_discrete(ref.rewards), rtol=1e-14)


@pytest.mark.parametrize("rate,exc", [
    (True, TypeError), (float('nan'), ValueError), (float('inf'), ValueError),
    (0.0, ValueError), (-1.0, ValueError)])
def test_fixed_rate_validation(rate, exc):
    with pytest.raises(exc, match="rate must be"):
        param_graph().discretize(rate)


def test_fixed_rate_with_kwargs_raises():
    with pytest.raises(TypeError, match="keyword arguments"):
        param_graph().discretize(0.1, mu=0.3)


def test_callback_receives_kwargs():
    d = param_graph().discretize(lambda s, mu: mu * sum(s), mu=0.3)
    ref = param_graph().discretize(lambda s: 0.3 * sum(s))
    d.update_weights([1.0, 1.0])
    ref.update_weights([1.0, 1.0])
    np.testing.assert_allclose(d.expectation_discrete(d.rewards),
                               ref.expectation_discrete(ref.rewards), rtol=1e-14)


@pytest.mark.parametrize("make", [param_graph, constant_graph])
@pytest.mark.parametrize("ret,exc", [
    ([0.1, 0.2], TypeError), (np.array([0.1]), TypeError), ("0.1", TypeError),
    (True, TypeError), (-0.1, ValueError), (float('nan'), ValueError),
    (float('inf'), ValueError)])
def test_callback_bad_returns_raise(make, ret, exc):
    with pytest.raises(exc, match="discretize"):
        make().discretize(lambda s: ret)


def test_callback_on_constant_graph_unchanged():
    d = constant_graph().discretize(lambda s: 0.1)
    assert not d.parameterized()
    assert d.param_length() == constant_graph().param_length()
