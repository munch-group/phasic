"""Discrete (DPH) correctness on the parameterized path: is_discrete propagation,
discrete reward transform, discrete moments, reward-length validation, and the
variance_discrete fix.

Background (pre-existing bugs fixed on branch fix/is-discrete-propagation):
  * is_discrete was a Python-only attribute that serialize() dropped, so the C++
    GraphBuilder (the whole parameterized/JAX path) was blind to discreteness and
    applied the CONTINUOUS reward transform and CONTINUOUS moments to a DPH.
  * variance_discrete() returned m[1]-2*m[0] (wrong) instead of m[1]-m[0]-m[0]^2.

Every assertion is anchored to an INDEPENDENT oracle: closed-form dph_pmf
summation, or reward_transform_discrete, never the code path under test.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")


# --------------------------------------------------------------------------- fixtures
def dph(probs=(1.0, 1.0), set_discrete=False):
    """Native DPH chain start -> [k] -> ... -> [1]; edge j carries coeff on theta[j],
    so weight = theta[j] is the per-step transition probability (deficit = self-loop)."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    n = len(probs)
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        coeff = [0.0] * n
        coeff[i] = 1.0
        vs[i].add_edge(vs[i + 1], coeff)
    if set_discrete:
        # is_discrete only (NOT was_dph): a native DPH already carries per-step
        # probabilities, so the was_dph auto-normalisation in update_weights would
        # rescale each vertex's out-edges to sum to 1 and collapse it to a
        # deterministic walk. (That native-DPH vs discretize() inconsistency is
        # tracked separately.)
        g.is_discrete = True
    return g


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def true_discrete_moments(probs, theta, K, upper=20000):
    """[E[N], ..., E[N^K]] by direct dph_pmf summation (independent oracle)."""
    g = dph(probs)
    g.update_weights(list(theta))
    ns = np.arange(1, upper)
    pn = np.array([g.pdf_discrete(int(n)) for n in ns])
    assert abs(pn.sum() - 1.0) < 1e-9, f"pmf mass {pn.sum()} (raise upper)"
    return [float((ns ** k * pn).sum()) for k in range(1, K + 1)]


CASES = [((1.0, 1.0), (0.3, 0.3)), ((1.0, 1.0), (0.5, 0.2)), ((1.0, 1.0, 1.0), (0.4, 0.3, 0.5))]


# --------------------------------------------------------------------------- variance_discrete
@pytest.mark.parametrize("probs,theta", CASES)
def test_variance_discrete_matches_summation(probs, theta):
    g = dph(probs)
    g.update_weights(list(theta))
    tm = true_discrete_moments(probs, theta, 2)
    true_var = tm[1] - tm[0] ** 2
    assert _rel(g.variance_discrete(), true_var) < 1e-9


# --------------------------------------------------------------------------- N1: discrete reward transform
def test_discrete_rewards_use_discrete_transform_not_continuous():
    """pmf_and_moments_from_graph(discrete=True, rewards) must match the DISCRETE
    reward transform, not the continuous one."""
    theta = np.array([0.3, 0.3])
    jumps = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    irew = np.array([1, 2, 1, 0])
    g = dph((1.0, 1.0)); g.update_weights(theta.tolist())
    ref_disc = np.array([g.reward_transform_discrete(irew).pdf_discrete(int(j)) for j in jumps])
    g2 = dph((1.0, 1.0)); g2.update_weights(theta.tolist())
    ref_cont = np.array([g2.reward_transform(irew.astype(float)).pdf_discrete(int(j)) for j in jumps])
    assert _rel(ref_disc, ref_cont) > 0.1, "oracle sanity: the two transforms must differ"

    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=True)
    pmf = np.asarray(m(jnp.asarray(theta), jnp.asarray(jumps), jnp.asarray(irew, float))[0])
    assert _rel(pmf, ref_disc) < 1e-9
    assert _rel(pmf, ref_cont) > 0.1


def test_non_integer_discrete_rewards_raise():
    """Discrete reward transform requires integer rewards; a fractional one must raise."""
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=True)
    with pytest.raises(Exception):
        m(jnp.asarray([0.3, 0.3]), jnp.asarray([2.0, 3.0]), jnp.asarray([0.5, 1.2, 1.0, 0.0]))


# --------------------------------------------------------------------------- is_discrete propagation
def test_is_discrete_serialized():
    ser = dph((1.0, 1.0), set_discrete=True).serialize(theta_dim=2)
    assert ser.get("is_discrete") is True
    assert dph((1.0, 1.0)).serialize(theta_dim=2).get("is_discrete") is False


def test_is_discrete_propagates_without_per_call_flag():
    """A graph flagged is_discrete produces the discrete answer even when the caller
    passes discrete=False -- discreteness travels with the graph via serialize()."""
    theta = np.array([0.3, 0.3])
    tm = true_discrete_moments((1.0, 1.0), theta, 2)
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0), set_discrete=True), nr_moments=2, discrete=False)
    _, mom = m(jnp.asarray(theta), jnp.asarray([2.0, 3.0]))
    assert _rel(np.asarray(mom), tm) < 1e-6


# --------------------------------------------------------------------------- N2: discrete moments
@pytest.mark.parametrize("probs,theta", CASES)
def test_discrete_moments_match_summation(probs, theta):
    tm = true_discrete_moments(probs, theta, 3)
    m = Graph.pmf_and_moments_from_graph(dph(probs), nr_moments=3, discrete=True)
    _, mom = m(jnp.asarray(theta), jnp.asarray([2.0, 3.0, 4.0]))
    assert _rel(np.asarray(mom), tm) < 1e-6


@pytest.mark.parametrize("probs,theta", CASES)
def test_graph_moments_discrete_matches_summation(probs, theta):
    """The Graph.moments(discrete=True) instance method (was unbound) gives discrete raw
    moments matching the dph_pmf summation, orders 1-3."""
    g = dph(probs, set_discrete=True)
    g.update_weights(list(theta))
    tm = true_discrete_moments(probs, theta, 3)
    got = np.asarray(g.moments(3, discrete=True))
    assert _rel(got, tm) < 1e-6


def test_moments_discrete_requires_is_discrete():
    g = dph((1.0, 1.0))  # not flagged discrete
    g.update_weights([0.3, 0.3])
    with pytest.raises(ValueError):
        g.moments(2, discrete=True)


def test_discrete_reward_moments_match_transformed_summation():
    """Discrete moments WITH rewards == moments of the reward_transform_discrete graph."""
    theta = np.array([0.3, 0.3])
    irew = np.array([1, 2, 1, 0])
    g = dph((1.0, 1.0)); g.update_weights(theta.tolist())
    tg = g.reward_transform_discrete(irew)
    ns = np.arange(1, 20000)
    pn = np.array([tg.pdf_discrete(int(n)) for n in ns])
    tm = [float((ns ** k * pn).sum()) for k in (1, 2)]
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=True)
    _, mom = m(jnp.asarray(theta), jnp.asarray([3.0, 4.0]), jnp.asarray(irew, float))
    assert _rel(np.asarray(mom), tm) < 1e-5


def test_discrete_moments_ffi_matches_pybind():
    theta = np.array([0.3, 0.3])
    mp = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=3, discrete=True, use_ffi=False)
    mf = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=3, discrete=True, use_ffi=True)
    _, momp = mp(jnp.asarray(theta), jnp.asarray([2.0, 3.0, 4.0]))
    _, momf = mf(jnp.asarray(theta), jnp.asarray([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(momf), np.asarray(momp), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- regression: continuous unchanged
def test_continuous_moments_unchanged():
    g = dph((1.0, 1.0)); g.update_weights([2.0, 3.0])
    cont = np.asarray(g.moments(2))  # [E[T], E[T^2]]
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=False)
    _, mom = m(jnp.asarray([2.0, 3.0]), jnp.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(mom), cont, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- B2: was_dph round-trip
def _erlang():
    """2-stage parameterized Erlang (continuous) to feed discretize()."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([2])
    b = g.find_or_create_vertex([1])
    s.add_edge(a, 1.0)
    a.add_edge(b, [1.0])  # rate = theta[0]
    return g


def _chain(n):
    """n-stage parameterized chain start -> [n+1] -> ... -> [1] (continuous), to
    feed discretize(). discretize() adds ONE aux vertex per transient vertex, all
    with the all-zero state -- so n>=2 exposes any from_serialized vertex-merge."""
    g = phasic.Graph(1)
    s = g.starting_vertex()
    vs = [g.find_or_create_vertex([n + 1 - i]) for i in range(n + 1)]
    s.add_edge(vs[0], 1.0)
    for i in range(n):
        vs[i].add_edge(vs[i + 1], [1.0 if j == i else 0.0 for j in range(n)])
    return g


def test_was_dph_serialized():
    """serialize() carries was_dph: a discretize()'d graph has it; a native DPH
    (is_discrete only) does not."""
    d = _erlang().discretize(0.5)
    assert d.serialize().get("was_dph") is True
    g = dph((1.0, 1.0), set_discrete=True)  # is_discrete only, no was_dph
    assert g.serialize(theta_dim=2).get("was_dph") is False


def test_discretized_roundtrip_preserves_was_dph():
    """A discretize()'d graph round-trips with was_dph=True, so update_weights
    auto-normalises and it stays a VALID DPH (mass 1). If was_dph were dropped to
    False, update_weights would leave outgoing rate > 1 and pdf_discrete would raise."""
    d = _erlang().discretize(0.5)
    d2 = Graph.from_serialized(d.serialize())
    assert d2.get_was_dph() is True
    d2.update_weights([1.0] * d2.param_length())
    p = np.array([d2.pdf_discrete(n) for n in range(1, 3000)])
    assert abs(p.sum() - 1.0) < 1e-6


def test_native_dph_roundtrip_stays_unnormalised():
    """A native DPH (is_discrete, was_dph NOT set) round-trips WITHOUT was_dph, so
    update_weights does not renormalise and collapse it to a deterministic walk."""
    g = dph((1.0, 1.0), set_discrete=True)
    assert g.get_was_dph() is False
    ser = g.serialize(theta_dim=2)
    assert ser.get("was_dph") is False
    g2 = Graph.from_serialized(ser)
    assert g2.is_discrete and g2.get_was_dph() is False
    g2.update_weights([0.5, 0.2])
    tm = true_discrete_moments((1.0, 1.0), (0.5, 0.2), 2)
    got = np.asarray(g2.moments(2, discrete=True))
    assert _rel(got, tm) < 1e-6, f"native DPH collapsed on round-trip: {got} vs {tm}"


def test_pre_b2_cache_discretized_stays_valid():
    """Cross-version: a pre-B2 serialized payload (no 'was_dph' key) for a discretized
    graph must reload as a valid DPH (was_dph defaults to is_discrete, not False)."""
    d = _erlang().discretize(0.5)
    d.update_weights([1.0] * d.param_length())
    ser = dict(d.serialize())
    ser.pop("was_dph", None)  # simulate a pre-B2 cache: key absent
    assert ser.get("is_discrete") is True
    d2 = Graph.from_serialized(ser)
    assert d2.get_was_dph() is True  # absent-key default = is_discrete
    d2.update_weights([1.0] * d2.param_length())
    d2.pdf_discrete(2)  # must not raise


# --------------------------------------------------------------------------- native vs parameterized/FFI for a discretize()'d (was_dph) graph
def _discretized_native_moments(theta, K, upper=20000):
    """Independent oracle: raw discrete moments E[N^k] of _erlang().discretize(0.5)
    at `theta`, via pdf_discrete summation on the NATIVE pybind path (which
    respects the was_dph auto-normalisation)."""
    d = _erlang().discretize(0.5)
    d.update_weights(list(theta))
    ns = np.arange(1, upper)
    pn = np.array([d.pdf_discrete(int(n)) for n in ns])
    assert abs(pn.sum() - 1.0) < 1e-9, f"pmf mass {pn.sum()} (raise upper)"
    return [float((ns ** k * pn).sum()) for k in range(1, K + 1)]


@pytest.mark.parametrize("use_ffi", [False, True])
def test_parameterized_discretized_matches_native(use_ffi):
    """was_dph must propagate to the parameterized/FFI backend: a discretize()'d
    parameterized graph gives the SAME discrete moments there as on the native
    pybind path. Before the fix GraphBuilder dropped was_dph, skipping the
    per-theta renormalisation -> E[N]=4.0 (param) vs 2.333 (native)."""
    theta = [0.3, 0.4]
    native = _discretized_native_moments(theta, 2)
    m = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True, use_ffi=use_ffi)
    _, mom = m(jnp.asarray(theta), jnp.asarray([2.0, 3.0, 4.0, 5.0]))
    assert _rel(np.asarray(mom), native) < 1e-6, (
        f"param(use_ffi={use_ffi}) {np.asarray(mom)} != native {native}")


def test_parameterized_discretized_no_rate_overflow():
    """At a theta that would push an un-normalised out-rate > 1, the parameterized
    path must stay a valid DPH (the was_dph renorm keeps rows <= 1) rather than
    raising 'outgoing rate <= 1'."""
    m = Graph.pmf_and_moments_from_graph(
        _erlang().discretize(0.5), nr_moments=2, discrete=True)
    _, mom = m(jnp.asarray([1.0, 1.0]), jnp.asarray([2.0, 3.0]))
    assert np.all(np.isfinite(np.asarray(mom)))


def test_native_dph_reward_transform_stays_unnormalised():
    """Fix 1b: reward-transforming a NATIVE DPH (was_dph=False) must NOT latch
    was_dph=True (which would collapse it to a deterministic walk on the next
    update_weights). A discretize() source keeps was_dph=True (covered by
    test_reward_transform_discrete_flag)."""
    g = dph((1.0, 1.0), set_discrete=True)  # native: was_dph False
    g.update_weights([0.3, 0.3])
    irew = np.array([1, 2, 1, 0])
    assert g.reward_transform_discrete(irew).get_was_dph() is False


@pytest.mark.parametrize("n_stages", [1, 3])
def test_from_serialized_discretized_roundtrip_matches_native(n_stages):
    """Batch 3: a discretize()'d parameterized graph must round-trip through
    serialize -> from_serialized reproducing the native discrete distribution.
    Two coupled bugs had to be fixed: (a) from_serialized dropped the aux
    back-edges (constant_edges) -> E[N]=1.0; (b) it merged the all-zero-state aux
    vertices via find_or_create_vertex (Q6b), so a >=2-aux graph mis-routed the
    back-edges -> a SILENT wrong distribution (mass still 1.0). n=1 has a single
    aux (can't merge); n=3 has three aux (exercises both fixes)."""
    d0 = _chain(n_stages).discretize(0.5)
    pl = d0.param_length()
    theta = [0.2, 0.5, 0.9, 0.5, 0.3, 0.7][:pl]
    d0.update_weights(theta)
    ns = np.arange(1, 40000)
    pn0 = np.array([d0.pdf_discrete(int(x)) for x in ns])
    assert abs(pn0.sum() - 1.0) < 1e-9, f"native mass {pn0.sum()}"
    EN0 = float((ns * pn0).sum())

    d2 = Graph.from_serialized(_chain(n_stages).discretize(0.5).serialize())
    assert d2.vertices_length() == d0.vertices_length(), "aux vertices merged on round-trip"
    d2.update_weights(theta)
    pn2 = np.array([d2.pdf_discrete(int(x)) for x in ns])
    assert abs(pn2.sum() - 1.0) < 1e-9, f"roundtrip mass {pn2.sum()}"
    EN2 = float((ns * pn2).sum())
    assert _rel([EN2], [EN0]) < 1e-6, f"n={n_stages}: from_serialized E[N]={EN2} vs native {EN0}"
