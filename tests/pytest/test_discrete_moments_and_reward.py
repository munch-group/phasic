"""Discrete (DPH) correctness on the parameterized path: is_discrete propagation,
discrete reward transform, discrete moments, reward-length validation, and the
variance_discrete fix.

Background (pre-existing bugs fixed on branch fix/is-discrete-propagation):
  * is_discrete was a Python-only attribute that serialize() dropped, so the C++
    GraphBuilder (the whole parameterized/JAX path) was blind to discreteness and
    applied the CONTINUOUS reward transform and CONTINUOUS moments to a DPH.
  * variance_discrete() returned m[1]-2*m[0] (wrong) instead of m[1]-m[0]-m[0]^2.
  * pmf_and_moments_from_graph took the reward length from the reward array shape
    and read out of bounds for a short reward vector.

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
        g.is_discrete = True
        g.set_was_dph(True)
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


# --------------------------------------------------------------------------- N4: reward-length validation
@pytest.mark.parametrize("badlen", [1, 3, 8])  # graph has 4 vertices
def test_wrong_length_rewards_raise_not_oob(badlen):
    """A reward vector whose length != n_vertices must raise, not read out of bounds."""
    rw = np.linspace(0.5, 2.0, badlen)
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=False)
    with pytest.raises(Exception):
        m(jnp.asarray([1.0, 2.0]), jnp.asarray([0.5, 1.0]), jnp.asarray(rw))


def test_correct_length_rewards_ok():
    rw = np.linspace(0.5, 2.0, 4)
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=False)
    pmf, _ = m(jnp.asarray([1.0, 2.0]), jnp.asarray([0.5, 1.0]), jnp.asarray(rw))
    assert np.all(np.isfinite(np.asarray(pmf)))


# --------------------------------------------------------------------------- regression: continuous unchanged
def test_continuous_moments_unchanged():
    g = dph((1.0, 1.0)); g.update_weights([2.0, 3.0])
    cont = np.asarray(g.moments(2))  # [E[T], E[T^2]]
    m = Graph.pmf_and_moments_from_graph(dph((1.0, 1.0)), nr_moments=2, discrete=False)
    _, mom = m(jnp.asarray([2.0, 3.0]), jnp.asarray([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(mom), cont, rtol=1e-12, atol=1e-12)
