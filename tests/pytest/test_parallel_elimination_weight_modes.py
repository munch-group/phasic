"""parallel_elimination must not change expectation() for any weight mode.

The SCC composer route in ``ptd_expected_waiting_time`` used to re-derive edge
weights as the linear sum c·theta, on both the parent graph and every
synthetic SCC graph. That silently replaced log / callback / formula weights
with linear ones (wrong expectations on cyclic graphs), and when the synth
re-weighting failed (formula with more coefficients than theta) it left a
stale thread-local ``ptd_err`` that made the NEXT ``update_weights()`` raise
"Coefficient length mismatch". The route now composes at the live edge
weights, exactly what the monolithic eliminator reads.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph, cache

THETA = np.array([0.7, 3.0])


def _cyclic_graph(n_coeff=2):
    # a <-> b is a non-trivial SCC, so the composer has real work to do.
    g = Graph(1)
    g.set_param_length(2)
    s = g.starting_vertex()
    a = g.find_or_create_vertex([1])
    b = g.find_or_create_vertex([2])
    c = g.find_or_create_vertex([3])
    z = g.find_or_create_vertex([0])
    pad = [0.0] * (n_coeff - 2)
    s.add_edge(a, 1.0)
    a.add_edge(b, [2.0, 1.0] + pad)
    b.add_edge(a, [1.0, 3.0] + pad)
    a.add_edge(c, [1.0, 2.0] + pad)
    b.add_edge(c, [0.5, 1.0] + pad)
    c.add_edge(z, [1.0, 1.0] + pad)
    return g


def _callback(t, c):
    return float(np.sum(np.asarray(c) ** 2 * np.asarray(t)))


def _set_weights(g, mode):
    if mode == 'linear':
        g.update_weights(THETA)
    elif mode == 'log':
        g.update_weights(THETA, log=True)
    elif mode == 'callback':
        g.update_weights(THETA, callback=_callback)
    elif mode == 'formula':
        g.weight_formula = "c0*t0*t1 + c1"
        g.update_weights(THETA)


def _expectation(mode, parallel):
    g = _cyclic_graph()
    _set_weights(g, mode)
    with phasic.configure(parallel_elimination=parallel):
        cache.reset_scc_compose_stats()
        e = g.expectation()
        calls = cache.scc_compose_stats()['compose_calls']
    return e, calls


@pytest.mark.parametrize('mode', ['linear', 'log', 'callback', 'formula'])
def test_parallel_matches_monolithic(mode):
    mono, mono_calls = _expectation(mode, parallel=False)
    par, par_calls = _expectation(mode, parallel=True)
    assert mono_calls == 0
    assert par_calls == 1, "composer route was not taken"
    assert par == pytest.approx(mono, rel=1e-12)


def _moment_quantities(mode, parallel):
    # Every moment routes through ptd_expected_waiting_time; only its
    # no-reward call composes, but its FULL per-vertex vector seeds every
    # higher moment, so all of these depend on the composer being right.
    g = _cyclic_graph()
    _set_weights(g, mode)
    n = g.vertices_length()
    r1 = np.arange(1.0, n + 1)
    r2 = np.linspace(2.0, 0.5, n)
    with phasic.configure(parallel_elimination=parallel):
        cache.reset_scc_compose_stats()
        q = {
            'expected_waiting_time': np.array(g.expected_waiting_time()),
            'moments': np.array(g.moments(4)),
            'moments_rewards': np.array(g.moments(3, rewards=r1)),
            'variance': g.variance(),
            'variance_rewards': g.variance(rewards=r1),
            'covariance': g.covariance(r1, r2),
        }
        calls = cache.scc_compose_stats()['compose_calls']
    return q, calls


@pytest.mark.parametrize('mode', ['linear', 'log', 'callback', 'formula'])
def test_parallel_higher_moments_match_monolithic(mode):
    mono, mono_calls = _moment_quantities(mode, parallel=False)
    par, par_calls = _moment_quantities(mode, parallel=True)
    assert mono_calls == 0
    assert par_calls > 0, "composer route was not taken"
    for key in mono:
        np.testing.assert_allclose(par[key], mono[key], rtol=1e-12, err_msg=key)


def test_weight_modes_give_distinct_expectations():
    # Guards the test itself: if the modes coincided on this fixture, a
    # composer that ignored the weight mode would still pass the test above.
    values = [_expectation(m, parallel=False)[0]
              for m in ('linear', 'log', 'callback', 'formula')]
    assert len({round(v, 9) for v in values}) == 4


def test_update_weights_after_parallel_expectation_formula():
    # The reported symptom: formula with more coefficients than theta; the
    # assigned formula must keep working after expectation().
    g = _cyclic_graph(n_coeff=5)
    g.weight_formula = "c0*t0*t1 + c1"
    g.update_weights(THETA)
    with phasic.configure(parallel_elimination=False):
        mono = g.expectation()
    with phasic.configure(parallel_elimination=True):
        par = g.expectation()
        g.update_weights(THETA)        # raised before the fix
        assert g.expectation() == pytest.approx(mono, rel=1e-12)
    assert par == pytest.approx(mono, rel=1e-12)
