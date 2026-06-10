"""Fix B1: theta_dim < coefficient length in formula mode (non-daisy path).

A graph built with Graph(..., theta_dim=k) / set_param_length(k) keeps
param_length = k (the optimized theta dimension) while its edges carry longer
coefficient vectors (per-edge data). The weight_formula references t0..t(k-1)
(theta) and c0..c(m-1) (the full coefficients). Works on the direct
update_weights/pdf path AND the FFI/svgd path.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
import phasic
from phasic import Graph, ExpStepSize, with_ipv, configure
import jax.numpy as jnp

TIMES = np.linspace(0.1, 4.0, 30)
GRAN = 128


# Module-level (hashable) callback so the graph_cache is active. Death chain
# 3->2->1 with 2-coefficient edges but theta_dim=1 (decoupled).
@with_ipv([3])
def _chain_2coeff_gc(state, indexer=None):
    n = int(state[0])
    if n <= 1:
        return []
    return [[np.array([n - 1]), [float(n), 1.0]]]   # 2 coeffs, theta_dim 1


def formula_graph():
    """param_length=2 (theta dim) but a 3-coefficient edge; rate = c0*t0+c1+c2*t1."""
    g = Graph(1)
    g.set_param_length(2)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5, 3.0])
    g.weight_formula = "c0*t0 + c1 + c2*t1"
    return g


def const_graph(w):
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, float(w))
    return g


def linear_ref():
    """1-coeff parameterized edge = weight 4.5 (for FFI same-granularity compare)."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [4.5])
    return g


def test_param_length_is_theta_dim_not_coeff_length():
    g = formula_graph()
    assert g.param_length() == 2          # theta dim, not the 3-coeff edge length


def test_live_path_theta_dim_lt_coeff_length():
    # weight = 2*0.5 + 0.5 + 3*1.0 = 4.5
    g = formula_graph()
    g.update_weights([0.5, 1.0])          # theta length 2 (== param_length)
    pdf = np.asarray(g.pdf(TIMES, granularity=GRAN))
    ref = np.asarray(const_graph(4.5).pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(pdf, ref, rtol=0, atol=1e-12)


def test_ffi_path_theta_dim_lt_coeff_length():
    # FFI must read the FULL coefficient row (not truncate to param_length) and
    # keep param_length = theta dim. Compare to a linear ref of the same weight
    # on the same (auto) granularity path.
    m_f = Graph.pmf_from_graph(formula_graph(), discrete=False)
    m_r = Graph.pmf_from_graph(linear_ref(), discrete=False)
    pdf_f = np.asarray(m_f(jnp.array([0.5, 1.0]), jnp.asarray(TIMES)))
    pdf_r = np.asarray(m_r(jnp.array([1.0]), jnp.asarray(TIMES)))
    np.testing.assert_allclose(pdf_f, pdf_r, rtol=0, atol=1e-12)


def test_default_build_still_requires_full_theta():
    # Without set_param_length, param_length locks to the 3-coeff edge length,
    # so a length-2 theta is rejected (the original blocker / user error).
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5, 3.0])
    assert g.param_length() == 3
    g.weight_formula = "c0*t0 + c1 + c2*t1"
    with pytest.raises(Exception):
        g.update_weights([0.5, 1.0])      # len 2 != param_length 3


def test_graph_cache_preserves_theta_dim():
    # The graph_cache save/load roundtrip must NOT drop theta_dim (from_serialized
    # would otherwise lock param_length to the coefficient length -> a silent
    # failure for decoupled weight_formula models built with graph_cache=True).
    phasic.clear_caches()
    configure(graph_cache=True)
    try:
        g1 = Graph(_chain_2coeff_gc, theta_dim=1)     # cache miss -> build + save
        g2 = Graph(_chain_2coeff_gc, theta_dim=1)     # cache hit  -> load
        assert g1.param_length() == 1                 # theta dim, not the 2-coeff length
        assert g2.param_length() == 1                 # preserved through the cache
    finally:
        configure(graph_cache=False)


def test_svgd_runs_with_theta_dim_lt_coeff_length():
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=1.0 / 4.5, size=120)
    g = formula_graph()
    res = g.svgd(
        data, theta_dim=2, n_iterations=10, n_particles=8, seed=1,
        fixed=[(1, 1.0)],                 # fix t1; optimize t0
        positive_params=True,
        learning_rate=ExpStepSize(first_step=0.02, last_step=0.002, tau=20.0),
    )
    parts = np.asarray(res.particles)
    assert parts.shape == (8, 2)
    assert np.all(np.isfinite(parts))


# --- svgd theta_dim resolution per weight mode -------------------------------
# param_length = 3 (the edge coefficient length); NO construction theta_dim, so
# the "infer from coefficient length" default would give 3.
def _g_3coeff():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5, 3.0])
    return g


_SVGD_DATA = np.random.default_rng(0).exponential(1.0 / 4.5, size=60)
_SVGD_KW = dict(n_iterations=2, n_particles=4, seed=0, positive_params=True)


def test_svgd_callback_property_requires_theta_dim():
    # A callback set via the PROPERTY (not the kwarg) must also require an
    # explicit theta_dim — a callback is a black box and can't be inferred.
    from phasic.exceptions import SvgdConfigError
    g = _g_3coeff()
    g.weight_callback = lambda th, c: float(c[0] * th[0])
    with pytest.raises(SvgdConfigError,
                       match="callback weight mode requires an explicit theta_dim"):
        g.svgd(_SVGD_DATA, **_SVGD_KW)


def test_svgd_formula_infers_theta_dim_from_n_theta():
    # Formula declares its parameters (n_theta=2); theta_dim is inferred from it,
    # NOT from param_length (=3, the coefficient length).
    g = _g_3coeff()
    g.weight_formula = "c0*t0 + c2*t1"          # n_theta = 2
    res = g.svgd(_SVGD_DATA, **_SVGD_KW)
    assert np.asarray(res.particles).shape == (4, 2)


def test_svgd_formula_theta_dim_below_n_theta_errors():
    from phasic.exceptions import SvgdConfigError
    g = _g_3coeff()
    g.weight_formula = "c0*t0 + c2*t1"          # n_theta = 2
    with pytest.raises(SvgdConfigError, match=r"n_theta=2"):
        g.svgd(_SVGD_DATA, theta_dim=1, **_SVGD_KW)


def test_svgd_formula_explicit_theta_dim_above_n_theta_ok():
    # Explicit theta_dim may exceed n_theta (reserve extra parameters).
    g = _g_3coeff()
    g.weight_formula = "c0*t0"                  # n_theta = 1
    res = g.svgd(_SVGD_DATA, theta_dim=2, **_SVGD_KW)
    assert np.asarray(res.particles).shape == (4, 2)


# --- method_of_moments uses the same per-mode theta_dim resolution -----------
# (callback/formula reach MoM only via the graph's properties; MoM has no
# callback=/weight_formula=/epoch_starts kwargs.)
def test_mom_callback_property_requires_theta_dim():
    from phasic.exceptions import SvgdConfigError
    g = _g_3coeff()
    g.weight_callback = lambda th, c: float(c[0] * th[0])
    with pytest.raises(SvgdConfigError,
                       match="callback weight mode requires an explicit theta_dim"):
        g.method_of_moments(_SVGD_DATA, verbose=False)


def test_mom_formula_infers_theta_dim_from_n_theta():
    g = _g_3coeff()
    g.weight_formula = "c0*t0"                  # n_theta = 1 (not param_length 3)
    res = g.method_of_moments(_SVGD_DATA, verbose=False)
    assert np.asarray(res.theta).shape == (1,)


def test_mom_formula_theta_dim_below_n_theta_errors():
    from phasic.exceptions import SvgdConfigError
    g = _g_3coeff()
    g.weight_formula = "c0*t0 + c2*t1"          # n_theta = 2
    with pytest.raises(SvgdConfigError, match=r"n_theta=2"):
        g.method_of_moments(_SVGD_DATA, theta_dim=1, verbose=False)


def test_mom_linear_uses_param_length():
    g = _g_3coeff()                             # linear, param_length = 3
    res = g.method_of_moments(_SVGD_DATA, verbose=False)
    assert np.asarray(res.theta).shape == (3,)
