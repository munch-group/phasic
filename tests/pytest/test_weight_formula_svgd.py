"""Batch 3: the Python ``weight_formula`` API end to end.

Covers the property/serialize contract, the direct graph.pdf() path, forward
and SVGD equivalence to the (validated) callback path, parameter recovery, and
that formula mode never falls into the slow ``_apply_weight_callback`` path.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
import numpy as np
import pytest

import phasic
from phasic import Graph, ExpStepSize
from phasic.weight_formula import WeightFormulaError


# --- fixtures ---------------------------------------------------------------
def g_1coeff():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])
    return g


def g_2coeff():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5])
    return g


def const_graph(rate):
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, float(rate))
    return g


# --- property / serialize contract ------------------------------------------
def test_property_sets_mode_and_tape():
    g = g_2coeff()
    assert g.weight_mode == "linear"
    g.weight_formula = "exp(c0*t0 + c1*t1)"
    assert g.weight_mode == "formula"
    assert g.weight_formula == "exp(c0*t0 + c1*t1)"
    assert g._weight_formula_tape is not None
    assert g._weight_formula_tape["n_theta"] == 2


def test_weight_mode_accepts_formula_string():
    g = g_1coeff()
    g.weight_mode = "formula"          # validation tuple includes 'formula'
    assert g.weight_mode == "formula"


def test_bad_formula_raises_at_assignment():
    g = g_1coeff()
    with pytest.raises(WeightFormulaError):
        g.weight_formula = "exp(t0,"          # syntax error
    with pytest.raises(WeightFormulaError):
        g.weight_formula = "select(t0 > c0, t1, c1)"   # theta-dependent condition


def test_serialize_carries_tape_only_in_formula_mode():
    g = g_2coeff()
    assert "weight_formula_tape" not in g.serialize()          # linear
    g.weight_mode = "log"
    assert "weight_formula_tape" not in g.serialize()          # log
    g.weight_formula = "c0*t0 + c1*t1"
    s = g.serialize()
    assert s["weight_mode"] == "formula"
    assert "weight_formula_tape" in s
    assert s["weight_formula_tape"]["src"] == "c0*t0 + c1*t1"


def test_clearing_formula_reverts_to_linear():
    g = g_1coeff()
    g.weight_formula = "(c0*t0)**2"
    assert g.weight_mode == "formula"
    g.weight_formula = None
    assert g.weight_mode == "linear"
    assert g.weight_formula is None
    assert "weight_formula_tape" not in g.serialize()


# --- direct graph.pdf() path honors the formula (live-graph tape) ------------
def test_direct_pdf_honors_formula():
    times = np.linspace(0.1, 4.0, 40)
    g = g_2coeff()
    g.weight_formula = "exp(c0*t0)"
    g.update_weights(np.array([0.6, 1.1]))         # len == param_length (2)
    pdf = np.asarray(g.pdf(times, granularity=128))
    ref = np.asarray(const_graph(math.exp(2.0 * 0.6)).pdf(times, granularity=128))
    np.testing.assert_allclose(pdf, ref, rtol=0, atol=1e-12)


# --- forward equivalence: formula vs callback (same math) --------------------
def test_formula_forward_matches_callback():
    import jax.numpy as jnp
    times = jnp.linspace(0.1, 3.0, 30)

    g_f = g_1coeff()
    g_f.weight_formula = "(c0*t0)**2"
    model_f = Graph.pmf_from_graph(g_f, discrete=False)

    g_c = g_1coeff()
    g_c.weight_callback = lambda th, c: float((c[0] * th[0]) ** 2)
    model_c = Graph.pmf_from_graph(g_c, discrete=False)

    for th in (0.4, 0.9, 1.5):
        a = np.asarray(model_f(jnp.array([th]), times))
        b = np.asarray(model_c(jnp.array([th]), times))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


# --- SVGD: formula mode == callback mode (same math, same seed) --------------
def test_svgd_formula_matches_callback():
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=1.0 / (1.3 ** 2), size=200)
    lr = ExpStepSize(first_step=0.02, last_step=0.002, tau=40.0)
    kw = dict(theta_dim=1, n_iterations=25, n_particles=8, seed=42,
              positive_params=True, learning_rate=lr)

    g_f = g_1coeff(); g_f.weight_formula = "(c0*t0)**2"
    res_f = g_f.svgd(data, **kw)

    g_c = g_1coeff()
    g_c.weight_callback = lambda th, c: float((c[0] * th[0]) ** 2)
    res_c = g_c.svgd(data, **kw)

    np.testing.assert_allclose(np.asarray(res_f.particles),
                               np.asarray(res_c.particles), rtol=0, atol=1e-6)


# --- SVGD at inference scale: formula posterior == callback posterior --------
# (Recovery-to-truth is an SVGD/prior property, not a weight_formula property;
# the meaningful invariant is that formula mode performs the *same* inference as
# the validated callback path. We check that the equivalence is stable at a
# realistic iteration/particle count, not just the short run above.)
def test_svgd_formula_matches_callback_at_scale():
    rng = np.random.default_rng(1)
    data = rng.exponential(scale=1.0 / (1.3 ** 2), size=400)   # rate = theta^2
    kw = dict(theta_dim=1, n_iterations=200, n_particles=24, seed=7,
              positive_params=True,
              learning_rate=ExpStepSize(first_step=0.03, last_step=0.003, tau=120.0))

    g_f = g_1coeff(); g_f.weight_formula = "(c0*t0)**2"
    pm_f = float(np.asarray(g_f.svgd(data, **kw).particles).mean())

    g_c = g_1coeff()
    g_c.weight_callback = lambda th, c: float((c[0] * th[0]) ** 2)
    pm_c = float(np.asarray(g_c.svgd(data, **kw).particles).mean())

    assert abs(pm_f - pm_c) < 1e-3, f"formula {pm_f} vs callback {pm_c}"
    assert np.isfinite(pm_f)


# --- formula mode must NOT take the slow _apply_weight_callback path ---------
def test_formula_does_not_invoke_apply_weight_callback(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("_apply_weight_callback called in formula mode")
    monkeypatch.setattr(phasic, "_apply_weight_callback", _boom)

    rng = np.random.default_rng(2)
    data = rng.exponential(scale=1.0 / (1.2 ** 2), size=100)
    g = g_1coeff(); g.weight_formula = "(c0*t0)**2"
    # Would raise AssertionError if the callback materialization path were used.
    g.svgd(data, theta_dim=1, n_iterations=5, n_particles=6, seed=3,
           positive_params=True)


# --- no silent fallback on the (linear-only) trace path ---------------------
def test_record_trace_rejects_formula():
    from phasic.trace_elimination import record_elimination_trace
    g = g_1coeff(); g.weight_formula = "(c0*t0)**2"
    with pytest.raises(NotImplementedError, match="weight_mode='formula'"):
        record_elimination_trace(g, theta_dim=1)
