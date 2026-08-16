"""D1-E2 guard: exact-gradient cores DECLINE on synthetic SCC graphs.

Synthetic SCC graphs (SCCVertex.as_synthetic_graph) carry PLACEHOLDER
coefficients on Type-A/phantom edges; the shipped exact-gradient
functions used to contract those as real dw/dtheta and return
plausible-but-wrong Jacobians (b3-d1-derisk-findings.md E2, the
pre-guard record: a full-size finite J on this very fixture). The
guard is a `synthetic` marker checked at the top of the two shared
gradient cores. Plan: b3-d1-e2-guard-plan.md.
"""
import logging
import os

import numpy as np
import pytest

import phasic
from phasic import Graph, set_log_level

set_log_level("WARNING")

THETA = [1.0, 2.0]


def two_scc_graph():
    """Two cyclic SCCs in series (the E2 de-risk fixture)."""
    g = Graph(1)
    s = g.starting_vertex()
    a1 = g.find_or_create_vertex([10])
    a2 = g.find_or_create_vertex([11])
    b1 = g.find_or_create_vertex([20])
    b2 = g.find_or_create_vertex([21])
    ab = g.find_or_create_vertex([1])
    s.add_edge(a1, 1.0)
    a1.add_edge(a2, [1.0, 0.0])
    a2.add_edge(a1, [0.5, 0.0])
    a2.add_edge(b1, [1.0, 0.0])
    b1.add_edge(b2, [0.0, 1.0])
    b2.add_edge(b1, [0.0, 0.5])
    b2.add_edge(ab, [1.0, 1.0])
    return g


def make_synth():
    g = two_scc_graph()
    g.update_weights(THETA)
    scc = g.scc_decomposition()
    for i in range(scc.n_sccs()):
        sg, meta = scc.scc_at(i).as_synthetic_graph()
        if meta.n_channels > 0:
            return g, sg
    raise AssertionError("no SCC with channels in the fixture")


def test_moments_declines_on_synthetic(monkeypatch):
    # Threshold lifted sky-high: pins that the decline is the synthetic
    # guard, NOT the MPFR conditioning gate (the moments analogue of
    # the sojourn test's skip_condition_gate=True).
    monkeypatch.setenv("PHASIC_CONDITION_THRESHOLD", "1e300")
    _, synth = make_synth()
    synth.update_weights(THETA)
    J = np.asarray(synth._moments_grad_theta(1))
    assert J.size == 0, (
        "guard broken: exact moments gradient ACCEPTED a synthetic SCC "
        f"graph (J={J.tolist()})")


def test_moments_declines_on_synthetic_clone(monkeypatch):
    # Pins the clone-propagation line: the Python exact path computes
    # on a private graph.clone(), so a dropped marker would silently
    # re-arm the landmine through pmf_and_moments_from_graph.
    monkeypatch.setenv("PHASIC_CONDITION_THRESHOLD", "1e300")
    _, synth = make_synth()
    synth.update_weights(THETA)
    J = np.asarray(synth.clone()._moments_grad_theta(1))
    assert J.size == 0, "guard broken: clone dropped the synthetic marker"


def test_sojourn_declines_on_synthetic_both_gate_settings():
    _, synth = make_synth()
    synth.update_weights(THETA)
    for skip in (False, True):
        J = synth._sojourn_grad_theta_subset([0], skip_condition_gate=skip)
        assert len(J) == 0, (
            f"guard broken: sojourn gradient (skip_condition_gate={skip}) "
            "ACCEPTED a synthetic SCC graph -- and skip=True pins that "
            "the synthetic check is not the conditioning gate")


def test_regular_graph_unaffected():
    # Non-regression: the PARENT parameterized graph is not synthetic;
    # both families must still return non-empty finite Jacobians.
    g = two_scc_graph()
    g.update_weights(THETA)
    Jm = np.asarray(g._moments_grad_theta(1))
    assert Jm.size > 0 and np.all(np.isfinite(Jm)), (
        "guard over-fires: regular graph's exact moments gradient declined")
    # BOTH sojourn wrappers (gated and gate-skipping) -- a guard that
    # over-fires in only one of them must not slip through.
    for skip in (False, True):
        Js = g._sojourn_grad_theta_subset([1], skip_condition_gate=skip)
        assert len(Js) > 0 and np.all(np.isfinite(np.asarray(Js))), (
            f"guard over-fires: regular graph's sojourn gradient "
            f"(skip_condition_gate={skip}) declined")


def test_marker_does_not_leak_onto_parent(monkeypatch):
    """The marker must land on the SYNTH, never on the parent it was
    decomposed from -- a `parent->synthetic = true` slip in
    ptd_scc_build_synthetic_graph would otherwise pass every other test
    in this file (G4 refuter A, finding M2)."""
    monkeypatch.setenv("PHASIC_CONDITION_THRESHOLD", "1e300")
    parent, synth = make_synth()          # decomposition already happened
    parent.update_weights(THETA)
    Jp = np.asarray(parent._moments_grad_theta(1))
    assert Jp.size > 0 and np.all(np.isfinite(Jp)), (
        "marker LEAKED onto the parent graph: its exact moments gradient "
        "declined after as_synthetic_graph() was called on its SCC")
    for skip in (False, True):
        Jps = parent._sojourn_grad_theta_subset([1], skip_condition_gate=skip)
        assert len(Jps) > 0, (
            f"marker LEAKED onto the parent (sojourn, skip={skip})")


def test_python_graph_clone_route_declines(monkeypatch):
    """The production exact path clones via the PYTHON wrapper
    (`Graph(super().clone())`, __init__.py), not the raw pybind clone --
    pin that route too (G4 refuter A, finding n5)."""
    monkeypatch.setenv("PHASIC_CONDITION_THRESHOLD", "1e300")
    _, synth = make_synth()
    synth.update_weights(THETA)
    wrapped = phasic.Graph(synth)          # the wrapper production uses
    J = np.asarray(wrapped.clone()._moments_grad_theta(1))
    assert J.size == 0, (
        "guard broken: the Python Graph.clone() route (the one the "
        "production exact path actually takes) kept the exact gradient")


def test_distributed_roundtrip_keeps_the_marker():
    """A synth serialized for the SLURM per-SCC worker and rebuilt must
    STILL decline: the marker is not serialized, so deserialize_scc_synth
    re-applies it. Without that, the round-trip returns the pre-guard
    landmine Jacobian (G4 refuter A, finding M1 -- measured [-1.0, 0.0])."""
    from phasic.distributed_scc import (serialize_scc_synth,
                                        deserialize_scc_synth)
    g = two_scc_graph()
    g.update_weights(THETA)
    scc = g.scc_decomposition()
    payload = None
    for i in range(scc.n_sccs()):
        sg, meta = scc.scc_at(i).as_synthetic_graph()
        if meta.n_channels > 0:
            payload = serialize_scc_synth(scc.scc_at(i))
            break
    assert payload is not None, "no SCC with channels in the fixture"
    rebuilt = deserialize_scc_synth(payload)
    rebuilt.update_weights(THETA)
    J = np.asarray(rebuilt._moments_grad_theta(1))
    assert J.size == 0, (
        f"guard bypassed across the distributed serialize/deserialize "
        f"boundary: got J={J.tolist()} (pre-guard landmine was [-1.0, 0.0])")
    Js = rebuilt._sojourn_grad_theta_subset([1], skip_condition_gate=True)
    assert len(Js) == 0, "guard bypassed across the distributed boundary (sojourn)"


def test_decline_logs_c_warning(caplog):
    # The guard's log contract: a WARNING naming the synthetic cause,
    # visible at the DEFAULT level. phasic sets propagate=False, so
    # attach to the phasic loggers directly (the
    # test_exposure_daisy_chain.py precedent).
    # Force the level FIRST: another test in the same session may have
    # left the C-side bridge level above WARNING (set_log_level is the
    # only call that propagates to C -- the precedent test does this
    # too).
    set_log_level("WARNING")
    phasic_root = logging.getLogger("phasic")
    phasic_c = logging.getLogger("phasic.c")
    saved = (phasic_root.propagate, phasic_c.propagate)
    phasic_root.propagate = True
    phasic_c.propagate = True
    caplog.set_level(logging.WARNING, logger="phasic.c")
    caplog.set_level(logging.WARNING, logger="phasic")
    try:
        _, synth = make_synth()
        synth.update_weights(THETA)
        np.asarray(synth._moments_grad_theta(1))
        assert any("synthetic" in r.getMessage().lower()
                   for r in caplog.records), (
            "guard declined without the WARNING log line naming the "
            "synthetic cause")
    finally:
        phasic_root.propagate, phasic_c.propagate = saved
