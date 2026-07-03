"""Regression test: weight mode/formula tape must survive graph derivation.

Constructing Graph from a C++ _Graph resets _weight_mode='linear' and nulls
the formula tape, and the C-level clone never copies the tape. clone()/copy(),
from_serialized(), _rebuild_with_wider_layout() (add_epoch/discretize) and
laplace_transform() all failed to restore it, so a formula-mode graph silently
reverted to plain inner-product (linear) weights after any of these — a
silently wrong pdf, or a mode='formula'/tape=None inconsistency that raises on
serialize.

These tests build a formula graph, derive a new graph, update_weights() with a
FRESH theta on the derivative, and assert its pdf matches a fresh formula graph
(not the linear fallback).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from phasic import Graph

TIMES = np.linspace(0.1, 4.0, 30)
GRAN = 128
FORMULA = "exp(c0*t0)"
THETA1 = np.array([0.6, 1.1])
THETA2 = np.array([0.9, 0.4])  # different t0 => different weights


def _formula_graph():
    """Parameterized 3-state graph; the single param edge has coeffs [2.0, 0.5]."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5])
    return g


def _reference_pdf(theta):
    g = _formula_graph()
    g.weight_formula = FORMULA
    g.update_weights(theta)
    return np.asarray(g.pdf(TIMES, granularity=GRAN))


def test_copy_preserves_formula_mode_and_pdf():
    src = _formula_graph()
    src.weight_formula = FORMULA
    src.update_weights(THETA1)

    cp = src.copy()
    assert cp._weight_mode == "formula", "copy() lost formula weight mode"
    assert cp._weight_formula_tape is not None, "copy() lost the compiled tape"

    # Re-evaluate with a fresh theta: linear fallback would give wrong weights.
    cp.update_weights(THETA2)
    got = np.asarray(cp.pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(got, _reference_pdf(THETA2), rtol=1e-6, atol=1e-9)


def test_from_serialized_preserves_formula_mode_and_pdf():
    src = _formula_graph()
    src.weight_formula = FORMULA
    src.update_weights(THETA1)

    data = src.serialize()
    assert data.get("weight_mode") == "formula"
    assert data.get("weight_formula_tape") is not None

    de = Graph.from_serialized(data)
    assert de._weight_mode == "formula", "from_serialized lost formula weight mode"
    assert de._weight_formula_tape is not None

    de.update_weights(THETA2)
    got = np.asarray(de.pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(got, _reference_pdf(THETA2), rtol=1e-6, atol=1e-9)


def test_copy_of_formula_graph_serializes_without_error():
    """mode='formula' with tape=None used to raise here."""
    src = _formula_graph()
    src.weight_formula = FORMULA
    src.update_weights(THETA1)
    cp = src.copy()
    d = cp.serialize()  # must not raise
    assert d["weight_mode"] == "formula"
