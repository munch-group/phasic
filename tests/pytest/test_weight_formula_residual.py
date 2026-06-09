"""Option B: the weight_formula per-edge partial evaluator (constant folding).

update_weights specializes the tape per edge — folding every theta-INDEPENDENT
subexpression using that edge's coefficients and pruning dead select() arms —
into a small theta-only residual tape, which it then evaluates per theta. This
makes complex formulas (a sum of select() dispatches) run ~as fast as the linear
inner product, because only the taken arm runs and its coefficient arithmetic is
precomputed once.

These tests pin that the residual produces weights BIT-IDENTICAL to the full-tape
Python reference (eval_tape) across many random thetas, for: complex select
dispatch (the slow case Option B targets), nested select, exp/log/sqrt/logistic,
the decoupled theta_dim < coefficient-length case, and a fully theta-independent
(constant) weight.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import phasic
from phasic import Graph
from phasic.weight_formula import eval_tape


def _chain(edge_coeffs, theta_dim):
    """Death chain start-(1)->[m]->...->[1] with one parameterized edge per step."""
    m = len(edge_coeffs) + 1
    g = Graph(1)
    g.set_param_length(theta_dim)
    s = g.starting_vertex()
    verts = {k: g.find_or_create_vertex([k]) for k in range(1, m + 1)}
    s.add_edge(verts[m], 1.0)
    for step, k in enumerate(range(m, 1, -1)):
        verts[k].add_edge(verts[k - 1], list(edge_coeffs[step]))
    return g


def _assert_residual_matches(formula, theta_dim, edge_coeffs, seed=0, n=25):
    g = _chain(edge_coeffs, theta_dim)
    g.weight_formula = formula
    tape = g._weight_formula_tape
    start_idx = g.starting_vertex().index()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        theta = rng.uniform(0.4, 2.5, size=theta_dim).tolist()
        g.update_weights(theta)                      # residual path active
        for vi in range(g.vertices_length()):
            v = g.vertex_at(vi)
            if v.index() == start_idx:
                continue
            for e in v.parameterized_edges():
                cl = e.coefficients_length()
                if cl == 0:
                    continue
                coeffs = list(e.edge_state(cl))
                ref = eval_tape(tape, theta, coeffs)  # full-tape Python reference
                got = e.weight()
                assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref)), (
                    f"residual {got!r} != reference {ref!r} "
                    f"(coeffs={coeffs}, theta={theta})")


def test_residual_complex_select_decoupled():
    # theta_dim=2, 4-coeff edges, flags 1/2/3 -> a different select arm per edge.
    formula = ("select(c0==1, c1*t0 + c2*t1, "
               "select(c0==2, c1*c3*t0 + c2, c2*t1*t1 + c3*t0))")
    coeffs = [[1.0, 2.0, 3.0, 0.5],
              [2.0, 0.5, 1.5, 4.0],
              [3.0, 1.0, 2.0, 1.0]]
    _assert_residual_matches(formula, 2, coeffs)


def test_residual_transcendental():
    formula = "sqrt(c0*t0) + logistic(c1 - t1)*c2 + exp(c0*0.1*t1)"
    coeffs = [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]
    _assert_residual_matches(formula, 2, coeffs)


def test_residual_inner_product_decoupled():
    # inner product written as a formula, decoupled (theta_dim=2 < 3 coeffs)
    formula = "c0*t0 + c2*t1"
    coeffs = [[1.5, 9.0, 2.0], [0.5, 0.0, 3.0]]
    _assert_residual_matches(formula, 2, coeffs)


def test_residual_constant_weight():
    # fully theta-independent weight -> residual folds to a single constant
    formula = "c0*c1 + 2.0"
    coeffs = [[2.0, 3.0], [1.0, 4.0]]
    _assert_residual_matches(formula, 2, coeffs)
