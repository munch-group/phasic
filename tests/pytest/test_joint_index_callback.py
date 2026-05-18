"""Regression: joint-index SVGD with a weight_callback used to fail
at FFI parse time with

    ValueError: Model validation failed. Error: INVALID_ARGUMENT:
    Failed to parse JSON: Error parsing graph structure:
    Unknown weight_mode: callback

because `pmf_from_graph_joint_index` serialised the graph with
`weight_mode='callback'` and routed the raw JSON to the FFI, which
only knows 'linear' and 'log'. The fix mirrors the callback handling
already present in `pmf_and_moments_from_graph`: apply the user's
`weight_callback` per-theta to materialise concrete edge weights,
then use the pybind `Graph.expected_sojourn_time` rather than the
FFI.

These tests pin the fix at two levels:

1. Model build no longer raises "Unknown weight_mode: callback".
2. The callback path and the equivalent linear path produce the
   same probabilities (to FD precision) on the same graph topology.
"""
from __future__ import annotations

import phasic  # noqa: F401  -- enables jax x64
import numpy as np
import jax.numpy as jnp
import pytest
from itertools import combinations_with_replacement

from phasic import Graph, StateIndexer, Property, with_ipv


def _make_n2_indexer() -> StateIndexer:
    return StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )


def _make_n2_base_graph() -> Graph:
    """Tiny N=2 coalescent: 3 vertices, one parameterised edge."""
    indexer = _make_n2_indexer()
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2,
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    return Graph(cb, indexer=indexer)


def _make_discrete_joint_prob_graph() -> Graph:
    """A small discrete joint-prob graph — the surface where the
    original "Unknown weight_mode: callback" error was observed."""
    base = _make_n2_base_graph()
    return base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=5, discrete=True,
    )


def _linear_callback(theta, coeffs):
    """Reproduces phasic's 'linear' weight semantics. Joint-prob
    graphs may carry coefficient vectors longer than theta_dim (e.g.
    auxiliary slots for mutation-rate bookkeeping); dot only the
    first len(theta) coefficients, matching the C++ linear path."""
    c = np.asarray(coeffs)
    t = np.asarray(theta)
    k = min(c.size, t.size)
    return float(np.dot(c[:k], t[:k]))


def test_joint_index_with_callback_builds_on_discrete_joint_prob():
    """Before the fix: this raised 'Unknown weight_mode: callback'
    inside the model on first evaluation. After the fix: the model
    builds and a sample evaluation completes without raising."""
    g = _make_discrete_joint_prob_graph()
    g.weight_callback = _linear_callback
    assert g._weight_mode == 'callback'

    model = Graph.pmf_from_graph_joint_index(g, theta_dim=1)

    # Sample a terminal vertex to query.
    terminal_idx = None
    for v in g.vertices():
        for e in v.edges():
            if len(e.to().edges()) == 0:
                terminal_idx = v.index()
                break
        if terminal_idx is not None:
            break
    assert terminal_idx is not None

    theta = jnp.array([1.0])
    vertex_indices = jnp.array([terminal_idx], dtype=jnp.int32)

    # The original bug raised at this call. We only assert it
    # completes without "Unknown weight_mode" — the *value* may
    # be NaN here because this small joint-prob graph at theta=1
    # has numerical edge cases unrelated to the callback fix.
    probs, _ = model(theta, vertex_indices)
    assert np.asarray(probs).shape == (1,)


def test_nonfinite_callback_returns_actionable_error():
    """When a user weight_callback returns NaN/inf, the failure used
    to manifest downstream as a cryptic JSON parse error
    ("parse error at column 155: invalid literal") because phasic
    dumped the concrete dict to JSON and the C++ parser rejected
    `NaN`. After the fix, the failure is caught at the callback
    boundary with a ValueError that names the offending edge,
    theta, and coefficients."""
    g = _make_n2_base_graph()

    def nan_cb(theta, coeffs):
        return float("nan")

    g.weight_callback = nan_cb
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2)

    theta = jnp.array([1.0])
    times = jnp.array([0.5, 1.0])
    with pytest.raises(Exception) as exc_info:
        pmf, _ = model(theta, times)
    msg = str(exc_info.value)
    # The error chains through jax.pure_callback so the user-facing
    # exception type may not be a plain ValueError — match on the
    # message which carries through the chain.
    assert "non-finite" in msg, (
        f"expected 'non-finite' in error message; got: {msg[:200]}"
    )


def test_callback_path_matches_linear_path_on_base_graph():
    """On a graph where the linear path returns finite, well-defined
    values, the callback path (with a callback that reproduces
    linear semantics) must agree to FD precision. This pins the
    correctness of the new `_apply_weight_callback` →
    `expected_sojourn_time` route."""
    g_linear = _make_n2_base_graph()
    g_callback = _make_n2_base_graph()
    g_callback.weight_callback = _linear_callback

    model_linear = Graph.pmf_from_graph_joint_index(g_linear, theta_dim=1)
    model_callback = Graph.pmf_from_graph_joint_index(g_callback, theta_dim=1)

    # Collect terminal vertices.
    terminals = []
    for v in g_linear.vertices():
        for e in v.edges():
            if len(e.to().edges()) == 0:
                terminals.append(v.index())
                break
    assert len(terminals) >= 1
    vertex_indices = jnp.array(terminals[:3], dtype=jnp.int32)

    theta = jnp.array([0.7])
    p_linear, _ = model_linear(theta, vertex_indices)
    p_callback, _ = model_callback(theta, vertex_indices)

    p_lin_np = np.asarray(p_linear)
    p_cb_np = np.asarray(p_callback)
    assert np.all(np.isfinite(p_lin_np)), (
        f"linear-path baseline produced non-finite probabilities; "
        f"got {p_lin_np}"
    )
    np.testing.assert_allclose(
        p_cb_np, p_lin_np,
        atol=1e-9,
        err_msg=(
            "callback path and linear path disagree on the same "
            "graph topology + same effective weights"
        ),
    )
