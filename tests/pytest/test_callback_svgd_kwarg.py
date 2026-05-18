"""Tests for the ``Graph.svgd(callback=...)`` kwarg.

Plan: typed-riding-truffle. The kwarg is sugar over the existing
``graph.weight_callback = fn`` attribute form. Three properties to pin:

1. **Parity.** ``graph.svgd(..., callback=fn)`` produces the same
   posterior particles as ``graph.weight_callback = fn;
   graph.svgd(...)`` at the same seed.
2. **Restoration.** After ``graph.svgd(..., callback=fn)`` returns
   (successfully or by exception), ``graph.weight_callback`` and
   ``graph._weight_mode`` match their pre-call values.
3. **Daisy-chain rejection.** Combining the kwarg with
   ``epoch_starts=...`` raises ``SvgdConfigError`` (validator rule
   R21), with no graph-state mutation.
"""
from __future__ import annotations

import os

# JAX x64 required by the SVGD pipeline; set before any jax import in
# test collection to avoid the late-x64 warning.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from phasic import Graph, ExpStepSize
from phasic.exceptions import SvgdConfigError


# ---------------------------------------------------------------------------
# Tiny single-parameter exponential fixture. ``callback(theta, coeffs)``
# returns ``coeffs[0] * theta[0]`` — bit-identical to the default
# linear weight_mode, so kwarg and attribute paths must produce
# bit-identical posteriors at the same seed.
# ---------------------------------------------------------------------------


def _build_exp_graph() -> Graph:
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def _linear_callback(theta, coeffs):
    """Equivalent to weight_mode='linear': Σ c_k · θ_k."""
    return float(coeffs[0]) * float(theta[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_callback_kwarg_matches_attribute_form():
    """Particles from `svgd(callback=fn)` equal particles from
    `weight_callback = fn; svgd(...)` at the same seed."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=30)

    # Form A: attribute.
    g_a = _build_exp_graph()
    g_a.weight_callback = _linear_callback
    svgd_a = g_a.svgd(
        data, theta_dim=1, n_iterations=8, n_particles=8, seed=42,
        learning_rate=ExpStepSize(
            first_step=0.05, last_step=0.005, tau=20.0,
        ),
        positive_params=True,
    )

    # Form B: kwarg.
    g_b = _build_exp_graph()
    svgd_b = g_b.svgd(
        data, theta_dim=1, n_iterations=8, n_particles=8, seed=42,
        learning_rate=ExpStepSize(
            first_step=0.05, last_step=0.005, tau=20.0,
        ),
        positive_params=True,
        callback=_linear_callback,
    )

    np.testing.assert_allclose(
        np.asarray(svgd_a.particles),
        np.asarray(svgd_b.particles),
        rtol=0, atol=1e-12,
    )


def test_callback_kwarg_restores_attribute_on_success():
    """After a successful kwarg call, graph.weight_callback and
    graph._weight_mode return to their pre-call values."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)

    g = _build_exp_graph()
    # Pre-call: no callback set; weight_mode is the default.
    assert g.weight_callback is None
    pre_mode = g._weight_mode

    _ = g.svgd(
        data, theta_dim=1, n_iterations=4, n_particles=4, seed=1,
        positive_params=True,
        callback=_linear_callback,
    )

    # Post-call: callback restored to None; mode restored to default.
    assert g.weight_callback is None, (
        "kwarg leaked: weight_callback is set after svgd returned"
    )
    assert g._weight_mode == pre_mode, (
        f"kwarg leaked: weight_mode changed from {pre_mode!r} to "
        f"{g._weight_mode!r} after svgd returned"
    )


def test_callback_kwarg_restores_attribute_on_exception():
    """If the SVGD optimize step raises, the kwarg's attribute
    override is STILL rolled back via finally."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)

    g = _build_exp_graph()
    pre_mode = g._weight_mode

    def _exploding_callback(theta, coeffs):
        raise RuntimeError("boom")

    with pytest.raises(Exception):
        # The callback raises inside the model evaluation; SVGD will
        # propagate it. We just need to confirm the finally runs.
        g.svgd(
            data, theta_dim=1, n_iterations=2, n_particles=4, seed=1,
            positive_params=True,
            callback=_exploding_callback,
        )

    assert g.weight_callback is None, (
        "kwarg leaked on exception: weight_callback still set"
    )
    assert g._weight_mode == pre_mode, (
        f"kwarg leaked on exception: weight_mode changed from "
        f"{pre_mode!r} to {g._weight_mode!r}"
    )


def test_callback_kwarg_preserves_user_attribute_form():
    """If the user already had graph.weight_callback set BEFORE the
    svgd(callback=...) call, it's restored to that prior callable
    afterwards (not nulled out)."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)

    def _user_cb(theta, coeffs):
        return float(coeffs[0]) * float(theta[0])

    def _override_cb(theta, coeffs):
        # Same numerics as _user_cb; just a different identity to
        # detect leaked references.
        return float(coeffs[0]) * float(theta[0])

    g = _build_exp_graph()
    g.weight_callback = _user_cb
    assert g.weight_callback is _user_cb

    _ = g.svgd(
        data, theta_dim=1, n_iterations=4, n_particles=4, seed=1,
        positive_params=True,
        callback=_override_cb,
    )

    assert g.weight_callback is _user_cb, (
        "kwarg leaked: prior user attribute was not restored "
        "(got a different callable identity)"
    )


def test_callback_kwarg_requires_explicit_theta_dim_or_theta_init():
    """`callback=` without either `theta_dim=` or `theta_init=` is
    rejected so the user's assumed parameter dimensionality is never
    silently replaced by `graph.param_length()` (which reflects the
    edge coefficient length, not necessarily what the callback treats
    as parameters)."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)

    g = _build_exp_graph()
    pre_mode = g._weight_mode
    assert g.weight_callback is None

    with pytest.raises(
        SvgdConfigError,
        match=r"callback= requires an explicit theta_dim= or theta_init=",
    ):
        g.svgd(
            data,
            n_iterations=2, n_particles=4, seed=1,
            positive_params=True,
            callback=_linear_callback,
            # NB: no theta_dim, no theta_init.
        )

    # Error fired before the kwarg's attribute override, so the
    # graph's weight_callback / weight_mode are untouched.
    assert g.weight_callback is None
    assert g._weight_mode == pre_mode


def test_callback_kwarg_accepts_theta_init_alone():
    """Supplying `theta_init` (without `theta_dim`) is sufficient: its
    second dim defines the parameter dimensionality unambiguously."""
    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=20)

    g = _build_exp_graph()
    theta_init = np.abs(np.random.randn(4, 1)) + 0.1  # (n_particles, theta_dim)

    res = g.svgd(
        data,
        n_iterations=2, n_particles=4, seed=1,
        positive_params=True,
        callback=_linear_callback,
        theta_init=theta_init,
    )
    assert np.asarray(res.particles).shape[1] == 1


def test_callback_kwarg_rejects_epoch_starts(tmp_path):
    """Validator R21 fires before any graph mutation when the kwarg is
    combined with epoch_starts=... (daisy-chain SVGD)."""
    from itertools import combinations_with_replacement
    from phasic import StateIndexer, Property, with_ipv

    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
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

    g = Graph(cb, indexer=indexer)
    jp = g.joint_prob_graph(
        indexer, mutation_rate=0.1, reward_limit=2, discrete=False,
    )

    pre_mode = jp._weight_mode
    assert jp.weight_callback is None

    with pytest.raises(SvgdConfigError, match=r"callback.*not supported.*epoch_starts"):
        jp.svgd(
            [(0, 0)] * 4,
            epoch_starts=[0.0, 0.5],
            n_iterations=2,
            n_particles=4,
            callback=_linear_callback,
        )

    # Validator raised BEFORE the kwarg's attribute override fired,
    # so weight_callback is still None and weight_mode is unchanged.
    assert jp.weight_callback is None
    assert jp._weight_mode == pre_mode
