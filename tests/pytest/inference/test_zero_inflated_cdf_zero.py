"""Tests for the fused cdf_zero path used by the zero-inflated likelihood.

Pins two invariants:

1. **Equivalence**: at the same theta, `1 - model._cdf_zero_fn(theta, rewards)`
   matches the legacy `compute_reward_visit_probability_ffi` result
   to numerical tolerance. The two paths must agree for SVGD posteriors
   to be invariant to which one is wired.

2. **Wiring**: when `Graph.svgd(observed_data, rewards=...)` is called
   with partial-coverage rewards, the model exposes `_cdf_zero_fn`
   and the zero-inflated `_zero_inflated_p_fn` uses it (not
   `compute_reward_visit_probability_ffi`). Asserted by monkey-
   patching the legacy entry point and checking it is never called.
"""

import phasic  # noqa: F401  -- activates jax x64
import numpy as np
import jax.numpy as jnp
import pytest
from phasic import Graph


def _build_branch_graph():
    """3-state branch + absorbing graph with normalised IPV.
    State [1] is the only rewarded vertex; partial-coverage atom = 0.5."""
    g = Graph(1)
    v0 = g.starting_vertex()
    v_a = g.find_or_create_vertex([1])
    v_b = g.find_or_create_vertex([2])
    v_abs = g.find_or_create_vertex([3])
    v0.add_edge(v_a, [0.5])
    v0.add_edge(v_b, [0.5])
    v_a.add_edge(v_abs, [1.0])
    v_b.add_edge(v_abs, [1.0])
    return g


def test_cdf_zero_matches_visit_probability_continuous():
    """1 - cdf_zero(theta, r) == p_visit(theta, r) on the same graph
    in continuous mode."""
    g = _build_branch_graph()
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

    rewards_1d = jnp.array([0.0, 1.0, 0.0, 0.0])
    theta = jnp.array([1.0])

    # cdf_zero from the new pybind path.
    cdf_zero = np.asarray(model._cdf_zero_fn(theta, rewards_1d))
    p_new = 1.0 - cdf_zero[0]

    # Legacy path.
    p_legacy = float(g.reward_visit_probability(
        np.asarray(rewards_1d), theta=np.asarray(theta),
    ))

    assert np.isclose(p_new, p_legacy, atol=1e-10), (
        f"cdf_zero path disagrees with legacy reward_visit_probability: "
        f"p_new={p_new}, p_legacy={p_legacy}"
    )


def test_cdf_zero_2d_per_feature_continuous():
    """For 2D rewards, cdf_zero is shape (n_features,) and matches
    legacy visit-probability per feature."""
    g = _build_branch_graph()
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

    rewards_2d = jnp.array([
        [0.0, 1.0, 0.0, 0.0],   # partial coverage (atom = 0.5)
        [0.0, 1.0, 1.0, 0.0],   # full coverage (atom = 0.0)
    ])
    theta = jnp.array([1.0])

    cdf_zero = np.asarray(model._cdf_zero_fn(theta, rewards_2d))
    assert cdf_zero.shape == (2,)
    p_new = 1.0 - cdf_zero

    p_legacy = np.array([
        float(g.reward_visit_probability(np.asarray(rewards_2d[j]),
                                         theta=np.asarray(theta)))
        for j in range(2)
    ])

    assert np.allclose(p_new, p_legacy, atol=1e-10), (
        f"per-feature cdf_zero diverges from legacy: "
        f"p_new={p_new}, p_legacy={p_legacy}"
    )


def test_svgd_partial_coverage_uses_cdf_zero_path():
    """When `Graph.svgd` is called with partial-coverage 1D rewards
    via the default pybind path, the wired `_zero_inflated_p_fn` reads
    from `_cdf_zero_fn` and does NOT call
    `compute_reward_visit_probability_ffi`."""
    g = _build_branch_graph()

    # Construct partial-coverage observations: a few zeros, a few positives.
    rewards = jnp.array([0.0, 1.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    pos_samples = rng.exponential(scale=2.0, size=5)
    observed_data = jnp.array(
        np.concatenate([np.zeros(3, dtype=np.float64), pos_samples])
    )

    # Monkey-patch the legacy FFI call site to fail if invoked. Apply
    # the patch on the module-level binding the wiring would import.
    import phasic.ffi_wrappers as fw
    sentinel = []
    original = fw.compute_reward_visit_probability_ffi

    def _trip(*a, **kw):
        sentinel.append(True)
        return original(*a, **kw)

    fw.compute_reward_visit_probability_ffi = _trip
    try:
        svgd = g.svgd(
            observed_data=observed_data,
            rewards=rewards,
            theta_dim=1,
            n_particles=4,
            n_iterations=2,
            progress=False,
        )
    finally:
        fw.compute_reward_visit_probability_ffi = original

    # The legacy entry point must not have been called by the
    # zero-inflated wiring on the default pybind path.
    assert not sentinel, (
        f"compute_reward_visit_probability_ffi was called "
        f"{len(sentinel)} time(s) during SVGD; the new cdf_zero path "
        "should make this unnecessary."
    )
    # And introspection must still show feature 0 as zero-inflated.
    assert 0 in svgd.zero_inflated_features
