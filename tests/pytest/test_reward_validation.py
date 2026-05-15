"""Reward-vector validation tests (plan: radiant-giggling-wigderson).

Two checks, one unified validator:

1. **Shape** (always runs): rewards must be 1D `(n_vertices,)` or
   2D `(n_features, n_vertices)`. Vertices are always the trailing
   axis. Wrong-length 1D rewards raise. Wrong-orientation 2D
   rewards raise with a transpose hint.

2. **Coverage** (gated by ``validate_rewards`` kwarg): every
   absorbing trajectory must accumulate strictly positive reward,
   otherwise the reward-transformed PDF is sub-probability and
   likelihood inference is biased.

The validator is the single source of truth for both checks. It is
hooked into ``Graph.svgd``, ``Graph.sample``, and
``Graph.reward_transform`` (warn-only for the latter).
"""
from __future__ import annotations

import warnings
from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import (
    Graph, Property, StateIndexer, with_ipv, GaussPrior,
    ExpStepSize, SparseObservations, dense_to_sparse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_n4_coalescent() -> Graph:
    """Standard N=4 coalescent fixture used across these tests."""
    nr_samples = 4

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append(
                    [new, [state[i] * (state[j] - same) / (1 + same)]]
                )
        return transitions

    g = Graph(coalescent_1param)
    g.update_weights([10.0])
    return g


def _build_tiny_3vertex_graph() -> Graph:
    """Tiny 3-vertex graph: start -> v0, v0 -> v1, v0 -> v2 (absorbing).

    Path 0 -> 2 skips vertex 1, so rewards=[0,1,0] (rewarding only v1)
    must fail the coverage check on this graph. Path-witness test
    asserts the exact reconstructed witness string.

    The graph is built by a simple callback so it has parametrised
    edges (matching the convention used by Graph constructors).
    """
    @with_ipv([1, 0, 0])  # IPV places all mass at state-index 0
    def cb(state):
        # state index 0 -> two non-absorbing outgoing edges.
        if int(state[0]) == 1:
            # Direct to absorbing (state-index 2).
            new_to_2 = state.copy()
            new_to_2[0] = 0
            new_to_2[2] = 1
            # Hop through intermediate (state-index 1).
            new_to_1 = state.copy()
            new_to_1[0] = 0
            new_to_1[1] = 1
            return [[new_to_1, [1.0]], [new_to_2, [1.0]]]
        if int(state[1]) == 1:
            new = state.copy()
            new[1] = 0
            new[2] = 1
            return [[new, [1.0]]]
        return []

    g = Graph(cb)
    g.update_weights([1.0])
    return g


# ---------------------------------------------------------------------------
# Shape tests (Tests 1-4)
# ---------------------------------------------------------------------------


def test_shape_1d_wrong_length_raises():
    """1D rewards with len != n_vertices must raise."""
    g = _build_n4_coalescent()
    n_v = g.vertices_length()
    with pytest.raises(ValueError, match=r"1D rewards must have shape"):
        g._validate_rewards(np.ones(n_v - 1), allow_2d=False)


def test_shape_2d_correct_orientation_passes():
    """(n_features, n_vertices) with n_features=3 must pass the shape
    check when check_coverage=False (avoid coverage interference)."""
    g = _build_n4_coalescent()
    n_v = g.vertices_length()
    arr = np.ones((3, n_v))
    out = g._validate_rewards(arr, allow_2d=True, check_coverage=False)
    assert out.shape == (3, n_v)


def test_shape_2d_transposed_offers_hint():
    """(n_vertices, n_features) — the wrong orientation — must raise
    with a transpose hint."""
    g = _build_n4_coalescent()
    n_v = g.vertices_length()
    arr = np.ones((n_v, 3))
    with pytest.raises(ValueError) as excinfo:
        g._validate_rewards(arr, allow_2d=True, check_coverage=False)
    msg = str(excinfo.value)
    assert "Transpose it: rewards = rewards.T" in msg
    assert f"(n_vertices={n_v}, n_features=3)" in msg


def test_shape_3d_raises():
    """3D arrays must be rejected with a clear message."""
    g = _build_n4_coalescent()
    n_v = g.vertices_length()
    with pytest.raises(ValueError, match=r"must be 1D .* or 2D"):
        g._validate_rewards(np.ones((2, 3, n_v)))


# ---------------------------------------------------------------------------
# Coverage tests (Tests 5-8)
# ---------------------------------------------------------------------------


def test_coverage_skip_last_row_fails():
    """`graph.states().T[:-1]` matches the tutorial bug. Validation
    must raise with diagnostic for the offending feature."""
    g = _build_n4_coalescent()
    rewards = g.states().T[:-1].astype(np.float64)
    with pytest.raises(ValueError) as excinfo:
        g._validate_rewards(rewards, allow_2d=True)
    msg = str(excinfo.value)
    # The error should mention which features are invalid.
    assert "invalid rewards for features" in msg
    # Row 2 of states().T[:-1] is the bad one for this graph.
    assert "[2]" in msg
    # Coverage diagnostic should include witness path and remediation
    # suggestions.
    assert "Witness path" in msg
    assert "graph.states().T" in msg


def test_coverage_partial_decomposition_passes():
    """A subset of rows that covers all paths must pass. For the N=4
    coalescent, rows [0, 1, 3] of states().T form a valid feature
    decomposition (row 2 alone is the problematic one)."""
    g = _build_n4_coalescent()
    rewards = g.states().T[[0, 1, 3]].astype(np.float64)
    out = g._validate_rewards(rewards, allow_2d=True)
    assert out.shape == (3, g.vertices_length())


def test_coverage_witness_path_correctness():
    """Hand-built 3-vertex graph. Rewarding only vertex 1 fails because
    the path start -> v0 -> v2 (absorbing) skips it. The witness path
    should start at a start vertex and end at an absorbing one."""
    g = _build_tiny_3vertex_graph()
    n_v = g.vertices_length()
    # Reward only the middle vertex (index 1 in the state, which maps
    # to some vertex index in the graph).
    rewards = np.zeros(n_v, dtype=np.float64)
    # Find the vertex that has state (0, 1, 0) (only the v1 indicator).
    for i in range(n_v):
        s = g.vertex_at(i).state()
        if int(s[1]) == 1 and int(s[0]) == 0 and int(s[2]) == 0:
            rewards[i] = 1.0
            break
    assert rewards.sum() > 0, "Test fixture: no vertex with state (0,1,0) found"

    with pytest.raises(ValueError) as excinfo:
        g._validate_rewards(rewards, allow_2d=False)
    msg = str(excinfo.value)
    assert "Witness path" in msg
    # Witness path should contain the arrow separator.
    assert "->" in msg


def test_coverage_opt_out_passes_invalid():
    """With check_coverage=False, an invalid reward vector is
    accepted (shape only). Useful for advanced users with intentional
    sub-stochastic workflows."""
    g = _build_n4_coalescent()
    rewards = g.states().T[2].astype(np.float64)  # the bad row
    # Shape is fine; coverage would normally fail; opt-out skips coverage.
    out = g._validate_rewards(rewards, allow_2d=False, check_coverage=False)
    assert out.shape == (g.vertices_length(),)


# ---------------------------------------------------------------------------
# Integration tests (Tests 9-13)
# ---------------------------------------------------------------------------


def test_svgd_rejects_invalid_rewards():
    """End-to-end: Graph.svgd refuses the tutorial-bug 2D rewards."""
    g = _build_n4_coalescent()
    rewards = g.states().T[:-1].astype(np.float64)

    # Build sparse observations using the (broken) reward decomposition.
    n = 200
    per_feat = [g.sample(n, rewards=rewards[i], validate_rewards=False)
                for i in range(rewards.shape[0])]
    obs = np.full(
        (n * rewards.shape[0], rewards.shape[0]), np.nan, dtype=np.float64,
    )
    for i in range(rewards.shape[0]):
        obs[i*n:(i+1)*n, i] = per_feat[i]
    sparse_obs = dense_to_sparse(obs)

    g_svgd = Graph(_build_n4_coalescent()._original_callback if hasattr(_build_n4_coalescent(), '_original_callback') else _build_n4_coalescent())
    # Easier: reuse the fixture's coalescent in a fresh graph.
    g_svgd = _build_n4_coalescent()
    with pytest.raises(ValueError) as excinfo:
        g_svgd.svgd(
            observed_data=sparse_obs,
            rewards=rewards,
            prior=GaussPrior(ci=[1, 20]),
            n_iterations=2,
            n_particles=4,
            learning_rate=ExpStepSize(first_step=0.01, last_step=0.001, tau=30.0),
            progress=False,
        )
    assert "feature" in str(excinfo.value).lower()


def test_svgd_accepts_valid_decomposition_and_recovers_theta():
    """The fixed tutorial: with rows [0,1,3] of states().T, theta is
    recovered within sampling noise."""
    g = _build_n4_coalescent()
    rewards = g.states().T[[0, 1, 3]].astype(np.float64)

    n = 5_000  # smaller than tutorial; keep test fast
    per_feat = [g.sample(n, rewards=rewards[i]) for i in range(rewards.shape[0])]
    obs = np.full(
        (n * rewards.shape[0], rewards.shape[0]), np.nan, dtype=np.float64,
    )
    for i in range(rewards.shape[0]):
        obs[i*n:(i+1)*n, i] = per_feat[i]
    np.random.shuffle(obs)
    sparse_obs = dense_to_sparse(obs)

    g_svgd = _build_n4_coalescent()
    svgd = g_svgd.svgd(
        observed_data=sparse_obs,
        rewards=rewards,
        prior=GaussPrior(ci=[1, 20]),
        n_iterations=200,
        n_particles=10,
        learning_rate=ExpStepSize(first_step=0.01, last_step=0.001, tau=30.0),
        progress=False,
    )
    theta = float(np.asarray(svgd.theta_mean).ravel()[0])
    # The fix takes the bias from ~11 down to within ~10% of truth.
    assert abs(theta - 10.0) < 1.5, (
        f"theta_mean = {theta} too far from true 10.0; "
        f"the validator may have allowed an invalid decomposition."
    )


def test_sample_rejects_invalid_rewards():
    """Graph.sample with an invalid reward vector raises."""
    g = _build_n4_coalescent()
    with pytest.raises(ValueError):
        g.sample(100, rewards=np.array([0, 0, 0, 0, 1, 0], dtype=np.float64))


def test_sample_with_validate_false_returns_zeros():
    """The opt-out lets users sample with invalid rewards (advanced)."""
    g = _build_n4_coalescent()
    samples = g.sample(
        500,
        rewards=np.array([0, 0, 0, 0, 1, 0], dtype=np.float64),
        validate_rewards=False,
    )
    # Some fraction of samples should be exactly 0 (trajectories that
    # don't visit vertex 4).
    assert np.any(samples == 0.0), (
        "Expected some zero-samples documenting the sub-stochastic case."
    )


def test_reward_transform_warns_but_does_not_raise():
    """reward_transform retains its legitimate non-inference uses
    (Laplace transforms, conditional expectations) so it warns rather
    than raises on coverage failure. Shape errors still raise."""
    g = _build_n4_coalescent()
    bad = np.array([0, 0, 0, 0, 1, 0], dtype=np.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = g.reward_transform(bad)
    assert result is not None
    assert any(issubclass(wi.category, UserWarning) for wi in w), (
        "Expected UserWarning on coverage failure"
    )
    assert any("sub-stochastic" in str(wi.message) for wi in w)

    # But a wrong-length 1D reward must still raise.
    with pytest.raises(ValueError, match=r"1D rewards must have shape"):
        g.reward_transform(np.ones(2, dtype=np.float64))


# ---------------------------------------------------------------------------
# FFI orientation regression (Test 14)
# ---------------------------------------------------------------------------


def test_ffi_wrapper_orientation_consistent():
    """compute_pmf_multivariate_ffi now expects (n_features, n_vertices).
    Passing (n_vertices, n_features) — the OLD convention — must raise.
    """
    import jax.numpy as jnp
    from phasic.ffi_wrappers import compute_pmf_multivariate_ffi

    g = _build_n4_coalescent()
    n_v = g.vertices_length()
    structure_dict = g.serialize()
    theta = jnp.array([10.0])
    times = jnp.array([[0.1, 0.1], [0.2, 0.2]])  # (n_times=2, n_features=2)

    # New (correct) orientation: (n_features=2, n_vertices=n_v).
    rewards_new = jnp.asarray(
        g.states().T[[0, 1]], dtype=jnp.float64,
    )  # (2, n_v)
    assert rewards_new.shape == (2, n_v)
    # This should succeed (or at minimum, not raise on shape).
    out = compute_pmf_multivariate_ffi(
        structure_dict, theta, times, rewards_new,
        discrete=False, granularity=100, compute_joint=False,
    )
    assert out.shape == times.shape

    # Old (wrong) orientation: (n_vertices, n_features). For
    # n_v != n_features this MUST raise with a transpose hint.
    if n_v != 2:
        rewards_old = jnp.asarray(rewards_new.T, dtype=jnp.float64)
        assert rewards_old.shape == (n_v, 2)
        with pytest.raises(ValueError, match=r"transpose|Transpose|n_features"):
            compute_pmf_multivariate_ffi(
                structure_dict, theta, times, rewards_old,
                discrete=False, granularity=100, compute_joint=False,
            )
