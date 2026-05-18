"""WP-5 tests: SCC composition produces parent-equivalent results.

Verifies that ``SCCGraph.compose`` (wrapping the C
``ptd_compose_scc_prcs``) reproduces the monolithic
``Graph.expected_waiting_time`` to machine precision on every toy
variant at multiple theta vectors.

This is the gate test for the entire branch: numerical equivalence
between hierarchical (per-SCC PRC + composition) and monolithic
elimination on cyclic SCCs.

See wp5-investigation.md for the design (Option B with per-channel
absorbing + phantom-injection).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from toy_model import BUILDERS


# Fixed theta vectors used for equivalence tests.
THETAS = [
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 2.0, 1.0, 0.7],
    [2.5, 0.3, 1.5, 0.9],
    [0.1, 5.0, 2.0, 0.5],
]


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
@pytest.mark.parametrize("theta_idx", list(range(len(THETAS))))
def test_compose_matches_monolithic(toy_name, theta_idx, monkeypatch):
    """For every toy variant and theta, the composer must produce
    a result vector numerically equivalent to the monolithic
    ``expected_waiting_time``."""
    # Disable the disk cache to avoid stale entries from prior
    # test runs. WP-5 itself works with caching, but determinism
    # in tests is easier without it. Cache defaults to off; remove
    # any opt-in left by a prior test.
    monkeypatch.delenv("PHASIC_REWARD_COMPUTE_CACHE", raising=False)

    builder = BUILDERS[toy_name]
    theta = THETAS[theta_idx]

    # Monolithic reference.
    g_mono = builder()
    g_mono.update_weights(theta)
    mono = np.asarray(g_mono.expected_waiting_time())

    # Hierarchical composition.
    g_comp = builder()
    scc_decomp = g_comp.scc_decomposition()
    comp = np.asarray(scc_decomp.compose(g_comp, theta))

    assert mono.shape == comp.shape
    np.testing.assert_allclose(
        comp, mono, rtol=1e-12, atol=1e-12,
        err_msg=f"{toy_name} theta={theta}: hierarchical compose disagrees "
                f"with monolithic"
    )


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_compose_with_disk_cache(toy_name, monkeypatch):
    """The composer must work end-to-end with the disk cache
    enabled (opt in via PHASIC_REWARD_COMPUTE_CACHE=1), exercising
    the cache-write and cache-read code paths."""
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    builder = BUILDERS[toy_name]
    theta = [1.0, 1.0, 1.0, 1.0]

    g_mono = builder()
    g_mono.update_weights(theta)
    mono = np.asarray(g_mono.expected_waiting_time())

    g_comp = builder()
    scc_decomp = g_comp.scc_decomposition()
    comp = np.asarray(scc_decomp.compose(g_comp, theta))

    np.testing.assert_allclose(comp, mono, rtol=1e-12, atol=1e-12)


def test_compose_is_deterministic():
    """Multiple compose calls on the same graph at the same theta
    produce identical output."""
    from toy_model import build_toy_base
    g = build_toy_base()
    scc_decomp = g.scc_decomposition()
    theta = [1.0, 1.0, 1.0, 1.0]

    r1 = np.asarray(scc_decomp.compose(g, theta))
    r2 = np.asarray(scc_decomp.compose(g, theta))
    np.testing.assert_array_equal(r1, r2)


def test_compose_varies_with_theta():
    """Compose results change when theta changes (sanity)."""
    from toy_model import build_toy_base
    g = build_toy_base()
    scc_decomp = g.scc_decomposition()

    r1 = np.asarray(scc_decomp.compose(g, [1.0, 1.0, 1.0, 1.0]))
    r2 = np.asarray(scc_decomp.compose(g, [0.5, 2.0, 1.0, 0.7]))
    assert not np.allclose(r1, r2)
