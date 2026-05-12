"""WP-7 tests: PHASIC_HIERAR_ELIMINATION env var gates the
hierarchical pipeline behind ptd_expected_waiting_time.

When the env var is set to '1', user-facing methods like
Graph.expectation, Graph.variance, Graph.expected_waiting_time
take the hierarchical path (per-SCC compose) transparently.
Output must be numerically equivalent to the monolithic path
within machine precision.

When the env var is NOT set (default), the monolithic path is
used unchanged — all existing tests continue to pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from toy_model import BUILDERS


THETAS = [
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 2.0, 1.0, 0.7],
    [2.5, 0.3, 1.5, 0.9],
]


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
@pytest.mark.parametrize("theta_idx", list(range(len(THETAS))))
def test_expected_waiting_time_under_env_var(toy_name, theta_idx, monkeypatch):
    """Graph.expected_waiting_time produces same result with and
    without PHASIC_HIERAR_ELIMINATION=1."""
    builder = BUILDERS[toy_name]
    theta = THETAS[theta_idx]

    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    g_mono = builder()
    g_mono.update_weights(theta)
    mono = np.asarray(g_mono.expected_waiting_time())

    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    g_hier = builder()
    g_hier.update_weights(theta)
    hier = np.asarray(g_hier.expected_waiting_time())

    np.testing.assert_allclose(hier, mono, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_expectation_under_env_var(toy_name, monkeypatch):
    """Graph.expectation matches under both modes."""
    builder = BUILDERS[toy_name]
    theta = [1.0, 1.0, 1.0, 1.0]

    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    g_mono = builder()
    g_mono.update_weights(theta)
    e_mono = g_mono.expectation()

    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    g_hier = builder()
    g_hier.update_weights(theta)
    e_hier = g_hier.expectation()

    assert e_mono == pytest.approx(e_hier, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_variance_under_env_var(toy_name, monkeypatch):
    """Graph.variance matches under both modes. variance() reads
    expected_waiting_time twice (with reward vector for the second
    moment) — when rewards != None, the hierarchical path falls
    back to monolithic, so this exercises the fallback path."""
    builder = BUILDERS[toy_name]
    theta = [1.0, 1.0, 1.0, 1.0]

    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    g_mono = builder()
    g_mono.update_weights(theta)
    v_mono = g_mono.variance()

    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    g_hier = builder()
    g_hier.update_weights(theta)
    v_hier = g_hier.variance()

    assert v_mono == pytest.approx(v_hier, rel=1e-12, abs=1e-12)


def test_env_var_disabled_uses_monolithic(monkeypatch):
    """Without PHASIC_HIERAR_ELIMINATION, the monolithic path is
    used. We can't directly observe path choice from Python, but
    we can verify the result is correct."""
    from toy_model import build_toy_base
    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    g = build_toy_base()
    g.update_weights([1.0, 1.0, 1.0, 1.0])
    e = g.expectation()
    assert e == pytest.approx(2.32, rel=1e-12)
