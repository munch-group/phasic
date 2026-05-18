"""WP-8 tests: SCC composer telemetry.

The composer maintains process-wide counters of:
  - cache_hits / cache_misses for the per-SCC PRC disk cache
  - compose_calls for ptd_compose_scc_prcs invocations
  - total_compose_ns for cumulative wall time

This suite verifies the counters behave as documented.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pytest

import phasic.cache as cache
from toy_model import BUILDERS, build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _force_cache_all_sccs(monkeypatch):
    """This file tests cache mechanics, not the size threshold.
    Force PHASIC_MIN_SCC_SIZE_TO_CACHE=0 so toy_b's small SCCs
    (some have synth size < 4) participate in the cache and
    contribute to cache_hits/cache_misses."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


def _clear_disk_cache():
    """Wipe ~/.phasic_cache/parameterized_reward_compute/ between tests
    that need controlled cache state."""
    cache_dir = os.path.expanduser("~/.phasic_cache/parameterized_reward_compute")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def test_stats_get_returns_dict_with_required_keys():
    """scc_compose_stats() returns a dict with the documented keys."""
    cache.reset_scc_compose_stats()
    stats = cache.scc_compose_stats()
    assert isinstance(stats, dict)
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "compose_calls" in stats
    assert "total_compose_ns" in stats


def test_stats_reset_zeroes_all_counters():
    """After reset, all counters are 0."""
    cache.reset_scc_compose_stats()
    stats = cache.scc_compose_stats()
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0
    assert stats["compose_calls"] == 0
    assert stats["total_compose_ns"] == 0


def test_compose_call_bumps_compose_calls(monkeypatch):
    """Calling Graph.expected_waiting_time under
    PHASIC_HIERAR_ELIMINATION=1 bumps compose_calls by 1."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 1


def test_compose_call_bumps_total_compose_ns(monkeypatch):
    """compose_calls > 0 implies total_compose_ns > 0
    (compose work always takes some time)."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 1
    assert stats["total_compose_ns"] > 0


def test_no_compose_when_env_var_off(monkeypatch):
    """Without PHASIC_HIERAR_ELIMINATION the monolithic path is
    taken — compose_calls stays at 0."""
    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 0


def test_first_compose_records_only_misses(monkeypatch):
    """With cleared disk cache, the first compose call records
    only misses (one per SCC, no hits)."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    _clear_disk_cache()
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] > 0
    # Toy-B has 5 SCCs; each one misses on the first build.
    assert stats["cache_misses"] == 5


def test_repeat_compose_records_only_hits(monkeypatch):
    """Once the disk cache is populated, subsequent composes for
    the same graph record only hits (no misses)."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    _clear_disk_cache()

    # Warm-up (populates disk cache).
    g1 = build_toy_b()
    g1.update_weights(THETA)
    _ = g1.expected_waiting_time()

    # Reset and re-run.
    cache.reset_scc_compose_stats()
    g2 = build_toy_b()
    g2.update_weights(THETA)
    _ = g2.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["cache_misses"] == 0
    assert stats["cache_hits"] > 0


def test_hits_plus_misses_equals_n_sccs_per_call(monkeypatch):
    """For toy_b (5 SCCs), each compose call records exactly 5
    hit-or-miss events."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    _clear_disk_cache()
    cache.reset_scc_compose_stats()

    # First compose: 5 misses.
    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()
    s1 = cache.scc_compose_stats()
    assert s1["cache_hits"] + s1["cache_misses"] == 5

    # Second compose: 5 hits.
    g2 = build_toy_b()
    g2.update_weights(THETA)
    _ = g2.expected_waiting_time()
    s2 = cache.scc_compose_stats()
    assert s2["cache_hits"] + s2["cache_misses"] == 10  # cumulative


def test_disabled_cache_records_no_hits_or_misses(monkeypatch):
    """When the reward-compute cache is disabled (the default
    policy; PHASIC_REWARD_COMPUTE_CACHE unset), the load/save path
    is skipped, so no hits or misses are recorded."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.delenv("PHASIC_REWARD_COMPUTE_CACHE", raising=False)
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0
    assert stats["compose_calls"] == 1


def test_counters_accumulate_across_calls(monkeypatch):
    """Counters accumulate across multiple compose calls without
    a reset in between."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    _clear_disk_cache()
    cache.reset_scc_compose_stats()

    for _ in range(3):
        g = build_toy_b()
        g.update_weights(THETA)
        _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 3
    # First call: 5 misses. Calls 2-3: 5 hits each.
    assert stats["cache_misses"] == 5
    assert stats["cache_hits"] == 10


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_compose_calls_counter_per_toy(toy_name, monkeypatch):
    """For every toy model, one expected_waiting_time call under
    the env var produces exactly one compose_calls bump."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    g = BUILDERS[toy_name]()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 1


def test_total_compose_ns_monotone_increasing(monkeypatch):
    """total_compose_ns is monotone-non-decreasing across calls."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    measurements = []
    for _ in range(3):
        g = build_toy_b()
        g.update_weights(THETA)
        _ = g.expected_waiting_time()
        measurements.append(cache.scc_compose_stats()["total_compose_ns"])

    assert measurements[0] <= measurements[1] <= measurements[2]
