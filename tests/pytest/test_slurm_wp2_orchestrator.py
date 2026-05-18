"""SLURM-WP-2: orchestrator level-set computation.

Verifies the Python helpers an orchestrator uses to plan
distributed SCC work:

  - compute_scc_levels: groups SCCs into independence-classes
    (sink-first BFS layers in the condensation).
  - find_missing_sccs: identifies which SCCs need (re)computation
    based on cache file presence.
  - plan_distributed_work: combines the two into a per-level
    work plan.
  - scc_cache_path_for_synth: matches the C composer's cache
    path so orchestrator and worker agree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import phasic.cache as cache
from phasic import distributed_scc
from toy_model import BUILDERS, build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _force_cache_all_sccs(monkeypatch):
    """Override the size threshold so every SCC participates in
    the cache; orchestrator tests check missing/cached SCC
    counts and would otherwise depend on toy_b SCC sizes."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


def test_compute_levels_partitions_all_sccs():
    """Every SCC index appears exactly once across all levels."""
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    n = len(scc_decomp)

    levels = distributed_scc.compute_scc_levels(scc_decomp)
    flat = [i for lvl in levels for i in lvl]
    assert sorted(flat) == list(range(n))


def test_compute_levels_respects_dependency_order():
    """If SCC i has an edge to SCC j, then level(j) < level(i).

    (Sink-first: lower level = closer to sink = computed first.)
    """
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()

    levels = distributed_scc.compute_scc_levels(scc_decomp)
    level_of = {}
    for l, lvl in enumerate(levels):
        for i in lvl:
            level_of[i] = l

    n = len(scc_decomp)
    for i in range(n):
        scc = scc_decomp.scc_at(i)
        for j in scc.outgoing_scc_edges():
            assert level_of[j] < level_of[i], (
                f"SCC {i} (level {level_of[i]}) has edge to "
                f"SCC {j} (level {level_of[j]}); expected j to be "
                f"at a strictly lower level (closer to sinks).")


def test_compute_levels_level_zero_are_sinks():
    """Level 0 contains exactly the SCCs with no outgoing edges."""
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    levels = distributed_scc.compute_scc_levels(scc_decomp)

    level_zero = set(levels[0])
    sinks = {i for i in range(len(scc_decomp))
             if len(scc_decomp.scc_at(i).outgoing_scc_edges()) == 0}
    assert level_zero == sinks


@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_compute_levels_works_for_every_toy(toy_name):
    """Smoke: the level computation succeeds and partitions
    correctly for every toy model."""
    g = BUILDERS[toy_name]()
    scc_decomp = g.scc_decomposition()
    n = len(scc_decomp)

    levels = distributed_scc.compute_scc_levels(scc_decomp)
    flat = [i for lvl in levels for i in lvl]
    assert sorted(flat) == list(range(n))


def test_cache_path_matches_what_c_writes(tmp_path, monkeypatch):
    """scc_cache_path_for_synth predicts the exact path the C
    composer writes when caching a synth's PRC."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc = g.scc_decomposition().scc_at(1)
    synth, _ = scc.as_synthetic_graph()

    predicted = distributed_scc.scc_cache_path_for_synth(synth)

    distributed_scc.cache_synth_prc(synth)

    assert os.path.exists(predicted), (
        f"Predicted path {predicted} does not exist after compute. "
        f"Available files: "
        f"{list((tmp_path / 'parameterized_reward_compute').glob('*'))}")


def test_find_missing_when_cache_empty(tmp_path, monkeypatch):
    """All SCCs missing when cache is empty."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()

    missing = distributed_scc.find_missing_sccs(scc_decomp)
    assert sorted(missing) == list(range(len(scc_decomp)))


def test_find_missing_after_full_population(tmp_path, monkeypatch):
    """No SCCs missing after running hierarchical compose to
    populate the cache."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    scc_decomp = g.scc_decomposition()
    missing = distributed_scc.find_missing_sccs(scc_decomp)
    assert missing == []


def test_find_missing_partial(tmp_path, monkeypatch):
    """Compute one specific SCC, others remain missing."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()

    # Cache exactly SCC 0.
    scc0 = scc_decomp.scc_at(0)
    synth0, _ = scc0.as_synthetic_graph()
    distributed_scc.cache_synth_prc(synth0)

    missing = distributed_scc.find_missing_sccs(scc_decomp)
    n = len(scc_decomp)
    expected = [i for i in range(n) if i != 0]
    assert sorted(missing) == expected


def test_plan_only_missing_drops_cached(tmp_path, monkeypatch):
    """plan_distributed_work(only_missing=True) returns levels
    with cached SCCs filtered out."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()

    # Populate one SCC.
    scc1 = scc_decomp.scc_at(1)
    synth1, _ = scc1.as_synthetic_graph()
    distributed_scc.cache_synth_prc(synth1)

    plan = distributed_scc.plan_distributed_work(
            scc_decomp, only_missing=True)
    flat = [i for lvl in plan for i in lvl]
    assert 1 not in flat, (
        f"Cached SCC 1 should be filtered from plan; got {plan}")


def test_plan_no_filter_returns_all(tmp_path, monkeypatch):
    """only_missing=False returns all levels regardless of cache."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc_decomp = g.scc_decomposition()

    plan_all = distributed_scc.plan_distributed_work(
            scc_decomp, only_missing=False)
    flat = [i for lvl in plan_all for i in lvl]
    assert sorted(flat) == list(range(len(scc_decomp)))


def test_plan_when_fully_cached_is_empty(tmp_path, monkeypatch):
    """After populating all SCCs, the plan has no work units."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    scc_decomp = g.scc_decomposition()
    plan = distributed_scc.plan_distributed_work(
            scc_decomp, only_missing=True)
    flat = [i for lvl in plan for i in lvl]
    assert flat == []
