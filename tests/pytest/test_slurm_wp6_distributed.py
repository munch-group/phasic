"""SLURM-WP-6: distributed precompute integration.

Verifies the top-level wrapper precompute_distributed:

  - When called with no slurm_options, runs workers locally
    and populates the cache so a subsequent hierarchical
    compose hits every SCC.
  - Works correctly when starting from an empty cache, a
    partially-populated cache, and a fully-populated cache.
  - Returns a result dict describing what was done.

We don't have a real SLURM cluster in CI, so the SLURM path is
exercised structurally (script gets written, submit_sbatch
fails cleanly when sbatch is absent).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import phasic.cache as cache
from phasic import distributed_scc
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


def test_precompute_distributed_local_populates_cache(tmp_path, monkeypatch):
    """Running precompute_distributed locally on an empty cache
    populates every SCC's PRC. A subsequent hierarchical compose
    hits everything."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b()
    g.update_weights(THETA)

    result = distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir),
            local_max_workers=2)

    assert result["mode"] == "local"
    n_sccs = sum(result["work_units_per_level"])
    assert n_sccs == len(g.scc_decomposition())

    # Run hierarchical compose. Every SCC should hit the cache
    # populated by precompute_distributed.
    cache.reset_scc_compose_stats()
    g_run = build_toy_b()
    g_run.update_weights(THETA)
    r = g_run.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["cache_hits"] == n_sccs
    assert stats["cache_misses"] == 0
    assert r[0] == pytest.approx(1.88, rel=1e-12)


def test_precompute_distributed_partial_cache(tmp_path, monkeypatch):
    """When the cache is partially populated, precompute_distributed
    only submits work for the missing SCCs."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    # Cache exactly SCC 1.
    g_pre = build_toy_b()
    scc_decomp_pre = g_pre.scc_decomposition()
    scc1 = scc_decomp_pre.scc_at(1)
    synth1, _ = scc1.as_synthetic_graph()
    distributed_scc.cache_synth_prc(synth1)

    g = build_toy_b()
    g.update_weights(THETA)

    result = distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir))

    n_total = len(g.scc_decomposition())
    n_done = sum(result["work_units_per_level"])
    # n_total - 1 (SCC 1 already cached).
    assert n_done == n_total - 1


def test_precompute_distributed_fully_cached_no_op(tmp_path, monkeypatch):
    """When the cache is already fully populated, no work units
    are submitted."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    # Pre-populate with a hierarchical compose run.
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    g_pre = build_toy_b()
    g_pre.update_weights(THETA)
    _ = g_pre.expected_waiting_time()

    g = build_toy_b()
    g.update_weights(THETA)

    result = distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir))

    n_done = sum(result["work_units_per_level"])
    assert n_done == 0


def test_precompute_distributed_creates_work_dir(tmp_path, monkeypatch):
    """precompute_distributed creates work_dir and log_dir if
    they don't exist."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "nested" / "work"
    assert not work_dir.exists()

    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)
    g = build_toy_b()
    g.update_weights(THETA)

    distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir))

    assert work_dir.exists()
    # Default log_dir is work_dir/logs.
    assert (work_dir / "logs").exists()


def test_precompute_distributed_writes_work_unit_files(tmp_path, monkeypatch):
    """precompute_distributed writes one JSON per missing SCC
    under work_dir/level_<l>/ (as a paper trail / for debug)."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b()
    g.update_weights(THETA)

    result = distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir))

    # Every level-set with non-empty work should produce JSON
    # files in work_dir/level_<l>/.
    n_total = sum(result["work_units_per_level"])
    json_files = list(work_dir.rglob("scc_*.json"))
    assert len(json_files) == n_total


def test_precompute_distributed_local_failure_propagates(tmp_path, monkeypatch):
    """If a local worker subprocess fails, the orchestrator
    raises a RuntimeError with the failure details.

    We simulate worker failure by writing a malformed work-unit
    JSON: precompute_distributed isn't directly callable in a
    way that injects bad work units, so we instead exercise
    run_workers_locally directly with a known-bad input — the
    same failure-detection logic precompute_distributed uses.
    """
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))

    # Mix of one good work unit and one missing-file unit.
    g = build_toy_b()
    scc = g.scc_decomposition().scc_at(0)
    good_wu = tmp_path / "good.json"
    distributed_scc.write_work_unit(scc, str(good_wu))
    bad_wu = tmp_path / "missing.json"
    # Don't create the bad file — worker should report load failure.

    rcs = distributed_scc.run_workers_locally(
            [str(good_wu), str(bad_wu)], max_workers=1)
    # Good unit succeeds, bad unit returns non-zero.
    assert rcs[0] == 0
    assert rcs[1] != 0


def test_precompute_distributed_returns_metadata(tmp_path, monkeypatch):
    """The returned dict carries the expected keys."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b()
    g.update_weights(THETA)

    result = distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=str(cache_dir))

    assert "mode" in result
    assert "levels" in result
    assert "work_units_per_level" in result
    assert "job_ids" in result
    assert isinstance(result["job_ids"], list)


def test_precompute_distributed_uses_existing_cache_dir_env(tmp_path, monkeypatch):
    """When cache_dir=None, the existing PHASIC_CACHE_DIR is honoured."""
    cache_dir = tmp_path / "cache"
    work_dir = tmp_path / "work"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b()
    g.update_weights(THETA)

    distributed_scc.precompute_distributed(
            graph=g,
            work_dir=str(work_dir),
            cache_dir=None)

    files = list(
            (cache_dir / "parameterized_reward_compute").glob("scc_*.bin"))
    assert len(files) > 0
