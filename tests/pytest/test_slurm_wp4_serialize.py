"""SLURM-WP-4: SCC subgraph serialisation.

Verifies that:
  - serialize_scc_synth produces a JSON-friendly dict
  - deserialize_scc_synth round-trips to a structurally
    equivalent graph
  - cache_synth_prc populates the cache for a deserialised synth
  - The cache entry produced by the worker path is hit by the
    orchestrator when it later runs hierarchical compose
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import phasic.cache as cache
from phasic import distributed_scc
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _force_cache_all_sccs(monkeypatch):
    """Override the size threshold so every SCC participates in
    the cache; tests here validate the worker-orchestrator
    cache hand-off, not the threshold."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


def test_serialize_returns_json_friendly_dict():
    """Serialised work unit contains only plain Python types
    that survive json.dumps()."""
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    scc0 = scc_decomp.scc_at(0)

    data = distributed_scc.serialize_scc_synth(scc0)
    # Must round-trip through JSON unchanged.
    s = json.dumps(data)
    parsed = json.loads(s)
    assert parsed["scc_index"] == int(scc0.index())
    assert "graph" in parsed
    assert "states" in parsed["graph"]
    assert isinstance(parsed["graph"]["states"], list)


def test_deserialize_reconstructs_synth():
    """deserialize_scc_synth produces a graph with the same
    vertex count as the original synth."""
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    scc0 = scc_decomp.scc_at(0)
    synth_orig, _ = scc0.as_synthetic_graph()

    data = distributed_scc.serialize_scc_synth(scc0)
    json_str = json.dumps(data)
    parsed = json.loads(json_str)

    synth_reconstructed = distributed_scc.deserialize_scc_synth(parsed)
    assert synth_reconstructed.vertices_length() == synth_orig.vertices_length()


@pytest.mark.parametrize("scc_idx", list(range(5)))  # toy_b has 5 SCCs
def test_round_trip_all_sccs_in_toy_b(scc_idx):
    """Every SCC in toy_b round-trips serialisation correctly."""
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    scc = scc_decomp.scc_at(scc_idx)
    synth_orig, _ = scc.as_synthetic_graph()

    data = distributed_scc.serialize_scc_synth(scc)
    json_str = json.dumps(data)
    parsed = json.loads(json_str)

    synth_recon = distributed_scc.deserialize_scc_synth(parsed)
    assert synth_recon.vertices_length() == synth_orig.vertices_length()


def test_cache_synth_prc_writes_cache_file(tmp_path, monkeypatch):
    """cache_synth_prc creates a cache file for the synth's
    content hash."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    scc = g.scc_decomposition().scc_at(1)
    synth, _ = scc.as_synthetic_graph()

    cache_subdir = tmp_path / "parameterized_reward_compute"
    before = list(cache_subdir.glob("scc_*.bin")) if cache_subdir.exists() else []

    distributed_scc.cache_synth_prc(synth)

    after = list(cache_subdir.glob("scc_*.bin"))
    assert len(after) > len(before)


def test_worker_cache_hit_at_orchestrator(tmp_path, monkeypatch):
    """End-to-end: worker pre-populates the cache for one SCC,
    then orchestrator's hierarchical compose hits that entry."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    # Phase 1: orchestrator-side serialisation.
    g_orch = build_toy_b()
    scc_decomp = g_orch.scc_decomposition()
    n_sccs = len(scc_decomp)

    work_units = []
    for i in range(n_sccs):
        scc = scc_decomp.scc_at(i)
        wu_path = tmp_path / f"work_unit_{i}.json"
        distributed_scc.write_work_unit(scc, str(wu_path))
        work_units.append(str(wu_path))

    # Phase 2: workers process the units (simulated locally).
    cache.reset_scc_compose_stats()
    for wu_path in work_units:
        data = distributed_scc.read_work_unit(wu_path)
        synth = distributed_scc.deserialize_scc_synth(data)
        distributed_scc.cache_synth_prc(synth)

    worker_stats = cache.scc_compose_stats()
    assert worker_stats["cache_misses"] == n_sccs
    assert worker_stats["cache_hits"] == 0

    # Phase 3: orchestrator runs hierarchical compose. All SCCs
    # should hit the cache populated by workers.
    cache.reset_scc_compose_stats()
    g_run = build_toy_b()
    g_run.update_weights(THETA)
    result = g_run.expected_waiting_time()

    final_stats = cache.scc_compose_stats()
    assert final_stats["cache_hits"] == n_sccs
    assert final_stats["cache_misses"] == 0
    assert result[0] == pytest.approx(1.88, rel=1e-12)


def test_write_and_read_work_unit_round_trip(tmp_path):
    """write_work_unit followed by read_work_unit yields the
    same dict."""
    g = build_toy_b()
    scc = g.scc_decomposition().scc_at(0)

    wu_path = tmp_path / "wu.json"
    distributed_scc.write_work_unit(scc, str(wu_path))

    assert wu_path.exists()
    data = distributed_scc.read_work_unit(str(wu_path))
    assert "graph" in data
    assert "scc_index" in data
