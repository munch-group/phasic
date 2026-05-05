"""Stage B2 verification — disk-persistent symbolic elimination cache.

After Stage B2, ``ptd_precompute_reward_compute_graph`` consults
``~/.phasic_cache/parameterized_reward_compute/<hash>.bin`` before
running the O(n^3) Gaussian elimination, and writes back on a miss.
The cache is keyed by ``ptd_graph_content_hash`` (theta-independent),
so the same parameterised model always hits the same key.

These tests verify:

1. Correctness — cached path and fresh-elimination path produce
   bit-identical results.
2. Cross-process — a subprocess populates the cache, then a fresh
   subprocess loads it.
3. ``PHASIC_DISABLE_CACHE=1`` round-trips correctly (no read, no write).
4. Format-version mismatch falls back to rebuild without surfacing as
   a user-visible error.
5. The in-memory persistent graph (Stage A1) and the disk cache
   compose correctly: in-memory hits short-circuit before disk is
   consulted; disk hits prime the persistent graph for subsequent
   in-memory hits.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import phasic
from phasic import Graph

GraphBuilder = phasic.parameterized.GraphBuilder


def _build_chain_json(n_states):
    def cb(state, **kw):
        n = state[0]
        if n <= 1:
            return []
        return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

    g = Graph(cb, ipv=[n_states])
    sd = g.serialize()
    return json.dumps(
        {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sd.items()}
    )


def _cache_root():
    return Path(os.path.expanduser("~/.phasic_cache/parameterized_reward_compute"))


@pytest.fixture
def empty_cache(monkeypatch):
    """Run the test against a fresh, empty cache directory under a
    HOME pointing at a temporary directory. Restores HOME after the
    test."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("HOME", tmp)
        yield Path(tmp) / ".phasic_cache" / "parameterized_reward_compute"


# ---------------------------------------------------------------------------
# Correctness: cached vs fresh
# ---------------------------------------------------------------------------


def test_cached_and_fresh_produce_identical_results(empty_cache, monkeypatch):
    """Run the same compute twice in the same process: first time
    populates the cache, second time hits the in-memory persistent
    graph (Stage A1). Both must produce identical results, and a
    third call after disabling the cache must also match."""
    structure_json = _build_chain_json(40)
    theta = np.array([1.5])

    b1 = GraphBuilder(structure_json)
    m1 = b1.compute_moments(theta, 3)

    # Cache should now contain one file.
    files = list(empty_cache.glob("*.bin")) if empty_cache.exists() else []
    assert len(files) == 1, f"expected 1 cache file, got {files}"

    # Fresh builder, same structure → cache hit.
    b2 = GraphBuilder(structure_json)
    m2 = b2.compute_moments(theta, 3)
    np.testing.assert_allclose(m2, m1, rtol=1e-12)

    # Fresh builder with cache disabled → fresh elimination.
    monkeypatch.setenv("PHASIC_DISABLE_CACHE", "1")
    b3 = GraphBuilder(structure_json)
    m3 = b3.compute_moments(theta, 3)
    np.testing.assert_allclose(m3, m1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cross-process round-trip
# ---------------------------------------------------------------------------


_SUBPROCESS_SCRIPT = """
import json, sys, numpy as np, phasic
from phasic import Graph
GraphBuilder = phasic.parameterized.GraphBuilder

def cb(state, **kw):
    n = state[0]
    if n <= 1: return []
    return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

g = Graph(cb, ipv=[40])
sd = g.serialize()
sj = json.dumps({k: (v.tolist() if hasattr(v,'tolist') else v) for k,v in sd.items()})
b = GraphBuilder(sj)
m = b.compute_moments(np.array([2.0]), 2)
print(json.dumps(list(m)))
"""


def _run_subprocess(env_overrides):
    """Run _SUBPROCESS_SCRIPT in a fresh subprocess with given env
    overrides. Returns the parsed moments output. The subprocess
    prints the JSON on its last stdout line; earlier lines may be
    noise from pixi/jax/matplotlib."""
    env = os.environ.copy()
    env.update(env_overrides)
    res = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        env=env, capture_output=True, text=True, check=True,
    )
    # Take the last non-empty line that parses as JSON.
    for line in reversed(res.stdout.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            return np.array(json.loads(line))
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        f"subprocess produced no JSON output. stdout={res.stdout!r} "
        f"stderr={res.stderr!r}")


def test_cross_process_cache_round_trip(empty_cache):
    """Process A populates the cache; process B loads it. Both must
    produce identical results. Process B's runtime should not include
    the elimination cost — but we don't time-assert here (timing on
    small graphs is too noisy); we only check correctness."""
    # The empty_cache fixture sets HOME via monkeypatch on the parent
    # process; subprocess inherits via os.environ.copy(). The cache
    # path therefore lives under <HOME>/.phasic_cache/parameterized_reward_compute/.
    home = os.environ.get("HOME")
    expected_cache_dir = Path(home) / ".phasic_cache" / "parameterized_reward_compute"

    # Process A: cold cache, populates it.
    m_a = _run_subprocess({})  # inherit HOME from parent
    files_after_a = list(expected_cache_dir.glob("*.bin")) \
            if expected_cache_dir.exists() else []
    assert len(files_after_a) == 1, (
        f"Process A should have written one cache file at {expected_cache_dir}; "
        f"got {files_after_a}. Cache root contents: "
        f"{list(expected_cache_dir.parent.iterdir()) if expected_cache_dir.parent.exists() else 'parent does not exist'}"
    )

    # Process B: warm cache, should load.
    m_b = _run_subprocess({})
    np.testing.assert_allclose(m_b, m_a, rtol=1e-12)


# ---------------------------------------------------------------------------
# PHASIC_DISABLE_CACHE
# ---------------------------------------------------------------------------


def test_disable_cache_skips_writes(empty_cache, monkeypatch):
    monkeypatch.setenv("PHASIC_DISABLE_CACHE", "1")
    structure_json = _build_chain_json(20)
    b = GraphBuilder(structure_json)
    b.compute_moments(np.array([1.0]), 2)

    # Cache directory may or may not exist; either way, no .bin file.
    if empty_cache.exists():
        assert not list(empty_cache.glob("*.bin"))


def test_disable_cache_skips_reads(empty_cache, monkeypatch):
    """Pre-populate the cache, then run with the env var set; verify
    that we still get correct results (the load is skipped, so we go
    through fresh elimination)."""
    structure_json = _build_chain_json(20)

    # Populate
    b1 = GraphBuilder(structure_json)
    m1 = b1.compute_moments(np.array([1.0]), 2)
    files = list(empty_cache.glob("*.bin")) if empty_cache.exists() else []
    assert len(files) == 1

    # Now set the disable flag and verify correctness still holds.
    monkeypatch.setenv("PHASIC_DISABLE_CACHE", "1")
    b2 = GraphBuilder(structure_json)
    m2 = b2.compute_moments(np.array([1.0]), 2)
    np.testing.assert_allclose(m2, m1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Format-version mismatch
# ---------------------------------------------------------------------------


def test_corrupt_cache_falls_back_to_rebuild(empty_cache):
    """Write a malformed file at the expected cache path; the load
    should fail, the elimination should rebuild, and the file should
    be overwritten with a valid one. End result: correct moments."""
    structure_json = _build_chain_json(20)
    # Run once to discover where the file lands, then corrupt it.
    b = GraphBuilder(structure_json)
    b.compute_moments(np.array([1.0]), 2)
    files = list(empty_cache.glob("*.bin"))
    assert len(files) == 1
    cache_path = files[0]

    # Overwrite with garbage.
    cache_path.write_bytes(b"GARBAGE_NOT_PHASIC" + b"\x00" * 200)

    # New compute should detect corrupt magic, rebuild, and overwrite.
    b2 = GraphBuilder(structure_json)
    m2 = b2.compute_moments(np.array([2.0]), 2)
    assert np.all(np.isfinite(m2))

    # File should now be a valid cache again (size > a few bytes).
    assert cache_path.stat().st_size > 100


# ---------------------------------------------------------------------------
# Compose: in-memory persistent graph + disk cache
# ---------------------------------------------------------------------------


def test_in_memory_and_disk_cache_compose(empty_cache):
    """First call: cache miss → elimination + save + in-memory cache.
    Second call same builder: in-memory hit (Stage A1).
    Third call new builder same JSON: in-memory miss, disk hit, then
    in-memory cache.
    All three must produce identical results across many thetas."""
    structure_json = _build_chain_json(30)

    rng = np.random.default_rng(0)
    thetas = rng.uniform(0.5, 3.0, size=(5, 1))

    b1 = GraphBuilder(structure_json)
    results = []
    for theta in thetas:
        results.append(b1.compute_moments(theta, 2))

    # Fresh builder; should hit disk on first call, then in-memory.
    b2 = GraphBuilder(structure_json)
    for i, theta in enumerate(thetas):
        m = b2.compute_moments(theta, 2)
        np.testing.assert_allclose(m, results[i], rtol=1e-12)
