"""Tests for Graph.pull_cache() / Graph.push_cache().

Exercises the compute-sharing system end-to-end against a temp
local mock registry (file:// URLs). No network access. No git, no
gh, no IPFS daemon required.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import phasic
import phasic.compute_repository as cr
from phasic.compute_repository import (
    ComputeRegistry,
    _graph_hash_hex,
    _param_compute_cache_dir,
)
from phasic.exceptions import PTDBackendError, PTDFormatError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Defend against test-isolation leakage: other test files
    (notably test_config_scc_controls.py) leave PHASIC_DISABLE_CACHE
    or similar set on ``os.environ`` directly, which would cause the
    C path to refuse to write a .bin and the fixture's
    ``assert local_bin.exists()`` to fail. Clear them before each
    test in this file."""
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    monkeypatch.delenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", raising=False)
    monkeypatch.delenv("PHASIC_MAX_PARALLEL_SCCS", raising=False)
    # PHASIC_CACHE_DIR is set per-test via fixture; clear any leak.
    monkeypatch.delenv("PHASIC_CACHE_DIR", raising=False)
    # Also reset any cached phasic config object so it doesn't
    # remember stale fields.
    try:
        from phasic.config import reset_config, _phasic_assigned_env
        reset_config()
        _phasic_assigned_env.clear()
    except ImportError:
        pass
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _two_step_chain(theta: float = 2.0) -> phasic.Graph:
    """Build a small parameterised graph used across these tests."""
    g = phasic.Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0])
    v1.add_edge(v2, [1.0])
    g.update_weights([theta])
    return g


@pytest.fixture
def temp_registry(tmp_path: Path):
    """Yield a (ComputeRegistry, hash_hex, local_bin_path) triple for a
    published demo graph, with a fresh tmp_path-rooted registry+cache."""
    # Stand up a fake registry directory.
    fake_origin = tmp_path / "origin"; fake_origin.mkdir()
    reg_cache   = tmp_path / "registry_cache"; reg_cache.mkdir()
    phasic_cache = tmp_path / "phasic_cache"
    phasic_cache.mkdir()
    monkeypath = phasic_cache / "parameterized_reward_compute"
    monkeypath.mkdir()

    # Use PHASIC_CACHE_DIR so the C path and the Python helper agree.
    import os
    old_cache = os.environ.get("PHASIC_CACHE_DIR")
    os.environ["PHASIC_CACHE_DIR"] = str(phasic_cache)

    try:
        # Build a publisher graph, run elimination, snapshot the .bin out.
        g = _two_step_chain()
        g.expectation()
        hash_hex = _graph_hash_hex(g)
        local_bin = _param_compute_cache_dir() / f"{hash_hex}.bin"
        assert local_bin.exists(), "expectation() should have populated cache"

        published = fake_origin / f"{hash_hex}.bin"
        shutil.copy2(local_bin, published)
        sha = hashlib.sha256(published.read_bytes()).hexdigest()

        registry_data = {
            "version": "2.0", "format": "ptd_pcg",
            "computes": {
                "demo": {
                    "graph_hash":      hash_hex,
                    "format_revision": 2,
                    "artifacts": {
                        "parent": {
                            "cid_or_path": f"file://{published}",
                            "sha256":      sha,
                            "size_bytes":  published.stat().st_size,
                        },
                        "scc": [],
                    },
                    "metadata": {
                        "description": "demo",
                        "domain":      "demo",
                        "model_type":  "chain",
                        "vertices":    g.vertices_length(),
                        "param_length": 1,
                    },
                },
            },
        }
        (reg_cache / "registry.json").write_text(json.dumps(registry_data))

        registry = ComputeRegistry(
            registry_repo="demo/demo",
            cache_dir=reg_cache,
            auto_update=False,
        )
        # Install as the process-wide default so Graph.pull_cache uses it.
        cr._default_registry = registry

        yield registry, hash_hex, local_bin
    finally:
        cr._default_registry = None
        if old_cache is None:
            os.environ.pop("PHASIC_CACHE_DIR", None)
        else:
            os.environ["PHASIC_CACHE_DIR"] = old_cache


# ---------------------------------------------------------------------------
# Pull tests
# ---------------------------------------------------------------------------


def test_pull_cache_hit_returns_true(temp_registry):
    registry, hash_hex, local_bin = temp_registry
    # Wipe the local cache; pull should re-download from file://.
    if local_bin.exists():
        local_bin.unlink()
    g = _two_step_chain()
    assert g.pull_cache() is True
    assert local_bin.exists()


def test_pull_cache_hit_reuse_via_expectation(temp_registry):
    registry, hash_hex, local_bin = temp_registry
    if local_bin.exists():
        local_bin.unlink()
    g = _two_step_chain()
    assert g.pull_cache() is True
    # The C path picks up the file transparently.
    assert abs(g.expectation() - 0.5) < 1e-10


def test_pull_cache_miss_returns_false(temp_registry):
    """A graph not in the registry returns False, no download, no error."""
    registry, *_ = temp_registry
    # Build a different graph (3 hops instead of 2).
    g = phasic.Graph(1)
    v0 = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    v0.add_edge(v1, [1.0])
    v1.add_edge(v2, [1.0])
    v2.add_edge(v3, [1.0])
    g.update_weights([2.0])
    assert g.pull_cache() is False


def test_pull_cache_already_present_returns_true_no_download(temp_registry):
    """If the local cache already has the file, pull is a no-op hit."""
    registry, hash_hex, local_bin = temp_registry
    assert local_bin.exists()  # left in place by the fixture
    mtime_before = local_bin.stat().st_mtime
    g = _two_step_chain()
    assert g.pull_cache() is True
    assert local_bin.stat().st_mtime == mtime_before  # no re-download


def test_pull_cache_force_redownloads(temp_registry):
    """force=True re-downloads even when local file exists."""
    registry, hash_hex, local_bin = temp_registry
    assert local_bin.exists()
    # Touch the local file to a different mtime so we can detect re-download.
    import os
    new_mtime = local_bin.stat().st_mtime - 100
    os.utime(local_bin, (new_mtime, new_mtime))
    g = _two_step_chain()
    assert g.pull_cache(force=True) is True
    # File was rewritten by os.replace, so mtime changes.
    assert local_bin.stat().st_mtime > new_mtime


def test_pull_cache_sha256_mismatch_raises(tmp_path: Path):
    """Corrupted artifact -> PTDBackendError; no file installed."""
    import os
    fake_origin = tmp_path / "origin"; fake_origin.mkdir()
    reg_cache   = tmp_path / "registry_cache"; reg_cache.mkdir()
    phasic_cache = tmp_path / "phasic_cache"
    (phasic_cache / "parameterized_reward_compute").mkdir(parents=True)
    old_cache = os.environ.get("PHASIC_CACHE_DIR")
    os.environ["PHASIC_CACHE_DIR"] = str(phasic_cache)
    try:
        g = _two_step_chain()
        g.expectation()
        hash_hex = _graph_hash_hex(g)
        local_bin = _param_compute_cache_dir() / f"{hash_hex}.bin"
        published = fake_origin / f"{hash_hex}.bin"
        shutil.copy2(local_bin, published)
        local_bin.unlink()  # consumer side starts cold

        # Use a wrong sha so the verify step fails.
        bad_sha = "0" * 64
        registry_data = {
            "version": "2.0", "format": "ptd_pcg",
            "computes": {
                "demo": {
                    "graph_hash":      hash_hex,
                    "format_revision": 2,
                    "artifacts": {
                        "parent": {
                            "cid_or_path": f"file://{published}",
                            "sha256":      bad_sha,
                            "size_bytes":  published.stat().st_size,
                        },
                        "scc": [],
                    },
                    "metadata": {"description": "demo"},
                },
            },
        }
        (reg_cache / "registry.json").write_text(json.dumps(registry_data))
        registry = ComputeRegistry(
            registry_repo="demo/demo", cache_dir=reg_cache,
            auto_update=False)
        cr._default_registry = registry

        with pytest.raises(PTDBackendError, match="SHA-256 mismatch"):
            g.pull_cache()
        # No file installed and no .tmp leftover.
        assert not local_bin.exists()
        assert not (_param_compute_cache_dir() / f"{hash_hex}.bin.tmp").exists()
    finally:
        cr._default_registry = None
        if old_cache is None:
            os.environ.pop("PHASIC_CACHE_DIR", None)
        else:
            os.environ["PHASIC_CACHE_DIR"] = old_cache


def test_pull_cache_format_revision_too_new(tmp_path: Path):
    """Entry's format_revision > local build's supported -> PTDFormatError."""
    import os
    fake_origin = tmp_path / "origin"; fake_origin.mkdir()
    reg_cache   = tmp_path / "registry_cache"; reg_cache.mkdir()
    phasic_cache = tmp_path / "phasic_cache"
    (phasic_cache / "parameterized_reward_compute").mkdir(parents=True)
    old_cache = os.environ.get("PHASIC_CACHE_DIR")
    os.environ["PHASIC_CACHE_DIR"] = str(phasic_cache)
    try:
        g = _two_step_chain()
        g.expectation()
        hash_hex = _graph_hash_hex(g)
        published = fake_origin / f"{hash_hex}.bin"
        shutil.copy2(_param_compute_cache_dir() / f"{hash_hex}.bin", published)
        (_param_compute_cache_dir() / f"{hash_hex}.bin").unlink()
        sha = hashlib.sha256(published.read_bytes()).hexdigest()

        registry_data = {
            "version": "2.0", "format": "ptd_pcg",
            "computes": {
                "demo": {
                    "graph_hash":      hash_hex,
                    "format_revision": 99,   # way above local support
                    "artifacts": {
                        "parent": {
                            "cid_or_path": f"file://{published}",
                            "sha256":      sha,
                            "size_bytes":  published.stat().st_size,
                        },
                        "scc": [],
                    },
                    "metadata": {"description": "demo"},
                },
            },
        }
        (reg_cache / "registry.json").write_text(json.dumps(registry_data))
        registry = ComputeRegistry(
            registry_repo="demo/demo", cache_dir=reg_cache,
            auto_update=False)
        cr._default_registry = registry

        with pytest.raises(PTDFormatError, match="format_revision"):
            g.pull_cache()
    finally:
        cr._default_registry = None
        if old_cache is None:
            os.environ.pop("PHASIC_CACHE_DIR", None)
        else:
            os.environ["PHASIC_CACHE_DIR"] = old_cache


# ---------------------------------------------------------------------------
# Push tests (dry-run only — never touches network)
# ---------------------------------------------------------------------------


def test_push_cache_dry_run_returns_valid_json(tmp_path: Path):
    """dry_run=True returns a JSON string of the entry; no network."""
    import os
    phasic_cache = tmp_path / "phasic_cache"
    (phasic_cache / "parameterized_reward_compute").mkdir(parents=True)
    old_cache = os.environ.get("PHASIC_CACHE_DIR")
    os.environ["PHASIC_CACHE_DIR"] = str(phasic_cache)
    try:
        g = _two_step_chain()
        g.expectation()  # populate cache so save will succeed
        out = g.push_cache(
            id="demo_chain_v1",
            description="demo",
            domain="demo",
            model_type="chain",
            tags=["demo"],
            dry_run=True,
        )
        assert isinstance(out, str)
        parsed = json.loads(out)
        assert "demo_chain_v1" in parsed
        entry = parsed["demo_chain_v1"]
        assert entry["format_revision"] == 2
        assert len(entry["graph_hash"]) == 64
        assert entry["metadata"]["domain"] == "demo"
        assert entry["metadata"]["model_type"] == "chain"
        assert entry["metadata"]["tags"] == ["demo"]
        assert entry["metadata"]["vertices"] == g.vertices_length()
        assert entry["metadata"]["param_length"] == 1
        # Parent artifact reference points at the cache-relative path.
        assert entry["artifacts"]["parent"]["cid_or_path"].endswith(".bin")
        assert len(entry["artifacts"]["parent"]["sha256"]) == 64
    finally:
        if old_cache is None:
            os.environ.pop("PHASIC_CACHE_DIR", None)
        else:
            os.environ["PHASIC_CACHE_DIR"] = old_cache


def test_push_cache_populates_cache_if_missing(tmp_path: Path):
    """push_cache calls expectation() implicitly if the C cache is empty."""
    import os
    phasic_cache = tmp_path / "phasic_cache"
    (phasic_cache / "parameterized_reward_compute").mkdir(parents=True)
    old_cache = os.environ.get("PHASIC_CACHE_DIR")
    os.environ["PHASIC_CACHE_DIR"] = str(phasic_cache)
    try:
        g = _two_step_chain()
        # Do NOT call expectation(); push_cache should trigger it.
        assert not g._has_param_compute_graph_cache()
        out = g.push_cache(
            id="demo_chain_v1",
            description="demo",
            dry_run=True,
        )
        assert isinstance(out, str)
        # After push_cache the C-side cache must be populated.
        assert g._has_param_compute_graph_cache()
    finally:
        if old_cache is None:
            os.environ.pop("PHASIC_CACHE_DIR", None)
        else:
            os.environ["PHASIC_CACHE_DIR"] = old_cache


# ---------------------------------------------------------------------------
# Removed-API gate
# ---------------------------------------------------------------------------


def test_removed_top_level_symbols_are_gone():
    """The IPFS / transport / fetch-compute names should not re-appear."""
    for name in (
        "fetch_compute", "compute_source", "pin_compute",
        "TransportBackend", "ComputeRegistry", "IPFSBackend",
        "get_trace", "get_trace_by_hash", "TraceRegistry",
    ):
        assert not hasattr(phasic, name), (
            f"phasic.{name} should have been removed in the sharing rewrite")
