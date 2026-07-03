"""Regression test: a corrupt rev-3 PCG cache file must be a safe cache miss.

The rev-3 loader trusted untrusted header counts and per-command operand
offsets: an overflowing n_commands wrapped the malloc/section-size math (heap
overflow / OOB), and an out-of-range MEM offset or INPUT index drove an
arbitrary write / OOB read in the executor. A corrupt or attacker-supplied
.bin (reachable via the shared reward-compute cache) must instead be rejected
and recomputed, never crash the process.

The loader only runs for a graph that reads the cache from disk, so we write
the cache with one graph object, corrupt it on disk, then load it with a fresh
graph object of identical structure (same cache key).

rev-3 header layout (little-endian): magic[8], n_commands u64, mem_doubles
u64, n_inputs u64, reserved u64.
"""
from __future__ import annotations

import os
import struct

import pytest

pytest.importorskip("toy_model")
from toy_model import build_toy_base  # noqa: E402


@pytest.fixture(autouse=True)
def _enable_cache(monkeypatch):
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


def _first_scc(g):
    for scc in g.scc_decomposition().sccs_in_topo_order():
        return scc
    raise AssertionError("graph has no SCCs")


def _write_cold_cache():
    g = build_toy_base()
    scc = _first_scc(g)
    path = scc.cache_file_path()
    if os.path.exists(path):
        os.remove(path)
    scc.get_or_compute_prc()  # cold -> writes a valid .bin
    assert os.path.exists(path)
    return path


def _load_with_fresh_object():
    """Fresh graph of identical structure -> hits the on-disk cache loader."""
    g = build_toy_base()
    scc = _first_scc(g)
    scc.get_or_compute_prc()  # must not crash; miss => recompute
    return scc.cache_file_path()


def _read_u64(path, offset):
    with open(path, "rb") as f:
        f.seek(offset)
        return struct.unpack("<Q", f.read(8))[0]


def _clobber_u64(path, offset, value):
    data = bytearray(open(path, "rb").read())
    struct.pack_into("<Q", data, offset, value)
    with open(path, "wb") as f:
        f.write(data)


def test_baseline_warm_load_across_objects_ok():
    """Sanity: an uncorrupted cache loads across fresh objects (no crash)."""
    _write_cold_cache()
    _load_with_fresh_object()


def test_overflowing_n_commands_is_safe_miss():
    path = _write_cold_cache()
    _clobber_u64(path, 8, 0xFFFFFFFFFFFFFFFF)  # n_commands -> overflow
    # Fresh object loads the corrupt file: must not crash, treated as a miss.
    _load_with_fresh_object()
    assert _read_u64(path, 8) != 0xFFFFFFFFFFFFFFFF, (
        "corrupt cache should have been rejected and recomputed"
    )


def test_out_of_range_mem_offset_is_safe_miss():
    path = _write_cold_cache()
    original = _read_u64(path, 16)  # mem_doubles
    _clobber_u64(path, 16, 0)  # every MEM operand offset now out of range
    _load_with_fresh_object()
    assert _read_u64(path, 16) == original, (
        "validator should have rejected mem_doubles=0 and recomputed"
    )
