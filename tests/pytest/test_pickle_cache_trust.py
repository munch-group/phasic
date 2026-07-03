"""Regression test: refuse untrusted pickle trace caches (cross-user RCE).

load_trace_from_cache fell back to pickle.load on <hash>.pkl in a directory
that honours PHASIC_CACHE_DIR (documented as a shared filesystem for
distributed SLURM workers). <hash> is a deterministic graph hash any peer can
compute, and pickle.load runs arbitrary __reduce__ code — so a peer could
plant a malicious <hash>.pkl and get code execution in the victim's account.

The fix only loads pickles owned by the current user and not group/world-
writable (nor in a group/world-writable directory).
"""
import os
import pickle
import stat

import pytest

import phasic.trace_serialization as ts

pytestmark = pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="POSIX ownership model required"
)

HASH = "a" * 64


def _mk(tmp_path, mode_file=0o600, mode_dir=0o700):
    d = tmp_path / "traces"
    d.mkdir()
    f = d / "trace.pkl"
    f.write_bytes(b"x")
    os.chmod(f, mode_file)
    os.chmod(d, mode_dir)
    return f


def test_trusted_file_accepted(tmp_path):
    f = _mk(tmp_path, mode_file=0o600, mode_dir=0o700)
    assert ts._pickle_source_is_trusted(f) is True


def test_world_writable_file_rejected(tmp_path):
    f = _mk(tmp_path, mode_file=0o666, mode_dir=0o700)
    assert ts._pickle_source_is_trusted(f) is False


def test_group_writable_file_rejected(tmp_path):
    f = _mk(tmp_path, mode_file=0o660, mode_dir=0o700)
    assert ts._pickle_source_is_trusted(f) is False


def test_world_writable_dir_rejected(tmp_path):
    f = _mk(tmp_path, mode_file=0o600, mode_dir=0o777)
    assert ts._pickle_source_is_trusted(f) is False


def test_foreign_owner_rejected(tmp_path, monkeypatch):
    f = _mk(tmp_path, mode_file=0o600, mode_dir=0o700)
    # Simulate the file being owned by a different user (attacker-planted):
    # our effective uid differs from the file's st_uid.
    monkeypatch.setattr(os, "getuid", lambda: os.stat(f).st_uid + 1)
    assert ts._pickle_source_is_trusted(f) is False


def test_load_refuses_and_does_not_execute_untrusted_pickle(tmp_path, monkeypatch):
    """End-to-end: a world-writable malicious pickle must not be loaded/executed."""
    sentinel = tmp_path / "PWNED"

    class Evil:
        def __reduce__(self):
            # If unpickled, this creates the sentinel file (proof of execution).
            return (open, (str(sentinel), "w"))

    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    traces = tmp_path / "traces"
    traces.mkdir()
    os.chmod(traces, 0o700)

    pkl = traces / f"{HASH}.pkl"
    pkl.write_bytes(pickle.dumps(Evil()))
    os.chmod(pkl, 0o666)  # world-writable => untrusted

    result = ts.load_trace_from_cache(HASH)

    assert result is None, "untrusted pickle must be treated as a cache miss"
    assert not sentinel.exists(), "pickle payload executed — RCE not blocked!"
