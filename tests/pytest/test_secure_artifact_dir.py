"""Regression test: generated sources/libraries must not use predictable /tmp.

Compiled artifacts used predictable, world-writable /tmp paths
(/tmp/trace_log_lik_<hash>.so, /tmp/graph_model_<hash>.cpp) trusted on load.
On a shared node a peer could pre-plant a malicious .so (the hash is derivable
from a public model) that ctypes then loads — RCE — or clobber a generated
source through a symlink. Artifacts now live in a per-user 0700 directory with
an unguessable name.
"""
import os
import stat
import subprocess

import pytest

import phasic
from phasic import _secure_artifact_dir, _compile_trace_library

pytestmark = pytest.mark.skipif(
    not hasattr(os, "getuid"), reason="POSIX ownership model required"
)


def test_artifact_dir_is_private_and_stable():
    d = _secure_artifact_dir()
    assert os.path.isdir(d)
    st = os.stat(d)
    assert st.st_uid == os.getuid(), "artifact dir must be owned by current user"
    # No group/other permissions at all (0700).
    assert st.st_mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
    # Stable within the process (so within-process caching still works).
    assert _secure_artifact_dir() == d
    # Not a predictable, shared, world-guessable location.
    assert not os.path.basename(d).startswith("trace_log_lik_")


def test_compile_trace_library_writes_into_secure_dir(monkeypatch, tmp_path):
    """The compiled .so path must live under the private artifact dir."""
    def fake_run(cmd, **kwargs):
        out = cmd[cmd.index("-o") + 1]
        open(out, "w").close()  # simulate a successful g++ producing the .so

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    lib = _compile_trace_library("// dummy source", "deadbeef12345678")

    d = _secure_artifact_dir()
    assert lib == os.path.join(d, "trace_log_lik_deadbeef12345678.so")
    assert os.path.commonpath([os.path.realpath(lib), os.path.realpath(d)]) == \
        os.path.realpath(d)
    # The old predictable path must NOT be used.
    assert not lib.startswith("/tmp/trace_log_lik_")
    # And the containing directory is private.
    assert os.stat(os.path.dirname(lib)).st_mode & (stat.S_IRWXG | stat.S_IRWXO) == 0
