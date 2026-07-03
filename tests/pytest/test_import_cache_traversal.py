"""Regression test: CacheManager.import_cache must reject path traversal.

import_cache called tar.extract(member, cache_dir) using member.name
unchecked, with no filter and no rejection of '..', absolute paths, or
symlinks (CVE-2007-4559). On Python < 3.14 that legacy behavior lets a
crafted tarball write outside cache_dir (e.g. overwrite ~/.bashrc).
"""
import io
import tarfile

import pytest

from phasic.cache_manager import CacheManager


def _tar_with(members, base_dir):
    """Build a .tar.gz at base_dir/evil.tar.gz containing (name, data) members."""
    path = base_dir / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_traversal_member_is_rejected(tmp_path):
    outside = tmp_path / "victim.txt"
    outside.write_text("original")

    cache = tmp_path / "cache"
    cache.mkdir()
    # member escapes cache/ to overwrite ../victim.txt
    tar_path = _tar_with([("../victim.txt", b"PWNED")], tmp_path)

    mgr = CacheManager(cache)
    with pytest.raises(ValueError, match="traversal|absolute|escapes"):
        mgr.import_cache(tar_path)

    # The file outside the cache dir must be untouched.
    assert outside.read_text() == "original"


def test_absolute_path_member_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    tar_path = _tar_with([("/tmp/phasic_abs_evil", b"x")], tmp_path)
    mgr = CacheManager(cache)
    with pytest.raises(ValueError):
        mgr.import_cache(tar_path)


def test_symlink_member_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        link = tarfile.TarInfo(name="link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    mgr = CacheManager(cache)
    with pytest.raises(ValueError, match="link"):
        mgr.import_cache(path)


def test_benign_tarball_extracts_normally(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    tar_path = _tar_with(
        [("compiled/module_abc.o", b"data"), ("keys.txt", b"k")], tmp_path
    )
    mgr = CacheManager(cache)
    mgr.import_cache(tar_path)
    assert (cache / "compiled" / "module_abc.o").read_bytes() == b"data"
    assert (cache / "keys.txt").read_bytes() == b"k"
