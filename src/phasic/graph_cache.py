"""
Persistent disk cache for Graph objects built with ``Graph(..., cache_key=...)``.

Graphs are cached ONLY when the user names the entry with ``cache_key``;
the entry key (``phasic.callback_hash.cache_entry_key``) combines that name
with everything phasic can read exactly from the call (callback code and
closures, keyword arguments and ipv by value, phasic build, numpy version).

Cache directory: ``$PHASIC_CACHE_DIR/graphs`` when ``PHASIC_CACHE_DIR`` is
set and non-empty (``phasic.configure(cache_dir=...)`` sets it), otherwise
``DEFAULT_CACHE_DIR`` (``~/.phasic_cache/graphs``).
Entry file: ``<first 32 hex characters of the entry digest>.json``, a JSON
object whose first bytes are ``{"format":`` followed by the format string,
holding the serialised graph plus metadata.

This module uses a stdlib logger and imports nothing that loads
``phasic_pybind`` at module level (it is imported before
``OMP_NUM_THREADS`` is set).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from .callback_hash import (
    GRAPH_CACHE_FORMAT,
    GraphCacheKeyError,   # noqa: F401  (re-exported for convenience)
    GraphCacheLoadError,
)

if TYPE_CHECKING:
    from . import Graph

logger = logging.getLogger("phasic.graph_cache")

# Legacy / default cache directory. Read at CALL time everywhere (tests
# monkeypatch it).
DEFAULT_CACHE_DIR = Path.home() / ".phasic_cache" / "graphs"

# First bytes of every entry written by this format (and any later one):
# ``format`` is written first, as these exact bytes followed by the format
# string. The legacy sweep tests the same constant.
_ENTRY_HEADER = b'{"format":'
# First bytes of an entry written by the previous (master) format:
# json.dump(..., indent=2) with "version" first.
_LEGACY_ENTRY_PREFIX = b'{\n  "version"'

_ENTRY_NAME_RE = re.compile(r"^[0-9a-f]{32}\.json$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTIAL_RE = re.compile(r"^\.tmp-.*\.partial$")
_PARTIAL_STALE_SECONDS = 3600.0
_WRITE_SLICE = 8 * 1024 * 1024
_FSYNC_IGNORED = {errno.EINVAL, errno.EROFS, getattr(errno, "ENOTSUP", -1),
                  getattr(errno, "EOPNOTSUPP", -1)}
_META_KEYS = frozenset({"cache_key", "callback_name"})

_LOAD_WAYS_OUT = (
    "Ways out: delete this file, or use another cache_key. "
    "(phasic.clear_all_graph_caches() removes only files that still start "
    "with an entry header, so a file whose header is damaged must be "
    "deleted by hand.)"
)
_DIR_WAYS_OUT = (
    "Ways out: choose another cache directory with "
    "phasic.configure(cache_dir=...) (or PHASIC_CACHE_DIR), or build without "
    "cache_key."
)


def _resolve_cache_dir() -> Path:
    """The graph cache directory, absolute.

    Same rule as ``phasic.cache._cache_root`` and the C caches (an empty
    ``PHASIC_CACHE_DIR`` counts as unset), implemented inline: importing
    ``phasic.cache`` would load ``phasic_pybind`` (via
    ``trace_serialization``) before ``OMP_NUM_THREADS`` is set.
    """
    env = os.environ.get("PHASIC_CACHE_DIR")
    if env:
        return (Path(env) / "graphs").resolve()
    return Path(DEFAULT_CACHE_DIR).resolve()


def _open_entry_writer(fd: int):
    """File object the save path writes an entry through (binary,
    buffered: loops over short writes)."""
    return os.fdopen(fd, "wb")


def _open_entry_reader(path: Path):
    """File object the load path reads an entry through."""
    return open(path, "r", encoding="utf-8")


def _reraise_oserror(e: OSError, message: str) -> OSError:
    """A new exception of the same OSError subclass carrying ``message``."""
    try:
        new = (type(e)(e.errno, message) if e.errno is not None
               else type(e)(message))
    except Exception:
        new = OSError(e.errno, message) if e.errno is not None else OSError(message)
    return new


def _owner_of(path: Path) -> str:
    try:
        uid = os.stat(path).st_uid
    except Exception:
        return "owner unknown"
    try:
        import pwd
        return f"owned by {pwd.getpwuid(uid).pw_name}"
    except Exception:
        return f"owned by uid {uid}"


def _first_bytes(path: Path, n: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


class GraphCache:
    """
    On-disk cache of serialised Graph objects, keyed by entry digests.

    Directory: ``$PHASIC_CACHE_DIR/graphs`` when ``PHASIC_CACHE_DIR`` is set
    and non-empty, otherwise ``DEFAULT_CACHE_DIR`` (``~/.phasic_cache/graphs``);
    resolved once, at construction, and made absolute.

    Users normally do not construct this class: ``Graph(callback, ...,
    cache_key="name")`` looks the graph up and saves it. ``GraphCache().clear()``
    removes this directory's entries.

    Parameters
    ----------
    cache_dir : path-like, optional
        Explicit directory (made absolute). Default: resolved as above.
    create : bool, default False
        Create the directory (``Graph.__init__``'s ``cache_key`` path does).
        The management helpers never create it.
    """

    def __init__(self, cache_dir: Path | str | None = None,
                 create: bool = False) -> None:
        if cache_dir is None:
            self.cache_dir = _resolve_cache_dir()
        else:
            self.cache_dir = Path(cache_dir).resolve()
        self.last_clear_skipped = 0
        if create:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise _reraise_oserror(
                    e, f"cannot create the graph cache directory "
                       f"{self.cache_dir} ({e.strerror or e}). {_DIR_WAYS_OUT}"
                ) from e
        logger.debug("GraphCache initialized: %s", self.cache_dir)

    # -- paths ---------------------------------------------------------------
    def entry_path(self, digest: str) -> Path:
        """Absolute path of the entry for a full 64-character digest."""
        if type(digest) is not str or not _DIGEST_RE.match(digest):
            raise ValueError(f"entry digest must be 64 lowercase hex "
                             f"characters, got {digest!r}")
        return self.cache_dir / f"{digest[:32]}.json"

    def _probe_writable(self) -> None:
        """Create and remove a ``.tmp-<hex>.partial`` file (O_CREAT|O_EXCL);
        on failure raise the save-path OSError naming the directory."""
        probe = self.cache_dir / f".tmp-{secrets.token_hex(8)}.partial"
        try:
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            os.close(fd)
            os.unlink(probe)
        except OSError as e:
            raise _reraise_oserror(
                e, f"the graph cache directory {self.cache_dir} is not "
                   f"writable ({e.strerror or e}); nothing was built. "
                   f"{_DIR_WAYS_OUT}"
            ) from e

    # -- save ----------------------------------------------------------------
    def save(self, graph: Graph, digest: str, meta: dict) -> Path:
        """Write ``graph`` as the entry for ``digest`` (atomic replace).

        ``meta`` has exactly the keys ``cache_key`` (str) and
        ``callback_name`` (str or None). Returns the entry path. Any error
        propagates with its type unchanged; the temporary file is removed.
        """
        if type(meta) is not dict or set(meta) != _META_KEYS:
            raise ValueError(f"meta must be a dict with exactly the keys "
                             f"{sorted(_META_KEYS)}, got {meta!r}")
        if type(meta["cache_key"]) is not str or not (
                meta["callback_name"] is None
                or type(meta["callback_name"]) is str):
            raise TypeError("meta['cache_key'] must be a str and "
                            "meta['callback_name'] a str or None")
        path = self.entry_path(digest)
        try:
            graph_data = graph.serialize()
            encoded = _serialize_value(graph_data)
            if type(encoded) is not dict:
                raise TypeError("Graph.serialize() did not return a dict")
        except Exception as e:
            if not isinstance(e, MemoryError):
                e.add_note("Graph.serialize() failed: this graph cannot be "
                           "cached; build it without cache_key.")
            raise

        import phasic
        small = [
            ("entry_digest", digest),
            ("phasic_version", str(getattr(phasic, "__version__", "unknown"))),
            ("created_at", datetime.now().isoformat()),
            ("cache_key", meta["cache_key"]),
            ("callback_name", meta["callback_name"]),
        ]
        # A cache root removed during a long build is recreated.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_dir / f".tmp-{secrets.token_hex(8)}.partial"
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        replaced = False
        try:
            try:
                f = _open_entry_writer(fd)
            except BaseException:
                os.close(fd)
                raise
            with f:
                f.write(_ENTRY_HEADER
                        + json.dumps(GRAPH_CACHE_FORMAT).encode("ascii"))
                for k, v in small:
                    f.write(b"," + json.dumps(k).encode("ascii") + b":"
                            + json.dumps(v).encode("ascii"))
                f.write(b',"graph_data":{')
                first = True
                for k, v in encoded.items():
                    if not first:
                        f.write(b",")
                    first = False
                    f.write(json.dumps(k).encode("ascii") + b":")
                    s = json.dumps(v, separators=(",", ":"))
                    for i in range(0, len(s), _WRITE_SLICE):
                        f.write(s[i:i + _WRITE_SLICE].encode("ascii"))
                    del s
                f.write(b"}}")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    if e.errno not in _FSYNC_IGNORED:
                        raise
                    logger.info("fsync not supported for %s (%s); durability "
                                "is best effort", tmp, e)
            os.replace(tmp, path)
            replaced = True
        finally:
            if not replaced:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        logger.info("Saved graph to cache: %s (%d vertices)", path.name,
                    graph.vertices_length())
        return path

    # -- load ----------------------------------------------------------------
    def load(self, digest: str) -> Graph | None:
        """The cached graph for ``digest``, ``None`` if there is no entry.

        Raises
        ------
        GraphCacheLoadError
            The entry exists but its content cannot be used.
        OSError
            The entry cannot be read (same subclass, message naming the
            file and its owner).
        """
        path = self.entry_path(digest)
        entry = None
        for attempt in (0, 1):
            try:
                with _open_entry_reader(path) as f:
                    entry = json.load(f)
                break
            except FileNotFoundError:
                return None
            except OSError as e:
                if e.errno == errno.ESTALE and attempt == 0:
                    logger.info("stale file handle reading %s; re-reading "
                                "once", path)
                    continue
                raise _reraise_oserror(
                    e, f"cannot read graph cache entry {path} "
                       f"({_owner_of(path)}): {e.strerror or e}. Ways out: "
                       "use another cache_key, or another directory via "
                       "phasic.configure(cache_dir=...)."
                ) from e
            except ValueError as e:   # JSONDecodeError, UnicodeDecodeError
                raise GraphCacheLoadError(
                    f"graph cache entry {path} is not valid JSON "
                    f"({type(e).__name__}: {e}). {_LOAD_WAYS_OUT}") from e

        def bad(what: str) -> GraphCacheLoadError:
            return GraphCacheLoadError(
                f"graph cache entry {path} is unusable: {what}. "
                f"{_LOAD_WAYS_OUT}")

        if type(entry) is not dict:
            raise bad(f"top level is a {type(entry).__name__}, not an object")
        fmt = entry.get("format")
        if type(fmt) is not str:
            raise bad("field 'format' is missing or not a string")
        if fmt != GRAPH_CACHE_FORMAT:
            raise bad(f"format {fmt!r} != {GRAPH_CACHE_FORMAT!r}")
        for field, ok in (
                ("entry_digest", lambda v: type(v) is str),
                ("phasic_version", lambda v: type(v) is str),
                ("created_at", lambda v: type(v) is str),
                ("cache_key", lambda v: type(v) is str),
                ("callback_name", lambda v: v is None or type(v) is str),
                ("graph_data", lambda v: type(v) is dict)):
            if field not in entry or not ok(entry[field]):
                raise bad(f"field {field!r} is missing or of the wrong type")
        if entry["entry_digest"] != digest:
            raise bad("its entry_digest does not match the requested key (a "
                      "file copied or planted under the wrong name)")
        try:
            graph_data = _decode_graph_data(entry["graph_data"])
        except GraphCacheLoadError as e:
            raise GraphCacheLoadError(
                f"graph cache entry {path} is unusable: {e}. "
                f"{_LOAD_WAYS_OUT}") from e
        from phasic import Graph
        try:
            graph = Graph.from_serialized(graph_data)
        except (MemoryError, RecursionError):
            raise
        except Exception as e:
            raise GraphCacheLoadError(
                f"graph cache entry {path} could not be rebuilt "
                f"(Graph.from_serialized raised {type(e).__name__}: {e}). "
                f"{_LOAD_WAYS_OUT}") from e
        logger.info("Cache hit: %s (%d vertices)", path.name,
                    graph.vertices_length())
        return graph

    # -- management ------------------------------------------------------------
    def _clear_dir(self) -> tuple[int, int, int]:
        """Remove this directory's entries and stale ``.partial`` files.

        Returns (removed entries, skipped entry-named files, per-file
        errors). Never creates the directory.
        """
        removed = skipped = errors = 0
        try:
            dir_mtime = os.stat(self.cache_dir).st_mtime
            names = os.listdir(self.cache_dir)
        except FileNotFoundError:
            return 0, 0, 0
        for name in names:
            p = self.cache_dir / name
            if _ENTRY_NAME_RE.match(name):
                try:
                    head = _first_bytes(p, len(_LEGACY_ENTRY_PREFIX))
                    if not (head.startswith(_ENTRY_HEADER)
                            or head.startswith(_LEGACY_ENTRY_PREFIX)):
                        skipped += 1   # not an entry (e.g. a user's MD5 file)
                        continue
                    p.unlink()
                    removed += 1
                except OSError:
                    errors += 1
            elif _PARTIAL_RE.match(name):
                # Stale only when (directory mtime) - (file mtime) > 1 h,
                # both read from the same file server.
                try:
                    if dir_mtime - os.stat(p).st_mtime > _PARTIAL_STALE_SECONDS:
                        p.unlink()
                except OSError:
                    errors += 1
        return removed, skipped, errors

    def clear_graph_cache(self) -> int:
        """Remove all cache entries in this directory.

        Deletes only files named like entries (``<32 hex>.json``) whose
        first bytes are an entry header (this format's ``{"format":`` or
        the previous format's ``{\\n  "version"``); other entry-named files
        are left alone and counted in ``self.last_clear_skipped``. Also
        removes ``.tmp-*.partial`` files left by killed writers once they
        are more than an hour older than the directory's mtime. A missing
        directory gives 0 and is not created.

        Returns
        -------
        int
            Number of entry files removed (``.partial`` files not counted).
        """
        removed, skipped, _errors = self._clear_dir()
        self.last_clear_skipped = skipped
        logger.info("Cleared %d cached graphs (%d entry-named files skipped)",
                    removed, skipped)
        return removed

    def clear(self) -> int:
        """Alias for :meth:`clear_graph_cache`."""
        return self.clear_graph_cache()

    def get_cache_stats(self) -> dict[str, Any]:
        """Statistics over entry-named files (``<32 hex>.json``) only.

        Returns
        -------
        dict
            ``{'num_graphs': int, 'total_size_mb': float, 'cache_dir': str}``;
            a missing directory gives zeros and is not created.
        """
        count = 0
        total = 0
        try:
            names = os.listdir(self.cache_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if _ENTRY_NAME_RE.match(name):
                p = self.cache_dir / name
                try:
                    head = _first_bytes(p, len(_LEGACY_ENTRY_PREFIX))
                    if not (head.startswith(_ENTRY_HEADER)
                            or head.startswith(_LEGACY_ENTRY_PREFIX)):
                        continue   # foreign file with an entry-like name
                    total += os.stat(p).st_size
                    count += 1
                except OSError:
                    pass
        return {
            'num_graphs': count,
            'total_size_mb': total / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
        }


def _serialize_value(value: Any) -> Any:
    """
    Recursively encode ``Graph.serialize()`` output for JSON storage.

    Encodes plain JSON values (None, bool, int, float, str, lists, tuples,
    dicts), numpy scalars (as Python scalars) and numpy arrays (tagged
    ``{'__type__': 'ndarray', '__data__': [...]}``). Anything else raises
    ``TypeError``: the save side is symmetric with the restricted decoder,
    so no entry can be written that every later lookup would refuse.

    Examples
    --------
    >>> _serialize_value(42)
    42
    >>> _serialize_value(np.array([1, 2, 3]))
    {'__type__': 'ndarray', '__data__': [1, 2, 3]}
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return {'__type__': 'ndarray', '__data__': value.tolist()}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    raise TypeError(
        f"Cannot store a value of type {type(value).__name__} in a graph "
        f"cache entry (only plain JSON values and numpy arrays are stored)."
    )


def _decode_graph_data(value: Any) -> Any:
    """Restricted decoder for an entry's ``graph_data``.

    Plain JSON values plus the ``{'__type__': 'ndarray', '__data__': ...}``
    tag are all ``serialize()`` output needs; nothing is imported. Any other
    ``__type__`` tag, a tag without ``__data__``, or any exception while
    decoding (for example ragged ``__data__``), except ``MemoryError`` and
    ``RecursionError``, raises :class:`GraphCacheLoadError`.
    """
    try:
        return _decode(value)
    except (GraphCacheLoadError, MemoryError, RecursionError):
        raise
    except Exception as e:
        raise GraphCacheLoadError(
            f"graph_data could not be decoded ({type(e).__name__}: {e})"
        ) from e


def _decode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is list:
        return [_decode(v) for v in value]
    if type(value) is dict:
        if '__type__' in value:
            tag = value['__type__']
            if tag != 'ndarray':
                raise GraphCacheLoadError(
                    f"graph_data carries the type tag {tag!r}; only "
                    "'ndarray' is accepted")
            if '__data__' not in value:
                raise GraphCacheLoadError(
                    "graph_data carries an 'ndarray' tag without '__data__'")
            return np.array(_decode(value['__data__']))
        return {k: _decode(v) for k, v in value.items()}
    raise GraphCacheLoadError(
        f"graph_data holds a value of type {type(value).__name__}")


# Module-level convenience functions (resolve the directory without mkdir)
def get_cache_dir() -> Path:
    """The graph cache directory (``$PHASIC_CACHE_DIR/graphs`` or
    ``DEFAULT_CACHE_DIR``), absolute; not created."""
    return _resolve_cache_dir()


def get_graph_cache_stats() -> dict[str, Any]:
    """
    Get graph cache statistics.

    Convenience wrapper for ``GraphCache().get_cache_stats()``; never
    creates the directory.

    Returns
    -------
    dict
        Dictionary with cache statistics:
        - num_graphs: Number of cached graphs
        - total_size_mb: Total cache size in MB
        - cache_dir: Cache directory path

    Examples
    --------
    >>> from phasic import get_graph_cache_stats
    >>> stats = get_graph_cache_stats()
    >>> print(f"Cached graphs: {stats['num_graphs']}")
    """
    return GraphCache().get_cache_stats()


def print_graph_cache_info() -> None:
    """
    Print formatted graph cache information.

    Displays statistics about cached Graph objects in a human-readable format.

    Examples
    --------
    >>> from phasic import print_graph_cache_info
    >>> print_graph_cache_info()
    Cache directory: /Users/you/.phasic_cache/graphs
    Cached graphs: 5
    Total size: 2.34 MB
    """
    stats = get_graph_cache_stats()

    print(f"Cache directory: {stats['cache_dir']}")

    if stats['num_graphs'] == 0:
        print("Status: No cached graphs")
    else:
        print(f"Cached graphs: {stats['num_graphs']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")


def clear_all_graph_caches() -> dict[str, int]:
    """
    Clear all graph cache entries.

    Clears the current graph cache directory (``$PHASIC_CACHE_DIR/graphs``
    or ``DEFAULT_CACHE_DIR``, as :class:`GraphCache` resolves it; entries
    and stale ``.partial`` files, see :meth:`GraphCache.clear_graph_cache`)
    and ALSO sweeps the legacy default directory (``DEFAULT_CACHE_DIR``,
    read at call time), where the previous phasic always wrote entries:
    there only entry-named files that start with the PREVIOUS phasic's
    entry prefix (old-format entries) are deleted, so entries of this or a
    newer phasic, and any foreign file, survive. The legacy directory is never created; a read or
    unlink error on a single file is skipped and counted, never fatal.

    Returns
    -------
    dict
        ``{"removed": int, "skipped": int, "foreign": int}``: entry files
        removed over both directories, per-file errors, and entry-named
        files left alone because their content is not an entry.
    """
    cur = GraphCache()
    removed, foreign, errors = cur._clear_dir()
    legacy = Path(DEFAULT_CACHE_DIR).resolve()
    if legacy != cur.cache_dir:
        try:
            names = os.listdir(legacy)
        except (FileNotFoundError, NotADirectoryError):
            names = []
        except OSError:
            names = []
            errors += 1
        for name in names:
            if not _ENTRY_NAME_RE.match(name):
                continue
            p = legacy / name
            try:
                head = _first_bytes(p, len(_LEGACY_ENTRY_PREFIX))
                if not head.startswith(_LEGACY_ENTRY_PREFIX):
                    if not head.startswith(_ENTRY_HEADER):
                        foreign += 1
                    continue   # this or a newer format, or foreign: keep
                p.unlink()
                removed += 1
            except OSError:
                errors += 1
    logger.info("Cleared %d graph cache entries (%d skipped, %d foreign)",
                removed, errors, foreign)
    return {"removed": removed, "skipped": errors, "foreign": foreign}
