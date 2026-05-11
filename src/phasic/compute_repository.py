"""Compute-graph sharing repository.

This module provides ``ComputeRegistry`` and a handful of top-level
helpers for sharing pre-computed elimination output between machines.
The artifact shared is the binary
``parameterized_reward_compute_graph`` (``.bin``) file written by
``ptd_save_parameterized_reward_compute_graph`` on the C side, keyed
by the graph's content hash (``ptd_graph_content_hash``).

Why this replaces the legacy ``trace_repository``:

- ``graph.expectation() / .pdf() / .moments()`` now route directly to
  the C path, which reads/writes
  ``~/.phasic_cache/parameterized_reward_compute/<hash>.bin``.
- The Python ``EliminationTrace`` cache is deprecated (see
  ``phasic.cache.clear_trace_cache``'s deprecation notice). Nothing
  in the public API populates it any more.
- The legacy ``ipfshttpclient`` dependency is dead upstream and
  rejects modern kubo daemons; this module uses the new
  :mod:`phasic.transport` stack instead.

Consumer workflow
-----------------

>>> import phasic
>>> graph = phasic.Graph(my_callback, nr_samples=5)
>>> phasic.fetch_compute(graph)   # True on cache hit
>>> # Next expectation() call reuses the published elimination
>>> expected = graph.expectation()

Publisher workflow
------------------

See :mod:`phasic.publish_cli` for the ``phasic publish-compute`` CLI
that fronts :class:`ComputeRegistry.publish_compute`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import PTDBackendError, PTDFormatError
from .logging_config import get_logger
from .transport import TransportBackend

if TYPE_CHECKING:
    from . import Graph

logger = get_logger(__name__)


# Format expected for parent-graph artifacts. Bump when the on-disk
# parameterized_reward_compute_graph layout changes incompatibly.
_REGISTRY_FORMAT = "ptd_pcg"

# Maximum format_revision this build can load. Must track the C-side
# ``PTD_PCG_FORMAT_REVISION`` constant (currently 2 — rev-1 = monolithic
# parent file, rev-2 = SCC file with EXTERNAL anchors).
_LOCAL_FORMAT_REVISION = 2

# On-disk header layout — matches the C ``ptd_pcg_disk_header`` struct
# in src/c/phasic.c. ``magic`` (8 bytes) + uint32 version + uint32
# format_revision + uint64 graph_hash_truncated + 3 × uint64.
_PCG_HEADER_STRUCT = struct.Struct("<8sIIQQQQ")
_PCG_HEADER_MAGIC = b"PTDPRMC1"


def _peek_format_revision(path: Path) -> tuple[int, int]:
    """Return ``(version, format_revision)`` from a ``.bin`` header.

    Reads only the first 40 bytes. Used by the fetch path to refuse
    files that this build cannot load.

    Raises
    ------
    PTDFormatError
        If the file is too short or has the wrong magic bytes.
    """
    try:
        with path.open("rb") as fh:
            buf = fh.read(_PCG_HEADER_STRUCT.size)
    except OSError as e:
        raise PTDFormatError(f"cannot read header from {path}: {e}") from e
    if len(buf) < _PCG_HEADER_STRUCT.size:
        raise PTDFormatError(
            f"{path}: file too short to contain a PTDPRMC1 header")
    magic, version, format_revision, *_ = _PCG_HEADER_STRUCT.unpack(buf)
    if magic != _PCG_HEADER_MAGIC:
        raise PTDFormatError(
            f"{path}: bad magic {magic!r}, expected {_PCG_HEADER_MAGIC!r}")
    return version, format_revision


def _sha256_of(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _graph_hash_hex(graph: Graph) -> str:
    """Return the canonical hex hash for *graph*.

    Wraps :func:`phasic.hash.compute_graph_hash`. Kept as a private
    helper so call sites read consistently and so future caching is
    straightforward.
    """
    from . import hash as _phasic_hash
    return _phasic_hash.compute_graph_hash(graph).hash_hex


def _param_compute_cache_dir() -> Path:
    """Return the on-disk directory that the C path consults.

    Honours ``PHASIC_CACHE_DIR`` so it agrees with the C side's
    ``ptd_cache_root_dir``.
    """
    override = os.environ.get("PHASIC_CACHE_DIR")
    root = Path(override) if override else (Path.home() / ".phasic_cache")
    return root / "parameterized_reward_compute"


# ============================================================================
# ComputeRegistry
# ============================================================================


class ComputeRegistry:
    """Browse, fetch, pin, and publish C-elimination compute artifacts.

    Parameters
    ----------
    registry_repo : str, default 'munch-group/phasic-traces'
        GitHub ``owner/name`` for the registry repo. The registry
        JSON is fetched from
        ``https://raw.githubusercontent.com/<repo>/master/registry.json``.
    cache_dir : Path, optional
        Local cache directory for the registry index. Defaults to
        ``~/.phasic_traces``. Per-artifact ``.bin`` files always land
        in the C-side cache (``_param_compute_cache_dir()``) so the
        next ``expectation()`` call picks them up transparently.
    backend : TransportBackend, optional
        Custom transport backend. Defaults to a lazily-constructed
        :class:`phasic.transport.ipfs.IPFSBackend`.
    auto_update : bool, default True
        If True, fetch the registry index from GitHub on init.
    """

    REGISTRY_URL_TEMPLATE = (
        "https://raw.githubusercontent.com/{repo}/master/registry.json")

    def __init__(
        self,
        registry_repo: str = "munch-group/phasic-traces",
        cache_dir: Path | None = None,
        backend: TransportBackend | None = None,
        auto_update: bool = True,
    ) -> None:
        self.registry_repo = registry_repo
        self.cache_dir = (Path(cache_dir) if cache_dir
                          else Path.home() / ".phasic_traces")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self.index: dict[str, Any] | None = None

        self._load_cached_index()
        if auto_update:
            try:
                self.update_index()
            except PTDBackendError as e:
                if self.index is None:
                    # No cache fallback — re-raise so consumers know.
                    raise
                logger.warning(
                    "Failed to update registry, using cached version: %s", e)

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------

    @property
    def backend(self) -> TransportBackend:
        """Transport backend; constructed lazily on first access."""
        if self._backend is None:
            from .transport.ipfs import IPFSBackend
            self._backend = IPFSBackend()
        return self._backend

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.cache_dir / "registry.json"

    def _load_cached_index(self) -> None:
        path = self._index_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cached registry unreadable (%s); ignoring", e)
            return
        if not isinstance(data, dict) or "computes" not in data:
            logger.warning("Cached registry %s has wrong shape; ignoring",
                           path)
            return
        self.index = data

    def update_index(self) -> None:
        """Fetch the registry index from GitHub and cache it locally.

        Raises
        ------
        PTDBackendError
            If the index cannot be fetched and no cached copy exists.
        """
        from .transport._retry import request_with_retry
        import requests

        url = self.REGISTRY_URL_TEMPLATE.format(repo=self.registry_repo)
        try:
            response = request_with_retry(url, timeout=10.0)
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            raise PTDBackendError(
                f"Failed to fetch registry from {url}: {e}") from e

        if not isinstance(data, dict) or "computes" not in data:
            raise PTDBackendError(
                f"Registry at {url} missing 'computes' field; got keys "
                f"{list(data.keys()) if isinstance(data, dict) else type(data)}")

        self.index = data
        try:
            self._index_path().write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning("Failed to cache registry: %s", e)

    def _require_index(self) -> dict[str, Any]:
        if self.index is None:
            raise PTDBackendError(
                "Registry index not loaded. Call update_index() or "
                "construct the registry with auto_update=True.")
        return self.index

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_computes(
        self,
        domain: str | None = None,
        model_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List published compute artifacts, optionally filtered.

        Filtering keys come from each entry's ``metadata`` block.
        Retracted entries are skipped.

        Returns
        -------
        list[dict]
            One dict per published artifact. Each carries a
            ``compute_id`` field (the registry key) plus the entry's
            ``graph_hash`` and flattened metadata.
        """
        idx = self._require_index()
        results: list[dict[str, Any]] = []
        for compute_id, entry in idx["computes"].items():
            if entry.get("retracted"):
                continue
            meta = entry.get("metadata", {})
            if domain is not None and meta.get("domain") != domain:
                continue
            if model_type is not None and meta.get("model_type") != model_type:
                continue
            if tags is not None:
                entry_tags = meta.get("tags", [])
                if not all(t in entry_tags for t in tags):
                    continue
            results.append({
                "compute_id": compute_id,
                "graph_hash": entry.get("graph_hash"),
                "format_revision": entry.get("format_revision"),
                "description": meta.get("description", ""),
                **meta,
            })
        return results

    # ------------------------------------------------------------------
    # Source introspection
    # ------------------------------------------------------------------

    def compute_source(self, graph: Graph) -> dict[str, Any]:
        """Report where a fetch for *graph* would come from.

        Without downloading anything, return a small dict describing
        the cheapest source: local on-disk cache, a daemon-backed
        IPFS fetch, an HTTP-gateway fetch, or unavailable.
        """
        hash_hex = _graph_hash_hex(graph)
        local_path = _param_compute_cache_dir() / f"{hash_hex}.bin"
        info: dict[str, Any] = {
            "graph_hash": hash_hex,
            "local_path": str(local_path),
            "cached_locally": local_path.exists(),
        }
        if local_path.exists():
            info["source"] = "local_cache"
            return info

        idx = self.index
        entry = None
        if idx is not None:
            for compute_id, e in idx.get("computes", {}).items():
                if e.get("graph_hash") == hash_hex and not e.get("retracted"):
                    entry = (compute_id, e)
                    break
        info["registered"] = entry is not None
        if entry is None:
            info["source"] = "unavailable"
            return info

        info["compute_id"] = entry[0]
        info["backend"] = self.backend.name
        # IPFSBackend is the common case; treat 'daemon vs gateway'
        # as a structured field so callers can present it nicely.
        ipfs = self._backend  # avoid lazy-constructing if not needed
        if ipfs is not None and hasattr(ipfs, "kubo"):
            info["source"] = ("ipfs_daemon" if ipfs.kubo.is_alive()
                              else "http_gateway")
        else:
            info["source"] = self.backend.name
        return info

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_compute(
        self,
        graph: Graph,
        force_download: bool = False,
    ) -> bool:
        """Download the C-elimination artifact for *graph* if available.

        On a hit the file is placed at
        ``$PHASIC_CACHE_DIR/parameterized_reward_compute/<hash>.bin``
        with atomic write-then-rename, the SHA-256 is verified, and
        the format revision is checked against the local build.

        Parameters
        ----------
        graph : phasic.Graph
            The graph whose compute artifact should be fetched. The
            graph's content hash decides what is downloaded.
        force_download : bool, default False
            If True, re-download even when a local cache file exists.

        Returns
        -------
        bool
            ``True`` if a fresh download succeeded or a local cache
            file was already present (and not ``force_download``).
            ``False`` if the registry has no entry for the graph's
            hash.

        Raises
        ------
        PTDBackendError
            If the registry entry exists but cannot be fetched, or
            the checksum does not match.
        PTDFormatError
            If the artifact's format revision exceeds the local
            build's capability.
        """
        hash_hex = _graph_hash_hex(graph)
        cache_dir = _param_compute_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{hash_hex}.bin"

        if target.exists() and not force_download:
            return True

        idx = self._require_index()
        match: tuple[str, dict[str, Any]] | None = None
        for compute_id, entry in idx["computes"].items():
            if entry.get("graph_hash") == hash_hex and not entry.get("retracted"):
                match = (compute_id, entry)
                break
        if match is None:
            logger.info(
                "No compute artifact registered for hash %s...", hash_hex[:16])
            return False

        compute_id, entry = match
        format_revision = entry.get("format_revision")
        if (format_revision is not None
                and format_revision > _LOCAL_FORMAT_REVISION):
            raise PTDFormatError(
                f"Compute artifact '{compute_id}' has format_revision="
                f"{format_revision} but this phasic build only "
                f"supports up to {_LOCAL_FORMAT_REVISION}. "
                f"Upgrade phasic to read newer artifacts.")

        parent = entry.get("artifacts", {}).get("parent")
        if not parent or not parent.get("cid_or_path"):
            raise PTDBackendError(
                f"Registry entry '{compute_id}' has no parent artifact "
                f"reference")
        sha = parent.get("sha256")
        if not sha:
            raise PTDBackendError(
                f"Registry entry '{compute_id}' missing parent.sha256")

        cid_or_path = self._resolve_artifact_url(parent["cid_or_path"])

        tmp = target.with_suffix(".bin.tmp")
        try:
            self.backend.get(cid_or_path, tmp)
            actual = _sha256_of(tmp)
            if actual != sha:
                tmp.unlink(missing_ok=True)
                raise PTDBackendError(
                    f"SHA-256 mismatch for '{compute_id}' parent: "
                    f"expected {sha}, got {actual}")
            # Header sanity-check before installing the file.
            try:
                _peek_format_revision(tmp)
            except PTDFormatError:
                tmp.unlink(missing_ok=True)
                raise
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        # Per-SCC artifacts are optional. The C side recomputes any
        # missing per-SCC entry on demand, so we don't refuse the
        # parent file on a partial fetch.
        for scc in entry.get("artifacts", {}).get("scc", []):
            self._fetch_scc(compute_id, scc, cache_dir)

        logger.info(
            "Fetched compute artifact '%s' (%d-byte parent, hash %s...)",
            compute_id, target.stat().st_size, hash_hex[:16])
        return True

    def _resolve_artifact_url(self, cid_or_path: str) -> str:
        """Turn ``cid_or_path`` into something the backend can fetch.

        - Any ``<scheme>://...`` URL (``http``, ``https``, ``file``,
          ``s3``, etc.) is returned as-is.
        - Bare CID (``"bafy..."``, ``"Qm..."``) is returned as-is.
        - Anything else is treated as a path relative to the
          registry repo and rewritten to a ``raw.githubusercontent.com``
          URL.
        """
        if "://" in cid_or_path:
            return cid_or_path
        if cid_or_path.startswith(("bafy", "bafk", "Qm")):
            return cid_or_path
        # Repo-relative path.
        return (f"https://raw.githubusercontent.com/{self.registry_repo}"
                f"/master/{cid_or_path}")

    def _fetch_scc(
        self,
        compute_id: str,
        scc: dict[str, Any],
        cache_dir: Path,
    ) -> None:
        """Fetch one per-SCC artifact, mirroring the C-side naming."""
        cid_or_path = scc.get("cid_or_path")
        sha = scc.get("sha256")
        scc_hash = scc.get("scc_hash")
        if not (cid_or_path and sha and scc_hash):
            logger.warning(
                "Skipping malformed SCC artifact in '%s': %r",
                compute_id, scc)
            return
        target = cache_dir / f"scc_{scc_hash}.bin"
        if target.exists():
            return
        url = self._resolve_artifact_url(cid_or_path)
        tmp = target.with_suffix(".bin.tmp")
        try:
            self.backend.get(url, tmp)
            actual = _sha256_of(tmp)
            if actual != sha:
                tmp.unlink(missing_ok=True)
                logger.warning(
                    "Skipping SCC artifact with bad SHA-256 in '%s' "
                    "(expected %s, got %s)",
                    compute_id, sha[:16], actual[:16])
                return
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------

    def pin_compute(self, graph: Graph) -> None:
        """Pin the artifact for *graph* on the local kubo daemon.

        Useful for nodes that want to mirror published artifacts.
        Requires a running daemon (raises ``PTDBackendError``
        otherwise via :meth:`TransportBackend.pin`).
        """
        hash_hex = _graph_hash_hex(graph)
        idx = self._require_index()
        for compute_id, entry in idx["computes"].items():
            if entry.get("graph_hash") != hash_hex or entry.get("retracted"):
                continue
            parent = entry.get("artifacts", {}).get("parent", {})
            cid = parent.get("cid_or_path", "")
            if cid.startswith(("bafy", "bafk", "Qm")):
                self.backend.pin(cid)
            for scc in entry.get("artifacts", {}).get("scc", []):
                cid = scc.get("cid_or_path", "")
                if cid.startswith(("bafy", "bafk", "Qm")):
                    self.backend.pin(cid)
            return
        raise PTDBackendError(
            f"No registered compute artifact for hash {hash_hex}")

    # ------------------------------------------------------------------
    # Publish (helper used by the CLI)
    # ------------------------------------------------------------------

    def build_publish_entry(
        self,
        graph: Graph,
        *,
        compute_id: str,
        metadata: dict[str, Any],
        artifact_paths: dict[str, Path],
    ) -> dict[str, Any]:
        """Construct a registry entry from a populated graph.

        Used by :mod:`phasic.publish_cli`. *artifact_paths* is a dict
        with at least key ``'parent'`` mapping to a local ``.bin``
        file; optionally a ``'scc'`` key with a list of paths. The
        files must already exist; SHA-256s are computed from them.

        Returns the entry dict (not yet inserted into a registry).
        """
        hash_hex = _graph_hash_hex(graph)
        parent_path = Path(artifact_paths["parent"])
        if not parent_path.is_file():
            raise PTDBackendError(
                f"parent artifact missing: {parent_path}")

        _peek_format_revision(parent_path)  # raises on garbage

        entry: dict[str, Any] = {
            "graph_hash": hash_hex,
            "format_revision": _LOCAL_FORMAT_REVISION,
            "artifacts": {
                "parent": {
                    "cid_or_path": f"artifacts/{hash_hex}.bin",
                    "sha256": _sha256_of(parent_path),
                    "size_bytes": parent_path.stat().st_size,
                },
                "scc": [],
            },
            "metadata": dict(metadata),
        }
        for scc_path in artifact_paths.get("scc", []):
            scc_path = Path(scc_path)
            if not scc_path.is_file():
                continue
            # File name: scc_<hash>.bin — strip prefix and .bin.
            stem = scc_path.stem
            if stem.startswith("scc_"):
                scc_hash = stem[len("scc_"):]
            else:
                scc_hash = stem
            entry["artifacts"]["scc"].append({
                "cid_or_path": f"artifacts/{scc_path.name}",
                "scc_hash": scc_hash,
                "sha256": _sha256_of(scc_path),
                "size_bytes": scc_path.stat().st_size,
            })

        return entry


# ============================================================================
# Top-level helpers (re-exported from phasic.__init__)
# ============================================================================


_default_registry: ComputeRegistry | None = None


def _get_default_registry() -> ComputeRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ComputeRegistry()
    return _default_registry


def fetch_compute(graph: Graph, force_download: bool = False) -> bool:
    """Fetch the published compute artifact for *graph* if available.

    Convenience wrapper around
    :meth:`ComputeRegistry.fetch_compute` that uses a module-level
    default registry (constructed on first call).
    """
    return _get_default_registry().fetch_compute(graph, force_download)


def compute_source(graph: Graph) -> dict[str, Any]:
    """Inspect where a fetch for *graph* would come from."""
    return _get_default_registry().compute_source(graph)


def pin_compute(graph: Graph) -> None:
    """Pin the published artifact for *graph* on the local daemon."""
    _get_default_registry().pin_compute(graph)


def list_computes(
    domain: str | None = None,
    model_type: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List published compute artifacts, optionally filtered."""
    return _get_default_registry().list_computes(
        domain=domain, model_type=model_type, tags=tags)


__all__ = [
    "ComputeRegistry",
    "fetch_compute",
    "compute_source",
    "pin_compute",
    "list_computes",
]
