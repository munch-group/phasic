"""Compute-graph sharing via the phasic-traces GitHub registry.

This module backs the ``Graph.pull_cache()`` and ``Graph.push_cache()``
methods. The artifact shared is the binary
``parameterized_reward_compute_graph`` (``.bin``) file written by
``ptd_save_parameterized_reward_compute_graph`` on the C side, keyed
by the graph's content hash (``ptd_graph_content_hash``).

Storage model
-------------
Artifacts and the registry live in a public GitHub repository
(by default ``munch-group/phasic-traces``). Consumers read
``registry.json`` from ``raw.githubusercontent.com`` and download
``.bin`` files the same way — no auth needed. Publishers use the
``gh`` CLI to clone the repo, splice an entry in, push a branch,
and open a PR.

There is no IPFS in the picture. Earlier versions of this module
had a kubo-daemon path; it was dead code in practice and was removed
to simplify the UX.

Public surface
--------------
- :class:`ComputeRegistry` — registry index + fetch + publish, all
  internal except for the methods that ``Graph.pull_cache`` /
  ``Graph.push_cache`` call.
- :func:`list_computes` — browse the registry without a graph.

``Graph.pull_cache(force=False) -> bool`` and
``Graph.push_cache(*, id, description, ...) -> str`` are defined on
the Graph class in ``phasic/__init__.py`` and use this module
internally.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from ._http_retry import request_with_retry
from .exceptions import PTDBackendError, PTDFormatError
from .logging_config import get_logger

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


# ============================================================================
# Low-level helpers
# ============================================================================


def _peek_format_revision(path: Path) -> tuple[int, int]:
    """Return ``(version, format_revision)`` from a ``.bin`` header.

    Reads only the first 40 bytes. Used by the fetch path to refuse
    files this build cannot load.

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
    """Return the canonical hex hash for *graph*."""
    from . import hash as _phasic_hash
    return _phasic_hash.compute_graph_hash(graph).hash_hex


def _param_compute_cache_dir() -> Path:
    """Return the on-disk directory the C path consults.

    Honours ``PHASIC_CACHE_DIR`` so it agrees with the C side's
    ``ptd_cache_root_dir``.
    """
    override = os.environ.get("PHASIC_CACHE_DIR")
    root = Path(override) if override else (Path.home() / ".phasic_cache")
    return root / "parameterized_reward_compute"


def _http_fetch(url: str, output_path: Path, timeout: float = 30.0) -> None:
    """GET *url* and stream the body to *output_path*.

    Used for both ``raw.githubusercontent.com`` URLs (the default
    case) and ``file://`` URLs (used by tests).
    """
    if url.startswith("file://"):
        # Direct copy; no requests round-trip.
        src = url[len("file://"):]
        shutil.copyfile(src, output_path)
        return

    response = request_with_retry(url, timeout=timeout, stream=True)
    with output_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                fh.write(chunk)


# ============================================================================
# git / gh helpers (used by push_cache)
# ============================================================================


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> str:
    """Run *cmd*, capturing output. Raises on non-zero exit."""
    logger.debug("$ %s", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    if check and proc.returncode != 0:
        raise PTDBackendError(
            f"command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}")
    return proc.stdout


def _git_config_value(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "--get", key],
            capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _default_author() -> str:
    name = _git_config_value("user.name")
    email = _git_config_value("user.email")
    if name and email:
        return f"{name} <{email}>"
    if name:
        return name
    if email:
        return email
    return "unknown"


def _check_gh_ready() -> None:
    """Raise PTDBackendError if `gh` is missing or unauthenticated.

    A 5-second `gh auth status` probe distinguishes "not installed"
    from "installed but not logged in".
    """
    if not shutil.which("gh"):
        raise PTDBackendError(
            "`gh` (GitHub CLI) is not installed.\n"
            "  Install:  brew install gh   (macOS)\n"
            "            apt install gh    (Debian/Ubuntu)\n"
            "  Or see https://cli.github.com/")
    try:
        proc = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True, text=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise PTDBackendError(f"`gh auth status` failed: {e}") from e
    if proc.returncode != 0:
        raise PTDBackendError(
            "`gh` is not authenticated for this user.\n"
            "  Run once in a terminal: gh auth login\n"
            "  Then retry push_cache().")


def _populate_prc_cache(graph) -> None:
    """Force the C-side parameterized_reward_compute_graph to be built.

    ``_save_param_compute_graph`` refuses to save if it's NULL; the
    cheapest populator is ``expectation()``.
    """
    for fn_name in ("expectation", "moments", "pdf"):
        fn = getattr(graph, fn_name, None)
        if fn is None:
            continue
        try:
            if fn_name == "pdf":
                fn(1.0)
            elif fn_name == "moments":
                fn(1)
            else:
                fn()
            return
        except Exception as e:  # pragma: no cover — model-dependent
            logger.debug("%s() failed: %s", fn_name, e)
            continue
    raise PTDBackendError(
        "could not populate the C-side compute cache: expectation(), "
        "moments(), and pdf() all failed on this graph")


def _save_artifacts(graph, staging_dir: Path) -> tuple[Path, list[Path]]:
    """Save the parent ``.bin`` and bundle any per-SCC files.

    Returns ``(parent_path, scc_paths)``. The parent file is always
    present. The per-SCC list contains only the cache files that
    belong to *this* graph — derived from its own SCC decomposition,
    not from a global glob. That keeps unrelated leftover SCC files
    (from previous coalescent runs etc.) out of the bundle.
    """
    hash_hex = _graph_hash_hex(graph)
    parent = staging_dir / f"{hash_hex}.bin"

    if not getattr(graph, "_has_param_compute_graph_cache", lambda: False)():
        _populate_prc_cache(graph)

    graph._save_param_compute_graph(str(parent))

    scc_paths: list[Path] = []
    local_cache = _param_compute_cache_dir()
    if not local_cache.exists():
        return parent, scc_paths

    expected_scc_files = _expected_scc_filenames(graph)
    for filename in expected_scc_files:
        src = local_cache / filename
        if not src.is_file():
            continue
        target = staging_dir / filename
        shutil.copy2(src, target)
        scc_paths.append(target)

    return parent, scc_paths


def _expected_scc_filenames(graph) -> list[str]:
    """Return the ``scc_<hash>.bin`` filenames for *graph*'s SCCs.

    Mirrors ``phasic.distributed_scc.cache_path_for_scc`` but
    returns just the filename, not the absolute path. If the SCC
    decomposition can't be computed (e.g. the graph is too small
    or some C-side error), returns an empty list — callers should
    treat per-SCC artifacts as best-effort.
    """
    try:
        scc_graph = graph.scc_decomposition()
        from phasic.phasic_pybind import hash as _hash_mod
    except Exception as e:
        logger.debug("scc_decomposition failed: %s", e)
        return []

    filenames: list[str] = []
    try:
        for scc in scc_graph:
            try:
                synth = scc.as_synthetic_graph()
            except Exception:
                continue
            h = _hash_mod.compute_graph_hash(synth)
            filenames.append(f"scc_{h.hash_hex}.bin")
    except Exception as e:
        logger.debug("SCC hash enumeration failed: %s", e)
        return []
    return filenames


# ============================================================================
# ComputeRegistry
# ============================================================================


class ComputeRegistry:
    """Internal helper backing Graph.pull_cache / push_cache and list_computes.

    Not exported at the top level. Construct directly only when you
    need a non-default ``registry_repo``.

    Parameters
    ----------
    registry_repo : str, default 'munch-group/phasic-traces'
        GitHub ``owner/name``. Registry JSON is read from
        ``raw.githubusercontent.com/<repo>/master/registry.json``.
    cache_dir : Path, optional
        Local directory for caching ``registry.json``. Defaults to
        ``~/.phasic_traces``. Per-artifact ``.bin`` files always land
        in ``_param_compute_cache_dir()`` so the next ``expectation()``
        call picks them up.
    auto_update : bool, default True
        Fetch the registry index from GitHub on init.
    """

    REGISTRY_URL_TEMPLATE = (
        "https://raw.githubusercontent.com/{repo}/master/registry.json")

    def __init__(
        self,
        registry_repo: str = "munch-group/phasic-traces",
        cache_dir: Path | None = None,
        auto_update: bool = True,
    ) -> None:
        self.registry_repo = registry_repo
        self.cache_dir = (Path(cache_dir) if cache_dir
                          else Path.home() / ".phasic_traces")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, Any] | None = None

        self._load_cached_index()
        if auto_update:
            try:
                self.update_index()
            except PTDBackendError:
                if self.index is None:
                    raise

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
        """Fetch the registry index from GitHub and cache it locally."""
        url = self.REGISTRY_URL_TEMPLATE.format(repo=self.registry_repo)
        # Allow file:// URLs so tests can point at a local mock.
        if url.startswith("file://"):
            try:
                data = json.loads(Path(url[len("file://"):]).read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise PTDBackendError(
                    f"Failed to read registry from {url}: {e}") from e
        else:
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
                "Registry index not loaded. Construct the registry "
                "with auto_update=True or call update_index().")
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

        Returns one dict per artifact with ``compute_id`` and the
        entry's metadata flattened in. Retracted entries are skipped.
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
    # Pull (fetch)
    # ------------------------------------------------------------------

    def pull(self, graph: Graph, *, force: bool = False) -> bool:
        """Download the compute artifact for *graph* if registered.

        Internal counterpart of ``Graph.pull_cache``.

        Returns True on a fresh download or an existing local file,
        False on registry miss. Raises on network errors, bad
        SHA-256, or format_revision exceeding this build.
        """
        hash_hex = _graph_hash_hex(graph)
        cache_dir = _param_compute_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{hash_hex}.bin"

        if target.exists() and not force:
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
                f"{format_revision} but this phasic build only supports "
                f"up to {_LOCAL_FORMAT_REVISION}. Upgrade phasic to read "
                f"newer artifacts.")

        parent = entry.get("artifacts", {}).get("parent")
        if not parent or not parent.get("cid_or_path"):
            raise PTDBackendError(
                f"Registry entry '{compute_id}' has no parent artifact "
                f"reference")
        sha = parent.get("sha256")
        if not sha:
            raise PTDBackendError(
                f"Registry entry '{compute_id}' missing parent.sha256")

        url = self._resolve_artifact_url(parent["cid_or_path"])
        tmp = target.with_suffix(".bin.tmp")
        try:
            _http_fetch(url, tmp)
            actual = _sha256_of(tmp)
            if actual != sha:
                tmp.unlink(missing_ok=True)
                raise PTDBackendError(
                    f"SHA-256 mismatch for '{compute_id}' parent: "
                    f"expected {sha}, got {actual}")
            _peek_format_revision(tmp)  # raises on bad/missing magic
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

        for scc in entry.get("artifacts", {}).get("scc", []):
            self._fetch_scc(compute_id, scc, cache_dir)

        logger.info(
            "Fetched compute artifact '%s' (%d-byte parent, hash %s...)",
            compute_id, target.stat().st_size, hash_hex[:16])
        return True

    def _resolve_artifact_url(self, cid_or_path: str) -> str:
        """Turn ``cid_or_path`` into a fetchable URL.

        - Any ``<scheme>://...`` URL is returned as-is (so tests can
          inject ``file://``).
        - Anything else is treated as a path relative to the registry
          repo and rewritten to a ``raw.githubusercontent.com`` URL.
        """
        if "://" in cid_or_path:
            return cid_or_path
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
            _http_fetch(url, tmp)
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
    # Push (publish)
    # ------------------------------------------------------------------

    def push(
        self,
        graph: Graph,
        *,
        compute_id: str,
        metadata: dict[str, Any],
        dry_run: bool = False,
        overwrite_branch: bool = False,
    ) -> str:
        """Publish *graph*'s compute artifact.

        Internal counterpart of ``Graph.push_cache``.

        If ``dry_run`` is True, returns a JSON string of the would-be
        registry entry without touching the network. Otherwise clones
        the registry repo, splices the entry in, pushes a branch, and
        opens a PR — returning the PR URL.

        If a branch named ``phasic-publish/<compute_id>`` already exists
        on the remote, the push refuses unless ``overwrite_branch=True``,
        in which case a force-with-lease push replaces it.
        """
        # Fast-fail BEFORE any work: refresh the index and check whether
        # this graph's hash is already registered (under any id) or the
        # chosen id is already taken. We re-check inside the clone too;
        # this is just to avoid expensive setup work on a known reject.
        # Skipped for dry_run since dry_run is offline-safe.
        if not dry_run:
            try:
                self.update_index()
            except PTDBackendError:
                # Network unreachable but no cached index either; fall
                # through and let _open_pr's clone-time check catch it.
                pass
            new_hash = _graph_hash_hex(graph)
            for other_id, other_entry in (self.index or {}).get(
                    "computes", {}).items():
                if other_entry.get("retracted"):
                    continue
                if other_entry.get("graph_hash") == new_hash:
                    raise PTDBackendError(
                        f"This graph (hash {new_hash[:16]}...) is already "
                        f"registered as '{other_id}' in "
                        f"{self.registry_repo}/registry.json. Use "
                        f"`graph.pull_cache()` to reuse the existing "
                        f"artifact instead of publishing a duplicate.")
                if other_id == compute_id:
                    raise PTDBackendError(
                        f"compute id '{compute_id}' already exists in "
                        f"{self.registry_repo}/registry.json (registered "
                        f"with graph_hash {other_entry.get('graph_hash', '?')[:16]}...).")

        with tempfile.TemporaryDirectory(prefix="phasic_publish_") as tmp:
            staging = Path(tmp)
            parent_path, scc_paths = _save_artifacts(graph, staging)
            entry = self._build_entry(
                graph, compute_id=compute_id, metadata=metadata,
                parent_path=parent_path, scc_paths=scc_paths)

            if dry_run:
                return json.dumps({compute_id: entry}, indent=2)

            _check_gh_ready()
            return self._open_pr(
                compute_id, entry, parent_path, scc_paths,
                overwrite_branch=overwrite_branch,
            )

    def _build_entry(
        self,
        graph: Graph,
        *,
        compute_id: str,
        metadata: dict[str, Any],
        parent_path: Path,
        scc_paths: list[Path],
    ) -> dict[str, Any]:
        """Construct the registry entry dict."""
        hash_hex = _graph_hash_hex(graph)
        if not parent_path.is_file():
            raise PTDBackendError(
                f"parent artifact missing: {parent_path}")
        _peek_format_revision(parent_path)

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
        for scc_path in scc_paths:
            scc_path = Path(scc_path)
            if not scc_path.is_file():
                continue
            stem = scc_path.stem
            scc_hash = stem[len("scc_"):] if stem.startswith("scc_") else stem
            entry["artifacts"]["scc"].append({
                "cid_or_path": f"artifacts/{scc_path.name}",
                "scc_hash": scc_hash,
                "sha256": _sha256_of(scc_path),
                "size_bytes": scc_path.stat().st_size,
            })
        return entry

    def _guard_remote_branch(
        self,
        branch: str,
        *,
        overwrite_branch: bool,
    ) -> None:
        """Refuse to push if *branch* already exists on the remote.

        Probes the GitHub API via ``gh api``. If the branch exists
        and ``overwrite_branch`` is False, raises with the two
        recovery paths (delete the stale branch, or retry with
        ``overwrite_branch=True``).
        """
        # `gh api repos/.../branches/<name>` returns HTTP 404 (exit 1
        # from gh's perspective) when the branch is absent. We don't
        # want to surface that as an error; only the success case
        # matters here.
        proc = subprocess.run(
            ["gh", "api", f"repos/{self.registry_repo}/branches/{branch}"],
            capture_output=True, text=True, timeout=15,
        )
        exists = proc.returncode == 0
        if exists and not overwrite_branch:
            raise PTDBackendError(
                f"Remote branch '{branch}' already exists in "
                f"{self.registry_repo}.\n\n"
                f"  Pick a different --id and retry, OR\n"
                f"  Delete the stale branch:\n"
                f"    gh api -X DELETE "
                f"repos/{self.registry_repo}/git/refs/heads/{branch}\n"
                f"  Or pass overwrite_branch=True to force-with-lease.")

    def _open_pr(
        self,
        compute_id: str,
        entry: dict[str, Any],
        parent_path: Path,
        scc_paths: list[Path],
        *,
        overwrite_branch: bool = False,
    ) -> str:
        """Clone the registry repo, splice in the entry, open a PR.

        Returns the PR URL printed by ``gh pr create``.
        """
        branch = f"phasic-publish/{compute_id}"
        self._guard_remote_branch(branch, overwrite_branch=overwrite_branch)
        with tempfile.TemporaryDirectory(prefix="phasic_pr_") as clone_tmp:
            clone_dir = Path(clone_tmp) / "registry"
            _run(["gh", "repo", "clone", self.registry_repo, str(clone_dir)])
            registry_path = clone_dir / "registry.json"
            if not registry_path.is_file():
                data: dict[str, Any] = {
                    "version": "2.0", "format": "ptd_pcg", "computes": {}}
            else:
                data = json.loads(registry_path.read_text())
                if "computes" not in data:
                    data = {"version": "2.0", "format": "ptd_pcg",
                            "computes": {}, **{k: v for k, v in data.items()
                                               if k not in ("traces",)}}
            if compute_id in data["computes"]:
                raise PTDBackendError(
                    f"compute id '{compute_id}' already exists in "
                    f"{self.registry_repo}/registry.json")
            new_hash = entry["graph_hash"]
            for other_id, other_entry in data["computes"].items():
                if other_entry.get("retracted"):
                    continue
                if other_entry.get("graph_hash") == new_hash:
                    raise PTDBackendError(
                        f"This graph (hash {new_hash[:16]}...) is already "
                        f"registered as '{other_id}' in "
                        f"{self.registry_repo}/registry.json. Use "
                        f"`graph.pull_cache()` to reuse the existing "
                        f"artifact instead of publishing a duplicate.")
            data["computes"][compute_id] = entry

            artifacts_dir = clone_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            shutil.copy2(parent_path, artifacts_dir / parent_path.name)
            for p in scc_paths:
                shutil.copy2(p, artifacts_dir / p.name)

            registry_path.write_text(json.dumps(data, indent=2) + "\n")

            _run(["git", "checkout", "-b", branch], cwd=clone_dir)
            _run(["git", "add", "registry.json", "artifacts/"], cwd=clone_dir)
            _run(["git", "commit", "-m",
                  f"Add compute artifact {compute_id}\n\n"
                  f"graph_hash: {entry['graph_hash']}\n"
                  f"format_revision: {entry['format_revision']}"],
                 cwd=clone_dir)
            push_cmd = ["git", "push", "-u", "origin", branch]
            if overwrite_branch:
                push_cmd.insert(2, "--force-with-lease")
            _run(push_cmd, cwd=clone_dir)

            out = _run([
                "gh", "pr", "create",
                "--title", f"Add compute artifact: {compute_id}",
                "--body", (
                    f"Adds the C-elimination compute artifact for "
                    f"`{compute_id}`.\n\n"
                    f"- graph_hash: `{entry['graph_hash']}`\n"
                    f"- format_revision: {entry['format_revision']}\n"
                    f"- vertices: "
                    f"{entry['metadata'].get('vertices', 'unspecified')}"),
            ], cwd=clone_dir)
            return out.strip()


# ============================================================================
# Module-level cache + top-level helpers
# ============================================================================


_default_registry: ComputeRegistry | None = None


def _get_default_registry() -> ComputeRegistry:
    """Return a process-wide default ComputeRegistry, constructed lazily."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ComputeRegistry()
    return _default_registry


def list_computes(
    domain: str | None = None,
    model_type: str | None = None,
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """List published compute artifacts, optionally filtered.

    Examples
    --------
    >>> import phasic
    >>> for entry in phasic.list_computes(domain='population-genetics'):
    ...     print(entry['compute_id'], entry['graph_hash'][:12])
    """
    return _get_default_registry().list_computes(
        domain=domain, model_type=model_type, tags=tags)


__all__ = ["ComputeRegistry", "list_computes"]
