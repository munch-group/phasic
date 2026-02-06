"""
IPFS-based trace repository for PtDAlgorithms.

This module provides a decentralized repository system for pre-computed
elimination traces using IPFS for content distribution and GitHub for
human-readable indexing.

Key Features:
- Progressive enhancement: Works without IPFS, better with daemon, optimal with service
- Auto-start IPFS daemon when installed
- Automatic fallback to HTTP gateways
- Content integrity via cryptographic hashing
- Offline-first with local caching
- Zero hosting costs for maintainers
- Pluggable transport backends (IPFS, S3, GCS, Azure via TransportBackend ABC)

Usage Examples
--------------
Download and use a trace:

    >>> from phasic.trace_repository import get_trace
    >>> trace = get_trace("coalescent_n5_theta1")
    >>> # Use trace with trace_to_log_likelihood()

Browse available traces:

    >>> from phasic.trace_repository import TraceRegistry
    >>> registry = TraceRegistry()
    >>> traces = registry.list_traces(domain="population-genetics")
    >>> for t in traces:
    ...     print(f"{t['trace_id']}: {t['description']}")

Publish a new trace:

    >>> from phasic.trace_elimination import record_elimination_trace
    >>> trace = record_elimination_trace(graph, theta_dim=1)
    >>> registry.publish_trace(
    ...     trace=trace,
    ...     trace_id="my_model",
    ...     metadata={...},
    ...     submit_pr=True
    ... )

Architecture
------------
The system uses a three-tier approach:

1. **Tier 1 (Zero Config)**: HTTP gateways work out of the box
2. **Tier 2 (Auto-Start)**: Python auto-starts daemon when IPFS installed
3. **Tier 3 (Optimal)**: IPFS installed as system service

References
----------
- IPFS Documentation: https://docs.ipfs.tech/
- PtDAlgorithms Paper: Røikjer, Hobolth & Munch (2022)
  https://doi.org/10.1007/s11222-022-10155-6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import gzip
import hashlib
import os
import re
import warnings
import time
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from .logging_config import get_logger
from .exceptions import PTDBackendError

if TYPE_CHECKING:
    from .trace_elimination import EliminationTrace

logger = get_logger(__name__)

# Try importing ipfshttpclient (optional dependency)
try:
    import ipfshttpclient
    HAS_IPFS_CLIENT = True
except ImportError:
    HAS_IPFS_CLIENT = False

# Trace format version for forward compatibility
TRACE_FORMAT_VERSION = "1.0"

# Required keys in serialized trace dicts
_REQUIRED_TRACE_KEYS = frozenset({
    'operations', 'vertex_rates', 'edge_probs', 'vertex_targets',
    'states', 'starting_vertex_idx', 'n_vertices', 'param_length',
    'state_length',
})

# Required keys for registry.json top-level
_REQUIRED_REGISTRY_KEYS = frozenset({'version', 'traces'})


# ============================================================================
# Transport Backend ABC
# ============================================================================

class TransportBackend(ABC):
    """
    Abstract base class for trace transport backends.

    Implementations provide content-addressed storage for trace packages.
    The default implementation is :class:`IPFSBackend`; future backends
    may target S3, GCS, or Azure Blob Storage.

    Subclasses must implement :meth:`get`, :meth:`add`, and the
    :attr:`name` property.  Optionally override
    :attr:`supports_directories` and :meth:`get_directory`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name (e.g. ``'ipfs'``)."""

    @abstractmethod
    def get(self, cid: str, output_path: Path | None = None) -> bytes:
        """
        Retrieve content by content identifier.

        Parameters
        ----------
        cid : str
            Content identifier (CID for IPFS, key for cloud stores).
        output_path : Path, optional
            Write content to this file instead of returning bytes.

        Returns
        -------
        bytes or None
            Content data, or ``None`` when *output_path* is given.
        """

    @abstractmethod
    def add(self, path: Path) -> str:
        """
        Publish a file or directory and return its content identifier.

        Parameters
        ----------
        path : Path
            Local file or directory.

        Returns
        -------
        str
            Content identifier for the published content.
        """

    @property
    def supports_directories(self) -> bool:
        """Whether this backend can retrieve whole directories."""
        return False

    def get_directory(self, cid: str, output_dir: Path) -> None:
        """
        Retrieve an entire directory by CID.

        The default implementation raises :class:`NotImplementedError`.
        Override in backends that support directory operations.
        """
        raise NotImplementedError(
            f"{self.name} backend does not support directory retrieval"
        )


# ============================================================================
# CID / Path Validation Helpers
# ============================================================================

# CIDv0: starts with Qm, 46 chars, base58
_CIDV0_RE = re.compile(r'^Qm[1-9A-HJ-NP-Za-km-z]{44}$')
# CIDv1: starts with bafy (base32) or b (other base), variable length
_CIDV1_RE = re.compile(r'^b[a-z2-7]{58,}$')


def _validate_cid(cid: str) -> None:
    """
    Validate an IPFS CID, optionally with a ``CID/subpath`` suffix.

    Raises
    ------
    ValueError
        If the CID is empty, contains path-traversal components, or
        does not look like a valid CIDv0 or CIDv1.
    """
    if not cid or not isinstance(cid, str):
        raise ValueError("CID must be a non-empty string")

    # Split CID from optional subpath
    parts = cid.split('/', 1)
    bare_cid = parts[0]

    # Validate bare CID format
    if not (_CIDV0_RE.match(bare_cid) or _CIDV1_RE.match(bare_cid)):
        raise ValueError(
            f"Invalid CID format: {bare_cid!r}. "
            "Expected CIDv0 (Qm..., 46 chars) or CIDv1 (bafy..., base32)."
        )

    # Validate subpath if present
    if len(parts) == 2:
        subpath = parts[1]
        for component in subpath.split('/'):
            if component in ('', '.', '..'):
                raise ValueError(
                    f"Invalid subpath in CID: {cid!r}. "
                    "Path components must not be empty, '.', or '..'."
                )


def _validate_output_path(output_path: Path, allowed_root: Path) -> None:
    """
    Ensure *output_path* resolves inside *allowed_root*.

    Raises
    ------
    ValueError
        If the resolved path escapes *allowed_root*.
    """
    resolved = output_path.resolve()
    root_resolved = allowed_root.resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(
            f"Output path {output_path} resolves outside allowed "
            f"directory {allowed_root}"
        )


# ============================================================================
# Swarm Key Helpers (Private IPFS Networks)
# ============================================================================

# Expected swarm key format:
#   /key/swarm/psk/1.0.0/
#   /base16/
#   <64 hex chars>
_SWARM_KEY_RE = re.compile(
    r'^/key/swarm/psk/1\.0\.0/\n/base16/\n[0-9a-f]{64}\n?$'
)


def get_ipfs_dir() -> Path:
    """
    Return the IPFS configuration directory.

    Respects the ``IPFS_PATH`` environment variable.  Falls back to
    ``~/.ipfs`` when the variable is not set.

    Returns
    -------
    Path
        Absolute path to the IPFS config directory.
    """
    env = os.environ.get('IPFS_PATH')
    if env:
        return Path(env)
    return Path.home() / ".ipfs"


def _validate_swarm_key_content(content: str) -> None:
    """
    Validate a swarm key string against the expected 3-line format.

    Raises
    ------
    ValueError
        If the content does not match the expected format.
    """
    if not _SWARM_KEY_RE.match(content):
        raise ValueError(
            "Invalid swarm key format. Expected:\n"
            "  /key/swarm/psk/1.0.0/\n"
            "  /base16/\n"
            "  <64 hex characters>"
        )


def generate_swarm_key() -> str:
    """
    Generate a new 32-byte swarm key for a private IPFS network.

    The returned string is in the standard 3-line format expected by
    the ``go-ipfs`` daemon (written to ``swarm.key``).

    Returns
    -------
    str
        Swarm key in the format::

            /key/swarm/psk/1.0.0/
            /base16/
            <64 hex characters>
    """
    key_bytes = os.urandom(32)
    hex_key = key_bytes.hex()
    return f"/key/swarm/psk/1.0.0/\n/base16/\n{hex_key}\n"


def install_swarm_key(key_content: str, ipfs_dir: Path | None = None) -> Path:
    """
    Install a swarm key into the IPFS configuration directory.

    The key file is written with mode ``0o600`` (owner read/write only).
    Refuses to overwrite an existing ``swarm.key`` file.

    Parameters
    ----------
    key_content : str
        Swarm key content (3-line format from :func:`generate_swarm_key`).
    ipfs_dir : Path, optional
        IPFS configuration directory.  Defaults to :func:`get_ipfs_dir`.

    Returns
    -------
    Path
        Path to the installed ``swarm.key`` file.

    Raises
    ------
    ValueError
        If *key_content* is not valid swarm key format.
    FileExistsError
        If ``swarm.key`` already exists in the target directory.
    """
    _validate_swarm_key_content(key_content)

    target_dir = ipfs_dir or get_ipfs_dir()
    key_path = target_dir / "swarm.key"

    if key_path.exists():
        raise FileExistsError(
            f"Swarm key already exists at {key_path}. "
            "Remove it first with remove_swarm_key() to replace."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key_content)
    key_path.chmod(0o600)

    logger.info("Installed swarm key at %s", key_path)
    return key_path


def detect_swarm_key(ipfs_dir: Path | None = None) -> Path | None:
    """
    Check whether a swarm key is installed.

    Parameters
    ----------
    ipfs_dir : Path, optional
        IPFS configuration directory.  Defaults to :func:`get_ipfs_dir`.

    Returns
    -------
    Path or None
        Path to ``swarm.key`` if it exists, ``None`` otherwise.
    """
    target_dir = ipfs_dir or get_ipfs_dir()
    key_path = target_dir / "swarm.key"
    if key_path.exists():
        return key_path
    return None


def remove_swarm_key(ipfs_dir: Path | None = None) -> bool:
    """
    Remove the swarm key to return the IPFS node to the public network.

    Parameters
    ----------
    ipfs_dir : Path, optional
        IPFS configuration directory.  Defaults to :func:`get_ipfs_dir`.

    Returns
    -------
    bool
        ``True`` if a key was removed, ``False`` if no key was found.
    """
    target_dir = ipfs_dir or get_ipfs_dir()
    key_path = target_dir / "swarm.key"
    if key_path.exists():
        key_path.unlink()
        logger.info("Removed swarm key from %s", target_dir)
        return True
    return False


def configure_bootstrap_peers(
    peers: list[str],
    ipfs_dir: Path | None = None,
) -> None:
    """
    Replace the IPFS bootstrap peer list.

    This removes all existing bootstrap peers and adds the given ones.
    Useful for private networks where only specific peers should be
    discoverable.

    Parameters
    ----------
    peers : list[str]
        Multiaddr strings of bootstrap peers to add.
    ipfs_dir : Path, optional
        IPFS configuration directory (passed via ``IPFS_PATH`` env).

    Raises
    ------
    ValueError
        If *peers* is empty.
    FileNotFoundError
        If the ``ipfs`` binary is not installed.
    subprocess.SubprocessError
        If a bootstrap command fails.
    """
    if not peers:
        raise ValueError("peers list must not be empty")

    ipfs_path = shutil.which('ipfs')
    if not ipfs_path:
        raise FileNotFoundError(
            "IPFS binary not found. Install IPFS first.\n"
            "  macOS:  brew install ipfs\n"
            "  Linux:  See https://docs.ipfs.tech/install/"
        )

    env = dict(os.environ)
    if ipfs_dir is not None:
        env['IPFS_PATH'] = str(ipfs_dir)

    # Remove all existing bootstrap peers
    subprocess.run(
        [ipfs_path, 'bootstrap', 'rm', '--all'],
        capture_output=True, check=True, timeout=30, env=env,
    )

    # Add each peer
    for peer in peers:
        subprocess.run(
            [ipfs_path, 'bootstrap', 'add', peer],
            capture_output=True, check=True, timeout=30, env=env,
        )

    logger.info("Configured %d bootstrap peers", len(peers))


# ============================================================================
# IPFS Backend with Auto-Start and Gateway Fallback
# ============================================================================

class IPFSBackend(TransportBackend):
    """
    IPFS backend with automatic daemon management and HTTP gateway fallback.

    This class provides a progressive enhancement strategy:
    - Tries to connect to existing IPFS daemon
    - Auto-starts daemon if IPFS is installed but not running
    - Falls back to HTTP gateways if IPFS not available

    Parameters
    ----------
    daemon_addr : str, default="/ip4/127.0.0.1/tcp/5001"
        IPFS daemon API address
    gateways : list[str], optional
        List of HTTP gateway URLs. If None, uses default public gateways.
    auto_start : bool, default=True
        If True, automatically start IPFS daemon if installed but not running
    timeout : int, default=30
        Timeout in seconds for IPFS operations

    Attributes
    ----------
    client : ipfshttpclient.Client or None
        IPFS HTTP client if daemon available, None otherwise
    daemon_process : subprocess.Popen or None
        Process handle for auto-started daemon, None if not started
    gateways : list[str]
        List of HTTP gateway URLs for fallback

    Examples
    --------
    >>> # Auto-start daemon if available, fallback to gateways
    >>> backend = IPFSBackend()
    Started IPFS daemon automatically

    >>> # Download content
    >>> content = backend.get("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")

    >>> # Or use gateway fallback
    >>> backend = IPFSBackend(auto_start=False)
    IPFS daemon not running. Using HTTP gateways.
    >>> content = backend.get("bafybeig...")
    """

    @property
    def name(self) -> str:
        return "ipfs"

    @property
    def supports_directories(self) -> bool:
        return self.client is not None

    def __init__(
        self,
        daemon_addr: str = "/ip4/127.0.0.1/tcp/5001",
        gateways: list[str] | None = None,
        auto_start: bool = True,
        timeout: int = 30,
    ) -> None:
        self.daemon_addr = daemon_addr
        self.timeout = timeout
        self.auto_start = auto_start
        self.client = None
        self.daemon_process = None
        self._ipfs_dir = get_ipfs_dir()

        # Detect private network mode
        self.private_network = detect_swarm_key(self._ipfs_dir) is not None

        # Configure HTTP gateway fallbacks
        user_passed_gateways = gateways is not None
        if gateways is None:
            self.gateways = [
                "https://ipfs.io",
                "https://cloudflare-ipfs.com",
                "https://dweb.link",
                "https://gateway.pinata.cloud"
            ]
        else:
            self.gateways = gateways

        # In private network mode, public gateways cannot access content
        if self.private_network:
            if user_passed_gateways and self.gateways:
                logger.warning(
                    "Swarm key detected: public HTTP gateways cannot access "
                    "private network content. Gateways will be disabled."
                )
            self.gateways = []
            logger.info(
                "Private IPFS network detected (swarm key at %s). "
                "HTTP gateways disabled.",
                self._ipfs_dir / "swarm.key",
            )

        # Try connecting to IPFS daemon
        if HAS_IPFS_CLIENT:
            try:
                # Try existing daemon first
                self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                version = self.client.version()
                logger.info("Connected to IPFS daemon (version %s)", version['Version'])
            except Exception as e:
                # Try auto-starting daemon
                if auto_start and self._start_daemon():
                    time.sleep(2)  # Give daemon time to initialize
                    try:
                        self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                        logger.info("Started IPFS daemon automatically")
                    except Exception:
                        warnings.warn("IPFS daemon started but connection failed. Using HTTP gateways.")
                else:
                    if not auto_start:
                        logger.info("IPFS daemon not running. Using HTTP gateways.")
                    else:
                        warnings.warn(f"IPFS not available. Using HTTP gateways.")
        else:
            logger.info("ipfshttpclient not installed. Using HTTP gateways only.")
            logger.info("  Install for faster downloads: pip install ipfshttpclient")

    def _start_daemon(self) -> bool:
        """
        Attempt to start IPFS daemon in background.

        Returns
        -------
        bool
            True if daemon started successfully, False otherwise
        """
        # Check if ipfs is installed
        ipfs_path = shutil.which('ipfs')
        if not ipfs_path:
            return False

        try:
            # Check if daemon already running (cross-platform)
            if self._is_daemon_running():
                return True

            # Check if IPFS is initialized
            ipfs_dir = self._ipfs_dir
            if not ipfs_dir.exists():
                # Initialize IPFS
                subprocess.run(
                    [ipfs_path, 'init'],
                    capture_output=True,
                    check=True,
                    timeout=30
                )
                logger.info("Initialized IPFS repository")

            # Start daemon in background (detached from parent)
            self.daemon_process = subprocess.Popen(
                [ipfs_path, 'daemon'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )

            return True
        except (subprocess.SubprocessError, OSError) as e:
            warnings.warn(f"Failed to start IPFS daemon: {e}")
            return False

    @staticmethod
    def _is_daemon_running() -> bool:
        """Check if an IPFS daemon process is running (cross-platform)."""
        try:
            if sys.platform == 'win32':
                result = subprocess.run(
                    ['tasklist', '/FI', 'IMAGENAME eq ipfs.exe'],
                    capture_output=True, text=True, timeout=10
                )
                return 'ipfs.exe' in result.stdout
            else:
                # Unix: use ps instead of pgrep for portability
                result = subprocess.run(
                    ['ps', '-eo', 'comm'],
                    capture_output=True, text=True, timeout=10
                )
                return 'ipfs' in result.stdout.split()
        except (subprocess.SubprocessError, OSError):
            return False

    def get(self, cid: str, output_path: Path | None = None) -> bytes:
        """
        Get content from IPFS by CID.

        Tries local daemon first, falls back to HTTP gateways.

        Parameters
        ----------
        cid : str
            IPFS Content Identifier (CID)
        output_path : Path, optional
            If provided, write content to this file instead of returning bytes

        Returns
        -------
        bytes
            Content data (if output_path is None)

        Raises
        ------
        PTDBackendError
            If content cannot be retrieved from any source
        ValueError
            If *cid* is not a valid CID
        """
        _validate_cid(cid)

        # Try local daemon first
        if self.client is not None:
            try:
                content = self.client.cat(cid, timeout=self.timeout)
                if output_path is not None:
                    output_path.write_bytes(content)
                    return None
                return content
            except Exception as e:
                logger.debug("IPFS daemon failed, trying gateways: %s", e)

        # In private network mode with no gateways, the daemon is required
        if self.private_network and not self.gateways:
            raise PTDBackendError(
                "Private IPFS network requires a running daemon to retrieve content.\n"
                "Public HTTP gateways cannot access private network content.\n"
                "  Start daemon with: ipfs daemon"
            )

        # Fallback to HTTP gateways
        return self._get_via_gateway(cid, output_path)

    def _get_via_gateway(self, cid: str, output_path: Path | None = None) -> bytes:
        """Retrieve content through HTTP gateways with retry logic."""
        last_error = None
        for gateway in self.gateways:
            url = f"{gateway}/ipfs/{cid}"
            try:
                response = _request_with_retry(url, timeout=self.timeout)
                content = response.content
                if output_path is not None:
                    output_path.write_bytes(content)
                    return None
                return content
            except Exception as e:
                logger.debug("Gateway %s failed: %s", gateway, e)
                last_error = e
                continue

        raise PTDBackendError(
            f"Failed to retrieve {cid} from IPFS daemon and all HTTP gateways"
        )

    def get_directory(self, cid: str, output_dir: Path) -> None:
        """
        Get entire directory from IPFS by CID.

        Parameters
        ----------
        cid : str
            IPFS Content Identifier (CID) for directory
        output_dir : Path
            Local directory to write contents

        Raises
        ------
        PTDBackendError
            If directory cannot be retrieved
        ValueError
            If *cid* is not a valid CID
        """
        _validate_cid(cid)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try local daemon first
        if self.client is not None:
            try:
                self.client.get(cid, target=str(output_dir))
                return
            except Exception as e:
                logger.debug("IPFS daemon failed for directory, trying gateways: %s", e)

        # For gateways, we need to know the file structure
        raise PTDBackendError(
            "Directory download via HTTP gateways requires knowing file structure. "
            "Install IPFS daemon for full functionality."
        )

    def add(self, path: Path) -> str:
        """
        Add file or directory to IPFS.

        Parameters
        ----------
        path : Path
            Path to file or directory to add

        Returns
        -------
        str
            CID of added content

        Raises
        ------
        PTDBackendError
            If IPFS daemon not available (publishing requires daemon)
        """
        if self.client is None:
            raise PTDBackendError(
                "Publishing requires IPFS daemon. Install IPFS and try again.\n"
                "  macOS:  brew install ipfs\n"
                "  Linux:  See https://docs.ipfs.tech/install/"
            )

        try:
            result = self.client.add(str(path), recursive=path.is_dir())
            if isinstance(result, list):
                # Multiple files - return last (root) CID
                return result[-1]['Hash']
            else:
                return result['Hash']
        except Exception as e:
            raise PTDBackendError(f"Failed to add to IPFS: {e}")

    # ----------------------------------------------------------------
    # Daemon status
    # ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """
        Return a summary of the backend's connection state.

        Returns
        -------
        dict[str, Any]
            Keys: ``daemon`` (bool), ``daemon_version`` (str or None),
            ``gateways`` (list of gateway URLs), ``ipfs_installed`` (bool).
        """
        swarm_key_path = detect_swarm_key(self._ipfs_dir)
        info: dict[str, Any] = {
            "daemon": self.client is not None,
            "daemon_version": None,
            "gateways": list(self.gateways),
            "ipfs_installed": shutil.which('ipfs') is not None,
            "private_network": self.private_network,
            "swarm_key_path": str(swarm_key_path) if swarm_key_path else None,
        }
        if self.client is not None:
            try:
                ver = self.client.version()
                info["daemon_version"] = ver.get("Version")
            except Exception:
                pass
        return info

    # ----------------------------------------------------------------
    # Pinning
    # ----------------------------------------------------------------

    def pin(self, cid: str) -> None:
        """
        Pin *cid* on the local IPFS node so it is not garbage-collected.

        Parameters
        ----------
        cid : str
            Content identifier to pin.

        Raises
        ------
        PTDBackendError
            If no daemon is available.
        """
        _validate_cid(cid)
        if self.client is None:
            raise PTDBackendError(
                "Pinning requires a running IPFS daemon.\n"
                "  Start one with: ipfs daemon"
            )
        try:
            self.client.pin.add(cid)
            logger.info("Pinned %s", cid)
        except Exception as e:
            raise PTDBackendError(f"Failed to pin {cid}: {e}")

    def unpin(self, cid: str) -> None:
        """
        Unpin *cid* from the local IPFS node.

        After unpinning the content may be garbage-collected by the IPFS
        daemon, freeing local disk space.

        Parameters
        ----------
        cid : str
            Content identifier to unpin.

        Raises
        ------
        PTDBackendError
            If no daemon is available.
        """
        _validate_cid(cid)
        if self.client is None:
            raise PTDBackendError(
                "Unpinning requires a running IPFS daemon.\n"
                "  Start one with: ipfs daemon"
            )
        try:
            self.client.pin.rm(cid)
            logger.info("Unpinned %s", cid)
        except Exception as e:
            raise PTDBackendError(f"Failed to unpin {cid}: {e}")

    def is_pinned(self, cid: str) -> bool:
        """
        Check whether *cid* is pinned on the local IPFS node.

        Returns
        -------
        bool
            ``True`` if pinned, ``False`` otherwise (including when no
            daemon is available).
        """
        if self.client is None:
            return False
        try:
            pins = self.client.pin.ls(type='all')
            return cid in pins.get('Keys', {})
        except Exception:
            return False

    def __del__(self) -> None:
        """Clean up on object destruction.

        Note: We intentionally do NOT kill the daemon here, as it should
        persist for other processes and future use.
        """
        pass


# ============================================================================
# HTTP Retry Helper
# ============================================================================

def _request_with_retry(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_base: float = 1.0
) -> requests.Response:
    """
    GET *url* with exponential-backoff retry on transient failures.

    Retries on connection errors and 5xx status codes.

    Parameters
    ----------
    url : str
        URL to fetch.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Maximum number of attempts.
    backoff_base : float
        Base delay in seconds (doubles each retry).

    Returns
    -------
    requests.Response
        Successful response.

    Raises
    ------
    requests.RequestException
        After all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code >= 500 and attempt < max_retries - 1:
                # Server error — retry
                delay = backoff_base * (2 ** attempt)
                logger.debug(
                    "Server error %d from %s, retrying in %.1fs",
                    response.status_code, url, delay
                )
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.ConnectionError as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.debug("Connection error for %s, retrying in %.1fs", url, delay)
                time.sleep(delay)
            continue
        except requests.RequestException:
            raise
    # All retries exhausted
    raise last_exc  # type: ignore[misc]


# ============================================================================
# Registry Validation
# ============================================================================

def _validate_registry(data: Any) -> None:
    """
    Validate top-level registry JSON structure.

    Raises
    ------
    PTDBackendError
        If required keys are missing or ``traces`` is not a dict.
    """
    if not isinstance(data, dict):
        raise PTDBackendError("Registry JSON must be a dict")
    missing = _REQUIRED_REGISTRY_KEYS - data.keys()
    if missing:
        raise PTDBackendError(f"Registry JSON missing required keys: {missing}")
    if not isinstance(data.get('traces'), dict):
        raise PTDBackendError("Registry 'traces' field must be a dict")


# ============================================================================
# Trace Registry
# ============================================================================

class TraceRegistry:
    """
    Main API for browsing and downloading pre-computed elimination traces.

    The registry maintains a catalog of available traces stored on IPFS,
    with human-readable metadata hosted on GitHub.

    Parameters
    ----------
    registry_repo : str, default="munch-group/phasic-traces"
        GitHub repository containing registry.json
    cache_dir : Path, optional
        Local cache directory. Defaults to ~/.phasic_traces
    backend : TransportBackend, optional
        Transport backend for content retrieval.  If None, an
        :class:`IPFSBackend` is created lazily on first use.
    ipfs_backend : IPFSBackend, optional
        **Deprecated** — use *backend* instead.
    auto_update : bool, default=True
        Automatically update registry from GitHub on initialization

    Attributes
    ----------
    registry : dict
        Loaded registry data from GitHub
    cache_dir : Path
        Local cache directory for downloaded traces
    backend : TransportBackend
        Transport backend for content retrieval

    Examples
    --------
    >>> # Browse available traces
    >>> registry = TraceRegistry()
    >>> traces = registry.list_traces(domain="population-genetics")
    >>> print(traces[0])
    {
        'trace_id': 'coalescent_n5_theta1',
        'description': 'Kingman coalescent, n=5 samples, 1 parameter',
        'cid': 'bafybeig...',
        'vertices': 5,
        'param_length': 1
    }

    >>> # Download and use trace
    >>> trace = registry.get_trace("coalescent_n5_theta1")
    >>> # trace is now loaded and ready for use
    """

    REGISTRY_URL = "https://raw.githubusercontent.com/{}/master/registry.json"

    def __init__(
        self,
        registry_repo: str = "munch-group/phasic-traces",
        cache_dir: Path | None = None,
        backend: TransportBackend | None = None,
        ipfs_backend: IPFSBackend | None = None,
        auto_update: bool = True,
    ) -> None:
        self.registry_repo = registry_repo
        self.cache_dir = cache_dir or (Path.home() / ".phasic_traces")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Handle deprecated ipfs_backend parameter
        if ipfs_backend is not None:
            if backend is not None:
                raise ValueError(
                    "Cannot specify both 'backend' and 'ipfs_backend'. "
                    "Use 'backend' (ipfs_backend is deprecated)."
                )
            warnings.warn(
                "The 'ipfs_backend' parameter is deprecated. "
                "Use 'backend' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            backend = ipfs_backend

        # Store backend (or None for lazy init)
        self._backend = backend

        # Registry data
        self.registry = None

        # Load cached registry
        self._load_cached_registry()

        # Update from GitHub if requested
        if auto_update:
            self.update_registry()

    @property
    def backend(self) -> TransportBackend:
        """Transport backend, lazily initialised on first access."""
        if self._backend is None:
            self._backend = IPFSBackend()
        return self._backend

    # Backward-compatible alias
    @property
    def ipfs(self) -> TransportBackend:
        """Deprecated alias for :attr:`backend`."""
        return self.backend

    def _load_cached_registry(self) -> None:
        """Load registry from local cache if available."""
        registry_path = self.cache_dir / "registry.json"
        if registry_path.exists():
            try:
                data = json.loads(registry_path.read_text())
                _validate_registry(data)
                self.registry = data
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning("Failed to load cached registry: %s", e)

    def update_registry(self) -> None:
        """
        Update registry from GitHub.

        Downloads the latest registry.json from the configured GitHub repository.
        """
        url = self.REGISTRY_URL.format(self.registry_repo)
        try:
            logger.info("Updating registry from %s...", self.registry_repo)
            response = _request_with_retry(url, timeout=10)
            data = response.json()
            _validate_registry(data)
            self.registry = data

            # Save to cache
            registry_path = self.cache_dir / "registry.json"
            registry_path.write_text(json.dumps(self.registry, indent=2))
            logger.info("Registry updated")
        except Exception as e:
            if self.registry is None:
                raise PTDBackendError(
                    f"Failed to download registry from {url}: {e}\n"
                    "  Ensure you have internet connection and the repository exists."
                )
            else:
                warnings.warn(f"Failed to update registry, using cached version: {e}")

    def list_traces(
        self,
        domain: str | None = None,
        model_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        List available traces with optional filtering.

        Parameters
        ----------
        domain : str, optional
            Filter by domain (e.g., "population-genetics")
        model_type : str, optional
            Filter by model type (e.g., "coalescent")
        tags : list[str], optional
            Filter by tags (all must match)

        Returns
        -------
        list[dict[str, Any]]
            List of trace metadata dictionaries
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded. Call update_registry() first.")

        traces = []
        for trace_id, info in self.registry['traces'].items():
            # Skip retracted traces
            if info.get('retracted'):
                continue

            metadata = info.get('metadata', {})

            # Apply filters
            if domain is not None and metadata.get('domain') != domain:
                continue
            if model_type is not None and metadata.get('model_type') != model_type:
                continue
            if tags is not None:
                trace_tags = metadata.get('tags', [])
                if not all(tag in trace_tags for tag in tags):
                    continue

            # Build result entry
            traces.append({
                'trace_id': trace_id,
                'description': info.get('description', ''),
                'cid': info.get('cid', ''),
                **metadata
            })

        return traces

    def get_trace(self, trace_id: str, force_download: bool = False) -> EliminationTrace:
        """
        Download and load a trace by ID.

        Parameters
        ----------
        trace_id : str
            Trace identifier (e.g., "coalescent_n5_theta1")
        force_download : bool, default=False
            If True, re-download even if cached locally

        Returns
        -------
        EliminationTrace
            Loaded trace object

        Raises
        ------
        PTDBackendError
            If trace not found in registry or download fails

        Examples
        --------
        >>> trace = registry.get_trace("coalescent_n5_theta1")
        >>> # Use with trace_to_log_likelihood
        >>> from phasic.trace_elimination import trace_to_log_likelihood
        >>> log_lik = trace_to_log_likelihood(trace, observed_times)
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded. Call update_registry() first.")

        # Look up trace in registry
        if trace_id not in self.registry['traces']:
            raise PTDBackendError(
                f"Trace '{trace_id}' not found in registry.\n"
                f"  Available traces: {list(self.registry['traces'].keys())}\n"
                f"  Use list_traces() to browse."
            )

        trace_info = self.registry['traces'][trace_id]

        # Check if trace has been retracted
        if trace_info.get('retracted'):
            reason = trace_info.get('retracted_reason', 'No reason given')
            raise PTDBackendError(
                f"Trace '{trace_id}' has been retracted: {reason}\n"
                f"  This trace is no longer available for download."
            )

        # Check local cache
        trace_cache_dir = self.cache_dir / "traces" / trace_id
        trace_file = trace_cache_dir / "trace.json.gz"

        if not force_download and trace_file.exists():
            logger.info("Using cached trace: %s", trace_file)
        else:
            # Download from backend
            logger.info("Downloading trace '%s'...", trace_id)
            trace_cache_dir.mkdir(parents=True, exist_ok=True)

            # Validate output path stays within cache_dir
            _validate_output_path(trace_file, self.cache_dir)

            # Get the trace.json.gz file from the directory CID
            dir_cid = trace_info['cid']
            file_path_in_ipfs = f"{dir_cid}/trace.json.gz"
            self.backend.get(file_path_in_ipfs, output_path=trace_file)

            # Verify checksum if available
            expected_checksum = trace_info.get('checksum')
            if expected_checksum:
                self._verify_checksum(trace_file, expected_checksum)

            logger.info("Downloaded to %s", trace_file)

        # Load and decompress trace
        with gzip.open(trace_file, 'rt') as f:
            trace_dict = json.load(f)

        # Use helper to deserialize
        return self._deserialize_trace(trace_dict)

    def _verify_checksum(self, file_path: Path, expected: str) -> None:
        """
        Verify SHA-256 checksum of a downloaded file.

        Parameters
        ----------
        file_path : Path
            Path to the file to verify.
        expected : str
            Expected checksum in ``sha256:<hex>`` format.

        Raises
        ------
        PTDBackendError
            If the checksum does not match (corrupted file is deleted).
        """
        if not expected.startswith('sha256:'):
            logger.warning("Unknown checksum format: %s", expected)
            return

        expected_hex = expected[len('sha256:'):]
        actual_hex = hashlib.sha256(file_path.read_bytes()).hexdigest()

        if actual_hex != expected_hex:
            file_path.unlink(missing_ok=True)
            raise PTDBackendError(
                f"Checksum mismatch for {file_path.name}: "
                f"expected {expected_hex[:16]}..., got {actual_hex[:16]}..."
            )
        logger.debug("Checksum verified for %s", file_path.name)

    def get_trace_by_hash(self, graph_hash: str, force_download: bool = False) -> EliminationTrace | None:
        """
        Download and load a trace by graph structure hash.

        This enables automatic trace discovery: build a graph, compute its hash,
        and check if someone has already computed the elimination trace.

        Parameters
        ----------
        graph_hash : str
            SHA-256 hash of graph structure (64 hex characters)
        force_download : bool, default=False
            If True, re-download even if cached locally

        Returns
        -------
        EliminationTrace or None
            Loaded trace if found, None otherwise

        Examples
        --------
        >>> from phasic import Graph
        >>> import phasic.hash
        >>> graph = Graph(my_callback, nr_samples=5)
        >>> hash_result = phasic.hash.compute_graph_hash(graph)
        >>> trace = registry.get_trace_by_hash(hash_result.hash_hex)
        >>> if trace:
        ...     print("Found existing trace!")
        ... else:
        ...     print("Need to record new trace")
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded. Call update_registry() first.")

        # Check local hash-based cache first
        hash_cache_dir = self.cache_dir / "by_hash" / graph_hash
        trace_file = hash_cache_dir / "trace.json.gz"

        if not force_download and trace_file.exists():
            logger.info("Using cached trace (by hash): %s", trace_file)
            # Load from cache
            with gzip.open(trace_file, 'rt') as f:
                trace_dict = json.load(f)
            return self._deserialize_trace(trace_dict)

        # Search registry for matching graph_hash
        for trace_id, info in self.registry['traces'].items():
            if info.get('graph_hash') == graph_hash:
                logger.info("Found trace '%s' with matching hash", trace_id)
                # Download using existing get_trace method
                trace = self.get_trace(trace_id, force_download=force_download)

                # Also cache by hash for faster future lookups
                hash_cache_dir.mkdir(parents=True, exist_ok=True)
                trace_by_id_file = self.cache_dir / "traces" / trace_id / "trace.json.gz"
                if trace_by_id_file.exists():
                    shutil.copy(trace_by_id_file, trace_file)
                    logger.debug("Cached by hash: %s", hash_cache_dir)

                return trace

        # Not found in registry
        logger.info("No trace found for hash %s...", graph_hash[:16])
        return None

    # ----------------------------------------------------------------
    # Source & cache inspection
    # ----------------------------------------------------------------

    def trace_source(self, trace_id: str) -> dict[str, Any]:
        """
        Report where a trace would be loaded from, without downloading it.

        Parameters
        ----------
        trace_id : str
            Trace identifier.

        Returns
        -------
        dict[str, Any]
            ``source`` is one of ``"local_cache"``, ``"ipfs_daemon"``,
            ``"http_gateway"``, or ``"unavailable"``.  Additional keys
            give the ``cache_path``, ``cid``, and ``backend`` name.

        Raises
        ------
        PTDBackendError
            If the registry is not loaded or *trace_id* is unknown.
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded. Call update_registry() first.")
        if trace_id not in self.registry['traces']:
            raise PTDBackendError(f"Trace '{trace_id}' not found in registry.")

        trace_info = self.registry['traces'][trace_id]
        cache_path = self.cache_dir / "traces" / trace_id / "trace.json.gz"

        is_private = getattr(self.backend, 'private_network', False)

        info: dict[str, Any] = {
            "trace_id": trace_id,
            "cid": trace_info.get("cid"),
            "cache_path": str(cache_path),
            "cached": cache_path.exists(),
            "backend": self.backend.name,
            "private_network": is_private,
        }

        if cache_path.exists():
            info["source"] = "local_cache"
        elif is_private and getattr(self.backend, 'client', None) is None:
            info["source"] = "unavailable"
        elif self.backend.name == "ipfs" and getattr(self.backend, 'client', None) is not None:
            info["source"] = "ipfs_daemon"
        elif self.backend.name == "ipfs":
            info["source"] = "http_gateway"
        else:
            info["source"] = self.backend.name

        return info

    def clear_cache(self, trace_id: str | None = None) -> int:
        """
        Delete locally cached trace files.

        Parameters
        ----------
        trace_id : str, optional
            If given, clear only this trace.  If ``None``, clear **all**
            cached traces.

        Returns
        -------
        int
            Number of cache entries removed.
        """
        removed = 0
        if trace_id is not None:
            cache_path = self.cache_dir / "traces" / trace_id
            if cache_path.exists():
                shutil.rmtree(cache_path)
                removed += 1
                logger.info("Cleared cache for %s", trace_id)
        else:
            traces_dir = self.cache_dir / "traces"
            if traces_dir.exists():
                for child in traces_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                        removed += 1
                logger.info("Cleared %d cached traces", removed)
            by_hash_dir = self.cache_dir / "by_hash"
            if by_hash_dir.exists():
                for child in by_hash_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child)
                        removed += 1
        return removed

    # ----------------------------------------------------------------
    # Retraction
    # ----------------------------------------------------------------

    def retract_trace(self, trace_id: str, reason: str = "") -> None:
        """
        Mark a trace as retracted in the **local** registry cache.

        This does **not** modify the upstream GitHub registry.  To
        permanently retract a faulty trace, submit a pull request that
        adds ``"retracted": true`` to the entry in ``registry.json``.

        After retraction, :meth:`get_trace` will refuse to load the
        trace and :meth:`list_traces` will skip it.

        Parameters
        ----------
        trace_id : str
            Trace to retract.
        reason : str, optional
            Free-form explanation (stored locally).

        Raises
        ------
        PTDBackendError
            If the registry is not loaded or *trace_id* is unknown.
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded.")
        if trace_id not in self.registry['traces']:
            raise PTDBackendError(f"Trace '{trace_id}' not found in registry.")

        self.registry['traces'][trace_id]['retracted'] = True
        if reason:
            self.registry['traces'][trace_id]['retracted_reason'] = reason

        # Persist to local cache
        registry_path = self.cache_dir / "registry.json"
        registry_path.write_text(json.dumps(self.registry, indent=2))

        # Also delete cached trace data
        self.clear_cache(trace_id)

        logger.info("Retracted trace '%s': %s", trace_id, reason or "(no reason given)")

    def _deserialize_trace(self, trace_dict: dict[str, Any]) -> EliminationTrace:
        """
        Deserialize a trace dict into an :class:`EliminationTrace`.

        Validates required keys, operation types, and integer constraints
        before constructing the trace object.

        Raises
        ------
        PTDBackendError
            If the trace dict is missing required keys, contains invalid
            operation types, or has negative integer fields.
        """
        from .trace_elimination import EliminationTrace, Operation, OpType
        import numpy as np

        # ---- Validate required keys ----
        missing = _REQUIRED_TRACE_KEYS - trace_dict.keys()
        if missing:
            raise PTDBackendError(
                f"Trace data missing required keys: {missing}"
            )

        # ---- Validate integer fields are non-negative ----
        for int_key in ('starting_vertex_idx', 'n_vertices', 'param_length', 'state_length'):
            val = trace_dict[int_key]
            if not isinstance(val, int) or val < 0:
                raise PTDBackendError(
                    f"Trace field '{int_key}' must be a non-negative integer, "
                    f"got {val!r}"
                )

        # ---- Validate and reconstruct operations ----
        valid_op_names = {member.name for member in OpType}
        operations = []
        for i, op_dict in enumerate(trace_dict['operations']):
            op_name = op_dict.get('op_type')
            if op_name not in valid_op_names:
                raise PTDBackendError(
                    f"Invalid op_type '{op_name}' in operation {i}. "
                    f"Valid types: {sorted(valid_op_names)}"
                )
            op_type = OpType[op_name]
            operands = op_dict.get('operands', [])
            coefficients = op_dict.get('coefficients')
            if coefficients is not None and not isinstance(coefficients, np.ndarray):
                coefficients = np.array(coefficients)
            operations.append(Operation(
                op_type=op_type,
                operands=operands,
                const_value=op_dict.get('const_value'),
                param_idx=op_dict.get('param_idx'),
                coefficients=coefficients
            ))

        # ---- Convert lists to numpy arrays ----
        vertex_rates = np.array(trace_dict['vertex_rates']) if isinstance(trace_dict['vertex_rates'], list) else trace_dict['vertex_rates']
        edge_probs = np.array(trace_dict['edge_probs'], dtype=object) if isinstance(trace_dict['edge_probs'], list) else trace_dict['edge_probs']
        vertex_targets = np.array(trace_dict['vertex_targets'], dtype=object) if isinstance(trace_dict['vertex_targets'], list) else trace_dict['vertex_targets']
        states = np.array(trace_dict['states']) if isinstance(trace_dict['states'], list) else trace_dict['states']

        # Optional fields with defaults
        vertex_indices_data = trace_dict.get('vertex_indices')
        if vertex_indices_data is not None:
            vertex_indices = np.array(vertex_indices_data)
        else:
            vertex_indices = np.array([])

        # ---- Construct EliminationTrace ----
        trace = EliminationTrace(
            operations=operations,
            vertex_rates=vertex_rates,
            edge_probs=edge_probs,
            vertex_targets=vertex_targets,
            states=states,
            vertex_indices=vertex_indices,
            starting_vertex_idx=trace_dict['starting_vertex_idx'],
            n_vertices=trace_dict['n_vertices'],
            param_length=trace_dict['param_length'],
            state_length=trace_dict['state_length'],
            reward_length=trace_dict.get('reward_length', 0),
            is_discrete=trace_dict.get('is_discrete', False),
            metadata=trace_dict.get('metadata', {})
        )

        return trace

    def publish_trace(
        self,
        trace: dict[str, Any],
        trace_id: str,
        metadata: dict[str, Any],
        construction_code: str | None = None,
        example_code: str | None = None,
        submit_pr: bool = False,
    ) -> str:
        """
        Publish a trace to IPFS.

        This creates a trace package on IPFS and optionally prints instructions
        for submitting to the public registry.

        Parameters
        ----------
        trace : dict
            Trace data from record_elimination_trace()
        trace_id : str
            Unique identifier for this trace
        metadata : dict
            Metadata dictionary (see plan for schema)
        construction_code : str, optional
            Python code that builds this trace
        example_code : str, optional
            Usage example code
        submit_pr : bool, default=False
            If True, print instructions for submitting PR to registry

        Returns
        -------
        str
            IPFS CID of published trace package

        Examples
        --------
        >>> from phasic.trace_elimination import record_elimination_trace
        >>> trace = record_elimination_trace(graph, theta_dim=1)
        >>>
        >>> metadata = {
        ...     "model_type": "coalescent",
        ...     "domain": "population-genetics",
        ...     "param_length": 1,
        ...     "description": "Kingman coalescent for n=5",
        ...     "author": "Kasper Munch <kaspermunch@birc.au.dk>",
        ...     "license": "MIT"
        ... }
        >>>
        >>> cid = registry.publish_trace(
        ...     trace=trace,
        ...     trace_id="my_coalescent_model",
        ...     metadata=metadata,
        ...     submit_pr=True
        ... )
        """
        # Create temporary directory for trace package
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_dir = Path(tmpdir) / trace_id
            pkg_dir.mkdir()

            # Write trace.json.gz
            trace_file = pkg_dir / "trace.json.gz"
            with gzip.open(trace_file, 'wt') as f:
                json.dump(trace, f)

            # Write metadata.json
            metadata_file = pkg_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2))

            # Write construction code if provided
            if construction_code is not None:
                code_file = pkg_dir / "construction_code.py"
                code_file.write_text(construction_code)

            # Write example code if provided
            if example_code is not None:
                example_file = pkg_dir / "example.py"
                example_file.write_text(example_code)

            # Compute checksum
            checksum = hashlib.sha256(trace_file.read_bytes()).hexdigest()
            checksum_file = pkg_dir / "checksum.sha256"
            checksum_file.write_text(checksum)

            # Add to backend
            logger.info("Publishing %s...", trace_id)
            cid = self.backend.add(pkg_dir)
            logger.info("Published to IPFS: %s", cid)

            # Print PR instructions if requested
            if submit_pr:
                self._print_pr_instructions(trace_id, cid, metadata, checksum)

            return cid

    def _print_pr_instructions(self, trace_id: str, cid: str, metadata: dict[str, Any], checksum: str) -> None:
        """Print instructions for submitting trace to public registry."""
        # Intentionally uses print() — this is user-facing interactive output
        print("\n" + "="*70)
        print("To add this trace to the public registry:")
        print("="*70)
        print(f"\n1. Fork repository: https://github.com/{self.registry_repo}")
        print(f"\n2. Add this entry to registry.json:\n")

        entry = {
            trace_id: {
                "cid": cid,
                "description": metadata.get("description", ""),
                "metadata": metadata,
                "files": {
                    "trace.json.gz": {"cid": cid + "/trace.json.gz"},
                    "metadata.json": {"cid": cid + "/metadata.json"}
                },
                "checksum": f"sha256:{checksum}",
                "license": metadata.get("license", "MIT")
            }
        }

        print(json.dumps(entry, indent=2))
        print(f"\n3. Submit pull request")
        print(f"\n4. Maintainer will pin to pinning services and merge")
        print("="*70 + "\n")


# ============================================================================
# Helper Functions
# ============================================================================

def get_trace(trace_id: str, force_download: bool = False) -> EliminationTrace:
    """
    Download and load a trace by ID (convenience function).

    This is a shortcut for creating a TraceRegistry and calling get_trace().

    Parameters
    ----------
    trace_id : str
        Trace identifier (e.g., "coalescent_n5_theta1")
    force_download : bool, default=False
        If True, re-download even if cached locally

    Returns
    -------
    EliminationTrace
        Loaded trace object

    Examples
    --------
    >>> from phasic.trace_repository import get_trace
    >>> trace = get_trace("coalescent_n5_theta1")
    >>> # Use with trace_to_log_likelihood
    """
    registry = TraceRegistry()
    return registry.get_trace(trace_id, force_download=force_download)


def install_trace_library(collection: str | None = None) -> None:
    """
    Download a collection of traces for offline use.

    Parameters
    ----------
    collection : str, optional
        Collection name (e.g., "coalescent_basic").
        If None, downloads all available traces.

    Examples
    --------
    >>> # Download all basic coalescent models
    >>> install_trace_library("coalescent_basic")
    Downloading 10 traces...
    Downloaded coalescent_n5_theta1
    Downloaded coalescent_n10_theta2
    ...
    """
    registry = TraceRegistry()

    if collection is not None:
        # Download collection
        if collection not in registry.registry.get('collections', {}):
            raise PTDBackendError(
                f"Collection '{collection}' not found.\n"
                f"  Available: {list(registry.registry.get('collections', {}).keys())}"
            )

        trace_ids = registry.registry['collections'][collection]['traces']
        logger.info("Downloading collection '%s' (%d traces)...", collection, len(trace_ids))
    else:
        # Download all traces
        trace_ids = list(registry.registry['traces'].keys())
        logger.info("Downloading all traces (%d total)...", len(trace_ids))

    # Download each trace
    for trace_id in trace_ids:
        try:
            registry.get_trace(trace_id)
            logger.info("Downloaded %s", trace_id)
        except Exception as e:
            warnings.warn(f"Failed to download {trace_id}: {e}")
