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
    >>> trace = record_elimination_trace(graph, param_length=1)
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

import json
import gzip
import hashlib
import warnings
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin

import requests

# Try importing ipfshttpclient (optional dependency)
try:
    import ipfshttpclient
    HAS_IPFS_CLIENT = True
except ImportError:
    HAS_IPFS_CLIENT = False

from .exceptions import PTDBackendError


# ============================================================================
# IPFS Backend with Auto-Start and Gateway Fallback
# ============================================================================

class IPFSBackend:
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
    gateways : List[str], optional
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
    gateways : List[str]
        List of HTTP gateway URLs for fallback

    Examples
    --------
    >>> # Auto-start daemon if available, fallback to gateways
    >>> backend = IPFSBackend()
    ✓ Started IPFS daemon automatically

    >>> # Download content
    >>> content = backend.get("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")

    >>> # Or use gateway fallback
    >>> backend = IPFSBackend(auto_start=False)
    IPFS daemon not running. Using HTTP gateways.
    >>> content = backend.get("bafybeig...")
    """

    def __init__(
        self,
        daemon_addr: str = "/ip4/127.0.0.1/tcp/5001",
        gateways: Optional[List[str]] = None,
        auto_start: bool = True,
        timeout: int = 30
    ):
        self.daemon_addr = daemon_addr
        self.timeout = timeout
        self.auto_start = auto_start
        self.client = None
        self.daemon_process = None

        # Configure HTTP gateway fallbacks
        if gateways is None:
            self.gateways = [
                "https://ipfs.io",
                "https://cloudflare-ipfs.com",
                "https://dweb.link",
                "https://gateway.pinata.cloud"
            ]
        else:
            self.gateways = gateways

        # Try connecting to IPFS daemon
        if HAS_IPFS_CLIENT:
            try:
                # Try existing daemon first
                self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                version = self.client.version()
                print(f"✓ Connected to IPFS daemon (version {version['Version']})")
            except Exception as e:
                # Try auto-starting daemon
                if auto_start and self._start_daemon():
                    time.sleep(2)  # Give daemon time to initialize
                    try:
                        self.client = ipfshttpclient.connect(daemon_addr, timeout=5)
                        print(f"✓ Started IPFS daemon automatically")
                    except:
                        warnings.warn("IPFS daemon started but connection failed. Using HTTP gateways.")
                else:
                    if not auto_start:
                        print("IPFS daemon not running. Using HTTP gateways.")
                    else:
                        warnings.warn(f"IPFS not available. Using HTTP gateways.")
        else:
            print("ipfshttpclient not installed. Using HTTP gateways only.")
            print("  Install for faster downloads: pip install ipfshttpclient")

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
            # Check if daemon already running
            result = subprocess.run(
                ['pgrep', '-x', 'ipfs'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True  # Already running

            # Check if IPFS is initialized
            ipfs_dir = Path.home() / ".ipfs"
            if not ipfs_dir.exists():
                # Initialize IPFS
                subprocess.run(
                    ['ipfs', 'init'],
                    capture_output=True,
                    check=True
                )
                print("✓ Initialized IPFS repository")

            # Start daemon in background (detached from parent)
            self.daemon_process = subprocess.Popen(
                ['ipfs', 'daemon'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Detach from parent process
            )

            return True
        except Exception as e:
            warnings.warn(f"Failed to start IPFS daemon: {e}")
            return False

    def get(self, cid: str, output_path: Optional[Path] = None) -> bytes:
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
        """
        # Try local daemon first
        if self.client is not None:
            try:
                content = self.client.cat(cid, timeout=self.timeout)
                if output_path is not None:
                    output_path.write_bytes(content)
                    return None
                return content
            except Exception as e:
                warnings.warn(f"IPFS daemon failed, trying gateways: {e}")

        # Fallback to HTTP gateways
        for gateway in self.gateways:
            url = f"{gateway}/ipfs/{cid}"
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                content = response.content
                if output_path is not None:
                    output_path.write_bytes(content)
                    return None
                return content
            except Exception as e:
                warnings.warn(f"Gateway {gateway} failed: {e}")
                continue

        raise PTDBackendError(
            f"Failed to retrieve {cid} from IPFS daemon and all HTTP gateways"
        )

    def get_directory(self, cid: str, output_dir: Path):
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
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try local daemon first
        if self.client is not None:
            try:
                self.client.get(cid, target=str(output_dir))
                return
            except Exception as e:
                warnings.warn(f"IPFS daemon failed for directory, trying gateways: {e}")

        # For gateways, we need to know the file structure
        # This is a simplified version - in practice, we'd need the directory manifest
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

    def __del__(self):
        """
        Cleanup on object destruction.

        Note: We intentionally do NOT kill the daemon here, as it should
        persist for other processes and future use.
        """
        pass


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
    ipfs_backend : IPFSBackend, optional
        Custom IPFS backend. If None, creates default backend.
    auto_update : bool, default=True
        Automatically update registry from GitHub on initialization

    Attributes
    ----------
    registry : Dict
        Loaded registry data from GitHub
    cache_dir : Path
        Local cache directory for downloaded traces
    ipfs : IPFSBackend
        IPFS backend for content retrieval

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
        cache_dir: Optional[Path] = None,
        ipfs_backend: Optional[IPFSBackend] = None,
        auto_update: bool = True
    ):
        self.registry_repo = registry_repo
        self.cache_dir = cache_dir or (Path.home() / ".phasic_traces")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize IPFS backend
        self.ipfs = ipfs_backend or IPFSBackend()

        # Registry data
        self.registry = None

        # Load cached registry
        self._load_cached_registry()

        # Update from GitHub if requested
        if auto_update:
            self.update_registry()

    def _load_cached_registry(self):
        """Load registry from local cache if available."""
        registry_path = self.cache_dir / "registry.json"
        if registry_path.exists():
            try:
                self.registry = json.loads(registry_path.read_text())
            except:
                pass

    def update_registry(self):
        """
        Update registry from GitHub.

        Downloads the latest registry.json from the configured GitHub repository.
        """
        url = self.REGISTRY_URL.format(self.registry_repo)
        try:
            print(f"Updating registry from {self.registry_repo}...")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            self.registry = response.json()

            # Save to cache
            registry_path = self.cache_dir / "registry.json"
            registry_path.write_text(json.dumps(self.registry, indent=2))
            print("✓ Registry updated")
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
        domain: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available traces with optional filtering.

        Parameters
        ----------
        domain : str, optional
            Filter by domain (e.g., "population-genetics")
        model_type : str, optional
            Filter by model type (e.g., "coalescent")
        tags : List[str], optional
            Filter by tags (all must match)

        Returns
        -------
        List[Dict[str, Any]]
            List of trace metadata dictionaries
        """
        if self.registry is None:
            raise PTDBackendError("Registry not loaded. Call update_registry() first.")

        traces = []
        for trace_id, info in self.registry['traces'].items():
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

    def get_trace(self, trace_id: str, force_download: bool = False) -> Dict:
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
        Dict
            Loaded trace data (result from record_elimination_trace)

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
        cid = trace_info['cid']

        # Check local cache
        trace_cache_dir = self.cache_dir / "traces" / trace_id
        trace_file = trace_cache_dir / "trace.json.gz"

        if not force_download and trace_file.exists():
            print(f"✓ Using cached trace: {trace_file}")
        else:
            # Download from IPFS
            print(f"Downloading trace '{trace_id}' from IPFS...")
            trace_cache_dir.mkdir(parents=True, exist_ok=True)

            # Get the trace.json.gz file from the directory CID
            dir_cid = trace_info['cid']
            file_path_in_ipfs = f"{dir_cid}/trace.json.gz"
            self.ipfs.get(file_path_in_ipfs, output_path=trace_file)

            print(f"✓ Downloaded to {trace_file}")

        # Load and decompress trace
        with gzip.open(trace_file, 'rt') as f:
            trace_dict = json.load(f)

        # Use helper to deserialize
        return self._deserialize_trace(trace_dict)

    def get_trace_by_hash(self, graph_hash: str, force_download: bool = False):
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
        >>> graph = Graph(callback=my_callback, parameterized=True, nr_samples=5)
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
            print(f"✓ Using cached trace (by hash): {trace_file}")
            # Load from cache
            with gzip.open(trace_file, 'rt') as f:
                trace_dict = json.load(f)
            return self._deserialize_trace(trace_dict)

        # Search registry for matching graph_hash
        for trace_id, info in self.registry['traces'].items():
            if info.get('graph_hash') == graph_hash:
                print(f"✓ Found trace '{trace_id}' with matching hash")
                # Download using existing get_trace method
                trace = self.get_trace(trace_id, force_download=force_download)

                # Also cache by hash for faster future lookups
                hash_cache_dir.mkdir(parents=True, exist_ok=True)
                trace_by_id_file = self.cache_dir / "traces" / trace_id / "trace.json.gz"
                if trace_by_id_file.exists():
                    import shutil
                    shutil.copy(trace_by_id_file, trace_file)
                    print(f"✓ Cached by hash: {hash_cache_dir}")

                return trace

        # Not found in registry
        print(f"✗ No trace found for hash {graph_hash[:16]}...")
        return None

    def _deserialize_trace(self, trace_dict):
        """
        Helper to deserialize trace from dict (extracted from get_trace).

        This is the deserialization logic from get_trace(), factored out
        for reuse in get_trace_by_hash().
        """
        from .trace_elimination import EliminationTrace, Operation, OpType
        import numpy as np

        # Reconstruct operations
        operations = []
        for op_dict in trace_dict['operations']:
            op_type = OpType[op_dict['op_type']]  # Convert string back to enum
            operands = op_dict.get('operands', [])
            coefficients = op_dict.get('coefficients')
            # Convert coefficients to numpy array if present
            if coefficients is not None and not isinstance(coefficients, np.ndarray):
                coefficients = np.array(coefficients)
            operations.append(Operation(
                op_type=op_type,
                operands=operands,
                const_value=op_dict.get('const_value'),
                param_idx=op_dict.get('param_idx'),
                coefficients=coefficients
            ))

        # Convert lists back to numpy arrays
        vertex_rates = np.array(trace_dict['vertex_rates']) if isinstance(trace_dict['vertex_rates'], list) else trace_dict['vertex_rates']
        edge_probs = np.array(trace_dict['edge_probs'], dtype=object) if isinstance(trace_dict['edge_probs'], list) else trace_dict['edge_probs']
        vertex_targets = np.array(trace_dict['vertex_targets'], dtype=object) if isinstance(trace_dict['vertex_targets'], list) else trace_dict['vertex_targets']
        states = np.array(trace_dict['states']) if isinstance(trace_dict['states'], list) else trace_dict['states']

        # Reconstruct EliminationTrace object
        trace = EliminationTrace(
            operations=operations,
            vertex_rates=vertex_rates,
            edge_probs=edge_probs,
            vertex_targets=vertex_targets,
            states=states,
            starting_vertex_idx=trace_dict['starting_vertex_idx'],
            n_vertices=trace_dict['n_vertices'],
            param_length=trace_dict['param_length'],
            state_length=trace_dict['state_length'],
            is_discrete=trace_dict.get('is_discrete', False),
            metadata=trace_dict.get('metadata', {})
        )

        return trace

    def publish_trace(
        self,
        trace: Dict,
        trace_id: str,
        metadata: Dict,
        construction_code: Optional[str] = None,
        example_code: Optional[str] = None,
        submit_pr: bool = False
    ) -> str:
        """
        Publish a trace to IPFS.

        This creates a trace package on IPFS and optionally prints instructions
        for submitting to the public registry.

        Parameters
        ----------
        trace : Dict
            Trace data from record_elimination_trace()
        trace_id : str
            Unique identifier for this trace
        metadata : Dict
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
        >>> trace = record_elimination_trace(graph, param_length=1)
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

            # Add to IPFS
            print(f"Publishing {trace_id} to IPFS...")
            cid = self.ipfs.add(pkg_dir)
            print(f"✓ Published to IPFS: {cid}")

            # Print PR instructions if requested
            if submit_pr:
                self._print_pr_instructions(trace_id, cid, metadata, checksum)

            return cid

    def _print_pr_instructions(self, trace_id: str, cid: str, metadata: Dict, checksum: str):
        """Print instructions for submitting trace to public registry."""
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

def get_trace(trace_id: str, force_download: bool = False) -> Dict:
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
    Dict
        Loaded trace data

    Examples
    --------
    >>> from phasic.trace_repository import get_trace
    >>> trace = get_trace("coalescent_n5_theta1")
    >>> # Use with trace_to_log_likelihood
    """
    registry = TraceRegistry()
    return registry.get_trace(trace_id, force_download=force_download)


def install_trace_library(collection: Optional[str] = None):
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
    ✓ Downloaded coalescent_n5_theta1
    ✓ Downloaded coalescent_n10_theta2
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
        print(f"Downloading collection '{collection}' ({len(trace_ids)} traces)...")
    else:
        # Download all traces
        trace_ids = list(registry.registry['traces'].keys())
        print(f"Downloading all traces ({len(trace_ids)} total)...")

    # Download each trace
    for trace_id in trace_ids:
        try:
            registry.get_trace(trace_id)
            print(f"✓ Downloaded {trace_id}")
        except Exception as e:
            warnings.warn(f"Failed to download {trace_id}: {e}")
