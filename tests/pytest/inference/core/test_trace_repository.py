"""
Tests for IPFS-based trace repository.

These tests verify:
1. IPFSBackend initialization with different configurations
2. Gateway fallback when IPFS daemon unavailable
3. TraceRegistry registry loading and caching
4. Mock downloads (without requiring actual IPFS setup)
5. Security: CID validation, checksum verification, output path validation
6. Transport abstraction: TransportBackend ABC, custom backends
7. Edge cases: lazy init, deprecation warnings, format version, retry
8. Swarm key support for private IPFS networks
"""

import json
import gzip
import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from phasic.trace_repository import (
    IPFSBackend,
    TraceRegistry,
    TransportBackend,
    get_trace,
    _validate_cid,
    _validate_output_path,
    _validate_registry,
    _request_with_retry,
    TRACE_FORMAT_VERSION,
    _REQUIRED_TRACE_KEYS,
    get_ipfs_dir,
    generate_swarm_key,
    install_swarm_key,
    detect_swarm_key,
    remove_swarm_key,
    configure_bootstrap_peers,
    _validate_swarm_key_content,
    _SWARM_KEY_RE,
)
from phasic.exceptions import PTDBackendError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_registry():
    """Create mock registry data."""
    return {
        "version": "1.0.0",
        "updated": "2025-10-21T12:00:00Z",
        "traces": {
            "test_trace_1": {
                "cid": "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                "description": "Test trace 1",
                "checksum": "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
                "metadata": {
                    "model_type": "coalescent",
                    "domain": "population-genetics",
                    "param_length": 1,
                    "vertices": 5,
                    "tags": ["test", "coalescent"]
                },
                "files": {
                    "trace.json.gz": {
                        "cid": "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                        "size_bytes": 1000
                    }
                }
            },
            "test_trace_2": {
                "cid": "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                "description": "Test trace 2",
                "metadata": {
                    "model_type": "structured-coalescent",
                    "domain": "population-genetics",
                    "param_length": 2,
                    "vertices": 10,
                    "tags": ["test", "structured"]
                },
                "files": {
                    "trace.json.gz": {
                        "cid": "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
                        "size_bytes": 2000
                    }
                }
            }
        },
        "collections": {
            "test_collection": {
                "description": "Test collection",
                "traces": ["test_trace_1", "test_trace_2"]
            }
        }
    }


@pytest.fixture
def mock_trace_data():
    """Create mock trace data as a valid serialized dict."""
    return {
        "operations": [
            {"op_type": "CONST", "operands": [], "const_value": 1.0},
            {"op_type": "CONST", "operands": [], "const_value": 0.0},
        ],
        "vertex_rates": [0, 1],
        "edge_probs": [[0], []],
        "vertex_targets": [[1], []],
        "states": [[0], [1]],
        "starting_vertex_idx": 0,
        "n_vertices": 2,
        "param_length": 0,
        "state_length": 1,
        "is_discrete": False,
        "metadata": {},
    }


@pytest.fixture
def mock_trace_data_simple():
    """Simpler mock trace data (raw dict, not requiring deserialization)."""
    return {
        "vertex_rates": [[1.0], [0.0]],
        "edge_probs": [[0.5, 0.5]],
        "vertex_targets": [[1], []]
    }


def _write_gzipped_json(path: Path, data: dict):
    """Helper: write gzipped JSON to *path*."""
    with gzip.open(path, 'wt') as f:
        json.dump(data, f)


def _make_registry_with_cache(tmp_path, registry_data):
    """Helper: set up cache_dir with a cached registry.json."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "registry.json").write_text(json.dumps(registry_data))
    return cache_dir


class DummyBackend(TransportBackend):
    """Minimal concrete backend for testing the ABC."""

    @property
    def name(self) -> str:
        return "dummy"

    def get(self, cid, output_path=None):
        return b"dummy content"

    def add(self, path):
        return "dummy_cid"


# ============================================================================
# Original Tests — IPFSBackend
# ============================================================================

class TestIPFSBackend:
    """Tests for IPFSBackend class."""

    def test_init_without_ipfs_client(self):
        """Test initialization when ipfshttpclient not installed."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            assert backend.client is None
            assert backend.gateways is not None
            assert len(backend.gateways) > 0

    def test_init_with_custom_gateways(self):
        """Test initialization with custom gateway list."""
        custom_gateways = ["https://custom.gateway.com"]
        backend = IPFSBackend(gateways=custom_gateways, auto_start=False)
        assert backend.gateways == custom_gateways

    def test_gateway_fallback(self):
        """Test HTTP gateway fallback when daemon unavailable."""
        mock_content = b"test content"

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository._request_with_retry') as mock_get:
                mock_response = mock.Mock()
                mock_response.content = mock_content
                mock_get.return_value = mock_response

                backend = IPFSBackend()
                content = backend.get("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

                assert content == mock_content
                assert mock_get.called

    def test_get_with_output_path(self):
        """Test downloading to file."""
        mock_content = b"test content"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"

            with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
                with mock.patch('phasic.trace_repository._request_with_retry') as mock_get:
                    mock_response = mock.Mock()
                    mock_response.content = mock_content
                    mock_get.return_value = mock_response

                    backend = IPFSBackend()
                    result = backend.get("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG", output_path=output_path)

                    assert result is None
                    assert output_path.exists()
                    assert output_path.read_bytes() == mock_content

    def test_get_failure_all_gateways(self):
        """Test failure when all gateways fail."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch(
                'phasic.trace_repository._request_with_retry',
                side_effect=Exception("Network error")
            ):
                backend = IPFSBackend()

                with pytest.raises(PTDBackendError, match="Failed to retrieve"):
                    backend.get("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_add_without_daemon(self):
        """Test that add() fails without daemon."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()

            with pytest.raises(PTDBackendError, match="Publishing requires IPFS daemon"):
                backend.add(Path("/fake/path"))

    def test_is_transport_backend(self):
        """IPFSBackend must be a TransportBackend."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            assert isinstance(backend, TransportBackend)
            assert backend.name == "ipfs"


# ============================================================================
# Original Tests — TraceRegistry
# ============================================================================

class TestTraceRegistry:
    """Tests for TraceRegistry class."""

    def test_init_with_cached_registry(self, tmp_path, mock_registry):
        """Test initialization with cached registry."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            assert registry.registry is not None
            assert "test_trace_1" in registry.registry["traces"]

    def test_update_registry(self, tmp_path, mock_registry):
        """Test updating registry from GitHub."""
        cache_dir = tmp_path / "cache"

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            with mock.patch('phasic.trace_repository._request_with_retry') as mock_get:
                mock_response = mock.Mock()
                mock_response.json.return_value = mock_registry
                mock_get.return_value = mock_response

                registry = TraceRegistry(cache_dir=cache_dir, auto_update=True)

                assert registry.registry is not None
                assert (cache_dir / "registry.json").exists()

    def test_list_traces_no_filter(self, tmp_path, mock_registry):
        """Test listing all traces without filters."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces()

            assert len(traces) == 2
            assert traces[0]['trace_id'] == 'test_trace_1'
            assert traces[1]['trace_id'] == 'test_trace_2'

    def test_list_traces_with_domain_filter(self, tmp_path, mock_registry):
        """Test filtering traces by domain."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(domain="population-genetics")

            assert len(traces) == 2

    def test_list_traces_with_model_type_filter(self, tmp_path, mock_registry):
        """Test filtering traces by model type."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(model_type="coalescent")

            assert len(traces) == 1
            assert traces[0]['trace_id'] == 'test_trace_1'

    def test_list_traces_with_tags_filter(self, tmp_path, mock_registry):
        """Test filtering traces by tags."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(tags=["structured"])

            assert len(traces) == 1
            assert traces[0]['trace_id'] == 'test_trace_2'

    def test_get_trace_from_cache(self, tmp_path, mock_registry, mock_trace_data):
        """Test loading trace from local cache (deserialized)."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Create cached trace
        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", mock_trace_data)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            trace = registry.get_trace("test_trace_1")

            # Should return an EliminationTrace, not a raw dict
            from phasic.trace_elimination import EliminationTrace
            assert isinstance(trace, EliminationTrace)
            assert trace.n_vertices == 2
            assert trace.param_length == 0

    def test_get_trace_download(self, tmp_path, mock_registry, mock_trace_data):
        """Test downloading trace via backend."""
        # Remove fake checksum so download doesn't fail on verification
        mock_registry["traces"]["test_trace_1"].pop("checksum", None)
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"

        def mock_get(cid, output_path=None):
            _write_gzipped_json(output_path, mock_trace_data)

        mock_backend.get = mock_get

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            trace = registry.get_trace("test_trace_1")

            from phasic.trace_elimination import EliminationTrace
            assert isinstance(trace, EliminationTrace)

    def test_get_trace_not_found(self, tmp_path, mock_registry):
        """Test error when trace not in registry."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)

            with pytest.raises(PTDBackendError, match="not found in registry"):
                registry.get_trace("nonexistent_trace")

    def test_publish_trace(self, tmp_path, mock_trace_data_simple):
        """Test publishing a trace to IPFS."""
        cache_dir = tmp_path / "cache"

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.add.return_value = "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"

        metadata = {
            "model_type": "test",
            "domain": "test",
            "param_length": 1,
            "description": "Test trace"
        }

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            registry.registry = {"version": "1.0", "traces": {}}

            cid = registry.publish_trace(
                trace=mock_trace_data_simple,
                trace_id="new_test_trace",
                metadata=metadata,
                submit_pr=False
            )

            assert cid == "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
            assert mock_backend.add.called


# ============================================================================
# Helper function tests
# ============================================================================

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_trace_helper(self, tmp_path):
        """Test get_trace() convenience function."""
        with mock.patch('phasic.trace_repository.TraceRegistry') as MockRegistry:
            mock_registry_instance = mock.Mock()
            mock_registry_instance.get_trace.return_value = {"test": "data"}
            MockRegistry.return_value = mock_registry_instance

            result = get_trace("test_trace")

            assert result == {"test": "data"}
            mock_registry_instance.get_trace.assert_called_once_with("test_trace", force_download=False)


# ============================================================================
# Security Validation Tests
# ============================================================================

class TestSecurityValidation:
    """Tests for CID validation, checksum, deserialization, and path safety."""

    # ---- CID validation ----

    def test_validate_cid_valid_v0(self):
        """Valid CIDv0 (Qm..., 46 chars base58)."""
        cid = "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        _validate_cid(cid)  # should not raise

    def test_validate_cid_valid_v1(self):
        """Valid CIDv1 (bafy..., base32)."""
        cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"
        _validate_cid(cid)  # should not raise

    def test_validate_cid_with_subpath(self):
        """CID/subpath format is accepted."""
        cid = "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG/trace.json.gz"
        _validate_cid(cid)  # should not raise

    def test_validate_cid_empty(self):
        """Empty CID raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_cid("")

    def test_validate_cid_none(self):
        """None CID raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            _validate_cid(None)

    def test_validate_cid_path_traversal(self):
        """Path traversal in subpath is rejected."""
        cid = "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG/../../../etc/passwd"
        with pytest.raises(ValueError, match="Path components"):
            _validate_cid(cid)

    def test_validate_cid_invalid_format(self):
        """Random string is rejected."""
        with pytest.raises(ValueError, match="Invalid CID format"):
            _validate_cid("not-a-cid")

    # ---- Output path validation ----

    def test_validate_output_path_inside(self, tmp_path):
        """Path inside allowed root is accepted."""
        out = tmp_path / "sub" / "file.txt"
        _validate_output_path(out, tmp_path)  # should not raise

    def test_validate_output_path_traversal(self, tmp_path):
        """Path escaping allowed root is rejected."""
        out = tmp_path / ".." / "escaped.txt"
        with pytest.raises(ValueError, match="resolves outside"):
            _validate_output_path(out, tmp_path / "sub")

    # ---- Checksum verification ----

    def test_checksum_pass(self, tmp_path, mock_registry, mock_trace_data):
        """Correct checksum passes verification."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Write trace file
        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        trace_file = trace_dir / "trace.json.gz"
        _write_gzipped_json(trace_file, mock_trace_data)

        # Compute correct checksum
        correct_checksum = hashlib.sha256(trace_file.read_bytes()).hexdigest()

        # Patch registry to have correct checksum
        mock_registry["traces"]["test_trace_1"]["checksum"] = f"sha256:{correct_checksum}"
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            # Should not raise
            trace = registry.get_trace("test_trace_1")
            from phasic.trace_elimination import EliminationTrace
            assert isinstance(trace, EliminationTrace)

    def test_checksum_fail(self, tmp_path, mock_registry, mock_trace_data):
        """Wrong checksum raises PTDBackendError and deletes file."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Patch registry with wrong checksum
        mock_registry["traces"]["test_trace_1"]["checksum"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"

        def mock_get(cid, output_path=None):
            _write_gzipped_json(output_path, mock_trace_data)

        mock_backend.get = mock_get

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)

            with pytest.raises(PTDBackendError, match="Checksum mismatch"):
                registry.get_trace("test_trace_1", force_download=True)

        # File should be deleted after checksum failure
        trace_file = cache_dir / "traces" / "test_trace_1" / "trace.json.gz"
        assert not trace_file.exists()

    # ---- Deserialization validation ----

    def test_deserialize_missing_keys(self, tmp_path, mock_registry):
        """Deserialization rejects dicts missing required keys."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        incomplete = {"operations": [], "vertex_rates": []}  # missing many keys

        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", incomplete)

        # Remove checksum so it doesn't fail on checksum first
        mock_registry["traces"]["test_trace_1"].pop("checksum", None)
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="missing required keys"):
                registry.get_trace("test_trace_1")

    def test_deserialize_invalid_op_type(self, tmp_path, mock_registry):
        """Deserialization rejects invalid op_type values."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        bad_trace = {
            "operations": [{"op_type": "BOGUS", "operands": []}],
            "vertex_rates": [0],
            "edge_probs": [[]],
            "vertex_targets": [[]],
            "states": [[0]],
            "starting_vertex_idx": 0,
            "n_vertices": 1,
            "param_length": 0,
            "state_length": 1,
        }

        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", bad_trace)

        mock_registry["traces"]["test_trace_1"].pop("checksum", None)
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="Invalid op_type"):
                registry.get_trace("test_trace_1")

    def test_deserialize_negative_n_vertices(self, tmp_path, mock_registry):
        """Deserialization rejects negative integer fields."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        bad_trace = {
            "operations": [],
            "vertex_rates": [],
            "edge_probs": [],
            "vertex_targets": [],
            "states": [],
            "starting_vertex_idx": 0,
            "n_vertices": -1,
            "param_length": 0,
            "state_length": 1,
        }

        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", bad_trace)

        mock_registry["traces"]["test_trace_1"].pop("checksum", None)
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="non-negative integer"):
                registry.get_trace("test_trace_1")


# ============================================================================
# Transport Backend Tests
# ============================================================================

class TestTransportBackend:
    """Tests for the TransportBackend ABC and backend integration."""

    def test_abc_cannot_instantiate(self):
        """TransportBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TransportBackend()

    def test_dummy_backend_works(self):
        """A concrete subclass can be instantiated and used."""
        b = DummyBackend()
        assert b.name == "dummy"
        assert b.get("any_cid") == b"dummy content"
        assert b.add(Path("/fake")) == "dummy_cid"
        assert b.supports_directories is False

    def test_get_directory_not_implemented(self):
        """Default get_directory raises NotImplementedError."""
        b = DummyBackend()
        with pytest.raises(NotImplementedError, match="dummy"):
            b.get_directory("cid", Path("/tmp/out"))

    def test_ipfs_backend_is_transport_backend(self):
        """IPFSBackend inherits from TransportBackend."""
        assert issubclass(IPFSBackend, TransportBackend)

    def test_registry_accepts_custom_backend(self, tmp_path, mock_registry):
        """TraceRegistry works with a custom TransportBackend."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        backend = DummyBackend()

        registry = TraceRegistry(
            cache_dir=cache_dir, auto_update=False, backend=backend
        )
        assert registry.backend is backend
        assert registry.backend.name == "dummy"

    def test_deprecated_ipfs_backend_warns(self, tmp_path, mock_registry):
        """Using ipfs_backend= emits DeprecationWarning."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        backend = DummyBackend()

        with pytest.warns(DeprecationWarning, match="ipfs_backend.*deprecated"):
            registry = TraceRegistry(
                cache_dir=cache_dir, auto_update=False, ipfs_backend=backend
            )
        assert registry.backend is backend

    def test_cannot_specify_both_backend_and_ipfs_backend(self, tmp_path, mock_registry):
        """Passing both backend= and ipfs_backend= raises ValueError."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with pytest.raises(ValueError, match="Cannot specify both"):
            TraceRegistry(
                cache_dir=cache_dir,
                auto_update=False,
                backend=DummyBackend(),
                ipfs_backend=DummyBackend(),
            )


# ============================================================================
# TraceRegistry Edge Cases
# ============================================================================

class TestTraceRegistryEdgeCases:
    """Tests for lazy init, registry validation, format version, and more."""

    def test_lazy_backend_init(self, tmp_path, mock_registry):
        """Backend is not created until first use."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Pass no backend — should not create IPFSBackend during __init__
        with mock.patch('phasic.trace_repository.IPFSBackend') as MockIPFS:
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            MockIPFS.assert_not_called()  # lazy: not created yet

            # Access backend property triggers creation
            _ = registry.backend
            MockIPFS.assert_called_once()

    def test_registry_validation_rejects_bad_json(self):
        """_validate_registry rejects non-dict and missing keys."""
        with pytest.raises(PTDBackendError, match="must be a dict"):
            _validate_registry("not a dict")

        with pytest.raises(PTDBackendError, match="missing required keys"):
            _validate_registry({"version": "1.0"})  # missing 'traces'

        with pytest.raises(PTDBackendError, match="must be a dict"):
            _validate_registry({"version": "1.0", "traces": "not-a-dict"})

    def test_registry_validation_accepts_valid(self, mock_registry):
        """_validate_registry accepts valid registry data."""
        _validate_registry(mock_registry)  # should not raise

    def test_format_version_exists(self):
        """TRACE_FORMAT_VERSION is defined."""
        assert TRACE_FORMAT_VERSION == "1.0"

    def test_required_trace_keys(self):
        """_REQUIRED_TRACE_KEYS includes all mandatory fields."""
        assert 'operations' in _REQUIRED_TRACE_KEYS
        assert 'n_vertices' in _REQUIRED_TRACE_KEYS
        assert 'param_length' in _REQUIRED_TRACE_KEYS

    def test_reward_length_preserved(self, tmp_path, mock_registry):
        """reward_length and vertex_indices survive round-trip deserialization."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        trace_data = {
            "operations": [
                {"op_type": "CONST", "operands": [], "const_value": 1.0},
            ],
            "vertex_rates": [0],
            "edge_probs": [[]],
            "vertex_targets": [[]],
            "states": [[0]],
            "vertex_indices": [42],
            "starting_vertex_idx": 0,
            "n_vertices": 1,
            "param_length": 0,
            "state_length": 1,
            "reward_length": 3,
            "is_discrete": False,
            "metadata": {},
        }

        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", trace_data)

        mock_registry["traces"]["test_trace_1"].pop("checksum", None)
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            trace = registry.get_trace("test_trace_1")

        assert trace.reward_length == 3
        import numpy as np
        np.testing.assert_array_equal(trace.vertex_indices, [42])

    def test_cached_registry_corrupt_json(self, tmp_path):
        """Corrupt cached registry.json is handled gracefully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text("{invalid json!!!")

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            # Should not raise — falls back to None
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            assert registry.registry is None

    def test_retry_on_transient_failure(self):
        """_request_with_retry retries on connection errors."""
        ok_response = mock.Mock()
        ok_response.status_code = 200
        ok_response.raise_for_status = mock.Mock()

        with mock.patch('requests.get') as mock_get:
            mock_get.side_effect = [
                __import__('requests').ConnectionError("transient"),
                ok_response,
            ]
            with mock.patch('time.sleep'):  # skip actual sleep
                result = _request_with_retry("http://example.com", max_retries=3, backoff_base=0.01)
            assert result is ok_response
            assert mock_get.call_count == 2

    def test_retry_exhausted(self):
        """_request_with_retry raises after all retries exhausted."""
        with mock.patch('requests.get') as mock_get:
            mock_get.side_effect = __import__('requests').ConnectionError("fail")
            with mock.patch('time.sleep'):
                with pytest.raises(__import__('requests').ConnectionError):
                    _request_with_retry("http://example.com", max_retries=2, backoff_base=0.01)
            assert mock_get.call_count == 2

    def test_ipfs_backward_compat_alias(self, tmp_path, mock_registry):
        """registry.ipfs is a backward-compatible alias for registry.backend."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        backend = DummyBackend()
        registry = TraceRegistry(
            cache_dir=cache_dir, auto_update=False, backend=backend
        )
        assert registry.ipfs is registry.backend


# ============================================================================
# IPFSBackend Status & Pinning Tests
# ============================================================================

class TestIPFSBackendStatusPinning:
    """Tests for IPFSBackend.status(), pin(), unpin(), is_pinned()."""

    def test_status_without_daemon(self):
        """status() works when no daemon is running."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            st = backend.status()
            assert st["daemon"] is False
            assert st["daemon_version"] is None
            assert isinstance(st["gateways"], list)
            assert "ipfs_installed" in st

    def test_status_with_daemon(self):
        """status() reports daemon version when connected."""
        mock_client = mock.Mock()
        mock_client.version.return_value = {"Version": "0.18.1"}

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', True):
            with mock.patch('phasic.trace_repository.ipfshttpclient') as mock_ipfs:
                mock_ipfs.connect.return_value = mock_client
                backend = IPFSBackend()
                st = backend.status()

                assert st["daemon"] is True
                assert st["daemon_version"] == "0.18.1"

    def test_pin_without_daemon(self):
        """pin() raises when no daemon available."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            with pytest.raises(PTDBackendError, match="Pinning requires"):
                backend.pin("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_pin_with_daemon(self):
        """pin() calls client.pin.add on success."""
        mock_client = mock.Mock()

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', True):
            with mock.patch('phasic.trace_repository.ipfshttpclient') as mock_ipfs:
                mock_ipfs.connect.return_value = mock_client
                backend = IPFSBackend()
                backend.pin("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

                mock_client.pin.add.assert_called_once_with(
                    "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
                )

    def test_unpin_without_daemon(self):
        """unpin() raises when no daemon available."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            with pytest.raises(PTDBackendError, match="Unpinning requires"):
                backend.unpin("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_unpin_with_daemon(self):
        """unpin() calls client.pin.rm on success."""
        mock_client = mock.Mock()

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', True):
            with mock.patch('phasic.trace_repository.ipfshttpclient') as mock_ipfs:
                mock_ipfs.connect.return_value = mock_client
                backend = IPFSBackend()
                backend.unpin("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

                mock_client.pin.rm.assert_called_once_with(
                    "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
                )

    def test_is_pinned_without_daemon(self):
        """is_pinned() returns False when no daemon is available."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()
            assert backend.is_pinned("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG") is False

    def test_is_pinned_true(self):
        """is_pinned() returns True when CID is in pin list."""
        cid = "QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        mock_client = mock.Mock()
        mock_client.pin.ls.return_value = {"Keys": {cid: {"Type": "recursive"}}}

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', True):
            with mock.patch('phasic.trace_repository.ipfshttpclient') as mock_ipfs:
                mock_ipfs.connect.return_value = mock_client
                backend = IPFSBackend()
                assert backend.is_pinned(cid) is True

    def test_is_pinned_false(self):
        """is_pinned() returns False when CID is not in pin list."""
        mock_client = mock.Mock()
        mock_client.pin.ls.return_value = {"Keys": {}}

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', True):
            with mock.patch('phasic.trace_repository.ipfshttpclient') as mock_ipfs:
                mock_ipfs.connect.return_value = mock_client
                backend = IPFSBackend()
                assert backend.is_pinned("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG") is False


# ============================================================================
# TraceRegistry Source, Cache & Retraction Tests
# ============================================================================

class TestTraceRegistrySourceCacheRetraction:
    """Tests for trace_source(), clear_cache(), retract_trace()."""

    def test_trace_source_local_cache(self, tmp_path, mock_registry, mock_trace_data):
        """trace_source() reports 'local_cache' when file is cached."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Create cached file
        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", mock_trace_data)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "local_cache"
            assert info["cached"] is True
            assert info["trace_id"] == "test_trace_1"

    def test_trace_source_ipfs_daemon(self, tmp_path, mock_registry):
        """trace_source() reports 'ipfs_daemon' when daemon available, no cache."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.client = mock.Mock()  # daemon available

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "ipfs_daemon"
            assert info["cached"] is False

    def test_trace_source_http_gateway(self, tmp_path, mock_registry):
        """trace_source() reports 'http_gateway' when no daemon, no cache."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.client = None  # no daemon

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "http_gateway"

    def test_trace_source_custom_backend(self, tmp_path, mock_registry):
        """trace_source() reports backend name for non-IPFS backends."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)
        backend = DummyBackend()

        registry = TraceRegistry(cache_dir=cache_dir, auto_update=False, backend=backend)
        info = registry.trace_source("test_trace_1")

        assert info["source"] == "dummy"
        assert info["backend"] == "dummy"

    def test_trace_source_unknown_trace(self, tmp_path, mock_registry):
        """trace_source() raises for unknown trace_id."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="not found"):
                registry.trace_source("nonexistent")

    def test_clear_cache_single(self, tmp_path, mock_registry, mock_trace_data):
        """clear_cache(trace_id) removes only the specified trace."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        for tid in ("test_trace_1", "test_trace_2"):
            d = cache_dir / "traces" / tid
            d.mkdir(parents=True)
            _write_gzipped_json(d / "trace.json.gz", mock_trace_data)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            removed = registry.clear_cache("test_trace_1")

            assert removed == 1
            assert not (cache_dir / "traces" / "test_trace_1").exists()
            assert (cache_dir / "traces" / "test_trace_2").exists()

    def test_clear_cache_all(self, tmp_path, mock_registry, mock_trace_data):
        """clear_cache() with no argument removes all cached traces."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        for tid in ("test_trace_1", "test_trace_2"):
            d = cache_dir / "traces" / tid
            d.mkdir(parents=True)
            _write_gzipped_json(d / "trace.json.gz", mock_trace_data)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            removed = registry.clear_cache()

            assert removed == 2
            # traces dir might still exist but should be empty
            traces_dir = cache_dir / "traces"
            assert not any(traces_dir.iterdir()) if traces_dir.exists() else True

    def test_retract_trace(self, tmp_path, mock_registry, mock_trace_data):
        """retract_trace() marks entry and persists to disk."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        # Create cached file so retraction can clear it
        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        _write_gzipped_json(trace_dir / "trace.json.gz", mock_trace_data)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            registry.retract_trace("test_trace_1", reason="Bad data")

            # In-memory registry is updated
            entry = registry.registry['traces']['test_trace_1']
            assert entry['retracted'] is True
            assert entry['retracted_reason'] == "Bad data"

            # Cache was cleared
            assert not trace_dir.exists()

            # Persisted to disk
            on_disk = json.loads((cache_dir / "registry.json").read_text())
            assert on_disk['traces']['test_trace_1']['retracted'] is True

    def test_retract_unknown_trace(self, tmp_path, mock_registry):
        """retract_trace() raises for unknown trace_id."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="not found"):
                registry.retract_trace("nonexistent")

    def test_list_traces_skips_retracted(self, tmp_path, mock_registry):
        """list_traces() skips entries where retracted=True."""
        mock_registry['traces']['test_trace_1']['retracted'] = True
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces()

            assert len(traces) == 1
            assert traces[0]['trace_id'] == 'test_trace_2'

    def test_get_trace_rejects_retracted(self, tmp_path, mock_registry):
        """get_trace() raises for retracted traces."""
        mock_registry['traces']['test_trace_1']['retracted'] = True
        mock_registry['traces']['test_trace_1']['retracted_reason'] = "Corrupt"
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            with pytest.raises(PTDBackendError, match="retracted.*Corrupt"):
                registry.get_trace("test_trace_1")


# ============================================================================
# Swarm Key Generation Tests
# ============================================================================

class TestSwarmKeyGeneration:
    """Tests for generate_swarm_key() output format and uniqueness."""

    def test_format(self):
        """Generated key matches the 3-line format."""
        key = generate_swarm_key()
        lines = key.strip().split('\n')
        assert len(lines) == 3
        assert lines[0] == "/key/swarm/psk/1.0.0/"
        assert lines[1] == "/base16/"
        assert len(lines[2]) == 64
        # All hex chars
        int(lines[2], 16)

    def test_uniqueness(self):
        """Two generated keys are different."""
        k1 = generate_swarm_key()
        k2 = generate_swarm_key()
        assert k1 != k2

    def test_passes_validation(self):
        """Generated key passes _validate_swarm_key_content."""
        key = generate_swarm_key()
        _validate_swarm_key_content(key)  # should not raise

    def test_regex_matches(self):
        """Generated key matches _SWARM_KEY_RE."""
        key = generate_swarm_key()
        assert _SWARM_KEY_RE.match(key)


# ============================================================================
# Swarm Key Validation Tests
# ============================================================================

class TestSwarmKeyValidation:
    """Tests for _validate_swarm_key_content edge cases."""

    def test_valid_key(self):
        """Valid 3-line key passes."""
        content = "/key/swarm/psk/1.0.0/\n/base16/\n" + "ab" * 32 + "\n"
        _validate_swarm_key_content(content)  # should not raise

    def test_valid_key_no_trailing_newline(self):
        """Valid key without trailing newline passes."""
        content = "/key/swarm/psk/1.0.0/\n/base16/\n" + "ab" * 32
        _validate_swarm_key_content(content)  # should not raise

    def test_wrong_version(self):
        """Wrong protocol version is rejected."""
        content = "/key/swarm/psk/2.0.0/\n/base16/\n" + "ab" * 32 + "\n"
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content(content)

    def test_wrong_encoding(self):
        """Non-base16 encoding line is rejected."""
        content = "/key/swarm/psk/1.0.0/\n/base64/\n" + "ab" * 32 + "\n"
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content(content)

    def test_short_key(self):
        """Key shorter than 64 hex chars is rejected."""
        content = "/key/swarm/psk/1.0.0/\n/base16/\n" + "ab" * 16 + "\n"
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content(content)

    def test_long_key(self):
        """Key longer than 64 hex chars is rejected."""
        content = "/key/swarm/psk/1.0.0/\n/base16/\n" + "ab" * 33 + "\n"
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content(content)

    def test_non_hex_key(self):
        """Non-hex characters in key are rejected."""
        content = "/key/swarm/psk/1.0.0/\n/base16/\n" + "zz" * 32 + "\n"
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content(content)

    def test_empty_string(self):
        """Empty string is rejected."""
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content("")

    def test_random_text(self):
        """Random text is rejected."""
        with pytest.raises(ValueError, match="Invalid swarm key format"):
            _validate_swarm_key_content("not a swarm key")


# ============================================================================
# Swarm Key Installation Tests
# ============================================================================

class TestSwarmKeyInstallation:
    """Tests for install, detect, and remove swarm key operations."""

    def test_install_and_detect(self, tmp_path):
        """install_swarm_key writes file, detect_swarm_key finds it."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        key_content = generate_swarm_key()
        result = install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        assert result == ipfs_dir / "swarm.key"
        assert result.exists()
        assert result.read_text() == key_content

        detected = detect_swarm_key(ipfs_dir=ipfs_dir)
        assert detected == result

    def test_install_creates_dir(self, tmp_path):
        """install_swarm_key creates the IPFS directory if it doesn't exist."""
        ipfs_dir = tmp_path / "new_ipfs_dir"
        assert not ipfs_dir.exists()

        key_content = generate_swarm_key()
        install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        assert ipfs_dir.exists()
        assert (ipfs_dir / "swarm.key").exists()

    def test_install_permissions(self, tmp_path):
        """Installed swarm key has 0o600 permissions."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        key_content = generate_swarm_key()
        path = install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_install_refuses_overwrite(self, tmp_path):
        """install_swarm_key raises FileExistsError if key already exists."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        key_content = generate_swarm_key()
        install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        with pytest.raises(FileExistsError, match="already exists"):
            install_swarm_key(key_content, ipfs_dir=ipfs_dir)

    def test_install_validates_content(self, tmp_path):
        """install_swarm_key validates key format before writing."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid swarm key format"):
            install_swarm_key("bad content", ipfs_dir=ipfs_dir)

        # File should not have been created
        assert not (ipfs_dir / "swarm.key").exists()

    def test_detect_no_key(self, tmp_path):
        """detect_swarm_key returns None when no key exists."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        assert detect_swarm_key(ipfs_dir=ipfs_dir) is None

    def test_detect_no_dir(self, tmp_path):
        """detect_swarm_key returns None when IPFS dir doesn't exist."""
        ipfs_dir = tmp_path / "nonexistent"
        assert detect_swarm_key(ipfs_dir=ipfs_dir) is None

    def test_remove_existing_key(self, tmp_path):
        """remove_swarm_key deletes the key file."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        key_content = generate_swarm_key()
        install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        result = remove_swarm_key(ipfs_dir=ipfs_dir)
        assert result is True
        assert not (ipfs_dir / "swarm.key").exists()

    def test_remove_no_key(self, tmp_path):
        """remove_swarm_key returns False when no key exists."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        result = remove_swarm_key(ipfs_dir=ipfs_dir)
        assert result is False

    def test_get_ipfs_dir_default(self):
        """get_ipfs_dir returns ~/.ipfs by default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove IPFS_PATH if set
            env = os.environ.copy()
            env.pop('IPFS_PATH', None)
            with mock.patch.dict(os.environ, env, clear=True):
                result = get_ipfs_dir()
                assert result == Path.home() / ".ipfs"

    def test_get_ipfs_dir_env_var(self, tmp_path):
        """get_ipfs_dir respects IPFS_PATH environment variable."""
        custom_path = str(tmp_path / "custom_ipfs")
        with mock.patch.dict(os.environ, {'IPFS_PATH': custom_path}):
            result = get_ipfs_dir()
            assert result == Path(custom_path)


# ============================================================================
# Swarm Key Backend Integration Tests
# ============================================================================

class TestSwarmKeyBackendIntegration:
    """Tests for IPFSBackend private network mode."""

    def test_private_network_flag_set(self, tmp_path):
        """IPFSBackend sets private_network=True when swarm key exists."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()
        key_content = generate_swarm_key()
        install_swarm_key(key_content, ipfs_dir=ipfs_dir)

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()
                assert backend.private_network is True

    def test_private_network_flag_not_set(self, tmp_path):
        """IPFSBackend sets private_network=False when no swarm key."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()
                assert backend.private_network is False

    def test_gateways_disabled_in_private_mode(self, tmp_path):
        """IPFSBackend clears gateways when in private network mode."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()
        install_swarm_key(generate_swarm_key(), ipfs_dir=ipfs_dir)

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()
                assert backend.gateways == []

    def test_explicit_gateways_warning_in_private_mode(self, tmp_path):
        """IPFSBackend warns when explicit gateways given in private mode."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()
        install_swarm_key(generate_swarm_key(), ipfs_dir=ipfs_dir)

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                # Should still disable gateways, but log a warning
                backend = IPFSBackend(gateways=["https://custom.gateway.com"])
                assert backend.gateways == []
                assert backend.private_network is True

    def test_get_raises_in_private_mode_without_daemon(self, tmp_path):
        """get() raises PTDBackendError in private mode with no daemon."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()
        install_swarm_key(generate_swarm_key(), ipfs_dir=ipfs_dir)

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()

                with pytest.raises(PTDBackendError, match="Private IPFS network requires"):
                    backend.get("QmYwAPJzv5CZsnN625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_status_includes_private_network_fields(self, tmp_path):
        """status() includes private_network and swarm_key_path."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()
        install_swarm_key(generate_swarm_key(), ipfs_dir=ipfs_dir)

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()
                st = backend.status()

                assert st["private_network"] is True
                assert st["swarm_key_path"] == str(ipfs_dir / "swarm.key")

    def test_status_no_private_network(self, tmp_path):
        """status() reports private_network=False when no swarm key."""
        ipfs_dir = tmp_path / ".ipfs"
        ipfs_dir.mkdir()

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('phasic.trace_repository.get_ipfs_dir', return_value=ipfs_dir):
                backend = IPFSBackend()
                st = backend.status()

                assert st["private_network"] is False
                assert st["swarm_key_path"] is None


# ============================================================================
# Bootstrap Peer Configuration Tests
# ============================================================================

class TestBootstrapPeerConfiguration:
    """Tests for configure_bootstrap_peers()."""

    def test_configure_peers(self, tmp_path):
        """configure_bootstrap_peers calls ipfs bootstrap rm --all + add."""
        peers = [
            "/ip4/192.168.1.1/tcp/4001/p2p/QmPeer1",
            "/ip4/192.168.1.2/tcp/4001/p2p/QmPeer2",
        ]
        ipfs_dir = tmp_path / ".ipfs"

        with mock.patch('shutil.which', return_value='/usr/local/bin/ipfs'):
            with mock.patch('subprocess.run') as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                configure_bootstrap_peers(peers, ipfs_dir=ipfs_dir)

                # First call: rm --all
                calls = mock_run.call_args_list
                assert len(calls) == 3  # 1 rm + 2 adds
                assert calls[0][0][0] == ['/usr/local/bin/ipfs', 'bootstrap', 'rm', '--all']
                assert calls[1][0][0] == ['/usr/local/bin/ipfs', 'bootstrap', 'add', peers[0]]
                assert calls[2][0][0] == ['/usr/local/bin/ipfs', 'bootstrap', 'add', peers[1]]

                # IPFS_PATH should be set in env
                for call in calls:
                    assert call[1]['env']['IPFS_PATH'] == str(ipfs_dir)

    def test_empty_peers_error(self):
        """configure_bootstrap_peers raises ValueError for empty list."""
        with pytest.raises(ValueError, match="must not be empty"):
            configure_bootstrap_peers([])

    def test_missing_ipfs_binary(self):
        """configure_bootstrap_peers raises FileNotFoundError without ipfs."""
        with mock.patch('shutil.which', return_value=None):
            with pytest.raises(FileNotFoundError, match="IPFS binary not found"):
                configure_bootstrap_peers(["/ip4/10.0.0.1/tcp/4001/p2p/QmPeer1"])


# ============================================================================
# Trace Registry Private Network Tests
# ============================================================================

class TestTraceRegistryPrivateNetwork:
    """Tests for TraceRegistry behavior in private network mode."""

    def test_trace_source_private_mode_no_daemon(self, tmp_path, mock_registry):
        """trace_source reports 'unavailable' in private mode with no daemon."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.client = None  # no daemon
        mock_backend.private_network = True

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "unavailable"
            assert info["private_network"] is True

    def test_trace_source_private_mode_with_daemon(self, tmp_path, mock_registry):
        """trace_source reports 'ipfs_daemon' in private mode with daemon."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.client = mock.Mock()  # daemon available
        mock_backend.private_network = True

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "ipfs_daemon"
            assert info["private_network"] is True

    def test_trace_source_public_mode(self, tmp_path, mock_registry):
        """trace_source reports private_network=False in public mode."""
        cache_dir = _make_registry_with_cache(tmp_path, mock_registry)

        mock_backend = mock.Mock(spec=IPFSBackend)
        mock_backend.name = "ipfs"
        mock_backend.client = None
        mock_backend.private_network = False

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_backend):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            info = registry.trace_source("test_trace_1")

            assert info["source"] == "http_gateway"
            assert info["private_network"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
