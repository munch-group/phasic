"""
Tests for IPFS-based trace repository.

These tests verify:
1. IPFSBackend initialization with different configurations
2. Gateway fallback when IPFS daemon unavailable
3. TraceRegistry registry loading and caching
4. Mock downloads (without requiring actual IPFS setup)
"""

import json
import gzip
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from phasic.trace_repository import IPFSBackend, TraceRegistry, get_trace
from phasic.exceptions import PTDBackendError


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
        # Mock requests to simulate gateway response
        mock_content = b"test content"

        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('requests.get') as mock_get:
                mock_response = mock.Mock()
                mock_response.content = mock_content
                mock_response.raise_for_status = mock.Mock()
                mock_get.return_value = mock_response

                backend = IPFSBackend()
                content = backend.get("fake_cid")

                assert content == mock_content
                assert mock_get.called

    def test_get_with_output_path(self):
        """Test downloading to file."""
        mock_content = b"test content"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"

            with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
                with mock.patch('requests.get') as mock_get:
                    mock_response = mock.Mock()
                    mock_response.content = mock_content
                    mock_response.raise_for_status = mock.Mock()
                    mock_get.return_value = mock_response

                    backend = IPFSBackend()
                    result = backend.get("fake_cid", output_path=output_path)

                    assert result is None
                    assert output_path.exists()
                    assert output_path.read_bytes() == mock_content

    def test_get_failure_all_gateways(self):
        """Test failure when all gateways fail."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            with mock.patch('requests.get', side_effect=Exception("Network error")):
                backend = IPFSBackend()

                with pytest.raises(PTDBackendError, match="Failed to retrieve"):
                    backend.get("fake_cid")

    def test_add_without_daemon(self):
        """Test that add() fails without daemon."""
        with mock.patch('phasic.trace_repository.HAS_IPFS_CLIENT', False):
            backend = IPFSBackend()

            with pytest.raises(PTDBackendError, match="Publishing requires IPFS daemon"):
                backend.add(Path("/fake/path"))


class TestTraceRegistry:
    """Tests for TraceRegistry class."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry data."""
        return {
            "version": "1.0.0",
            "updated": "2025-10-21T12:00:00Z",
            "traces": {
                "test_trace_1": {
                    "cid": "bafytest123",
                    "description": "Test trace 1",
                    "metadata": {
                        "model_type": "coalescent",
                        "domain": "population-genetics",
                        "param_length": 1,
                        "vertices": 5,
                        "tags": ["test", "coalescent"]
                    },
                    "files": {
                        "trace.json.gz": {
                            "cid": "bafytrace123",
                            "size_bytes": 1000
                        }
                    }
                },
                "test_trace_2": {
                    "cid": "bafytest456",
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
                            "cid": "bafytrace456",
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
    def mock_trace_data(self):
        """Create mock trace data."""
        return {
            "vertex_rates": [[1.0], [0.0]],
            "edge_probs": [[0.5, 0.5]],
            "vertex_targets": [[1], []]
        }

    def test_init_with_cached_registry(self, tmp_path, mock_registry):
        """Test initialization with cached registry."""
        # Create cache directory with registry
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        registry_file = cache_dir / "registry.json"
        registry_file.write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            assert registry.registry is not None
            assert "test_trace_1" in registry.registry["traces"]

    def test_update_registry(self, tmp_path, mock_registry):
        """Test updating registry from GitHub."""
        cache_dir = tmp_path / "cache"

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            with mock.patch('requests.get') as mock_get:
                mock_response = mock.Mock()
                mock_response.json.return_value = mock_registry
                mock_response.raise_for_status = mock.Mock()
                mock_get.return_value = mock_response

                registry = TraceRegistry(cache_dir=cache_dir, auto_update=True)

                assert registry.registry is not None
                assert (cache_dir / "registry.json").exists()

    def test_list_traces_no_filter(self, tmp_path, mock_registry):
        """Test listing all traces without filters."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces()

            assert len(traces) == 2
            assert traces[0]['trace_id'] == 'test_trace_1'
            assert traces[1]['trace_id'] == 'test_trace_2'

    def test_list_traces_with_domain_filter(self, tmp_path, mock_registry):
        """Test filtering traces by domain."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(domain="population-genetics")

            assert len(traces) == 2

    def test_list_traces_with_model_type_filter(self, tmp_path, mock_registry):
        """Test filtering traces by model type."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(model_type="coalescent")

            assert len(traces) == 1
            assert traces[0]['trace_id'] == 'test_trace_1'

    def test_list_traces_with_tags_filter(self, tmp_path, mock_registry):
        """Test filtering traces by tags."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            traces = registry.list_traces(tags=["structured"])

            assert len(traces) == 1
            assert traces[0]['trace_id'] == 'test_trace_2'

    def test_get_trace_from_cache(self, tmp_path, mock_registry, mock_trace_data):
        """Test loading trace from local cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        # Create cached trace
        trace_dir = cache_dir / "traces" / "test_trace_1"
        trace_dir.mkdir(parents=True)
        trace_file = trace_dir / "trace.json.gz"
        with gzip.open(trace_file, 'wt') as f:
            json.dump(mock_trace_data, f)

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            trace = registry.get_trace("test_trace_1")

            assert trace == mock_trace_data

    def test_get_trace_download(self, tmp_path, mock_registry, mock_trace_data):
        """Test downloading trace from IPFS."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        # Mock IPFS backend
        mock_ipfs = mock.Mock()

        def mock_get(cid, output_path=None):
            # Write trace data to output_path
            with gzip.open(output_path, 'wt') as f:
                json.dump(mock_trace_data, f)

        mock_ipfs.get = mock_get

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_ipfs):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            trace = registry.get_trace("test_trace_1")

            assert trace == mock_trace_data

    def test_get_trace_not_found(self, tmp_path, mock_registry):
        """Test error when trace not in registry."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "registry.json").write_text(json.dumps(mock_registry))

        with mock.patch('phasic.trace_repository.IPFSBackend'):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)

            with pytest.raises(PTDBackendError, match="not found in registry"):
                registry.get_trace("nonexistent_trace")

    def test_publish_trace(self, tmp_path, mock_trace_data):
        """Test publishing a trace to IPFS."""
        cache_dir = tmp_path / "cache"

        # Mock IPFS backend
        mock_ipfs = mock.Mock()
        mock_ipfs.add.return_value = "bafynewcid123"

        metadata = {
            "model_type": "test",
            "domain": "test",
            "param_length": 1,
            "description": "Test trace"
        }

        with mock.patch('phasic.trace_repository.IPFSBackend', return_value=mock_ipfs):
            registry = TraceRegistry(cache_dir=cache_dir, auto_update=False)
            registry.registry = {"traces": {}}  # Initialize empty registry

            cid = registry.publish_trace(
                trace=mock_trace_data,
                trace_id="new_test_trace",
                metadata=metadata,
                submit_pr=False
            )

            assert cid == "bafynewcid123"
            assert mock_ipfs.add.called


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_trace_helper(self, tmp_path):
        """Test get_trace() convenience function."""
        # This is a simple wrapper, so we just verify it calls TraceRegistry
        with mock.patch('phasic.trace_repository.TraceRegistry') as MockRegistry:
            mock_registry_instance = mock.Mock()
            mock_registry_instance.get_trace.return_value = {"test": "data"}
            MockRegistry.return_value = mock_registry_instance

            result = get_trace("test_trace")

            assert result == {"test": "data"}
            mock_registry_instance.get_trace.assert_called_once_with("test_trace", force_download=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
