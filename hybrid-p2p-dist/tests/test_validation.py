"""Tests for validation module."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from hybrid_p2p.validation import (
    ContentManifest,
    ContentValidator,
    FileEntry,
    SignatureMetadata,
    ProvenanceInfo,
    DistributionMetadata,
    create_manifest_from_files,
)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample test file."""
    file_path = tmp_path / "test.txt"
    content = b"Hello, World!\n" * 100
    file_path.write_bytes(content)
    return file_path


@pytest.fixture
def sample_files(tmp_path: Path) -> list[Path]:
    """Create multiple sample files."""
    files = []
    
    # Text file
    text_file = tmp_path / "text.txt"
    text_file.write_text("Sample text content\n")
    files.append(text_file)
    
    # JSON file
    json_file = tmp_path / "data.json"
    json_file.write_text('{"key": "value"}\n')
    files.append(json_file)
    
    # Binary file
    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
    files.append(bin_file)
    
    return files


class TestFileEntry:
    """Tests for FileEntry model."""
    
    def test_valid_file_entry(self):
        """Test creating a valid FileEntry."""
        entry = FileEntry(
            path="test.txt",
            size=1024,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        assert entry.path == "test.txt"
        assert entry.size == 1024
    
    def test_invalid_hash_format(self):
        """Test that invalid hash format raises error."""
        with pytest.raises(ValueError):
            FileEntry(
                path="test.txt",
                size=1024,
                sha256="invalid",
                mime_type="text/plain",
            )
    
    def test_invalid_mime_type_format(self):
        """Test that invalid MIME type format raises error."""
        with pytest.raises(ValueError):
            FileEntry(
                path="test.txt",
                size=1024,
                sha256="a" * 64,
                mime_type="invalid",
            )


class TestContentValidator:
    """Tests for ContentValidator."""
    
    def test_validate_file(self, sample_file: Path):
        """Test validating a file."""
        validator = ContentValidator()
        entry = validator.validate_file(sample_file)
        
        assert entry.path == sample_file.name
        assert entry.size == sample_file.stat().st_size
        assert len(entry.sha256) == 64
        assert entry.mime_type == "text/plain"
    
    def test_validate_nonexistent_file(self):
        """Test validating nonexistent file raises error."""
        validator = ContentValidator()
        with pytest.raises(ValueError, match="File not found"):
            validator.validate_file(Path("/nonexistent/file.txt"))
    
    def test_validate_file_against_entry(self, sample_file: Path):
        """Test validating file against entry."""
        validator = ContentValidator()
        
        # Create entry
        entry = validator.validate_file(sample_file)
        
        # Validate should pass
        assert validator.validate_file_against_entry(sample_file, entry)
    
    def test_validate_file_size_mismatch(self, sample_file: Path):
        """Test that size mismatch raises error."""
        validator = ContentValidator()
        
        # Create entry with wrong size
        entry = FileEntry(
            path=sample_file.name,
            size=9999,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        with pytest.raises(ValueError, match="Size mismatch"):
            validator.validate_file_against_entry(sample_file, entry)
    
    def test_validate_file_hash_mismatch(self, sample_file: Path):
        """Test that hash mismatch raises error."""
        validator = ContentValidator()
        
        # Create entry with wrong hash
        entry = FileEntry(
            path=sample_file.name,
            size=sample_file.stat().st_size,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        with pytest.raises(ValueError, match="Hash mismatch"):
            validator.validate_file_against_entry(sample_file, entry)


class TestContentManifest:
    """Tests for ContentManifest model."""
    
    def test_valid_manifest(self):
        """Test creating a valid manifest."""
        file_entry = FileEntry(
            path="test.txt",
            size=100,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        manifest = ContentManifest(
            content_id="test-id",
            version="1.0.0",
            name="test-package",
            files=[file_entry],
            total_size=100,
            signature=SignatureMetadata(
                public_key="test-key",
                signature="test-sig",
            ),
            provenance=ProvenanceInfo(
                uploader_id="alice",
            ),
        )
        
        assert manifest.name == "test-package"
        assert manifest.version == "1.0.0"
        assert len(manifest.files) == 1
    
    def test_invalid_version_format(self):
        """Test that invalid version format raises error."""
        with pytest.raises(ValueError):
            ContentManifest(
                content_id="test-id",
                version="invalid",
                name="test-package",
                files=[],
                total_size=0,
                signature=SignatureMetadata(
                    public_key="key",
                    signature="sig",
                ),
                provenance=ProvenanceInfo(uploader_id="alice"),
            )
    
    def test_total_size_mismatch(self):
        """Test that total size mismatch raises error."""
        file_entry = FileEntry(
            path="test.txt",
            size=100,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        with pytest.raises(ValueError, match="total_size"):
            ContentManifest(
                content_id="test-id",
                version="1.0.0",
                name="test-package",
                files=[file_entry],
                total_size=999,  # Wrong!
                signature=SignatureMetadata(
                    public_key="key",
                    signature="sig",
                ),
                provenance=ProvenanceInfo(uploader_id="alice"),
            )
    
    def test_canonical_json(self):
        """Test canonical JSON generation."""
        file_entry = FileEntry(
            path="test.txt",
            size=100,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        manifest = ContentManifest(
            content_id="test-id",
            version="1.0.0",
            name="test-package",
            files=[file_entry],
            total_size=100,
            signature=SignatureMetadata(
                public_key="key",
                signature="sig",
            ),
            provenance=ProvenanceInfo(uploader_id="alice"),
        )
        
        json_str = manifest.to_canonical_json()
        
        # Should not contain signature
        assert "signature" not in json_str
        
        # Should be valid JSON
        import json
        data = json.loads(json_str)
        assert data["name"] == "test-package"
    
    def test_compute_manifest_hash(self):
        """Test manifest hash computation."""
        file_entry = FileEntry(
            path="test.txt",
            size=100,
            sha256="a" * 64,
            mime_type="text/plain",
        )
        
        manifest = ContentManifest(
            content_id="test-id",
            version="1.0.0",
            name="test-package",
            files=[file_entry],
            total_size=100,
            signature=SignatureMetadata(
                public_key="key",
                signature="sig",
            ),
            provenance=ProvenanceInfo(uploader_id="alice"),
        )
        
        hash1 = manifest.compute_manifest_hash()
        hash2 = manifest.compute_manifest_hash()
        
        # Should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64


class TestCreateManifest:
    """Tests for create_manifest_from_files helper."""
    
    def test_create_manifest_from_files(self, sample_files: list[Path]):
        """Test creating manifest from files."""
        manifest = create_manifest_from_files(
            files=sample_files,
            name="test-package",
            version="1.0.0",
            uploader_id="alice",
            description="Test package",
        )
        
        assert manifest.name == "test-package"
        assert manifest.version == "1.0.0"
        assert len(manifest.files) == len(sample_files)
        
        # Check total size
        expected_size = sum(f.stat().st_size for f in sample_files)
        assert manifest.total_size == expected_size
    
    def test_create_manifest_empty_files(self):
        """Test that empty file list raises error."""
        with pytest.raises(ValueError):
            create_manifest_from_files(
                files=[],
                name="test",
                version="1.0.0",
                uploader_id="alice",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
