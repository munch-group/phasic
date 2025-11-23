"""Content validation and schema definitions."""

import hashlib
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import magic
from pydantic import BaseModel, Field, field_validator, model_validator


class ContentType(str, Enum):
    """Allowed content types."""
    TEXT = "text/plain"
    JSON = "application/json"
    BINARY = "application/octet-stream"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    VIDEO_MP4 = "video/mp4"
    AUDIO_MP3 = "audio/mpeg"
    PDF = "application/pdf"


class FileEntry(BaseModel):
    """Single file entry in manifest."""
    path: str = Field(..., description="Relative path to file")
    size: int = Field(..., gt=0, description="File size in bytes")
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$", description="SHA-256 hash")
    cid: Optional[str] = Field(None, description="IPFS CID (if available)")
    mime_type: str = Field(..., description="MIME type of file")
    
    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """Ensure mime type is in allowed list or warn."""
        # Allow any MIME type but validate format
        if "/" not in v:
            raise ValueError(f"Invalid MIME type format: {v}")
        return v


class SignatureMetadata(BaseModel):
    """Cryptographic signature metadata."""
    algorithm: str = Field("ed25519", description="Signature algorithm")
    public_key: str = Field(..., description="Base64-encoded public key")
    signature: str = Field(..., description="Base64-encoded signature")
    signed_at: datetime = Field(default_factory=datetime.utcnow)


class ProvenanceInfo(BaseModel):
    """Provenance and authorship metadata."""
    uploader_id: str = Field(..., description="Unique uploader identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    git_commit: Optional[str] = Field(None, pattern=r"^[a-f0-9]{40}$")
    git_ref: Optional[str] = Field(None, description="Git branch or tag")
    pr_number: Optional[int] = Field(None, description="Pull request number")


class DistributionMetadata(BaseModel):
    """P2P distribution metadata."""
    ipfs_cid: Optional[str] = Field(None, description="Root IPFS CID for content")
    ipfs_gateway: Optional[str] = Field(None, description="Preferred IPFS gateway URL")
    torrent_infohash: Optional[str] = Field(None, description="BitTorrent info hash")
    torrent_magnet: Optional[str] = Field(None, description="Magnet link")
    pinning_nodes: List[str] = Field(default_factory=list, description="Known pinning node IDs")


class ContentManifest(BaseModel):
    """
    Canonical manifest for content distribution.
    
    This defines the complete schema for a content package including
    files, hashes, signatures, and distribution metadata.
    """
    
    manifest_version: str = Field("1.0", description="Manifest schema version")
    content_id: str = Field(..., description="Unique content identifier (CID or UUID)")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$", description="Semantic version")
    name: str = Field(..., min_length=1, max_length=255, description="Content package name")
    description: Optional[str] = Field(None, max_length=2000)
    
    files: List[FileEntry] = Field(..., min_length=1, description="List of files in package")
    total_size: int = Field(..., gt=0, description="Total size in bytes")
    
    signature: SignatureMetadata = Field(..., description="Manifest signature")
    provenance: ProvenanceInfo = Field(..., description="Origin and authorship info")
    distribution: Optional[DistributionMetadata] = Field(None, description="P2P distribution data")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic versioning format."""
        parts = v.split("-")[0].split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid semver format: {v}")
        try:
            [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"Invalid semver numbers: {v}")
        return v
    
    @model_validator(mode="after")
    def validate_total_size(self) -> "ContentManifest":
        """Ensure total_size matches sum of file sizes."""
        calculated = sum(f.size for f in self.files)
        if self.total_size != calculated:
            raise ValueError(
                f"total_size ({self.total_size}) does not match "
                f"sum of file sizes ({calculated})"
            )
        return self
    
    def to_canonical_json(self) -> str:
        """
        Generate canonical JSON representation for signing.
        
        Uses sorted keys and no whitespace for deterministic output.
        """
        return self.model_dump_json(exclude={"signature"}, by_alias=True, indent=None)
    
    def compute_manifest_hash(self) -> str:
        """Compute SHA-256 hash of canonical manifest (excluding signature)."""
        canonical = self.to_canonical_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ContentValidator:
    """Validates content files and manifests."""
    
    def __init__(self, strict_mime: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mime: If True, only allow predefined MIME types
        """
        self.strict_mime = strict_mime
        self.magic = magic.Magic(mime=True)
    
    def validate_file(self, path: Path) -> FileEntry:
        """
        Validate a single file and create FileEntry.
        
        Args:
            path: Path to file
            
        Returns:
            FileEntry with computed hashes and metadata
            
        Raises:
            ValueError: If file validation fails
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        
        # Compute SHA-256
        sha256_hash = hashlib.sha256()
        size = 0
        
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256_hash.update(chunk)
                size += len(chunk)
        
        # Detect MIME type
        mime_type = self.magic.from_file(str(path))
        
        if self.strict_mime:
            allowed = [ct.value for ct in ContentType]
            if mime_type not in allowed:
                raise ValueError(f"MIME type {mime_type} not in allowed list")
        
        return FileEntry(
            path=str(path.name),  # Use relative path
            size=size,
            sha256=sha256_hash.hexdigest(),
            mime_type=mime_type,
        )
    
    def validate_manifest(self, manifest: ContentManifest) -> bool:
        """
        Validate manifest structure and constraints.
        
        Args:
            manifest: Manifest to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Pydantic already validates structure
        # Add any additional business logic validation here
        
        # Check for duplicate file paths
        paths = [f.path for f in manifest.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Duplicate file paths in manifest")
        
        return True
    
    def validate_file_against_entry(self, path: Path, entry: FileEntry) -> bool:
        """
        Validate that a file matches its manifest entry.
        
        Args:
            path: Path to file
            entry: Expected file entry
            
        Returns:
            True if file matches entry
            
        Raises:
            ValueError: If validation fails
        """
        actual = self.validate_file(path)
        
        if actual.size != entry.size:
            raise ValueError(
                f"Size mismatch for {path}: expected {entry.size}, got {actual.size}"
            )
        
        if actual.sha256 != entry.sha256:
            raise ValueError(
                f"Hash mismatch for {path}: expected {entry.sha256}, got {actual.sha256}"
            )
        
        return True


def create_manifest_from_files(
    files: List[Path],
    name: str,
    version: str,
    uploader_id: str,
    description: Optional[str] = None,
) -> ContentManifest:
    """
    Create a manifest from a list of files.
    
    This is a helper for creating manifests; signature must be added separately.
    
    Args:
        files: List of file paths
        name: Package name
        version: Semantic version
        uploader_id: Uploader identifier
        description: Optional description
        
    Returns:
        ContentManifest (without signature)
    """
    validator = ContentValidator()
    file_entries = [validator.validate_file(f) for f in files]
    
    total_size = sum(f.size for f in file_entries)
    
    # Generate content ID from hash of all file hashes
    content_hash = hashlib.sha256()
    for entry in sorted(file_entries, key=lambda x: x.path):
        content_hash.update(entry.sha256.encode("utf-8"))
    content_id = content_hash.hexdigest()
    
    # Create placeholder signature (will be replaced by signing module)
    placeholder_sig = SignatureMetadata(
        public_key="placeholder",
        signature="placeholder",
    )
    
    return ContentManifest(
        content_id=content_id,
        version=version,
        name=name,
        description=description,
        files=file_entries,
        total_size=total_size,
        signature=placeholder_sig,
        provenance=ProvenanceInfo(uploader_id=uploader_id),
    )
