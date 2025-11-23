"""
Hybrid GitHub + P2P (IPFS/BitTorrent) content distribution library.

This library provides a hybrid storage/distribution model combining:
- Central authoritative repository (GitHub) for manifests
- IPFS for content-addressed P2P distribution
- BitTorrent for high-throughput legacy compatibility
- Cryptographic signing and verification
- Format validation and schema enforcement
"""

__version__ = "0.1.0"

from .client import Client, ClientConfig
from .ipfs_adapter import IPFSAdapter, IPFSError
from .bittorrent_adapter import TorrentCreator, TorrentSeeder, BitTorrentError
from .signing import (
    KeyManager,
    Signer,
    Verifier,
    SigningError,
    VerificationError,
    generate_keypair,
)
from .validation import (
    ContentManifest,
    ContentValidator,
    FileEntry,
    SignatureMetadata,
    ProvenanceInfo,
    DistributionMetadata,
    create_manifest_from_files,
)

__all__ = [
    # Main client
    "Client",
    "ClientConfig",
    
    # IPFS
    "IPFSAdapter",
    "IPFSError",
    
    # BitTorrent
    "TorrentCreator",
    "TorrentSeeder",
    "BitTorrentError",
    
    # Signing
    "KeyManager",
    "Signer",
    "Verifier",
    "SigningError",
    "VerificationError",
    "generate_keypair",
    
    # Validation
    "ContentManifest",
    "ContentValidator",
    "FileEntry",
    "SignatureMetadata",
    "ProvenanceInfo",
    "DistributionMetadata",
    "create_manifest_from_files",
]
